"""Near-duplicate news item detection using MinHash LSH.

Algorithm
---------
1. Each article's headline+summary is tokenised into character 3-grams (shingles).
2. A 128-hash MinHash signature is computed.
3. An in-process LSH index (band-based) answers "find items with Jaccard ≥ threshold".
4. If a candidate matches any stored item above the similarity threshold the item
   is classified as a near-duplicate and discarded.
5. Fingerprints (exact SHA-256) are stored in a secondary set for O(1) exact-match
   elimination before the more expensive MinHash step.

The implementation is self-contained and does not require the ``datasketch`` library
(though it mirrors the datasketch MinHash API closely). When ``datasketch`` is
installed, it is used directly for better performance.

Configuration
-------------
- ``threshold``        : Jaccard similarity above which two items are duplicates (default 0.85).
- ``num_perm``         : Number of hash permutations for MinHash (default 128).
- ``band_count``       : LSH band count (derived from threshold automatically if not set).
- ``max_age_seconds``  : Items older than this are pruned from the index (default 86400 = 24h).
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import struct
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from src.services.news.base import RawNewsItem

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to use datasketch for better performance; fall back to manual
# ---------------------------------------------------------------------------
try:
    from datasketch import MinHash, MinHashLSH

    _DATASKETCH_AVAILABLE = True
    logger.debug("deduplicator_backend", backend="datasketch")
except ImportError:
    _DATASKETCH_AVAILABLE = False
    logger.debug("deduplicator_backend", backend="manual_minhash")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _char_shingles(text: str, k: int = 3) -> set[str]:
    """Return the set of k-character shingles from ``text``."""
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _word_shingles(text: str, k: int = 2) -> set[str]:
    """Return the set of k-word shingles from ``text``."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# Manual MinHash implementation (no external dependency)
# ---------------------------------------------------------------------------

# Large Mersenne prime used as the hash modulus
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _generate_hash_params(num_perm: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Generate (a, b) pairs for the universal hash family: h(x) = (ax + b) mod p."""
    rng = random.Random(seed)
    a = [rng.randint(1, _MERSENNE_PRIME) for _ in range(num_perm)]
    b = [rng.randint(0, _MERSENNE_PRIME) for _ in range(num_perm)]
    return a, b


def _minhash_signature(shingles: set[str], a: list[int], b: list[int]) -> list[int]:
    """Compute a MinHash signature vector for ``shingles``."""
    num_perm = len(a)
    sig = [_MAX_HASH] * num_perm
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16) & _MAX_HASH
        for i in range(num_perm):
            candidate = ((a[i] * h + b[i]) % _MERSENNE_PRIME) & _MAX_HASH
            if candidate < sig[i]:
                sig[i] = candidate
    return sig


def _jaccard_from_signatures(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
    return matches / len(sig1)


# LSH band parameters: b bands of r rows each; threshold ≈ (1/b)^(1/r)
def _optimal_band_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """Return (bands, rows) that minimise false neg/pos around threshold."""
    best_bands = 1
    best_rows = num_perm
    best_error = float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        # Approximate false positive probability
        t = (1.0 / b) ** (1.0 / r)
        error = abs(t - threshold)
        if error < best_error:
            best_error = error
            best_bands = b
            best_rows = r
    return best_bands, best_rows


# ---------------------------------------------------------------------------
# Stored entry
# ---------------------------------------------------------------------------

@dataclass
class _IndexEntry:
    fingerprint: str
    signature: list[int]
    stored_at: float  # monotonic timestamp


# ---------------------------------------------------------------------------
# Main deduplicator
# ---------------------------------------------------------------------------


class NewsDeduplicator:
    """Near-duplicate news item filter.

    Parameters
    ----------
    threshold:
        Jaccard similarity threshold above which two items are considered duplicates.
        Default 0.85.
    num_perm:
        Number of MinHash permutations. Higher → more accurate but slower.
        Default 128.
    max_age_seconds:
        Items older than this are pruned from the in-memory index on the next
        ``filter_novel`` call.  Default 86400 (24 hours).
    use_word_shingles:
        Use word 2-grams instead of character 3-grams (better for very short headlines).
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        max_age_seconds: float = 86400.0,
        use_word_shingles: bool = True,
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.max_age_seconds = max_age_seconds
        self.use_word_shingles = use_word_shingles

        # Exact-match fingerprint cache
        self._fingerprints: set[str] = set()

        if _DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._minhash_index: dict[str, MinHash] = {}  # fp → minhash object
            self._age_index: dict[str, float] = {}         # fp → stored_at
        else:
            # Manual LSH
            self._a, self._b = _generate_hash_params(num_perm)
            self._bands, self._rows = _optimal_band_params(threshold, num_perm)
            # band_buckets[band_idx][bucket_key] → list of fingerprints
            self._band_buckets: list[dict[bytes, list[str]]] = [
                {} for _ in range(self._bands)
            ]
            self._sig_store: dict[str, _IndexEntry] = {}  # fp → IndexEntry

        logger.info(
            "deduplicator_init",
            threshold=threshold,
            num_perm=num_perm,
            backend="datasketch" if _DATASKETCH_AVAILABLE else "manual",
        )

    # ------------------------------------------------------------------
    # Shingle helpers
    # ------------------------------------------------------------------

    def _shingles(self, text: str) -> set[str]:
        normalised = _normalise(text)
        if self.use_word_shingles:
            return _word_shingles(normalised, k=2) | _char_shingles(normalised, k=5)
        return _char_shingles(normalised, k=3)

    def _text_for_item(self, item: RawNewsItem) -> str:
        return item.headline if not item.summary else f"{item.headline} {item.summary[:200]}"

    # ------------------------------------------------------------------
    # datasketch path
    # ------------------------------------------------------------------

    def _ds_is_duplicate(self, fingerprint: str, mh: "MinHash") -> bool:
        """Query the datasketch LSH index."""
        results = self._lsh.query(mh)
        return len(results) > 0

    def _ds_add(self, fingerprint: str, mh: "MinHash") -> None:
        try:
            self._lsh.insert(fingerprint, mh)
        except Exception:
            # Already inserted (possible race in concurrent use)
            pass
        self._minhash_index[fingerprint] = mh
        self._age_index[fingerprint] = time.monotonic()

    def _ds_prune_old(self) -> None:
        now = time.monotonic()
        stale = [fp for fp, t in self._age_index.items() if now - t > self.max_age_seconds]
        for fp in stale:
            try:
                self._lsh.remove(fp)
            except Exception:
                pass
            self._minhash_index.pop(fp, None)
            self._age_index.pop(fp, None)
            self._fingerprints.discard(fp)

    # ------------------------------------------------------------------
    # Manual MinHash path
    # ------------------------------------------------------------------

    def _manual_is_duplicate(self, fingerprint: str, sig: list[int]) -> bool:
        """Check LSH buckets for candidate matches and verify with exact Jaccard."""
        candidates: set[str] = set()
        for b in range(self._bands):
            start = b * self._rows
            end = start + self._rows
            band_data = sig[start:end]
            bucket_key = struct.pack(f">{self._rows}I", *band_data)
            bucket = self._band_buckets[b].get(bucket_key, [])
            candidates.update(bucket)

        for candidate_fp in candidates:
            entry = self._sig_store.get(candidate_fp)
            if entry is None:
                continue
            j = _jaccard_from_signatures(sig, entry.signature)
            if j >= self.threshold:
                return True
        return False

    def _manual_add(self, fingerprint: str, sig: list[int]) -> None:
        entry = _IndexEntry(
            fingerprint=fingerprint,
            signature=sig,
            stored_at=time.monotonic(),
        )
        self._sig_store[fingerprint] = entry
        for b in range(self._bands):
            start = b * self._rows
            end = start + self._rows
            band_data = sig[start:end]
            bucket_key = struct.pack(f">{self._rows}I", *band_data)
            bucket = self._band_buckets[b].setdefault(bucket_key, [])
            bucket.append(fingerprint)

    def _manual_prune_old(self) -> None:
        now = time.monotonic()
        stale = [
            fp for fp, entry in self._sig_store.items()
            if now - entry.stored_at > self.max_age_seconds
        ]
        for fp in stale:
            self._sig_store.pop(fp, None)
            self._fingerprints.discard(fp)
        # Rebuild band buckets (simple O(n) pass — acceptable for small indices)
        if stale:
            self._band_buckets = [{} for _ in range(self._bands)]
            for fp, entry in self._sig_store.items():
                for b in range(self._bands):
                    start = b * self._rows
                    end = start + self._rows
                    band_data = entry.signature[start:end]
                    bucket_key = struct.pack(f">{self._rows}I", *band_data)
                    self._band_buckets[b].setdefault(bucket_key, []).append(fp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, item: RawNewsItem) -> bool:
        """Return True if ``item`` is a near-duplicate of a previously seen item."""
        fp = item.fingerprint

        # Fast path: exact URL/headline match
        if fp in self._fingerprints:
            return True

        text = self._text_for_item(item)
        shingles = self._shingles(text)
        if not shingles:
            return False

        if _DATASKETCH_AVAILABLE:
            mh = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))
            return self._ds_is_duplicate(fp, mh)
        else:
            sig = _minhash_signature(shingles, self._a, self._b)
            return self._manual_is_duplicate(fp, sig)

    def add(self, item: RawNewsItem) -> None:
        """Add ``item`` to the duplicate-detection index."""
        fp = item.fingerprint
        if fp in self._fingerprints:
            return  # already stored

        self._fingerprints.add(fp)
        text = self._text_for_item(item)
        shingles = self._shingles(text)
        if not shingles:
            return

        if _DATASKETCH_AVAILABLE:
            mh = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))
            self._ds_add(fp, mh)
        else:
            sig = _minhash_signature(shingles, self._a, self._b)
            self._manual_add(fp, sig)

    def filter_novel(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        """Return only the novel (non-duplicate) items from ``items``.

        Items are processed in order; if two items in the batch are near-duplicates
        of each other, only the first is kept.  All novel items are added to the index.
        """
        # Prune stale entries first
        if _DATASKETCH_AVAILABLE:
            self._ds_prune_old()
        else:
            self._manual_prune_old()

        novel: list[RawNewsItem] = []
        duplicates = 0
        for item in items:
            if self.is_duplicate(item):
                duplicates += 1
                logger.debug(
                    "deduplicator_duplicate_dropped",
                    headline=item.headline[:60],
                    source=item.source,
                )
            else:
                self.add(item)
                novel.append(item)

        logger.info(
            "deduplicator_filter_complete",
            input_count=len(items),
            novel_count=len(novel),
            duplicate_count=duplicates,
        )
        return novel

    def index_size(self) -> int:
        """Return the number of items currently in the index."""
        return len(self._fingerprints)

    def clear(self) -> None:
        """Reset the index completely."""
        self._fingerprints.clear()
        if _DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            self._minhash_index.clear()
            self._age_index.clear()
        else:
            self._band_buckets = [{} for _ in range(self._bands)]
            self._sig_store.clear()
        logger.info("deduplicator_cleared")
