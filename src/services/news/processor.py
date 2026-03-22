"""FinBERT-based NLP processor for financial news sentiment analysis.

Model: ProsusAI/finbert (fine-tuned BERT on financial texts)
  - Labels: positive (index 0) | negative (index 1) | neutral (index 2)
    NOTE: the actual label order in the HuggingFace checkpoint is
          {0: positive, 1: negative, 2: neutral}

Features
--------
- Lazy model loading (loaded on first use, not on import)
- Batch inference for throughput
- Thread-safe singleton pattern for the model
- Graceful degradation: if torch/transformers are unavailable, returns neutral
- GPU auto-detection; falls back to CPU
- Text truncation at 512 tokens
- Configurable batch size (default 16)
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import structlog

from src.services.news.base import NewsAnalysis, RawNewsItem

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are only resolved at runtime when the model is loaded
# ---------------------------------------------------------------------------
_torch = None
_transformers = None
_TORCH_AVAILABLE: Optional[bool] = None


def _check_torch() -> bool:
    global _torch, _transformers, _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        import torch
        import transformers

        _torch = torch
        _transformers = transformers
        _TORCH_AVAILABLE = True
        logger.debug("finbert_torch_available", torch_version=torch.__version__)
    except ImportError:
        _TORCH_AVAILABLE = False
        logger.warning(
            "finbert_torch_unavailable",
            message="torch/transformers not installed; sentiment will default to neutral",
        )
    return _TORCH_AVAILABLE  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fallback analysis when the model is unavailable
# ---------------------------------------------------------------------------

_FALLBACK_ANALYSIS = NewsAnalysis(
    sentiment_score=0.0,
    positive_prob=0.333,
    negative_prob=0.333,
    neutral_prob=0.334,
    confidence=0.334,
)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

@dataclass
class _ModelBundle:
    tokenizer: object
    model: object
    device: str
    label_map: dict[int, str]


_MODEL_LOCK = threading.Lock()
_MODEL_BUNDLE: Optional[_ModelBundle] = None


def _load_model(model_name: str = "ProsusAI/finbert") -> Optional[_ModelBundle]:
    """Load and return the FinBERT model bundle (thread-safe singleton)."""
    global _MODEL_BUNDLE

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None:
            return _MODEL_BUNDLE

        if not _check_torch():
            return None

        torch = _torch
        transformers = _transformers

        try:
            logger.info("finbert_loading_model", model=model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

            # Device selection: CUDA > MPS (Apple Silicon) > CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            model = model.to(device)
            model.eval()

            # Build label_map from the model's config
            # ProsusAI/finbert: id2label = {0: positive, 1: negative, 2: neutral}
            id2label: dict[int, str] = getattr(model.config, "id2label", {
                0: "positive", 1: "negative", 2: "neutral"
            })

            _MODEL_BUNDLE = _ModelBundle(
                tokenizer=tokenizer,
                model=model,
                device=device,
                label_map={v.lower(): k for k, v in id2label.items()},
            )
            logger.info("finbert_model_loaded", device=device, model=model_name)
        except Exception as exc:
            logger.exception("finbert_model_load_failed", error=str(exc))
            return None

    return _MODEL_BUNDLE


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference_batch(
    bundle: _ModelBundle,
    texts: list[str],
    max_length: int = 512,
) -> list[NewsAnalysis]:
    """Run FinBERT inference on a batch of texts. Runs synchronously (CPU/GPU)."""
    torch = _torch

    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    # Determine label indices
    lmap = bundle.label_map
    pos_idx = lmap.get("positive", 0)
    neg_idx = lmap.get("negative", 1)
    neu_idx = lmap.get("neutral", 2)

    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results: list[NewsAnalysis] = []
    for row in probs.cpu().tolist():
        pos = row[pos_idx]
        neg = row[neg_idx]
        neu = row[neu_idx]
        score = pos - neg  # range [-1, +1]
        results.append(
            NewsAnalysis(
                sentiment_score=round(score, 4),
                positive_prob=round(pos, 4),
                negative_prob=round(neg, 4),
                neutral_prob=round(neu, 4),
                confidence=round(max(pos, neg, neu), 4),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public processor class
# ---------------------------------------------------------------------------

class NewsProcessor:
    """FinBERT-backed news sentiment processor.

    The HuggingFace model is loaded lazily on the first call to ``analyze``
    or ``analyze_batch``.  Calling ``preload()`` warms it up eagerly.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Default: "ProsusAI/finbert".
    batch_size:
        Maximum number of texts per inference batch.  Default: 16.
    max_length:
        Maximum token length fed to the model.  Default: 512.
    executor:
        Optional ThreadPoolExecutor for running blocking inference
        in async contexts.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 16,
        max_length: int = 512,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="finbert")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def preload(self) -> bool:
        """Eagerly load the model.  Returns True if successful."""
        bundle = _load_model(self.model_name)
        return bundle is not None

    @property
    def is_available(self) -> bool:
        """True if the model is loaded and ready."""
        return _MODEL_BUNDLE is not None

    # ------------------------------------------------------------------
    # Synchronous inference
    # ------------------------------------------------------------------

    def analyze(self, headline: str, summary: str = "") -> NewsAnalysis:
        """Analyse a single news item.

        Args:
            headline: Article headline.
            summary: Optional article body snippet.

        Returns:
            NewsAnalysis with sentiment_score, positive/negative/neutral probabilities
            and confidence.
        """
        text = f"{headline}. {summary}".strip() if summary else headline
        results = self._analyze_texts([text])
        return results[0]

    def analyze_batch(self, items: list[RawNewsItem]) -> list[NewsAnalysis]:
        """Analyse a list of RawNewsItem objects in batches.

        Args:
            items: News items to analyse.

        Returns:
            List of NewsAnalysis, one per input item (same order).
        """
        if not items:
            return []

        texts = [
            f"{item.headline}. {item.summary}".strip() if item.summary else item.headline
            for item in items
        ]
        return self._analyze_texts(texts)

    def _analyze_texts(self, texts: list[str]) -> list[NewsAnalysis]:
        """Analyse a list of raw strings, batching as configured."""
        bundle = _load_model(self.model_name)
        if bundle is None:
            logger.debug("finbert_fallback_neutral", reason="model_unavailable", count=len(texts))
            return [_FALLBACK_ANALYSIS] * len(texts)

        results: list[NewsAnalysis] = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_results = _run_inference_batch(bundle, batch, max_length=self.max_length)
                results.extend(batch_results)
        except Exception as exc:
            logger.exception("finbert_inference_error", error=str(exc), text_count=len(texts))
            # Pad with fallback for remaining items
            while len(results) < len(texts):
                results.append(_FALLBACK_ANALYSIS)

        return results

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def analyze_async(self, headline: str, summary: str = "") -> NewsAnalysis:
        """Async wrapper around ``analyze``.  Offloads to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.analyze, headline, summary
        )

    async def analyze_batch_async(self, items: list[RawNewsItem]) -> list[NewsAnalysis]:
        """Async wrapper around ``analyze_batch``.  Offloads to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.analyze_batch, items
        )

    # ------------------------------------------------------------------
    # Relevance scoring (rule-based complement to FinBERT)
    # ------------------------------------------------------------------

    _HIGH_IMPACT_KEYWORDS: frozenset[str] = frozenset({
        "earnings", "revenue", "eps", "guidance", "outlook",
        "upgrade", "downgrade", "price target", "buy", "sell", "hold",
        "fda", "approval", "merger", "acquisition", "buyout",
        "dividend", "split", "buyback", "lawsuit", "fraud", "sec",
        "bankruptcy", "default", "restatement",
    })

    def estimate_relevance(self, item: RawNewsItem, ticker: str) -> float:
        """Estimate how relevant a news item is to ``ticker`` (0–1).

        Combines:
        - Ticker mention in headline/summary (+0.5)
        - High-impact keyword presence (+0.3)
        - Confidence proxy from FinBERT analysis if available
        """
        text = item.text.lower()
        score = 0.0

        # Direct ticker mention
        if ticker.lower() in text:
            score += 0.5
        elif ticker.upper() in [t.upper() for t in item.tickers]:
            score += 0.4

        # High-impact financial keywords
        keyword_hits = sum(1 for kw in self._HIGH_IMPACT_KEYWORDS if kw in text)
        score += min(0.3, keyword_hits * 0.1)

        # Recency bonus: items fetched recently are more likely relevant
        score += 0.2  # base presence score

        return round(min(1.0, score), 3)
