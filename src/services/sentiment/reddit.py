"""Reddit sentiment adapter.

Uses Reddit's public JSON API (no OAuth required for read-only access):
  GET https://www.reddit.com/r/{subreddit}/new.json
  GET https://www.reddit.com/r/{subreddit}/hot.json
  GET https://www.reddit.com/r/{subreddit}/search.json

Subreddits monitored:
  - r/wallstreetbets  (high-volume, speculative)
  - r/stocks          (research-oriented)
  - r/investing       (longer-term / fundamental)

Ticker extraction:
  - $AAPL style ($-prefix with 1-5 uppercase letters)
  - Bare uppercase 2-5 letter words against a known-ticker check if provided
  - Mention context filtering (avoids false positives like "I" or "A")

Sentiment scoring:
  - Keyword lexicon (fast, no model needed)
  - Optional FinBERT upgrade if NewsProcessor is injected

Rate limiting: Reddit's public JSON API allows ~60 requests/minute with a
proper User-Agent.  We throttle to 30 requests/minute to be conservative.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from src.core.config import SentimentConfig
from src.core.exceptions import DataProviderError
from src.services.sentiment.base import MentionSource, SentimentProvider, SocialMention

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Subreddit configuration
# ---------------------------------------------------------------------------

_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
]

# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------

_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Common false positives — all-caps words that are NOT tickers
_TICKER_STOP_WORDS = frozenset({
    "I", "A", "IS", "IT", "IN", "TO", "BE", "DO", "GO", "NO", "SO",
    "MY", "ME", "US", "WE", "HE", "SHE", "AM", "AN", "OR", "IF",
    "AT", "AS", "OF", "UP", "BY", "ON", "DO", "RE", "OK", "DD",
    "CEO", "CFO", "COO", "SEC", "IPO", "ETF", "OTC", "ATH", "ATL",
    "EPS", "PE", "GDP", "CPI", "FED", "IMO", "FWIW", "YOLO", "HODL",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HAS", "MORE", "WAS", "HIS", "HER", "ITS", "OUR", "OUT", "NEW",
    "ANY", "TWO", "WAY", "WHO", "HOW", "WHEN", "THEN", "EACH", "SOME",
})


def _extract_tickers(text: str, target_ticker: str | None = None) -> list[str]:
    """Extract ticker symbols from social media text.

    Priority:
    1. $-prefixed tickers (highest precision)
    2. Bare uppercase words not in stop-word list

    If ``target_ticker`` is set, returns only matches containing that symbol.
    """
    found: set[str] = set()

    # High-confidence: $AAPL style
    for m in _DOLLAR_TICKER_RE.finditer(text):
        sym = m.group(1).upper()
        if sym not in _TICKER_STOP_WORDS:
            found.add(sym)

    # Lower confidence: bare uppercase words
    if not found:
        for m in _BARE_TICKER_RE.finditer(text):
            sym = m.group(1).upper()
            if sym not in _TICKER_STOP_WORDS and len(sym) >= 2:
                found.add(sym)

    tickers = list(found)

    if target_ticker:
        target_upper = target_ticker.upper()
        # Keep all tickers but flag this mention only if target appears
        if target_upper not in tickers:
            return []

    return tickers


# ---------------------------------------------------------------------------
# Lexicon-based sentiment scoring
# ---------------------------------------------------------------------------

_BULLISH_WORDS = frozenset({
    "bullish", "bull", "long", "buy", "calls", "rocket", "moon", "green",
    "gains", "profit", "upgrade", "breaking out", "breakout", "ath",
    "oversold", "bounce", "recovery", "strong", "beat", "outperform",
    "undervalued", "cheap", "loading", "accumulate", "dip buy",
    "hold", "diamond hands", "to the moon", "squeeze", "yolo",
})

_BEARISH_WORDS = frozenset({
    "bearish", "bear", "short", "puts", "sell", "crash", "dump", "red",
    "loss", "downgrade", "breakdown", "overbought", "overvalued", "expensive",
    "bubble", "miss", "disappoint", "disappointing", "fraud",
    "bankruptcy", "default", "cut", "warning", "weak", "avoid",
})

_INTENSIFIERS = frozenset({"very", "extremely", "super", "ultra", "massive", "huge", "strong"})
_NEGATORS = frozenset({"not", "no", "never", "don't", "doesn't", "isn't", "wasn't", "can't"})


def _keyword_polarity(text: str) -> float:
    """Return a polarity score in [-1, +1] using a financial keyword lexicon.

    Simple approach: count bullish/bearish words with optional negation handling.
    Intensifiers multiply the adjacent word's weight by 1.5.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    bull_score = 0.0
    bear_score = 0.0

    for i, word in enumerate(words):
        # Check negation in 2-word window before current word
        negated = any(
            j >= 0 and words[j] in _NEGATORS
            for j in range(max(0, i - 2), i)
        )
        # Check intensifier in 1-word window before
        intensified = i > 0 and words[i - 1] in _INTENSIFIERS

        multiplier = 1.5 if intensified else 1.0

        if word in _BULLISH_WORDS:
            if negated:
                bear_score += 0.5 * multiplier
            else:
                bull_score += 1.0 * multiplier
        elif word in _BEARISH_WORDS:
            if negated:
                bull_score += 0.5 * multiplier
            else:
                bear_score += 1.0 * multiplier

    total = bull_score + bear_score
    if total == 0:
        return 0.0
    return (bull_score - bear_score) / total


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RequestThrottle:
    def __init__(self, rps: float = 0.5) -> None:  # 30 req/min
        self._interval = 1.0 / rps
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep = self._interval - (now - self._last_call)
            if sleep > 0:
                await asyncio.sleep(sleep)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class RedditSentimentProvider(SentimentProvider):
    """Reddit sentiment adapter using the public (unauthenticated) JSON API.

    Parameters
    ----------
    config:
        SentimentConfig; api_keys['reddit'] is used for OAuth if provided,
        but the adapter works without it.
    subreddits:
        Subreddits to query.  Defaults to WSB + stocks + investing.
    post_limit:
        Maximum posts to fetch per subreddit per call.  Default 100.
    finbert_processor:
        Optional NewsProcessor instance for higher-quality polarity.
        If not provided, keyword lexicon is used.
    """

    _BASE = "https://www.reddit.com"
    _HEADERS = {
        "User-Agent": "SwingTraderBot/1.0 (research tool; contact: bot@swingtrader.local)",
        "Accept": "application/json",
    }

    def __init__(
        self,
        config: SentimentConfig,
        subreddits: list[str] | None = None,
        post_limit: int = 100,
        finbert_processor=None,  # Optional[NewsProcessor] — avoid circular import
    ) -> None:
        self._config = config
        self._subreddits = subreddits or list(_SUBREDDITS)
        self._post_limit = min(post_limit, 100)  # Reddit hard-caps at 100
        self._finbert = finbert_processor
        self._throttle = _RequestThrottle(rps=0.5)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._BASE,
                timeout=httpx.Timeout(15.0, connect=5.0),
                headers=self._HEADERS,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _fetch_subreddit(
        self, subreddit: str, sort: str = "new", limit: int = 100
    ) -> list[dict]:
        """Fetch posts from a subreddit via the public JSON endpoint."""
        await self._throttle.wait()
        client = await self._get_client()
        url = f"/r/{subreddit}/{sort}.json"
        try:
            resp = await client.get(url, params={"limit": limit, "raw_json": 1})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("reddit_rate_limited", subreddit=subreddit)
                await asyncio.sleep(60)
                return []
            raise DataProviderError(
                f"Reddit HTTP {exc.response.status_code} for r/{subreddit}"
            ) from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"Reddit request error: {exc}") from exc

        posts = []
        children = data.get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            if post:
                posts.append(post)
        return posts

    async def _search_subreddit(
        self, subreddit: str, query: str, since_utc: float
    ) -> list[dict]:
        """Search a subreddit for ``query``."""
        await self._throttle.wait()
        client = await self._get_client()
        url = f"/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "true",
            "sort": "new",
            "limit": self._post_limit,
            "t": "week",
            "raw_json": 1,
        }
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("reddit_search_rate_limited", subreddit=subreddit)
                return []
            raise DataProviderError(
                f"Reddit search HTTP {exc.response.status_code} for r/{subreddit}"
            ) from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"Reddit search error: {exc}") from exc

        posts = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post and float(post.get("created_utc", 0)) >= since_utc:
                posts.append(post)
        return posts

    def _post_to_mention(
        self,
        post: dict,
        ticker: str,
        subreddit: str,
    ) -> SocialMention | None:
        """Convert a Reddit post dict to a SocialMention."""
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        full_text = f"{title} {selftext}".strip()

        if not full_text:
            return None

        # Verify ticker appears in this post
        tickers = _extract_tickers(full_text, target_ticker=ticker)
        if not tickers:
            return None

        created_utc = float(post.get("created_utc", 0))
        timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        # Polarity
        if self._finbert is not None:
            try:
                analysis = self._finbert.analyze(title, selftext[:300])
                polarity = float(analysis.sentiment_score)
                method = "finbert"
            except Exception:
                polarity = _keyword_polarity(full_text)
                method = "keyword"
        else:
            polarity = _keyword_polarity(full_text)
            method = "keyword"

        upvotes = max(0, int(post.get("score", 0)))
        post_id = str(post.get("id", ""))

        return SocialMention(
            source=MentionSource.REDDIT,
            ticker=ticker.upper(),
            text=full_text[:500],
            author=str(post.get("author", "unknown")),
            timestamp=timestamp,
            polarity=polarity,
            polarity_method=method,
            upvotes=upvotes,
            post_id=post_id,
            url=f"https://www.reddit.com{post.get('permalink', '')}",
            subreddit=subreddit,
        )

    async def fetch_mentions(self, ticker: str, since: datetime) -> list[SocialMention]:
        """Fetch Reddit mentions of ``ticker`` from monitored subreddits.

        Combines:
        1. Search for ticker symbol in each subreddit (direct match).
        2. Scan new/hot posts for $TICKER mentions.
        """
        since_utc = since.timestamp() if since.tzinfo else since.replace(tzinfo=timezone.utc).timestamp()
        ticker_upper = ticker.upper()
        query = f"${ticker_upper} OR \"{ticker_upper}\""

        logger.info("reddit_fetch_mentions", ticker=ticker_upper, subreddits=self._subreddits)

        all_mentions: list[SocialMention] = []
        seen_ids: set[str] = set()

        tasks = [
            self._search_subreddit(sub, query, since_utc)
            for sub in self._subreddits
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sub, result in zip(self._subreddits, results):
            if isinstance(result, Exception):
                logger.warning("reddit_subreddit_error", subreddit=sub, error=str(result))
                continue
            for post in result:
                pid = post.get("id", "")
                if pid and pid in seen_ids:
                    continue
                mention = self._post_to_mention(post, ticker_upper, sub)
                if mention:
                    seen_ids.add(pid)
                    all_mentions.append(mention)

        # Also scan new posts (catches ticker mentions without exact search match)
        scan_tasks = [
            self._fetch_subreddit(sub, sort="new", limit=50)
            for sub in self._subreddits[:2]  # limit to WSB + stocks for perf
        ]
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)

        for sub, result in zip(self._subreddits[:2], scan_results):
            if isinstance(result, Exception):
                continue
            for post in result:
                pid = post.get("id", "")
                if pid and pid in seen_ids:
                    continue
                created_utc = float(post.get("created_utc", 0))
                if created_utc < since_utc:
                    continue
                mention = self._post_to_mention(post, ticker_upper, sub)
                if mention:
                    seen_ids.add(pid)
                    all_mentions.append(mention)

        # Sort newest first
        all_mentions.sort(key=lambda m: m.timestamp, reverse=True)

        logger.info(
            "reddit_mentions_fetched",
            ticker=ticker_upper,
            count=len(all_mentions),
        )
        return all_mentions

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await self._throttle.wait()
            resp = await client.get("/r/stocks/new.json", params={"limit": 1})
            resp.raise_for_status()
            return True
        except Exception:
            return False
