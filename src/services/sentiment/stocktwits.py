"""StockTwits sentiment adapter.

API reference: https://api.stocktwits.com/developers/docs

Public endpoints used (no auth required for basic access):
  GET https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
    - Returns the 30 most recent messages for a symbol
  GET https://api.stocktwits.com/api/2/streams/trending.json
    - Returns trending symbols

Each message may have a ``sentiment`` field with ``{ basic: "Bullish" | "Bearish" }``
when the user explicitly tagged their post.  We use this as the ground-truth label
when present and fall back to keyword polarity otherwise.

Rate limits: 200 requests/hour unauthenticated (≈3.3/min).
We throttle to 2 requests/minute to leave headroom.
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
# Lexicon for posts without explicit labels
# ---------------------------------------------------------------------------

_BULLISH_TOKENS = frozenset({
    "bullish", "bull", "long", "buy", "calls", "moon", "rocket", "green",
    "breakout", "squeeze", "rally", "up", "gains", "profit", "strong",
    "oversold", "bounce", "undervalued",
})
_BEARISH_TOKENS = frozenset({
    "bearish", "bear", "short", "puts", "sell", "crash", "dump", "red",
    "breakdown", "overbought", "overvalued", "weak", "loss", "miss",
    "avoid", "warning", "fraud",
})


def _keyword_polarity(text: str) -> float:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    bull = len(words & _BULLISH_TOKENS)
    bear = len(words & _BEARISH_TOKENS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _label_to_polarity(label: Optional[str]) -> Optional[float]:
    """Convert StockTwits explicit bullish/bearish label to polarity."""
    if label is None:
        return None
    l = label.lower()
    if l == "bullish":
        return 0.7
    if l == "bearish":
        return -0.7
    return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RequestThrottle:
    def __init__(self, rps: float = 1 / 30) -> None:  # 2 req/min
        self._interval = 1.0 / rps
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep = self._interval - (now - self._last)
            if sleep > 0:
                await asyncio.sleep(sleep)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class StockTwitsProvider(SentimentProvider):
    """StockTwits sentiment adapter.

    Fetches the symbol stream and extracts bullish/bearish labels along with
    keyword-based polarity for unlabelled posts.

    Parameters
    ----------
    config:
        SentimentConfig.
    access_token:
        Optional OAuth access token. Not required for basic symbol stream
        access, but raises the rate limit significantly.
    finbert_processor:
        Optional NewsProcessor for higher-quality polarity on unlabelled posts.
    """

    _BASE = "https://api.stocktwits.com/api/2"

    def __init__(
        self,
        config: SentimentConfig,
        access_token: Optional[str] = None,
        finbert_processor=None,
    ) -> None:
        self._config = config
        self._access_token = access_token or config.api_keys.get("stocktwits", "")
        self._finbert = finbert_processor
        self._throttle = _RequestThrottle(rps=1 / 30)  # ~2 req/min
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._BASE,
                timeout=httpx.Timeout(15.0, connect=5.0),
                headers={"User-Agent": "SwingTraderBot/1.0"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        await self._throttle.wait()
        client = await self._get_client()
        p = dict(params or {})
        if self._access_token:
            p["access_token"] = self._access_token

        try:
            resp = await client.get(path, params=p)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                logger.warning("stocktwits_rate_limited")
                await asyncio.sleep(60)
                return {}
            if code == 404:
                return {}  # symbol not found
            raise DataProviderError(f"StockTwits HTTP {code} for {path}") from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"StockTwits request error: {exc}") from exc

        # API-level errors
        resp_status = (data.get("response") or {}).get("status")
        if resp_status and resp_status != 200:
            msg = (data.get("errors") or [{}])[0].get("message", "unknown error")
            logger.warning("stocktwits_api_error", path=path, message=msg)
            return {}

        return data

    def _parse_message(self, msg: dict, ticker: str, since_ts: float) -> SocialMention | None:
        """Convert a StockTwits message dict to a SocialMention."""
        body = msg.get("body", "").strip()
        if not body:
            return None

        # Timestamp
        created_at_str = msg.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        if timestamp.timestamp() < since_ts:
            return None

        # Sentiment label
        sentiment_obj = msg.get("entities", {}).get("sentiment") or msg.get("sentiment")
        label: Optional[str] = None
        if isinstance(sentiment_obj, dict):
            label = sentiment_obj.get("basic")

        # Polarity
        label_polarity = _label_to_polarity(label)
        if label_polarity is not None:
            polarity = label_polarity
            method = "label"
        elif self._finbert is not None:
            try:
                analysis = self._finbert.analyze(body)
                polarity = float(analysis.sentiment_score)
                method = "finbert"
            except Exception:
                polarity = _keyword_polarity(body)
                method = "keyword"
        else:
            polarity = _keyword_polarity(body)
            method = "keyword"

        user = msg.get("user") or {}
        author = str(user.get("username") or user.get("id") or "unknown")
        msg_id = str(msg.get("id", ""))

        return SocialMention(
            source=MentionSource.STOCKTWITS,
            ticker=ticker.upper(),
            text=body[:500],
            author=author,
            timestamp=timestamp,
            polarity=polarity,
            polarity_method=method,
            bullish_label=(label == "Bullish") if label else None,
            bearish_label=(label == "Bearish") if label else None,
            upvotes=int(msg.get("likes", {}).get("total", 0)),
            post_id=msg_id,
            url=f"https://stocktwits.com/{author}/message/{msg_id}",
        )

    async def fetch_symbol_stream(
        self,
        ticker: str,
        since: datetime,
        max_messages: int = 30,
    ) -> list[SocialMention]:
        """Fetch the latest StockTwits messages for ``ticker``.

        The public API returns the 30 most recent messages for a symbol.
        Paginates backwards (using ``max`` cursor) until ``since`` is reached
        or we run out of messages.
        """
        since_ts = since.timestamp()
        ticker_upper = ticker.upper()
        mentions: list[SocialMention] = []
        cursor: Optional[int] = None  # StockTwits uses max/since integer cursors
        page = 0

        while len(mentions) < max_messages:
            params: dict = {}
            if cursor is not None:
                params["max"] = cursor  # fetch messages older than cursor

            data = await self._get(f"/streams/symbol/{ticker_upper}.json", params=params)
            if not data:
                break

            messages = data.get("messages") or []
            if not messages:
                break

            oldest_ts = None
            for msg in messages:
                mention = self._parse_message(msg, ticker_upper, since_ts)
                if mention:
                    mentions.append(mention)

                # Track oldest message timestamp for pagination
                try:
                    ts = datetime.fromisoformat(
                        (msg.get("created_at") or "").replace("Z", "+00:00")
                    ).timestamp()
                    if oldest_ts is None or ts < oldest_ts:
                        oldest_ts = ts
                        cursor = msg.get("id")
                except (ValueError, AttributeError):
                    pass

            page += 1
            # Stop if oldest fetched message is already older than `since`
            if oldest_ts is not None and oldest_ts < since_ts:
                break
            # StockTwits cursor-based pagination: need to continue
            if len(messages) < 30:
                break  # no more messages available

        return mentions

    async def fetch_mentions(self, ticker: str, since: datetime) -> list[SocialMention]:
        """Fetch StockTwits mentions of ``ticker`` since ``since``."""
        logger.info("stocktwits_fetch_mentions", ticker=ticker.upper())
        try:
            mentions = await self.fetch_symbol_stream(ticker, since)
        except DataProviderError:
            logger.exception("stocktwits_fetch_failed", ticker=ticker)
            return []

        mentions.sort(key=lambda m: m.timestamp, reverse=True)
        logger.info("stocktwits_mentions_fetched", ticker=ticker, count=len(mentions))
        return mentions

    async def fetch_trending_symbols(self) -> list[str]:
        """Return the current list of trending ticker symbols on StockTwits."""
        data = await self._get("/trending/symbols.json")
        symbols = data.get("symbols") or []
        return [s.get("symbol", "") for s in symbols if s.get("symbol")]

    async def health_check(self) -> bool:
        try:
            data = await self._get("/streams/symbol/AAPL.json", params={"limit": 1})
            return bool(data)
        except DataProviderError:
            return False
