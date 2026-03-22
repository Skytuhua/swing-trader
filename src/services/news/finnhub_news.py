"""Finnhub news provider adapter.

Endpoints used:
  - /news                    → general market news (category filter)
  - /company-news            → company-specific news
  - /press-releases          → company press releases
  - /news-sentiment          → aggregated sentiment from Finnhub

Rate limit: 60 calls / minute on the free tier.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from src.core.config import NewsConfig
from src.core.exceptions import DataProviderError
from src.services.news.base import NewsProvider, RawNewsItem

logger = structlog.get_logger(__name__)

# Finnhub category names that map to our internal categories
_CATEGORY_MAP: dict[str, str] = {
    "general": "general",
    "forex": "forex",
    "crypto": "crypto",
    "merger": "merger",
    "top news": "general",
}

# Keyword→category inference when no explicit category is available
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "revenue", "eps", "quarterly results", "guidance"],
    "analyst": ["upgrade", "downgrade", "price target", "initiate", "outperform", "underperform"],
    "merger": ["merger", "acquisition", "takeover", "deal", "buyout"],
}


def _infer_category(headline: str) -> str:
    lower = headline.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return cat
    return "general"


def _parse_timestamp(ts: Any) -> datetime:
    """Convert unix timestamp or ISO string to UTC datetime."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


class _RateLimiter:
    """Token-bucket rate limiter for Finnhub's 60 calls/minute free tier."""

    def __init__(self, calls_per_minute: int = 55) -> None:
        self._max = calls_per_minute
        self._tokens = float(calls_per_minute)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens at the rate of calls_per_minute / 60 per second
            self._tokens = min(self._max, self._tokens + elapsed * (self._max / 60.0))
            self._last_refill = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / (self._max / 60.0)
                logger.debug("finnhub_rate_limit_wait", wait_seconds=round(wait, 2))
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


class FinnhubNewsProvider(NewsProvider):
    """Async Finnhub news adapter backed by httpx.

    Usage::

        provider = FinnhubNewsProvider(config)
        items = await provider.fetch_company_news("AAPL", date.today() - timedelta(days=7))
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, config: NewsConfig) -> None:
        self._config = config
        self._api_key: str = config.api_keys.get("finnhub", "")
        if not self._api_key:
            logger.warning("finnhub_no_api_key", message="Finnhub API key not configured")
        self._rate_limiter = _RateLimiter(calls_per_minute=55)
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={"X-Finnhub-Token": self._api_key},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> Any:
        await self._rate_limiter.acquire()
        client = await self._get_client()
        try:
            response = await client.get(path, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise DataProviderError(
                f"Finnhub HTTP {exc.response.status_code} for {path}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"Finnhub request error for {path}: {exc}") from exc

    @staticmethod
    def _parse_news_item(item: dict, source_category: str = "general") -> RawNewsItem | None:
        """Convert a Finnhub news JSON object to RawNewsItem."""
        headline = (item.get("headline") or item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not headline or not url:
            return None

        published_ts = item.get("datetime") or item.get("publishedDate") or 0
        published_at = _parse_timestamp(published_ts)

        # Tickers: Finnhub may include a 'related' string or 'symbol' field
        related = item.get("related", "") or ""
        tickers: list[str] = [
            t.strip().upper() for t in related.split(",") if t.strip()
        ]
        if not tickers and item.get("symbol"):
            tickers = [str(item["symbol"]).upper()]

        category_raw = (item.get("category") or source_category).lower()
        category = _CATEGORY_MAP.get(category_raw, _infer_category(headline))

        return RawNewsItem(
            source="finnhub",
            headline=headline,
            url=url,
            published_at=published_at,
            summary=item.get("summary", ""),
            tickers=tickers,
            sectors=[],
            category=category,
            image_url=item.get("image", ""),
            source_id=str(item.get("id", "")),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_general_news(self, category: str = "general") -> list[RawNewsItem]:
        """Fetch market-wide news from Finnhub.

        Args:
            category: One of general | forex | crypto | merger.
        """
        logger.info("finnhub_fetch_general_news", category=category)
        valid_categories = {"general", "forex", "crypto", "merger"}
        if category not in valid_categories:
            category = "general"

        try:
            data = await self._get("/news", params={"category": category, "minId": 0})
        except DataProviderError:
            logger.exception("finnhub_general_news_failed", category=category)
            return []

        if not isinstance(data, list):
            logger.warning("finnhub_unexpected_response", path="/news", type=type(data).__name__)
            return []

        items: list[RawNewsItem] = []
        for raw in data:
            parsed = self._parse_news_item(raw, source_category=category)
            if parsed:
                items.append(parsed)

        logger.info("finnhub_general_news_fetched", count=len(items), category=category)
        return items

    async def fetch_company_news(
        self, ticker: str, from_date: date, to_date: date | None = None
    ) -> list[RawNewsItem]:
        """Fetch company-specific news for ``ticker`` from Finnhub.

        Args:
            ticker: Equity symbol, e.g. 'AAPL'.
            from_date: Inclusive start date.
            to_date: Inclusive end date (defaults to today).
        """
        if to_date is None:
            to_date = date.today()

        logger.info("finnhub_fetch_company_news", ticker=ticker, from_date=str(from_date))

        params = {
            "symbol": ticker.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        }

        items: list[RawNewsItem] = []

        # 1. Company news endpoint
        try:
            data = await self._get("/company-news", params=params)
            if isinstance(data, list):
                for raw in data:
                    # Inject ticker directly since we know the query symbol
                    raw.setdefault("symbol", ticker)
                    parsed = self._parse_news_item(raw, source_category="general")
                    if parsed:
                        if ticker.upper() not in parsed.tickers:
                            parsed.tickers.insert(0, ticker.upper())
                        items.append(parsed)
        except DataProviderError:
            logger.exception("finnhub_company_news_failed", ticker=ticker)

        # 2. Press releases (company-specific)
        try:
            pr_data = await self._get("/press-releases", params={"symbol": ticker.upper()})
            press_releases = pr_data.get("majorDevelopment", []) if isinstance(pr_data, dict) else []
            for pr in press_releases:
                headline = pr.get("headline", "").strip()
                if not headline:
                    continue
                ts = pr.get("datetime") or pr.get("date") or 0
                published_at = _parse_timestamp(ts)
                from_dt = datetime.combine(from_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                if published_at < from_dt:
                    continue
                items.append(
                    RawNewsItem(
                        source="finnhub",
                        headline=headline,
                        url=pr.get("url", ""),
                        published_at=published_at,
                        summary=pr.get("description", ""),
                        tickers=[ticker.upper()],
                        sectors=[],
                        category="general",
                        source_id=str(pr.get("id", "")),
                    )
                )
        except DataProviderError:
            logger.debug("finnhub_press_releases_unavailable", ticker=ticker)

        # Deduplicate by URL within this batch
        seen_urls: set[str] = set()
        unique: list[RawNewsItem] = []
        for item in items:
            if item.url and item.url not in seen_urls:
                seen_urls.add(item.url)
                unique.append(item)

        # Sort newest first
        unique.sort(key=lambda x: x.published_at, reverse=True)

        logger.info("finnhub_company_news_fetched", ticker=ticker, count=len(unique))
        return unique

    async def fetch_sector_news(self, sector: str) -> list[RawNewsItem]:
        """Fetch news for a broad sector.

        Finnhub does not have a native sector endpoint; we use general news
        and filter by sector keywords in the headline/summary.
        """
        logger.info("finnhub_fetch_sector_news", sector=sector)
        general = await self.fetch_general_news(category="general")

        sector_keywords: dict[str, list[str]] = {
            "technology": ["tech", "software", "semiconductor", "ai", "cloud", "cybersecurity"],
            "healthcare": ["pharma", "biotech", "drug", "fda", "clinical", "health"],
            "financials": ["bank", "insurance", "fintech", "fed", "rate", "lending"],
            "energy": ["oil", "gas", "energy", "renewable", "solar", "esg"],
            "consumer": ["retail", "consumer", "spending", "amazon", "walmart"],
            "industrials": ["industrial", "manufacturing", "supply chain", "aerospace", "defense"],
            "materials": ["mining", "metals", "chemicals", "commodities"],
            "utilities": ["utility", "power", "electricity", "grid"],
            "real_estate": ["reit", "real estate", "housing", "property"],
        }

        keywords = sector_keywords.get(sector.lower(), [sector.lower()])
        relevant = [
            item for item in general
            if any(kw in item.headline.lower() or kw in item.summary.lower() for kw in keywords)
        ]
        for item in relevant:
            if sector.lower() not in item.sectors:
                item.sectors.append(sector.lower())

        logger.info("finnhub_sector_news_fetched", sector=sector, count=len(relevant))
        return relevant

    async def health_check(self) -> bool:
        try:
            await self._get("/news", params={"category": "general", "minId": 0})
            return True
        except DataProviderError:
            return False
