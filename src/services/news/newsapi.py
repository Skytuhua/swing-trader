"""NewsAPI.org news provider adapter.

Endpoints:
  - /v2/top-headlines  → top headlines (by category or source)
  - /v2/everything     → keyword / date-range search

Rate limits:
  - Developer plan: 100 requests/day
  - We rate-limit to ~1 req/second to stay safe.

Docs: https://newsapi.org/docs
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx
import structlog

from src.core.config import NewsConfig
from src.core.exceptions import DataProviderError
from src.services.news.base import NewsProvider, RawNewsItem

logger = structlog.get_logger(__name__)

# NewsAPI category names
_VALID_CATEGORIES = {
    "business", "entertainment", "general", "health",
    "science", "sports", "technology",
}

# Map our internal sector names → NewsAPI category + search query
_SECTOR_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "technology":  ("technology", ""),
    "healthcare":  ("health",     ""),
    "financials":  ("business",   "finance OR bank OR stocks"),
    "energy":      ("business",   "oil OR gas OR energy"),
    "consumer":    ("business",   "retail OR consumer spending"),
    "industrials": ("business",   "manufacturing OR aerospace"),
    "materials":   ("business",   "mining OR metals OR commodities"),
    "utilities":   ("business",   "utility OR electricity"),
    "real_estate": ("business",   "real estate OR housing OR REIT"),
}

# Keywords that hint at specific financial event categories
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "revenue", "eps", "quarterly results", "guidance", "beats estimates"],
    "analyst":  ["upgrade", "downgrade", "price target", "initiate", "outperform", "underperform", "buy rating"],
    "merger":   ["merger", "acquisition", "takeover", "deal", "buyout", "agrees to acquire"],
}


def _infer_category(headline: str, description: str = "") -> str:
    text = f"{headline} {description}".lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return cat
    return "general"


def _parse_iso(ts_str: str) -> datetime:
    if not ts_str:
        return datetime.now(tz=timezone.utc)
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(tz=timezone.utc)


class _RequestThrottle:
    """Simple leaky-bucket throttle: at most ``rps`` requests per second."""

    def __init__(self, rps: float = 1.0) -> None:
        self._interval = 1.0 / rps
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep_time = self._interval - (now - self._last_call)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self._last_call = time.monotonic()


class NewsAPIProvider(NewsProvider):
    """Async NewsAPI.org adapter backed by httpx.

    Usage::

        provider = NewsAPIProvider(config)
        items = await provider.fetch_company_news("TSLA", date.today() - timedelta(days=7))
    """

    BASE_URL = "https://newsapi.org/v2"

    # Maximum ``pageSize`` the API allows per request
    _MAX_PAGE_SIZE = 100

    def __init__(self, config: NewsConfig) -> None:
        self._config = config
        self._api_key: str = config.api_keys.get("newsapi", "")
        if not self._api_key:
            logger.warning("newsapi_no_api_key", message="NewsAPI key not configured")
        self._throttle = _RequestThrottle(rps=0.8)  # stay well under 1 req/sec
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(15.0, connect=5.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "User-Agent": "SwingTraderBot/1.0",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> dict:
        await self._throttle.wait()
        client = await self._get_client()
        try:
            response = await client.get(path, params=params or {})
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise DataProviderError(
                f"NewsAPI HTTP {exc.response.status_code} for {path}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"NewsAPI request error for {path}: {exc}") from exc

        status = data.get("status", "")
        if status != "ok":
            code = data.get("code", "unknown")
            msg = data.get("message", "")
            raise DataProviderError(f"NewsAPI error [{code}]: {msg}")

        return data

    def _parse_article(self, article: dict, tickers: list[str] | None = None) -> RawNewsItem | None:
        """Parse a single NewsAPI article dict into a RawNewsItem."""
        headline = (article.get("title") or "").strip()
        url = (article.get("url") or "").strip()
        if not headline or not url or headline.lower() == "[removed]":
            return None

        description = article.get("description") or ""
        content = article.get("content") or ""
        summary = description or content[:300]

        published_at = _parse_iso(article.get("publishedAt", ""))

        source_name = (article.get("source") or {}).get("name") or "newsapi"

        category = _infer_category(headline, summary)

        return RawNewsItem(
            source="newsapi",
            headline=headline,
            url=url,
            published_at=published_at,
            summary=summary,
            tickers=list(tickers) if tickers else [],
            sectors=[],
            category=category,
            image_url=article.get("urlToImage") or "",
            source_id=f"newsapi_{hash(url) & 0xFFFFFFFF}",
        )

    def _collect_articles(
        self, data: dict, tickers: list[str] | None = None
    ) -> list[RawNewsItem]:
        articles = data.get("articles") or []
        items: list[RawNewsItem] = []
        for article in articles:
            parsed = self._parse_article(article, tickers=tickers)
            if parsed:
                items.append(parsed)
        return items

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_general_news(self, category: str = "general") -> list[RawNewsItem]:
        """Fetch top headlines for a NewsAPI category.

        Args:
            category: One of business | technology | health | general | science.
        """
        if category not in _VALID_CATEGORIES:
            category = "business"  # closest to financial news

        logger.info("newsapi_fetch_general_news", category=category)
        try:
            data = await self._get(
                "/top-headlines",
                params={
                    "category": category,
                    "language": "en",
                    "pageSize": self._MAX_PAGE_SIZE,
                    "country": "us",
                },
            )
        except DataProviderError:
            logger.exception("newsapi_general_news_failed", category=category)
            return []

        items = self._collect_articles(data)
        logger.info("newsapi_general_news_fetched", count=len(items), category=category)
        return items

    async def fetch_company_news(self, ticker: str, from_date: date) -> list[RawNewsItem]:
        """Fetch news for ``ticker`` using keyword search in /v2/everything.

        Args:
            ticker: Stock symbol, e.g. 'AAPL'.
            from_date: Start date (UTC).
        """
        ticker_upper = ticker.upper()
        logger.info("newsapi_fetch_company_news", ticker=ticker_upper, from_date=str(from_date))

        # Build a query that finds ticker mentions in headlines
        # e.g., "AAPL OR Apple Inc"
        query = ticker_upper

        # Cap look-back at 30 days (developer plan limitation)
        from_clamped = max(from_date, date.today() - timedelta(days=29))

        params: dict[str, Any] = {
            "q": query,
            "language": "en",
            "from": from_clamped.strftime("%Y-%m-%dT00:00:00"),
            "to": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "publishedAt",
            "pageSize": self._MAX_PAGE_SIZE,
        }

        try:
            data = await self._get("/everything", params=params)
        except DataProviderError:
            logger.exception("newsapi_company_news_failed", ticker=ticker_upper)
            return []

        items = self._collect_articles(data, tickers=[ticker_upper])
        logger.info("newsapi_company_news_fetched", ticker=ticker_upper, count=len(items))
        return items

    async def fetch_sector_news(self, sector: str) -> list[RawNewsItem]:
        """Fetch sector-themed news.

        Uses a combination of category filter and keyword query.
        """
        logger.info("newsapi_fetch_sector_news", sector=sector)
        category, keywords = _SECTOR_CATEGORY_MAP.get(
            sector.lower(), ("business", sector.lower())
        )

        items: list[RawNewsItem] = []

        # Approach 1: top headlines for the category
        try:
            data = await self._get(
                "/top-headlines",
                params={
                    "category": category,
                    "language": "en",
                    "country": "us",
                    "pageSize": 50,
                },
            )
            items.extend(self._collect_articles(data))
        except DataProviderError:
            logger.exception("newsapi_sector_headlines_failed", sector=sector)

        # Approach 2: keyword search for the past 3 days
        if keywords:
            try:
                since = (date.today() - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00")
                data2 = await self._get(
                    "/everything",
                    params={
                        "q": keywords,
                        "language": "en",
                        "from": since,
                        "sortBy": "relevancy",
                        "pageSize": 50,
                    },
                )
                items.extend(self._collect_articles(data2))
            except DataProviderError:
                logger.debug("newsapi_sector_keyword_search_failed", sector=sector)

        # Attach sector tag and deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[RawNewsItem] = []
        for item in items:
            item.sectors.append(sector.lower())
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique.append(item)

        unique.sort(key=lambda x: x.published_at, reverse=True)
        logger.info("newsapi_sector_news_fetched", sector=sector, count=len(unique))
        return unique

    async def search_news(
        self,
        query: str,
        from_date: date | None = None,
        to_date: date | None = None,
        sort_by: str = "publishedAt",
        page_size: int = 50,
    ) -> list[RawNewsItem]:
        """Open-ended keyword search in /v2/everything.

        Args:
            query: Free-text query, supports AND/OR/NOT operators.
            from_date: Optional start date.
            to_date: Optional end date.
            sort_by: One of publishedAt | relevancy | popularity.
            page_size: Number of results (max 100).
        """
        logger.info("newsapi_search", query=query[:80])
        params: dict[str, Any] = {
            "q": query,
            "language": "en",
            "sortBy": sort_by,
            "pageSize": min(page_size, self._MAX_PAGE_SIZE),
        }
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT00:00:00")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT23:59:59")

        try:
            data = await self._get("/everything", params=params)
        except DataProviderError:
            logger.exception("newsapi_search_failed", query=query[:80])
            return []

        return self._collect_articles(data)

    async def health_check(self) -> bool:
        try:
            await self._get(
                "/top-headlines",
                params={"category": "business", "language": "en", "pageSize": 1, "country": "us"},
            )
            return True
        except DataProviderError:
            return False
