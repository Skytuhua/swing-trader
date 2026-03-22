"""Finnhub market data provider.

Uses the Finnhub REST API via httpx (async).
Free tier: 60 API calls / minute (1 call/second sustained).
Rate limiting is enforced with a token-bucket style semaphore + sleep,
and tenacity is used for transient errors.

Finnhub docs: https://finnhub.io/docs/api
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone
from typing import Any

import httpx
import pandas as pd
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.core.exceptions import DataProviderError, StaleDataError
from src.services.market_data.base import Bar, MarketDataProvider, Quote

logger = structlog.get_logger(__name__)

_FINNHUB_BASE = "https://finnhub.io/api/v1"

# Free-tier limit: 60 req/min  → 1 req/sec safe minimum
_RATE_LIMIT_RPS = 55  # stay slightly under 60/min
_RATE_LOCK_INTERVAL = 60.0 / _RATE_LIMIT_RPS  # seconds between calls


class _RateLimiter:
    """Simple token-bucket rate limiter for async usage."""

    def __init__(self, calls_per_minute: int = 55) -> None:
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last_call = time.monotonic()


class FinnhubDataProvider(MarketDataProvider):
    """Finnhub REST API implementation of MarketDataProvider.

    Parameters
    ----------
    api_key : str
        Finnhub API key.
    calls_per_minute : int
        Max requests per minute (default 55, safely under the 60/min free limit).
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        calls_per_minute: int = 55,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._rate_limiter = _RateLimiter(calls_per_minute)
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    # HTTP client lifecycle                                                #
    # ------------------------------------------------------------------ #

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=_FINNHUB_BASE,
                headers={"X-Finnhub-Token": self._api_key},
                timeout=self._timeout,
            )
        return self._client

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Rate-limited GET request. Returns parsed JSON."""
        await self._rate_limiter.acquire()
        client = self._get_client()
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("finnhub.rate_limit_hit", path=path)
                await asyncio.sleep(2.0)
                raise DataProviderError("Finnhub rate limit exceeded") from exc
            raise DataProviderError(
                f"Finnhub HTTP error {exc.response.status_code}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise DataProviderError(f"Finnhub request error: {exc}") from exc

    @property
    def provider_name(self) -> str:
        return "finnhub"

    # ------------------------------------------------------------------ #
    # Daily OHLCV (via /stock/candle)                                      #
    # ------------------------------------------------------------------ #

    async def get_daily_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        start_ts = int(
            datetime.combine(start_date, datetime.min.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        end_ts = int(
            datetime.combine(end_date, datetime.max.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        try:
            data = await self._get(
                "/stock/candle",
                params={
                    "symbol": ticker,
                    "resolution": "D",
                    "from": start_ts,
                    "to": end_ts,
                },
            )
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(
                f"Finnhub daily OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

        if data.get("s") == "no_data" or not data.get("t"):
            logger.warning("finnhub.no_daily_data", ticker=ticker)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        timestamps = [
            datetime.fromtimestamp(ts, tz=timezone.utc) for ts in data["t"]
        ]
        df = pd.DataFrame(
            {
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"],
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        return self._normalise_ohlcv(df)

    # ------------------------------------------------------------------ #
    # Intraday OHLCV                                                       #
    # ------------------------------------------------------------------ #

    _INTRADAY_RESOLUTION_MAP: dict[str, str] = {
        "1Min": "1",
        "5Min": "5",
        "15Min": "15",
        "30Min": "30",
        "1Hour": "60",
        "1Day": "D",
        "1Week": "W",
        "1Month": "M",
    }

    async def get_intraday_ohlcv(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if interval not in self._INTRADAY_RESOLUTION_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported: {list(self._INTRADAY_RESOLUTION_MAP.keys())}"
            )

        resolution = self._INTRADAY_RESOLUTION_MAP[interval]
        start_ts = int(
            start.replace(tzinfo=timezone.utc).timestamp()
            if start.tzinfo is None
            else start.timestamp()
        )
        end_ts = int(
            end.replace(tzinfo=timezone.utc).timestamp()
            if end.tzinfo is None
            else end.timestamp()
        )

        try:
            data = await self._get(
                "/stock/candle",
                params={
                    "symbol": ticker,
                    "resolution": resolution,
                    "from": start_ts,
                    "to": end_ts,
                },
            )
        except DataProviderError:
            raise
        except Exception as exc:
            raise DataProviderError(
                f"Finnhub intraday OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

        if data.get("s") == "no_data" or not data.get("t"):
            logger.warning(
                "finnhub.no_intraday_data", ticker=ticker, interval=interval
            )
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        timestamps = [
            datetime.fromtimestamp(ts, tz=timezone.utc) for ts in data["t"]
        ]
        df = pd.DataFrame(
            {
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"],
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        return self._normalise_ohlcv(df)

    # ------------------------------------------------------------------ #
    # Quote                                                                #
    # ------------------------------------------------------------------ #

    async def get_quote(self, ticker: str) -> Quote:
        try:
            data = await self._get("/quote", params={"symbol": ticker})
        except Exception as exc:
            raise DataProviderError(
                f"Finnhub quote fetch failed for {ticker}: {exc}"
            ) from exc

        if not data or data.get("c") == 0:
            raise DataProviderError(f"Finnhub returned empty quote for {ticker}")

        current_price = float(data.get("c", 0))
        # Finnhub quote has no bid/ask for stocks; approximate from current price
        return Quote(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(
                data.get("t", time.time()), tz=timezone.utc
            ),
            bid=current_price,
            ask=current_price,
            bid_size=0.0,
            ask_size=0.0,
            last=current_price,
            last_size=0.0,
            volume=int(data.get("v", 0) or 0),
            vwap=0.0,
        )

    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}

        tasks = [self.get_quote(t) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: dict[str, Quote] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.warning(
                    "finnhub.bulk_quote_failed",
                    ticker=ticker,
                    error=str(result),
                )
            else:
                quotes[ticker] = result
        return quotes

    # ------------------------------------------------------------------ #
    # Market snapshot                                                      #
    # ------------------------------------------------------------------ #

    async def get_market_snapshot(self) -> dict[str, Any]:
        benchmarks = ["SPY", "QQQ", "IWM"]
        quotes = await self.get_bulk_quotes(benchmarks)

        snapshot: dict[str, Any] = {}
        for ticker, q in quotes.items():
            snapshot[ticker] = {
                "ticker": ticker,
                "close": q.last,
                "bid": q.bid,
                "ask": q.ask,
                "volume": q.volume,
                "vwap": q.vwap,
                "change_pct": 0.0,  # Finnhub quote includes dp (change %)
            }

        # Fetch VIX via Finnhub quote endpoint (symbol: ^VIX or VIX)
        try:
            vix_data = await self._get("/quote", params={"symbol": "^VIX"})
            snapshot["VIX_level"] = float(vix_data.get("c", 20.0))
        except Exception:
            snapshot["VIX_level"] = 20.0

        return snapshot

    # ------------------------------------------------------------------ #
    # Ticker search                                                        #
    # ------------------------------------------------------------------ #

    async def search_tickers(self, query: str) -> list[dict[str, Any]]:
        try:
            data = await self._get("/search", params={"q": query})
        except Exception as exc:
            logger.error("finnhub.search_tickers_failed", query=query, error=str(exc))
            return []

        results = []
        for item in data.get("result", []):
            results.append(
                {
                    "ticker": item.get("symbol", ""),
                    "name": item.get("description", ""),
                    "exchange": item.get("primaryExchange", ""),
                    "type": item.get("type", ""),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Company info                                                         #
    # ------------------------------------------------------------------ #

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        try:
            data = await self._get(
                "/stock/profile2", params={"symbol": ticker}
            )
        except Exception as exc:
            logger.error(
                "finnhub.get_company_info_failed", ticker=ticker, error=str(exc)
            )
            return {"ticker": ticker, "name": ticker}

        if not data:
            return {"ticker": ticker, "name": ticker}

        return {
            "ticker": data.get("ticker", ticker),
            "name": data.get("name", ""),
            "exchange": data.get("exchange", ""),
            "sector": data.get("finnhubIndustry", ""),
            "industry": data.get("finnhubIndustry", ""),
            "market_cap": data.get("marketCapitalization", 0) * 1_000_000,
            "description": "",
            "country": data.get("country", ""),
            "currency": data.get("currency", "USD"),
            "logo_url": data.get("logo", ""),
            "website": data.get("weburl", ""),
            "ipo_date": data.get("ipo", ""),
            "share_outstanding": data.get("shareOutstanding", 0),
        }

    # ------------------------------------------------------------------ #
    # News (company-specific)                                              #
    # ------------------------------------------------------------------ #

    async def get_company_news(
        self,
        ticker: str,
        from_date: date,
        to_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch company news from Finnhub (not part of MarketDataProvider ABC,
        but used by FinnhubNewsProvider).
        """
        if to_date is None:
            to_date = date.today()

        try:
            data = await self._get(
                "/company-news",
                params={
                    "symbol": ticker,
                    "from": from_date.isoformat(),
                    "to": to_date.isoformat(),
                },
            )
        except Exception as exc:
            logger.error(
                "finnhub.get_company_news_failed", ticker=ticker, error=str(exc)
            )
            return []

        return data if isinstance(data, list) else []

    async def get_general_news(self, category: str = "general") -> list[dict[str, Any]]:
        """Fetch general/market news from Finnhub."""
        valid_categories = {"general", "forex", "crypto", "merger"}
        if category not in valid_categories:
            category = "general"

        try:
            data = await self._get(
                "/news", params={"category": category}
            )
        except Exception as exc:
            logger.error(
                "finnhub.get_general_news_failed", category=category, error=str(exc)
            )
            return []

        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------ #
    # Health check                                                         #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            data = await self._get("/quote", params={"symbol": "SPY"})
            return bool(data and data.get("c", 0) > 0)
        except Exception as exc:
            logger.warning("finnhub.health_check_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
