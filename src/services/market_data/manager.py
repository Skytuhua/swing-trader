"""DataManager – primary/fallback provider orchestration with Redis caching.

Responsibilities:
  - Try primary provider; fall back to secondary on failure.
  - Cache all results in Redis with configurable TTL.
  - Detect stale data and emit structured warnings.
  - Score data quality per response.
  - Bulk-fetch with configurable concurrency limits.
  - Provide get_enriched_bars() with dollar_volume and avg_dollar_volume.
  - Provide get_universe_snapshot() for all universe tickers.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.core.enums import DataQuality
from src.core.exceptions import DataProviderError, StaleDataError
from src.services.market_data.base import MarketDataProvider, Quote

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cache TTL defaults (seconds)
# ---------------------------------------------------------------------------

_TTL_DAILY_BARS = 3600 * 4         # 4 hours for daily bars (end-of-day stable)
_TTL_INTRADAY_BARS = 60 * 5        # 5 minutes
_TTL_QUOTE = 15                    # 15 seconds
_TTL_SNAPSHOT = 30                 # 30 seconds
_TTL_COMPANY_INFO = 3600 * 24 * 7  # 7 days

# Stale-data thresholds
_STALE_DAILY_HOURS = 72            # Daily bar older than 72 hours → stale
_STALE_QUOTE_SECONDS = 60          # Quote older than 60 seconds → stale


class _SerdesHelper:
    """JSON serialisation helpers for DataFrames and Quotes."""

    @staticmethod
    def df_to_json(df: pd.DataFrame) -> str:
        """Serialise DataFrame to JSON, handling Timestamps properly."""
        return df.to_json(orient="split", date_format="iso")

    @staticmethod
    def df_from_json(raw: str) -> pd.DataFrame:
        """Deserialise DataFrame from JSON."""
        df = pd.read_json(raw, orient="split")
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                pass
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = "timestamp"
        return df

    @staticmethod
    def quote_to_dict(q: Quote) -> dict[str, Any]:
        return q.to_dict()

    @staticmethod
    def quote_from_dict(d: dict[str, Any]) -> Quote:
        return Quote(
            ticker=d["ticker"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            bid=d["bid"],
            ask=d["ask"],
            bid_size=d.get("bid_size", 0.0),
            ask_size=d.get("ask_size", 0.0),
            last=d.get("last", 0.0),
            last_size=d.get("last_size", 0.0),
            volume=d.get("volume", 0),
            vwap=d.get("vwap", 0.0),
        )


class DataManager:
    """Manages market data access across one or two providers.

    Parameters
    ----------
    primary : MarketDataProvider
        Primary data source (e.g. Alpaca).
    fallback : MarketDataProvider, optional
        Fallback source used when primary fails (e.g. yfinance).
    redis_client : optional
        An async Redis client (aioredis / redis-py async). If None,
        caching is disabled (in-process dict used as a minimal stand-in).
    cache_prefix : str
        Prefix for all Redis cache keys.
    max_concurrent_fetches : int
        Semaphore limit for bulk fetch concurrency.
    stale_daily_hours : int
        How many hours before a daily-bar cache entry is considered stale.
    stale_quote_seconds : int
        How many seconds before a quote is considered stale.
    """

    def __init__(
        self,
        primary: MarketDataProvider,
        fallback: MarketDataProvider | None = None,
        redis_client: Any = None,
        cache_prefix: str = "mktdata",
        max_concurrent_fetches: int = 10,
        stale_daily_hours: int = _STALE_DAILY_HOURS,
        stale_quote_seconds: int = _STALE_QUOTE_SECONDS,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self._redis = redis_client
        self._prefix = cache_prefix
        self._sem = asyncio.Semaphore(max_concurrent_fetches)
        self._stale_daily_hours = stale_daily_hours
        self._stale_quote_seconds = stale_quote_seconds
        self._mem_cache: dict[str, tuple[float, Any]] = {}  # fallback when redis=None

    # ------------------------------------------------------------------ #
    # Cache helpers                                                        #
    # ------------------------------------------------------------------ #

    def _key(self, *parts: str) -> str:
        return ":".join([self._prefix] + list(parts))

    async def _cache_get(self, key: str) -> Any | None:
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception as exc:
                logger.warning("cache.get_failed", key=key, error=str(exc))
                return None
        else:
            entry = self._mem_cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._mem_cache[key]
                return None
            return value

    async def _cache_set(self, key: str, value: Any, ttl: int) -> None:
        if self._redis is not None:
            try:
                await self._redis.setex(key, ttl, json.dumps(value, default=str))
            except Exception as exc:
                logger.warning("cache.set_failed", key=key, error=str(exc))
        else:
            self._mem_cache[key] = (time.monotonic() + ttl, value)

    async def _cache_delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(key)
            except Exception as exc:
                logger.warning("cache.delete_failed", key=key, error=str(exc))
        else:
            self._mem_cache.pop(key, None)

    # ------------------------------------------------------------------ #
    # Provider fallback helper                                             #
    # ------------------------------------------------------------------ #

    async def _with_fallback(
        self,
        primary_coro,
        fallback_coro,
        operation: str,
        ticker: str = "",
    ) -> tuple[Any, str]:
        """Try primary; fall back if it raises DataProviderError."""
        try:
            result = await primary_coro
            return result, self.primary.provider_name
        except (DataProviderError, Exception) as exc:
            logger.warning(
                "data_manager.primary_failed",
                provider=self.primary.provider_name,
                operation=operation,
                ticker=ticker,
                error=str(exc),
            )
            if self.fallback is not None:
                try:
                    result = await fallback_coro
                    return result, self.fallback.provider_name
                except Exception as exc2:
                    logger.error(
                        "data_manager.fallback_failed",
                        provider=self.fallback.provider_name,
                        operation=operation,
                        ticker=ticker,
                        error=str(exc2),
                    )
                    raise DataProviderError(
                        f"Both providers failed for {operation} {ticker}: {exc2}"
                    ) from exc2
            raise

    # ------------------------------------------------------------------ #
    # Daily OHLCV                                                          #
    # ------------------------------------------------------------------ #

    async def get_daily_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_key = self._key(
            "daily", ticker, start_date.isoformat(), end_date.isoformat()
        )

        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                df = _SerdesHelper.df_from_json(cached)
                logger.debug(
                    "data_manager.cache_hit", key="daily_ohlcv", ticker=ticker
                )
                return df

        df, source = await self._with_fallback(
            self.primary.get_daily_ohlcv(ticker, start_date, end_date),
            self.fallback.get_daily_ohlcv(ticker, start_date, end_date)
            if self.fallback
            else self.primary.get_daily_ohlcv(ticker, start_date, end_date),
            operation="get_daily_ohlcv",
            ticker=ticker,
        )

        if not df.empty:
            await self._cache_set(
                cache_key, _SerdesHelper.df_to_json(df), _TTL_DAILY_BARS
            )

        logger.info(
            "data_manager.daily_ohlcv_fetched",
            ticker=ticker,
            rows=len(df),
            source=source,
        )
        return df

    # ------------------------------------------------------------------ #
    # Intraday OHLCV                                                       #
    # ------------------------------------------------------------------ #

    async def get_intraday_ohlcv(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_key = self._key(
            "intraday",
            ticker,
            interval,
            start.strftime("%Y%m%d%H%M"),
            end.strftime("%Y%m%d%H%M"),
        )

        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                return _SerdesHelper.df_from_json(cached)

        df, source = await self._with_fallback(
            self.primary.get_intraday_ohlcv(ticker, interval, start, end),
            self.fallback.get_intraday_ohlcv(ticker, interval, start, end)
            if self.fallback
            else self.primary.get_intraday_ohlcv(ticker, interval, start, end),
            operation="get_intraday_ohlcv",
            ticker=ticker,
        )

        if not df.empty:
            await self._cache_set(
                cache_key, _SerdesHelper.df_to_json(df), _TTL_INTRADAY_BARS
            )

        return df

    # ------------------------------------------------------------------ #
    # Quote                                                                #
    # ------------------------------------------------------------------ #

    async def get_quote(self, ticker: str, force_refresh: bool = False) -> Quote:
        cache_key = self._key("quote", ticker)

        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                q = _SerdesHelper.quote_from_dict(cached)
                age = (datetime.now(timezone.utc) - q.timestamp).total_seconds()
                if age < self._stale_quote_seconds:
                    return q
                logger.debug("data_manager.stale_quote", ticker=ticker, age_s=age)

        quote, source = await self._with_fallback(
            self.primary.get_quote(ticker),
            self.fallback.get_quote(ticker) if self.fallback else self.primary.get_quote(ticker),
            operation="get_quote",
            ticker=ticker,
        )

        await self._cache_set(
            cache_key, _SerdesHelper.quote_to_dict(quote), _TTL_QUOTE
        )
        return quote

    # ------------------------------------------------------------------ #
    # Bulk quotes                                                          #
    # ------------------------------------------------------------------ #

    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}

        # Check cache first, identify misses
        result: dict[str, Quote] = {}
        misses: list[str] = []

        for ticker in tickers:
            cached = await self._cache_get(self._key("quote", ticker))
            if cached is not None:
                q = _SerdesHelper.quote_from_dict(cached)
                age = (datetime.now(timezone.utc) - q.timestamp).total_seconds()
                if age < self._stale_quote_seconds:
                    result[ticker] = q
                    continue
            misses.append(ticker)

        if not misses:
            return result

        # Fetch misses in bulk
        try:
            fetched = await self.primary.get_bulk_quotes(misses)
        except Exception as exc:
            logger.warning(
                "data_manager.bulk_quotes_primary_failed",
                error=str(exc),
                count=len(misses),
            )
            fetched = {}
            if self.fallback:
                try:
                    fetched = await self.fallback.get_bulk_quotes(misses)
                except Exception as exc2:
                    logger.error(
                        "data_manager.bulk_quotes_fallback_failed", error=str(exc2)
                    )

        for ticker, q in fetched.items():
            result[ticker] = q
            await self._cache_set(
                self._key("quote", ticker),
                _SerdesHelper.quote_to_dict(q),
                _TTL_QUOTE,
            )

        return result

    # ------------------------------------------------------------------ #
    # Market snapshot                                                      #
    # ------------------------------------------------------------------ #

    async def get_market_snapshot(self, force_refresh: bool = False) -> dict[str, Any]:
        cache_key = self._key("snapshot", "market")

        if not force_refresh:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                return cached

        snap, source = await self._with_fallback(
            self.primary.get_market_snapshot(),
            self.fallback.get_market_snapshot()
            if self.fallback
            else self.primary.get_market_snapshot(),
            operation="get_market_snapshot",
        )

        await self._cache_set(cache_key, snap, _TTL_SNAPSHOT)
        return snap

    # ------------------------------------------------------------------ #
    # Company info                                                         #
    # ------------------------------------------------------------------ #

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        cache_key = self._key("company", ticker)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        info, _ = await self._with_fallback(
            self.primary.get_company_info(ticker),
            self.fallback.get_company_info(ticker)
            if self.fallback
            else self.primary.get_company_info(ticker),
            operation="get_company_info",
            ticker=ticker,
        )

        await self._cache_set(cache_key, info, _TTL_COMPANY_INFO)
        return info

    # ------------------------------------------------------------------ #
    # Enriched bars                                                        #
    # ------------------------------------------------------------------ #

    async def get_enriched_bars(
        self,
        ticker: str,
        lookback_days: int = 252,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV and add dollar_volume and rolling avg_dollar_volume.

        Parameters
        ----------
        lookback_days : int
            Number of calendar days to look back (default 252 ≈ 1 trading year).
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days + 30)  # buffer

        df = await self.get_daily_ohlcv(
            ticker, start_date, end_date, force_refresh=force_refresh
        )

        if df.empty:
            return df

        df = df.copy()
        df["dollar_volume"] = df["close"] * df["volume"]
        df["avg_dollar_volume"] = df["dollar_volume"].rolling(window=20, min_periods=5).mean()

        # Trim to requested lookback
        cutoff = pd.Timestamp(end_date - timedelta(days=lookback_days), tz="UTC")
        df = df[df.index >= cutoff]

        return df

    # ------------------------------------------------------------------ #
    # Universe snapshot                                                    #
    # ------------------------------------------------------------------ #

    async def get_universe_snapshot(
        self, tickers: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch current data for all tickers in the universe.

        Returns a dict mapping ticker → dict with keys: quote, enriched_bars,
        data_quality.
        """
        if not tickers:
            return {}

        # Bulk fetch quotes first
        quotes = await self.get_bulk_quotes(tickers)

        # Fetch enriched bars concurrently with concurrency limit
        async def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
            async with self._sem:
                try:
                    df = await self.get_enriched_bars(ticker, lookback_days=90)
                    return ticker, df
                except Exception as exc:
                    logger.warning(
                        "data_manager.universe_bars_failed",
                        ticker=ticker,
                        error=str(exc),
                    )
                    return ticker, pd.DataFrame()

        bar_tasks = [_fetch_one(t) for t in tickers]
        bar_results = await asyncio.gather(*bar_tasks)
        bars_map: dict[str, pd.DataFrame] = dict(bar_results)

        result: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            quote = quotes.get(ticker)
            bars = bars_map.get(ticker, pd.DataFrame())
            quality = self.score_data_quality(ticker, quote, bars)

            result[ticker] = {
                "quote": quote.to_dict() if quote else None,
                "bars": bars,
                "data_quality": quality.value,
                "last_price": quote.last if quote else None,
                "avg_dollar_volume": (
                    float(bars["avg_dollar_volume"].iloc[-1])
                    if not bars.empty and "avg_dollar_volume" in bars.columns
                    else None
                ),
            }

        return result

    # ------------------------------------------------------------------ #
    # Bulk OHLCV fetch                                                     #
    # ------------------------------------------------------------------ #

    async def get_bulk_daily_ohlcv(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily OHLCV for many tickers concurrently."""

        async def _fetch(ticker: str) -> tuple[str, pd.DataFrame]:
            async with self._sem:
                try:
                    df = await self.get_daily_ohlcv(ticker, start_date, end_date)
                    return ticker, df
                except Exception as exc:
                    logger.warning(
                        "data_manager.bulk_daily_failed",
                        ticker=ticker,
                        error=str(exc),
                    )
                    return ticker, pd.DataFrame()

        results = await asyncio.gather(*[_fetch(t) for t in tickers])
        return {ticker: df for ticker, df in results if not df.empty}

    # ------------------------------------------------------------------ #
    # Data quality scoring                                                 #
    # ------------------------------------------------------------------ #

    def score_data_quality(
        self,
        ticker: str,
        quote: Quote | None,
        bars: pd.DataFrame,
    ) -> DataQuality:
        """Assign a DataQuality enum value based on data freshness and completeness."""
        if quote is None and bars.empty:
            return DataQuality.UNAVAILABLE

        issues: list[str] = []

        # Quote freshness
        if quote is not None:
            age_s = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
            if age_s > 3600:  # > 1 hour old
                issues.append(f"stale_quote:{age_s:.0f}s")

        # Bars completeness
        if not bars.empty:
            last_bar_age = (
                pd.Timestamp.now(tz="UTC") - bars.index[-1]
            ).total_seconds() / 3600

            if last_bar_age > self._stale_daily_hours:
                issues.append(f"stale_bars:{last_bar_age:.0f}h")

            # Check for NaN in close
            nan_pct = bars["close"].isna().mean()
            if nan_pct > 0.05:
                issues.append(f"nan_close:{nan_pct:.1%}")

            # Minimum bar count
            if len(bars) < 20:
                issues.append(f"insufficient_bars:{len(bars)}")

        if not issues:
            return DataQuality.GOOD
        if len(issues) == 1 and any(
            i.startswith("stale_quote") or i.startswith("insufficient") for i in issues
        ):
            return DataQuality.DEGRADED
        return DataQuality.STALE

    def score_data_quality_numeric(
        self, ticker: str, quote: Quote | None, bars: pd.DataFrame
    ) -> float:
        """Return a 0–100 numeric quality score (100 = perfect)."""
        quality = self.score_data_quality(ticker, quote, bars)
        mapping = {
            DataQuality.GOOD: 100.0,
            DataQuality.DEGRADED: 65.0,
            DataQuality.STALE: 30.0,
            DataQuality.UNAVAILABLE: 0.0,
        }
        return mapping[quality]

    # ------------------------------------------------------------------ #
    # Health check                                                         #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        results[self.primary.provider_name] = await self.primary.health_check()
        if self.fallback:
            results[self.fallback.provider_name] = await self.fallback.health_check()
        return results
