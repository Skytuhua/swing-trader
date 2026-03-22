"""yfinance fallback market data provider.

No API key required. Suitable for daily OHLCV data and backtesting.
Slower than Alpaca/Finnhub; should be used as a fallback only.

Notes
-----
* yfinance uses Yahoo Finance, which is not an official API and may
  change without notice. Always prefer Alpaca or Finnhub in production.
* All requests are executed in a thread-pool executor because yfinance
  is synchronous.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone, timedelta
from typing import Any

import pandas as pd
import structlog

from src.core.exceptions import DataProviderError
from src.services.market_data.base import Bar, MarketDataProvider, Quote

logger = structlog.get_logger(__name__)

# yfinance interval mapping
_INTERVAL_MAP: dict[str, str] = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "30Min": "30m",
    "1Hour": "1h",
    "1Day": "1d",
    "1Week": "1wk",
    "1Month": "1mo",
}

# yfinance period for recent data snapshot
_SNAPSHOT_PERIOD = "2d"


class YFinanceDataProvider(MarketDataProvider):
    """yfinance-backed MarketDataProvider.

    Suitable as a no-API-key fallback for daily OHLCV data. Not recommended
    for live/paper trading critical paths.
    """

    def __init__(self) -> None:
        self._yf: Any = None  # lazy import

    def _yf_module(self) -> Any:
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError as exc:
                raise ImportError(
                    "yfinance is required for YFinanceDataProvider. "
                    "Install it with: pip install yfinance"
                ) from exc
        return self._yf

    @property
    def provider_name(self) -> str:
        return "yfinance"

    # ------------------------------------------------------------------ #
    # Daily OHLCV                                                          #
    # ------------------------------------------------------------------ #

    async def get_daily_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                self._fetch_daily_sync,
                ticker,
                start_date,
                end_date,
            )
            return self._normalise_ohlcv(df)
        except Exception as exc:
            logger.error(
                "yfinance.get_daily_ohlcv_failed", ticker=ticker, error=str(exc)
            )
            raise DataProviderError(
                f"yfinance daily OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_daily_sync(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        yf = self._yf_module()

        # yfinance end is exclusive, so add 1 day
        end_exclusive = end_date + timedelta(days=1)

        df = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=end_exclusive.isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if df.empty:
            logger.warning("yfinance.no_daily_data", ticker=ticker)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        # yfinance uses capitalised column names
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        # Drop multi-index if present (happens with single ticker too in newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        # Ensure we only keep OHLCV
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"
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
    ) -> pd.DataFrame:
        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported: {list(_INTERVAL_MAP.keys())}"
            )

        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                self._fetch_intraday_sync,
                ticker,
                interval,
                start,
                end,
            )
            return self._normalise_ohlcv(df)
        except Exception as exc:
            logger.error(
                "yfinance.get_intraday_ohlcv_failed",
                ticker=ticker,
                interval=interval,
                error=str(exc),
            )
            raise DataProviderError(
                f"yfinance intraday OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_intraday_sync(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        yf = self._yf_module()
        yf_interval = _INTERVAL_MAP[interval]

        df = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d %H:%M:%S"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"
        return df

    # ------------------------------------------------------------------ #
    # Quote                                                                #
    # ------------------------------------------------------------------ #

    async def get_quote(self, ticker: str) -> Quote:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_quote_sync, ticker
            )
        except Exception as exc:
            logger.error("yfinance.get_quote_failed", ticker=ticker, error=str(exc))
            raise DataProviderError(
                f"yfinance quote fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_quote_sync(self, ticker: str) -> Quote:
        yf = self._yf_module()
        t = yf.Ticker(ticker)
        info = t.fast_info

        try:
            last_price = float(info.last_price or 0)
            volume = int(info.three_month_average_volume or 0)
        except (AttributeError, TypeError):
            last_price = 0.0
            volume = 0

        # yfinance does not provide bid/ask; approximate
        return Quote(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            bid=last_price,
            ask=last_price,
            bid_size=0.0,
            ask_size=0.0,
            last=last_price,
            last_size=0.0,
            volume=volume,
            vwap=0.0,
        )

    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_bulk_quotes_sync, tickers
            )
        except Exception as exc:
            logger.error(
                "yfinance.get_bulk_quotes_failed",
                count=len(tickers),
                error=str(exc),
            )
            raise DataProviderError(f"yfinance bulk quotes failed: {exc}") from exc

    def _fetch_bulk_quotes_sync(self, tickers: list[str]) -> dict[str, Quote]:
        yf = self._yf_module()

        # Download last 2 days of data for all tickers at once (efficient)
        df = yf.download(
            tickers=" ".join(tickers),
            period=_SNAPSHOT_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by="ticker",
        )

        quotes: dict[str, Quote] = {}

        if df.empty:
            return quotes

        now = datetime.now(timezone.utc)

        if len(tickers) == 1:
            # Single ticker: flat columns
            ticker = tickers[0]
            df.columns = [c.lower() for c in df.columns]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            if not df.empty and "close" in df.columns:
                last_row = df.iloc[-1]
                quotes[ticker] = Quote(
                    ticker=ticker,
                    timestamp=now,
                    bid=float(last_row["close"]),
                    ask=float(last_row["close"]),
                    last=float(last_row["close"]),
                    volume=int(last_row.get("volume", 0)),
                )
        else:
            # Multi-ticker: multi-level columns
            for ticker in tickers:
                try:
                    sub = df[ticker]
                    sub.columns = [c.lower() for c in sub.columns]
                    if not sub.empty and "close" in sub.columns:
                        last_row = sub.iloc[-1]
                        quotes[ticker] = Quote(
                            ticker=ticker,
                            timestamp=now,
                            bid=float(last_row["close"]),
                            ask=float(last_row["close"]),
                            last=float(last_row["close"]),
                            volume=int(last_row.get("volume", 0)),
                        )
                except (KeyError, IndexError) as exc:
                    logger.warning(
                        "yfinance.bulk_quote_ticker_failed",
                        ticker=ticker,
                        error=str(exc),
                    )

        return quotes

    # ------------------------------------------------------------------ #
    # Market snapshot                                                      #
    # ------------------------------------------------------------------ #

    async def get_market_snapshot(self) -> dict[str, Any]:
        benchmarks = ["SPY", "QQQ", "IWM", "^VIX"]
        try:
            quotes = await self.get_bulk_quotes(benchmarks[:3])
        except Exception:
            quotes = {}

        # Fetch VIX separately
        vix_level = 20.0
        try:
            vix_df = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_vix_sync
            )
            if not vix_df.empty:
                vix_level = float(vix_df["close"].iloc[-1])
        except Exception:
            pass

        snapshot: dict[str, Any] = {}
        for ticker, q in quotes.items():
            snapshot[ticker] = {
                "ticker": ticker,
                "close": q.last,
                "bid": q.bid,
                "ask": q.ask,
                "volume": q.volume,
                "vwap": q.vwap,
                "change_pct": 0.0,
            }

        snapshot["VIX_level"] = vix_level
        return snapshot

    def _fetch_vix_sync(self) -> pd.DataFrame:
        yf = self._yf_module()
        df = yf.download(
            "^VIX", period="2d", interval="1d", progress=False, auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Ticker search                                                        #
    # ------------------------------------------------------------------ #

    async def search_tickers(self, query: str) -> list[dict[str, Any]]:
        """yfinance does not expose a search endpoint; returns an empty list."""
        logger.warning("yfinance.search_not_supported", query=query)
        return []

    # ------------------------------------------------------------------ #
    # Company info                                                         #
    # ------------------------------------------------------------------ #

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_info_sync, ticker
            )
        except Exception as exc:
            logger.warning(
                "yfinance.get_company_info_failed", ticker=ticker, error=str(exc)
            )
            return {"ticker": ticker, "name": ticker}

    def _fetch_info_sync(self, ticker: str) -> dict[str, Any]:
        yf = self._yf_module()
        t = yf.Ticker(ticker)
        info = t.info or {}

        return {
            "ticker": ticker,
            "name": info.get("longName", info.get("shortName", ticker)),
            "exchange": info.get("exchange", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "description": info.get("longBusinessSummary", ""),
            "country": info.get("country", ""),
            "currency": info.get("currency", "USD"),
            "logo_url": info.get("logo_url", ""),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees", 0),
            "pe_ratio": info.get("trailingPE", None),
            "beta": info.get("beta", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
        }

    # ------------------------------------------------------------------ #
    # Health check                                                         #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            yf = self._yf_module()
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.download(
                    "SPY", period="5d", interval="1d", progress=False
                ),
            )
            return not df.empty
        except Exception as exc:
            logger.warning("yfinance.health_check_failed", error=str(exc))
            return False
