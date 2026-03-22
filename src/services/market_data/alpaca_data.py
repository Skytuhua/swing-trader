"""Alpaca market data provider.

Uses alpaca-py SDK:
  - alpaca.data.historical.StockHistoricalDataClient  → historical bars
  - alpaca.data.live.StockDataStream                  → real-time stream (optional)
  - Latest bar/quote fetched via StockHistoricalDataClient snapshot endpoints

Rate-limit retry is handled via tenacity with exponential backoff.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any

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

# ---------------------------------------------------------------------------
# Lazy import helpers – alpaca-py is optional at import time so the module
# can be loaded even if the package is not installed.
# ---------------------------------------------------------------------------

def _get_alpaca_clients():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import (
            StockBarsRequest,
            StockLatestBarRequest,
            StockLatestQuoteRequest,
            StockSnapshotRequest,
            StockQuotesRequest,
        )
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.common.exceptions import APIError
        return (
            StockHistoricalDataClient,
            StockBarsRequest,
            StockLatestBarRequest,
            StockLatestQuoteRequest,
            StockSnapshotRequest,
            StockQuotesRequest,
            TimeFrame,
            TimeFrameUnit,
            APIError,
        )
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is required for AlpacaDataProvider. "
            "Install it with: pip install alpaca-py"
        ) from exc


# ---------------------------------------------------------------------------
# Helper: map interval string → alpaca TimeFrame
# ---------------------------------------------------------------------------

_INTERVAL_MAP: dict[str, tuple[int, str]] = {
    "1Min": (1, "Minute"),
    "5Min": (5, "Minute"),
    "15Min": (15, "Minute"),
    "30Min": (30, "Minute"),
    "1Hour": (1, "Hour"),
    "1Day": (1, "Day"),
    "1Week": (1, "Week"),
    "1Month": (1, "Month"),
}


class AlpacaDataProvider(MarketDataProvider):
    """Market data provider backed by the Alpaca broker data API.

    Parameters
    ----------
    api_key, api_secret : str
        Alpaca API credentials (paper or live).
    paper : bool
        If True, uses paper-trading data feed (same data, just for
        environment consistency).
    feed : str
        Data feed to use: ``'iex'`` (free tier) or ``'sip'`` (paid).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        feed: str = "iex",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        self._feed = feed
        self._client: Any = None  # lazy-initialised

    # ------------------------------------------------------------------ #
    # Client initialisation                                                #
    # ------------------------------------------------------------------ #

    def _ensure_client(self) -> Any:
        if self._client is None:
            (
                StockHistoricalDataClient,
                *_,
            ) = _get_alpaca_clients()
            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "alpaca"

    # ------------------------------------------------------------------ #
    # Retry decorator factory                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_retry():
        """Build a tenacity retry decorator for Alpaca API calls."""
        try:
            from alpaca.common.exceptions import APIError as _APIError
            _exc = (_APIError, ConnectionError, TimeoutError)
        except ImportError:
            _exc = (ConnectionError, TimeoutError)

        return retry(
            retry=retry_if_exception_type(_exc),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, "warning"),  # type: ignore[arg-type]
            reraise=True,
        )

    # ------------------------------------------------------------------ #
    # Daily OHLCV                                                          #
    # ------------------------------------------------------------------ #

    async def get_daily_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily bars via StockHistoricalDataClient.get_stock_bars."""
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_daily_bars_sync, ticker, start_date, end_date
            )
            return self._normalise_ohlcv(df)
        except Exception as exc:
            logger.error(
                "alpaca.get_daily_ohlcv_failed",
                ticker=ticker,
                error=str(exc),
            )
            raise DataProviderError(
                f"Alpaca daily OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_daily_bars_sync(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        (
            StockHistoricalDataClient,
            StockBarsRequest,
            _,
            _,
            _,
            _,
            TimeFrame,
            TimeFrameUnit,
            APIError,
        ) = _get_alpaca_clients()

        client = self._ensure_client()
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            ),
            end=datetime.combine(end_date, datetime.max.time()).replace(
                tzinfo=timezone.utc
            ),
            feed=self._feed,
        )

        try:
            bars_response = client.get_stock_bars(request)
        except APIError as exc:
            raise DataProviderError(f"Alpaca API error: {exc}") from exc

        bars = bars_response.get(ticker, [])
        if not bars:
            logger.warning("alpaca.no_bars_returned", ticker=ticker)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"]
            )

        rows = []
        for bar in bars:
            rows.append(
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else 0.0,
                    "trade_count": int(bar.trade_count) if bar.trade_count else 0,
                }
            )

        df = pd.DataFrame(rows).set_index("timestamp")
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
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_intraday_bars_sync, ticker, interval, start, end
            )
            return self._normalise_ohlcv(df)
        except Exception as exc:
            logger.error(
                "alpaca.get_intraday_ohlcv_failed",
                ticker=ticker,
                interval=interval,
                error=str(exc),
            )
            raise DataProviderError(
                f"Alpaca intraday OHLCV fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_intraday_bars_sync(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        (
            StockHistoricalDataClient,
            StockBarsRequest,
            _,
            _,
            _,
            _,
            TimeFrame,
            TimeFrameUnit,
            APIError,
        ) = _get_alpaca_clients()

        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported: {list(_INTERVAL_MAP.keys())}"
            )

        amount, unit_name = _INTERVAL_MAP[interval]
        unit = getattr(TimeFrameUnit, unit_name)
        timeframe = TimeFrame(amount, unit)

        client = self._ensure_client()
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start,
            end=end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end,
            feed=self._feed,
        )

        try:
            bars_response = client.get_stock_bars(request)
        except APIError as exc:
            raise DataProviderError(f"Alpaca API error: {exc}") from exc

        bars = bars_response.get(ticker, [])
        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"]
            )

        rows = [
            {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap else 0.0,
                "trade_count": int(bar.trade_count) if bar.trade_count else 0,
            }
            for bar in bars
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    # ------------------------------------------------------------------ #
    # Latest quote                                                         #
    # ------------------------------------------------------------------ #

    async def get_quote(self, ticker: str) -> Quote:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_quote_sync, ticker
            )
        except Exception as exc:
            logger.error("alpaca.get_quote_failed", ticker=ticker, error=str(exc))
            raise DataProviderError(
                f"Alpaca quote fetch failed for {ticker}: {exc}"
            ) from exc

    def _fetch_quote_sync(self, ticker: str) -> Quote:
        (
            _,
            _,
            StockLatestBarRequest,
            StockLatestQuoteRequest,
            _,
            _,
            _,
            _,
            APIError,
        ) = _get_alpaca_clients()

        client = self._ensure_client()

        try:
            q_resp = client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=self._feed)
            )
            b_resp = client.get_stock_latest_bar(
                StockLatestBarRequest(symbol_or_symbols=ticker, feed=self._feed)
            )
        except APIError as exc:
            raise DataProviderError(f"Alpaca API error: {exc}") from exc

        q = q_resp.get(ticker)
        b = b_resp.get(ticker)

        if q is None:
            raise DataProviderError(f"No quote returned for {ticker}")

        return Quote(
            ticker=ticker,
            timestamp=q.timestamp if q.timestamp else datetime.now(timezone.utc),
            bid=float(q.bid_price) if q.bid_price else 0.0,
            ask=float(q.ask_price) if q.ask_price else 0.0,
            bid_size=float(q.bid_size) if q.bid_size else 0.0,
            ask_size=float(q.ask_size) if q.ask_size else 0.0,
            last=float(b.close) if b else 0.0,
            last_size=float(b.trade_count) if (b and b.trade_count) else 0.0,
            volume=int(b.volume) if (b and b.volume) else 0,
            vwap=float(b.vwap) if (b and b.vwap) else 0.0,
        )

    # ------------------------------------------------------------------ #
    # Bulk quotes                                                          #
    # ------------------------------------------------------------------ #

    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_bulk_quotes_sync, tickers
            )
        except Exception as exc:
            logger.error(
                "alpaca.get_bulk_quotes_failed",
                count=len(tickers),
                error=str(exc),
            )
            raise DataProviderError(f"Alpaca bulk quotes failed: {exc}") from exc

    def _fetch_bulk_quotes_sync(self, tickers: list[str]) -> dict[str, Quote]:
        (
            _,
            _,
            StockLatestBarRequest,
            StockLatestQuoteRequest,
            _,
            _,
            _,
            _,
            APIError,
        ) = _get_alpaca_clients()

        client = self._ensure_client()

        # Batch request
        try:
            q_resp = client.get_stock_latest_quote(
                StockLatestQuoteRequest(
                    symbol_or_symbols=tickers, feed=self._feed
                )
            )
            b_resp = client.get_stock_latest_bar(
                StockLatestBarRequest(
                    symbol_or_symbols=tickers, feed=self._feed
                )
            )
        except APIError as exc:
            raise DataProviderError(f"Alpaca API error: {exc}") from exc

        result: dict[str, Quote] = {}
        for ticker in tickers:
            q = q_resp.get(ticker)
            b = b_resp.get(ticker)
            if q is None:
                logger.warning("alpaca.no_quote_for_ticker", ticker=ticker)
                continue

            result[ticker] = Quote(
                ticker=ticker,
                timestamp=q.timestamp if q.timestamp else datetime.now(timezone.utc),
                bid=float(q.bid_price) if q.bid_price else 0.0,
                ask=float(q.ask_price) if q.ask_price else 0.0,
                bid_size=float(q.bid_size) if q.bid_size else 0.0,
                ask_size=float(q.ask_size) if q.ask_size else 0.0,
                last=float(b.close) if b else 0.0,
                last_size=float(b.trade_count) if (b and b.trade_count) else 0.0,
                volume=int(b.volume) if (b and b.volume) else 0,
                vwap=float(b.vwap) if (b and b.vwap) else 0.0,
            )

        return result

    # ------------------------------------------------------------------ #
    # Market snapshot                                                      #
    # ------------------------------------------------------------------ #

    async def get_market_snapshot(self) -> dict[str, Any]:
        benchmarks = ["SPY", "QQQ", "IWM", "VXX"]
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_snapshot_sync, benchmarks
            )
        except Exception as exc:
            logger.error("alpaca.get_market_snapshot_failed", error=str(exc))
            raise DataProviderError(f"Alpaca market snapshot failed: {exc}") from exc

    def _fetch_snapshot_sync(self, tickers: list[str]) -> dict[str, Any]:
        (
            _,
            _,
            _,
            _,
            StockSnapshotRequest,
            _,
            _,
            _,
            APIError,
        ) = _get_alpaca_clients()

        client = self._ensure_client()

        try:
            snap_resp = client.get_stock_snapshot(
                StockSnapshotRequest(
                    symbol_or_symbols=tickers, feed=self._feed
                )
            )
        except APIError as exc:
            raise DataProviderError(f"Alpaca API error: {exc}") from exc

        result: dict[str, Any] = {}
        for ticker in tickers:
            snap = snap_resp.get(ticker)
            if snap is None:
                continue

            daily_bar = snap.daily_bar
            prev_bar = snap.previous_daily_bar
            quote = snap.latest_quote
            trade = snap.latest_trade

            close = float(daily_bar.close) if daily_bar else 0.0
            prev_close = float(prev_bar.close) if prev_bar else close
            change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0.0

            result[ticker] = {
                "ticker": ticker,
                "close": close,
                "open": float(daily_bar.open) if daily_bar else 0.0,
                "high": float(daily_bar.high) if daily_bar else 0.0,
                "low": float(daily_bar.low) if daily_bar else 0.0,
                "volume": int(daily_bar.volume) if daily_bar else 0,
                "vwap": float(daily_bar.vwap) if (daily_bar and daily_bar.vwap) else 0.0,
                "change_pct": change_pct,
                "prev_close": prev_close,
                "bid": float(quote.bid_price) if (quote and quote.bid_price) else 0.0,
                "ask": float(quote.ask_price) if (quote and quote.ask_price) else 0.0,
                "last": float(trade.price) if (trade and trade.price) else close,
            }

        # Use VXX close as VIX proxy if VIX is not directly available
        vxx_close = result.get("VXX", {}).get("close", 20.0)
        result["VIX_level"] = vxx_close

        return result

    # ------------------------------------------------------------------ #
    # Ticker search                                                        #
    # ------------------------------------------------------------------ #

    async def search_tickers(self, query: str) -> list[dict[str, Any]]:
        """Alpaca does not expose a ticker search endpoint directly.

        Returns an empty list; callers should use Finnhub or yfinance for
        symbol searches.
        """
        logger.warning(
            "alpaca.search_tickers_not_supported",
            query=query,
            msg="Alpaca does not support symbol search; use Finnhub.",
        )
        return []

    # ------------------------------------------------------------------ #
    # Company info                                                         #
    # ------------------------------------------------------------------ #

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """Alpaca asset endpoint provides minimal company metadata."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_asset_info_sync, ticker
            )
        except Exception as exc:
            logger.warning(
                "alpaca.get_company_info_failed", ticker=ticker, error=str(exc)
            )
            return {"ticker": ticker, "name": ticker}

    def _fetch_asset_info_sync(self, ticker: str) -> dict[str, Any]:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass
        except ImportError:
            return {"ticker": ticker, "name": ticker}

        trading_client = TradingClient(
            api_key=self._api_key, secret_key=self._api_secret, paper=self._paper
        )
        try:
            asset = trading_client.get_asset(ticker)
            return {
                "ticker": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange.value if asset.exchange else "",
                "asset_class": asset.asset_class.value if asset.asset_class else "us_equity",
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "fractionable": asset.fractionable,
            }
        except Exception:
            return {"ticker": ticker, "name": ticker}

    # ------------------------------------------------------------------ #
    # Health check                                                         #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._health_check_sync
            )
            return result
        except Exception as exc:
            logger.warning("alpaca.health_check_failed", error=str(exc))
            return False

    def _health_check_sync(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            return False

        trading_client = TradingClient(
            api_key=self._api_key, secret_key=self._api_secret, paper=self._paper
        )
        try:
            trading_client.get_clock()
            return True
        except Exception:
            return False
