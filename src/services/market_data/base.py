"""Abstract base class for market data providers.

All market data provider implementations must subclass MarketDataProvider
and implement every abstract method.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Quote:
    """Real-time or delayed quote for a single ticker."""

    ticker: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float = 0.0
    last_size: float = 0.0
    volume: int = 0
    vwap: float = 0.0

    @property
    def mid(self) -> float:
        """Mid-point price."""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last

    @property
    def spread(self) -> float:
        """Absolute spread."""
        return self.ask - self.bid if self.bid and self.ask else 0.0

    @property
    def spread_pct(self) -> float:
        """Spread as a percentage of mid price."""
        mid = self.mid
        return (self.spread / mid) * 100 if mid > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last": self.last,
            "last_size": self.last_size,
            "volume": self.volume,
            "vwap": self.vwap,
            "mid": self.mid,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
        }


@dataclass
class Bar:
    """Single OHLCV bar (daily or intraday)."""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    trade_count: int = 0

    @property
    def dollar_volume(self) -> float:
        return self.close * self.volume

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "dollar_volume": self.dollar_volume,
        }


class MarketDataProvider(abc.ABC):
    """Abstract market data provider.

    All concrete implementations (Alpaca, Finnhub, yfinance) must
    subclass this and implement every abstract method.

    DataFrame column conventions:
        Daily OHLCV:  open, high, low, close, volume, vwap (optional)
                      Index: DatetimeIndex (tz-aware UTC preferred)
        Intraday OHLCV: same columns, finer DatetimeIndex
    """

    # ------------------------------------------------------------------ #
    # Required properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. 'alpaca', 'finnhub'."""
        ...

    # ------------------------------------------------------------------ #
    # Historical bar data                                                  #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def get_daily_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for *ticker* between *start_date* and *end_date* (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume [, vwap]
            Index: DatetimeIndex (UTC)
        Raises
        ------
        DataProviderError
            On API errors, missing data, or deserialization failures.
        """
        ...

    @abc.abstractmethod
    async def get_intraday_ohlcv(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch intraday bars.

        Parameters
        ----------
        interval : str
            Bar size, e.g. ``'1Min'``, ``'5Min'``, ``'15Min'``, ``'1Hour'``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Real-time / latest data                                              #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def get_quote(self, ticker: str) -> Quote:
        """Fetch latest quote for *ticker*."""
        ...

    @abc.abstractmethod
    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch latest quotes for multiple tickers in a single round-trip where possible.

        Returns
        -------
        dict[str, Quote]
            Mapping of ticker → Quote.  Missing tickers are omitted.
        """
        ...

    @abc.abstractmethod
    async def get_market_snapshot(self) -> dict[str, Any]:
        """Fetch a snapshot of key market benchmarks: SPY, QQQ, IWM, VIX.

        Returns a dict with at least the keys ``'SPY'``, ``'QQQ'``, ``'IWM'``,
        ``'VIX_level'``.  Each benchmark entry should expose ``close``,
        ``change_pct``, ``volume``, and ``vwap``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Metadata / search                                                    #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def search_tickers(self, query: str) -> list[dict[str, Any]]:
        """Search for tickers matching *query*.

        Returns a list of dicts, each with at least ``ticker``, ``name``,
        and ``exchange``.
        """
        ...

    @abc.abstractmethod
    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """Return company metadata for *ticker*.

        Keys: name, ticker, exchange, sector, industry, market_cap,
        description, country, currency, logo_url, website.
        """
        ...

    # ------------------------------------------------------------------ #
    # Operational                                                          #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and responsive."""
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce column names to lowercase and ensure required columns exist.

        Raises ValueError if required columns are missing after normalisation.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLCV DataFrame index must be a DatetimeIndex")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.sort_index(inplace=True)
        return df

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r}>"
