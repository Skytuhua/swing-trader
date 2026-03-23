"""Incremental indicator update support.

Instead of recalculating indicators from scratch on every new bar,
this module maintains cached indicator state and updates incrementally
with new data points.

Supports:
- EMA (exponential moving average) — O(1) update per bar
- SMA (simple moving average) — O(1) with circular buffer
- RSI — O(1) incremental update
- ATR — O(1) incremental update

Usage::

    cache = IncrementalIndicatorCache()
    cache.update_ema("AAPL", "ema_20", new_close=150.0, period=20)
    current_ema = cache.get("AAPL", "ema_20")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _EMAState:
    """Internal state for incremental EMA."""

    period: int
    value: float = 0.0
    count: int = 0
    multiplier: float = 0.0

    def __post_init__(self) -> None:
        self.multiplier = 2.0 / (self.period + 1)


@dataclass
class _SMAState:
    """Internal state for incremental SMA with circular buffer."""

    period: int
    buffer: deque = field(default_factory=deque)
    total: float = 0.0

    @property
    def value(self) -> float:
        if len(self.buffer) == 0:
            return 0.0
        return self.total / len(self.buffer)


@dataclass
class _RSIState:
    """Internal state for incremental RSI."""

    period: int
    avg_gain: float = 0.0
    avg_loss: float = 0.0
    prev_close: float = 0.0
    count: int = 0

    @property
    def value(self) -> float:
        if self.avg_loss == 0:
            return 100.0 if self.avg_gain > 0 else 50.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class _ATRState:
    """Internal state for incremental ATR."""

    period: int
    value: float = 0.0
    prev_close: float = 0.0
    count: int = 0


class IncrementalIndicatorCache:
    """Cache for incrementally-updated indicator values.

    Maintains per-ticker, per-indicator state so that new bars can be
    processed in O(1) time instead of recalculating from the full history.
    """

    def __init__(self) -> None:
        self._ema: dict[str, dict[str, _EMAState]] = {}
        self._sma: dict[str, dict[str, _SMAState]] = {}
        self._rsi: dict[str, dict[str, _RSIState]] = {}
        self._atr: dict[str, dict[str, _ATRState]] = {}

    def get(self, ticker: str, indicator_key: str) -> float | None:
        """Get the current value of a cached indicator.

        Returns None if not yet initialized.
        """
        for store in (self._ema, self._sma, self._rsi, self._atr):
            if ticker in store and indicator_key in store[ticker]:
                return store[ticker][indicator_key].value
        return None

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def update_ema(
        self, ticker: str, key: str, new_value: float, period: int
    ) -> float:
        """Update EMA incrementally with a new data point.

        Parameters
        ----------
        ticker : str
            Symbol.
        key : str
            Indicator key (e.g. "ema_20").
        new_value : float
            New price or value to incorporate.
        period : int
            EMA period.

        Returns
        -------
        float
            Updated EMA value.
        """
        if ticker not in self._ema:
            self._ema[ticker] = {}
        if key not in self._ema[ticker]:
            self._ema[ticker][key] = _EMAState(period=period)

        state = self._ema[ticker][key]
        state.count += 1

        if state.count == 1:
            state.value = new_value
        else:
            state.value = (new_value * state.multiplier) + (state.value * (1 - state.multiplier))

        return state.value

    # ------------------------------------------------------------------
    # SMA
    # ------------------------------------------------------------------

    def update_sma(
        self, ticker: str, key: str, new_value: float, period: int
    ) -> float:
        """Update SMA incrementally using a circular buffer.

        Returns
        -------
        float
            Updated SMA value.
        """
        if ticker not in self._sma:
            self._sma[ticker] = {}
        if key not in self._sma[ticker]:
            self._sma[ticker][key] = _SMAState(period=period, buffer=deque(maxlen=period))

        state = self._sma[ticker][key]

        # Remove oldest if buffer is full
        if len(state.buffer) == state.period:
            state.total -= state.buffer[0]

        state.buffer.append(new_value)
        state.total += new_value

        return state.value

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def update_rsi(
        self, ticker: str, key: str, new_close: float, period: int = 14
    ) -> float:
        """Update RSI incrementally with a new close price.

        Uses Wilder's smoothing method for incremental updates.

        Returns
        -------
        float
            Updated RSI value (0-100).
        """
        if ticker not in self._rsi:
            self._rsi[ticker] = {}
        if key not in self._rsi[ticker]:
            self._rsi[ticker][key] = _RSIState(period=period, prev_close=new_close)
            return 50.0  # Neutral on first bar

        state = self._rsi[ticker][key]
        state.count += 1

        change = new_close - state.prev_close
        gain = max(0.0, change)
        loss = max(0.0, -change)
        state.prev_close = new_close

        if state.count <= period:
            # Initial period: accumulate simple averages
            state.avg_gain = (state.avg_gain * (state.count - 1) + gain) / state.count
            state.avg_loss = (state.avg_loss * (state.count - 1) + loss) / state.count
        else:
            # Wilder's smoothing
            state.avg_gain = (state.avg_gain * (period - 1) + gain) / period
            state.avg_loss = (state.avg_loss * (period - 1) + loss) / period

        return state.value

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def update_atr(
        self,
        ticker: str,
        key: str,
        high: float,
        low: float,
        close: float,
        period: int = 14,
    ) -> float:
        """Update ATR incrementally with a new OHLC bar.

        Returns
        -------
        float
            Updated ATR value.
        """
        if ticker not in self._atr:
            self._atr[ticker] = {}
        if key not in self._atr[ticker]:
            self._atr[ticker][key] = _ATRState(period=period, prev_close=close)
            state = self._atr[ticker][key]
            state.value = high - low  # First bar: TR = range
            return state.value

        state = self._atr[ticker][key]
        state.count += 1

        # True range
        tr = max(
            high - low,
            abs(high - state.prev_close),
            abs(low - state.prev_close),
        )
        state.prev_close = close

        if state.count < period:
            # Simple average during warmup
            state.value = (state.value * state.count + tr) / (state.count + 1)
        else:
            # Wilder's smoothing
            state.value = (state.value * (period - 1) + tr) / period

        return state.value

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self, ticker: str | None = None) -> None:
        """Clear cached state for a ticker (or all tickers if None)."""
        for store in (self._ema, self._sma, self._rsi, self._atr):
            if ticker is None:
                store.clear()
            else:
                store.pop(ticker, None)

    def has_data(self, ticker: str) -> bool:
        """Check if any indicator data exists for a ticker."""
        return any(
            ticker in store and store[ticker]
            for store in (self._ema, self._sma, self._rsi, self._atr)
        )
