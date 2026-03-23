"""Rolling window data management.

Prevents unbounded memory growth by maintaining a fixed-size window of
historical data. When new data arrives, the oldest data is dropped.

Usage::

    window = RollingWindow(max_bars=500)
    window.append(new_bar_df)
    current_data = window.data  # Always ≤ 500 bars
"""

from __future__ import annotations

import pandas as pd


class RollingWindow:
    """Fixed-size rolling window for OHLCV data.

    Parameters
    ----------
    max_bars : int
        Maximum number of bars to retain. Default 500 (~2 years daily).
    """

    def __init__(self, max_bars: int = 500) -> None:
        self._max_bars = max_bars
        self._data: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        """Return the current data window."""
        if self._data is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return self._data

    @property
    def size(self) -> int:
        """Number of bars currently in the window."""
        return len(self._data) if self._data is not None else 0

    def set_data(self, df: pd.DataFrame) -> None:
        """Replace the window contents, trimming to max_bars.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        """
        if df is None or df.empty:
            self._data = None
            return
        self._data = df.tail(self._max_bars).copy()

    def append(self, new_data: pd.DataFrame) -> None:
        """Append new bars and trim to max_bars.

        Parameters
        ----------
        new_data : pd.DataFrame
            New bars to append (must have compatible columns and index).
        """
        if new_data is None or new_data.empty:
            return

        if self._data is None or self._data.empty:
            self._data = new_data.tail(self._max_bars).copy()
            return

        # Concatenate, drop duplicates (by index), sort, and trim
        combined = pd.concat([self._data, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        self._data = combined.tail(self._max_bars)

    def get_latest(self, n: int = 1) -> pd.DataFrame:
        """Return the last *n* bars."""
        if self._data is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return self._data.tail(n)

    def clear(self) -> None:
        """Clear all data."""
        self._data = None
