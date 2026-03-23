"""
MarketSimulator: provide day-by-day historical data slices with no lookahead.

Usage
-----
sim = MarketSimulator(data, slippage_pct=0.001, commission_per_share=0.005)
trading_days = sim.get_trading_days("2023-01-01", "2024-01-01")
for day in trading_days:
    sim.advance_to(day)
    bar = sim.get_bar("AAPL", day)
    candidates = sim.get_candidates(day, exclude=set(), universe=["AAPL"])
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(d: str | date | datetime | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the most-recent ATR(14) from a OHLCV DataFrame slice."""
    if len(df) < period:
        last_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0
        return last_close * 0.02
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    val = atr_series.dropna()
    return float(val.iloc[-1]) if len(val) > 0 else float(df["close"].iloc[-1]) * 0.02


def _simple_trend_score(df: pd.DataFrame) -> float:
    """Return a quick 0-100 trend score for the last bar of a DataFrame slice."""
    if len(df) < 20:
        return 50.0
    close = df["close"]
    last = float(close.iloc[-1])

    score = 50.0

    # Price vs SMA20
    sma20 = close.rolling(20).mean().iloc[-1]
    if not np.isnan(sma20) and sma20 > 0:
        score += 20.0 if last > sma20 else -20.0

    # Price vs SMA50
    if len(close) >= 50:
        sma50 = close.rolling(50).mean().iloc[-1]
        if not np.isnan(sma50) and sma50 > 0:
            score += 15.0 if last > sma50 else -15.0

    # 10-day momentum
    if len(close) >= 11:
        roc10 = (last / float(close.iloc[-11]) - 1.0) * 100.0
        score += min(15.0, max(-15.0, roc10))

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# MarketSimulator
# ---------------------------------------------------------------------------


class MarketSimulator:
    """Day-by-day OHLCV simulator with no-lookahead guarantee.

    Parameters
    ----------
    data : pd.DataFrame | dict[str, pd.DataFrame]
        Historical OHLCV data. Single-ticker DataFrame OR dict of ticker → DataFrame.
        Each DataFrame must have columns [open, high, low, close, volume] and a
        DatetimeIndex (or date-parseable index).
    slippage_pct : float
        Per-side slippage expressed as a fraction (0.001 = 0.1%).
    commission_per_share : float
        Flat commission per share (in dollars).
    """

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        slippage_pct: float = 0.001,
        commission_per_share: float = 0.005,
        simulation_config: Any = None,
    ) -> None:
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share

        # Configurable slippage model (if SimulationConfig is available)
        self._slippage_model = None
        if simulation_config is not None:
            try:
                from src.simulation.slippage import SlippageModel
                self._slippage_model = SlippageModel(simulation_config)
            except Exception:
                pass

        # Caches for volume and ATR (populated during advance_to)
        self._avg_daily_volumes: dict[str, float] = {}
        self._atr_cache: dict[str, float] = {}

        # Normalize to dict[ticker, DataFrame]
        if isinstance(data, pd.DataFrame):
            self._data: dict[str, pd.DataFrame] = {"STOCK": data}
        else:
            self._data = dict(data)

        # Ensure DatetimeIndex and sorted
        for ticker in list(self._data.keys()):
            df = self._data[ticker].copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=False)
            df = df.sort_index()
            df.columns = [c.lower() for c in df.columns]
            self._data[ticker] = df

        # Current simulation date (set by advance_to)
        self._current_date: date | None = None

        # Cached data slice up to current_date (per ticker)
        self._slices: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_trading_days(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[date]:
        """Return sorted list of dates that appear in any ticker's data."""
        all_dates: set[date] = set()
        for df in self._data.values():
            for idx in df.index:
                d = idx.date() if isinstance(idx, datetime) else idx
                all_dates.add(d)

        days = sorted(all_dates)

        if start_date:
            sd = _to_date(start_date)
            days = [d for d in days if d >= sd]
        if end_date:
            ed = _to_date(end_date)
            days = [d for d in days if d <= ed]

        return days

    def advance_to(self, target_date: date) -> None:
        """Advance the simulation cursor to target_date.  Clears cached slices.

        Also updates ATR and average volume caches for slippage model use.
        """
        self._current_date = target_date
        self._slices = {}

        # Update volume and ATR caches for configurable slippage
        if self._slippage_model is not None:
            for ticker in self._data:
                df_slice = self.get_slice(ticker, target_date, bars=30)
                if len(df_slice) >= 5:
                    close = float(df_slice["close"].iloc[-1])
                    avg_vol = float(df_slice["volume"].tail(20).mean())
                    self._avg_daily_volumes[ticker] = avg_vol * close
                    self._atr_cache[ticker] = _compute_atr(df_slice) / close if close > 0 else 0.015

    # ------------------------------------------------------------------
    # Bar access
    # ------------------------------------------------------------------

    def get_bar(
        self,
        ticker: str,
        on_date: date,
    ) -> dict[str, float] | None:
        """Return the OHLCV bar for a ticker on a given date.

        Returns None if no bar is available (e.g. holiday, delisted).
        """
        df = self._data.get(ticker)
        if df is None:
            return None

        target = pd.Timestamp(on_date)
        try:
            row = df.loc[target]
        except KeyError:
            # Try fuzzy: find the last available row on or before that date
            mask = df.index <= target
            if not mask.any():
                return None
            row = df[mask].iloc[-1]

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def get_slice(self, ticker: str, up_to_date: date, bars: int = 200) -> pd.DataFrame:
        """Return up to *bars* rows of history strictly up to *up_to_date* (inclusive).

        The returned slice has no knowledge of future prices — no-lookahead guaranteed.
        """
        cache_key = (ticker, up_to_date, bars)
        if ticker in self._slices:
            return self._slices[ticker]

        df = self._data.get(ticker)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        cutoff = pd.Timestamp(up_to_date)
        mask = df.index <= cutoff
        available = df[mask]
        result = available.tail(bars)
        self._slices[ticker] = result
        return result

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        on_date: date,
        exclude: set[str] | None = None,
        universe: list[str] | None = None,
        min_bars: int = 30,
        top_n: int = 20,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return a ranked list of (ticker, score, indicators) for the given date.

        Scoring is purely technical (trend + momentum). In live mode the full
        pipeline would run here. This simplified version is suitable for
        backtest simulation.

        Returns
        -------
        list of (ticker, score, indicators) sorted descending by score.
        """
        exclude = exclude or set()
        tickers = universe if universe else list(self._data.keys())
        tickers = [t for t in tickers if t not in exclude]

        candidates: list[tuple[str, float, dict[str, Any]]] = []
        for ticker in tickers:
            df_slice = self.get_slice(ticker, on_date)
            if len(df_slice) < min_bars:
                continue

            bar = self.get_bar(ticker, on_date)
            if bar is None:
                continue

            # Ensure today's bar is not "future" by checking we have ≥ 1 bar
            score = _simple_trend_score(df_slice)
            atr = _compute_atr(df_slice)
            close = float(df_slice["close"].iloc[-1])

            # Relative volume (last bar vs. 20-bar avg)
            if len(df_slice) >= 21:
                avg_vol = float(df_slice["volume"].iloc[-21:-1].mean())
                last_vol = float(df_slice["volume"].iloc[-1])
                rel_vol = last_vol / avg_vol if avg_vol > 0 else 1.0
            else:
                rel_vol = 1.0

            indicators: dict[str, Any] = {
                "atr_14": round(atr, 4),
                "last_close": round(close, 4),
                "relative_volume": round(rel_vol, 3),
                "trend_score": round(score, 2),
            }
            candidates.append((ticker, score, indicators))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def fill_buy(self, ticker: str, reference_price: float) -> tuple[float, float]:
        """Simulate a buy fill: price + slippage. Returns (fill_price, slippage_dollars).

        If a SimulationConfig-backed SlippageModel is available, uses it for
        volume/volatility-aware slippage. Otherwise falls back to flat percentage.
        """
        if self._slippage_model is not None:
            addv = self._avg_daily_volumes.get(ticker, 100_000_000.0)
            atr_pct = self._atr_cache.get(ticker, 0.015)
            order_value = reference_price * 100  # estimate
            result = self._slippage_model.calculate(
                order_dollar_value=order_value,
                avg_daily_dollar_volume=addv,
                atr_pct=atr_pct,
                side="buy",
            )
            slip_pct = result.slippage_pct
        else:
            slip_pct = self.slippage_pct

        fill = reference_price * (1.0 + slip_pct)
        slippage_dollars = reference_price * slip_pct
        return round(fill, 6), round(slippage_dollars, 6)

    def fill_sell(self, ticker: str, reference_price: float) -> tuple[float, float]:
        """Simulate a sell fill: price - slippage. Returns (fill_price, slippage_dollars).

        Uses configurable slippage model if available.
        """
        if self._slippage_model is not None:
            addv = self._avg_daily_volumes.get(ticker, 100_000_000.0)
            atr_pct = self._atr_cache.get(ticker, 0.015)
            order_value = reference_price * 100
            result = self._slippage_model.calculate(
                order_dollar_value=order_value,
                avg_daily_dollar_volume=addv,
                atr_pct=atr_pct,
                side="sell",
            )
            slip_pct = result.slippage_pct
        else:
            slip_pct = self.slippage_pct

        fill = reference_price * (1.0 - slip_pct)
        slippage_dollars = reference_price * slip_pct
        return round(fill, 6), round(slippage_dollars, 6)

    # ------------------------------------------------------------------
    # Capital tracking helper
    # ------------------------------------------------------------------

    def compute_portfolio_value(
        self,
        cash: float,
        positions: dict[str, Any],  # ticker → {shares: int, entry_price: float}
        on_date: date,
    ) -> float:
        """Compute total portfolio value (cash + mark-to-market positions)."""
        total = cash
        for ticker, pos in positions.items():
            bar = self.get_bar(ticker, on_date)
            price = float(bar["close"]) if bar is not None else float(pos.get("entry_price", 0))
            total += price * int(pos.get("shares", 0))
        return total
