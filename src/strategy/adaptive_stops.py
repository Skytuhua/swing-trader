"""
Adaptive ATR Trailing Stop System — dynamic stop management that adapts to market conditions.

This module provides three complementary stop-loss mechanisms:

1. ATR Trailing Stop
   - Long: stop = highest_high - (ATR x multiplier)
   - Short: stop = lowest_low + (ATR x multiplier)
   - Multiplier adapts: 2.0-3.0 for trending, 1.5-2.0 for ranging markets

2. Chandelier Exit (enhanced variant)
   - Uses HHV (Highest High Value) / LLV (Lowest Low Value) for precise placement
   - Best suited for trending markets where you want to ride momentum

3. Scale-Out Mechanism
   - At first profit target: sell ~50%, move stop to breakeven
   - Remaining position gets extended target and trailing stop
   - Balances "lock in profits" vs "let winners run"

The multiplier automatically adjusts based on the market regime:
- Trending (ADX > 25): wider stops (2.0-3.0x ATR) to avoid premature exits
- Ranging  (ADX < 20): tighter stops (1.5-2.0x ATR) for mean-reversion trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class StopType(str, Enum):
    """Type of trailing stop being applied."""

    ATR_TRAILING = "atr_trailing"
    CHANDELIER = "chandelier"
    BREAKEVEN = "breakeven"
    FIXED = "fixed"


@dataclass
class StopLevel:
    """Computed stop-loss level with metadata."""

    price: float
    stop_type: StopType
    atr_multiplier: float
    atr_value: float
    lookback_high: float = 0.0
    lookback_low: float = 0.0
    is_trailing: bool = True
    notes: str = ""


@dataclass
class ScaleOutResult:
    """Result of a scale-out evaluation."""

    should_scale_out: bool = False
    scale_out_pct: float = 0.0  # fraction of position to sell (e.g. 0.5)
    new_stop: float = 0.0  # new stop price after scale-out
    move_stop_to_breakeven: bool = False
    target_hit: str = ""  # which target was hit (tp1, tp2)
    notes: str = ""


@dataclass
class AdaptiveStopState:
    """Tracks the state of an adaptive stop for a position."""

    ticker: str
    side: str  # "long" or "short"
    entry_price: float
    current_stop: float
    highest_high: float
    lowest_low: float
    atr: float
    multiplier: float
    stop_type: StopType = StopType.ATR_TRAILING
    tp1_hit: bool = False
    tp2_hit: bool = False
    scale_out_done: bool = False
    bars_since_entry: int = 0


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute the Average True Range for the most recent bar.

    ATR = Wilder-smoothed average of True Range over ``period`` bars.
    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|).
    """
    n = len(closes)
    if n < period + 1:
        if n >= 2:
            return float(np.mean(highs[-n:] - lows[-n:]))
        return 0.0

    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Wilder smoothing for ATR
    atr = float(np.mean(tr[1 : period + 1]))
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr = atr * (1 - alpha) + tr[i] * alpha

    return atr


class AdaptiveStopManager:
    """Manage adaptive ATR-based trailing stops for open positions.

    Parameters
    ----------
    config : dict or dataclass, optional
        Configuration with:
        - atr_period: int (default 14)
        - trailing_multiplier: float (default 2.5)
        - chandelier_enabled: bool (default True)
        - scale_out_enabled: bool (default True)
        - scale_out_pct: float (default 0.5)
        - first_target_atr_mult: float (default 2.0)
        - second_target_atr_mult: float (default 4.0)
    """

    def __init__(self, config: Any = None) -> None:
        self._cfg = config or {}
        self.atr_period: int = self._get("atr_period", 14)
        self.base_multiplier: float = self._get("trailing_multiplier", 2.5)
        self.chandelier_enabled: bool = self._get("chandelier_enabled", True)
        self.scale_out_enabled: bool = self._get("scale_out_enabled", True)
        self.scale_out_pct: float = self._get("scale_out_pct", 0.5)
        self.first_target_atr_mult: float = self._get("first_target_atr_mult", 2.0)
        self.second_target_atr_mult: float = self._get("second_target_atr_mult", 4.0)

        # Active stop states keyed by ticker
        self._states: dict[str, AdaptiveStopState] = {}

    def _get(self, key: str, default: Any) -> Any:
        if isinstance(self._cfg, dict):
            return self._cfg.get(key, default)
        return getattr(self._cfg, key, default)

    # ------------------------------------------------------------------
    # Stop level computation
    # ------------------------------------------------------------------

    def compute_stop(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        side: str = "long",
        regime_trending: bool = True,
        atr_override: float | None = None,
    ) -> StopLevel:
        """Compute the adaptive stop level for a position.

        Parameters
        ----------
        highs, lows, closes : np.ndarray
            OHLC price arrays.
        side : str
            'long' or 'short'.
        regime_trending : bool
            True if market is trending (wider stops), False if ranging (tighter).
        atr_override : float, optional
            Override the computed ATR value.

        Returns
        -------
        StopLevel with the computed stop price and metadata.
        """
        atr = atr_override if atr_override is not None else compute_atr(highs, lows, closes, self.atr_period)

        # Adapt multiplier to regime
        multiplier = self._adapt_multiplier(regime_trending)

        if self.chandelier_enabled:
            return self._chandelier_stop(highs, lows, closes, atr, multiplier, side)
        return self._atr_trailing_stop(highs, lows, closes, atr, multiplier, side)

    def _adapt_multiplier(self, regime_trending: bool) -> float:
        """Adjust the ATR multiplier based on market regime.

        Trending markets get wider stops (2.0-3.0x ATR) to avoid noise exits.
        Ranging markets get tighter stops (1.5-2.0x ATR) for faster exits.
        """
        if regime_trending:
            return max(2.0, min(3.0, self.base_multiplier))
        return max(1.5, min(2.0, self.base_multiplier * 0.7))

    def _atr_trailing_stop(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: float,
        multiplier: float,
        side: str,
    ) -> StopLevel:
        """Standard ATR trailing stop.

        Long:  stop = highest_high(lookback) - ATR * multiplier
        Short: stop = lowest_low(lookback) + ATR * multiplier
        """
        lookback = max(1, self.atr_period)
        hh = float(np.max(highs[-lookback:])) if len(highs) >= lookback else float(highs[-1])
        ll = float(np.min(lows[-lookback:])) if len(lows) >= lookback else float(lows[-1])

        if side == "long":
            stop = hh - atr * multiplier
        else:
            stop = ll + atr * multiplier

        return StopLevel(
            price=round(stop, 4),
            stop_type=StopType.ATR_TRAILING,
            atr_multiplier=multiplier,
            atr_value=round(atr, 4),
            lookback_high=round(hh, 4),
            lookback_low=round(ll, 4),
            is_trailing=True,
            notes=f"ATR trailing: {side} stop @ {stop:.2f} ({multiplier:.1f}x ATR {atr:.2f})",
        )

    def _chandelier_stop(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: float,
        multiplier: float,
        side: str,
    ) -> StopLevel:
        """Chandelier Exit — HHV/LLV based trailing stop.

        Uses the Highest High Value (HHV) and Lowest Low Value (LLV) over
        the ATR lookback period for more precise stop placement.

        Long:  stop = HHV(period) - ATR * multiplier
        Short: stop = LLV(period) + ATR * multiplier
        """
        period = max(1, self.atr_period)
        n = min(period, len(highs))

        hhv = float(np.max(highs[-n:]))
        llv = float(np.min(lows[-n:]))

        if side == "long":
            stop = hhv - atr * multiplier
        else:
            stop = llv + atr * multiplier

        return StopLevel(
            price=round(stop, 4),
            stop_type=StopType.CHANDELIER,
            atr_multiplier=multiplier,
            atr_value=round(atr, 4),
            lookback_high=round(hhv, 4),
            lookback_low=round(llv, 4),
            is_trailing=True,
            notes=f"Chandelier: {side} stop @ {stop:.2f} (HHV={hhv:.2f}, LLV={llv:.2f}, {multiplier:.1f}x ATR)",
        )

    # ------------------------------------------------------------------
    # Trailing stop update (for live position management)
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        state: AdaptiveStopState,
        current_high: float,
        current_low: float,
        current_close: float,
        atr: float,
    ) -> float:
        """Update the trailing stop for an existing position.

        The stop only ratchets in the favorable direction — it never moves
        against the position.

        Parameters
        ----------
        state : AdaptiveStopState
            Current stop state for the position.
        current_high, current_low, current_close : float
            Current bar's OHLC.
        atr : float
            Current ATR value.

        Returns
        -------
        The new (potentially updated) stop price.
        """
        multiplier = state.multiplier
        state.bars_since_entry += 1

        if state.side == "long":
            state.highest_high = max(state.highest_high, current_high)
            new_stop = state.highest_high - atr * multiplier
            # Stop only moves up for longs
            if new_stop > state.current_stop:
                state.current_stop = round(new_stop, 4)
                logger.debug(
                    "trailing_stop_updated",
                    ticker=state.ticker,
                    new_stop=state.current_stop,
                    highest_high=state.highest_high,
                )
        else:
            state.lowest_low = min(state.lowest_low, current_low)
            new_stop = state.lowest_low + atr * multiplier
            # Stop only moves down for shorts
            if new_stop < state.current_stop:
                state.current_stop = round(new_stop, 4)
                logger.debug(
                    "trailing_stop_updated",
                    ticker=state.ticker,
                    new_stop=state.current_stop,
                    lowest_low=state.lowest_low,
                )

        return state.current_stop

    # ------------------------------------------------------------------
    # Scale-out logic
    # ------------------------------------------------------------------

    def evaluate_scale_out(
        self,
        state: AdaptiveStopState,
        current_price: float,
        atr: float,
    ) -> ScaleOutResult:
        """Evaluate whether a position should scale out (partial profit-take).

        Scale-out rules:
        1. At TP1 (first_target_atr_mult x ATR from entry): sell scale_out_pct,
           move stop to breakeven.
        2. At TP2 (second_target_atr_mult x ATR from entry): sell remainder.

        Parameters
        ----------
        state : AdaptiveStopState
            Current position state.
        current_price : float
            Current market price.
        atr : float
            Current ATR value.

        Returns
        -------
        ScaleOutResult indicating whether to scale out and new stop level.
        """
        if not self.scale_out_enabled:
            return ScaleOutResult()

        tp1_distance = atr * self.first_target_atr_mult
        tp2_distance = atr * self.second_target_atr_mult

        if state.side == "long":
            tp1_price = state.entry_price + tp1_distance
            tp2_price = state.entry_price + tp2_distance
            price_above_tp1 = current_price >= tp1_price
            price_above_tp2 = current_price >= tp2_price
        else:
            tp1_price = state.entry_price - tp1_distance
            tp2_price = state.entry_price - tp2_distance
            price_above_tp1 = current_price <= tp1_price
            price_above_tp2 = current_price <= tp2_price

        # TP2 hit (full exit of remaining position)
        if not state.tp2_hit and price_above_tp2:
            state.tp2_hit = True
            return ScaleOutResult(
                should_scale_out=True,
                scale_out_pct=1.0,  # exit remaining
                new_stop=state.current_stop,
                target_hit="tp2",
                notes=f"TP2 hit @ {current_price:.2f} (target {tp2_price:.2f}). Full exit of remaining.",
            )

        # TP1 hit (partial exit + move stop to breakeven)
        if not state.tp1_hit and price_above_tp1:
            state.tp1_hit = True
            state.scale_out_done = True
            # Move stop to breakeven
            state.current_stop = state.entry_price
            return ScaleOutResult(
                should_scale_out=True,
                scale_out_pct=self.scale_out_pct,
                new_stop=state.entry_price,
                move_stop_to_breakeven=True,
                target_hit="tp1",
                notes=(
                    f"TP1 hit @ {current_price:.2f} (target {tp1_price:.2f}). "
                    f"Scaling out {self.scale_out_pct:.0%}, moving stop to breakeven {state.entry_price:.2f}."
                ),
            )

        return ScaleOutResult()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def create_state(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        initial_stop: float,
        atr: float,
        regime_trending: bool = True,
    ) -> AdaptiveStopState:
        """Create and register a new adaptive stop state for a position."""
        multiplier = self._adapt_multiplier(regime_trending)
        state = AdaptiveStopState(
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_high=entry_price,
            lowest_low=entry_price,
            atr=atr,
            multiplier=multiplier,
        )
        self._states[ticker] = state
        return state

    def get_state(self, ticker: str) -> AdaptiveStopState | None:
        """Retrieve the stop state for a ticker."""
        return self._states.get(ticker)

    def remove_state(self, ticker: str) -> None:
        """Remove stop state when a position is closed."""
        self._states.pop(ticker, None)

    @property
    def active_states(self) -> dict[str, AdaptiveStopState]:
        """All currently tracked stop states."""
        return dict(self._states)
