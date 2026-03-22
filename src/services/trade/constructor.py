"""
Trade constructor: compute entry zone, stop loss, take-profit targets,
trailing stop rule, and invalidation conditions from a scored candidate.

All price levels are derived from ATR, structural key levels, and
realistic swing-trade expectations (3-5 day hold horizon).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import EntryMethod

if TYPE_CHECKING:
    from src.services.scoring.engine import ScoredCandidate

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TradeSetup:
    """Complete trade setup ready for sizing and execution."""

    ticker: str

    # Entry
    entry_low: float            # Lower bound of acceptable entry zone
    entry_high: float           # Upper bound / limit price
    entry_method: EntryMethod

    # Risk management
    stop_loss: float            # Hard stop level
    take_profit_1: float        # Primary target (2:1 R:R)
    take_profit_2: float        # Stretch target (3:1 R:R)
    trailing_stop_rule: str     # e.g. "trail_atr_1.5"

    # Metadata
    risk_reward_ratio: float    # Calculated R:R (based on entry_high and TP1)
    atr: float                  # ATR at construction time
    max_hold_days: int = 5

    # Conditions under which the thesis is invalidated
    invalidation_conditions: list[str] = field(default_factory=list)

    # Diagnostics
    constructed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    notes: str = ""

    # Extra (key levels used during construction)
    key_levels_used: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

# Quote duck-type – the real Quote comes from market_data service
_QuoteT = Any


class TradeConstructor:
    """Construct a complete TradeSetup from a scored candidate and live quote.

    Rules (aligned with spec):
    - Entry zone: current_price ±0.5% for most setups; adjusted for
      breakout / pullback entries.
    - Stop loss: below nearest support OR 2x ATR below entry, whichever
      is lower; minimum 0.5% below price.
    - TP1: 2:1 R:R, capped at 70% of 4-ATR theoretical maximum.
    - TP2: 3:1 R:R, capped at 4-ATR theoretical maximum.
    - Trailing stop: activates after 1% gain; trails at 1.5x ATR.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct(
        self,
        candidate: "ScoredCandidate",
        daily_df: Any,          # pd.DataFrame
        quote: _QuoteT,
    ) -> TradeSetup:
        """Build a TradeSetup for the given candidate.

        Args:
            candidate:  The top-ranked ScoredCandidate.
            daily_df:   200 days of daily OHLCV as a DataFrame.
            quote:      Current market quote with at minimum a `.price` field.

        Returns:
            Populated TradeSetup ready for position sizing.
        """
        technical = candidate.screened_candidate.technical
        indicators = technical.indicators
        key_levels: dict[str, Any] = getattr(technical, "key_levels", {}) or {}

        atr: float = float(indicators.get("atr_14", 0) or 0)
        current_price: float = float(quote.price)

        if atr <= 0:
            # Fallback: estimate ATR as 2% of price
            atr = current_price * 0.02
            logger.warning(
                "trade_constructor_atr_missing",
                ticker=candidate.ticker,
                fallback_atr=round(atr, 4),
            )

        # ---- Entry zone ----
        setup_type: str = getattr(technical, "setup_type", "default") or "default"
        entry_low, entry_high, entry_method = self._compute_entry(
            setup_type, current_price, key_levels
        )

        # ---- Stop loss ----
        stop_loss = self._compute_stop(
            current_price, atr, key_levels
        )

        # ---- Take-profit targets ----
        take_profit_1, take_profit_2 = self._compute_targets(
            current_price, stop_loss, atr
        )

        # ---- R:R ratio (relative to entry_high, which is worst-case fill) ----
        risk_dollars = max(entry_high - stop_loss, 1e-8)
        reward_dollars = take_profit_1 - entry_high
        risk_reward_ratio = reward_dollars / risk_dollars if risk_dollars > 0 else 0.0

        # ---- Trailing stop rule ----
        trailing_stop_rule = f"trail_atr_{1.5}"

        # ---- Invalidation conditions ----
        invalidation = [
            f"close_below_{stop_loss:.2f}",
            "volume_collapse_below_50pct_avg",
            "regime_turns_unfavorable",
            "major_negative_news_catalyst",
        ]

        setup = TradeSetup(
            ticker=candidate.ticker,
            entry_low=round(entry_low, 4),
            entry_high=round(entry_high, 4),
            entry_method=entry_method,
            stop_loss=round(stop_loss, 4),
            take_profit_1=round(take_profit_1, 4),
            take_profit_2=round(take_profit_2, 4),
            trailing_stop_rule=trailing_stop_rule,
            risk_reward_ratio=round(risk_reward_ratio, 3),
            atr=round(atr, 4),
            max_hold_days=5,
            invalidation_conditions=invalidation,
            key_levels_used=dict(key_levels),
        )

        logger.info(
            "trade_constructor_setup",
            ticker=candidate.ticker,
            setup_type=setup_type,
            entry_low=setup.entry_low,
            entry_high=setup.entry_high,
            stop_loss=setup.stop_loss,
            tp1=setup.take_profit_1,
            tp2=setup.take_profit_2,
            rr=setup.risk_reward_ratio,
        )
        return setup

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entry(
        setup_type: str,
        current_price: float,
        key_levels: dict[str, Any],
    ) -> tuple[float, float, EntryMethod]:
        """Compute entry zone and method based on setup type."""
        if setup_type == "breakout":
            resistance = key_levels.get("resistance") or key_levels.get("nearest_resistance")
            entry_high = float(resistance) if resistance else current_price * 1.01
            entry_low = current_price * 0.998
            return entry_low, entry_high, EntryMethod.BREAKOUT_CONFIRMATION

        if setup_type == "pullback":
            support = key_levels.get("support") or key_levels.get("nearest_support")
            entry_low = float(support) if support else current_price * 0.99
            entry_high = current_price * 1.002
            return entry_low, entry_high, EntryMethod.PULLBACK

        # Default: limit order within a tight range around current price
        entry_low = current_price * 0.998
        entry_high = current_price * 1.005
        return entry_low, entry_high, EntryMethod.LIMIT

    @staticmethod
    def _compute_stop(
        current_price: float,
        atr: float,
        key_levels: dict[str, Any],
    ) -> float:
        """Compute stop loss level.

        Logic:
        1. ATR stop: current_price - 2.0 * ATR
        2. Structural stop: just below nearest support with 0.5% buffer
        3. Take the lower of (1) and (2) for maximum protection
        4. Never closer than 0.5% below entry
        """
        atr_stop = current_price - (atr * 2.0)

        support = (
            key_levels.get("nearest_support")
            or key_levels.get("support")
        )
        if support and float(support) > atr_stop:
            structural_stop = float(support) * 0.995  # 0.5% below support
            stop_loss = min(structural_stop, atr_stop)
        else:
            stop_loss = atr_stop

        # Enforce minimum stop distance of 0.5% below current price
        min_stop = current_price * 0.995
        stop_loss = min(stop_loss, min_stop)

        return stop_loss

    @staticmethod
    def _compute_targets(
        current_price: float,
        stop_loss: float,
        atr: float,
    ) -> tuple[float, float]:
        """Compute TP1 (2:1) and TP2 (3:1), capped for realistic 3-5 day swings."""
        risk = current_price - stop_loss
        tp1_raw = current_price + (risk * 2.0)
        tp2_raw = current_price + (risk * 3.0)

        # Cap at realistic 3-5 day move: ~4 ATR from entry
        # Cap is applied as an offset above current_price, never below it
        atr_cap_1 = current_price + (atr * 2.8)   # ~70% of 4-ATR range
        atr_cap_2 = current_price + (atr * 4.0)
        tp1 = min(tp1_raw, atr_cap_1)
        tp2 = min(tp2_raw, atr_cap_2)

        # Ensure TP1 is always above current_price (at least 0.5% up)
        tp1 = max(tp1, current_price * 1.005)
        # TP2 must always be above TP1
        tp2 = max(tp2, tp1 * 1.01)

        return tp1, tp2
