"""
Position sizer: map confidence → allocation, apply regime adjustments,
enforce hard caps, and check total risk budget.

Returns a PositionSize describing the exact number of shares to buy,
dollar exposure, allocation %, risk %, and whether to scale in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.core.enums import MarketRegime

if TYPE_CHECKING:
    from src.core.config import RiskConfig
    from src.services.pipeline.screener import RegimeAssessment
    from src.services.trade.constructor import TradeSetup

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PositionSize:
    """Recommended position size with full context."""

    shares: int
    dollar_size: float
    allocation_pct: float   # % of portfolio value
    risk_pct: float         # % of portfolio at risk (shares × risk_per_share / portfolio)
    risk_per_share: float   # Entry − stop_loss

    # When True the caller should scale in (partial fill first, add on strength)
    scale_in: bool = False

    # Diagnostics
    sizing_notes: str = ""

    @classmethod
    def zero(cls, reason: str = "") -> "PositionSize":
        """Convenience constructor for a zero-sized position."""
        return cls(
            shares=0,
            dollar_size=0.0,
            allocation_pct=0.0,
            risk_pct=0.0,
            risk_per_share=0.0,
            scale_in=False,
            sizing_notes=reason or "Zero position: blocked by sizing logic.",
        )

    @property
    def is_tradeable(self) -> bool:
        return self.shares > 0


# ---------------------------------------------------------------------------
# Sizer
# ---------------------------------------------------------------------------

# Confidence band thresholds
_HIGH_CONF = 80.0
_MID_CONF = 60.0
_LOW_CONF = 40.0


class PositionSizer:
    """Translate confidence and regime into a concrete share count.

    Confidence → base allocation mapping:
    - ≥ 80 : up to max_position_pct (100% of max)
    - 60-80 : 50–100% of max (linear)
    - 40-60 : 25–50% of max (linear)
    -  < 40 : no trade

    Regime adjustments:
    - FAVORABLE   : no reduction
    - MIXED       : × 0.60
    - UNFAVORABLE : 0 (no trade)
    """

    def __init__(self, risk_config: "RiskConfig") -> None:
        self.config = risk_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        setup: "TradeSetup",
        confidence: float,
        regime: "RegimeAssessment",
        portfolio_value: float,
        current_risk_pct: float,
    ) -> PositionSize:
        """Compute position size.

        Args:
            setup:             TradeSetup with entry_high and stop_loss.
            confidence:        Scoring engine confidence (0-100).
            regime:            Current market regime.
            portfolio_value:   Total portfolio equity in dollars.
            current_risk_pct:  Currently committed risk % across open positions.

        Returns:
            PositionSize with shares, dollar size, and diagnostics.
        """
        # ---- Regime gate ----
        if regime.regime == MarketRegime.UNFAVORABLE:
            logger.info("sizer_zero_unfavorable_regime", ticker=setup.ticker)
            return PositionSize.zero("No position in UNFAVORABLE regime.")

        # ---- Total risk budget gate ----
        remaining_risk = self.config.max_total_risk_pct - current_risk_pct
        if remaining_risk <= 0:
            logger.info("sizer_zero_risk_budget_exhausted", ticker=setup.ticker)
            return PositionSize.zero("Total risk budget exhausted.")

        # ---- Confidence → base allocation ----
        base_alloc_pct = self._confidence_to_allocation(confidence)

        if base_alloc_pct <= 0:
            logger.info("sizer_zero_low_confidence", confidence=confidence, ticker=setup.ticker)
            return PositionSize.zero(f"Confidence {confidence:.1f} too low for a trade.")

        # ---- Regime adjustment ----
        if regime.regime == MarketRegime.MIXED:
            base_alloc_pct *= 0.60

        # ---- Risk-per-share calculation ----
        risk_per_share = setup.entry_high - setup.stop_loss
        if risk_per_share <= 0:
            logger.warning(
                "sizer_invalid_risk_per_share",
                ticker=setup.ticker,
                entry_high=setup.entry_high,
                stop_loss=setup.stop_loss,
            )
            return PositionSize.zero("Invalid stop: entry_high ≤ stop_loss.")

        # ---- Method 1: risk-based shares ----
        max_risk_dollars = portfolio_value * (self.config.max_risk_per_trade_pct / 100.0)
        risk_based_shares = int(max_risk_dollars / risk_per_share)

        # ---- Method 2: allocation-based shares ----
        alloc_dollars = portfolio_value * (base_alloc_pct / 100.0)
        alloc_based_shares = int(alloc_dollars / setup.entry_high)

        # ---- Take the more conservative ----
        shares = min(risk_based_shares, alloc_based_shares)

        # ---- Hard position cap ----
        max_alloc_dollars = portfolio_value * (self.config.max_position_pct / 100.0)
        max_cap_shares = int(max_alloc_dollars / setup.entry_high)
        shares = min(shares, max_cap_shares)

        if shares <= 0:
            logger.info("sizer_zero_computed_shares", ticker=setup.ticker)
            return PositionSize.zero("Computed share count is 0.")

        # ---- Final metrics ----
        dollar_size = shares * setup.entry_high
        allocation_pct = (dollar_size / portfolio_value) * 100.0
        risk_pct = (shares * risk_per_share / portfolio_value) * 100.0

        # Scale-in flag: use staged entry when not highly confident
        scale_in = confidence < 65.0

        notes_parts: list[str] = [
            f"confidence={confidence:.1f}",
            f"base_alloc={base_alloc_pct:.1f}%",
            f"risk_based_shares={risk_based_shares}",
            f"alloc_based_shares={alloc_based_shares}",
        ]
        if regime.regime == MarketRegime.MIXED:
            notes_parts.append("regime_adjustment=×0.60")

        pos = PositionSize(
            shares=shares,
            dollar_size=round(dollar_size, 2),
            allocation_pct=round(allocation_pct, 3),
            risk_pct=round(risk_pct, 4),
            risk_per_share=round(risk_per_share, 4),
            scale_in=scale_in,
            sizing_notes="; ".join(notes_parts),
        )

        logger.info(
            "sizer_result",
            ticker=setup.ticker,
            shares=shares,
            dollar_size=round(dollar_size, 2),
            allocation_pct=round(allocation_pct, 2),
            risk_pct=round(risk_pct, 3),
            scale_in=scale_in,
        )
        return pos

    # ------------------------------------------------------------------
    # Confidence → allocation
    # ------------------------------------------------------------------

    def _confidence_to_allocation(self, confidence: float) -> float:
        """Map confidence (0-100) to a base allocation percentage.

        Returns allocation as % of portfolio value (0-max_position_pct).
        """
        max_pct = float(self.config.max_position_pct)

        if confidence >= _HIGH_CONF:
            # Full allocation
            return max_pct

        if confidence >= _MID_CONF:
            # Linear interpolation: 50-100% of max between 60 and 80
            fraction = 0.50 + 0.50 * (confidence - _MID_CONF) / (_HIGH_CONF - _MID_CONF)
            return max_pct * fraction

        if confidence >= _LOW_CONF:
            # Linear interpolation: 25-50% of max between 40 and 60
            fraction = 0.25 + 0.25 * (confidence - _LOW_CONF) / (_MID_CONF - _LOW_CONF)
            return max_pct * fraction

        # Below 40: no trade
        return 0.0
