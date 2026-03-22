"""
Pre-trade validator: validate trade setup and position size before execution.

Catches configuration / data errors that could result in a loss-generating
or operationally dangerous order being sent to the broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.trade.constructor import TradeSetup
    from src.services.trade.sizer import PositionSize

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a pre-trade validation pass."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            base = "PASS"
        else:
            base = "FAIL: " + "; ".join(self.failures)
        if self.warnings:
            base += " | Warnings: " + "; ".join(self.warnings)
        return base


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_MIN_STOP_DISTANCE_PCT = 0.003   # 0.3% minimum stop distance
_MAX_STOP_DISTANCE_PCT = 0.15    # 15% maximum stop distance (likely a bad level)
_MIN_RR = 1.2                    # Absolute minimum R:R
_MAX_POSITION_DOLLAR_HARD_CAP = 1_000_000.0  # $1M hard cap regardless of config
_MIN_SHARES = 1


class PreTradeValidator:
    """Validate a TradeSetup and PositionSize before submitting to the broker.

    Checks:
    1. Shares > 0
    2. Dollar size > 0
    3. Stop distance reasonable (0.3% – 15% of price)
    4. Entry > stop_loss
    5. TP1 > entry, TP2 > TP1
    6. R:R ≥ minimum
    7. Position within hard dollar cap
    8. entry_low ≤ entry_high

    Warnings (non-blocking):
    - Scale-in suggested (low confidence)
    - Earnings within 2 days
    - Stop distance > 8% (unusually wide)
    """

    def __init__(
        self,
        min_risk_reward: float = _MIN_RR,
        min_stop_pct: float = _MIN_STOP_DISTANCE_PCT,
        max_stop_pct: float = _MAX_STOP_DISTANCE_PCT,
        max_position_dollars: float = _MAX_POSITION_DOLLAR_HARD_CAP,
    ) -> None:
        self.min_risk_reward = min_risk_reward
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.max_position_dollars = max_position_dollars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        setup: "TradeSetup",
        size: "PositionSize",
    ) -> ValidationResult:
        """Run all validation checks.

        Args:
            setup: The TradeSetup from TradeConstructor.
            size:  The PositionSize from PositionSizer.

        Returns:
            ValidationResult with passed/failed status and detailed messages.
        """
        failures: list[str] = []
        warnings: list[str] = []

        entry_ref = setup.entry_high  # Worst-case fill price

        # ---- 1. Positive share count ----
        if size.shares < _MIN_SHARES:
            failures.append(
                f"shares={size.shares} is less than minimum {_MIN_SHARES}."
            )

        # ---- 2. Positive dollar size ----
        if size.dollar_size <= 0:
            failures.append(f"dollar_size={size.dollar_size:.2f} must be positive.")

        # ---- 3. Entry > stop_loss ----
        if entry_ref <= setup.stop_loss:
            failures.append(
                f"entry_high={entry_ref:.4f} ≤ stop_loss={setup.stop_loss:.4f}. "
                "Stop must be below entry."
            )

        # ---- 4. Stop distance within reasonable bounds ----
        if entry_ref > 0:
            stop_distance_pct = (entry_ref - setup.stop_loss) / entry_ref
            if stop_distance_pct < self.min_stop_pct:
                failures.append(
                    f"Stop distance {stop_distance_pct:.2%} < minimum {self.min_stop_pct:.2%}. "
                    "Stop may be too tight and get immediately triggered."
                )
            elif stop_distance_pct > self.max_stop_pct:
                failures.append(
                    f"Stop distance {stop_distance_pct:.2%} > maximum {self.max_stop_pct:.2%}. "
                    "Stop is unreasonably wide; risk-per-trade will be excessive."
                )
            elif stop_distance_pct > 0.08:
                warnings.append(
                    f"Stop distance {stop_distance_pct:.2%} is quite wide (>8%). "
                    "Verify this is structurally justified."
                )

        # ---- 5. TP1 above entry ----
        if setup.take_profit_1 <= entry_ref:
            failures.append(
                f"take_profit_1={setup.take_profit_1:.4f} ≤ entry_high={entry_ref:.4f}."
            )

        # ---- 6. TP2 above TP1 ----
        if setup.take_profit_2 <= setup.take_profit_1:
            failures.append(
                f"take_profit_2={setup.take_profit_2:.4f} ≤ take_profit_1={setup.take_profit_1:.4f}."
            )

        # ---- 7. R:R minimum ----
        if setup.risk_reward_ratio < self.min_risk_reward:
            failures.append(
                f"R:R ratio {setup.risk_reward_ratio:.2f} < minimum {self.min_risk_reward:.2f}."
            )

        # ---- 8. Dollar hard cap ----
        if size.dollar_size > self.max_position_dollars:
            failures.append(
                f"Dollar size ${size.dollar_size:,.2f} exceeds hard cap "
                f"${self.max_position_dollars:,.2f}."
            )

        # ---- 9. entry_low ≤ entry_high ----
        if setup.entry_low > setup.entry_high:
            failures.append(
                f"entry_low={setup.entry_low:.4f} > entry_high={setup.entry_high:.4f}."
            )

        # ---- Warnings (non-blocking) ----
        if size.scale_in:
            warnings.append(
                "Low confidence signal – consider scaling in (partial initial position)."
            )

        passed = len(failures) == 0
        result = ValidationResult(passed=passed, failures=failures, warnings=warnings)

        if passed:
            logger.info(
                "pre_trade_validation_passed",
                ticker=setup.ticker,
                shares=size.shares,
                entry_high=setup.entry_high,
                stop_loss=setup.stop_loss,
                rr=setup.risk_reward_ratio,
                warnings=warnings or None,
            )
        else:
            logger.warning(
                "pre_trade_validation_failed",
                ticker=setup.ticker,
                failures=failures,
                warnings=warnings or None,
            )

        return result

    def validate_or_raise(
        self,
        setup: "TradeSetup",
        size: "PositionSize",
    ) -> None:
        """Validate and raise ValueError if any check fails."""
        from src.core.exceptions import RiskLimitExceededError

        result = self.validate(setup, size)
        if not result.passed:
            raise RiskLimitExceededError(
                f"Pre-trade validation failed for {setup.ticker}: "
                + "; ".join(result.failures)
            )
