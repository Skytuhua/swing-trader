"""
Risk engine: orchestrate all pre-trade risk checks.

RiskEngine.pre_trade_check() is the single entry point that must return
RiskCheckResult(passed=True) before any order is submitted to the broker.

Checks (in order):
  1. Kill switch not active
  2. Market is open
  3. Daily loss within limit
  4. Portfolio drawdown within limit (activates kill switch if breached)
  5. Total risk budget not exhausted
  6. Position concentration within limit
  7. Data freshness
  8. Bid-ask spread within limit
  9. Duplicate signal cooldown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import KillSwitchActiveError

if TYPE_CHECKING:
    from src.core.config import RiskConfig
    from src.services.execution.base import BrokerAdapter
    from src.services.market_data.manager import DataManager
    from src.services.risk.calendar import MarketCalendar
    from src.services.risk.kill_switch import KillSwitch

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check pass."""

    passed: bool
    reason: str = ""
    checks_run: list[str] = field(default_factory=list)
    failed_check: str | None = None

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return f"PASS ({len(self.checks_run)} checks passed)"
        return f"FAIL at [{self.failed_check}]: {self.reason}"


@dataclass
class PortfolioState:
    """Snapshot of current portfolio risk metrics."""

    portfolio_value: float
    cash: float
    daily_pnl_pct: float          # Today's P&L as % (negative = loss)
    drawdown_pct: float           # Peak-to-trough drawdown as %
    total_risk_pct: float         # % of portfolio currently at risk across positions
    open_position_count: int = 0
    open_positions: list[str] = field(default_factory=list)
    peak_value: float = 0.0
    computed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingDecisionRisk:
    """Risk attributes of a proposed trade for pre-trade checking."""

    selected_ticker: str
    risk_pct: float         # Risk this trade adds to total portfolio risk %
    allocation_pct: float   # % of portfolio this trade would allocate
    data_timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Orchestrate all pre-trade risk checks.

    All checks are fail-fast: the first failure returns immediately without
    running subsequent checks (to avoid unnecessary calls in degenerate states).

    Usage::

        risk = RiskEngine(config, kill_switch, calendar, data_manager, broker)
        result = await risk.pre_trade_check(decision, portfolio)
        if not result:
            return  # Do not trade
    """

    def __init__(
        self,
        config: "RiskConfig",
        kill_switch: "KillSwitch",
        calendar: "MarketCalendar | None" = None,
        data_manager: "DataManager | None" = None,
        broker: "BrokerAdapter | None" = None,
        stale_data_max_minutes: float = 15.0,
    ) -> None:
        self.config = config
        self.kill_switch = kill_switch
        self.calendar = calendar
        self.data = data_manager
        self.broker = broker
        self._stale_threshold = timedelta(minutes=stale_data_max_minutes)

        # Lazy-import guards to avoid circular deps
        from src.services.risk.guards import (
            DailyLossGuard,
            DuplicateSignalGuard,
            MaxDrawdownGuard,
            MaxSpreadGuard,
            StaleDataGuard,
        )

        self._daily_loss_guard = DailyLossGuard(
            max_daily_loss_pct=float(
                getattr(config, "max_daily_loss_pct", 2.0) or 2.0
            )
        )
        self._drawdown_guard = MaxDrawdownGuard(
            max_drawdown_pct=float(
                getattr(config, "max_drawdown_pct", 10.0) or 10.0
            ),
            kill_switch=kill_switch,
        )
        self._spread_guard = MaxSpreadGuard(
            max_spread_pct=float(
                getattr(config, "max_spread_pct", 0.5) or 0.5
            )
        )
        self._stale_guard = StaleDataGuard(
            max_age_minutes=stale_data_max_minutes
        )
        self._duplicate_guard = DuplicateSignalGuard(
            cooldown_minutes=float(
                getattr(config, "duplicate_signal_cooldown_minutes", 60.0) or 60.0
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def pre_trade_check(
        self,
        decision: "TradingDecisionRisk",
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Run all pre-trade risk checks.

        Args:
            decision:  The proposed trade decision.
            portfolio: Current portfolio state snapshot.

        Returns:
            RiskCheckResult.  If passed=False, reason is populated.
        """
        checks: list[str] = []

        # ---- 1. Kill switch ----
        try:
            if await self.kill_switch.is_active:
                reason = await self.kill_switch.reason
                return RiskCheckResult(
                    passed=False,
                    reason=f"Kill switch is active: {reason or 'unknown'}.",
                    checks_run=checks,
                    failed_check="kill_switch",
                )
        except Exception as exc:
            # Fail safe: if we can't read kill switch status, block trading
            return RiskCheckResult(
                passed=False,
                reason=f"Kill switch status check failed: {exc}",
                checks_run=checks,
                failed_check="kill_switch",
            )
        checks.append("kill_switch")

        # ---- 2. Market hours ----
        if self.calendar is not None and not self.calendar.is_market_open():
            return RiskCheckResult(
                passed=False,
                reason="Market is closed.",
                checks_run=checks,
                failed_check="market_hours",
            )
        checks.append("market_hours")

        # ---- 3. Daily loss limit ----
        daily_result = self._daily_loss_guard.check(portfolio.daily_pnl_pct)
        if not daily_result:
            return RiskCheckResult(
                passed=False,
                reason=daily_result.reason,
                checks_run=checks,
                failed_check="daily_loss",
            )
        checks.append("daily_loss")

        # ---- 4. Max drawdown ----
        dd_result = self._drawdown_guard.check(portfolio.drawdown_pct)
        if not dd_result:
            # Activate kill switch
            try:
                await self.kill_switch.activate(
                    reason=f"Max drawdown {portfolio.drawdown_pct:.2f}% breached."
                )
            except Exception as exc:
                logger.error("risk_engine_kill_switch_activation_failed", error=str(exc))
            return RiskCheckResult(
                passed=False,
                reason=dd_result.reason,
                checks_run=checks,
                failed_check="max_drawdown",
            )
        checks.append("max_drawdown")

        # ---- 5. Total risk budget ----
        max_total_risk = float(getattr(self.config, "max_total_risk_pct", 6.0) or 6.0)
        projected_total_risk = portfolio.total_risk_pct + decision.risk_pct
        if projected_total_risk > max_total_risk:
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"Total risk budget: projected {projected_total_risk:.2f}% "
                    f"> max {max_total_risk:.2f}%."
                ),
                checks_run=checks,
                failed_check="total_risk_budget",
            )
        checks.append("total_risk_budget")

        # ---- 6. Position concentration ----
        max_pos_pct = float(getattr(self.config, "max_position_pct", 20.0) or 20.0)
        if decision.allocation_pct > max_pos_pct:
            return RiskCheckResult(
                passed=False,
                reason=(
                    f"Position allocation {decision.allocation_pct:.2f}% "
                    f"> max {max_pos_pct:.2f}%."
                ),
                checks_run=checks,
                failed_check="position_concentration",
            )
        checks.append("position_concentration")

        # ---- 7. Data freshness ----
        stale_result = self._stale_guard.check(decision.data_timestamp)
        if not stale_result:
            return RiskCheckResult(
                passed=False,
                reason=stale_result.reason,
                checks_run=checks,
                failed_check="data_freshness",
            )
        checks.append("data_freshness")

        # ---- 8. Spread check ----
        if self.broker is not None:
            try:
                quote = await self.broker.get_quote(decision.selected_ticker)
                spread_result = self._spread_guard.check(quote.spread_pct)
                if not spread_result:
                    return RiskCheckResult(
                        passed=False,
                        reason=spread_result.reason,
                        checks_run=checks,
                        failed_check="spread",
                    )
            except Exception as exc:
                logger.warning(
                    "risk_engine_spread_check_failed",
                    ticker=decision.selected_ticker,
                    error=str(exc),
                )
        checks.append("spread")

        # ---- 9. Duplicate signal ----
        dup_result = self._duplicate_guard.check(decision.selected_ticker)
        if not dup_result:
            return RiskCheckResult(
                passed=False,
                reason=dup_result.reason,
                checks_run=checks,
                failed_check="duplicate_signal",
            )
        checks.append("duplicate_signal")

        logger.info(
            "risk_engine_pre_trade_passed",
            ticker=decision.selected_ticker,
            checks=len(checks),
        )
        return RiskCheckResult(passed=True, checks_run=checks)

    # ------------------------------------------------------------------
    # Entry recording
    # ------------------------------------------------------------------

    def record_trade_entry(self, ticker: str) -> None:
        """Call after a successful order submission to update cooldown state."""
        self._duplicate_guard.record_entry(ticker)

    def record_trade_exit(self, ticker: str) -> None:
        """Call after full position exit to clear the cooldown."""
        self._duplicate_guard.clear_ticker(ticker)

    # ------------------------------------------------------------------
    # Portfolio state builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_portfolio_state(
        portfolio_value: float,
        cash: float,
        daily_pnl_pct: float,
        peak_value: float,
        total_risk_pct: float,
        open_positions: list[str] | None = None,
    ) -> PortfolioState:
        """Convenience constructor for PortfolioState."""
        drawdown_pct = (
            max(0.0, (peak_value - portfolio_value) / peak_value * 100.0)
            if peak_value > 0
            else 0.0
        )
        return PortfolioState(
            portfolio_value=portfolio_value,
            cash=cash,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=drawdown_pct,
            total_risk_pct=total_risk_pct,
            peak_value=peak_value,
            open_position_count=len(open_positions or []),
            open_positions=open_positions or [],
        )
