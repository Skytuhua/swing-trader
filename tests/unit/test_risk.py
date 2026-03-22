"""
Unit tests for RiskEngine and individual guards.

Tests: kill switch, daily loss limit, max drawdown trigger, stale data rejection,
spread guard, all-pass scenario.
At least 8 test cases.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.enums import DataQuality
from src.services.risk.engine import (
    PortfolioState,
    RiskCheckResult,
    RiskEngine,
    TradingDecisionRisk,
)
from src.services.risk.guards import (
    DailyLossGuard,
    MaxDrawdownGuard,
    MaxSpreadGuard,
    StaleDataGuard,
)
from src.services.risk.kill_switch import KillSwitch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(
    daily_pnl_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    total_risk_pct: float = 0.0,
    portfolio_value: float = 100_000.0,
) -> PortfolioState:
    return PortfolioState(
        portfolio_value=portfolio_value,
        cash=portfolio_value,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=drawdown_pct,
        total_risk_pct=total_risk_pct,
        open_positions=[],
        peak_value=portfolio_value,
    )


def _make_decision(
    ticker: str = "AAPL",
    risk_pct: float = 1.0,
    allocation_pct: float = 10.0,
    minutes_ago: float = 1.0,
) -> TradingDecisionRisk:
    return TradingDecisionRisk(
        selected_ticker=ticker,
        risk_pct=risk_pct,
        allocation_pct=allocation_pct,
        data_timestamp=datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago),
    )


async def _make_kill_switch_coro(active: bool, reason: str):
    """Async helper for building a kill switch mock."""
    pass


class _MockKillSwitch:
    """Simple mock kill switch where is_active and reason are async properties."""

    def __init__(self, active: bool = False, reason_str: str = "") -> None:
        self._active = active
        self._reason = reason_str
        self.activate = AsyncMock()
        self.deactivate = AsyncMock()

    @property
    async def is_active(self) -> bool:  # type: ignore[override]
        return self._active

    @property
    async def reason(self) -> str | None:  # type: ignore[override]
        return self._reason or None


def _make_kill_switch(active: bool = False, reason: str = "") -> "_MockKillSwitch":
    """Create a kill switch mock whose is_active and reason are async properties."""
    return _MockKillSwitch(active=active, reason_str=reason)


def _make_engine(
    kill_switch_active: bool = False,
    max_daily_loss: float = 2.0,
    max_drawdown: float = 10.0,
) -> tuple[RiskEngine, AsyncMock]:
    ks = _make_kill_switch(active=kill_switch_active, reason="test")
    config = MagicMock()
    config.max_daily_loss_pct = max_daily_loss
    config.max_drawdown_pct = max_drawdown
    config.max_total_risk_pct = 6.0
    config.max_position_pct = 20.0
    config.max_spread_pct = 0.5
    config.duplicate_signal_cooldown_minutes = 60.0

    engine = RiskEngine(
        config=config,
        kill_switch=ks,
        calendar=None,
        data_manager=None,
        broker=None,
        stale_data_max_minutes=15.0,
    )
    return engine, ks


# ---------------------------------------------------------------------------
# Individual guard tests
# ---------------------------------------------------------------------------


class TestDailyLossGuard:

    def test_pass_when_within_limit(self):
        """Guard passes when daily loss is within limit."""
        guard = DailyLossGuard(max_daily_loss_pct=2.0)
        result = guard.check(-1.5)
        assert result.passed is True

    def test_fail_when_limit_exceeded(self):
        """Guard fails when daily loss exceeds limit."""
        guard = DailyLossGuard(max_daily_loss_pct=2.0)
        result = guard.check(-2.5)
        assert result.passed is False
        assert "daily loss" in result.reason.lower()

    def test_pass_when_profitable(self):
        """Guard always passes when portfolio is profitable today."""
        guard = DailyLossGuard(max_daily_loss_pct=2.0)
        result = guard.check(1.5)
        assert result.passed is True


class TestMaxDrawdownGuard:

    def test_pass_within_drawdown_limit(self, mock_kill_switch):
        """Guard passes when drawdown is within limit."""
        guard = MaxDrawdownGuard(max_drawdown_pct=10.0, kill_switch=mock_kill_switch)
        result = guard.check(5.0)
        assert result.passed is True

    def test_fail_when_drawdown_breached(self, mock_kill_switch):
        """Guard fails when drawdown exceeds limit."""
        guard = MaxDrawdownGuard(max_drawdown_pct=10.0, kill_switch=mock_kill_switch)
        result = guard.check(12.0)
        assert result.passed is False
        assert "drawdown" in result.reason.lower()


class TestStaleDataGuard:

    def test_pass_for_fresh_data(self):
        """Guard passes for data that is 1 minute old."""
        guard = StaleDataGuard(max_age_minutes=15.0)
        fresh_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        result = guard.check(fresh_ts)
        assert result.passed is True

    def test_fail_for_stale_data(self):
        """Guard fails for data that is 20 minutes old with 15-minute threshold."""
        guard = StaleDataGuard(max_age_minutes=15.0)
        stale_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        result = guard.check(stale_ts)
        assert result.passed is False
        # Reason should mention age or minutes (exact wording may vary)
        assert result.reason is not None
        assert len(result.reason) > 0


class TestMaxSpreadGuard:

    def test_pass_within_spread_limit(self):
        """Guard passes when spread is within limit."""
        guard = MaxSpreadGuard(max_spread_pct=0.5)
        result = guard.check(0.3)
        assert result.passed is True

    def test_fail_when_spread_too_wide(self):
        """Guard fails when spread exceeds limit."""
        guard = MaxSpreadGuard(max_spread_pct=0.5)
        result = guard.check(1.0)
        assert result.passed is False
        assert "spread" in result.reason.lower()


# ---------------------------------------------------------------------------
# RiskEngine integration tests
# ---------------------------------------------------------------------------


class TestRiskEngine:

    def test_all_pass_scenario(self):
        """Pre-trade check passes when all conditions are healthy."""
        engine, _ = _make_engine(kill_switch_active=False)
        portfolio = _make_portfolio(daily_pnl_pct=0.5, drawdown_pct=2.0, total_risk_pct=2.0)
        decision = _make_decision(risk_pct=1.0, allocation_pct=10.0, minutes_ago=1.0)
        result = asyncio.get_event_loop().run_until_complete(
            engine.pre_trade_check(decision, portfolio)
        )
        assert result.passed is True
        assert len(result.checks_run) > 0

    def test_kill_switch_blocks_trading(self):
        """Pre-trade check fails immediately when kill switch is active."""
        # Engine is created with kill_switch_active=True; no need to set it again
        engine, ks = _make_engine(kill_switch_active=True)
        portfolio = _make_portfolio()
        decision = _make_decision()

        async def run():
            return await engine.pre_trade_check(decision, portfolio)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.passed is False
        assert result.failed_check == "kill_switch"

    def test_daily_loss_limit_blocks_trading(self):
        """Pre-trade check fails when daily loss limit is breached."""
        engine, _ = _make_engine(kill_switch_active=False, max_daily_loss=2.0)
        # Daily loss of -3% exceeds 2% limit
        portfolio = _make_portfolio(daily_pnl_pct=-3.0)
        decision = _make_decision()
        result = asyncio.get_event_loop().run_until_complete(
            engine.pre_trade_check(decision, portfolio)
        )
        assert result.passed is False
        assert result.failed_check == "daily_loss"

    def test_max_drawdown_blocks_trading(self):
        """Pre-trade check fails when max drawdown is breached."""
        engine, _ = _make_engine(kill_switch_active=False, max_drawdown=10.0)
        portfolio = _make_portfolio(drawdown_pct=12.0)
        decision = _make_decision()
        result = asyncio.get_event_loop().run_until_complete(
            engine.pre_trade_check(decision, portfolio)
        )
        assert result.passed is False
        assert result.failed_check == "max_drawdown"

    def test_stale_data_blocks_trading(self):
        """Pre-trade check fails when data timestamp is too old."""
        engine, _ = _make_engine()
        portfolio = _make_portfolio()
        # Data 30 minutes old with 15-minute threshold
        decision = _make_decision(minutes_ago=30.0)
        result = asyncio.get_event_loop().run_until_complete(
            engine.pre_trade_check(decision, portfolio)
        )
        assert result.passed is False
        assert result.failed_check == "data_freshness"

    def test_result_summary_method(self):
        """RiskCheckResult.summary() returns a human-readable string."""
        passed = RiskCheckResult(passed=True, checks_run=["kill_switch", "market_hours"])
        failed = RiskCheckResult(
            passed=False, reason="Kill switch active", failed_check="kill_switch"
        )
        assert "PASS" in passed.summary()
        assert "FAIL" in failed.summary()
        assert "kill_switch" in failed.summary()
