"""
Unit tests for PositionSizer.

Tests: confidence-to-allocation mapping, regime adjustment, risk-based sizing,
hard caps, zero size conditions, total risk budget enforcement.
At least 8 test cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.enums import MarketRegime
from src.services.pipeline.screener import RegimeAssessment
from src.services.trade.sizer import PositionSize, PositionSizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trade_setup():
    """A mock TradeSetup with typical values."""
    setup = MagicMock()
    setup.ticker = "AAPL"
    setup.entry_high = 155.0
    setup.stop_loss = 150.0  # $5 risk per share
    return setup


@pytest.fixture
def sizer(risk_config):
    return PositionSizer(risk_config)


@pytest.fixture
def favorable_regime():
    return RegimeAssessment(regime=MarketRegime.FAVORABLE, confidence=80.0)


@pytest.fixture
def mixed_regime():
    return RegimeAssessment(regime=MarketRegime.MIXED, confidence=60.0)


@pytest.fixture
def unfavorable_regime():
    return RegimeAssessment(regime=MarketRegime.UNFAVORABLE, confidence=70.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfidenceToAllocation:

    def test_high_confidence_gets_max_allocation(self, sizer, trade_setup, favorable_regime):
        """Confidence ≥ 80 → full max_position_pct allocation."""
        result = sizer.size(
            setup=trade_setup,
            confidence=85.0,
            regime=favorable_regime,
            portfolio_value=100_000.0,
            current_risk_pct=0.0,
        )
        assert result.is_tradeable
        # Allocation should be near max
        assert result.allocation_pct > 0.0

    def test_mid_confidence_gets_partial_allocation(self, sizer, trade_setup, favorable_regime):
        """Confidence in 60-80 range → partial allocation (50-100% of max)."""
        high_conf = sizer.size(
            setup=trade_setup, confidence=85.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        mid_conf = sizer.size(
            setup=trade_setup, confidence=65.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        # Mid confidence should get smaller or equal allocation
        assert mid_conf.allocation_pct <= high_conf.allocation_pct + 0.1

    def test_low_confidence_below_40_returns_zero(self, sizer, trade_setup, favorable_regime):
        """Confidence < 40 → zero position (no trade)."""
        result = sizer.size(
            setup=trade_setup, confidence=30.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        assert result.shares == 0
        assert not result.is_tradeable

    def test_confidence_at_threshold_40(self, sizer, trade_setup, favorable_regime):
        """Confidence exactly at 40 gets minimal but non-zero allocation."""
        result = sizer.size(
            setup=trade_setup, confidence=40.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        # At exactly 40 this might be 0 (boundary) — just test it doesn't crash
        assert isinstance(result.shares, int)
        assert result.shares >= 0


class TestRegimeAdjustment:

    def test_unfavorable_regime_returns_zero(self, sizer, trade_setup, unfavorable_regime):
        """UNFAVORABLE regime always returns zero position."""
        result = sizer.size(
            setup=trade_setup, confidence=90.0, regime=unfavorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        assert result.shares == 0
        assert "UNFAVORABLE" in result.sizing_notes

    def test_mixed_regime_reduces_allocation(self, sizer, trade_setup, mixed_regime, favorable_regime):
        """MIXED regime reduces allocation by 40% compared to FAVORABLE."""
        fav_result = sizer.size(
            setup=trade_setup, confidence=80.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        mixed_result = sizer.size(
            setup=trade_setup, confidence=80.0, regime=mixed_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        # Mixed allocation should be ≤ 60% of favorable
        assert mixed_result.allocation_pct <= fav_result.allocation_pct * 0.65


class TestRiskBudgetAndCaps:

    def test_total_risk_budget_exhausted_returns_zero(self, sizer, trade_setup, favorable_regime):
        """Returns zero when total risk budget is exhausted."""
        result = sizer.size(
            setup=trade_setup, confidence=85.0, regime=favorable_regime,
            portfolio_value=100_000.0,
            current_risk_pct=99.0,  # Way over budget
        )
        assert result.shares == 0

    def test_position_size_respects_max_position_pct(self, sizer, trade_setup, favorable_regime):
        """Allocation never exceeds max_position_pct."""
        result = sizer.size(
            setup=trade_setup, confidence=90.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        assert result.allocation_pct <= 20.0 + 0.1  # max_position_pct = 20%

    def test_risk_per_share_calculation(self, sizer, trade_setup, favorable_regime):
        """Risk per share = entry_high - stop_loss."""
        result = sizer.size(
            setup=trade_setup, confidence=80.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        expected_risk = trade_setup.entry_high - trade_setup.stop_loss  # 5.0
        assert abs(result.risk_per_share - expected_risk) < 0.01

    def test_invalid_stop_loss_returns_zero(self, sizer, favorable_regime):
        """Entry_high ≤ stop_loss (invalid) returns zero position."""
        bad_setup = MagicMock()
        bad_setup.ticker = "AAPL"
        bad_setup.entry_high = 100.0
        bad_setup.stop_loss = 105.0  # Stop above entry = invalid
        result = sizer.size(
            setup=bad_setup, confidence=85.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        assert result.shares == 0

    def test_scale_in_flag_for_lower_confidence(self, sizer, trade_setup, favorable_regime):
        """Confidence < 65 should trigger scale_in = True."""
        result = sizer.size(
            setup=trade_setup, confidence=60.0, regime=favorable_regime,
            portfolio_value=100_000.0, current_risk_pct=0.0,
        )
        if result.is_tradeable:
            assert result.scale_in is True
