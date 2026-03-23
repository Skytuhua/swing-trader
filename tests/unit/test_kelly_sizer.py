"""Tests for Kelly Criterion Position Sizing (src/risk/position_sizer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.position_sizer import KellyPositionSizer, KellyResult, TradeRecord


# ---------------------------------------------------------------------------
# Trade record helpers
# ---------------------------------------------------------------------------


def _make_trade_history(n_trades: int = 50, win_rate: float = 0.6, avg_win: float = 200.0, avg_loss: float = 150.0) -> list[TradeRecord]:
    """Generate synthetic trade history."""
    rng = np.random.default_rng(42)
    trades = []
    for _ in range(n_trades):
        if rng.random() < win_rate:
            pnl = avg_win * rng.uniform(0.5, 1.5)
        else:
            pnl = -avg_loss * rng.uniform(0.5, 1.5)
        trades.append(TradeRecord(pnl=pnl, pnl_pct=pnl / 10_000 * 100))
    return trades


# ---------------------------------------------------------------------------
# Kelly formula tests
# ---------------------------------------------------------------------------


class TestKellyFormula:
    def test_half_kelly_basic(self):
        """Test half-Kelly with known win rate and W/L ratio."""
        sizer = KellyPositionSizer({"method": "half_kelly", "min_trades_for_kelly": 5})
        # Seed with history
        for t in _make_trade_history(50, win_rate=0.6, avg_win=200, avg_loss=150):
            sizer.record_trade(t)

        result = sizer.size(
            portfolio_value=100_000,
            entry_price=150.0,
            stop_loss=145.0,
        )
        assert isinstance(result, KellyResult)
        assert result.method == "half_kelly"
        assert result.shares > 0
        assert result.position_value > 0
        assert result.full_kelly_fraction > 0

    def test_full_kelly(self):
        sizer = KellyPositionSizer({"method": "kelly", "min_trades_for_kelly": 5})
        for t in _make_trade_history(50, win_rate=0.6, avg_win=200, avg_loss=150):
            sizer.record_trade(t)

        result = sizer.size(100_000, 150.0, 145.0)
        assert result.method == "kelly"
        assert result.full_kelly_fraction >= result.fraction  # Half is always <= full

    def test_half_kelly_is_half_of_full(self):
        sizer_half = KellyPositionSizer({"method": "half_kelly", "min_trades_for_kelly": 5})
        sizer_full = KellyPositionSizer({"method": "kelly", "min_trades_for_kelly": 5})

        trades = _make_trade_history(50, 0.6, 200, 150)
        for t in trades:
            sizer_half.record_trade(t)
            sizer_full.record_trade(t)

        half_result = sizer_half.size(100_000, 150.0, 145.0)
        full_result = sizer_full.size(100_000, 150.0, 145.0)

        # Half-Kelly fraction should be roughly half of full Kelly
        assert abs(half_result.fraction - full_result.fraction / 2) < 0.01


# ---------------------------------------------------------------------------
# Fallback to fixed fraction
# ---------------------------------------------------------------------------


class TestFixedFractionFallback:
    def test_insufficient_history_uses_fixed(self):
        sizer = KellyPositionSizer({
            "method": "half_kelly",
            "min_trades_for_kelly": 30,
            "default_risk_per_trade_pct": 1.0,
        })
        # Only 5 trades — not enough for Kelly
        for t in _make_trade_history(5):
            sizer.record_trade(t)

        result = sizer.size(100_000, 150.0, 145.0)
        assert result.method == "fixed_fraction"
        assert result.shares > 0

    def test_no_history_uses_fixed(self):
        sizer = KellyPositionSizer({"method": "half_kelly"})
        result = sizer.size(100_000, 150.0, 145.0)
        assert result.method == "fixed_fraction"

    def test_fixed_fraction_sizing(self):
        sizer = KellyPositionSizer({
            "method": "fixed_fraction",
            "default_risk_per_trade_pct": 2.0,
            "max_position_pct": 100.0,  # No cap for this test
        })
        result = sizer.size(100_000, 150.0, 145.0)
        # Risk budget = 100_000 * 2% = 2_000
        # Risk per share = 150 - 145 = 5
        # Shares = 2_000 / 5 = 400
        assert result.shares == 400
        assert result.method == "fixed_fraction"


# ---------------------------------------------------------------------------
# Volatility-adjusted sizing
# ---------------------------------------------------------------------------


class TestVolatilityAdjusted:
    def test_volatility_adjusted_method(self):
        sizer = KellyPositionSizer({
            "method": "volatility_adjusted",
            "default_risk_per_trade_pct": 1.0,
        })
        result = sizer.size(100_000, 150.0, 145.0, atr=3.0)
        assert result.method == "volatility_adjusted"
        assert result.shares > 0

    def test_higher_atr_smaller_position(self):
        sizer = KellyPositionSizer({
            "method": "volatility_adjusted",
            "default_risk_per_trade_pct": 1.0,
            "max_position_pct": 100.0,  # No cap for this test
        })
        result_low_vol = sizer.size(100_000, 150.0, 145.0, atr=2.0)
        result_high_vol = sizer.size(100_000, 150.0, 145.0, atr=5.0)
        assert result_low_vol.shares > result_high_vol.shares


# ---------------------------------------------------------------------------
# Position cap enforcement
# ---------------------------------------------------------------------------


class TestPositionCap:
    def test_max_position_pct_enforced(self):
        sizer = KellyPositionSizer({
            "method": "half_kelly",
            "max_position_pct": 2.0,
            "min_trades_for_kelly": 5,
        })
        for t in _make_trade_history(50, 0.8, 300, 100):  # Very favorable stats
            sizer.record_trade(t)

        result = sizer.size(100_000, 150.0, 145.0)
        # Max position = 100_000 * 2% = 2_000 → ~13 shares
        assert result.position_value <= 100_000 * 0.02 + 150  # Allow 1 share rounding


# ---------------------------------------------------------------------------
# Regime adjustment
# ---------------------------------------------------------------------------


class TestRegimeAdjustment:
    def test_reduced_aggression(self):
        sizer = KellyPositionSizer({
            "method": "fixed_fraction",
            "default_risk_per_trade_pct": 2.0,
            "max_position_pct": 100.0,  # No cap so regime adjustment is visible
        })
        result_full = sizer.size(100_000, 150.0, 145.0, regime_aggression=1.0)
        result_half = sizer.size(100_000, 150.0, 145.0, regime_aggression=0.5)
        assert result_half.shares < result_full.shares


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_portfolio(self):
        sizer = KellyPositionSizer()
        result = sizer.size(0, 150.0, 145.0)
        assert result.shares == 0

    def test_zero_entry_price(self):
        sizer = KellyPositionSizer()
        result = sizer.size(100_000, 0, -5.0)
        assert result.shares == 0

    def test_stop_above_entry(self):
        sizer = KellyPositionSizer()
        result = sizer.size(100_000, 150.0, 155.0)
        # Should still work (risk_per_share = |150-155| = 5)
        assert result.shares >= 0

    def test_negative_kelly(self):
        """Poor win rate should result in 0 Kelly fraction (don't trade)."""
        sizer = KellyPositionSizer({"method": "kelly", "min_trades_for_kelly": 5})
        # 20% win rate with 1:1 ratio → negative edge
        for t in _make_trade_history(50, win_rate=0.2, avg_win=100, avg_loss=100):
            sizer.record_trade(t)

        result = sizer.size(100_000, 150.0, 145.0)
        assert result.full_kelly_fraction == 0.0


# ---------------------------------------------------------------------------
# Trade history management
# ---------------------------------------------------------------------------


class TestTradeHistory:
    def test_record_and_count(self):
        sizer = KellyPositionSizer()
        for t in _make_trade_history(10):
            sizer.record_trade(t)
        assert sizer.trade_count == 10

    def test_has_sufficient_history(self):
        sizer = KellyPositionSizer({"min_trades_for_kelly": 30})
        assert sizer.has_sufficient_history is False
        for t in _make_trade_history(30):
            sizer.record_trade(t)
        assert sizer.has_sufficient_history is True

    def test_get_stats(self):
        sizer = KellyPositionSizer()
        for t in _make_trade_history(50, 0.6, 200, 150):
            sizer.record_trade(t)
        stats = sizer.get_stats()
        assert "win_rate" in stats
        assert "kelly_fraction" in stats
        assert stats["trade_count"] == 50
        assert 0 < stats["win_rate"] < 1
