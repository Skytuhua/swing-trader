"""Tests for Adaptive ATR Trailing Stop System (src/strategy/adaptive_stops.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategy.adaptive_stops import (
    AdaptiveStopManager,
    AdaptiveStopState,
    ScaleOutResult,
    StopLevel,
    StopType,
    compute_atr,
)


# ---------------------------------------------------------------------------
# ATR computation tests
# ---------------------------------------------------------------------------


class TestComputeATR:
    def test_basic_atr(self):
        n = 30
        highs = np.linspace(101, 110, n)
        lows = np.linspace(99, 108, n)
        closes = np.linspace(100, 109, n)
        atr = compute_atr(highs, lows, closes, 14)
        assert atr > 0

    def test_atr_flat_prices(self):
        n = 30
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        closes = np.full(n, 100.0)
        atr = compute_atr(highs, lows, closes, 14)
        assert 1.5 <= atr <= 2.5  # Range is 2, so ATR ≈ 2

    def test_atr_insufficient_data(self):
        atr = compute_atr(np.array([101.0]), np.array([99.0]), np.array([100.0]), 14)
        assert atr == 0.0

    def test_atr_two_bars(self):
        atr = compute_atr(
            np.array([102.0, 104.0]),
            np.array([98.0, 100.0]),
            np.array([100.0, 103.0]),
            14,
        )
        assert atr > 0


# ---------------------------------------------------------------------------
# Stop computation tests
# ---------------------------------------------------------------------------


class TestAdaptiveStopManager:
    @pytest.fixture
    def manager(self):
        return AdaptiveStopManager({
            "atr_period": 14,
            "trailing_multiplier": 2.5,
            "chandelier_enabled": False,
            "scale_out_enabled": True,
            "scale_out_pct": 0.5,
            "first_target_atr_mult": 2.0,
            "second_target_atr_mult": 4.0,
        })

    @pytest.fixture
    def chandelier_manager(self):
        return AdaptiveStopManager({
            "atr_period": 14,
            "trailing_multiplier": 2.5,
            "chandelier_enabled": True,
        })

    @pytest.fixture
    def price_data(self):
        n = 30
        rng = np.random.default_rng(42)
        closes = 100 + np.cumsum(rng.normal(0.1, 0.5, n))
        highs = closes + rng.uniform(0.5, 1.5, n)
        lows = closes - rng.uniform(0.5, 1.5, n)
        return highs, lows, closes

    def test_compute_stop_long(self, manager, price_data):
        highs, lows, closes = price_data
        stop = manager.compute_stop(highs, lows, closes, side="long")
        assert isinstance(stop, StopLevel)
        assert stop.price < closes[-1]  # Stop below current price for longs
        assert stop.atr_value > 0
        assert stop.is_trailing is True

    def test_compute_stop_short(self, manager, price_data):
        highs, lows, closes = price_data
        stop = manager.compute_stop(highs, lows, closes, side="short")
        assert stop.price > closes[-1]  # Stop above current price for shorts

    def test_trending_wider_stops(self, manager, price_data):
        highs, lows, closes = price_data
        trending_stop = manager.compute_stop(highs, lows, closes, side="long", regime_trending=True)
        ranging_stop = manager.compute_stop(highs, lows, closes, side="long", regime_trending=False)
        # Trending stop should be wider (further from price)
        assert trending_stop.atr_multiplier >= ranging_stop.atr_multiplier

    def test_chandelier_stop(self, chandelier_manager, price_data):
        highs, lows, closes = price_data
        stop = chandelier_manager.compute_stop(highs, lows, closes, side="long")
        assert stop.stop_type == StopType.CHANDELIER
        assert stop.lookback_high > 0

    def test_atr_override(self, manager, price_data):
        highs, lows, closes = price_data
        stop = manager.compute_stop(highs, lows, closes, side="long", atr_override=5.0)
        assert stop.atr_value == 5.0

    def test_stop_notes(self, manager, price_data):
        highs, lows, closes = price_data
        stop = manager.compute_stop(highs, lows, closes, side="long")
        assert "ATR" in stop.notes


# ---------------------------------------------------------------------------
# Trailing stop update tests
# ---------------------------------------------------------------------------


class TestTrailingStopUpdate:
    def test_long_stop_ratchets_up(self):
        manager = AdaptiveStopManager()
        state = AdaptiveStopState(
            ticker="AAPL",
            side="long",
            entry_price=100.0,
            current_stop=95.0,
            highest_high=105.0,
            lowest_low=95.0,
            atr=2.0,
            multiplier=2.5,
        )

        # New high should ratchet stop up
        new_stop = manager.update_trailing_stop(state, 108.0, 104.0, 107.0, 2.0)
        assert new_stop > 95.0  # Stop moved up
        assert state.highest_high == 108.0

    def test_long_stop_never_moves_down(self):
        manager = AdaptiveStopManager()
        state = AdaptiveStopState(
            ticker="AAPL",
            side="long",
            entry_price=100.0,
            current_stop=100.0,  # Already at breakeven
            highest_high=108.0,
            lowest_low=95.0,
            atr=2.0,
            multiplier=2.5,
        )

        # Lower price shouldn't move stop down
        new_stop = manager.update_trailing_stop(state, 101.0, 99.0, 100.0, 2.0)
        assert new_stop >= 100.0

    def test_short_stop_ratchets_down(self):
        manager = AdaptiveStopManager()
        state = AdaptiveStopState(
            ticker="AAPL",
            side="short",
            entry_price=100.0,
            current_stop=105.0,
            highest_high=105.0,
            lowest_low=95.0,
            atr=2.0,
            multiplier=2.5,
        )

        # New low should ratchet stop down
        new_stop = manager.update_trailing_stop(state, 94.0, 90.0, 92.0, 2.0)
        assert new_stop < 105.0
        assert state.lowest_low == 90.0

    def test_bars_since_entry_increments(self):
        manager = AdaptiveStopManager()
        state = AdaptiveStopState(
            ticker="AAPL", side="long", entry_price=100.0,
            current_stop=95.0, highest_high=105.0, lowest_low=95.0,
            atr=2.0, multiplier=2.5,
        )
        assert state.bars_since_entry == 0
        manager.update_trailing_stop(state, 106.0, 104.0, 105.0, 2.0)
        assert state.bars_since_entry == 1


# ---------------------------------------------------------------------------
# Scale-out tests
# ---------------------------------------------------------------------------


class TestScaleOut:
    @pytest.fixture
    def manager(self):
        return AdaptiveStopManager({
            "scale_out_enabled": True,
            "scale_out_pct": 0.5,
            "first_target_atr_mult": 2.0,
            "second_target_atr_mult": 4.0,
        })

    def test_no_scale_out_below_tp1(self, manager):
        state = AdaptiveStopState(
            ticker="AAPL", side="long", entry_price=100.0,
            current_stop=95.0, highest_high=100.0, lowest_low=95.0,
            atr=2.0, multiplier=2.5,
        )
        result = manager.evaluate_scale_out(state, 102.0, 2.0)
        assert result.should_scale_out is False

    def test_scale_out_at_tp1(self, manager):
        state = AdaptiveStopState(
            ticker="AAPL", side="long", entry_price=100.0,
            current_stop=95.0, highest_high=100.0, lowest_low=95.0,
            atr=2.0, multiplier=2.5,
        )
        # TP1 = entry + 2.0 * ATR = 100 + 4 = 104
        result = manager.evaluate_scale_out(state, 104.5, 2.0)
        assert result.should_scale_out is True
        assert result.target_hit == "tp1"
        assert result.scale_out_pct == 0.5
        assert result.move_stop_to_breakeven is True
        assert result.new_stop == 100.0  # Breakeven
        assert state.tp1_hit is True

    def test_scale_out_at_tp2(self, manager):
        state = AdaptiveStopState(
            ticker="AAPL", side="long", entry_price=100.0,
            current_stop=100.0, highest_high=108.0, lowest_low=95.0,
            atr=2.0, multiplier=2.5, tp1_hit=True,
        )
        # TP2 = entry + 4.0 * ATR = 100 + 8 = 108
        result = manager.evaluate_scale_out(state, 109.0, 2.0)
        assert result.should_scale_out is True
        assert result.target_hit == "tp2"
        assert result.scale_out_pct == 1.0

    def test_scale_out_disabled(self):
        manager = AdaptiveStopManager({"scale_out_enabled": False})
        state = AdaptiveStopState(
            ticker="AAPL", side="long", entry_price=100.0,
            current_stop=95.0, highest_high=100.0, lowest_low=95.0,
            atr=2.0, multiplier=2.5,
        )
        result = manager.evaluate_scale_out(state, 110.0, 2.0)
        assert result.should_scale_out is False

    def test_short_scale_out(self, manager):
        state = AdaptiveStopState(
            ticker="AAPL", side="short", entry_price=100.0,
            current_stop=105.0, highest_high=105.0, lowest_low=100.0,
            atr=2.0, multiplier=2.5,
        )
        # TP1 for short = entry - 2.0 * ATR = 100 - 4 = 96
        result = manager.evaluate_scale_out(state, 95.0, 2.0)
        assert result.should_scale_out is True
        assert result.target_hit == "tp1"


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_create_and_get_state(self):
        manager = AdaptiveStopManager()
        state = manager.create_state("AAPL", "long", 150.0, 145.0, 3.0)
        assert state.ticker == "AAPL"
        assert state.entry_price == 150.0
        retrieved = manager.get_state("AAPL")
        assert retrieved is state

    def test_remove_state(self):
        manager = AdaptiveStopManager()
        manager.create_state("AAPL", "long", 150.0, 145.0, 3.0)
        manager.remove_state("AAPL")
        assert manager.get_state("AAPL") is None

    def test_active_states(self):
        manager = AdaptiveStopManager()
        manager.create_state("AAPL", "long", 150.0, 145.0, 3.0)
        manager.create_state("MSFT", "long", 300.0, 290.0, 5.0)
        assert len(manager.active_states) == 2
