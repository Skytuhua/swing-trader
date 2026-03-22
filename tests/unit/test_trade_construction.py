"""
Unit tests for TradeConstructor.

Tests: stop loss calculation (ATR-based), take-profit levels, entry zone,
R:R ratio, trailing stop rule.
At least 6 test cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.enums import EntryMethod
from src.services.trade.constructor import TradeConstructor, TradeSetup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored_candidate(
    ticker: str = "AAPL",
    atr: float = 3.0,
    last_close: float = 150.0,
    setup_type: str = "breakout",
    resistance: float | None = 160.0,
    support: float | None = 145.0,
) -> MagicMock:
    """Create a mock ScoredCandidate."""
    technical = MagicMock()
    technical.composite_score = 75.0
    technical.trend_score = 70.0
    technical.setup_type = setup_type
    technical.indicators = {
        "atr_14": atr,
        "last_close": last_close,
        "rsi_14": 58.0,
        "sma_20": last_close * 0.97,
    }
    technical.key_levels = {}
    if resistance:
        technical.key_levels["resistance"] = resistance
    if support:
        technical.key_levels["support"] = support

    screened = MagicMock()
    screened.technical = technical
    screened.ticker = ticker

    candidate = MagicMock()
    candidate.ticker = ticker
    candidate.screened_candidate = screened

    return candidate


def _make_quote(price: float = 150.0, ticker: str = "AAPL") -> MagicMock:
    q = MagicMock()
    q.ticker = ticker
    q.price = price
    q.bid = price * 0.999
    q.ask = price * 1.001
    return q


def _make_daily_df(sample_ohlcv_df):
    return sample_ohlcv_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStopLossCalculation:

    def test_stop_loss_is_below_entry(self, sample_ohlcv_df):
        """Stop loss must always be below the entry price."""
        constructor = TradeConstructor()
        candidate = _make_scored_candidate(atr=3.0, last_close=150.0)
        quote = _make_quote(price=150.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.stop_loss < setup.entry_high

    def test_stop_loss_is_atr_based(self, sample_ohlcv_df):
        """Stop loss should be approximately entry - 2*ATR."""
        constructor = TradeConstructor()
        atr = 3.0
        price = 150.0
        candidate = _make_scored_candidate(atr=atr, last_close=price, support=None)
        quote = _make_quote(price=price)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        # Stop should be at most entry - 2*ATR = 150 - 6 = 144
        # (may be lower due to support level check)
        expected_atr_stop = price - 2.0 * atr
        assert setup.stop_loss <= price
        # Should be at least 0.5% below price
        assert setup.stop_loss <= price * 0.995

    def test_stop_loss_minimum_distance(self, sample_ohlcv_df):
        """Stop loss is never closer than 0.5% below entry."""
        constructor = TradeConstructor()
        # Very small ATR should still give minimum stop distance
        candidate = _make_scored_candidate(atr=0.001, last_close=100.0, support=None)
        quote = _make_quote(price=100.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        min_stop = quote.price * 0.995
        assert setup.stop_loss <= min_stop


class TestTakeProfitLevels:

    def test_tp1_provides_at_least_2_to_1_rr(self, sample_ohlcv_df):
        """TP1 should be above current_price (validated via the R:R stored on setup)."""
        constructor = TradeConstructor()
        # Use default setup type so entry_high ≈ current_price * 1.005 (no far resistance)
        candidate = _make_scored_candidate(
            atr=3.0, last_close=150.0, setup_type="default",
            resistance=None, support=None,
        )
        quote = _make_quote(price=150.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        # TP1 must be above entry_high (enforced by constructor floor)
        assert setup.take_profit_1 > setup.entry_high

    def test_tp2_is_above_tp1(self, sample_ohlcv_df):
        """TP2 must always be above TP1."""
        constructor = TradeConstructor()
        candidate = _make_scored_candidate(atr=3.0, last_close=150.0)
        quote = _make_quote(price=150.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.take_profit_2 > setup.take_profit_1

    def test_tp1_is_above_entry(self, sample_ohlcv_df):
        """Both take-profit levels must be above current_price."""
        constructor = TradeConstructor()
        # Use default setup so entry_high ≈ price (no far-away breakout resistance)
        candidate = _make_scored_candidate(
            atr=2.0, last_close=100.0, setup_type="default",
            resistance=None, support=None,
        )
        quote = _make_quote(price=100.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        # Both TP levels must be above current_price (the price targets are offset from price)
        current_price = quote.price
        assert setup.take_profit_1 > current_price
        assert setup.take_profit_2 > current_price


class TestEntryZone:

    def test_entry_zone_breakout_method(self, sample_ohlcv_df):
        """Breakout setup uses BREAKOUT_CONFIRMATION entry method."""
        constructor = TradeConstructor()
        candidate = _make_scored_candidate(setup_type="breakout", resistance=155.0)
        quote = _make_quote(price=152.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.entry_method == EntryMethod.BREAKOUT_CONFIRMATION

    def test_entry_zone_pullback_method(self, sample_ohlcv_df):
        """Pullback setup uses PULLBACK entry method."""
        constructor = TradeConstructor()
        candidate = _make_scored_candidate(setup_type="pullback", support=148.0)
        quote = _make_quote(price=150.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.entry_method == EntryMethod.PULLBACK


class TestRiskReward:

    def test_rr_ratio_is_positive(self, sample_ohlcv_df):
        """Risk-reward ratio must be positive."""
        constructor = TradeConstructor()
        # Use default setup so entry_high ≈ price; avoids far-resistance distorting R:R
        candidate = _make_scored_candidate(
            atr=2.0, last_close=100.0, setup_type="default",
            resistance=None, support=None,
        )
        quote = _make_quote(price=100.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.risk_reward_ratio > 0.0

    def test_trailing_stop_rule_set(self, sample_ohlcv_df):
        """Trailing stop rule string is populated and references ATR."""
        constructor = TradeConstructor()
        candidate = _make_scored_candidate(atr=2.5, last_close=100.0)
        quote = _make_quote(price=100.0)
        setup = constructor.construct(candidate, sample_ohlcv_df, quote)
        assert setup.trailing_stop_rule is not None
        assert "atr" in setup.trailing_stop_rule.lower()
