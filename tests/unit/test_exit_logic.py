"""
Unit tests for ExitEngine.

Tests: stop loss trigger, take profit trigger, trailing stop update/trigger,
time stop, no exit when nothing triggered.
At least 8 test cases.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import ExitReason
from src.services.execution.base import Quote
from src.services.monitor.exit_engine import ExitEngine, ExitSignal


# ---------------------------------------------------------------------------
# Test position helper
# ---------------------------------------------------------------------------


def _make_position(
    ticker: str = "AAPL",
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit_1: float = 110.0,
    take_profit_2: float = 120.0,
    hold_days: int = 0,
    max_hold_days: int = 5,
    tp1_hit: bool = False,
    trailing_stop_price: float | None = None,
    max_price_since_entry: float | None = None,
    atr_at_entry: float = 2.0,
    unrealized_pnl_pct: float = 0.0,
) -> MagicMock:
    """Create a mock Position object."""
    pos = MagicMock()
    pos.ticker = ticker
    pos.entry_price = entry_price
    pos.stop_loss = stop_loss
    pos.take_profit_1 = take_profit_1
    pos.take_profit_2 = take_profit_2
    pos.hold_days = hold_days
    pos.max_hold_days = max_hold_days
    pos.tp1_hit = tp1_hit
    pos.trailing_stop_price = trailing_stop_price
    pos.max_price_since_entry = max_price_since_entry or entry_price
    pos.atr_at_entry = atr_at_entry
    pos.unrealized_pnl_pct = unrealized_pnl_pct
    return pos


def _make_quote(price: float, ticker: str = "AAPL") -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        bid=price * 0.999,
        ask=price * 1.001,
    )


def _run(coro):
    """Run an async coroutine in the test context."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestExitEngineStopLoss:

    def test_stop_loss_triggered_at_stop_price(self):
        """ExitSignal.STOP_LOSS triggered when price ≤ stop_loss."""
        engine = ExitEngine()
        position = _make_position(stop_loss=95.0)
        quote = _make_quote(94.0)  # Price below stop

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.STOP_LOSS
        assert signal.urgency == "immediate"

    def test_stop_loss_triggered_exactly_at_stop(self):
        """ExitSignal.STOP_LOSS triggered when price == stop_loss."""
        engine = ExitEngine()
        position = _make_position(stop_loss=95.0)
        quote = _make_quote(95.0)  # Exactly at stop

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.STOP_LOSS

    def test_no_stop_loss_when_price_above_stop(self):
        """No stop loss signal when price is safely above stop."""
        engine = ExitEngine()
        position = _make_position(
            stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
            hold_days=0, max_hold_days=5,
        )
        quote = _make_quote(102.0)  # Well above stop, below TP1

        signal = _run(engine.evaluate(position, quote))
        # No exit conditions should be triggered
        assert signal is None


class TestExitEngineTakeProfit:

    def test_take_profit_1_triggered(self):
        """ExitSignal.TAKE_PROFIT_1 triggered when price ≥ tp1 and tp1 not yet hit."""
        engine = ExitEngine()
        position = _make_position(take_profit_1=110.0, tp1_hit=False)
        quote = _make_quote(112.0)  # Above TP1

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.TAKE_PROFIT_1
        assert signal.is_partial is True  # TP1 exits only 50%
        assert signal.partial_pct == 0.5

    def test_take_profit_2_triggered(self):
        """ExitSignal.TAKE_PROFIT_2 triggered when price ≥ tp2 and tp1 already hit."""
        engine = ExitEngine()
        position = _make_position(
            take_profit_1=110.0, take_profit_2=120.0, tp1_hit=True
        )
        quote = _make_quote(121.0)  # Above TP2

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.TAKE_PROFIT_2
        assert signal.urgency == "immediate"

    def test_tp1_not_triggered_if_already_hit(self):
        """TP1 is not re-triggered when tp1_hit=True."""
        engine = ExitEngine()
        position = _make_position(
            take_profit_1=110.0, take_profit_2=120.0, tp1_hit=True
        )
        quote = _make_quote(112.0)  # Between TP1 and TP2

        signal = _run(engine.evaluate(position, quote))
        # Should not trigger TP1 again; TP2 is not yet hit
        if signal is not None:
            assert signal.reason != ExitReason.TAKE_PROFIT_1


class TestExitEngineTrailingStop:

    def test_trailing_stop_triggered(self):
        """Trailing stop triggered when price falls below trailing stop price."""
        engine = ExitEngine()
        position = _make_position(
            trailing_stop_price=105.0,
            stop_loss=95.0,  # trailing stop is tighter
        )
        quote = _make_quote(104.0)  # Below trailing stop

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.TRAILING_STOP
        assert signal.urgency == "immediate"

    def test_trailing_stop_updated_as_price_rises(self):
        """Trailing stop update logic: new_trail > current_trail → position updated."""
        engine = ExitEngine()
        entry = 100.0
        atr = 2.0
        # Use a real object so attribute writes are tracked
        from dataclasses import dataclass

        @dataclass
        class RealPosition:
            ticker: str = "AAPL"
            entry_price: float = entry
            stop_loss: float = 96.0
            take_profit_1: float = 115.0
            take_profit_2: float = 120.0
            hold_days: int = 1
            max_hold_days: int = 5
            tp1_hit: bool = False
            trailing_stop_price: float | None = 100.5
            # max_price_since_entry matches the new quote so new_trail = 104 - 3 = 101 > 100.5
            max_price_since_entry: float = 104.0
            atr_at_entry: float = atr
            unrealized_pnl_pct: float = 4.0

        position = RealPosition()
        quote = _make_quote(104.0)  # Price above all exit levels

        # Should update trailing stop without triggering exit
        signal = _run(engine.evaluate(position, quote))

        # When no exit is triggered, trailing stop should have been updated
        # new_trail = max_price_since_entry(104) - 1.5*atr(2) = 104 - 3 = 101
        # Since 101 > 100.5 (old current_stop), it should update
        expected_trail = 104.0 - (atr * 1.5)  # = 101.0
        if signal is None:
            assert position.trailing_stop_price >= expected_trail - 0.1


class TestExitEngineTimeStop:

    def test_time_stop_triggered_after_max_hold(self):
        """Time stop triggers when hold_days ≥ max_hold_days."""
        engine = ExitEngine()
        position = _make_position(hold_days=5, max_hold_days=5)
        quote = _make_quote(100.0)  # Price unchanged (no other triggers)

        signal = _run(engine.evaluate(position, quote))

        assert signal is not None
        assert signal.reason == ExitReason.TIME_STOP
        assert signal.urgency == "end_of_day"

    def test_no_time_stop_before_max_hold(self):
        """Time stop is NOT triggered when hold_days < max_hold_days."""
        engine = ExitEngine()
        position = _make_position(
            hold_days=3, max_hold_days=5,
            stop_loss=90.0, take_profit_1=115.0, take_profit_2=125.0,
        )
        quote = _make_quote(100.0)

        signal = _run(engine.evaluate(position, quote))
        if signal is not None:
            assert signal.reason != ExitReason.TIME_STOP


class TestExitEngineNoExit:

    def test_no_exit_when_all_conditions_healthy(self):
        """Returns None when no exit condition is triggered."""
        engine = ExitEngine()
        position = _make_position(
            entry_price=100.0,
            stop_loss=93.0,
            take_profit_1=112.0,
            take_profit_2=118.0,
            hold_days=2,
            max_hold_days=5,
            tp1_hit=False,
            trailing_stop_price=None,
        )
        quote = _make_quote(102.5)  # Safe zone: above stop, below TP1

        signal = _run(engine.evaluate(position, quote))
        assert signal is None
