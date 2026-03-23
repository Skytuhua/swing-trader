"""Tests for WebSocket Real-Time Data Streaming (src/data/websocket_manager.py)."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.data.websocket_manager import (
    BarAggregator,
    BarData,
    ConnectionState,
    TickData,
    WebSocketManager,
)


# ---------------------------------------------------------------------------
# BarAggregator tests
# ---------------------------------------------------------------------------


class TestBarAggregator:
    def test_single_tick_creates_bar(self):
        agg = BarAggregator(bar_size="1m")
        tick = TickData(symbol="AAPL", price=150.0, size=100, timestamp=1000000.0)
        completed = agg.process_tick(tick)
        # First tick starts a new bar, no completed bar yet
        assert completed is None
        current = agg.get_current_bar("AAPL")
        assert current is not None
        assert current.open == 150.0
        assert current.high == 150.0
        assert current.low == 150.0
        assert current.close == 150.0
        assert current.volume == 100

    def test_multiple_ticks_same_bar(self):
        agg = BarAggregator(bar_size="1m")
        base_time = 1000020.0  # Within same minute
        ticks = [
            TickData("AAPL", 150.0, 100, base_time),
            TickData("AAPL", 151.0, 200, base_time + 10),
            TickData("AAPL", 149.0, 150, base_time + 20),
            TickData("AAPL", 150.5, 300, base_time + 30),
        ]
        for tick in ticks:
            agg.process_tick(tick)

        bar = agg.get_current_bar("AAPL")
        assert bar is not None
        assert bar.open == 150.0
        assert bar.high == 151.0
        assert bar.low == 149.0
        assert bar.close == 150.5
        assert bar.volume == 750
        assert bar.trade_count == 4

    def test_new_bar_completes_previous(self):
        agg = BarAggregator(bar_size="1m")
        # First bar (minute 0)
        agg.process_tick(TickData("AAPL", 150.0, 100, 60.0))
        agg.process_tick(TickData("AAPL", 151.0, 200, 90.0))

        # New bar (minute 1) — should complete the first bar
        completed = agg.process_tick(TickData("AAPL", 152.0, 300, 120.0))
        assert completed is not None
        assert completed.is_complete is True
        assert completed.symbol == "AAPL"
        assert completed.open == 150.0
        assert completed.high == 151.0

    def test_multiple_symbols(self):
        agg = BarAggregator(bar_size="1m")
        agg.process_tick(TickData("AAPL", 150.0, 100, 60.0))
        agg.process_tick(TickData("MSFT", 300.0, 50, 60.0))

        assert agg.get_current_bar("AAPL") is not None
        assert agg.get_current_bar("MSFT") is not None
        assert agg.get_current_bar("AAPL").open == 150.0
        assert agg.get_current_bar("MSFT").open == 300.0

    def test_flush_bar(self):
        agg = BarAggregator(bar_size="1m")
        agg.process_tick(TickData("AAPL", 150.0, 100, 60.0))
        flushed = agg.flush("AAPL")
        assert flushed is not None
        assert flushed.is_complete is True
        assert agg.get_current_bar("AAPL") is None

    def test_completed_bars_list(self):
        agg = BarAggregator(bar_size="1m")
        agg.process_tick(TickData("AAPL", 150.0, 100, 60.0))
        agg.process_tick(TickData("AAPL", 151.0, 200, 120.0))  # Completes first bar
        agg.process_tick(TickData("AAPL", 152.0, 300, 180.0))  # Completes second bar

        bars = agg.get_completed_bars("AAPL")
        assert len(bars) == 2

    def test_5_minute_bars(self):
        agg = BarAggregator(bar_size="5m")
        # First 5-min bar
        agg.process_tick(TickData("AAPL", 150.0, 100, 300.0))  # Second 0 of bar
        agg.process_tick(TickData("AAPL", 151.0, 200, 450.0))  # 2.5 min in

        # Next 5-min bar (completes first)
        completed = agg.process_tick(TickData("AAPL", 152.0, 300, 600.0))
        assert completed is not None

    def test_bar_alignment(self):
        agg = BarAggregator(bar_size="1m")
        # Tick at timestamp 1000065 should align to bar starting at 1000020
        tick = TickData("AAPL", 150.0, 100, 1000065.0)
        agg.process_tick(tick)
        bar = agg.get_current_bar("AAPL")
        # Bar start should be aligned to 60-second boundary
        assert bar.timestamp == 1000020.0


# ---------------------------------------------------------------------------
# WebSocketManager tests
# ---------------------------------------------------------------------------


class TestWebSocketManager:
    def test_initial_state(self):
        mgr = WebSocketManager()
        assert mgr.state == ConnectionState.DISCONNECTED
        assert mgr.is_connected is False
        assert len(mgr.subscribed_symbols) == 0

    def test_stats(self):
        mgr = WebSocketManager()
        stats = mgr.stats
        assert stats["state"] == "disconnected"
        assert stats["messages_received"] == 0

    def test_subscribe_tracking(self):
        mgr = WebSocketManager()
        # Can't actually connect without a server, but subscriptions should be tracked
        mgr._subscriptions.add("AAPL")
        mgr._subscriptions.add("MSFT")
        assert "AAPL" in mgr.subscribed_symbols
        assert "MSFT" in mgr.subscribed_symbols

    @pytest.mark.asyncio
    async def test_connect_no_url(self):
        mgr = WebSocketManager(url="")
        success = await mgr.connect()
        assert success is False
        assert mgr.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect(self):
        mgr = WebSocketManager()
        await mgr.disconnect()
        assert mgr.state == ConnectionState.DISCONNECTED
        assert mgr._running is False

    @pytest.mark.asyncio
    async def test_get_message_timeout(self):
        mgr = WebSocketManager()
        msg = await mgr.get_message(timeout=0.01)
        assert msg is None

    def test_bar_aggregation_via_manager(self):
        mgr = WebSocketManager(bar_size="1m")
        current = mgr.get_current_bar("AAPL")
        assert current is None  # No data yet

    def test_completed_bars_via_manager(self):
        mgr = WebSocketManager()
        bars = mgr.get_completed_bars()
        assert bars == []


# ---------------------------------------------------------------------------
# TickData tests
# ---------------------------------------------------------------------------


class TestTickData:
    def test_tick_creation(self):
        tick = TickData(symbol="AAPL", price=150.0, size=100, timestamp=time.time())
        assert tick.symbol == "AAPL"
        assert tick.price == 150.0
        assert tick.size == 100

    def test_tick_conditions(self):
        tick = TickData(
            symbol="AAPL", price=150.0, size=100,
            timestamp=time.time(), conditions=["@", "T"],
        )
        assert len(tick.conditions) == 2


# ---------------------------------------------------------------------------
# BarData tests
# ---------------------------------------------------------------------------


class TestBarData:
    def test_bar_creation(self):
        bar = BarData(
            symbol="AAPL", timestamp=1000000.0,
            open=150.0, high=152.0, low=149.0, close=151.0,
            volume=10000, trade_count=50, bar_size="1m",
        )
        assert bar.symbol == "AAPL"
        assert bar.high == 152.0
        assert bar.is_complete is False

    def test_bar_complete_flag(self):
        bar = BarData(
            symbol="AAPL", timestamp=1000000.0,
            open=150.0, high=152.0, low=149.0, close=151.0,
            volume=10000, is_complete=True,
        )
        assert bar.is_complete is True
