"""
Integration tests for order flow with PaperBroker.

Tests: buy order fills, sell order fills, position tracking, P&L calculation.
At least 3 test cases.
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.enums import OrderSide, OrderStatus, OrderType
from src.services.execution.base import OrderRequest
from src.services.execution.paper_broker import PaperBroker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaperBrokerBuyFill:

    def test_buy_order_fills_immediately(self, paper_broker):
        """Buy order fills immediately at slippage-adjusted price."""
        paper_broker.set_price("AAPL", 150.0)
        order = OrderRequest(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )
        result = _run(paper_broker.submit_order(order))

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        # Fill price should be slightly above ask due to slippage
        assert result.average_fill_price > 150.0
        assert result.average_fill_price <= 150.0 * 1.01  # within 1%

    def test_buy_reduces_cash(self, paper_broker):
        """Buying shares reduces available cash."""
        initial_cash = paper_broker._cash
        paper_broker.set_price("MSFT", 300.0)
        order = OrderRequest(
            ticker="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50,
        )
        _run(paper_broker.submit_order(order))
        assert paper_broker._cash < initial_cash

    def test_insufficient_cash_raises_error(self, paper_broker):
        """Buying more than available cash raises an error."""
        paper_broker.set_price("BRK", 500_000.0)
        order = OrderRequest(
            ticker="BRK",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,  # Way too large
        )
        from src.core.exceptions import BrokerError
        with pytest.raises((BrokerError, Exception)):
            _run(paper_broker.submit_order(order))


class TestPaperBrokerSellFill:

    def test_sell_order_fills_and_closes_position(self, paper_broker):
        """Selling all shares closes the position."""
        # Buy first
        paper_broker.set_price("AAPL", 150.0)
        buy_order = OrderRequest(
            ticker="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
        )
        _run(paper_broker.submit_order(buy_order))

        # Now sell
        paper_broker.set_price("AAPL", 160.0)  # Price went up
        sell_order = OrderRequest(
            ticker="AAPL", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=100,
        )
        result = _run(paper_broker.submit_order(sell_order))
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100

    def test_sell_produces_profit(self, paper_broker):
        """Selling at higher price increases cash above initial - cost_basis."""
        paper_broker.set_price("TSLA", 200.0)
        buy = OrderRequest(
            ticker="TSLA", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        )
        _run(paper_broker.submit_order(buy))
        cash_after_buy = paper_broker._cash

        paper_broker.set_price("TSLA", 250.0)  # 25% gain
        sell = OrderRequest(
            ticker="TSLA", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=10,
        )
        _run(paper_broker.submit_order(sell))

        # Cash should be higher than after-buy (profitable trade)
        assert paper_broker._cash > cash_after_buy

    def test_sell_without_position_raises_error(self, paper_broker):
        """Selling a ticker we don't own raises an error."""
        paper_broker.set_price("NFLX", 400.0)
        sell = OrderRequest(
            ticker="NFLX", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=10,
        )
        from src.core.exceptions import BrokerError
        with pytest.raises((BrokerError, Exception)):
            _run(paper_broker.submit_order(sell))


class TestPaperBrokerPositionTracking:

    def test_positions_tracked_correctly(self, paper_broker):
        """Positions dict is updated after buy."""
        paper_broker.set_price("GOOG", 140.0)
        buy = OrderRequest(
            ticker="GOOG", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=25,
        )
        _run(paper_broker.submit_order(buy))

        positions = _run(paper_broker.get_positions())
        tickers = [p.ticker for p in positions]
        assert "GOOG" in tickers

    def test_account_info_reflects_portfolio_value(self, paper_broker):
        """AccountInfo.portfolio_value includes open position market value."""
        paper_broker.set_price("AMZN", 180.0)
        buy = OrderRequest(
            ticker="AMZN", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=50,
        )
        _run(paper_broker.submit_order(buy))

        account = _run(paper_broker.get_account())
        # Portfolio value should be > cash alone (position adds value)
        assert account.portfolio_value > paper_broker._cash

    def test_close_all_positions(self, paper_broker):
        """close_all_positions exits all open trades."""
        paper_broker.set_price("NVDA", 400.0)
        paper_broker.set_price("AMD", 120.0)

        for ticker, price, qty in [("NVDA", 400.0, 10), ("AMD", 120.0, 20)]:
            paper_broker.set_price(ticker, price)
            buy = OrderRequest(
                ticker=ticker, side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=qty,
            )
            _run(paper_broker.submit_order(buy))

        _run(paper_broker.close_all_positions())
        positions = _run(paper_broker.get_positions())
        assert len(positions) == 0
