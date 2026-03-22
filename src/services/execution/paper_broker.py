"""
In-memory paper broker for backtesting and development.

Simulates fills with configurable slippage.  Tracks positions, orders, and cash.
Implements the same BrokerAdapter interface as the live Alpaca broker.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.enums import OrderSide, OrderStatus, OrderType
from src.core.exceptions import BrokerError, OrderError
from src.services.execution.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    Quote,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_INITIAL_CASH = 100_000.0
_DEFAULT_SLIPPAGE_PCT = 0.001       # 0.1% per-side slippage
_DEFAULT_COMMISSION_PER_SHARE = 0.0  # Alpaca paper: $0 commission


class PaperBroker(BrokerAdapter):
    """In-memory simulated broker for paper trading / backtesting.

    - Fills are immediate (end of same bar) at mid ± slippage.
    - Positions and cash are tracked in memory.
    - Can be seeded with an initial cash balance.
    - Quote prices must be injected via ``set_price(ticker, price)``.
    """

    def __init__(
        self,
        initial_cash: float = _DEFAULT_INITIAL_CASH,
        slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
        commission_per_share: float = _DEFAULT_COMMISSION_PER_SHARE,
        simulate_partial_fills: bool = False,
    ) -> None:
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.simulate_partial_fills = simulate_partial_fills

        # State
        self._cash: float = initial_cash
        self._prices: dict[str, float] = {}
        self._orders: dict[str, BrokerOrder] = {}       # broker_order_id → order
        self._positions: dict[str, dict[str, Any]] = {} # ticker → {qty, avg_price, ...}
        self._market_open: bool = True

        # Cumulative PnL tracking
        self._realised_pnl: float = 0.0
        self._total_trades: int = 0

    # ------------------------------------------------------------------
    # Price injection (used by backtester / tests)
    # ------------------------------------------------------------------

    def set_price(self, ticker: str, price: float) -> None:
        """Set the simulated market price for a ticker."""
        self._prices[ticker] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        """Bulk-set market prices."""
        self._prices.update(prices)

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        portfolio_value = self._cash + self._portfolio_market_value()
        return AccountInfo(
            account_id="paper-0001",
            equity=portfolio_value,
            cash=self._cash,
            buying_power=self._cash * 1.0,  # No margin in paper mode
            portfolio_value=portfolio_value,
            currency="USD",
            status="ACTIVE",
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> BrokerOrder:
        """Simulate immediate fill with slippage."""
        if not self.is_market_open():
            raise OrderError("Market is closed; cannot submit paper order.")

        price = self._get_price(order.ticker)
        fill_price = self._apply_slippage(price, order.side)

        # Commission
        commission = self.commission_per_share * order.quantity

        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity + commission
            if cost > self._cash:
                raise OrderError(
                    f"Insufficient cash: need ${cost:.2f}, have ${self._cash:.2f}"
                )
            self._cash -= cost
            self._update_position(order.ticker, order.quantity, fill_price)
        else:  # SELL
            qty_to_sell = min(order.quantity, self._get_position_qty(order.ticker))
            if qty_to_sell <= 0:
                raise OrderError(f"No position in {order.ticker} to sell.")
            proceeds = fill_price * qty_to_sell - commission
            self._cash += proceeds
            self._close_position(order.ticker, qty_to_sell, fill_price)
            self._total_trades += 1

        now = datetime.now(tz=timezone.utc)
        broker_id = order.idempotency_key or str(uuid.uuid4())

        filled_qty = order.quantity
        if self.simulate_partial_fills and order.quantity > 100:
            # Simulate 50% partial fill for large orders
            filled_qty = order.quantity // 2

        broker_order = BrokerOrder(
            broker_order_id=broker_id,
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            filled_quantity=filled_qty,
            average_fill_price=fill_price,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            status=OrderStatus.FILLED if filled_qty == order.quantity else OrderStatus.PARTIAL,
            submitted_at=now,
            filled_at=now,
        )

        self._orders[broker_id] = broker_order
        logger.info(
            "paper_broker_fill",
            ticker=order.ticker,
            side=order.side.value,
            qty=filled_qty,
            price=round(fill_price, 4),
            cash_remaining=round(self._cash, 2),
        )
        return broker_order

    async def cancel_order(self, broker_order_id: str) -> bool:
        order = self._orders.get(broker_order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(tz=timezone.utc)
        logger.info("paper_broker_order_cancelled", broker_order_id=broker_order_id)
        return True

    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        order = self._orders.get(broker_order_id)
        if order is None:
            raise BrokerError(f"Order {broker_order_id} not found in paper broker.")
        return order

    async def get_open_orders(self) -> list[BrokerOrder]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING, OrderStatus.PARTIAL)
        ]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[BrokerPosition]:
        positions = []
        for ticker, pos in self._positions.items():
            if pos["qty"] <= 0:
                continue
            current_price = self._prices.get(ticker, pos["avg_price"])
            market_value = current_price * pos["qty"]
            cost_basis = pos["avg_price"] * pos["qty"]
            unrealised_pnl = market_value - cost_basis
            unrealised_pnl_pct = (unrealised_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
            positions.append(
                BrokerPosition(
                    ticker=ticker,
                    quantity=pos["qty"],
                    avg_entry_price=pos["avg_price"],
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealised_pnl,
                    unrealized_pnl_pct=unrealised_pnl_pct,
                    cost_basis=cost_basis,
                )
            )
        return positions

    async def close_position(self, ticker: str) -> BrokerOrder:
        qty = self._get_position_qty(ticker)
        if qty <= 0:
            raise BrokerError(f"No open position in {ticker}.")
        close_req = OrderRequest(
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=qty,
        )
        return await self.submit_order(close_req)

    async def close_all_positions(self) -> list[BrokerOrder]:
        orders = []
        tickers = [t for t, p in self._positions.items() if p["qty"] > 0]
        for ticker in tickers:
            try:
                order = await self.close_position(ticker)
                orders.append(order)
            except Exception as exc:
                logger.warning("paper_broker_close_all_error", ticker=ticker, error=str(exc))
        logger.warning("paper_broker_all_positions_closed", count=len(orders))
        return orders

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> Quote:
        price = self._prices.get(ticker)
        if price is None:
            raise BrokerError(f"No price set for {ticker} in paper broker.")
        spread = price * 0.001  # Simulated 0.1% spread
        return Quote(
            ticker=ticker,
            price=price,
            bid=price - spread / 2,
            ask=price + spread / 2,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        return self._market_open

    async def health_check(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict for reporting / testing."""
        portfolio_value = self._cash + self._portfolio_market_value()
        return {
            "cash": round(self._cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "realised_pnl": round(self._realised_pnl, 2),
            "total_return_pct": round(
                (portfolio_value - self.initial_cash) / self.initial_cash * 100.0, 3
            ),
            "total_trades": self._total_trades,
            "open_positions": len([p for p in self._positions.values() if p["qty"] > 0]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_price(self, ticker: str) -> float:
        price = self._prices.get(ticker)
        if price is None or price <= 0:
            raise OrderError(
                f"No valid price available for {ticker} in paper broker. "
                "Call set_price(ticker, price) before submitting orders."
            )
        return price

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply per-side slippage: buys pay more, sells receive less."""
        if side == OrderSide.BUY:
            return price * (1.0 + self.slippage_pct)
        return price * (1.0 - self.slippage_pct)

    def _get_position_qty(self, ticker: str) -> int:
        return self._positions.get(ticker, {}).get("qty", 0)

    def _update_position(self, ticker: str, qty: int, price: float) -> None:
        existing = self._positions.get(ticker)
        if existing and existing["qty"] > 0:
            # Update average price (weighted average)
            total_qty = existing["qty"] + qty
            avg = (existing["avg_price"] * existing["qty"] + price * qty) / total_qty
            existing["qty"] = total_qty
            existing["avg_price"] = avg
        else:
            self._positions[ticker] = {"qty": qty, "avg_price": price}

    def _close_position(self, ticker: str, qty: int, sell_price: float) -> None:
        pos = self._positions.get(ticker)
        if pos is None:
            return
        realised = (sell_price - pos["avg_price"]) * qty
        self._realised_pnl += realised
        pos["qty"] = max(0, pos["qty"] - qty)
        if pos["qty"] == 0:
            del self._positions[ticker]

    def _portfolio_market_value(self) -> float:
        total = 0.0
        for ticker, pos in self._positions.items():
            if pos["qty"] > 0:
                price = self._prices.get(ticker, pos["avg_price"])
                total += price * pos["qty"]
        return total
