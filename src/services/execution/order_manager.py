"""
Order manager: idempotent order placement, duplicate prevention,
entry orders, protective stops, take-profit orders, and reconciliation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import OrderSide, OrderStatus, OrderType
from src.core.exceptions import BrokerError, OrderError
from src.services.execution.base import OrderRequest

if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.services.execution.base import BrokerAdapter, BrokerOrder
    from src.services.trade.constructor import TradeSetup
    from src.services.trade.sizer import PositionSize

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight Order / TradingDecision DTOs (database-backed in production)
# ---------------------------------------------------------------------------


class Order:
    """In-process order record mirroring the database model."""

    def __init__(
        self,
        decision_id: str,
        ticker: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        limit_price: float | None,
        idempotency_key: str,
        is_entry: bool = False,
        is_stop: bool = False,
        is_take_profit: bool = False,
        stop_price: float | None = None,
    ) -> None:
        self.id: str = ""
        self.decision_id = decision_id
        self.ticker = ticker
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.idempotency_key = idempotency_key
        self.is_entry = is_entry
        self.is_stop = is_stop
        self.is_take_profit = is_take_profit
        self.broker_order_id: str | None = None
        self.status: OrderStatus = OrderStatus.PENDING
        self.created_at: datetime = datetime.now(tz=timezone.utc)
        self.updated_at: datetime = datetime.now(tz=timezone.utc)
        self.extra: dict[str, Any] = {}

    def to_broker_request(self) -> OrderRequest:
        """Convert to a broker-agnostic OrderRequest."""
        return OrderRequest(
            ticker=self.ticker,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            idempotency_key=self.idempotency_key,
        )


class TradingDecision:
    """Minimal DTO representing a trading decision from the pipeline."""

    def __init__(
        self,
        id: str,
        cycle_id: str,
        selected_ticker: str,
        risk_pct: float = 0.0,
        allocation_pct: float = 0.0,
    ) -> None:
        self.id = id
        self.cycle_id = cycle_id
        self.selected_ticker = selected_ticker
        self.risk_pct = risk_pct
        self.allocation_pct = allocation_pct


class Position:
    """In-process position record."""

    def __init__(
        self,
        ticker: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        atr_at_entry: float,
    ) -> None:
        self.ticker = ticker
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.atr_at_entry = atr_at_entry
        self.tp1_hit: bool = False
        self.current_price: float = entry_price
        self.trailing_stop_price: float | None = None


# ---------------------------------------------------------------------------
# Thin DB interface (real implementation uses SQLAlchemy session)
# ---------------------------------------------------------------------------


class _OrderStore:
    """In-memory order store – replace with real DB implementation."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}          # idempotency_key → Order
        self._by_id: dict[str, Order] = {}           # broker_order_id → Order

    async def get_order_by_idempotency_key(self, key: str) -> Order | None:
        return self._orders.get(key)

    async def save_order(self, order: Order) -> None:
        self._orders[order.idempotency_key] = order
        if order.broker_order_id:
            self._by_id[order.broker_order_id] = order

    async def get_open_orders(self) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
        ]

    async def update_order_status(self, broker_order_id: str, status: OrderStatus) -> None:
        order = self._by_id.get(broker_order_id)
        if order:
            order.status = status
            order.updated_at = datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Order Manager
# ---------------------------------------------------------------------------


class OrderManager:
    """Manages order lifecycle with idempotency and broker reconciliation.

    Key guarantees:
    - No duplicate orders: idempotency key checked before every submission.
    - Protective stops placed immediately after entry fill.
    - TP orders placed alongside protective stops.
    - Full reconciliation against broker state on demand.
    """

    def __init__(
        self,
        broker: "BrokerAdapter",
        db: "_OrderStore | None" = None,
    ) -> None:
        self.broker = broker
        self.db: _OrderStore = db or _OrderStore()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def place_entry_order(
        self,
        decision: TradingDecision,
        setup: "TradeSetup",
        size: "PositionSize",
    ) -> Order:
        """Place entry order with idempotency protection.

        Args:
            decision: The trading decision (contains cycle_id, ticker).
            setup:    TradeSetup with entry_high and entry_method.
            size:     PositionSize with share count.

        Returns:
            The persisted Order (may be a pre-existing duplicate).
        """
        idem_key = f"entry_{decision.cycle_id}_{decision.selected_ticker}"

        # Idempotency check
        existing = await self.db.get_order_by_idempotency_key(idem_key)
        if existing:
            logger.warning(
                "order_manager_duplicate_prevented",
                key=idem_key,
                existing_status=existing.status,
            )
            return existing

        # Determine order type from entry method
        from src.core.enums import EntryMethod
        if setup.entry_method == EntryMethod.MARKET:
            order_type = OrderType.MARKET
            limit_price = None
        else:
            order_type = OrderType.LIMIT
            limit_price = setup.entry_high

        order = Order(
            decision_id=decision.id,
            ticker=decision.selected_ticker,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=size.shares,
            limit_price=limit_price,
            idempotency_key=idem_key,
            is_entry=True,
        )

        try:
            broker_order = await self.broker.submit_order(order.to_broker_request())
            order.broker_order_id = broker_order.broker_order_id
            order.status = broker_order.status
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            await self.db.save_order(order)
            raise OrderError(
                f"Entry order submission failed for {decision.selected_ticker}: {exc}"
            ) from exc

        await self.db.save_order(order)
        logger.info(
            "order_manager_entry_placed",
            ticker=decision.selected_ticker,
            qty=size.shares,
            broker_id=order.broker_order_id,
            order_type=order_type,
        )
        return order

    # ------------------------------------------------------------------
    # Protective stop
    # ------------------------------------------------------------------

    async def place_protective_stop(self, position: Position) -> Order:
        """Place a stop-loss order for an open position.

        Uses a STOP order (market stop). If already placed, returns the
        existing order without creating a duplicate.
        """
        idem_key = f"stop_{position.ticker}_e{position.entry_price:.4f}"

        existing = await self.db.get_order_by_idempotency_key(idem_key)
        if existing:
            logger.debug(
                "order_manager_stop_already_exists",
                ticker=position.ticker,
                key=idem_key,
            )
            return existing

        order = Order(
            decision_id="",
            ticker=position.ticker,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=position.quantity,
            limit_price=None,
            stop_price=position.stop_loss,
            idempotency_key=idem_key,
            is_stop=True,
        )

        broker_order = await self.broker.submit_order(order.to_broker_request())
        order.broker_order_id = broker_order.broker_order_id
        order.status = broker_order.status
        await self.db.save_order(order)

        logger.info(
            "order_manager_stop_placed",
            ticker=position.ticker,
            stop_price=position.stop_loss,
            broker_id=order.broker_order_id,
        )
        return order

    # ------------------------------------------------------------------
    # Take-profit orders
    # ------------------------------------------------------------------

    async def place_take_profit_order(
        self,
        position: Position,
        target_price: float,
        quantity: int,
        label: str = "tp1",
    ) -> Order:
        """Place a limit sell (take-profit) order.

        Args:
            position:     Open position.
            target_price: Limit price for the take-profit.
            quantity:     Shares to sell at this target.
            label:        "tp1" or "tp2" for idempotency key namespacing.
        """
        idem_key = f"{label}_{position.ticker}_e{position.entry_price:.4f}"

        existing = await self.db.get_order_by_idempotency_key(idem_key)
        if existing:
            return existing

        order = Order(
            decision_id="",
            ticker=position.ticker,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=target_price,
            idempotency_key=idem_key,
            is_take_profit=True,
        )

        broker_order = await self.broker.submit_order(order.to_broker_request())
        order.broker_order_id = broker_order.broker_order_id
        order.status = broker_order.status
        await self.db.save_order(order)

        logger.info(
            "order_manager_tp_placed",
            ticker=position.ticker,
            label=label,
            target_price=target_price,
            qty=quantity,
            broker_id=order.broker_order_id,
        )
        return order

    # ------------------------------------------------------------------
    # Cancel helpers
    # ------------------------------------------------------------------

    async def cancel_stop_for_position(self, position: Position) -> bool:
        """Cancel the protective stop for a position (e.g. before manual close)."""
        idem_key = f"stop_{position.ticker}_e{position.entry_price:.4f}"
        existing = await self.db.get_order_by_idempotency_key(idem_key)
        if not existing or not existing.broker_order_id:
            return False
        cancelled = await self.broker.cancel_order(existing.broker_order_id)
        if cancelled:
            existing.status = OrderStatus.CANCELLED
            await self.db.save_order(existing)
        return cancelled

    async def cancel_all_open_orders(self) -> int:
        """Cancel all locally-tracked open orders. Returns count cancelled."""
        open_orders = await self.db.get_open_orders()
        count = 0
        for order in open_orders:
            if order.broker_order_id:
                try:
                    ok = await self.broker.cancel_order(order.broker_order_id)
                    if ok:
                        order.status = OrderStatus.CANCELLED
                        await self.db.save_order(order)
                        count += 1
                except Exception as exc:
                    logger.warning(
                        "order_manager_cancel_failed",
                        ticker=order.ticker,
                        broker_id=order.broker_order_id,
                        error=str(exc),
                    )
        logger.info("order_manager_cancelled_all", count=count)
        return count

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self) -> None:
        """Reconcile local order state with the broker's current state.

        For each open order in our local store, fetch the latest status
        from the broker and update if it has changed.
        """
        open_orders = await self.db.get_open_orders()
        logger.debug("order_manager_reconcile_start", open_count=len(open_orders))

        for order in open_orders:
            if not order.broker_order_id:
                continue
            try:
                broker_status = await self.broker.get_order_status(order.broker_order_id)
                if broker_status.status != order.status:
                    logger.info(
                        "order_manager_reconcile_status_change",
                        ticker=order.ticker,
                        broker_id=order.broker_order_id,
                        old_status=order.status,
                        new_status=broker_status.status,
                    )
                    await self._update_order_state(order, broker_status)
            except Exception as exc:
                logger.warning(
                    "order_manager_reconcile_error",
                    broker_id=order.broker_order_id,
                    error=str(exc),
                )

    async def _update_order_state(
        self,
        order: Order,
        broker_order: "BrokerOrder",
    ) -> None:
        order.status = broker_order.status
        order.updated_at = datetime.now(tz=timezone.utc)
        if broker_order.average_fill_price:
            order.extra["fill_price"] = broker_order.average_fill_price
        await self.db.save_order(order)
