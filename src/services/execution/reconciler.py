"""
Order reconciler: poll broker for order updates, synchronise local state,
handle partial fills, rejections, and cancellations.

Designed to run as a background task on a configurable poll interval.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from src.core.enums import OrderStatus

if TYPE_CHECKING:
    from src.services.execution.base import BrokerAdapter, BrokerOrder
    from src.services.execution.order_manager import Order, _OrderStore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Reconciliation event types
# ---------------------------------------------------------------------------

RECONCILE_FILLED = "filled"
RECONCILE_PARTIAL = "partial_fill"
RECONCILE_CANCELLED = "cancelled"
RECONCILE_REJECTED = "rejected"
RECONCILE_EXPIRED = "expired"

# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset(
    [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
)


class OrderReconciler:
    """Poll the broker for order updates and keep local state consistent.

    Features:
    - Configurable poll interval (default 15 s during market hours).
    - Processes partial fills: updates fill quantity and notifies callbacks.
    - Handles full fills, rejections, cancellations, and expirations.
    - Fires registered event callbacks on status transitions.
    - Can be started as a background loop or called manually per cycle.

    Usage::

        reconciler = OrderReconciler(broker, order_store, poll_interval=15)
        # Register a callback for fills
        reconciler.on_fill(handle_fill)
        # Run forever as background task
        asyncio.create_task(reconciler.run_forever())
        # Or call manually
        await reconciler.reconcile_once()
    """

    def __init__(
        self,
        broker: "BrokerAdapter",
        order_store: "_OrderStore",
        poll_interval: float = 15.0,
    ) -> None:
        self.broker = broker
        self.db = order_store
        self.poll_interval = poll_interval
        self._running = False

        # Event callbacks: event_type → list of async callables
        self._callbacks: dict[str, list[Callable[..., Coroutine[Any, Any, None]]]] = {
            RECONCILE_FILLED: [],
            RECONCILE_PARTIAL: [],
            RECONCILE_CANCELLED: [],
            RECONCILE_REJECTED: [],
            RECONCILE_EXPIRED: [],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Run reconciliation loop until stopped."""
        self._running = True
        logger.info("reconciler_started", poll_interval=self.poll_interval)
        while self._running:
            try:
                await self.reconcile_once()
            except Exception as exc:
                logger.error("reconciler_cycle_error", error=str(exc))
            await asyncio.sleep(self.poll_interval)
        logger.info("reconciler_stopped")

    def stop(self) -> None:
        """Signal the background loop to stop after the current cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------

    async def reconcile_once(self) -> dict[str, int]:
        """Perform one reconciliation pass.

        Returns:
            Dict with counts: {filled, partial, cancelled, rejected, expired, unchanged}.
        """
        open_orders = await self.db.get_open_orders()
        if not open_orders:
            return {
                "filled": 0, "partial": 0, "cancelled": 0,
                "rejected": 0, "expired": 0, "unchanged": 0,
            }

        stats: dict[str, int] = {
            "filled": 0, "partial": 0, "cancelled": 0,
            "rejected": 0, "expired": 0, "unchanged": 0,
        }

        # Fetch all open orders from broker in one call when possible
        try:
            broker_orders_list = await self.broker.get_open_orders()
            broker_map: dict[str, "BrokerOrder"] = {
                o.broker_order_id: o for o in broker_orders_list
            }
        except Exception as exc:
            logger.warning("reconciler_broker_fetch_failed", error=str(exc))
            broker_map = {}

        for order in open_orders:
            if not order.broker_order_id:
                continue

            try:
                # Try cached broker map first, then individual fetch
                broker_order = broker_map.get(order.broker_order_id)
                if broker_order is None:
                    # Not in open orders → likely terminal; fetch directly
                    broker_order = await self.broker.get_order_status(order.broker_order_id)

                change_type = await self._process_order(order, broker_order)
                if change_type:
                    stats[change_type] = stats.get(change_type, 0) + 1
                else:
                    stats["unchanged"] += 1

            except Exception as exc:
                logger.warning(
                    "reconciler_order_error",
                    ticker=order.ticker,
                    broker_id=order.broker_order_id,
                    error=str(exc),
                )

        logger.debug("reconciler_cycle_complete", **stats)
        return stats

    # ------------------------------------------------------------------
    # Per-order processing
    # ------------------------------------------------------------------

    async def _process_order(
        self,
        order: "Order",
        broker_order: "BrokerOrder",
    ) -> str | None:
        """Update local order if broker status differs.

        Returns the event type string if a change occurred, else None.
        """
        if broker_order.status == order.status:
            # Check for partial fill changes (quantity may have increased)
            if (
                broker_order.status == OrderStatus.PARTIAL
                and broker_order.filled_quantity != order.extra.get("filled_qty", 0)
            ):
                return await self._handle_partial(order, broker_order)
            return None  # Truly unchanged

        old_status = order.status

        # Update local record
        order.status = broker_order.status
        order.updated_at = datetime.now(tz=timezone.utc)
        if broker_order.average_fill_price:
            order.extra["fill_price"] = broker_order.average_fill_price
        if broker_order.filled_quantity:
            order.extra["filled_qty"] = broker_order.filled_quantity
        await self.db.save_order(order)

        logger.info(
            "reconciler_status_change",
            ticker=order.ticker,
            broker_id=order.broker_order_id,
            old=old_status,
            new=broker_order.status,
            fill_price=broker_order.average_fill_price,
        )

        # Dispatch to appropriate handler
        if broker_order.status == OrderStatus.FILLED:
            await self._fire(RECONCILE_FILLED, order, broker_order)
            return RECONCILE_FILLED

        if broker_order.status == OrderStatus.PARTIAL:
            await self._fire(RECONCILE_PARTIAL, order, broker_order)
            return RECONCILE_PARTIAL

        if broker_order.status == OrderStatus.CANCELLED:
            await self._fire(RECONCILE_CANCELLED, order, broker_order)
            return RECONCILE_CANCELLED

        if broker_order.status == OrderStatus.REJECTED:
            logger.error(
                "reconciler_order_rejected",
                ticker=order.ticker,
                broker_id=order.broker_order_id,
                is_entry=order.is_entry,
                is_stop=order.is_stop,
            )
            await self._fire(RECONCILE_REJECTED, order, broker_order)
            return RECONCILE_REJECTED

        if broker_order.status == OrderStatus.EXPIRED:
            await self._fire(RECONCILE_EXPIRED, order, broker_order)
            return RECONCILE_EXPIRED

        return None

    async def _handle_partial(
        self,
        order: "Order",
        broker_order: "BrokerOrder",
    ) -> str:
        """Handle an updated partial fill quantity."""
        old_filled = order.extra.get("filled_qty", 0)
        new_filled = broker_order.filled_quantity

        logger.info(
            "reconciler_partial_fill_update",
            ticker=order.ticker,
            broker_id=order.broker_order_id,
            old_filled=old_filled,
            new_filled=new_filled,
            fill_price=broker_order.average_fill_price,
        )
        order.extra["filled_qty"] = new_filled
        order.extra["fill_price"] = broker_order.average_fill_price
        order.updated_at = datetime.now(tz=timezone.utc)
        await self.db.save_order(order)
        await self._fire(RECONCILE_PARTIAL, order, broker_order)
        return RECONCILE_PARTIAL

    # ------------------------------------------------------------------
    # Callback registry
    # ------------------------------------------------------------------

    def on_fill(self, callback: Callable[..., Coroutine[Any, Any, None]]) -> None:
        """Register an async callback for full fills."""
        self._callbacks[RECONCILE_FILLED].append(callback)

    def on_partial(self, callback: Callable[..., Coroutine[Any, Any, None]]) -> None:
        self._callbacks[RECONCILE_PARTIAL].append(callback)

    def on_cancel(self, callback: Callable[..., Coroutine[Any, Any, None]]) -> None:
        self._callbacks[RECONCILE_CANCELLED].append(callback)

    def on_reject(self, callback: Callable[..., Coroutine[Any, Any, None]]) -> None:
        self._callbacks[RECONCILE_REJECTED].append(callback)

    def on_expire(self, callback: Callable[..., Coroutine[Any, Any, None]]) -> None:
        self._callbacks[RECONCILE_EXPIRED].append(callback)

    async def _fire(
        self,
        event_type: str,
        order: "Order",
        broker_order: "BrokerOrder",
    ) -> None:
        for cb in self._callbacks.get(event_type, []):
            try:
                await cb(order, broker_order)
            except Exception as exc:
                logger.error(
                    "reconciler_callback_error",
                    event=event_type,
                    ticker=order.ticker,
                    error=str(exc),
                )
