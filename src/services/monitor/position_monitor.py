"""
Position monitor: real-time monitoring loop for open positions.

Calls monitor_cycle() every N seconds during market hours.
On each cycle:
  - Fetches latest quotes
  - Updates position state (price, PnL, max_price, hold_days)
  - Evaluates exit conditions via ExitEngine
  - Executes exits via the broker
  - Persists updates and fires alerts on errors
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import ExitReason, OrderSide, OrderStatus, OrderType
from src.core.exceptions import BrokerError
from src.services.execution.base import OrderRequest

if TYPE_CHECKING:
    from src.services.execution.base import BrokerAdapter, Quote
    from src.services.execution.order_manager import Position
    from src.services.market_data.manager import DataManager
    from src.services.monitor.alert_manager import AlertManager
    from src.services.monitor.exit_engine import ExitEngine, ExitSignal

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Thin position store interface
# ---------------------------------------------------------------------------


class _PositionStore:
    """In-memory position store – replace with real DB in production."""

    def __init__(self) -> None:
        self._positions: dict[str, "Position"] = {}

    async def get_open_positions(self) -> list["Position"]:
        return list(self._positions.values())

    async def update_position(self, position: "Position") -> None:
        self._positions[position.ticker] = position

    async def close_position(self, ticker: str) -> None:
        pos = self._positions.get(ticker)
        if pos:
            pos_status = getattr(pos, "status", None)
            if hasattr(pos, "status"):
                pos.status = "closed"
            else:
                del self._positions[ticker]

    def add_position(self, position: "Position") -> None:
        self._positions[position.ticker] = position


# ---------------------------------------------------------------------------
# Position monitor
# ---------------------------------------------------------------------------


class PositionMonitor:
    """Real-time position monitoring with exit triggers.

    Usage::

        monitor = PositionMonitor(
            broker=broker,
            data=data_manager,
            exit_engine=exit_engine,
            alert_manager=alert_manager,
            monitor_interval=30,
        )
        asyncio.create_task(monitor.run_forever())
    """

    def __init__(
        self,
        broker: "BrokerAdapter",
        data: "DataManager",
        exit_engine: "ExitEngine",
        alert_manager: "AlertManager",
        position_store: "_PositionStore | None" = None,
        monitor_interval: float = 30.0,
    ) -> None:
        self.broker = broker
        self.data = data
        self.exit_engine = exit_engine
        self.alert_manager = alert_manager
        self.db: _PositionStore = position_store or _PositionStore()
        self.monitor_interval = monitor_interval
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Monitor loop – runs until stopped.  Pauses during market close."""
        self._running = True
        logger.info("position_monitor_started", interval=self.monitor_interval)
        while self._running:
            try:
                if self.broker.is_market_open():
                    await self.monitor_cycle()
                else:
                    logger.debug("position_monitor_market_closed")
            except Exception as exc:
                logger.error("position_monitor_loop_error", error=str(exc))
            await asyncio.sleep(self.monitor_interval)
        logger.info("position_monitor_stopped")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def monitor_cycle(self) -> None:
        """Process one monitoring cycle across all open positions.

        Called every monitor_interval seconds during market hours.
        """
        positions = await self.db.get_open_positions()
        if not positions:
            logger.debug("position_monitor_no_positions")
            return

        logger.debug("position_monitor_cycle_start", positions=len(positions))
        now_utc = datetime.now(tz=timezone.utc)

        tasks = [self._monitor_position(pos, now_utc) for pos in positions]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug("position_monitor_cycle_end", positions=len(positions))

    async def _monitor_position(
        self,
        position: "Position",
        now: datetime,
    ) -> None:
        """Update and evaluate a single position."""
        ticker = position.ticker
        log = logger.bind(ticker=ticker)

        try:
            # ---- Fetch latest quote ----
            quote = await self.data.get_quote(ticker)

            # ---- Update position state ----
            self._update_position_state(position, quote, now)

            # ---- Evaluate exit conditions ----
            exit_signal = await self.exit_engine.evaluate(position, quote)

            if exit_signal:
                await self._execute_exit(position, exit_signal)

            # ---- Persist ----
            position.last_monitored_at = now  # type: ignore[attr-defined]
            await self.db.update_position(position)

        except Exception as exc:
            log.error("position_monitor_error", error=str(exc))
            await self.alert_manager.create_alert(
                alert_type="monitor_error",
                message=f"Error monitoring {ticker}: {exc}",
                ticker=ticker,
                severity="error",
            )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    @staticmethod
    def _update_position_state(
        position: "Position",
        quote: "Quote",
        now: datetime,
    ) -> None:
        """Update price, PnL, max_price, and hold_days on the position object."""
        price = quote.price
        position.current_price = price  # type: ignore[attr-defined]
        position.unrealized_pnl = (price - position.entry_price) * position.quantity  # type: ignore[attr-defined]
        position.unrealized_pnl_pct = (price / position.entry_price - 1.0) * 100.0  # type: ignore[attr-defined]

        # Ratchet maximum price
        old_max = getattr(position, "max_price_since_entry", position.entry_price)
        position.max_price_since_entry = max(old_max, price)  # type: ignore[attr-defined]

        # Hold days (calendar days since entry)
        entry_date: date | None = getattr(position, "entry_date", None)
        if entry_date:
            position.hold_days = (now.date() - entry_date).days  # type: ignore[attr-defined]
        else:
            position.hold_days = getattr(position, "hold_days", 0)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    async def _execute_exit(
        self,
        position: "Position",
        exit_signal: "ExitSignal",
    ) -> None:
        """Place the appropriate exit order based on the exit signal."""
        ticker = position.ticker
        reason = exit_signal.reason
        urgency = exit_signal.urgency
        log = logger.bind(
            ticker=ticker,
            reason=reason,
            urgency=urgency,
        )

        # Determine quantity to sell
        total_qty = position.quantity
        if exit_signal.is_partial and exit_signal.partial_pct:
            qty = max(1, int(total_qty * exit_signal.partial_pct))
            log.info("exit_partial", qty=qty, pct=exit_signal.partial_pct)
        else:
            qty = total_qty

        # Choose order type based on urgency
        if urgency == "immediate" or reason == ExitReason.STOP_LOSS:
            order_type = OrderType.MARKET
            limit_price = None
        else:
            # next_bar or end_of_day: use limit at current price (fills quickly)
            order_type = OrderType.LIMIT
            limit_price = getattr(position, "current_price", None) or exit_signal.price_at_signal

        order_req = OrderRequest(
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=qty,
            limit_price=limit_price,
            time_in_force="day",
            idempotency_key=f"exit_{ticker}_{reason.value}_{int(datetime.now(tz=timezone.utc).timestamp())}",
        )

        try:
            broker_order = await self.broker.submit_order(order_req)
            log.info(
                "exit_order_submitted",
                broker_id=broker_order.broker_order_id,
                qty=qty,
                order_type=order_type,
            )

            # Mark TP1 as hit to prevent re-triggering
            if reason == ExitReason.TAKE_PROFIT_1:
                position.tp1_hit = True

            # If full exit, close position in local store
            if not exit_signal.is_partial:
                await self.db.close_position(ticker)

            # Fire alert
            await self.alert_manager.create_alert(
                alert_type="position_exit",
                message=(
                    f"{ticker} exit triggered: {reason.value}. "
                    f"Sold {qty} shares @ {exit_signal.price_at_signal or 'mkt'}. "
                    f"Detail: {exit_signal.detail}"
                ),
                ticker=ticker,
                severity="info" if reason in (
                    ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2
                ) else "warning",
            )

        except BrokerError as exc:
            log.error("exit_order_failed", error=str(exc))
            await self.alert_manager.create_alert(
                alert_type="exit_order_failed",
                message=f"CRITICAL: Failed to exit {ticker} on {reason.value}: {exc}",
                ticker=ticker,
                severity="critical",
            )
