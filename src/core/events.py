"""
In-process async event bus for the swing-trader system.

Provides a lightweight publish/subscribe mechanism that allows decoupled
modules to communicate without direct imports.  All event handlers are
coroutines invoked sequentially within the same event loop.

Events
------
Each event is a dataclass inheriting from ``BaseEvent``.  Predefined
event types cover the main trading lifecycle:

- ``ScanCompletedEvent``
- ``TradeOpenedEvent``
- ``TradeClosedEvent``
- ``ExitTriggeredEvent``
- ``KillSwitchActivatedEvent``
- ``RiskLimitBreachedEvent``
- ``DataQualityDegradedEvent``
- ``OrderFilledEvent``
- ``AlertRaisedEvent``

Usage
-----
    from src.core.events import event_bus, ScanCompletedEvent

    # Subscribe
    @event_bus.on(ScanCompletedEvent)
    async def handle_scan(event: ScanCompletedEvent) -> None:
        print(event.cycle_id, event.candidates_found)

    # Publish
    await event_bus.publish(ScanCompletedEvent(cycle_id="c-001", candidates_found=12))
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Type, TypeVar

logger = logging.getLogger(__name__)

# Type alias for async handler callables
Handler = Callable[..., Coroutine[Any, Any, None]]
E = TypeVar("E", bound="BaseEvent")


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


@dataclass
class BaseEvent:
    """Base class for all application events.

    Attributes
    ----------
    occurred_at:
        UTC timestamp when the event was created (auto-set).
    """

    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        init=False,
    )


# ---------------------------------------------------------------------------
# Trading lifecycle events
# ---------------------------------------------------------------------------


@dataclass
class ScanCompletedEvent(BaseEvent):
    """Emitted after the candidate scan and selection pipeline completes."""

    cycle_id: str = ""
    candidates_found: int = 0
    selected_ticker: str | None = None
    is_no_trade: bool = False
    duration_seconds: float = 0.0


@dataclass
class TradeOpenedEvent(BaseEvent):
    """Emitted when a new position is opened (entry order filled)."""

    trade_id: str = ""
    cycle_id: str = ""
    ticker: str = ""
    entry_price: float = 0.0
    quantity: int = 0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0


@dataclass
class TradeClosedEvent(BaseEvent):
    """Emitted when a position is fully closed."""

    trade_id: str = ""
    ticker: str = ""
    exit_reason: str = ""
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    hold_days: int = 0


@dataclass
class ExitTriggeredEvent(BaseEvent):
    """Emitted when an exit condition is detected (before the order is sent)."""

    position_id: str = ""
    ticker: str = ""
    exit_reason: str = ""
    current_price: float = 0.0
    stop_price: float = 0.0


@dataclass
class KillSwitchActivatedEvent(BaseEvent):
    """Emitted when the emergency kill switch is triggered."""

    trigger: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class RiskLimitBreachedEvent(BaseEvent):
    """Emitted when any risk limit is approached or exceeded."""

    limit_name: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    ticker: str | None = None


@dataclass
class DataQualityDegradedEvent(BaseEvent):
    """Emitted when data quality drops below acceptable thresholds."""

    provider: str = ""
    data_type: str = ""
    quality: str = ""
    detail: str = ""
    ticker: str | None = None


@dataclass
class OrderFilledEvent(BaseEvent):
    """Emitted when a broker order is fully or partially filled."""

    order_id: str = ""
    broker_order_id: str = ""
    ticker: str = ""
    side: str = ""
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    is_partial: bool = False


@dataclass
class AlertRaisedEvent(BaseEvent):
    """Emitted when a system alert is generated."""

    level: str = "info"
    category: str = ""
    message: str = ""
    ticker: str | None = None


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class EventBus:
    """Lightweight in-process async event bus.

    Handlers are registered per event type and called in registration
    order.  Errors in individual handlers are logged but do not prevent
    subsequent handlers from running.
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: Type[E], handler: Handler) -> None:
        """Register *handler* to be called when an event of *event_type* is published.

        Parameters
        ----------
        event_type:
            The event class to listen for.
        handler:
            An async callable that accepts a single argument of *event_type*.
        """
        self._handlers[event_type].append(handler)
        logger.debug(
            "Event handler registered.",
            extra={"event": event_type.__name__, "handler": handler.__qualname__},
        )

    def on(self, event_type: Type[E]) -> Callable[[Handler], Handler]:
        """Decorator version of ``subscribe``.

        Example
        -------
            @event_bus.on(TradeOpenedEvent)
            async def handle_opened(event: TradeOpenedEvent) -> None: ...
        """

        def decorator(handler: Handler) -> Handler:
            self.subscribe(event_type, handler)
            return handler

        return decorator

    def unsubscribe(self, event_type: Type[E], handler: Handler) -> None:
        """Remove a previously registered handler.

        No-ops if the handler is not currently registered.
        """
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    async def publish(self, event: BaseEvent) -> None:
        """Publish *event* to all registered handlers.

        Handlers are awaited sequentially.  Exceptions are caught, logged,
        and suppressed so that a failing handler does not disrupt others.

        Parameters
        ----------
        event:
            The event instance to dispatch.
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug("Event published with no subscribers.", extra={"event": event_type.__name__})
            return

        for handler in handlers:
            try:
                await handler(event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error in event handler.",
                    extra={"event": event_type.__name__, "handler": handler.__qualname__},
                )

    async def publish_nowait(self, event: BaseEvent) -> None:
        """Schedule *event* publication as a fire-and-forget asyncio task.

        The event is dispatched in the background without blocking the
        caller.  Use when the caller does not need to wait for handlers.
        """
        asyncio.create_task(self.publish(event))  # noqa: RUF006

    def handler_count(self, event_type: Type[E]) -> int:
        """Return the number of handlers registered for *event_type*."""
        return len(self._handlers.get(event_type, []))

    def clear(self, event_type: Type[E] | None = None) -> None:
        """Remove all handlers for *event_type*, or all handlers if None."""
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

event_bus: EventBus = EventBus()
"""Global event bus instance shared across the application."""
