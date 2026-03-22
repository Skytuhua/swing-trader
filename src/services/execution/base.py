"""
Abstract broker adapter and shared execution data models.

All broker implementations (Alpaca live/paper, in-memory paper)
must subclass BrokerAdapter and implement every abstract method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.enums import OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Shared data models
# ---------------------------------------------------------------------------


@dataclass
class AccountInfo:
    """Snapshot of brokerage account state."""

    account_id: str
    equity: float               # Total account equity
    cash: float                 # Available cash
    buying_power: float         # Buying power (may be 2× for marginable accounts)
    portfolio_value: float      # Equity including unrealised P&L
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    currency: str = "USD"
    status: str = "ACTIVE"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    """Parameters for submitting a new order to the broker."""

    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int

    # Price fields (only used for relevant order types)
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None  # For trailing stop orders

    # Time in force: "day" | "gtc" | "ioc" | "fok"
    time_in_force: str = "day"

    # Extended-hours trading flag
    extended_hours: bool = False

    # Client-side idempotency key (forwarded to broker as client_order_id)
    idempotency_key: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerOrder:
    """A broker-side order record returned after submission or status query."""

    broker_order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    status: OrderStatus

    filled_quantity: int = 0
    average_fill_price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    time_in_force: str = "day"

    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    expired_at: datetime | None = None

    # Raw broker payload for debugging
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPosition:
    """An open position as reported by the broker."""

    ticker: str
    quantity: int                   # Positive = long
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    asset_class: str = "us_equity"
    side: str = "long"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    raw: dict[str, Any] = field(default_factory=dict)


# Quote dataclass (shared across broker and market data)
@dataclass
class Quote:
    """Real-time or delayed bid/ask quote for an instrument."""

    ticker: str
    price: float            # Last trade price
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> float:
        if self.bid and self.ask and self.price > 0:
            return ((self.ask - self.bid) / self.price) * 100.0
        return 0.0

    @property
    def mid(self) -> float:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.price


# ---------------------------------------------------------------------------
# Abstract broker adapter
# ---------------------------------------------------------------------------


class BrokerAdapter(ABC):
    """Abstract base class for all broker integrations.

    All methods are coroutines (``async def``) except ``is_market_open``
    which is expected to return a cached / synchronous result.
    """

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Return current account snapshot."""
        ...

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> BrokerOrder:
        """Submit a new order to the broker.

        Raises:
            BrokerError: If the broker rejects the order.
            BrokerConnectionError: If the broker is unreachable.
        """
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            True if the cancellation was accepted; False if already terminal.
        """
        ...

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        """Fetch the latest status of an order by its broker-assigned ID."""
        ...

    @abstractmethod
    async def get_open_orders(self) -> list[BrokerOrder]:
        """Return all open orders across all symbols."""
        ...

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Return all current open positions."""
        ...

    @abstractmethod
    async def close_position(self, ticker: str) -> BrokerOrder:
        """Close an entire open position at market.

        Raises:
            BrokerError: If no position exists or the close fails.
        """
        ...

    @abstractmethod
    async def close_all_positions(self) -> list[BrokerOrder]:
        """Close every open position at market.  Used by kill switch."""
        ...

    # ------------------------------------------------------------------
    # Market data (lightweight, broker-sourced)
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_quote(self, ticker: str) -> Quote:
        """Fetch the latest quote for a single ticker from the broker."""
        ...

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the primary exchange is currently open for trading.

        This must not be a coroutine – callers may call it synchronously
        in tight loops.  Implementations should cache the result and
        refresh it asynchronously in the background.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the broker connection is healthy."""
        ...
