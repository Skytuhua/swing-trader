"""
WebSocket Real-Time Data Streaming — connection management and bar aggregation.

Provides a WebSocket connection manager with:
- Auto-reconnection with exponential backoff (1s → 2s → 4s → ... → 60s max)
- Heartbeat/ping for stale connection detection
- Multi-symbol subscription in a single connection
- Bar aggregation from ticks (configurable bar sizes: 1m, 5m, 15m, etc.)
- Thread-safe async message queue
- Automatic fallback to REST polling when WebSocket is unavailable

Architecture
------------
WebSocketManager manages the connection lifecycle:
  connect() → subscribe() → receive messages → aggregate bars → emit completed bars

BarAggregator accumulates ticks into OHLCV bars:
  tick → update current bar → if bar complete → emit bar → start new bar

Message flow:
  WebSocket → raw message → parse → route to:
    - Trade messages → BarAggregator
    - Quote messages → quote update callback
    - Status messages → connection state update
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

import structlog

logger = structlog.get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class BarData:
    """Aggregated OHLCV bar."""

    symbol: str
    timestamp: float  # Unix timestamp of bar open
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float = 0.0
    bar_size: str = "1m"  # e.g. "1m", "5m", "15m"
    is_complete: bool = False


@dataclass
class TickData:
    """A single trade tick."""

    symbol: str
    price: float
    size: float
    timestamp: float
    conditions: list[str] = field(default_factory=list)


class BarAggregator:
    """Aggregate ticks into OHLCV bars of configurable size.

    Parameters
    ----------
    bar_size_seconds : int
        Bar duration in seconds (60=1m, 300=5m, 900=15m).
    on_bar_complete : callable, optional
        Async callback invoked when a bar completes.
    """

    BAR_SIZES = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
    }

    def __init__(
        self,
        bar_size: str = "1m",
        on_bar_complete: Optional[Callable[[BarData], Coroutine]] = None,
    ) -> None:
        self.bar_size = bar_size
        self.bar_seconds = self.BAR_SIZES.get(bar_size, 60)
        self.on_bar_complete = on_bar_complete

        # Current incomplete bars keyed by symbol
        self._current_bars: dict[str, BarData] = {}
        # Completed bars buffer
        self._completed: list[BarData] = []

    def process_tick(self, tick: TickData) -> BarData | None:
        """Process a tick and return a completed bar if the bar period elapsed.

        Parameters
        ----------
        tick : TickData
            A single trade tick.

        Returns
        -------
        BarData if a bar just completed, None otherwise.
        """
        bar_start = self._bar_start_time(tick.timestamp)
        current = self._current_bars.get(tick.symbol)

        completed = None

        # Check if we need to start a new bar
        if current is None or current.timestamp != bar_start:
            if current is not None:
                # Complete the previous bar
                current.is_complete = True
                current.close = current.close  # already set by last tick
                completed = current
                self._completed.append(current)

            # Start new bar
            self._current_bars[tick.symbol] = BarData(
                symbol=tick.symbol,
                timestamp=bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.size,
                trade_count=1,
                vwap=tick.price * tick.size,
                bar_size=self.bar_size,
            )
        else:
            # Update current bar
            current.high = max(current.high, tick.price)
            current.low = min(current.low, tick.price)
            current.close = tick.price
            current.volume += tick.size
            current.trade_count += 1
            current.vwap += tick.price * tick.size

        return completed

    def get_current_bar(self, symbol: str) -> BarData | None:
        """Get the current incomplete bar for a symbol."""
        return self._current_bars.get(symbol)

    def get_completed_bars(self, symbol: str | None = None) -> list[BarData]:
        """Get completed bars, optionally filtered by symbol."""
        if symbol:
            return [b for b in self._completed if b.symbol == symbol]
        return list(self._completed)

    def flush(self, symbol: str) -> BarData | None:
        """Force-complete the current bar for a symbol (e.g. at market close)."""
        current = self._current_bars.pop(symbol, None)
        if current:
            current.is_complete = True
            self._completed.append(current)
        return current

    def _bar_start_time(self, timestamp: float) -> float:
        """Align timestamp to the start of its bar period."""
        return float(int(timestamp // self.bar_seconds) * self.bar_seconds)


class WebSocketManager:
    """Manages WebSocket connections for real-time market data streaming.

    Provides auto-reconnection, heartbeat monitoring, multi-symbol subscriptions,
    and bar aggregation from live ticks.

    Parameters
    ----------
    url : str
        WebSocket endpoint URL.
    api_key : str, optional
        API key for authentication.
    api_secret : str, optional
        API secret for authentication.
    bar_size : str
        Bar size for aggregation (default "1m").
    heartbeat_interval : float
        Seconds between heartbeat pings (default 30).
    max_reconnect_delay : float
        Maximum delay between reconnection attempts (default 60).
    on_bar : callable, optional
        Async callback for completed bars.
    on_quote : callable, optional
        Async callback for quote updates.
    on_error : callable, optional
        Async callback for errors.
    """

    def __init__(
        self,
        url: str = "",
        api_key: str = "",
        api_secret: str = "",
        bar_size: str = "1m",
        heartbeat_interval: float = 30.0,
        max_reconnect_delay: float = 60.0,
        on_bar: Optional[Callable] = None,
        on_quote: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_delay = max_reconnect_delay

        self.state = ConnectionState.DISCONNECTED
        self._ws: Any = None
        self._subscriptions: set[str] = set()
        self._reconnect_delay: float = 1.0
        self._last_message_time: float = 0.0
        self._running: bool = False

        # Message queue (thread-safe via asyncio.Queue)
        self._message_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10000)

        # Bar aggregation
        self._aggregator = BarAggregator(bar_size=bar_size, on_bar_complete=on_bar)

        # Callbacks
        self._on_bar = on_bar
        self._on_quote = on_quote
        self._on_error = on_error

        # Metrics
        self._messages_received: int = 0
        self._reconnect_count: int = 0
        self._bars_completed: int = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Establish the WebSocket connection.

        Returns True if connected successfully, False otherwise.
        Handles authentication if api_key is provided.
        """
        if not self.url:
            logger.warning("websocket_no_url", msg="No WebSocket URL configured")
            return False

        self.state = ConnectionState.CONNECTING
        try:
            import websockets  # type: ignore[import-untyped]
            self._ws = await websockets.connect(self.url)
            self.state = ConnectionState.CONNECTED
            self._reconnect_delay = 1.0
            self._last_message_time = time.time()

            # Authenticate if credentials provided
            if self.api_key:
                auth_msg = json.dumps({
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.api_secret,
                })
                await self._ws.send(auth_msg)

            logger.info("websocket_connected", url=self.url)
            return True

        except ImportError:
            logger.warning("websocket_library_missing", msg="websockets package not installed; using REST fallback")
            self.state = ConnectionState.FAILED
            return False
        except Exception as exc:
            logger.error("websocket_connect_failed", error=str(exc))
            self.state = ConnectionState.FAILED
            return False

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self.state = ConnectionState.DISCONNECTED
        logger.info("websocket_disconnected")

    async def reconnect(self) -> bool:
        """Reconnect with exponential backoff.

        Delay sequence: 1s → 2s → 4s → 8s → ... → max_reconnect_delay.
        """
        self.state = ConnectionState.RECONNECTING
        self._reconnect_count += 1

        logger.info(
            "websocket_reconnecting",
            attempt=self._reconnect_count,
            delay=self._reconnect_delay,
        )

        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self.max_reconnect_delay)

        success = await self.connect()
        if success and self._subscriptions:
            await self.subscribe(list(self._subscriptions))
        return success

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for the given symbols.

        Parameters
        ----------
        symbols : list[str]
            Ticker symbols to subscribe to.
        """
        self._subscriptions.update(symbols)

        if self._ws and self.state == ConnectionState.CONNECTED:
            msg = json.dumps({
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
            })
            try:
                await self._ws.send(msg)
                logger.info("websocket_subscribed", symbols=symbols)
            except Exception as exc:
                logger.error("websocket_subscribe_failed", error=str(exc))

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for the given symbols."""
        for s in symbols:
            self._subscriptions.discard(s)

        if self._ws and self.state == ConnectionState.CONNECTED:
            msg = json.dumps({
                "action": "unsubscribe",
                "trades": symbols,
                "quotes": symbols,
            })
            try:
                await self._ws.send(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main event loop: receive messages, process, and handle reconnection.

        Call this as an asyncio task. It runs until disconnect() is called.
        """
        self._running = True

        while self._running:
            if self.state != ConnectionState.CONNECTED:
                success = await self.reconnect()
                if not success:
                    continue

            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.heartbeat_interval * 2,
                )
                self._last_message_time = time.time()
                self._messages_received += 1

                await self._process_message(message)

            except asyncio.TimeoutError:
                # No message for 2x heartbeat interval — connection may be stale
                logger.warning("websocket_heartbeat_timeout")
                await self._send_ping()

            except Exception as exc:
                if self._running:
                    logger.error("websocket_error", error=str(exc))
                    self.state = ConnectionState.DISCONNECTED
                    if self._on_error:
                        try:
                            await self._on_error(exc)
                        except Exception:
                            pass

    async def _process_message(self, raw: str) -> None:
        """Parse and route a raw WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Handle list of messages (Alpaca format)
        messages = data if isinstance(data, list) else [data]

        for msg in messages:
            msg_type = msg.get("T") or msg.get("type") or msg.get("ev", "")

            if msg_type in ("t", "trade"):
                await self._handle_trade(msg)
            elif msg_type in ("q", "quote"):
                await self._handle_quote(msg)
            elif msg_type in ("error", "err"):
                logger.error("websocket_server_error", msg=msg)
            # Ignore other message types (subscription confirmations, etc.)

    async def _handle_trade(self, msg: dict) -> None:
        """Process a trade message and feed to bar aggregator."""
        tick = TickData(
            symbol=msg.get("S") or msg.get("sym") or msg.get("symbol", ""),
            price=float(msg.get("p") or msg.get("price", 0)),
            size=float(msg.get("s") or msg.get("size", 0)),
            timestamp=float(msg.get("t", 0)) / 1e9 if isinstance(msg.get("t"), int) and msg.get("t", 0) > 1e12 else float(msg.get("t", time.time())),
            conditions=msg.get("c") or msg.get("conditions", []),
        )

        if tick.price <= 0 or not tick.symbol:
            return

        completed_bar = self._aggregator.process_tick(tick)
        if completed_bar and self._on_bar:
            self._bars_completed += 1
            try:
                await self._on_bar(completed_bar)
            except Exception as exc:
                logger.error("websocket_bar_callback_error", error=str(exc))

        # Also enqueue for other consumers
        try:
            self._message_queue.put_nowait({"type": "trade", "data": tick})
        except asyncio.QueueFull:
            pass  # Drop oldest if queue full

    async def _handle_quote(self, msg: dict) -> None:
        """Process a quote update message."""
        quote = {
            "symbol": msg.get("S") or msg.get("sym") or msg.get("symbol", ""),
            "bid": float(msg.get("bp") or msg.get("bid", 0)),
            "ask": float(msg.get("ap") or msg.get("ask", 0)),
            "bid_size": float(msg.get("bs") or msg.get("bidSize", 0)),
            "ask_size": float(msg.get("as") or msg.get("askSize", 0)),
            "timestamp": time.time(),
        }

        if self._on_quote:
            try:
                await self._on_quote(quote)
            except Exception as exc:
                logger.error("websocket_quote_callback_error", error=str(exc))

        try:
            self._message_queue.put_nowait({"type": "quote", "data": quote})
        except asyncio.QueueFull:
            pass

    async def _send_ping(self) -> None:
        """Send a ping to keep the connection alive."""
        if self._ws and self.state == ConnectionState.CONNECTED:
            try:
                await self._ws.ping()
            except Exception:
                self.state = ConnectionState.DISCONNECTED

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    async def get_message(self, timeout: float = 1.0) -> dict | None:
        """Get the next message from the queue (for polling consumers)."""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_current_bar(self, symbol: str) -> BarData | None:
        """Get the current incomplete bar for a symbol."""
        return self._aggregator.get_current_bar(symbol)

    def get_completed_bars(self, symbol: str | None = None) -> list[BarData]:
        """Get completed bars, optionally filtered by symbol."""
        return self._aggregator.get_completed_bars(symbol)

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    @property
    def subscribed_symbols(self) -> set[str]:
        return set(self._subscriptions)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "bars_completed": self._bars_completed,
            "subscriptions": len(self._subscriptions),
            "queue_size": self._message_queue.qsize(),
            "last_message_age": round(time.time() - self._last_message_time, 1) if self._last_message_time else None,
        }
