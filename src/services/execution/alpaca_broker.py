"""
Alpaca broker adapter using alpaca-py TradingClient.

Supports paper and live trading via configurable base URL.
All order types: market, limit, stop, stop_limit, trailing_stop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.enums import OrderSide, OrderStatus, OrderType
from src.core.exceptions import BrokerConnectionError, BrokerError, OrderError
from src.services.execution.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    Quote,
)

if TYPE_CHECKING:
    from src.core.config import BrokerConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Alpaca SDK imports (optional dependency)
# ---------------------------------------------------------------------------

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import (
        AssetClass,
        OrderSide as AlpacaSide,
        OrderType as AlpacaOrderType,
        TimeInForce,
        OrderStatus as AlpacaOrderStatus,
        PositionSide,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
        TrailingStopOrderRequest,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning("alpaca_sdk_not_installed", hint="pip install alpaca-py")


# ---------------------------------------------------------------------------
# Status + type mappers
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.EXPIRED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "held": OrderStatus.SUBMITTED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
}

_SIDE_MAP: dict[str, OrderSide] = {
    "buy": OrderSide.BUY,
    "sell": OrderSide.SELL,
}

_TYPE_MAP: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "stop": OrderType.STOP,
    "stop_limit": OrderType.STOP_LIMIT,
    "trailing_stop": OrderType.TRAILING_STOP,
}


def _to_alpaca_side(side: OrderSide):  # type: ignore[return]
    if not _ALPACA_AVAILABLE:
        return None
    return AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL


def _to_alpaca_tif(tif: str):  # type: ignore[return]
    if not _ALPACA_AVAILABLE:
        return None
    mapping = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    return mapping.get(tif, TimeInForce.DAY)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AlpacaBroker(BrokerAdapter):
    """Alpaca broker adapter (paper and live).

    Usage::

        broker = AlpacaBroker(config)
        await broker.initialize()
        account = await broker.get_account()
    """

    def __init__(self, config: "BrokerConfig") -> None:
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )
        self.config = config
        self._trading_client: TradingClient | None = None
        self._data_client: StockHistoricalDataClient | None = None
        self._market_open_cache: bool = False
        self._cache_updated_at: datetime | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the Alpaca clients.  Must be called before first use."""
        paper = getattr(self.config, "paper", True)
        self._trading_client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
            paper=paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
        )
        logger.info("alpaca_broker_initialized", paper=paper)
        # Prime the market-open cache
        await self._refresh_market_open()

    def _client(self) -> TradingClient:
        if self._trading_client is None:
            raise BrokerConnectionError(
                "AlpacaBroker not initialised. Call await broker.initialize() first."
            )
        return self._trading_client

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_account(self) -> AccountInfo:
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, self._client().get_account
            )
            return AccountInfo(
                account_id=str(raw.id),
                equity=float(raw.equity),
                cash=float(raw.cash),
                buying_power=float(raw.buying_power),
                portfolio_value=float(raw.portfolio_value),
                day_trade_count=int(raw.daytrade_count or 0),
                pattern_day_trader=bool(raw.pattern_day_trader),
                currency=str(raw.currency or "USD"),
                status=str(raw.status),
            )
        except Exception as exc:
            raise BrokerConnectionError(f"get_account failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> BrokerOrder:
        """Submit an order and return the broker order record."""
        try:
            req = self._build_order_request(order)
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().submit_order(req)
            )
            broker_order = self._parse_order(raw)
            logger.info(
                "alpaca_order_submitted",
                ticker=order.ticker,
                side=order.side,
                type=order.order_type,
                qty=order.quantity,
                broker_id=broker_order.broker_order_id,
            )
            return broker_order
        except Exception as exc:
            raise OrderError(f"submit_order failed for {order.ticker}: {exc}") from exc

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().cancel_order_by_id(broker_order_id)
            )
            logger.info("alpaca_order_cancelled", broker_order_id=broker_order_id)
            return True
        except Exception as exc:
            logger.warning(
                "alpaca_cancel_failed",
                broker_order_id=broker_order_id,
                error=str(exc),
            )
            return False

    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().get_order_by_id(broker_order_id)
            )
            return self._parse_order(raw)
        except Exception as exc:
            raise BrokerError(f"get_order_status failed for {broker_order_id}: {exc}") from exc

    async def get_open_orders(self) -> list[BrokerOrder]:
        try:
            req = GetOrdersRequest(status="open")  # type: ignore[call-arg]
            raws = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().get_orders(req)
            )
            return [self._parse_order(r) for r in raws]
        except Exception as exc:
            raise BrokerError(f"get_open_orders failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[BrokerPosition]:
        try:
            raws = await asyncio.get_event_loop().run_in_executor(
                None, self._client().get_all_positions
            )
            return [self._parse_position(p) for p in raws]
        except Exception as exc:
            raise BrokerError(f"get_positions failed: {exc}") from exc

    async def close_position(self, ticker: str) -> BrokerOrder:
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().close_position(ticker)
            )
            logger.info("alpaca_position_closed", ticker=ticker)
            return self._parse_order(raw)
        except Exception as exc:
            raise BrokerError(f"close_position failed for {ticker}: {exc}") from exc

    async def close_all_positions(self) -> list[BrokerOrder]:
        try:
            raws = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client().close_all_positions(cancel_orders=True)
            )
            logger.warning("alpaca_all_positions_closed")
            return [self._parse_order(r) for r in (raws or [])]
        except Exception as exc:
            raise BrokerError(f"close_all_positions failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> Quote:
        if self._data_client is None:
            raise BrokerConnectionError("Data client not initialised.")
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._data_client.get_stock_latest_quote(req)  # type: ignore[union-attr]
            )
            raw_quote = result.get(ticker)
            if raw_quote is None:
                raise BrokerError(f"No quote returned for {ticker}")
            bid = float(raw_quote.bid_price or 0)
            ask = float(raw_quote.ask_price or 0)
            price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(bid or ask)
            return Quote(
                ticker=ticker,
                price=price,
                bid=bid,
                ask=ask,
                bid_size=float(raw_quote.bid_size or 0),
                ask_size=float(raw_quote.ask_size or 0),
            )
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"get_quote failed for {ticker}: {exc}") from exc

    # ------------------------------------------------------------------
    # Market status
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return cached market-open status.  Refreshed every 60 seconds."""
        return self._market_open_cache

    async def _refresh_market_open(self) -> None:
        try:
            clock = await asyncio.get_event_loop().run_in_executor(
                None, self._client().get_clock
            )
            self._market_open_cache = bool(clock.is_open)
            self._cache_updated_at = datetime.now(tz=timezone.utc)
        except Exception as exc:
            logger.warning("alpaca_clock_refresh_failed", error=str(exc))

    async def health_check(self) -> bool:
        try:
            await self.get_account()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _build_order_request(self, order: OrderRequest) -> Any:
        """Convert our OrderRequest to the appropriate alpaca-py request object."""
        side = _to_alpaca_side(order.side)
        tif = _to_alpaca_tif(order.time_in_force)
        kwargs: dict[str, Any] = dict(
            symbol=order.ticker,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
        )
        if order.idempotency_key:
            kwargs["client_order_id"] = order.idempotency_key[:48]

        ot = order.order_type
        if ot == OrderType.MARKET:
            return MarketOrderRequest(**kwargs)
        if ot == OrderType.LIMIT:
            return LimitOrderRequest(limit_price=order.limit_price, **kwargs)
        if ot == OrderType.STOP:
            return StopOrderRequest(stop_price=order.stop_price, **kwargs)
        if ot == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=order.stop_price,
                limit_price=order.limit_price,
                **kwargs,
            )
        if ot == OrderType.TRAILING_STOP:
            return TrailingStopOrderRequest(
                trail_percent=order.trail_percent,
                **kwargs,
            )
        raise OrderError(f"Unsupported order type: {ot}")

    @staticmethod
    def _parse_order(raw: Any) -> BrokerOrder:
        status_str = str(raw.status).lower().replace("orderstatus.", "")
        status = _STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)
        side_str = str(raw.side).lower().replace("orderside.", "")
        side = _SIDE_MAP.get(side_str, OrderSide.BUY)
        type_str = str(raw.type).lower().replace("ordertype.", "")
        order_type = _TYPE_MAP.get(type_str, OrderType.MARKET)

        return BrokerOrder(
            broker_order_id=str(raw.id),
            ticker=str(raw.symbol),
            side=side,
            order_type=order_type,
            quantity=int(raw.qty or 0),
            filled_quantity=int(raw.filled_qty or 0),
            average_fill_price=(
                float(raw.filled_avg_price) if raw.filled_avg_price else None
            ),
            limit_price=float(raw.limit_price) if raw.limit_price else None,
            stop_price=float(raw.stop_price) if raw.stop_price else None,
            trail_percent=float(raw.trail_percent) if getattr(raw, "trail_percent", None) else None,
            time_in_force=str(raw.time_in_force).lower().replace("timeinforce.", ""),
            status=status,
            submitted_at=raw.submitted_at,
            filled_at=raw.filled_at,
            cancelled_at=raw.canceled_at,
            expired_at=raw.expired_at,
        )

    @staticmethod
    def _parse_position(raw: Any) -> BrokerPosition:
        return BrokerPosition(
            ticker=str(raw.symbol),
            quantity=int(raw.qty or 0),
            avg_entry_price=float(raw.avg_entry_price or 0),
            current_price=float(raw.current_price or 0),
            market_value=float(raw.market_value or 0),
            unrealized_pnl=float(raw.unrealized_pl or 0),
            unrealized_pnl_pct=float(raw.unrealized_plpc or 0) * 100.0,
            cost_basis=float(raw.cost_basis or 0),
            side=str(raw.side).lower().replace("positionside.", ""),
        )
