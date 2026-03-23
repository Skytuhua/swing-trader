"""
In-memory paper broker with a realistic market microstructure model.

Replaces the original flat-slippage PaperBroker with a simulation that models:

  1. Variable bid-ask spreads   — function of average daily dollar volume
  2. Non-linear slippage        — square-root market impact model
  3. Volume-based fill probability — limit orders fill probabilistically
  4. Time-of-day liquidity      — opening / midday / closing spread/slippage multipliers
  5. Partial fills              — orders > 5 % of daily volume fill partially
  6. Order queue position       — limit orders away from market get random delay
  7. Execution quality metrics  — exposed via get_execution_metrics()

The class still implements the same BrokerAdapter interface as before and keeps
full backward-compatibility with existing test helpers:
  set_price(), set_prices(), set_market_open()
  set_avg_daily_volume(), set_atr(), set_time_of_day()

All fill logic is delegated to MarketMicrostructureModel (market_simulator.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timezone
from typing import Any, Optional

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
from src.services.execution.market_simulator import (
    MarketMicrostructureModel,
    SpreadInfo,
)
from src.simulation.fees import FeeCalculator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_INITIAL_CASH: float = 100_000.0
_DEFAULT_COMMISSION_PER_SHARE: float = 0.0   # Alpaca paper: $0 commission


class PaperBroker(BrokerAdapter):
    """In-memory simulated broker with realistic market microstructure.

    Designed for paper trading and backtesting.  All fills are computed
    deterministically (given the same RNG seed) using a microstructure model
    that accounts for spreads, impact, liquidity, and fill probability.

    Quick-start::

        broker = PaperBroker(initial_cash=100_000)
        broker.set_price("AAPL", 175.00)
        broker.set_avg_daily_volume("AAPL", 14_000_000_000.0)  # ~$14B ADDV
        broker.set_atr("AAPL", 2.50)
        broker.set_time_of_day(time(10, 30))                   # mid-morning

        order = OrderRequest(ticker="AAPL", side=OrderSide.BUY,
                             order_type=OrderType.MARKET, quantity=100)
        result = await broker.submit_order(order)
    """

    def __init__(
        self,
        initial_cash: float = _DEFAULT_INITIAL_CASH,
        commission_per_share: float = _DEFAULT_COMMISSION_PER_SHARE,
        # Legacy parameter — kept for backward compatibility but ignored.
        # Slippage is now computed by MarketMicrostructureModel.
        slippage_pct: float = 0.001,
        # Legacy parameter — kept for backward compatibility.
        # True enables partial fills via the microstructure model.
        simulate_partial_fills: bool = True,
        rng_seed: Optional[int] = None,
        simulation_config: Optional[Any] = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_per_share = commission_per_share
        # Kept as attributes so tests can read them
        self.slippage_pct = slippage_pct
        self.simulate_partial_fills = simulate_partial_fills

        # Load simulation config from global settings if not explicitly provided
        self._sim_config = simulation_config
        if self._sim_config is None:
            try:
                from src.core.config import get_settings
                self._sim_config = get_settings().simulation
            except Exception:
                pass  # Config not available, use defaults

        # Core microstructure engine (with simulation config for enhanced models)
        self._model = MarketMicrostructureModel(
            rng_seed=rng_seed, config=self._sim_config
        )

        # Fee calculator (uses simulation config or falls back to per-share)
        self._fee_calculator: Optional[FeeCalculator] = None
        if self._sim_config is not None:
            self._fee_calculator = FeeCalculator(self._sim_config)

        # Broker state
        self._cash: float = initial_cash
        self._prices: dict[str, float] = {}
        self._orders: dict[str, BrokerOrder] = {}       # broker_id → order
        self._positions: dict[str, dict[str, Any]] = {} # ticker → {qty, avg_price}
        self._market_open: bool = True

        # Time-of-day state (injected externally for backtesting)
        self._current_time: Optional[time] = None

        # Cumulative PnL / trade tracking
        self._realised_pnl: float = 0.0
        self._total_trades: int = 0

        # Pending limit / stop orders waiting in the queue
        # Maps broker_order_id → (OrderRequest, fill_delay_seconds)
        self._pending_orders: dict[str, tuple[OrderRequest, float]] = {}

    # ------------------------------------------------------------------
    # State injection (test helpers + backtester hooks)
    # ------------------------------------------------------------------

    def set_price(self, ticker: str, price: float) -> None:
        """Set (or update) the simulated mid price for a ticker."""
        self._prices[ticker] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        """Bulk-set mid prices."""
        self._prices.update(prices)

    def set_market_open(self, is_open: bool) -> None:
        """Override market-open flag (used by tests and pre-market checks)."""
        self._market_open = is_open

    def set_avg_daily_volume(self, ticker: str, avg_daily_volume: float) -> None:
        """Inject average daily dollar volume so the spread/impact model
        can calibrate to real liquidity.

        Args:
            ticker:            Equity symbol.
            avg_daily_volume:  Average daily *dollar* volume
                               (share price × daily share volume).
                               E.g. ~$14B for AAPL, ~$500M for a mid-cap.
        """
        self._model.set_avg_daily_volume(ticker, avg_daily_volume)

    def set_atr(self, ticker: str, atr: float) -> None:
        """Inject ATR (average true range in dollars) for volatility scaling.

        Args:
            ticker:  Equity symbol.
            atr:     ATR value in absolute dollar terms.  A stock priced at
                     $100 with a 2% daily range has ATR ≈ $2.00.
        """
        self._model.set_atr(ticker, atr)

    def set_time_of_day(self, t: time) -> None:
        """Set the simulated clock time so the broker applies the correct
        intraday liquidity multipliers.

        Args:
            t:  A ``datetime.time`` object in market-local time (US/Eastern).
        """
        self._current_time = t

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        portfolio_value = self._cash + self._portfolio_market_value()
        return AccountInfo(
            account_id="paper-0001",
            equity=portfolio_value,
            cash=self._cash,
            buying_power=self._cash,  # No margin in paper mode
            portfolio_value=portfolio_value,
            currency="USD",
            status="ACTIVE",
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> BrokerOrder:
        """Simulate order submission with realistic microstructure effects.

        Market orders fill immediately with spread + impact slippage.
        Limit orders may not fill (based on fill probability); if they don't
        fill on submission they remain in ``get_open_orders()`` as PENDING.
        Stop orders fill with gap-through risk applied.
        """
        if not self.is_market_open():
            raise OrderError("Market is closed; cannot submit paper order.")

        mid_price = self._get_price(order.ticker)
        now = datetime.now(tz=timezone.utc)
        broker_id = order.idempotency_key or str(uuid.uuid4())

        # --- Run the microstructure simulation ---
        fill_result = self._model.simulate_fill(
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            requested_qty=order.quantity,
            mid_price=mid_price,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            current_time=self._current_time,
        )

        # --- Handle unfilled limit order ---
        if fill_result.filled_qty == 0:
            broker_order = BrokerOrder(
                broker_order_id=broker_id,
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=0,
                average_fill_price=None,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                status=OrderStatus.PENDING,
                submitted_at=now,
                raw={
                    "fill_probability": fill_result.fill_probability,
                    "fill_delay_seconds": fill_result.fill_delay_seconds,
                },
            )
            self._orders[broker_id] = broker_order
            self._pending_orders[broker_id] = (order, fill_result.fill_delay_seconds)
            logger.info(
                "paper_broker_order_pending",
                ticker=order.ticker,
                side=order.side.value,
                order_type=order.order_type.value,
                fill_probability=round(fill_result.fill_probability, 3),
                delay_secs=round(fill_result.fill_delay_seconds, 1),
            )
            return broker_order

        filled_qty = fill_result.filled_qty
        fill_price = fill_result.fill_price

        # --- Apply commission/fees ---
        if self._fee_calculator is not None:
            fee_result = self._fee_calculator.calculate(
                quantity=filled_qty,
                fill_price=fill_price,
                side=order.side.value,
                order_type=order.order_type.value,
            )
            commission = fee_result.total_fee
        else:
            commission = self.commission_per_share * filled_qty

        # --- Update portfolio state ---
        if order.side == OrderSide.BUY:
            cost = fill_price * filled_qty + commission
            if cost > self._cash:
                raise OrderError(
                    f"Insufficient cash: need ${cost:.2f}, have ${self._cash:.2f}"
                )
            self._cash -= cost
            self._update_position(order.ticker, filled_qty, fill_price)
        else:  # SELL
            qty_to_sell = min(filled_qty, self._get_position_qty(order.ticker))
            if qty_to_sell <= 0:
                raise OrderError(f"No position in {order.ticker} to sell.")
            # If partial fill reduced qty further, adjust
            filled_qty = qty_to_sell
            proceeds = fill_price * filled_qty - commission
            self._cash += proceeds
            self._close_position(order.ticker, filled_qty, fill_price)
            self._total_trades += 1

        # --- Record execution in the metrics ledger ---
        self._model.record_execution(
            order_id=broker_id,
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            requested_qty=order.quantity,
            result=fill_result,
            current_time=self._current_time,
        )

        is_partial = fill_result.partial or (filled_qty < order.quantity)
        status = OrderStatus.PARTIAL if is_partial else OrderStatus.FILLED

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
            status=status,
            submitted_at=now,
            filled_at=now,
            raw={
                "expected_price": fill_result.expected_price,
                "realized_slippage": fill_result.realized_slippage,
                "realized_slippage_bps": fill_result.realized_slippage_bps,
                "spread_cost": fill_result.spread_cost,
                "impact_cost": fill_result.impact_cost,
                "fill_probability": fill_result.fill_probability,
                "fill_delay_seconds": fill_result.fill_delay_seconds,
                "partial": fill_result.partial,
            },
        )

        self._orders[broker_id] = broker_order

        logger.info(
            "paper_broker_fill",
            ticker=order.ticker,
            side=order.side.value,
            order_type=order.order_type.value,
            qty=filled_qty,
            requested_qty=order.quantity,
            fill_price=round(fill_price, 4),
            expected_price=round(fill_result.expected_price, 4),
            slippage_bps=round(fill_result.realized_slippage_bps, 2),
            spread_cost=round(fill_result.spread_cost, 4),
            impact_cost=round(fill_result.impact_cost, 4),
            partial=is_partial,
            cash_remaining=round(self._cash, 2),
        )

        return broker_order

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open or pending order."""
        order = self._orders.get(broker_order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(tz=timezone.utc)
        # Remove from pending queue if present
        self._pending_orders.pop(broker_order_id, None)
        logger.info("paper_broker_order_cancelled", broker_order_id=broker_order_id)
        return True

    async def get_order_status(self, broker_order_id: str) -> BrokerOrder:
        """Return order by broker ID, or raise BrokerError if not found."""
        order = self._orders.get(broker_order_id)
        if order is None:
            raise BrokerError(f"Order {broker_order_id} not found in paper broker.")
        return order

    async def get_open_orders(self) -> list[BrokerOrder]:
        """Return all orders that are not yet in a terminal state."""
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING, OrderStatus.PARTIAL)
        ]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[BrokerPosition]:
        """Return all open positions with current market values."""
        positions = []
        for ticker, pos in self._positions.items():
            if pos["qty"] <= 0:
                continue
            current_price = self._prices.get(ticker, pos["avg_price"])
            market_value = current_price * pos["qty"]
            cost_basis = pos["avg_price"] * pos["qty"]
            unrealised_pnl = market_value - cost_basis
            unrealised_pnl_pct = (
                (unrealised_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
            )
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
        """Close the entire open position for a ticker at market."""
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
        """Close every open position at market (kill-switch path)."""
        orders = []
        tickers = [t for t, p in self._positions.items() if p["qty"] > 0]
        for ticker in tickers:
            try:
                order = await self.close_position(ticker)
                orders.append(order)
            except Exception as exc:
                logger.warning(
                    "paper_broker_close_all_error", ticker=ticker, error=str(exc)
                )
        logger.warning("paper_broker_all_positions_closed", count=len(orders))
        return orders

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> Quote:
        """Return a simulated quote with realistic bid/ask spread."""
        mid = self._prices.get(ticker)
        if mid is None:
            raise BrokerError(f"No price set for {ticker} in paper broker.")

        spread_info: SpreadInfo = self._model.compute_spread(
            ticker=ticker,
            mid_price=mid,
            current_time=self._current_time,
        )

        return Quote(
            ticker=ticker,
            price=mid,
            bid=spread_info.bid,
            ask=spread_info.ask,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        return self._market_open

    async def health_check(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Execution quality metrics
    # ------------------------------------------------------------------

    def get_execution_metrics(self) -> dict[str, Any]:
        """Return aggregated execution quality metrics.

        Returns a dict with:
        - total_fills: number of fills recorded
        - total_partial_fills: number of partial fills
        - avg_slippage_bps: mean realized slippage in basis points
        - avg_spread_cost: mean spread cost per fill in dollars
        - avg_impact_cost: mean market impact cost per fill in dollars
        - fill_rate: ratio of filled to requested shares across all orders
        - by_ticker: per-ticker breakdown dict

        Additionally includes broker-level stats:
        - total_trades: cumulative completed round-trips
        - realised_pnl: cumulative realized P&L in dollars
        - open_positions: number of open positions
        - cash: available cash
        - portfolio_value: total account value (cash + market value)
        """
        model_metrics = self._model.get_execution_metrics()
        portfolio_value = self._cash + self._portfolio_market_value()
        model_metrics.update({
            "total_trades": self._total_trades,
            "realised_pnl": round(self._realised_pnl, 2),
            "open_positions": len([p for p in self._positions.values() if p["qty"] > 0]),
            "cash": round(self._cash, 2),
            "portfolio_value": round(portfolio_value, 2),
        })
        return model_metrics

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a high-level summary dict (backward compatible with original)."""
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
        """Retrieve the mid price for a ticker, raising OrderError if missing."""
        price = self._prices.get(ticker)
        if price is None or price <= 0:
            raise OrderError(
                f"No valid price available for {ticker} in paper broker. "
                "Call set_price(ticker, price) before submitting orders."
            )
        return price

    def _get_position_qty(self, ticker: str) -> int:
        return self._positions.get(ticker, {}).get("qty", 0)

    def _update_position(self, ticker: str, qty: int, price: float) -> None:
        """Add shares to an existing position (or open a new one) using
        weighted-average cost basis."""
        existing = self._positions.get(ticker)
        if existing and existing["qty"] > 0:
            total_qty = existing["qty"] + qty
            avg = (existing["avg_price"] * existing["qty"] + price * qty) / total_qty
            existing["qty"] = total_qty
            existing["avg_price"] = avg
        else:
            self._positions[ticker] = {"qty": qty, "avg_price": price}

    def _close_position(self, ticker: str, qty: int, sell_price: float) -> None:
        """Reduce a position and accrue realized P&L."""
        pos = self._positions.get(ticker)
        if pos is None:
            return
        realised = (sell_price - pos["avg_price"]) * qty
        self._realised_pnl += realised
        pos["qty"] = max(0, pos["qty"] - qty)
        if pos["qty"] == 0:
            del self._positions[ticker]

    def _portfolio_market_value(self) -> float:
        """Compute total market value of all open positions at current mid prices."""
        total = 0.0
        for ticker, pos in self._positions.items():
            if pos["qty"] > 0:
                price = self._prices.get(ticker, pos["avg_price"])
                total += price * pos["qty"]
        return total
