"""
Market Microstructure Model for realistic paper trading simulation.

Models the bid-ask spread, non-linear slippage, volume-based fill probability,
time-of-day liquidity variations, and partial fill mechanics that real equity
markets exhibit.  Designed to be used by PaperBroker but is fully independent.

Key research basis:
- Spread as function of average daily dollar volume (Glosten-Milgrom, Kyle)
- Square-root market impact model (Almgren et al. 2005, Toth et al. 2011)
- Intraday volume U-shape (Admati & Pfleiderer 1988)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import time
from typing import TYPE_CHECKING, Optional

from src.core.enums import OrderSide, OrderType

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpreadInfo:
    """Computed bid-ask spread for a given ticker at a given moment."""

    mid: float          # Mid price (reference)
    bid: float          # Simulated bid
    ask: float          # Simulated ask
    half_spread: float  # Half-spread in dollars
    spread_pct: float   # Spread as % of mid


@dataclass
class FillResult:
    """Outcome of a fill simulation for a single order."""

    filled_qty: int             # Shares actually filled (may be < requested)
    fill_price: float           # Average fill price (after spread + slippage)
    expected_price: float       # Mid price at time of order (no microstructure)
    realized_slippage: float    # fill_price - expected_price (+ = paid more)
    realized_slippage_bps: float  # Slippage in basis points
    spread_cost: float          # Cost attributable to spread (half-spread × qty)
    impact_cost: float          # Cost attributable to market impact
    partial: bool               # True if qty < requested
    fill_delay_seconds: float   # Simulated queue/delay in seconds (for limit orders)
    fill_probability: float     # Probability that was computed (0–1)


@dataclass
class ExecutionRecord:
    """Immutable record stored in the execution metrics ledger."""

    order_id: str
    ticker: str
    side: str               # "buy" | "sell"
    order_type: str         # "market" | "limit" | "stop" | …
    requested_qty: int
    filled_qty: int
    expected_price: float
    fill_price: float
    realized_slippage: float
    realized_slippage_bps: float
    spread_cost: float
    impact_cost: float
    partial: bool
    avg_daily_volume: float
    atr: float
    time_of_day_label: str  # "open_auction" | "mid_morning" | …


# ---------------------------------------------------------------------------
# Time-of-day buckets
# ---------------------------------------------------------------------------

_TOD_OPEN_START   = time(9, 30)
_TOD_OPEN_END     = time(9, 45)
_TOD_MID_MRN_END  = time(11, 30)
_TOD_MIDDAY_END   = time(14, 0)
_TOD_AFTN_END     = time(15, 30)
_TOD_CLOSE_END    = time(16, 0)


def _classify_time_of_day(t: time) -> str:
    """Return a label for the intraday liquidity bucket."""
    if _TOD_OPEN_START <= t < _TOD_OPEN_END:
        return "open_auction"
    elif _TOD_OPEN_END <= t < _TOD_MID_MRN_END:
        return "mid_morning"
    elif _TOD_MID_MRN_END <= t < _TOD_MIDDAY_END:
        return "midday"
    elif _TOD_MIDDAY_END <= t < _TOD_AFTN_END:
        return "afternoon"
    elif _TOD_AFTN_END <= t < _TOD_CLOSE_END:
        return "close_auction"
    else:
        # Outside regular hours — treat as midday (conservative)
        return "extended_hours"


# Multipliers (spread_mult, slippage_mult) per bucket
_TOD_MULTIPLIERS: dict[str, tuple[float, float]] = {
    "open_auction":    (2.0, 1.5),
    "mid_morning":     (1.0, 1.0),
    "midday":          (1.2, 1.0),
    "afternoon":       (1.0, 1.0),
    "close_auction":   (1.3, 1.2),
    "extended_hours":  (3.0, 2.0),  # Very wide spreads outside RTH
}


# ---------------------------------------------------------------------------
# Spread model parameters
# ---------------------------------------------------------------------------

# Tiers defined by average daily dollar volume (ADDV):
#   (min_addv, max_addv, min_spread_pct, max_spread_pct)
# Spread drawn uniformly within the tier's range (then scaled by TOD).
_SPREAD_TIERS: list[tuple[float, float, float, float]] = [
    (500_000_000.0, math.inf,        0.0001, 0.0003),  # Liquid (AAPL, MSFT, …)
    ( 50_000_000.0, 500_000_000.0,   0.0003, 0.0010),  # Mid-cap
    (  5_000_000.0,  50_000_000.0,   0.0010, 0.0050),  # Less liquid
    (           0.0,  5_000_000.0,   0.0050, 0.0200),  # Illiquid
]

# Fallback when no ADDV is set (assume mid-cap)
_DEFAULT_ADDV = 100_000_000.0
_DEFAULT_ATR_PCT = 0.015  # 1.5% daily ATR if not set


# ---------------------------------------------------------------------------
# MarketMicrostructureModel
# ---------------------------------------------------------------------------


class MarketMicrostructureModel:
    """All spread / slippage / fill logic, separated from broker state.

    Usage::

        model = MarketMicrostructureModel()
        model.set_avg_daily_volume("AAPL", 8_000_000_000.0)
        model.set_atr("AAPL", 2.50)  # ATR in dollars

        spread = model.compute_spread("AAPL", mid_price=175.0)
        result = model.simulate_fill(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_qty=500,
            mid_price=175.0,
            limit_price=None,
            current_time=time(10, 15),
        )
    """

    def __init__(
        self,
        rng_seed: Optional[int] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        # Per-ticker state (injected by broker / backtester)
        self._addv: dict[str, float] = {}      # Average daily dollar volume
        self._atr:  dict[str, float] = {}      # ATR in dollars (absolute)

        # RNG — deterministic seed supported for reproducible backtests
        self._rng = random.Random(rng_seed)

        # Execution metrics ledger
        self._records: list[ExecutionRecord] = []

        # Load simulation sub-models from config
        self._sim_config = config
        if config is not None:
            from src.simulation.slippage import SlippageModel
            from src.simulation.fees import FeeCalculator
            from src.simulation.spread import SpreadSimulator
            from src.simulation.partial_fills import PartialFillSimulator
            from src.simulation.latency import LatencySimulator
            from src.simulation.gaps import GapHandler

            self._slippage_model = SlippageModel(config)
            self._fee_calculator = FeeCalculator(config)
            self._spread_sim = SpreadSimulator(config, rng=self._rng)
            self._partial_fill_sim = PartialFillSimulator(config, rng=self._rng)
            self._latency_sim = LatencySimulator(config, rng=self._rng)
            self._gap_handler = GapHandler(config)
        else:
            self._slippage_model = None
            self._fee_calculator = None
            self._spread_sim = None
            self._partial_fill_sim = None
            self._latency_sim = None
            self._gap_handler = None

    # ------------------------------------------------------------------
    # State injection
    # ------------------------------------------------------------------

    def set_avg_daily_volume(self, ticker: str, addv: float) -> None:
        """Set average daily dollar volume for a ticker.

        Args:
            ticker: Equity symbol.
            addv:   Average daily *dollar* volume (price × share volume).
                    E.g. AAPL at $175 trading 80M shares → ~$14B ADDV.
        """
        self._addv[ticker] = max(addv, 1.0)

    def set_atr(self, ticker: str, atr: float) -> None:
        """Set average true range (in dollars) for a ticker.

        Args:
            ticker: Equity symbol.
            atr:    ATR value in absolute price terms (not percent).
        """
        self._atr[ticker] = max(atr, 0.0)

    # ------------------------------------------------------------------
    # Spread simulation
    # ------------------------------------------------------------------

    def compute_spread(
        self,
        ticker: str,
        mid_price: float,
        current_time: Optional[time] = None,
    ) -> SpreadInfo:
        """Compute a simulated bid-ask spread for the given ticker and time.

        The spread is drawn from a uniform distribution within the ADDV tier
        and then scaled by the time-of-day multiplier.
        """
        addv = self._addv.get(ticker, _DEFAULT_ADDV)
        tod_label = _classify_time_of_day(current_time) if current_time else "mid_morning"
        spread_mult, _ = _TOD_MULTIPLIERS[tod_label]

        # Pick the ADDV tier
        base_min_pct, base_max_pct = self._get_spread_range(addv)

        # Draw spread within tier
        raw_spread_pct = self._rng.uniform(base_min_pct, base_max_pct)
        effective_spread_pct = raw_spread_pct * spread_mult

        half_spread = mid_price * effective_spread_pct / 2.0
        bid = mid_price - half_spread
        ask = mid_price + half_spread

        return SpreadInfo(
            mid=mid_price,
            bid=round(bid, 4),
            ask=round(ask, 4),
            half_spread=round(half_spread, 6),
            spread_pct=round(effective_spread_pct * 100.0, 6),
        )

    # ------------------------------------------------------------------
    # Fill simulation (the main entry point)
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        ticker: str,
        side: OrderSide,
        order_type: OrderType,
        requested_qty: int,
        mid_price: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_time: Optional[time] = None,
        open_price: Optional[float] = None,
        previous_close: Optional[float] = None,
    ) -> FillResult:
        """Simulate realistic fill for a single order leg.

        Returns a FillResult.  If fill_probability < random draw, the result
        will have filled_qty == 0 (limit order not filled this bar).

        Steps:
        1. Compute spread (bid/ask).
        2. Compute fill probability (limits) or force-fill (market).
        3. Compute market impact slippage.
        4. Apply partial fill if order is large relative to ADDV.
        5. Apply execution latency price drift.
        6. Handle gap-through for stop orders.
        7. Compute fill delay for limit/stop orders.
        """
        # Normalize to enums — callers may pass plain strings ("buy"/"sell")
        if isinstance(side, str) and not isinstance(side, OrderSide):
            side = OrderSide(side)
        if isinstance(order_type, str) and not isinstance(order_type, OrderType):
            order_type = OrderType(order_type)

        addv = self._addv.get(ticker, _DEFAULT_ADDV)
        atr  = self._atr.get(ticker, mid_price * _DEFAULT_ATR_PCT)
        tod_label = _classify_time_of_day(current_time) if current_time else "mid_morning"
        spread_mult, slippage_mult = _TOD_MULTIPLIERS[tod_label]

        # --- Step 1: Spread ---
        spread_info = self.compute_spread(ticker, mid_price, current_time)

        # The reference fill price before impact (buys fill at ask, sells at bid)
        if side == OrderSide.BUY:
            base_fill = spread_info.ask
        else:
            base_fill = spread_info.bid

        # --- Step 2: Fill probability ---
        fill_prob, fill_delay = self._compute_fill_probability(
            order_type=order_type,
            side=side,
            limit_price=limit_price,
            stop_price=stop_price,
            spread_info=spread_info,
        )

        # Roll the dice
        if fill_prob < 1.0 and self._rng.random() > fill_prob:
            # Order not filled this bar
            return FillResult(
                filled_qty=0,
                fill_price=0.0,
                expected_price=mid_price,
                realized_slippage=0.0,
                realized_slippage_bps=0.0,
                spread_cost=0.0,
                impact_cost=0.0,
                partial=False,
                fill_delay_seconds=fill_delay,
                fill_probability=fill_prob,
            )

        # --- Step 3: Market impact slippage ---
        atr_pct = atr / mid_price if mid_price > 0 else _DEFAULT_ATR_PCT

        if self._slippage_model is not None:
            # Use configurable slippage model
            order_dollar_value = requested_qty * mid_price
            slip_result = self._slippage_model.calculate(
                order_dollar_value=order_dollar_value,
                avg_daily_dollar_volume=addv,
                atr_pct=atr_pct,
                side=side.value,
            )
            impact_pct = slip_result.slippage_pct
            # Apply TOD multiplier
            impact_pct *= slippage_mult
            # Add noise
            impact_pct *= self._rng.uniform(0.85, 1.15)
            # Cap
            impact_pct = min(impact_pct, 0.05)
        else:
            impact_pct = self._compute_market_impact(
                ticker=ticker,
                qty=requested_qty,
                mid_price=mid_price,
                addv=addv,
                atr=atr,
                slippage_mult=slippage_mult,
            )

        # Impact is always adverse: buys pay more, sells receive less
        if side == OrderSide.BUY:
            fill_price = base_fill * (1.0 + impact_pct)
        else:
            fill_price = base_fill * (1.0 - impact_pct)

        # --- Step 4: Execution latency price drift ---
        if self._latency_sim is not None:
            latency_result = self._latency_sim.simulate(
                signal_price=fill_price,
                atr_pct=atr_pct,
                side=side.value,
            )
            fill_price = latency_result.adjusted_price
            fill_delay += latency_result.delay_ms / 1000.0  # Convert ms to seconds

        # --- Step 5: Gap handling for stop orders ---
        if order_type == OrderType.STOP and self._gap_handler is not None:
            if open_price is not None and previous_close is not None:
                gap_result = self._gap_handler.check_gap_fill(
                    stop_price=stop_price or fill_price,
                    open_price=open_price,
                    previous_close=previous_close,
                    side=side.value,
                )
                if gap_result.is_gapped:
                    fill_price = gap_result.fill_price
        elif order_type == OrderType.STOP:
            # Legacy gap-through behavior
            gap_pct = self._rng.uniform(0.0005, 0.0020) * slippage_mult
            if side == OrderSide.SELL:
                fill_price *= (1.0 - gap_pct)
            else:
                fill_price *= (1.0 + gap_pct)

        # --- Step 6: Partial fills ---
        if self._partial_fill_sim is not None:
            pf_result = self._partial_fill_sim.calculate(
                requested_qty=requested_qty,
                mid_price=mid_price,
                avg_daily_dollar_volume=addv,
            )
            filled_qty = pf_result.filled_qty
            is_partial = pf_result.is_partial
        else:
            filled_qty, is_partial = self._compute_partial_fill(
                ticker=ticker,
                requested_qty=requested_qty,
                mid_price=mid_price,
                addv=addv,
            )

        # --- Step 7: Cost breakdown ---
        spread_cost_per_share = spread_info.half_spread  # one-way spread cost
        spread_total = spread_cost_per_share * filled_qty

        impact_cost_per_share = abs(fill_price - base_fill)
        impact_total = impact_cost_per_share * filled_qty

        realized_slippage = (fill_price - mid_price) if side == OrderSide.BUY \
                            else (mid_price - fill_price)
        realized_slippage_bps = (realized_slippage / mid_price * 10_000.0) if mid_price > 0 else 0.0

        return FillResult(
            filled_qty=filled_qty,
            fill_price=round(fill_price, 4),
            expected_price=mid_price,
            realized_slippage=round(realized_slippage, 6),
            realized_slippage_bps=round(realized_slippage_bps, 3),
            spread_cost=round(spread_total, 4),
            impact_cost=round(impact_total, 4),
            partial=is_partial,
            fill_delay_seconds=fill_delay,
            fill_probability=fill_prob,
        )

    # ------------------------------------------------------------------
    # Execution metrics
    # ------------------------------------------------------------------

    def record_execution(
        self,
        order_id: str,
        ticker: str,
        side: OrderSide,
        order_type: OrderType,
        requested_qty: int,
        result: FillResult,
        current_time: Optional[time] = None,
    ) -> None:
        """Persist a FillResult to the internal ledger."""
        # Normalize to enums — callers may pass plain strings
        if isinstance(side, str) and not isinstance(side, OrderSide):
            side = OrderSide(side)
        if isinstance(order_type, str) and not isinstance(order_type, OrderType):
            order_type = OrderType(order_type)

        tod_label = _classify_time_of_day(current_time) if current_time else "mid_morning"
        addv = self._addv.get(ticker, _DEFAULT_ADDV)
        atr  = self._atr.get(ticker, 0.0)

        self._records.append(ExecutionRecord(
            order_id=order_id,
            ticker=ticker,
            side=side.value,
            order_type=order_type.value,
            requested_qty=requested_qty,
            filled_qty=result.filled_qty,
            expected_price=result.expected_price,
            fill_price=result.fill_price,
            realized_slippage=result.realized_slippage,
            realized_slippage_bps=result.realized_slippage_bps,
            spread_cost=result.spread_cost,
            impact_cost=result.impact_cost,
            partial=result.partial,
            avg_daily_volume=addv,
            atr=atr,
            time_of_day_label=tod_label,
        ))

    def get_execution_metrics(self) -> dict:
        """Return aggregated execution quality metrics across all recorded fills.

        Returns a dict with:
        - total_fills: number of fills recorded
        - total_partial_fills: fills where filled_qty < requested_qty
        - avg_slippage_bps: mean realized slippage in basis points
        - avg_spread_cost: mean spread cost per fill (dollars)
        - avg_impact_cost: mean market impact cost per fill (dollars)
        - fill_rate: ratio of filled_qty to requested_qty across all orders
        - by_ticker: per-ticker breakdown dict
        """
        if not self._records:
            return {
                "total_fills": 0,
                "total_partial_fills": 0,
                "avg_slippage_bps": 0.0,
                "avg_spread_cost": 0.0,
                "avg_impact_cost": 0.0,
                "fill_rate": 1.0,
                "by_ticker": {},
            }

        total_fills = len(self._records)
        partial_fills = sum(1 for r in self._records if r.partial)
        total_requested = sum(r.requested_qty for r in self._records)
        total_filled = sum(r.filled_qty for r in self._records)
        avg_slippage_bps = sum(r.realized_slippage_bps for r in self._records) / total_fills
        avg_spread = sum(r.spread_cost for r in self._records) / total_fills
        avg_impact = sum(r.impact_cost for r in self._records) / total_fills
        fill_rate = (total_filled / total_requested) if total_requested > 0 else 1.0

        # Per-ticker breakdown
        by_ticker: dict[str, dict] = {}
        for r in self._records:
            t = r.ticker
            if t not in by_ticker:
                by_ticker[t] = {
                    "fills": 0,
                    "total_slippage_bps": 0.0,
                    "total_spread_cost": 0.0,
                    "total_impact_cost": 0.0,
                    "requested_qty": 0,
                    "filled_qty": 0,
                }
            bt = by_ticker[t]
            bt["fills"] += 1
            bt["total_slippage_bps"] += r.realized_slippage_bps
            bt["total_spread_cost"] += r.spread_cost
            bt["total_impact_cost"] += r.impact_cost
            bt["requested_qty"] += r.requested_qty
            bt["filled_qty"] += r.filled_qty

        # Compute averages per ticker
        for t, bt in by_ticker.items():
            n = bt["fills"]
            bt["avg_slippage_bps"] = round(bt["total_slippage_bps"] / n, 3)
            bt["avg_spread_cost"] = round(bt["total_spread_cost"] / n, 4)
            bt["avg_impact_cost"] = round(bt["total_impact_cost"] / n, 4)
            bt["fill_rate"] = round(bt["filled_qty"] / bt["requested_qty"], 4) \
                              if bt["requested_qty"] > 0 else 1.0

        return {
            "total_fills": total_fills,
            "total_partial_fills": partial_fills,
            "avg_slippage_bps": round(avg_slippage_bps, 3),
            "avg_spread_cost": round(avg_spread, 4),
            "avg_impact_cost": round(avg_impact, 4),
            "fill_rate": round(fill_rate, 4),
            "by_ticker": by_ticker,
        }

    def get_execution_records(self) -> list[ExecutionRecord]:
        """Return raw execution records (read-only copy)."""
        return list(self._records)

    def reset_metrics(self) -> None:
        """Clear all stored execution records."""
        self._records.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_spread_range(self, addv: float) -> tuple[float, float]:
        """Return (min_spread_pct, max_spread_pct) for the given ADDV tier."""
        for min_addv, max_addv, min_pct, max_pct in _SPREAD_TIERS:
            if min_addv <= addv < max_addv:
                return min_pct, max_pct
        # Fallback (should not happen given the tier structure)
        return _SPREAD_TIERS[-1][2], _SPREAD_TIERS[-1][3]

    def _compute_fill_probability(
        self,
        order_type: OrderType,
        side: OrderSide,
        limit_price: Optional[float],
        stop_price: Optional[float],
        spread_info: SpreadInfo,
    ) -> tuple[float, float]:
        """Return (fill_probability, fill_delay_seconds).

        Market orders: probability = 1.0, delay = 0.
        Limit orders: probability decreases as limit moves away from market.
        Stop orders: probability = 1.0 once triggered (we assume trigger has happened),
                     but with a small random delay for queue effects.
        """
        if order_type == OrderType.MARKET:
            return 1.0, 0.0

        if order_type in (OrderType.STOP, OrderType.TRAILING_STOP):
            # Stop has been triggered — treat as market with small delay
            delay = self._rng.uniform(0.5, 3.0)
            return 1.0, delay

        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if limit_price is None:
                # Degenerate — treat as market
                return 1.0, 0.0

            if side == OrderSide.BUY:
                # How far below the ask is the limit?
                reference = spread_info.ask
                distance_pct = (reference - limit_price) / reference if reference > 0 else 0.0
            else:
                # How far above the bid is the limit (i.e., sell limit below bid → aggressive)?
                reference = spread_info.bid
                distance_pct = (limit_price - reference) / reference if reference > 0 else 0.0

            fill_prob = self._limit_fill_prob(distance_pct)

            # Queue delay: closer to market → faster fill
            if distance_pct <= 0.0:
                delay = self._rng.uniform(0.1, 2.0)
            elif distance_pct <= 0.001:
                delay = self._rng.uniform(1.0, 10.0)
            elif distance_pct <= 0.005:
                delay = self._rng.uniform(5.0, 30.0)
            else:
                delay = self._rng.uniform(15.0, 60.0)

            return fill_prob, delay

        # Unknown order type — treat as market
        return 1.0, 0.0

    @staticmethod
    def _limit_fill_prob(distance_pct: float) -> float:
        """Map limit order distance from market to fill probability.

        distance_pct > 0 means limit is away from the market (passive).
        distance_pct <= 0 means limit is at or better than market (aggressive).

        Schedule:
          ≤ 0.0%  → 0.95
          0.1%    → 0.60
          0.5%    → 0.30
          ≥ 1.0%  → 0.10
        """
        if distance_pct <= 0.0:
            return 0.95
        elif distance_pct <= 0.001:
            # Linear interpolation 0.0%→0.95, 0.1%→0.60
            t = distance_pct / 0.001
            return 0.95 - t * (0.95 - 0.60)
        elif distance_pct <= 0.005:
            # Linear interpolation 0.1%→0.60, 0.5%→0.30
            t = (distance_pct - 0.001) / (0.005 - 0.001)
            return 0.60 - t * (0.60 - 0.30)
        elif distance_pct <= 0.010:
            # Linear interpolation 0.5%→0.30, 1.0%→0.10
            t = (distance_pct - 0.005) / (0.010 - 0.005)
            return 0.30 - t * (0.30 - 0.10)
        else:
            return 0.10

    def _compute_market_impact(
        self,
        ticker: str,
        qty: int,
        mid_price: float,
        addv: float,
        atr: float,
        slippage_mult: float,
    ) -> float:
        """Return market impact as a fraction of mid price.

        Uses the square-root model:
            impact_pct = base_slippage * sqrt(order_dollar_value / addv)

        Then scaled by a volatility multiplier:
            volatility_mult = 1 + (atr / mid_price) / default_atr_pct

        The base_slippage is calibrated so that:
          - 0.1% of ADDV → ~5 bps impact
          - 1% of ADDV   → ~16 bps impact
          - 10% of ADDV  → ~50 bps impact
        """
        if mid_price <= 0 or addv <= 0:
            return 0.0

        order_dollar_value = qty * mid_price
        participation = order_dollar_value / addv

        # Square-root impact model (base coefficient ≈ 0.05 = 5 bps at 1% participation)
        BASE_COEFFICIENT = 0.0050
        raw_impact = BASE_COEFFICIENT * math.sqrt(participation)

        # Volatility multiplier: ATR-based
        atr_pct = atr / mid_price if mid_price > 0 else _DEFAULT_ATR_PCT
        vol_mult = 1.0 + max(0.0, (atr_pct / _DEFAULT_ATR_PCT) - 1.0) * 0.5

        # Add small random noise to prevent identical fills
        noise = self._rng.uniform(0.85, 1.15)

        impact = raw_impact * vol_mult * slippage_mult * noise
        return min(impact, 0.05)  # Cap at 5% impact (extreme safety)

    def _compute_partial_fill(
        self,
        ticker: str,
        requested_qty: int,
        mid_price: float,
        addv: float,
    ) -> tuple[int, bool]:
        """Return (filled_qty, is_partial).

        Orders larger than 5% of average daily *share* volume trigger partial fills.
        We estimate daily share volume as addv / mid_price.

        Fill rate = min(1.0, 0.05 * adv_shares / requested_qty)
        """
        if mid_price <= 0 or requested_qty <= 0:
            return requested_qty, False

        adv_shares = addv / mid_price  # Estimated average daily share volume
        threshold_qty = 0.05 * adv_shares  # 5% threshold

        if requested_qty <= threshold_qty:
            return requested_qty, False

        # Compute fill rate per the specification
        fill_rate = min(1.0, (0.05 * adv_shares) / requested_qty)

        # Add randomness: actual fill between 80% and 110% of computed rate
        actual_rate = fill_rate * self._rng.uniform(0.80, 1.10)
        actual_rate = min(1.0, actual_rate)

        filled = max(1, int(requested_qty * actual_rate))
        is_partial = filled < requested_qty

        return filled, is_partial
