"""
Execution quality tracking for all trade fills.

Tracks slippage (expected vs actual fill price), fill rates for limit orders,
and average fill times.  Alerts when execution quality has deteriorated below
a rolling baseline, which may indicate broker issues, unusual market
conditions, or strategy changes that increase market impact.

Terminology
-----------
Slippage
    The difference between the expected fill price and the actual fill price,
    expressed in basis points (bps, i.e. 0.01%) or dollars.  For a buy order,
    positive slippage means we paid more than expected (unfavourable).  For a
    sell order, positive slippage means we received less than expected
    (unfavourable).  This module normalises slippage so that positive always
    means unfavourable to the trader.

Basis points (bps)
    1 bps = 0.01%.  Slippage of 10 bps on a $100 stock = $0.10 per share.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BPS_SCALE = 10_000.0  # multiplier to convert fractional to bps


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FillRecord:
    """Record of a single order fill.

    Attributes
    ----------
    ticker:
        Instrument.
    expected_price:
        The price used when the order was submitted (e.g., mid-price at signal
        time, limit price, or last trade price).
    actual_price:
        The price at which the order was actually filled.
    quantity:
        Shares filled in this record.
    side:
        ``"buy"`` or ``"sell"``.
    order_type:
        ``"market"``, ``"limit"``, ``"stop"``, etc.
    timestamp:
        UTC datetime of the fill.
    slippage_bps:
        Pre-computed slippage in basis points (positive = unfavourable).
    slippage_usd:
        Pre-computed slippage in USD (positive = unfavourable cost).
    fill_time_seconds:
        Optional: time elapsed between order submission and fill in seconds.
        None if not tracked.
    filled:
        False for cancelled/expired limit orders that never received a fill.
    """

    ticker: str
    expected_price: float
    actual_price: float
    quantity: int
    side: str        # "buy" | "sell"
    order_type: str  # "market" | "limit" | "stop" | ...
    timestamp: datetime
    slippage_bps: float
    slippage_usd: float
    fill_time_seconds: float | None = None
    filled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Aggregate execution quality statistics.

    Attributes
    ----------
    sample_size:
        Number of fill records used to compute these metrics.
    avg_slippage_bps:
        Mean slippage in basis points across all fills.
    median_slippage_bps:
        Median slippage in bps (more robust to outliers than mean).
    worst_slippage_bps:
        Single worst (highest) slippage event in bps.
    best_slippage_bps:
        Best (lowest / negative = price improvement) slippage event.
    stddev_slippage_bps:
        Standard deviation of slippage distribution.
    p95_slippage_bps:
        95th percentile slippage (tail risk indicator).
    fill_rate:
        Fraction of limit orders that received a complete fill (0–1).
    avg_fill_time_seconds:
        Mean fill latency for orders where fill time was recorded.
    total_slippage_usd:
        Cumulative dollar cost of slippage across all recorded fills.
    market_order_avg_slippage_bps:
        Average slippage for market orders only.
    limit_order_avg_slippage_bps:
        Average slippage for limit orders only.
    buy_avg_slippage_bps:
        Average slippage for buy-side fills.
    sell_avg_slippage_bps:
        Average slippage for sell-side fills.
    degraded:
        True when recent execution quality is materially worse than the
        longer-term baseline.
    """

    sample_size: int = 0
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    best_slippage_bps: float = 0.0
    stddev_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    fill_rate: float = 1.0
    avg_fill_time_seconds: float | None = None
    total_slippage_usd: float = 0.0
    market_order_avg_slippage_bps: float = 0.0
    limit_order_avg_slippage_bps: float = 0.0
    buy_avg_slippage_bps: float = 0.0
    sell_avg_slippage_bps: float = 0.0
    degraded: bool = False


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ExecutionTracker:
    """Track execution quality across all trades.

    Parameters
    ----------
    degradation_window:
        Number of recent fills to include in the "recent" window used to
        detect execution degradation.
    degradation_threshold_bps:
        If the recent mean slippage exceeds the long-term mean by this many
        bps, execution is considered degraded.
    degradation_min_samples:
        Minimum total fills required before degradation detection is active.
    """

    def __init__(
        self,
        degradation_window: int = 20,
        degradation_threshold_bps: float = 5.0,
        degradation_min_samples: int = 10,
    ) -> None:
        self.fills: list[FillRecord] = []
        self._degradation_window = degradation_window
        self._degradation_threshold_bps = degradation_threshold_bps
        self._degradation_min_samples = degradation_min_samples

    # ------------------------------------------------------------------
    # Recording fills
    # ------------------------------------------------------------------

    def record_fill(
        self,
        ticker: str,
        expected_price: float,
        actual_price: float,
        quantity: int,
        side: str,
        order_type: str,
        timestamp: datetime,
        fill_time_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> FillRecord:
        """Record a fill and compute slippage metrics.

        Parameters
        ----------
        ticker:
            Instrument symbol.
        expected_price:
            Reference price at order submission time.
        actual_price:
            Actual fill price.
        quantity:
            Number of shares filled.
        side:
            ``"buy"`` or ``"sell"``.
        order_type:
            ``"market"``, ``"limit"``, ``"stop"``, etc.
        timestamp:
            UTC datetime of the fill.
        fill_time_seconds:
            Optional: elapsed seconds from order submission to fill.
        extra:
            Optional dict of additional metadata.

        Returns
        -------
        FillRecord
            The recorded fill object.
        """
        side_norm = side.lower()
        if expected_price <= 0:
            logger.warning(
                "execution_tracker.invalid_expected_price",
                ticker=ticker,
                expected_price=expected_price,
            )
            slippage_bps = 0.0
            slippage_usd = 0.0
        else:
            # Raw fractional slippage: positive means actual > expected
            raw_fractional = (actual_price - expected_price) / expected_price

            # For sells, we want higher actual price = better, so invert
            if side_norm == "sell":
                raw_fractional = -raw_fractional

            # Now positive = unfavourable (paid more on buy / received less on sell)
            slippage_bps = raw_fractional * _BPS_SCALE
            # Dollar cost of slippage (positive = unfavourable cost)
            slippage_usd = abs(actual_price - expected_price) * quantity
            if raw_fractional < 0:
                # Price improvement – record as negative cost
                slippage_usd = -slippage_usd

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        record = FillRecord(
            ticker=ticker,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            side=side_norm,
            order_type=order_type.lower(),
            timestamp=timestamp,
            slippage_bps=slippage_bps,
            slippage_usd=slippage_usd,
            fill_time_seconds=fill_time_seconds,
            filled=True,
            extra=extra or {},
        )
        self.fills.append(record)

        logger.debug(
            "execution_tracker.fill_recorded",
            ticker=ticker,
            side=side_norm,
            order_type=order_type,
            slippage_bps=round(slippage_bps, 2),
            slippage_usd=round(slippage_usd, 4),
        )
        return record

    def record_unfilled_order(
        self,
        ticker: str,
        expected_price: float,
        quantity: int,
        side: str,
        order_type: str,
        timestamp: datetime,
        extra: dict[str, Any] | None = None,
    ) -> FillRecord:
        """Record an order that expired/was cancelled without a fill.

        Used to accurately track limit-order fill rates.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        record = FillRecord(
            ticker=ticker,
            expected_price=expected_price,
            actual_price=0.0,
            quantity=quantity,
            side=side.lower(),
            order_type=order_type.lower(),
            timestamp=timestamp,
            slippage_bps=0.0,
            slippage_usd=0.0,
            fill_time_seconds=None,
            filled=False,
            extra=extra or {},
        )
        self.fills.append(record)
        return record

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> ExecutionMetrics:
        """Compute aggregate execution quality metrics across all fills.

        Returns
        -------
        ExecutionMetrics
        """
        if not self.fills:
            return ExecutionMetrics()

        # Separate filled and unfilled
        filled_records = [f for f in self.fills if f.filled]
        limit_orders = [f for f in self.fills if f.order_type == "limit"]
        filled_limit = [f for f in limit_orders if f.filled]

        # Fill rate for limit orders
        fill_rate = (
            len(filled_limit) / len(limit_orders) if limit_orders else 1.0
        )

        if not filled_records:
            return ExecutionMetrics(
                sample_size=len(self.fills),
                fill_rate=fill_rate,
            )

        slippages = [f.slippage_bps for f in filled_records]
        slippages_arr = np.array(slippages, dtype=float)

        avg_slip = float(np.mean(slippages_arr))
        median_slip = float(np.median(slippages_arr))
        worst_slip = float(np.max(slippages_arr))
        best_slip = float(np.min(slippages_arr))
        stddev_slip = float(np.std(slippages_arr)) if len(slippages_arr) > 1 else 0.0
        p95_slip = float(np.percentile(slippages_arr, 95))

        total_slip_usd = sum(f.slippage_usd for f in filled_records)

        # By order type
        mkt_records = [f for f in filled_records if f.order_type == "market"]
        lmt_records = [f for f in filled_records if f.order_type == "limit"]
        mkt_avg = float(np.mean([f.slippage_bps for f in mkt_records])) if mkt_records else 0.0
        lmt_avg = float(np.mean([f.slippage_bps for f in lmt_records])) if lmt_records else 0.0

        # By side
        buy_records = [f for f in filled_records if f.side == "buy"]
        sell_records = [f for f in filled_records if f.side == "sell"]
        buy_avg = float(np.mean([f.slippage_bps for f in buy_records])) if buy_records else 0.0
        sell_avg = float(np.mean([f.slippage_bps for f in sell_records])) if sell_records else 0.0

        # Fill time (only for records that have it)
        fill_times = [f.fill_time_seconds for f in filled_records if f.fill_time_seconds is not None]
        avg_fill_time = float(statistics.mean(fill_times)) if fill_times else None

        metrics = ExecutionMetrics(
            sample_size=len(filled_records),
            avg_slippage_bps=avg_slip,
            median_slippage_bps=median_slip,
            worst_slippage_bps=worst_slip,
            best_slippage_bps=best_slip,
            stddev_slippage_bps=stddev_slip,
            p95_slippage_bps=p95_slip,
            fill_rate=fill_rate,
            avg_fill_time_seconds=avg_fill_time,
            total_slippage_usd=total_slip_usd,
            market_order_avg_slippage_bps=mkt_avg,
            limit_order_avg_slippage_bps=lmt_avg,
            buy_avg_slippage_bps=buy_avg,
            sell_avg_slippage_bps=sell_avg,
            degraded=self.is_execution_degraded(),
        )

        return metrics

    def get_recent_metrics(self, n: int | None = None) -> ExecutionMetrics:
        """Compute metrics for the most recent *n* fills only.

        Parameters
        ----------
        n:
            Number of recent fills to include.  Defaults to
            ``self._degradation_window``.
        """
        window = n or self._degradation_window
        tracker = ExecutionTracker(
            degradation_window=self._degradation_window,
            degradation_threshold_bps=self._degradation_threshold_bps,
            degradation_min_samples=self._degradation_min_samples,
        )
        tracker.fills = self.fills[-window:] if len(self.fills) > window else list(self.fills)
        return tracker.get_metrics()

    # ------------------------------------------------------------------
    # Degradation detection
    # ------------------------------------------------------------------

    def is_execution_degraded(self) -> bool:
        """Return True if recent execution quality has materially deteriorated.

        Compares the mean slippage of the most recent ``degradation_window``
        filled records against the long-term mean.  If the recent window is
        worse by more than ``degradation_threshold_bps``, returns True.

        Also flags degradation if fill rate for limit orders has dropped
        below 50% in the recent window.
        """
        filled = [f for f in self.fills if f.filled]

        if len(filled) < self._degradation_min_samples:
            return False  # Not enough data for reliable detection

        window = self._degradation_window
        if len(filled) <= window:
            # All fills are "recent"; use vs. a tight threshold
            recent_slip = float(np.mean([f.slippage_bps for f in filled]))
            return recent_slip > self._degradation_threshold_bps * 2

        recent_filled = filled[-window:]
        baseline_filled = filled[:-window]

        recent_mean = float(np.mean([f.slippage_bps for f in recent_filled]))
        baseline_mean = float(np.mean([f.slippage_bps for f in baseline_filled]))

        slippage_degraded = (recent_mean - baseline_mean) > self._degradation_threshold_bps

        # Check fill-rate degradation for limit orders in recent window
        recent_all = self.fills[-window:]
        recent_limits = [f for f in recent_all if f.order_type == "limit"]
        if len(recent_limits) >= 5:
            recent_fill_rate = sum(1 for f in recent_limits if f.filled) / len(recent_limits)
            fill_rate_degraded = recent_fill_rate < 0.50
        else:
            fill_rate_degraded = False

        degraded = slippage_degraded or fill_rate_degraded

        if degraded:
            logger.warning(
                "execution_tracker.degradation_detected",
                recent_mean_bps=round(recent_mean, 2),
                baseline_mean_bps=round(baseline_mean, 2),
                delta_bps=round(recent_mean - baseline_mean, 2),
                threshold_bps=self._degradation_threshold_bps,
                fill_rate_issue=fill_rate_degraded,
            )

        return degraded

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded fills (e.g., at start of a new trading session)."""
        self.fills.clear()

    def summary(self) -> str:
        """Return a one-line human-readable summary of execution quality."""
        m = self.get_metrics()
        status = "DEGRADED" if m.degraded else "OK"
        return (
            f"ExecutionTracker [{status}] "
            f"fills={m.sample_size} "
            f"avg_slip={m.avg_slippage_bps:.1f}bps "
            f"p95={m.p95_slippage_bps:.1f}bps "
            f"fill_rate={m.fill_rate:.1%} "
            f"total_cost=${m.total_slippage_usd:.2f}"
        )

    def __repr__(self) -> str:
        return (
            f"ExecutionTracker("
            f"fills={len(self.fills)}, "
            f"degradation_window={self._degradation_window})"
        )
