"""Partial fill simulation for paper trading.

When an order's share count exceeds a configurable threshold of average daily
volume, the order is partially filled. The fill rate decreases as order size
increases relative to market liquidity.

Unfilled portions are tracked and can be retried on subsequent bars.

Usage::

    from src.simulation.partial_fills import PartialFillSimulator
    from src.core.config import SimulationConfig

    sim = PartialFillSimulator(SimulationConfig())
    result = sim.calculate(
        requested_qty=10_000,
        mid_price=150.0,
        avg_daily_dollar_volume=500_000_000,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


@dataclass(frozen=True)
class PartialFillResult:
    """Result of a partial fill calculation."""

    filled_qty: int         # Shares actually filled
    unfilled_qty: int       # Shares remaining unfilled
    fill_rate: float        # Fraction of order that was filled (0-1)
    is_partial: bool        # True if not fully filled
    reason: str             # Why partial (or "full_fill")


class PartialFillSimulator:
    """Simulates partial fills based on order size vs market volume.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with partial fill parameters.
    rng : random.Random | None
        RNG for reproducibility.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: "random.Random | None" = None,
    ) -> None:
        import random

        self._enabled = config.partial_fill_enabled
        self._threshold_pct = config.partial_fill_volume_threshold_pct / 100.0
        self._min_rate = config.partial_fill_min_rate
        self._randomness = config.partial_fill_randomness
        self._rng = rng or random.Random()

    def calculate(
        self,
        requested_qty: int,
        mid_price: float,
        avg_daily_dollar_volume: float,
    ) -> PartialFillResult:
        """Determine how many shares can be filled.

        Parameters
        ----------
        requested_qty : int
            Number of shares requested.
        mid_price : float
            Current mid price for the instrument.
        avg_daily_dollar_volume : float
            Average daily dollar volume.

        Returns
        -------
        PartialFillResult
            Fill calculation result.
        """
        if not self._enabled or requested_qty <= 0 or mid_price <= 0:
            return PartialFillResult(
                filled_qty=requested_qty,
                unfilled_qty=0,
                fill_rate=1.0,
                is_partial=False,
                reason="full_fill",
            )

        if avg_daily_dollar_volume <= 0:
            avg_daily_dollar_volume = 100_000_000.0  # Default

        # Estimate average daily share volume
        adv_shares = avg_daily_dollar_volume / mid_price
        threshold_shares = adv_shares * self._threshold_pct

        if requested_qty <= threshold_shares:
            return PartialFillResult(
                filled_qty=requested_qty,
                unfilled_qty=0,
                fill_rate=1.0,
                is_partial=False,
                reason="full_fill",
            )

        # Fill rate decreases as order exceeds threshold
        # fill_rate = threshold_shares / requested_qty
        raw_rate = threshold_shares / requested_qty
        raw_rate = max(self._min_rate, min(1.0, raw_rate))

        # Add randomness
        jitter = self._rng.uniform(1.0 - self._randomness, 1.0 + self._randomness)
        actual_rate = max(self._min_rate, min(1.0, raw_rate * jitter))

        filled = max(1, int(requested_qty * actual_rate))
        unfilled = requested_qty - filled

        return PartialFillResult(
            filled_qty=filled,
            unfilled_qty=unfilled,
            fill_rate=round(actual_rate, 4),
            is_partial=filled < requested_qty,
            reason="volume_constraint" if filled < requested_qty else "full_fill",
        )
