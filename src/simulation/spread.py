"""Bid-ask spread simulation for paper trading.

Provides configurable spread models:
- **fixed**: Constant spread in cents.
- **percentage**: Constant percentage spread.
- **volume_tiered**: Spread based on average daily dollar volume tiers,
  following the Glosten-Milgrom / Kyle model. More liquid stocks have
  tighter spreads.

Buys fill at ask (mid + half_spread), sells fill at bid (mid - half_spread).

Usage::

    from src.simulation.spread import SpreadSimulator
    from src.core.config import SimulationConfig

    sim = SpreadSimulator(SimulationConfig())
    result = sim.calculate(
        mid_price=150.0,
        avg_daily_dollar_volume=5_000_000_000,
        time_of_day_label="mid_morning",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


# Volume-tiered spread ranges: (min_addv, max_addv, min_spread_pct, max_spread_pct)
# These are half-spread percentages (one-side cost).
_SPREAD_TIERS: list[tuple[float, float, float, float]] = [
    (500_000_000.0, float("inf"), 0.0001, 0.0003),  # Mega-liquid (AAPL, MSFT)
    (50_000_000.0,  500_000_000.0, 0.0003, 0.0010),  # Large-cap
    (5_000_000.0,   50_000_000.0,  0.0010, 0.0050),   # Mid-cap
    (0.0,           5_000_000.0,   0.0050, 0.0200),    # Small-cap / illiquid
]

# Time-of-day spread multipliers
_TOD_SPREAD_MULTIPLIERS: dict[str, float] = {
    "open_auction": 2.0,
    "mid_morning": 1.0,
    "midday": 1.2,
    "afternoon": 1.0,
    "close_auction": 1.3,
    "extended_hours": 3.0,
}


@dataclass(frozen=True)
class SpreadResult:
    """Computed bid-ask spread."""

    mid: float          # Mid/reference price
    bid: float          # Simulated bid price
    ask: float          # Simulated ask price
    half_spread: float  # Half-spread in dollars
    spread_pct: float   # Full spread as % of mid price
    model_used: str     # Which spread model was used


class SpreadSimulator:
    """Configurable bid-ask spread simulator.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with spread parameters.
    rng : random.Random | None
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: "random.Random | None" = None,
    ) -> None:
        import random

        self._model = config.spread_model
        self._fixed_cents = config.spread_fixed_cents
        self._fixed_pct = config.spread_fixed_pct / 100.0  # Convert % to fraction
        self._tod_scaling = config.spread_time_of_day_scaling
        self._rng = rng or random.Random()

    def calculate(
        self,
        mid_price: float,
        avg_daily_dollar_volume: float = 0.0,
        time_of_day_label: str = "mid_morning",
    ) -> SpreadResult:
        """Calculate the bid-ask spread for a given instrument.

        Parameters
        ----------
        mid_price : float
            Current mid/reference price.
        avg_daily_dollar_volume : float
            Average daily dollar volume (price × daily shares).
        time_of_day_label : str
            Intraday period label for TOD scaling.

        Returns
        -------
        SpreadResult
            Computed bid, ask, and spread metrics.
        """
        if mid_price <= 0:
            return SpreadResult(
                mid=mid_price, bid=mid_price, ask=mid_price,
                half_spread=0.0, spread_pct=0.0, model_used=self._model,
            )

        if self._model == "fixed":
            half_spread = self._fixed_cents / 100.0  # cents to dollars
        elif self._model == "percentage":
            half_spread = mid_price * self._fixed_pct
        elif self._model == "volume_tiered":
            half_spread = self._volume_tiered_spread(mid_price, avg_daily_dollar_volume)
        else:
            half_spread = mid_price * self._fixed_pct

        # Apply time-of-day scaling
        if self._tod_scaling:
            tod_mult = _TOD_SPREAD_MULTIPLIERS.get(time_of_day_label, 1.0)
            half_spread *= tod_mult

        bid = round(mid_price - half_spread, 4)
        ask = round(mid_price + half_spread, 4)
        spread_pct = (half_spread * 2 / mid_price * 100.0) if mid_price > 0 else 0.0

        return SpreadResult(
            mid=mid_price,
            bid=bid,
            ask=ask,
            half_spread=round(half_spread, 6),
            spread_pct=round(spread_pct, 6),
            model_used=self._model,
        )

    def _volume_tiered_spread(
        self, mid_price: float, addv: float
    ) -> float:
        """Calculate spread from volume tier with randomization."""
        if addv <= 0:
            addv = 100_000_000.0  # Default to mid-cap

        min_pct, max_pct = self._get_tier(addv)
        raw_pct = self._rng.uniform(min_pct, max_pct)
        return mid_price * raw_pct

    @staticmethod
    def _get_tier(addv: float) -> tuple[float, float]:
        """Find the spread tier for a given ADDV."""
        for min_addv, max_addv, min_pct, max_pct in _SPREAD_TIERS:
            if min_addv <= addv < max_addv:
                return min_pct, max_pct
        return _SPREAD_TIERS[-1][2], _SPREAD_TIERS[-1][3]
