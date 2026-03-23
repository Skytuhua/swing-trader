"""Order execution latency simulation for paper trading.

Simulates the delay between signal generation and order execution.
During this delay, the price may drift adversely, making the actual
fill price worse than the price at signal time.

Usage::

    from src.simulation.latency import LatencySimulator
    from src.core.config import SimulationConfig

    sim = LatencySimulator(SimulationConfig())
    result = sim.simulate(
        signal_price=150.0,
        atr_pct=0.02,
        side="buy",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


@dataclass(frozen=True)
class LatencyResult:
    """Result of latency simulation."""

    delay_ms: float         # Simulated delay in milliseconds
    price_drift_pct: float  # Price drift during delay (as fraction)
    adjusted_price: float   # Price after drift (signal_price * (1 + drift))


class LatencySimulator:
    """Simulates execution latency and resulting price drift.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with latency parameters.
    rng : random.Random | None
        RNG for reproducibility.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: "random.Random | None" = None,
    ) -> None:
        import random

        self._enabled = config.latency_enabled
        self._base_ms = config.latency_base_ms
        self._jitter_ms = config.latency_jitter_ms
        self._drift_bps_per_100ms = config.latency_price_drift_bps
        self._rng = rng or random.Random()

    def simulate(
        self,
        signal_price: float,
        atr_pct: float = 0.015,
        side: str = "buy",
    ) -> LatencyResult:
        """Simulate execution latency and compute the drifted fill price.

        Parameters
        ----------
        signal_price : float
            Price at the time the signal was generated.
        atr_pct : float
            ATR as a fraction of price (scales drift magnitude).
        side : str
            "buy" or "sell" — drift is adverse to the trader.

        Returns
        -------
        LatencyResult
            Latency delay and the adjusted price.
        """
        if not self._enabled or signal_price <= 0:
            return LatencyResult(
                delay_ms=0.0,
                price_drift_pct=0.0,
                adjusted_price=signal_price,
            )

        # Compute delay
        delay_ms = self._base_ms + self._rng.uniform(0, self._jitter_ms)

        # Compute price drift during delay
        # drift_bps = (delay_ms / 100) * drift_bps_per_100ms * vol_scaling
        periods_of_100ms = delay_ms / 100.0
        vol_scaling = max(0.5, atr_pct / 0.015)  # Normalize to 1.5% ATR baseline

        # Drift direction: random but biased adverse
        # ~60% chance of adverse drift, ~40% favorable
        drift_sign = 1.0 if self._rng.random() < 0.6 else -1.0
        if side == "sell":
            drift_sign *= -1.0  # For sells, adverse = price drops

        drift_bps = (
            periods_of_100ms
            * self._drift_bps_per_100ms
            * vol_scaling
            * drift_sign
            * self._rng.uniform(0.5, 1.5)  # Random magnitude
        )

        drift_pct = drift_bps / 10_000.0

        # For buys, adverse drift = price goes up
        # For sells, adverse drift = price goes down
        adjusted_price = signal_price * (1.0 + drift_pct)

        return LatencyResult(
            delay_ms=round(delay_ms, 1),
            price_drift_pct=round(drift_pct, 6),
            adjusted_price=round(adjusted_price, 4),
        )
