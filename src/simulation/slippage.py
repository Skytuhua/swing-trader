"""Configurable slippage models for paper trading simulation.

Provides four slippage models:
- **Fixed**: Constant percentage slippage per trade.
- **Volume-weighted**: Square-root market impact — larger orders relative to
  average daily volume incur more slippage (Almgren et al. 2005).
- **Volatility-adjusted**: Slippage scaled by ATR (average true range) to
  account for market turbulence.
- **Composite**: Combines volume-weighted and volatility-adjusted models.

Usage::

    from src.simulation.slippage import SlippageModel
    from src.core.config import SimulationConfig

    model = SlippageModel(SimulationConfig())
    slippage_pct = model.calculate(
        order_dollar_value=50_000,
        avg_daily_dollar_volume=1_000_000_000,
        atr_pct=0.02,
        side="buy",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


@dataclass(frozen=True)
class SlippageResult:
    """Result of a slippage calculation."""

    slippage_pct: float   # Slippage as a fraction of price (e.g. 0.001 = 0.1%)
    slippage_bps: float   # Slippage in basis points
    model_used: str       # Which model produced this result


class SlippageModel:
    """Configurable slippage calculator.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with slippage parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._model = config.slippage_model
        self._fixed_pct = config.slippage_fixed_pct / 100.0  # Convert from % to fraction
        self._base_bps = config.slippage_base_bps
        self._impact_exp = config.slippage_impact_exponent
        self._vol_weight = config.slippage_volatility_weight
        self._max_pct = config.slippage_max_pct / 100.0  # Convert from % to fraction

    def calculate(
        self,
        order_dollar_value: float,
        avg_daily_dollar_volume: float = 0.0,
        atr_pct: float = 0.0,
        side: str = "buy",
    ) -> SlippageResult:
        """Calculate slippage for an order.

        Parameters
        ----------
        order_dollar_value : float
            Dollar value of the order (price × shares).
        avg_daily_dollar_volume : float
            Average daily dollar volume for the instrument.
        atr_pct : float
            ATR as a fraction of price (e.g. 0.02 for 2%).
        side : str
            "buy" or "sell" — slippage is always adverse.

        Returns
        -------
        SlippageResult
            The computed slippage.
        """
        if self._model == "fixed":
            pct = self._fixed_slippage()
        elif self._model == "volume_weighted":
            pct = self._volume_weighted_slippage(order_dollar_value, avg_daily_dollar_volume)
        elif self._model == "volatility_adjusted":
            pct = self._volatility_adjusted_slippage(
                order_dollar_value, avg_daily_dollar_volume, atr_pct
            )
        elif self._model == "composite":
            pct = self._composite_slippage(
                order_dollar_value, avg_daily_dollar_volume, atr_pct
            )
        else:
            pct = self._fixed_slippage()

        # Cap at maximum
        pct = min(pct, self._max_pct)

        return SlippageResult(
            slippage_pct=pct,
            slippage_bps=pct * 10_000,
            model_used=self._model,
        )

    def _fixed_slippage(self) -> float:
        """Constant percentage slippage."""
        return self._fixed_pct

    def _volume_weighted_slippage(
        self, order_value: float, addv: float
    ) -> float:
        """Square-root market impact model.

        impact = base_coeff * (order_value / addv) ^ exponent

        Calibrated so that:
        - 0.1% of ADDV → ~5 bps impact
        - 1% of ADDV   → ~16 bps (with exponent=0.5)
        - 10% of ADDV  → ~50 bps
        """
        if addv <= 0 or order_value <= 0:
            return self._fixed_pct

        participation = order_value / addv
        base_coeff = self._base_bps / 10_000.0  # Convert bps to fraction
        impact = base_coeff * math.pow(participation, self._impact_exp)
        return impact

    def _volatility_adjusted_slippage(
        self, order_value: float, addv: float, atr_pct: float
    ) -> float:
        """Slippage scaled by volatility (ATR).

        Higher ATR = wider price swings = more likely adverse fills.
        Base slippage is scaled by (atr_pct / baseline_atr_pct).
        """
        baseline_atr = 0.015  # 1.5% daily ATR is "normal"
        vol_mult = 1.0 + max(0.0, (atr_pct / baseline_atr) - 1.0) * self._vol_weight

        base_slip = self._volume_weighted_slippage(order_value, addv)
        return base_slip * vol_mult

    def _composite_slippage(
        self, order_value: float, addv: float, atr_pct: float
    ) -> float:
        """Combined volume + volatility model.

        composite = volume_weighted * (1 + vol_adjustment)
        """
        return self._volatility_adjusted_slippage(order_value, addv, atr_pct)
