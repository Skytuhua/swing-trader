"""Overnight gap handling for paper trading simulation.

When a stock gaps through a stop-loss level at the market open, the stop
should fill at the open price (not the stop price), reflecting the reality
that stop-loss orders cannot fill between the stop level and the gap open.

Usage::

    from src.simulation.gaps import GapHandler
    from src.core.config import SimulationConfig

    handler = GapHandler(SimulationConfig())
    result = handler.check_gap_fill(
        stop_price=145.0,
        open_price=142.0,
        previous_close=148.0,
        side="sell",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


@dataclass(frozen=True)
class GapFillResult:
    """Result of a gap check on a stop order."""

    is_gapped: bool         # True if price gapped through the stop
    fill_price: float       # Actual fill price (open_price if gapped, stop_price if not)
    gap_pct: float          # Gap magnitude as % of previous close
    slippage_from_stop: float  # Additional adverse price movement from stop level


class GapHandler:
    """Handles overnight gap scenarios for stop-loss orders.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with gap handling parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._enabled = config.gap_handling_enabled
        self._fill_at_open = config.gap_stop_fill_at_open

    def check_gap_fill(
        self,
        stop_price: float,
        open_price: float,
        previous_close: float,
        side: str = "sell",
    ) -> GapFillResult:
        """Check if a stop order should fill at a gap-open price.

        Parameters
        ----------
        stop_price : float
            The stop-loss trigger price.
        open_price : float
            Today's opening price.
        previous_close : float
            Yesterday's closing price.
        side : str
            "sell" for long-position stop-loss, "buy" for short-cover stop.

        Returns
        -------
        GapFillResult
            Whether a gap occurred and the resulting fill price.
        """
        if not self._enabled or not self._fill_at_open:
            return GapFillResult(
                is_gapped=False,
                fill_price=stop_price,
                gap_pct=0.0,
                slippage_from_stop=0.0,
            )

        gap_pct = 0.0
        if previous_close > 0:
            gap_pct = (open_price - previous_close) / previous_close * 100.0

        is_gapped = False
        fill_price = stop_price
        slippage = 0.0

        if side == "sell":
            # Long position stop-loss: gapped down through stop
            if open_price < stop_price:
                is_gapped = True
                fill_price = open_price
                slippage = stop_price - open_price
        elif side == "buy":
            # Short-cover stop: gapped up through stop
            if open_price > stop_price:
                is_gapped = True
                fill_price = open_price
                slippage = open_price - stop_price

        return GapFillResult(
            is_gapped=is_gapped,
            fill_price=round(fill_price, 4),
            gap_pct=round(gap_pct, 3),
            slippage_from_stop=round(slippage, 4),
        )
