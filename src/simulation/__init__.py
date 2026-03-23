"""Simulation accuracy models for paper trading and backtesting.

This package provides configurable, research-backed models for:
- Slippage (fixed, volume-weighted, volatility-adjusted, composite)
- Fees and commissions (per-share, flat, percentage, maker/taker)
- Bid-ask spread simulation (fixed, percentage, volume-tiered)
- Partial fill simulation (volume-based)
- Execution latency simulation
- Overnight gap handling

All models are configurable via SimulationConfig in the config system.
"""

from src.simulation.fees import FeeCalculator
from src.simulation.gaps import GapHandler
from src.simulation.latency import LatencySimulator
from src.simulation.partial_fills import PartialFillSimulator
from src.simulation.slippage import SlippageModel
from src.simulation.spread import SpreadSimulator

__all__ = [
    "SlippageModel",
    "FeeCalculator",
    "SpreadSimulator",
    "PartialFillSimulator",
    "LatencySimulator",
    "GapHandler",
]
