"""Backtesting framework for the SwingTrader autonomous trading bot."""

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.reporter import BacktestReporter
from src.backtest.simulator import MarketSimulator
from src.backtest.walk_forward import WalkForwardValidator

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "MarketSimulator",
    "BacktestReporter",
    "WalkForwardValidator",
]
