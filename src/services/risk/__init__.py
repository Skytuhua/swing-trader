"""Risk management: pre-trade checks, kill switch, guards, market calendar."""

from .engine import RiskEngine, RiskCheckResult, PortfolioState
from .kill_switch import KillSwitch
from .guards import (
    MaxSlippageGuard,
    MaxSpreadGuard,
    MaxDrawdownGuard,
    DailyLossGuard,
    StaleDataGuard,
    DuplicateSignalGuard,
)
from .calendar import MarketCalendar

__all__ = [
    "RiskEngine",
    "RiskCheckResult",
    "PortfolioState",
    "KillSwitch",
    "MaxSlippageGuard",
    "MaxSpreadGuard",
    "MaxDrawdownGuard",
    "DailyLossGuard",
    "StaleDataGuard",
    "DuplicateSignalGuard",
    "MarketCalendar",
]
