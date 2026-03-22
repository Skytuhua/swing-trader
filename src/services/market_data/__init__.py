"""Market data service package.

Provides unified access to market data from multiple providers with
automatic fallback, Redis caching, and data quality scoring.
"""

from src.services.market_data.base import MarketDataProvider, Quote, Bar
from src.services.market_data.manager import DataManager

__all__ = [
    "MarketDataProvider",
    "Quote",
    "Bar",
    "DataManager",
]
