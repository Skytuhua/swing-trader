"""Technical analysis service package.

Provides a plugin-based indicator registry, a TechnicalEngine that
computes all indicators and produces a TechnicalProfile, and five
indicator categories: trend, momentum, volatility, volume, structure.

Usage
-----
    from src.services.technical import TechnicalEngine, IndicatorRegistry
    from src.services.technical.indicators import register_all

    register_all()
    engine = TechnicalEngine(IndicatorRegistry, config)
    profile = await engine.analyze("AAPL", daily_df)
"""

from src.services.technical.registry import BaseIndicator, IndicatorRegistry
from src.services.technical.engine import TechnicalEngine

__all__ = [
    "BaseIndicator",
    "IndicatorRegistry",
    "TechnicalEngine",
]
