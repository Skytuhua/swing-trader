"""Indicator plugin registry.

The IndicatorRegistry is a class-based registry that holds all registered
BaseIndicator subclasses.  Every indicator module decorates its classes
with @IndicatorRegistry.register("name") which inserts the class into the
global _indicators dict.

At runtime, IndicatorRegistry.compute_all(df, config) instantiates and
runs every registered indicator, collecting their results into a flat dict
keyed by indicator name.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

if TYPE_CHECKING:
    pass  # TechnicalConfig imported lazily to avoid circular imports

logger = structlog.get_logger(__name__)


class BaseIndicator(abc.ABC):
    """Abstract base class for all technical indicators.

    Subclasses must:
    1. Decorate the class with ``@IndicatorRegistry.register("slug")``.
    2. Implement the ``compute(df, config)`` method.
    3. Expose ``name`` and ``category`` properties.
    """

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        """Compute the indicator values from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data with columns: open, high, low, close, volume.
            Index: DatetimeIndex (UTC).
        config : TechnicalConfig
            Configuration parameters (periods, thresholds, etc.).

        Returns
        -------
        dict[str, Any]
            Flat dict of computed values. Keys should be descriptive and
            snake_cased.  Missing / uncomputable values should be ``None``.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short slug matching the registry key, e.g. ``'rsi'``."""
        ...

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Category: ``'trend'``, ``'momentum'``, ``'volatility'``,
        ``'volume'``, or ``'structure'``."""
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers available to all indicators                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_last(series: pd.Series) -> float | None:
        """Return the last non-NaN value of a series, or None."""
        cleaned = series.dropna()
        if cleaned.empty:
            return None
        return float(cleaned.iloc[-1])

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert a value to float, returning None on failure."""
        if value is None:
            return None
        try:
            f = float(value)
            import math
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _require_min_bars(df: pd.DataFrame, n: int) -> bool:
        """Return True if df has at least n rows of non-NaN close prices."""
        if df is None or df.empty:
            return False
        return df["close"].dropna().shape[0] >= n


class IndicatorRegistry:
    """Plugin registry for technical indicators.

    Class-level dict maps indicator slug → indicator class.
    Thread-safe for reads (writes only happen at module import time).
    """

    _indicators: dict[str, type[BaseIndicator]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers an indicator under *name*.

        Example
        -------
        @IndicatorRegistry.register("rsi")
        class RSIIndicator(BaseIndicator):
            ...
        """

        def wrapper(indicator_cls: type[BaseIndicator]) -> type[BaseIndicator]:
            if name in cls._indicators:
                logger.warning(
                    "indicator_registry.overwriting",
                    name=name,
                    old=cls._indicators[name].__name__,
                    new=indicator_cls.__name__,
                )
            cls._indicators[name] = indicator_cls
            return indicator_cls

        return wrapper

    @classmethod
    def get(cls, name: str) -> type[BaseIndicator]:
        """Return the indicator class registered under *name*.

        Raises
        ------
        KeyError
            If no indicator with that name is registered.
        """
        if name not in cls._indicators:
            raise KeyError(
                f"Indicator '{name}' is not registered. "
                f"Available: {sorted(cls._indicators.keys())}"
            )
        return cls._indicators[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Return sorted list of all registered indicator names."""
        return sorted(cls._indicators.keys())

    @classmethod
    def list_by_category(cls, category: str) -> list[str]:
        """Return names of all indicators in a given category."""
        return [
            name
            for name, klass in cls._indicators.items()
            if klass().category == category
        ]

    @classmethod
    def compute_all(cls, df: pd.DataFrame, config: Any) -> dict[str, dict | None]:
        """Instantiate and run every registered indicator.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (same as BaseIndicator.compute signature).
        config : TechnicalConfig
            Config passed through to each indicator.

        Returns
        -------
        dict[str, dict | None]
            Maps indicator name → result dict (or None on failure).
        """
        results: dict[str, dict | None] = {}
        for name, indicator_cls in cls._indicators.items():
            try:
                instance = indicator_cls()
                results[name] = instance.compute(df, config)
            except Exception as exc:
                logger.warning(
                    "indicator_registry.compute_failed",
                    indicator=name,
                    error=str(exc),
                    exc_info=True,
                )
                results[name] = None
        return results

    @classmethod
    def reset(cls) -> None:
        """Clear all registered indicators (useful in tests)."""
        cls._indicators.clear()
