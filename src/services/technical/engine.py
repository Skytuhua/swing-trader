"""Technical Analysis Engine.

TechnicalEngine.analyze() is the main entry point. It:
1. Calls IndicatorRegistry.compute_all() to run all registered indicators.
2. Scores each sub-category (trend, momentum, volatility, volume, structure) on 0-100.
3. Computes a weighted composite score.
4. Produces a TechnicalProfile dataclass with all scores, raw indicator values,
   setup classification, and key levels.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.services.technical.registry import IndicatorRegistry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# TechnicalProfile dataclass
# ---------------------------------------------------------------------------


@dataclass
class TechnicalProfile:
    """Complete technical analysis profile for a single ticker."""

    ticker: str
    analyzed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Sub-scores (0–100)
    trend_score: float = 50.0
    momentum_score: float = 50.0
    volatility_score: float = 50.0
    volume_score: float = 50.0
    structure_score: float = 50.0

    # Weighted composite (0–100)
    composite_score: float = 50.0

    # Raw indicator outputs
    indicators: dict[str, Any] = field(default_factory=dict)

    # Derived fields
    setup_type: str = "neutral"          # e.g. 'breakout', 'pullback', 'momentum_surge'
    key_levels: dict[str, float] = field(default_factory=dict)  # support, resistance, vwap

    # Metadata
    bar_count: int = 0
    data_quality_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "analyzed_at": self.analyzed_at.isoformat(),
            "trend_score": self.trend_score,
            "momentum_score": self.momentum_score,
            "volatility_score": self.volatility_score,
            "volume_score": self.volume_score,
            "structure_score": self.structure_score,
            "composite_score": self.composite_score,
            "setup_type": self.setup_type,
            "key_levels": self.key_levels,
            "bar_count": self.bar_count,
            "data_quality_note": self.data_quality_note,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TechnicalEngine:
    """Compute all technical indicators and produce a TechnicalProfile.

    Parameters
    ----------
    registry : IndicatorRegistry
        The populated registry (class, not instance).
    config : TechnicalConfig
        Configuration object.
    """

    # Composite score weights (must sum to 1.0)
    _WEIGHTS = {
        "trend": 0.30,
        "momentum": 0.25,
        "volume": 0.20,
        "structure": 0.15,
        "volatility": 0.10,
    }

    def __init__(self, registry: type[IndicatorRegistry], config: Any) -> None:
        self.registry = registry
        self.config = config

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def analyze(
        self,
        ticker: str,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame | None = None,
        benchmark_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
    ) -> TechnicalProfile:
        """Compute all indicators and produce a TechnicalProfile.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        daily_df : pd.DataFrame
            Daily OHLCV.
        intraday_df : pd.DataFrame | None
            Intraday OHLCV (optional, used by some indicators).
        benchmark_df : pd.DataFrame | None
            SPY daily bars for relative strength computation.
        sector_df : pd.DataFrame | None
            Sector ETF bars for relative strength computation.
        """
        if daily_df is None or daily_df.empty:
            logger.warning("technical_engine.empty_dataframe", ticker=ticker)
            return TechnicalProfile(
                ticker=ticker,
                data_quality_note="empty_dataframe",
            )

        # Attach benchmark / sector refs to the dataframe as metadata so
        # indicators can access them without changing the ABC signature.
        df = daily_df.copy()
        if benchmark_df is not None:
            df.attrs["benchmark_df"] = benchmark_df
        if sector_df is not None:
            df.attrs["sector_df"] = sector_df
        if intraday_df is not None:
            df.attrs["intraday_df"] = intraday_df

        # Run all indicators (CPU-bound, wrap in executor for async context)
        loop = asyncio.get_event_loop()
        indicators = await loop.run_in_executor(
            None, self.registry.compute_all, df, self.config
        )

        # Compute sub-scores
        trend_score = self._score_trend(indicators)
        momentum_score = self._score_momentum(indicators)
        volatility_score = self._score_volatility(indicators)
        volume_score = self._score_volume(indicators)
        structure_score = self._score_structure(indicators)

        # Weighted composite
        composite = (
            trend_score * self._WEIGHTS["trend"]
            + momentum_score * self._WEIGHTS["momentum"]
            + volume_score * self._WEIGHTS["volume"]
            + structure_score * self._WEIGHTS["structure"]
            + volatility_score * self._WEIGHTS["volatility"]
        )

        logger.info(
            "technical_engine.analysis_complete",
            ticker=ticker,
            composite=round(composite, 1),
            trend=round(trend_score, 1),
            momentum=round(momentum_score, 1),
            volume=round(volume_score, 1),
            structure=round(structure_score, 1),
            volatility=round(volatility_score, 1),
        )

        return TechnicalProfile(
            ticker=ticker,
            trend_score=round(trend_score, 2),
            momentum_score=round(momentum_score, 2),
            volatility_score=round(volatility_score, 2),
            volume_score=round(volume_score, 2),
            structure_score=round(structure_score, 2),
            composite_score=round(composite, 2),
            indicators=indicators,
            setup_type=self._classify_setup(indicators),
            key_levels=self._extract_key_levels(indicators),
            bar_count=len(daily_df),
        )

    # ------------------------------------------------------------------ #
    # Sub-category scorers                                                 #
    # ------------------------------------------------------------------ #

    def _score_trend(self, indicators: dict[str, Any]) -> float:
        """Score trend quality on 0-100."""
        score = 50.0
        points: list[float] = []

        sma = (indicators.get("sma") or {})
        ema = (indicators.get("ema") or {})
        regime = (indicators.get("trend_regime") or {})

        # 1. Regime classification
        regime_label = regime.get("regime", "sideways")
        if regime_label == "uptrend":
            points.append(80.0)
        elif regime_label == "sideways":
            points.append(50.0)
        elif regime_label == "downtrend":
            points.append(20.0)

        # 2. Price vs key MAs
        for key in ("above_sma_20", "above_sma_50", "above_sma_200"):
            val = sma.get(key)
            if val is True:
                points.append(70.0)
            elif val is False:
                points.append(30.0)

        # 3. MA alignment (short > medium > long)
        alignment = sma.get("ma_alignment") or regime.get("ma_alignment")
        if alignment == "bullish":
            points.append(85.0)
        elif alignment == "bearish":
            points.append(15.0)
        elif alignment == "neutral":
            points.append(50.0)

        # 4. Slope direction
        slope_50 = sma.get("slope_50") or ema.get("slope_50")
        if slope_50 is not None:
            if slope_50 > 0:
                points.append(min(50 + slope_50 * 500, 90.0))
            else:
                points.append(max(50 + slope_50 * 500, 10.0))

        # 5. Price relative to 200 SMA
        dist_200 = sma.get("dist_from_sma_200_pct")
        if dist_200 is not None:
            if 0 < dist_200 < 15:
                points.append(72.0)  # above but not extended
            elif dist_200 >= 15:
                points.append(55.0)  # extended above
            else:
                points.append(28.0)  # below 200

        return _clamp(_avg(points, default=50.0))

    def _score_momentum(self, indicators: dict[str, Any]) -> float:
        """Score momentum quality on 0-100."""
        points: list[float] = []

        rsi = (indicators.get("rsi") or {})
        macd = (indicators.get("macd") or {})
        stoch = (indicators.get("stochastic") or {})
        adx = (indicators.get("adx") or {})
        rs = (indicators.get("relative_strength") or {})

        # RSI (14) – ideal zone 50-70 for longs
        rsi_val = rsi.get("rsi_14")
        if rsi_val is not None:
            if 50 <= rsi_val <= 70:
                points.append(75.0)
            elif 40 <= rsi_val < 50:
                points.append(55.0)
            elif rsi_val > 70:
                points.append(45.0)  # overbought
            elif 30 <= rsi_val < 40:
                points.append(40.0)
            else:
                points.append(20.0)  # oversold

        # MACD histogram direction
        macd_hist = macd.get("histogram")
        macd_bullish_cross = macd.get("bullish_crossover")
        if macd_hist is not None:
            if macd_hist > 0:
                points.append(68.0)
            else:
                points.append(35.0)
        if macd_bullish_cross:
            points.append(80.0)

        # Stochastic
        stoch_k = stoch.get("stoch_k")
        stoch_d = stoch.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if 40 <= stoch_k <= 80 and stoch_k > stoch_d:
                points.append(70.0)
            elif stoch_k < 20:
                points.append(30.0)
            elif stoch_k > 80:
                points.append(45.0)
            else:
                points.append(50.0)

        # ADX (trend strength)
        adx_val = adx.get("adx")
        plus_di = adx.get("plus_di")
        minus_di = adx.get("minus_di")
        if adx_val is not None:
            if adx_val > 25 and (plus_di or 0) > (minus_di or 0):
                points.append(78.0)
            elif adx_val > 25:
                points.append(30.0)
            else:
                points.append(50.0)

        # Relative strength vs SPY
        rs_spy = rs.get("rs_vs_spy")
        if rs_spy is not None:
            if rs_spy > 1.05:
                points.append(75.0)
            elif rs_spy > 1.0:
                points.append(60.0)
            else:
                points.append(35.0)

        return _clamp(_avg(points, default=50.0))

    def _score_volatility(self, indicators: dict[str, Any]) -> float:
        """Score volatility quality on 0-100.

        For swing trading we want moderate volatility (not too quiet, not
        too explosive). Bollinger Band squeeze + low ATR% → higher score.
        """
        points: list[float] = []

        atr = (indicators.get("atr") or {})
        bb = (indicators.get("bollinger_bands") or {})
        gap = (indicators.get("gaps") or {})

        # ATR % of price – sweet spot 1-4%
        atr_pct = atr.get("atr_pct")
        if atr_pct is not None:
            if 1.0 <= atr_pct <= 4.0:
                points.append(75.0)
            elif 0.5 <= atr_pct < 1.0:
                points.append(60.0)
            elif 4.0 < atr_pct <= 8.0:
                points.append(50.0)
            else:
                points.append(30.0)

        # Bollinger Band width (squeeze = potential energy)
        bb_width = bb.get("bandwidth_pct")
        if bb_width is not None:
            if bb_width < 8:
                points.append(80.0)  # squeeze
            elif bb_width < 15:
                points.append(65.0)
            elif bb_width < 30:
                points.append(50.0)
            else:
                points.append(30.0)  # very wide = high risk

        # %B position: prefer middle of band (not at extremes)
        pct_b = bb.get("percent_b")
        if pct_b is not None:
            if 0.3 <= pct_b <= 0.7:
                points.append(65.0)
            elif 0.0 <= pct_b < 0.3:
                points.append(70.0)  # near lower band – potential reversal
            else:
                points.append(45.0)  # near upper band

        # Gap risk
        gap_present = gap.get("gap_present", False)
        if gap_present:
            gap_pct = abs(gap.get("gap_magnitude_pct", 0) or 0)
            if gap_pct > 5:
                points.append(20.0)
            else:
                points.append(45.0)
        else:
            points.append(65.0)

        return _clamp(_avg(points, default=50.0))

    def _score_volume(self, indicators: dict[str, Any]) -> float:
        """Score volume quality on 0-100."""
        points: list[float] = []

        vol_trend = (indicators.get("volume_trend") or {})
        vol_spike = (indicators.get("volume_spike") or {})
        vwap = (indicators.get("vwap") or {})

        # Relative volume (>1 is increasing participation)
        rel_vol = vol_trend.get("relative_volume")
        if rel_vol is not None:
            if rel_vol >= 2.0:
                points.append(85.0)
            elif rel_vol >= 1.3:
                points.append(72.0)
            elif rel_vol >= 0.8:
                points.append(55.0)
            else:
                points.append(35.0)

        # Volume trend direction
        trend_dir = vol_trend.get("volume_trend_direction")
        if trend_dir == "increasing":
            points.append(70.0)
        elif trend_dir == "decreasing":
            points.append(35.0)

        # Volume spike presence
        spike = vol_spike.get("spike_detected")
        if spike:
            points.append(78.0)

        # Distance from VWAP
        vwap_dist = vwap.get("distance_from_vwap_pct")
        if vwap_dist is not None:
            # Being near or above VWAP is a positive sign
            if -2 <= vwap_dist <= 3:
                points.append(65.0)
            elif vwap_dist > 3:
                points.append(60.0)
            else:
                points.append(40.0)  # below VWAP

        return _clamp(_avg(points, default=50.0))

    def _score_structure(self, indicators: dict[str, Any]) -> float:
        """Score price structure quality on 0-100."""
        points: list[float] = []

        sr = (indicators.get("support_resistance") or {})
        breakout = (indicators.get("breakout") or {})
        pullback = (indicators.get("pullback_quality") or {})
        consol = (indicators.get("consolidation") or {})
        candles = (indicators.get("candle_patterns") or {})

        # Breakout detection
        if breakout.get("breakout_detected"):
            if breakout.get("volume_confirmed"):
                points.append(90.0)
            else:
                points.append(68.0)

        # Near support (better entry risk/reward)
        near_support = sr.get("near_support")
        if near_support:
            points.append(75.0)
        else:
            points.append(50.0)

        # Pullback quality
        pullback_quality = pullback.get("quality")
        if pullback_quality == "orderly":
            points.append(78.0)
        elif pullback_quality == "chaotic":
            points.append(30.0)
        elif pullback_quality == "moderate":
            points.append(55.0)

        # Consolidation (potential energy building)
        if consol.get("in_consolidation"):
            duration = consol.get("duration_bars", 0)
            if duration >= 5:
                points.append(72.0)
            else:
                points.append(55.0)

        # Bullish candle patterns
        patterns = candles.get("patterns_detected", [])
        bullish_patterns = {"hammer", "bullish_engulfing", "morning_doji_star"}
        bearish_patterns = {"shooting_star", "bearish_engulfing"}
        if any(p in bullish_patterns for p in patterns):
            points.append(75.0)
        if any(p in bearish_patterns for p in patterns):
            points.append(25.0)

        return _clamp(_avg(points, default=50.0))

    # ------------------------------------------------------------------ #
    # Setup classification                                                 #
    # ------------------------------------------------------------------ #

    def _classify_setup(self, indicators: dict[str, Any]) -> str:
        """Classify the dominant setup type from indicator outputs."""
        breakout = (indicators.get("breakout") or {})
        pullback = (indicators.get("pullback_quality") or {})
        consol = (indicators.get("consolidation") or {})
        rsi = (indicators.get("rsi") or {})
        macd = (indicators.get("macd") or {})
        vol_spike = (indicators.get("volume_spike") or {})
        regime = (indicators.get("trend_regime") or {})

        if breakout.get("breakout_detected") and breakout.get("volume_confirmed"):
            return "high_volume_breakout"

        if breakout.get("breakout_detected"):
            return "breakout"

        if consol.get("in_consolidation") and (consol.get("duration_bars", 0) >= 5):
            return "consolidation_setup"

        rsi_val = rsi.get("rsi_14")
        if rsi_val is not None and 45 <= rsi_val <= 65 and macd.get("histogram", 0) > 0:
            if pullback.get("quality") == "orderly":
                return "healthy_pullback"

        if vol_spike.get("spike_detected"):
            return "volume_surge"

        if regime.get("regime") == "uptrend":
            if rsi_val and rsi_val > 55:
                return "momentum_continuation"

        return "neutral"

    # ------------------------------------------------------------------ #
    # Key levels extraction                                                #
    # ------------------------------------------------------------------ #

    def _extract_key_levels(
        self, indicators: dict[str, Any]
    ) -> dict[str, float]:
        """Extract price levels for stop placement and target setting."""
        levels: dict[str, float] = {}

        sr = (indicators.get("support_resistance") or {})
        vwap = (indicators.get("vwap") or {})
        sma = (indicators.get("sma") or {})
        bb = (indicators.get("bollinger_bands") or {})
        atr = (indicators.get("atr") or {})

        # Support / resistance
        nearest_support = sr.get("nearest_support")
        nearest_resistance = sr.get("nearest_resistance")
        if nearest_support:
            levels["support"] = nearest_support
        if nearest_resistance:
            levels["resistance"] = nearest_resistance

        # VWAP
        vwap_val = vwap.get("vwap")
        if vwap_val:
            levels["vwap"] = vwap_val

        # SMAs as dynamic S/R
        for period in (20, 50, 200):
            key = f"sma_{period}"
            val = sma.get(key)
            if val:
                levels[key] = val

        # Bollinger Bands
        for band in ("upper_band", "lower_band", "middle_band"):
            val = bb.get(band)
            if val:
                levels[f"bb_{band}"] = val

        # ATR value (useful for stop sizing)
        atr_val = atr.get("atr_14")
        if atr_val:
            levels["atr_14"] = atr_val

        return levels


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _avg(values: list[float], default: float = 50.0) -> float:
    """Mean of a list, returning *default* if list is empty."""
    if not values:
        return default
    return sum(values) / len(values)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))
