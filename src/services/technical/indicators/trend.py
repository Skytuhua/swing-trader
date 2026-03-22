"""Trend indicators: SMA, EMA, TrendRegime.

Registered indicators:
  - sma: SMAIndicator (SMA 9, 20, 50, 200; slopes; price-relative-to-MA)
  - ema: EMAIndicator (EMA 9, 20, 50; slopes)
  - trend_regime: TrendRegimeIndicator (uptrend/downtrend/sideways classification)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # optional acceleration
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False

import structlog

from src.services.technical.registry import BaseIndicator, IndicatorRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: compute linear slope over the last N bars, normalised by price
# ---------------------------------------------------------------------------

def _slope(series: pd.Series, window: int = 10) -> float | None:
    """Return the linear regression slope of the last *window* values, scaled by mean."""
    s = series.dropna()
    if len(s) < window:
        return None
    y = s.values[-window:]
    x = np.arange(window, dtype=float)
    # OLS slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope_raw = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    # Normalise by mean price so slopes are comparable across tickers
    return slope_raw / y_mean if y_mean != 0 else slope_raw


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=max(1, period // 2)).mean()


def _ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=max(1, period // 2)).mean()


# ---------------------------------------------------------------------------
# SMAIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("sma")
class SMAIndicator(BaseIndicator):
    """Simple Moving Averages: 9, 20, 50, 200-period.

    Outputs
    -------
    sma_9, sma_20, sma_50, sma_200 : float | None
        Current (last bar) SMA value.
    slope_9, slope_20, slope_50, slope_200 : float | None
        Normalised linear slope over the last 10 bars.
    above_sma_9, above_sma_20, above_sma_50, above_sma_200 : bool | None
        Whether the last close is above that SMA.
    dist_from_sma_20_pct, dist_from_sma_50_pct, dist_from_sma_200_pct : float | None
        (close - SMA) / SMA * 100.
    ma_alignment : str
        'bullish' if sma_9 > sma_20 > sma_50, 'bearish' if reversed, else 'neutral'.
    price_vs_ma_score : float
        0-100 score of how many MAs price is above (weighted by period).
    """

    @property
    def name(self) -> str:
        return "sma"

    @property
    def category(self) -> str:
        return "trend"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 20):
            return self._empty_result()

        close = df["close"]
        periods = getattr(config, "sma_windows", [9, 20, 50, 200])
        result: dict[str, Any] = {}

        sma_values: dict[int, float | None] = {}
        for p in periods:
            if len(close.dropna()) < p // 2:
                result[f"sma_{p}"] = None
                result[f"slope_{p}"] = None
                result[f"above_sma_{p}"] = None
                result[f"dist_from_sma_{p}_pct"] = None
                sma_values[p] = None
                continue

            sma_series = _sma(close, p)
            val = self._safe_last(sma_series)
            sma_values[p] = val
            result[f"sma_{p}"] = val
            result[f"slope_{p}"] = _slope(sma_series, window=min(10, len(sma_series.dropna())))

            last_close = self._safe_last(close)
            if val and last_close:
                result[f"above_sma_{p}"] = last_close > val
                result[f"dist_from_sma_{p}_pct"] = (last_close - val) / val * 100
            else:
                result[f"above_sma_{p}"] = None
                result[f"dist_from_sma_{p}_pct"] = None

        # MA alignment
        v9 = sma_values.get(9)
        v20 = sma_values.get(20)
        v50 = sma_values.get(50)
        v200 = sma_values.get(200)

        if v9 and v20 and v50:
            if v9 > v20 > v50:
                alignment = "bullish"
            elif v9 < v20 < v50:
                alignment = "bearish"
            else:
                alignment = "neutral"
        else:
            alignment = "neutral"
        result["ma_alignment"] = alignment

        # price_vs_ma_score: weighted score (200 has highest weight)
        weights = {9: 0.1, 20: 0.2, 50: 0.3, 200: 0.4}
        score = 50.0
        w_sum = 0.0
        s_sum = 0.0
        for p, w in weights.items():
            above = result.get(f"above_sma_{p}")
            if above is not None:
                s_sum += w * (100.0 if above else 0.0)
                w_sum += w
        result["price_vs_ma_score"] = s_sum / w_sum if w_sum > 0 else 50.0

        return result

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "sma_9": None, "sma_20": None, "sma_50": None, "sma_200": None,
            "slope_9": None, "slope_20": None, "slope_50": None, "slope_200": None,
            "above_sma_9": None, "above_sma_20": None, "above_sma_50": None, "above_sma_200": None,
            "dist_from_sma_20_pct": None, "dist_from_sma_50_pct": None, "dist_from_sma_200_pct": None,
            "ma_alignment": "neutral",
            "price_vs_ma_score": 50.0,
        }


# ---------------------------------------------------------------------------
# EMAIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("ema")
class EMAIndicator(BaseIndicator):
    """Exponential Moving Averages: 9, 20, 50-period.

    Outputs
    -------
    ema_9, ema_20, ema_50 : float | None
    slope_9, slope_20, slope_50 : float | None
    above_ema_9, above_ema_20, above_ema_50 : bool | None
    ema_9_vs_20_cross : str   'bullish_cross', 'bearish_cross', or 'none'
    """

    @property
    def name(self) -> str:
        return "ema"

    @property
    def category(self) -> str:
        return "trend"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 10):
            return {
                "ema_9": None, "ema_20": None, "ema_50": None,
                "slope_9": None, "slope_20": None, "slope_50": None,
                "above_ema_9": None, "above_ema_20": None, "above_ema_50": None,
                "ema_9_vs_20_cross": "none",
                "ema_alignment": "neutral",
            }

        close = df["close"]
        periods = getattr(config, "ema_windows", [9, 20, 50])
        result: dict[str, Any] = {}

        ema_series_map: dict[int, pd.Series] = {}
        ema_values: dict[int, float | None] = {}

        for p in periods:
            series = _ema_series(close, p)
            ema_series_map[p] = series
            val = self._safe_last(series)
            ema_values[p] = val
            result[f"ema_{p}"] = val
            result[f"slope_{p}"] = _slope(series, window=min(10, len(series.dropna())))

            last_close = self._safe_last(close)
            if val and last_close:
                result[f"above_ema_{p}"] = last_close > val
            else:
                result[f"above_ema_{p}"] = None

        # EMA 9 vs EMA 20 crossover (last 3 bars)
        if 9 in ema_series_map and 20 in ema_series_map:
            s9 = ema_series_map[9].dropna()
            s20 = ema_series_map[20].dropna()
            min_len = min(len(s9), len(s20))
            if min_len >= 2:
                aligned_9 = s9.iloc[-2:].values
                aligned_20 = s20.reindex(s9.index).iloc[-2:].values
                if (
                    len(aligned_9) == 2
                    and len(aligned_20) == 2
                    and not np.any(np.isnan(np.concatenate([aligned_9, aligned_20])))
                ):
                    prev_above = aligned_9[-2] > aligned_20[-2]
                    curr_above = aligned_9[-1] > aligned_20[-1]
                    if not prev_above and curr_above:
                        result["ema_9_vs_20_cross"] = "bullish_cross"
                    elif prev_above and not curr_above:
                        result["ema_9_vs_20_cross"] = "bearish_cross"
                    else:
                        result["ema_9_vs_20_cross"] = "none"
                else:
                    result["ema_9_vs_20_cross"] = "none"
            else:
                result["ema_9_vs_20_cross"] = "none"
        else:
            result["ema_9_vs_20_cross"] = "none"

        # EMA alignment
        v9 = ema_values.get(9)
        v20 = ema_values.get(20)
        v50 = ema_values.get(50)
        if v9 and v20 and v50:
            if v9 > v20 > v50:
                result["ema_alignment"] = "bullish"
            elif v9 < v20 < v50:
                result["ema_alignment"] = "bearish"
            else:
                result["ema_alignment"] = "neutral"
        else:
            result["ema_alignment"] = "neutral"

        return result


# ---------------------------------------------------------------------------
# TrendRegimeIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("trend_regime")
class TrendRegimeIndicator(BaseIndicator):
    """Classify the current price trend regime.

    Regime is one of: 'uptrend', 'downtrend', 'sideways'.

    Classification rules (majority vote from multiple signals):
    1. Price relative to SMA 50 and SMA 200.
    2. SMA 50 slope direction.
    3. Higher highs / higher lows pattern over the last 20 bars.
    4. Close relative to 10-bar range midpoint.

    Outputs
    -------
    regime : str
    strength : float   0-100 (how decisive the classification is)
    ma_alignment : str  'bullish', 'bearish', 'neutral'
    hh_hl : bool   higher highs and higher lows
    ll_lh : bool   lower lows and lower highs
    price_above_sma50 : bool | None
    price_above_sma200 : bool | None
    sma50_slope : float | None
    """

    @property
    def name(self) -> str:
        return "trend_regime"

    @property
    def category(self) -> str:
        return "trend"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 20):
            return {
                "regime": "sideways",
                "strength": 0.0,
                "ma_alignment": "neutral",
                "hh_hl": False,
                "ll_lh": False,
                "price_above_sma50": None,
                "price_above_sma200": None,
                "sma50_slope": None,
            }

        close = df["close"]
        high = df["high"]
        low = df["low"]
        last_close = float(close.iloc[-1])

        # ---- SMA signals ----
        sma50 = _sma(close, 50)
        sma200 = _sma(close, 200)
        sma50_val = self._safe_last(sma50)
        sma200_val = self._safe_last(sma200)
        sma50_slope = _slope(sma50, window=10)

        price_above_50 = (last_close > sma50_val) if sma50_val else None
        price_above_200 = (last_close > sma200_val) if sma200_val else None

        # MA alignment
        if sma50_val and sma200_val:
            if sma50_val > sma200_val and (price_above_50 or False):
                ma_alignment = "bullish"
            elif sma50_val < sma200_val and not (price_above_50 or True):
                ma_alignment = "bearish"
            else:
                ma_alignment = "neutral"
        else:
            ma_alignment = "neutral"

        # ---- Higher highs / higher lows (last 20 bars) ----
        n = min(20, len(df))
        recent_high = high.iloc[-n:].values
        recent_low = low.iloc[-n:].values

        # Split into two halves and compare
        mid = n // 2
        first_half_high = recent_high[:mid].max() if mid > 0 else 0
        second_half_high = recent_high[mid:].max() if mid > 0 else 0
        first_half_low = recent_low[:mid].min() if mid > 0 else 0
        second_half_low = recent_low[mid:].min() if mid > 0 else 0

        hh_hl = (second_half_high > first_half_high) and (second_half_low > first_half_low)
        ll_lh = (second_half_high < first_half_high) and (second_half_low < first_half_low)

        # ---- Majority vote ----
        bull_signals = 0
        bear_signals = 0
        total = 0

        if price_above_50 is not None:
            total += 1
            if price_above_50:
                bull_signals += 1
            else:
                bear_signals += 1

        if price_above_200 is not None:
            total += 1
            if price_above_200:
                bull_signals += 1
            else:
                bear_signals += 1

        if sma50_slope is not None:
            total += 1
            if sma50_slope > 0.0002:  # meaningful positive slope
                bull_signals += 1
            elif sma50_slope < -0.0002:
                bear_signals += 1

        total += 1
        if hh_hl:
            bull_signals += 1
        elif ll_lh:
            bear_signals += 1

        # 10-bar range midpoint
        range_10_high = high.iloc[-10:].max() if len(high) >= 10 else high.max()
        range_10_low = low.iloc[-10:].min() if len(low) >= 10 else low.min()
        midpoint = (range_10_high + range_10_low) / 2
        total += 1
        if last_close > midpoint:
            bull_signals += 1
        else:
            bear_signals += 1

        bull_ratio = bull_signals / total if total > 0 else 0.5
        bear_ratio = bear_signals / total if total > 0 else 0.5

        if bull_ratio >= 0.6:
            regime = "uptrend"
            strength = bull_ratio * 100
        elif bear_ratio >= 0.6:
            regime = "downtrend"
            strength = bear_ratio * 100
        else:
            regime = "sideways"
            strength = (0.5 - abs(bull_ratio - bear_ratio)) * 100

        return {
            "regime": regime,
            "strength": round(strength, 1),
            "ma_alignment": ma_alignment,
            "hh_hl": hh_hl,
            "ll_lh": ll_lh,
            "price_above_sma50": price_above_50,
            "price_above_sma200": price_above_200,
            "sma50_slope": sma50_slope,
            "bull_signals": bull_signals,
            "bear_signals": bear_signals,
            "total_signals": total,
        }
