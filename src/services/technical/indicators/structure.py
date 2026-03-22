"""Price structure indicators: Support/Resistance, Breakouts, Pullbacks,
Consolidation, and Candle Patterns.

Registered indicators:
  - support_resistance  : SupportResistanceIndicator
  - breakout            : BreakoutDetector
  - pullback_quality    : PullbackQualityIndicator
  - consolidation       : ConsolidationIndicator
  - candle_patterns     : CandlePatternIndicator
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import structlog

from src.services.technical.registry import BaseIndicator, IndicatorRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helper: local pivot detection
# ---------------------------------------------------------------------------

def _find_pivots(
    high: pd.Series,
    low: pd.Series,
    left: int = 5,
    right: int = 5,
) -> tuple[list[float], list[float]]:
    """Find pivot highs and lows.

    A bar is a pivot high if it is the highest in [i-left, i+right].
    Returns (pivot_highs, pivot_lows) as lists of price values.
    """
    n = len(high)
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(left, n - right):
        h_window = high.iloc[i - left: i + right + 1]
        l_window = low.iloc[i - left: i + right + 1]

        if high.iloc[i] == h_window.max():
            pivot_highs.append(float(high.iloc[i]))
        if low.iloc[i] == l_window.min():
            pivot_lows.append(float(low.iloc[i]))

    return pivot_highs, pivot_lows


def _cluster_levels(
    prices: list[float],
    tolerance_pct: float = 0.5,
) -> list[float]:
    """Group nearby price levels into single representative levels."""
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = [[sorted_prices[0]]]

    for price in sorted_prices[1:]:
        cluster_ref = clusters[-1][-1]
        if abs(price - cluster_ref) / cluster_ref * 100 <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    return [float(np.mean(c)) for c in clusters]


# ---------------------------------------------------------------------------
# SupportResistanceIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("support_resistance")
class SupportResistanceIndicator(BaseIndicator):
    """Identify key support and resistance levels from recent pivots.

    Outputs
    -------
    support_levels : list[float]    Clustered support levels (ascending).
    resistance_levels : list[float] Clustered resistance levels (ascending).
    nearest_support : float | None  Closest support below current price.
    nearest_resistance : float | None  Closest resistance above current price.
    near_support : bool             Price within 2% of nearest support.
    near_resistance : bool          Price within 2% of nearest resistance.
    support_strength : int          How many pivot lows cluster near nearest support.
    resistance_strength : int
    """

    @property
    def name(self) -> str:
        return "support_resistance"

    @property
    def category(self) -> str:
        return "structure"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 15):
            return self._empty()

        # Use last 60 bars (or all if fewer)
        lookback = min(60, len(df))
        sub = df.iloc[-lookback:]

        left_bars = getattr(config, "pivot_left", 5)
        right_bars = getattr(config, "pivot_right", 5)

        pivot_highs, pivot_lows = _find_pivots(
            sub["high"], sub["low"], left=left_bars, right=right_bars
        )

        resistance_levels = _cluster_levels(pivot_highs, tolerance_pct=0.8)
        support_levels = _cluster_levels(pivot_lows, tolerance_pct=0.8)

        last_close = self._safe_last(df["close"])
        if last_close is None:
            return self._empty()

        # Nearest support (below current price)
        supports_below = [s for s in support_levels if s < last_close]
        nearest_support = max(supports_below) if supports_below else None

        # Nearest resistance (above current price)
        resistances_above = [r for r in resistance_levels if r > last_close]
        nearest_resistance = min(resistances_above) if resistances_above else None

        # Near support / resistance (within 2%)
        near_threshold = getattr(config, "near_level_pct", 2.0)
        near_support = (
            abs(last_close - nearest_support) / last_close * 100 < near_threshold
            if nearest_support else False
        )
        near_resistance = (
            abs(nearest_resistance - last_close) / last_close * 100 < near_threshold
            if nearest_resistance else False
        )

        # Strength: count how many original pivots cluster near the nearest level
        tol = 0.8 / 100
        support_strength = sum(
            1 for p in pivot_lows
            if nearest_support and abs(p - nearest_support) / nearest_support <= tol
        )
        resistance_strength = sum(
            1 for p in pivot_highs
            if nearest_resistance and abs(p - nearest_resistance) / nearest_resistance <= tol
        )

        return {
            "support_levels": [round(s, 4) for s in support_levels[-5:]],
            "resistance_levels": [round(r, 4) for r in resistance_levels[-5:]],
            "nearest_support": round(nearest_support, 4) if nearest_support else None,
            "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else None,
            "near_support": near_support,
            "near_resistance": near_resistance,
            "support_strength": support_strength,
            "resistance_strength": resistance_strength,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "support_levels": [], "resistance_levels": [],
            "nearest_support": None, "nearest_resistance": None,
            "near_support": False, "near_resistance": False,
            "support_strength": 0, "resistance_strength": 0,
        }


# ---------------------------------------------------------------------------
# BreakoutDetector
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("breakout")
class BreakoutDetector(BaseIndicator):
    """Detect breakouts from consolidation zones and resistance levels.

    A breakout is defined as: today's close > highest close in the prior
    N-bar lookback (default 20), confirmed by above-average volume.

    Outputs
    -------
    breakout_detected : bool
    breakout_direction : str        'bullish', 'bearish', 'none'
    breakout_level : float | None   The resistance/support level broken.
    volume_confirmed : bool         Volume > 1.5x 20-day average on breakout bar.
    bars_above_breakout : int       How many consecutive bars closed above breakout level.
    failed_breakout : bool          Closed above, then fell back below within 3 bars.
    """

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def category(self) -> str:
        return "structure"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 22):
            return self._empty()

        lookback = getattr(config, "breakout_lookback", 20)
        vol_confirm_mult = getattr(config, "breakout_volume_mult", 1.5)

        close = df["close"]
        volume = df["volume"].astype(float)

        # Prior period highs/lows (exclude today)
        prior = close.iloc[-(lookback + 1):-1]
        prior_high = float(prior.max())
        prior_low = float(prior.min())
        last_close = float(close.iloc[-1])
        last_vol = float(volume.iloc[-1])

        vol_avg = float(volume.iloc[-(lookback + 1):-1].mean())

        breakout = False
        breakout_dir = "none"
        breakout_level = None
        vol_confirmed = False

        if last_close > prior_high:
            breakout = True
            breakout_dir = "bullish"
            breakout_level = prior_high
        elif last_close < prior_low:
            breakout = True
            breakout_dir = "bearish"
            breakout_level = prior_low

        if breakout and vol_avg > 0:
            vol_confirmed = last_vol >= vol_avg * vol_confirm_mult

        # Bars above breakout level (consecutive from today backwards)
        bars_above = 0
        if breakout_dir == "bullish" and breakout_level:
            for i in range(len(close) - 1, max(0, len(close) - 6), -1):
                if float(close.iloc[i]) > breakout_level:
                    bars_above += 1
                else:
                    break

        # Failed breakout: closed above then back below within 3 bars
        failed = False
        if not breakout and len(close) >= 4 and breakout_level is None:
            # Check if prior 3 bars saw a breakout that reversed
            recent_close = close.iloc[-4:].values
            older_high = float(close.iloc[-25:-4].max()) if len(close) >= 25 else prior_high
            crossed_above = any(c > older_high for c in recent_close[:-1])
            back_below = recent_close[-1] <= older_high
            failed = crossed_above and back_below

        return {
            "breakout_detected": breakout,
            "breakout_direction": breakout_dir,
            "breakout_level": round(breakout_level, 4) if breakout_level else None,
            "volume_confirmed": vol_confirmed,
            "bars_above_breakout": bars_above,
            "failed_breakout": failed,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "breakout_detected": False, "breakout_direction": "none",
            "breakout_level": None, "volume_confirmed": False,
            "bars_above_breakout": 0, "failed_breakout": False,
        }


# ---------------------------------------------------------------------------
# PullbackQualityIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("pullback_quality")
class PullbackQualityIndicator(BaseIndicator):
    """Assess the quality of a pullback within an uptrend.

    A "quality" pullback is orderly (low-volume, narrow candles, not
    giving back more than 50% of the prior up-move).

    Outputs
    -------
    in_pullback : bool
    pullback_depth_pct : float | None   How far price has pulled back from recent high.
    retracement_ratio : float | None    Pullback / prior move (Fibonacci context).
    quality : str                       'orderly', 'moderate', 'chaotic', 'none'
    pullback_volume_ratio : float | None  Pullback vol / breakout vol (lower = better)
    candle_quality : str                'tight', 'wide', 'unknown'
    days_in_pullback : int
    """

    @property
    def name(self) -> str:
        return "pullback_quality"

    @property
    def category(self) -> str:
        return "structure"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 15):
            return self._empty()

        close = df["close"]
        high = df["high"]
        volume = df["volume"].astype(float)

        # Find recent high (last 20 bars)
        n = min(20, len(df))
        recent = df.iloc[-n:]
        recent_high_idx = recent["high"].idxmax()
        recent_high = float(recent["high"].max())
        last_close = float(close.iloc[-1])

        # Is price pulling back?
        pullback_pct = (recent_high - last_close) / recent_high * 100
        in_pullback = pullback_pct >= 2.0 and pullback_pct <= 40.0

        if not in_pullback:
            return {
                "in_pullback": False, "pullback_depth_pct": round(pullback_pct, 2),
                "retracement_ratio": None, "quality": "none",
                "pullback_volume_ratio": None, "candle_quality": "unknown",
                "days_in_pullback": 0,
            }

        # Prior move: from the low before the recent high
        try:
            high_pos = df.index.get_loc(recent_high_idx)
        except KeyError:
            high_pos = len(df) - 1

        lookback_for_prior = max(0, high_pos - 20)
        prior_low = float(df["low"].iloc[lookback_for_prior:high_pos].min())
        prior_move = recent_high - prior_low

        retracement_ratio = (
            (recent_high - last_close) / prior_move if prior_move > 0 else None
        )

        # Days in pullback (bars since recent high)
        try:
            high_loc = df.index.get_loc(recent_high_idx)
            days_in_pullback = len(df) - 1 - high_loc
        except Exception:
            days_in_pullback = 1

        # Volume during pullback vs prior move
        pullback_vol = float(volume.iloc[-days_in_pullback:].mean()) if days_in_pullback > 0 else float(volume.iloc[-1])
        prior_vol = float(volume.iloc[max(0, high_pos - 10):high_pos].mean()) if high_pos > 0 else pullback_vol
        pullback_vol_ratio = (pullback_vol / prior_vol) if prior_vol > 0 else None

        # Candle quality during pullback (tight vs wide candles)
        pullback_bars = df.iloc[-days_in_pullback:] if days_in_pullback > 0 else df.iloc[-3:]
        if len(pullback_bars) > 0:
            avg_range = float((pullback_bars["high"] - pullback_bars["low"]).mean())
            overall_avg_range = float((df["high"] - df["low"]).mean())
            candle_quality = (
                "tight" if avg_range < overall_avg_range * 0.8 else
                "wide" if avg_range > overall_avg_range * 1.3 else
                "normal"
            )
        else:
            candle_quality = "unknown"

        # Quality assessment
        if retracement_ratio is not None:
            if (
                retracement_ratio <= 0.50
                and (pullback_vol_ratio or 1.0) < 0.8
                and candle_quality == "tight"
            ):
                quality = "orderly"
            elif retracement_ratio > 0.65 or (pullback_vol_ratio or 1.0) > 1.2:
                quality = "chaotic"
            else:
                quality = "moderate"
        else:
            quality = "moderate"

        return {
            "in_pullback": in_pullback,
            "pullback_depth_pct": round(pullback_pct, 2),
            "retracement_ratio": round(retracement_ratio, 3) if retracement_ratio else None,
            "quality": quality,
            "pullback_volume_ratio": round(pullback_vol_ratio, 3) if pullback_vol_ratio else None,
            "candle_quality": candle_quality,
            "days_in_pullback": days_in_pullback,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "in_pullback": False, "pullback_depth_pct": None,
            "retracement_ratio": None, "quality": "none",
            "pullback_volume_ratio": None, "candle_quality": "unknown",
            "days_in_pullback": 0,
        }


# ---------------------------------------------------------------------------
# ConsolidationIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("consolidation")
class ConsolidationIndicator(BaseIndicator):
    """Detect tight range consolidation (potential energy building).

    Consolidation: the close range over N bars is < X% of price, and
    the ATR is contracting.

    Outputs
    -------
    in_consolidation : bool
    range_pct : float | None        (high - low) / avg_close * 100 over window
    duration_bars : int             How many consecutive bars in consolidation
    consolidation_high : float | None
    consolidation_low : float | None
    tightness_score : float         0-100 (100 = perfectly flat)
    """

    @property
    def name(self) -> str:
        return "consolidation"

    @property
    def category(self) -> str:
        return "structure"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 10):
            return self._empty()

        consol_threshold_pct = getattr(config, "consolidation_range_pct", 5.0)
        min_bars = getattr(config, "consolidation_min_bars", 3)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Check rolling windows of increasing size until range threshold is violated
        max_window = 20
        consol_bars = 0
        consol_high = None
        consol_low = None

        for window in range(min_bars, min(max_window + 1, len(df) + 1)):
            sub_high = high.iloc[-window:]
            sub_low = low.iloc[-window:]
            sub_close = close.iloc[-window:]
            avg_close = float(sub_close.mean())
            range_val = float(sub_high.max() - sub_low.min())

            if avg_close == 0:
                break

            range_pct = range_val / avg_close * 100
            if range_pct > consol_threshold_pct:
                break

            consol_bars = window
            consol_high = float(sub_high.max())
            consol_low = float(sub_low.min())

        in_consolidation = consol_bars >= min_bars

        # Overall range_pct at the current window
        n = min(20, len(df))
        sub_h = high.iloc[-n:]
        sub_l = low.iloc[-n:]
        sub_c = close.iloc[-n:]
        avg_c = float(sub_c.mean())
        overall_range_pct = (
            float(sub_h.max() - sub_l.min()) / avg_c * 100 if avg_c > 0 else None
        )

        # Tightness score: inverse of range percentage
        if consol_bars > 0 and consol_high and consol_low and avg_c > 0:
            consol_range_pct = (consol_high - consol_low) / avg_c * 100
            tightness_score = max(0.0, 100.0 - consol_range_pct * (100.0 / consol_threshold_pct))
        else:
            tightness_score = 0.0

        return {
            "in_consolidation": in_consolidation,
            "range_pct": round(overall_range_pct, 3) if overall_range_pct else None,
            "duration_bars": consol_bars,
            "consolidation_high": round(consol_high, 4) if consol_high else None,
            "consolidation_low": round(consol_low, 4) if consol_low else None,
            "tightness_score": round(tightness_score, 2),
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "in_consolidation": False, "range_pct": None,
            "duration_bars": 0, "consolidation_high": None,
            "consolidation_low": None, "tightness_score": 0.0,
        }


# ---------------------------------------------------------------------------
# CandlePatternIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("candle_patterns")
class CandlePatternIndicator(BaseIndicator):
    """Basic candlestick pattern detection on the last 1-3 bars.

    Patterns detected:
      - Bullish: hammer, bullish_engulfing, doji_bullish_reversal
      - Bearish: shooting_star, bearish_engulfing, doji_bearish_reversal
      - Neutral: doji

    Outputs
    -------
    patterns_detected : list[str]   All detected patterns on the most recent bars.
    last_pattern : str              Most recent / dominant pattern.
    signal : str                    'bullish', 'bearish', 'neutral', 'none'
    body_to_range_ratio : float     |open - close| / (high - low) for last bar.
    is_doji : bool                  Last candle is a doji (tiny body).
    is_long_body : bool             Body > 70% of range.
    """

    # Thresholds
    _DOJI_RATIO = 0.1     # body/range < 10% → doji
    _LONG_BODY_RATIO = 0.7
    _HAMMER_SHADOW_RATIO = 2.0  # lower shadow >= 2x body

    @property
    def name(self) -> str:
        return "candle_patterns"

    @property
    def category(self) -> str:
        return "structure"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 2):
            return self._empty()

        # Work on last 3 bars
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        prev2 = df.iloc[-3] if len(df) >= 3 else None

        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        range_ = h - l
        body = abs(c - o)
        body_ratio = body / range_ if range_ > 0 else 0.0
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        patterns: list[str] = []

        # ---- Single-bar patterns ----

        # Doji
        is_doji = body_ratio < self._DOJI_RATIO
        if is_doji:
            patterns.append("doji")

        # Hammer (bullish): small body near top, long lower shadow, in downtrend
        is_hammer = (
            body_ratio < 0.35
            and lower_shadow >= body * self._HAMMER_SHADOW_RATIO
            and lower_shadow >= range_ * 0.5
            and upper_shadow < body * 0.5
        )
        if is_hammer:
            patterns.append("hammer")

        # Shooting Star (bearish): small body near bottom, long upper shadow, in uptrend
        is_shooting_star = (
            body_ratio < 0.35
            and upper_shadow >= body * self._HAMMER_SHADOW_RATIO
            and upper_shadow >= range_ * 0.5
            and lower_shadow < body * 0.5
        )
        if is_shooting_star:
            patterns.append("shooting_star")

        # Long bullish candle
        is_long_body = body_ratio >= self._LONG_BODY_RATIO
        is_bullish = c > o
        is_bearish = c < o
        if is_long_body and is_bullish:
            patterns.append("bullish_marubozu")
        if is_long_body and is_bearish:
            patterns.append("bearish_marubozu")

        # ---- Two-bar patterns ----
        if prev is not None:
            p_o = float(prev["open"])
            p_h = float(prev["high"])
            p_l = float(prev["low"])
            p_c = float(prev["close"])
            p_body = abs(p_c - p_o)

            # Bullish engulfing
            if (
                p_c < p_o  # prev was bearish
                and c > o  # curr is bullish
                and o < p_c  # curr open below prev close
                and c > p_o  # curr close above prev open
                and body >= p_body * 0.8
            ):
                patterns.append("bullish_engulfing")

            # Bearish engulfing
            if (
                p_c > p_o  # prev was bullish
                and c < o  # curr is bearish
                and o > p_c  # curr open above prev close
                and c < p_o  # curr close below prev open
                and body >= p_body * 0.8
            ):
                patterns.append("bearish_engulfing")

            # Doji after trend (reversal signal)
            if is_doji:
                if p_c < p_o:  # prior bearish → doji = bullish reversal
                    patterns.append("doji_bullish_reversal")
                elif p_c > p_o:  # prior bullish → doji = bearish reversal
                    patterns.append("doji_bearish_reversal")

        # ---- Signal aggregation ----
        bullish_set = {"hammer", "bullish_engulfing", "doji_bullish_reversal", "bullish_marubozu"}
        bearish_set = {"shooting_star", "bearish_engulfing", "doji_bearish_reversal", "bearish_marubozu"}

        bull_count = sum(1 for p in patterns if p in bullish_set)
        bear_count = sum(1 for p in patterns if p in bearish_set)

        if bull_count > bear_count:
            signal = "bullish"
        elif bear_count > bull_count:
            signal = "bearish"
        elif patterns:
            signal = "neutral"
        else:
            signal = "none"

        last_pattern = patterns[-1] if patterns else "none"

        return {
            "patterns_detected": patterns,
            "last_pattern": last_pattern,
            "signal": signal,
            "body_to_range_ratio": round(body_ratio, 4),
            "is_doji": is_doji,
            "is_long_body": is_long_body,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "patterns_detected": [], "last_pattern": "none",
            "signal": "none", "body_to_range_ratio": None,
            "is_doji": False, "is_long_body": False,
        }
