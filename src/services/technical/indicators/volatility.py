"""Volatility indicators: ATR, Bollinger Bands, StdDev, Gaps.

Registered indicators:
  - atr             : ATRIndicator
  - bollinger_bands : BollingerBandIndicator
  - std_dev         : StandardDevIndicator
  - gaps            : GapIndicator
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import structlog

from src.services.technical.registry import BaseIndicator, IndicatorRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Wilder-smoothed ATR."""
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# ATRIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("atr")
class ATRIndicator(BaseIndicator):
    """Average True Range (14).

    Outputs
    -------
    atr_14 : float | None         ATR value.
    atr_pct : float | None        ATR as % of close price.
    expanding : bool              ATR is expanding (increasing volatility).
    contracting : bool            ATR is contracting (decreasing volatility).
    atr_ratio_20 : float | None   ATR / 20-period ATR average.
    """

    @property
    def name(self) -> str:
        return "atr"

    @property
    def category(self) -> str:
        return "volatility"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 16):
            return {
                "atr_14": None, "atr_pct": None,
                "expanding": False, "contracting": False,
                "atr_ratio_20": None,
            }

        period = getattr(config, "atr_period", 14)
        atr_series = _atr(df["high"], df["low"], df["close"], period)
        atr_val = self._safe_last(atr_series)
        last_close = self._safe_last(df["close"])

        if atr_val is None or last_close is None or last_close == 0:
            return {
                "atr_14": None, "atr_pct": None,
                "expanding": False, "contracting": False,
                "atr_ratio_20": None,
            }

        atr_pct = (atr_val / last_close) * 100

        # Expanding / contracting: compare last ATR to 5-bar average ATR
        atr_clean = atr_series.dropna()
        expanding = False
        contracting = False
        atr_ratio_20 = None

        if len(atr_clean) >= 6:
            recent_atr = float(atr_clean.iloc[-1])
            prev_avg_atr = float(atr_clean.iloc[-6:-1].mean())
            if prev_avg_atr > 0:
                ratio = recent_atr / prev_avg_atr
                if ratio > 1.1:
                    expanding = True
                elif ratio < 0.9:
                    contracting = True

        if len(atr_clean) >= 20:
            avg_20 = float(atr_clean.iloc[-20:].mean())
            if avg_20 > 0:
                atr_ratio_20 = round(float(atr_clean.iloc[-1]) / avg_20, 3)

        return {
            "atr_14": round(atr_val, 4),
            "atr_pct": round(atr_pct, 3),
            "expanding": expanding,
            "contracting": contracting,
            "atr_ratio_20": atr_ratio_20,
        }


# ---------------------------------------------------------------------------
# BollingerBandIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("bollinger_bands")
class BollingerBandIndicator(BaseIndicator):
    """Bollinger Bands (20, 2.0).

    Outputs
    -------
    upper_band : float | None
    middle_band : float | None      (20-period SMA)
    lower_band : float | None
    bandwidth_pct : float | None    (upper - lower) / middle * 100
    percent_b : float | None        (close - lower) / (upper - lower)
    squeeze : bool                  bandwidth < 6% (historic low)
    price_above_upper : bool
    price_below_lower : bool
    squeeze_threshold_pct : float
    """

    @property
    def name(self) -> str:
        return "bollinger_bands"

    @property
    def category(self) -> str:
        return "volatility"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 20):
            return self._empty()

        period = getattr(config, "bb_period", 20)
        std_mult = getattr(config, "bb_std", 2.0)
        squeeze_threshold = getattr(config, "bb_squeeze_threshold_pct", 6.0)

        close = df["close"]
        sma = close.rolling(window=period, min_periods=period // 2).mean()
        std = close.rolling(window=period, min_periods=period // 2).std(ddof=0)

        upper = sma + std_mult * std
        lower = sma - std_mult * std

        upper_val = self._safe_last(upper)
        middle_val = self._safe_last(sma)
        lower_val = self._safe_last(lower)
        last_close = self._safe_last(close)

        if any(v is None for v in [upper_val, middle_val, lower_val]):
            return self._empty()

        band_width = upper_val - lower_val
        bandwidth_pct = (band_width / middle_val * 100) if middle_val else None
        pct_b = (
            (last_close - lower_val) / band_width
            if band_width and band_width > 0 and last_close
            else None
        )

        squeeze = (bandwidth_pct is not None and bandwidth_pct < squeeze_threshold)
        price_above_upper = (last_close or 0) > (upper_val or float("inf"))
        price_below_lower = (last_close or 0) < (lower_val or float("-inf"))

        return {
            "upper_band": round(upper_val, 4),
            "middle_band": round(middle_val, 4),
            "lower_band": round(lower_val, 4),
            "bandwidth_pct": round(bandwidth_pct, 3) if bandwidth_pct else None,
            "percent_b": round(pct_b, 4) if pct_b is not None else None,
            "squeeze": squeeze,
            "price_above_upper": price_above_upper,
            "price_below_lower": price_below_lower,
            "squeeze_threshold_pct": squeeze_threshold,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "upper_band": None, "middle_band": None, "lower_band": None,
            "bandwidth_pct": None, "percent_b": None,
            "squeeze": False, "price_above_upper": False,
            "price_below_lower": False, "squeeze_threshold_pct": 6.0,
        }


# ---------------------------------------------------------------------------
# StandardDevIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("std_dev")
class StandardDevIndicator(BaseIndicator):
    """Rolling standard deviation of returns.

    Outputs
    -------
    std_10 : float | None       10-period std dev of log returns (annualised %)
    std_20 : float | None
    std_60 : float | None
    realised_vol_20 : float | None   20-day annualised realised volatility (%)
    vol_regime : str            'low' (<15%), 'normal' (15-30%), 'high' (>30%)
    """

    _ANNUALISE = 252 ** 0.5  # sqrt(252)

    @property
    def name(self) -> str:
        return "std_dev"

    @property
    def category(self) -> str:
        return "volatility"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 12):
            return {
                "std_10": None, "std_20": None, "std_60": None,
                "realised_vol_20": None, "vol_regime": "normal",
            }

        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

        result: dict[str, Any] = {}
        for period in [10, 20, 60]:
            if len(log_returns) >= period:
                std = float(log_returns.rolling(window=period).std(ddof=1).iloc[-1])
                annualised = std * self._ANNUALISE * 100
                result[f"std_{period}"] = round(annualised, 3)
            else:
                result[f"std_{period}"] = None

        rv20 = result.get("std_20")
        result["realised_vol_20"] = rv20

        if rv20 is None:
            vol_regime = "normal"
        elif rv20 < 15:
            vol_regime = "low"
        elif rv20 < 30:
            vol_regime = "normal"
        else:
            vol_regime = "high"

        result["vol_regime"] = vol_regime
        return result


# ---------------------------------------------------------------------------
# GapIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("gaps")
class GapIndicator(BaseIndicator):
    """Gap detection: identifies gap-up / gap-down at open.

    A gap is defined as: today's open > yesterday's high (gap up)
    or today's open < yesterday's low (gap down).

    Outputs
    -------
    gap_present : bool
    gap_type : str              'gap_up', 'gap_down', 'none'
    gap_magnitude_pct : float | None    % gap relative to prev close
    gap_filled : bool           Whether the gap was filled intrabar
    gap_fill_probability : float   Estimated probability the gap fills (0-1)
    recent_gaps_count : int     Number of gaps in last 20 bars
    avg_gap_magnitude_pct : float | None
    """

    @property
    def name(self) -> str:
        return "gaps"

    @property
    def category(self) -> str:
        return "volatility"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 5):
            return self._empty()

        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Latest bar gap
        last_open = self._safe_float(open_.iloc[-1])
        prev_high = self._safe_float(high.iloc[-2]) if len(df) >= 2 else None
        prev_low = self._safe_float(low.iloc[-2]) if len(df) >= 2 else None
        prev_close = self._safe_float(close.iloc[-2]) if len(df) >= 2 else None
        last_low = self._safe_float(low.iloc[-1])
        last_high = self._safe_float(high.iloc[-1])

        gap_present = False
        gap_type = "none"
        gap_magnitude_pct = None
        gap_filled = False

        if all(v is not None for v in [last_open, prev_high, prev_low, prev_close]):
            if last_open > prev_high:  # type: ignore[operator]
                gap_present = True
                gap_type = "gap_up"
                gap_magnitude_pct = (last_open - prev_close) / prev_close * 100  # type: ignore[operator]
                # Filled if low crossed back into prev close
                gap_filled = (last_low or float("inf")) <= prev_close  # type: ignore[operator]
            elif last_open < prev_low:  # type: ignore[operator]
                gap_present = True
                gap_type = "gap_down"
                gap_magnitude_pct = (last_open - prev_close) / prev_close * 100  # type: ignore[operator]
                gap_filled = (last_high or float("-inf")) >= prev_close  # type: ignore[operator]

        # Recent gaps (last 20 bars)
        n = min(20, len(df))
        recent_df = df.iloc[-n:]
        gap_magnitudes: list[float] = []

        for i in range(1, len(recent_df)):
            o = self._safe_float(recent_df["open"].iloc[i])
            ph = self._safe_float(recent_df["high"].iloc[i - 1])
            pl = self._safe_float(recent_df["low"].iloc[i - 1])
            pc = self._safe_float(recent_df["close"].iloc[i - 1])
            if o and ph and pl and pc:
                if o > ph or o < pl:
                    gap_magnitudes.append(abs(o - pc) / pc * 100)

        # Gap fill probability: empirical – larger gaps fill less often
        gap_fill_prob = 0.5
        if gap_magnitude_pct is not None:
            mag = abs(gap_magnitude_pct)
            if mag < 1:
                gap_fill_prob = 0.75
            elif mag < 3:
                gap_fill_prob = 0.55
            elif mag < 5:
                gap_fill_prob = 0.40
            else:
                gap_fill_prob = 0.25

        return {
            "gap_present": gap_present,
            "gap_type": gap_type,
            "gap_magnitude_pct": round(gap_magnitude_pct, 3) if gap_magnitude_pct is not None else None,
            "gap_filled": gap_filled,
            "gap_fill_probability": gap_fill_prob,
            "recent_gaps_count": len(gap_magnitudes),
            "avg_gap_magnitude_pct": (
                round(float(np.mean(gap_magnitudes)), 3) if gap_magnitudes else None
            ),
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "gap_present": False, "gap_type": "none",
            "gap_magnitude_pct": None, "gap_filled": False,
            "gap_fill_probability": 0.5, "recent_gaps_count": 0,
            "avg_gap_magnitude_pct": None,
        }
