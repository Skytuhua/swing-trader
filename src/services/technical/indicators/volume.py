"""Volume indicators: VolumeTrend, VolumeSpike, VWAP.

Registered indicators:
  - volume_trend  : VolumeTrendIndicator
  - volume_spike  : VolumeSpikeIndicator
  - vwap          : VWAPIndicator
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import structlog

from src.services.technical.registry import BaseIndicator, IndicatorRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# VolumeTrendIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("volume_trend")
class VolumeTrendIndicator(BaseIndicator):
    """Volume trend analysis: SMA, relative volume, trend direction.

    Outputs
    -------
    volume_sma_20 : float | None       20-day volume SMA.
    volume_sma_50 : float | None
    last_volume : int | None
    relative_volume : float | None     last_volume / volume_sma_20.
    volume_trend_direction : str       'increasing', 'decreasing', 'flat'
    volume_vs_avg_pct : float | None   (last_vol - avg) / avg * 100
    on_balance_volume : float | None   Simplified OBV (last value).
    """

    @property
    def name(self) -> str:
        return "volume_trend"

    @property
    def category(self) -> str:
        return "volume"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 5):
            return self._empty()

        volume = df["volume"].astype(float)
        close = df["close"]

        last_vol = self._safe_float(volume.iloc[-1])
        vol_sma_20 = self._safe_last(volume.rolling(20, min_periods=5).mean())
        vol_sma_50 = self._safe_last(volume.rolling(50, min_periods=10).mean())

        # Relative volume
        relative_vol = None
        vol_vs_avg_pct = None
        if last_vol is not None and vol_sma_20 and vol_sma_20 > 0:
            relative_vol = round(last_vol / vol_sma_20, 3)
            vol_vs_avg_pct = round((last_vol - vol_sma_20) / vol_sma_20 * 100, 2)

        # Volume trend: compare 5-bar average vs 20-bar average
        vol_trend_dir = "flat"
        vol_clean = volume.dropna()
        if len(vol_clean) >= 10:
            recent_5 = float(vol_clean.iloc[-5:].mean())
            prior_15 = float(vol_clean.iloc[-20:-5].mean()) if len(vol_clean) >= 20 else float(vol_clean.mean())
            if prior_15 > 0:
                ratio = recent_5 / prior_15
                if ratio > 1.15:
                    vol_trend_dir = "increasing"
                elif ratio < 0.85:
                    vol_trend_dir = "decreasing"

        # On-Balance Volume (simplified)
        obv = self._compute_obv(close, volume)
        obv_last = self._safe_last(obv)

        return {
            "volume_sma_20": round(vol_sma_20, 0) if vol_sma_20 else None,
            "volume_sma_50": round(vol_sma_50, 0) if vol_sma_50 else None,
            "last_volume": int(last_vol) if last_vol is not None else None,
            "relative_volume": relative_vol,
            "volume_trend_direction": vol_trend_dir,
            "volume_vs_avg_pct": vol_vs_avg_pct,
            "on_balance_volume": round(obv_last, 0) if obv_last else None,
        }

    @staticmethod
    def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Simplified OBV: cumsum of signed volume."""
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "volume_sma_20": None, "volume_sma_50": None,
            "last_volume": None, "relative_volume": None,
            "volume_trend_direction": "flat", "volume_vs_avg_pct": None,
            "on_balance_volume": None,
        }


# ---------------------------------------------------------------------------
# VolumeSpikeIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("volume_spike")
class VolumeSpikeIndicator(BaseIndicator):
    """Detect volume spikes (> 2x 20-day average) and their price context.

    Outputs
    -------
    spike_detected : bool
    spike_multiplier : float | None    last_vol / avg_vol
    spike_direction : str              'up_spike', 'down_spike', 'none'
    consecutive_high_vol_days : int    Days in a row with vol > 1.5x avg
    recent_spike_count : int           Number of spikes in last 20 bars
    """

    _SPIKE_THRESHOLD = 2.0   # > 2x average = spike
    _HIGH_VOL_THRESHOLD = 1.5

    @property
    def name(self) -> str:
        return "volume_spike"

    @property
    def category(self) -> str:
        return "volume"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 5):
            return self._empty()

        volume = df["volume"].astype(float)
        close = df["close"]
        vol_sma20 = volume.rolling(20, min_periods=5).mean()

        last_vol = self._safe_float(volume.iloc[-1])
        avg_vol = self._safe_last(vol_sma20.shift(1))  # exclude today from avg

        if last_vol is None or avg_vol is None or avg_vol == 0:
            return self._empty()

        spike_mult = last_vol / avg_vol
        spike_detected = spike_mult >= self._SPIKE_THRESHOLD

        # Spike direction
        spike_dir = "none"
        if spike_detected:
            last_close = self._safe_float(close.iloc[-1])
            prev_close = self._safe_float(close.iloc[-2]) if len(close) >= 2 else None
            if last_close and prev_close:
                spike_dir = "up_spike" if last_close >= prev_close else "down_spike"

        # Consecutive high-volume days
        consec = 0
        vol_arr = volume.dropna().values
        avg_arr = vol_sma20.shift(1).dropna().values
        min_len = min(len(vol_arr), len(avg_arr))
        if min_len > 0:
            for i in range(min_len - 1, -1, -1):
                if avg_arr[i] > 0 and vol_arr[i] / avg_arr[i] >= self._HIGH_VOL_THRESHOLD:
                    consec += 1
                else:
                    break

        # Recent spike count (last 20 bars)
        n = min(20, min_len)
        recent_spikes = 0
        for i in range(min_len - n, min_len):
            if avg_arr[i] > 0 and vol_arr[i] / avg_arr[i] >= self._SPIKE_THRESHOLD:
                recent_spikes += 1

        return {
            "spike_detected": spike_detected,
            "spike_multiplier": round(spike_mult, 3),
            "spike_direction": spike_dir,
            "consecutive_high_vol_days": consec,
            "recent_spike_count": recent_spikes,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "spike_detected": False, "spike_multiplier": None,
            "spike_direction": "none", "consecutive_high_vol_days": 0,
            "recent_spike_count": 0,
        }


# ---------------------------------------------------------------------------
# VWAPIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("vwap")
class VWAPIndicator(BaseIndicator):
    """VWAP and distance from VWAP.

    For daily data, VWAP is computed as a rolling 20-day VWAP using
    typical price * volume.  If intraday_df is provided via df.attrs,
    the true intraday VWAP is computed.

    Outputs
    -------
    vwap : float | None               Current VWAP level.
    last_close : float | None
    distance_from_vwap_pct : float | None   (close - vwap) / vwap * 100
    above_vwap : bool
    vwap_slope : float | None          VWAP slope (trend direction)
    daily_vwap_20 : float | None       20-day rolling VWAP (daily bars)
    """

    @property
    def name(self) -> str:
        return "vwap"

    @property
    def category(self) -> str:
        return "volume"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 5):
            return self._empty()

        intraday_df: pd.DataFrame | None = df.attrs.get("intraday_df")

        if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 5:
            vwap_val = self._intraday_vwap(intraday_df)
        else:
            vwap_val = self._rolling_daily_vwap(df, window=20)

        last_close = self._safe_last(df["close"])

        if vwap_val is None or last_close is None:
            return self._empty()

        dist_pct = (last_close - vwap_val) / vwap_val * 100 if vwap_val else None
        above_vwap = last_close > vwap_val

        # VWAP slope (daily rolling VWAP over last 10 bars)
        daily_vwap_series = self._rolling_daily_vwap_series(df, window=20)
        vwap_slope = None
        if daily_vwap_series is not None:
            clean = daily_vwap_series.dropna()
            w = min(10, len(clean))
            if w >= 3:
                y = clean.values[-w:]
                x = np.arange(w, dtype=float)
                x_mean, y_mean = x.mean(), y.mean()
                denom = ((x - x_mean) ** 2).sum()
                if denom > 0:
                    slope_raw = float(((x - x_mean) * (y - y_mean)).sum() / denom)
                    vwap_slope = slope_raw / y_mean if y_mean != 0 else slope_raw

        # 20-day rolling VWAP
        daily_vwap_20 = self._rolling_daily_vwap(df, window=20)

        return {
            "vwap": round(vwap_val, 4),
            "last_close": round(last_close, 4),
            "distance_from_vwap_pct": round(dist_pct, 3) if dist_pct is not None else None,
            "above_vwap": above_vwap,
            "vwap_slope": round(vwap_slope, 6) if vwap_slope is not None else None,
            "daily_vwap_20": round(daily_vwap_20, 4) if daily_vwap_20 else None,
        }

    # ---- helpers ----

    @staticmethod
    def _intraday_vwap(df: pd.DataFrame) -> float | None:
        """True VWAP from intraday bars (today's session)."""
        if "volume" not in df.columns or "close" not in df.columns:
            return None

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None

        typical = (df["high"] + df["low"] + df["close"]) / 3
        tpv = typical * df["volume"].astype(float)
        total_vol = df["volume"].astype(float).sum()

        if total_vol == 0:
            return None
        return float(tpv.sum() / total_vol)

    @staticmethod
    def _rolling_daily_vwap(df: pd.DataFrame, window: int = 20) -> float | None:
        """Rolling VWAP using daily bars."""
        if "volume" not in df.columns:
            return None

        typical = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"].astype(float)

        rolling_tpv = (typical * volume).rolling(window=window, min_periods=5).sum()
        rolling_vol = volume.rolling(window=window, min_periods=5).sum()

        vwap_series = rolling_tpv / rolling_vol.replace(0, float("nan"))
        clean = vwap_series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])

    @staticmethod
    def _rolling_daily_vwap_series(df: pd.DataFrame, window: int = 20) -> pd.Series | None:
        """Return the full rolling VWAP series for slope computation."""
        if "volume" not in df.columns:
            return None

        typical = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"].astype(float)

        rolling_tpv = (typical * volume).rolling(window=window, min_periods=5).sum()
        rolling_vol = volume.rolling(window=window, min_periods=5).sum()
        return rolling_tpv / rolling_vol.replace(0, float("nan"))

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "vwap": None, "last_close": None,
            "distance_from_vwap_pct": None, "above_vwap": False,
            "vwap_slope": None, "daily_vwap_20": None,
        }
