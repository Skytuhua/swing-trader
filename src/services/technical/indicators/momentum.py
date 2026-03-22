"""Momentum indicators: RSI, MACD, Stochastic, ROC, ADX, RelativeStrength.

Registered indicators:
  - rsi             : RSIIndicator
  - macd            : MACDIndicator
  - stochastic      : StochasticIndicator
  - roc             : ROCIndicator
  - adx             : ADXIndicator
  - relative_strength: RelativeStrengthIndicator
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import structlog

from src.services.technical.registry import BaseIndicator, IndicatorRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python implementations (no pandas_ta dependency required,
# but pandas_ta is used when available for speed / accuracy).
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = ((close - lowest_low) / denom) * 100
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


def _roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change = (close / close_n_bars_ago - 1) * 100."""
    return ((close / close.shift(period)) - 1) * 100


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range series."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI)."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    smooth_plus = plus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    smooth_minus = minus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = (smooth_plus / atr.replace(0, np.nan)) * 100
    minus_di = (smooth_minus / atr.replace(0, np.nan)) * 100

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx, plus_di, minus_di


# ---------------------------------------------------------------------------
# RSIIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("rsi")
class RSIIndicator(BaseIndicator):
    """RSI(14) with overbought/oversold zones and basic divergence detection.

    Outputs
    -------
    rsi_14 : float | None          Current RSI value.
    overbought : bool              RSI > 70.
    oversold : bool                RSI < 30.
    zone : str                     'overbought', 'oversold', 'neutral'.
    rsi_slope : float | None       Slope of RSI over last 5 bars.
    bullish_divergence : bool      Price made lower low but RSI made higher low.
    bearish_divergence : bool      Price made higher high but RSI made lower high.
    """

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        min_bars = getattr(config, "rsi_period", 14) + 5
        if not self._require_min_bars(df, min_bars):
            return self._empty()

        close = df["close"]
        period = getattr(config, "rsi_period", 14)
        rsi_series = _rsi(close, period)

        rsi_val = self._safe_last(rsi_series)
        if rsi_val is None:
            return self._empty()

        overbought_threshold = getattr(config, "rsi_overbought", 70)
        oversold_threshold = getattr(config, "rsi_oversold", 30)

        overbought = rsi_val > overbought_threshold
        oversold = rsi_val < oversold_threshold
        zone = (
            "overbought" if overbought else "oversold" if oversold else "neutral"
        )

        # RSI slope over last 5 bars
        rsi_clean = rsi_series.dropna()
        window = min(5, len(rsi_clean))
        if window >= 2:
            recent = rsi_clean.values[-window:]
            x = np.arange(window, dtype=float)
            x_mean, y_mean = x.mean(), recent.mean()
            denom = ((x - x_mean) ** 2).sum()
            rsi_slope = float(((x - x_mean) * (recent - y_mean)).sum() / denom) if denom else 0.0
        else:
            rsi_slope = None

        # Divergence detection (last 30 bars, 2 swing comparison)
        bullish_div = False
        bearish_div = False
        if len(df) >= 20 and len(rsi_clean) >= 20:
            n = 20
            close_vals = close.dropna().values[-n:]
            rsi_vals = rsi_clean.values[-n:]

            # Find local extrema (simplified: compare midpoint vs endpoints)
            mid = n // 2
            price_prev_low = close_vals[:mid].min()
            price_curr_low = close_vals[mid:].min()
            rsi_prev_low = rsi_vals[:mid].min()
            rsi_curr_low = rsi_vals[mid:].min()
            price_prev_high = close_vals[:mid].max()
            price_curr_high = close_vals[mid:].max()
            rsi_prev_high = rsi_vals[:mid].max()
            rsi_curr_high = rsi_vals[mid:].max()

            bullish_div = (price_curr_low < price_prev_low) and (rsi_curr_low > rsi_prev_low)
            bearish_div = (price_curr_high > price_prev_high) and (rsi_curr_high < rsi_prev_high)

        return {
            "rsi_14": round(rsi_val, 2),
            "overbought": overbought,
            "oversold": oversold,
            "zone": zone,
            "rsi_slope": round(rsi_slope, 4) if rsi_slope is not None else None,
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "rsi_14": None, "overbought": False, "oversold": False,
            "zone": "neutral", "rsi_slope": None,
            "bullish_divergence": False, "bearish_divergence": False,
        }


# ---------------------------------------------------------------------------
# MACDIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("macd")
class MACDIndicator(BaseIndicator):
    """MACD (12, 26, 9) with crossover detection and histogram trend.

    Outputs
    -------
    macd_line : float | None
    signal_line : float | None
    histogram : float | None
    bullish_crossover : bool        MACD crossed above signal on last bar.
    bearish_crossover : bool        MACD crossed below signal on last bar.
    histogram_trend : str           'rising', 'falling', 'flat'
    above_zero : bool               MACD line > 0
    """

    @property
    def name(self) -> str:
        return "macd"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 35):
            return self._empty()

        close = df["close"]
        fast = getattr(config, "macd_fast", 12)
        slow = getattr(config, "macd_slow", 26)
        signal = getattr(config, "macd_signal", 9)

        macd_line, signal_line, histogram = _macd(close, fast, slow, signal)

        macd_val = self._safe_last(macd_line)
        signal_val = self._safe_last(signal_line)
        hist_val = self._safe_last(histogram)

        if macd_val is None or signal_val is None:
            return self._empty()

        # Crossover detection (last 2 bars)
        hist_clean = histogram.dropna()
        bullish_cross = False
        bearish_cross = False
        if len(hist_clean) >= 2:
            prev_hist = float(hist_clean.iloc[-2])
            curr_hist = float(hist_clean.iloc[-1])
            bullish_cross = prev_hist < 0 and curr_hist >= 0
            bearish_cross = prev_hist > 0 and curr_hist <= 0

        # Histogram trend (last 4 bars)
        hist_trend = "flat"
        if len(hist_clean) >= 4:
            recent = hist_clean.values[-4:]
            slope = float(np.polyfit(np.arange(4), recent, 1)[0])
            if slope > 0.001:
                hist_trend = "rising"
            elif slope < -0.001:
                hist_trend = "falling"

        return {
            "macd_line": round(macd_val, 4),
            "signal_line": round(signal_val, 4),
            "histogram": round(hist_val, 4) if hist_val is not None else None,
            "bullish_crossover": bullish_cross,
            "bearish_crossover": bearish_cross,
            "histogram_trend": hist_trend,
            "above_zero": macd_val > 0,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "macd_line": None, "signal_line": None, "histogram": None,
            "bullish_crossover": False, "bearish_crossover": False,
            "histogram_trend": "flat", "above_zero": False,
        }


# ---------------------------------------------------------------------------
# StochasticIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("stochastic")
class StochasticIndicator(BaseIndicator):
    """Stochastic Oscillator (%K, %D) with overbought/oversold detection.

    Outputs
    -------
    stoch_k : float | None
    stoch_d : float | None
    overbought : bool     %K > 80
    oversold : bool       %K < 20
    bullish_cross : bool  %K crossed above %D from oversold territory
    bearish_cross : bool  %K crossed below %D from overbought territory
    """

    @property
    def name(self) -> str:
        return "stochastic"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 16):
            return self._empty()

        k_period = getattr(config, "stoch_k", 14)
        d_period = getattr(config, "stoch_d", 3)

        k_series, d_series = _stochastic(df["high"], df["low"], df["close"], k_period, d_period)

        k_val = self._safe_last(k_series)
        d_val = self._safe_last(d_series)

        if k_val is None or d_val is None:
            return self._empty()

        overbought = k_val > 80
        oversold = k_val < 20

        # Cross detection (last 2 bars)
        k_clean = k_series.dropna()
        d_clean = d_series.dropna()
        bullish_cross = False
        bearish_cross = False
        if len(k_clean) >= 2 and len(d_clean) >= 2:
            prev_k = float(k_clean.iloc[-2])
            curr_k = float(k_clean.iloc[-1])
            prev_d = float(d_clean.iloc[-2])
            curr_d = float(d_clean.iloc[-1])

            if prev_k < prev_d and curr_k >= curr_d and prev_k < 40:
                bullish_cross = True
            if prev_k > prev_d and curr_k <= curr_d and prev_k > 60:
                bearish_cross = True

        return {
            "stoch_k": round(k_val, 2),
            "stoch_d": round(d_val, 2),
            "overbought": overbought,
            "oversold": oversold,
            "bullish_cross": bullish_cross,
            "bearish_cross": bearish_cross,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "stoch_k": None, "stoch_d": None,
            "overbought": False, "oversold": False,
            "bullish_cross": False, "bearish_cross": False,
        }


# ---------------------------------------------------------------------------
# ROCIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("roc")
class ROCIndicator(BaseIndicator):
    """Rate of Change over multiple periods.

    Outputs
    -------
    roc_5, roc_10, roc_20, roc_60 : float | None   Percentage ROC
    momentum_class : str    'strong_up', 'up', 'flat', 'down', 'strong_down'
    """

    @property
    def name(self) -> str:
        return "roc"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 10):
            return {
                "roc_5": None, "roc_10": None, "roc_20": None, "roc_60": None,
                "momentum_class": "flat",
            }

        close = df["close"]
        roc_vals: dict[str, float | None] = {}

        for period in [5, 10, 20, 60]:
            if len(close.dropna()) > period:
                roc_series = _roc(close, period)
                roc_vals[f"roc_{period}"] = self._safe_last(roc_series)
            else:
                roc_vals[f"roc_{period}"] = None

        # Round
        result = {k: (round(v, 3) if v is not None else None) for k, v in roc_vals.items()}

        # Classification based on roc_20
        roc20 = result.get("roc_20")
        if roc20 is None:
            momentum_class = "flat"
        elif roc20 >= 10:
            momentum_class = "strong_up"
        elif roc20 >= 3:
            momentum_class = "up"
        elif roc20 <= -10:
            momentum_class = "strong_down"
        elif roc20 <= -3:
            momentum_class = "down"
        else:
            momentum_class = "flat"

        result["momentum_class"] = momentum_class
        return result


# ---------------------------------------------------------------------------
# ADXIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("adx")
class ADXIndicator(BaseIndicator):
    """Average Directional Index (14): trend strength classification.

    Outputs
    -------
    adx : float | None
    plus_di : float | None
    minus_di : float | None
    trend_strength : str    'strong' (>25), 'moderate' (20-25), 'weak' (<20)
    di_bullish : bool       +DI > -DI
    trend_present : bool    ADX > 20
    """

    @property
    def name(self) -> str:
        return "adx"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 20):
            return self._empty()

        period = getattr(config, "adx_period", 14)
        adx_series, plus_di_series, minus_di_series = _adx(
            df["high"], df["low"], df["close"], period
        )

        adx_val = self._safe_last(adx_series)
        plus_di_val = self._safe_last(plus_di_series)
        minus_di_val = self._safe_last(minus_di_series)

        if adx_val is None:
            return self._empty()

        if adx_val > 25:
            strength = "strong"
        elif adx_val > 20:
            strength = "moderate"
        else:
            strength = "weak"

        di_bullish = (plus_di_val or 0) > (minus_di_val or 0)

        return {
            "adx": round(adx_val, 2),
            "plus_di": round(plus_di_val, 2) if plus_di_val else None,
            "minus_di": round(minus_di_val, 2) if minus_di_val else None,
            "trend_strength": strength,
            "di_bullish": di_bullish,
            "trend_present": adx_val > 20,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "adx": None, "plus_di": None, "minus_di": None,
            "trend_strength": "weak", "di_bullish": False, "trend_present": False,
        }


# ---------------------------------------------------------------------------
# RelativeStrengthIndicator
# ---------------------------------------------------------------------------


@IndicatorRegistry.register("relative_strength")
class RelativeStrengthIndicator(BaseIndicator):
    """Relative strength of the ticker vs SPY and vs sector ETF.

    RS is computed as the ratio of 20-period returns.
    If benchmark_df is not available via df.attrs, returns None values.

    Outputs
    -------
    rs_vs_spy : float | None      ticker_return / spy_return
    rs_vs_sector : float | None
    rs_vs_spy_52w : float | None  52-week version
    outperforming_spy : bool
    outperforming_sector : bool
    """

    @property
    def name(self) -> str:
        return "relative_strength"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, df: pd.DataFrame, config: Any) -> dict[str, Any]:
        if not self._require_min_bars(df, 20):
            return self._empty()

        close = df["close"].dropna()
        benchmark_df: pd.DataFrame | None = df.attrs.get("benchmark_df")
        sector_df: pd.DataFrame | None = df.attrs.get("sector_df")

        def _rs(ticker_close: pd.Series, bench_close: pd.Series, period: int) -> float | None:
            if len(ticker_close) < period or len(bench_close) < period:
                return None
            # Align on common index
            combined = pd.concat([ticker_close, bench_close], axis=1, join="inner")
            if len(combined) < period:
                return None
            t_ret = (combined.iloc[-1, 0] / combined.iloc[-period, 0]) - 1
            b_ret = (combined.iloc[-1, 1] / combined.iloc[-period, 1]) - 1
            if b_ret == 0:
                return None
            # Ratio: >1 means outperforming
            return (1 + t_ret) / (1 + b_ret) if b_ret != -1 else None

        rs_spy = None
        rs_spy_52w = None
        if benchmark_df is not None and not benchmark_df.empty:
            bench_close = benchmark_df["close"].dropna()
            rs_spy = _rs(close, bench_close, 20)
            rs_spy_52w = _rs(close, bench_close, min(252, len(close), len(bench_close)))

        rs_sector = None
        if sector_df is not None and not sector_df.empty:
            sector_close = sector_df["close"].dropna()
            rs_sector = _rs(close, sector_close, 20)

        return {
            "rs_vs_spy": round(rs_spy, 4) if rs_spy is not None else None,
            "rs_vs_sector": round(rs_sector, 4) if rs_sector is not None else None,
            "rs_vs_spy_52w": round(rs_spy_52w, 4) if rs_spy_52w is not None else None,
            "outperforming_spy": (rs_spy or 0) > 1.0,
            "outperforming_sector": (rs_sector or 0) > 1.0,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "rs_vs_spy": None, "rs_vs_sector": None, "rs_vs_spy_52w": None,
            "outperforming_spy": False, "outperforming_sector": False,
        }
