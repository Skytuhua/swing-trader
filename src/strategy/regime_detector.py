"""
Market Regime Detection Filter — multi-indicator, multi-timeframe regime classification.

Classifies the market into one of six regimes based on ADX trend strength,
EMA alignment, Bollinger Band width, RSI momentum, and volume confirmation.
Each regime maps to a strategy adaptation (bias, aggression, trailing multiplier)
that downstream components use to filter trades and adjust sizing.

Regimes
-------
BULL        — Strong uptrend: ADX>25, EMA aligned up, momentum bullish.
BEAR        — Strong downtrend: ADX>25, EMA aligned down, momentum bearish.
SIDEWAYS    — Range-bound: ADX<20, narrow BB width, no clear direction.
BULL_WEAK   — Weakening uptrend: some bull signals but mixed confirmation.
BEAR_WEAK   — Weakening downtrend: some bear signals but mixed confirmation.
UNCERTAIN   — Conflicting signals across timeframes or indicators.

Multi-timeframe approach
------------------------
Daily bars provide the primary regime classification; 4-hour bars provide
confirmation.  A regime is high-confidence only when both timeframes agree.

Strategy adaptation per regime
------------------------------
BULL:      long_only bias, full aggression (1.0), trailing multiplier 2.5
BEAR:      short_only/cash, low aggression (0.3), tight trailing 1.5
SIDEWAYS:  mean_reversion bias, moderate aggression (0.5), trailing 2.0
BULL_WEAK: long bias with caution, reduced aggression (0.7), trailing 2.0
BEAR_WEAK: defensive/cash, low aggression (0.4), trailing 1.8
UNCERTAIN: cash, zero aggression, no trailing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


class DetailedRegime(str, Enum):
    """Fine-grained market regime classification."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    BULL_WEAK = "bull_weak"
    BEAR_WEAK = "bear_weak"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Strategy adaptation map
# ---------------------------------------------------------------------------

REGIME_STRATEGY: dict[DetailedRegime, dict[str, Any]] = {
    DetailedRegime.BULL: {
        "bias": "long_only",
        "aggression": 1.0,
        "trail_mult": 2.5,
        "position_scale": 1.0,
    },
    DetailedRegime.BEAR: {
        "bias": "short_only_or_cash",
        "aggression": 0.3,
        "trail_mult": 1.5,
        "position_scale": 0.3,
    },
    DetailedRegime.SIDEWAYS: {
        "bias": "mean_reversion",
        "aggression": 0.5,
        "trail_mult": 2.0,
        "position_scale": 0.5,
    },
    DetailedRegime.BULL_WEAK: {
        "bias": "long_cautious",
        "aggression": 0.7,
        "trail_mult": 2.0,
        "position_scale": 0.7,
    },
    DetailedRegime.BEAR_WEAK: {
        "bias": "defensive",
        "aggression": 0.4,
        "trail_mult": 1.8,
        "position_scale": 0.4,
    },
    DetailedRegime.UNCERTAIN: {
        "bias": "cash",
        "aggression": 0.0,
        "trail_mult": 0.0,
        "position_scale": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Regime assessment dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegimeDetection:
    """Full regime detection output."""

    regime: DetailedRegime
    confidence: float  # 0-1 how confident we are in the classification
    adx: float = 0.0
    adx_trending: bool = False
    ema_alignment: str = "neutral"  # bullish / bearish / neutral
    bb_width: float = 0.0
    rsi: float = 50.0
    volume_ratio: float = 1.0
    strategy: dict = field(default_factory=dict)
    primary_timeframe: str = "1d"
    confirmation_timeframe: str = "4h"
    primary_regime: Optional[str] = None
    confirmation_regime: Optional[str] = None
    scores: dict = field(default_factory=dict)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Indicator computation helpers
# ---------------------------------------------------------------------------


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute the Average Directional Index.

    ADX measures trend strength regardless of direction.
    ADX > 25 indicates a trending market; ADX < 20 indicates ranging.

    Uses the Wilder smoothing method: EMA with alpha = 1/period.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Wilder smoothing
    alpha = 1.0 / period
    atr = _wilder_smooth(tr[1:], period, alpha)
    plus_di_raw = _wilder_smooth(plus_dm[1:], period, alpha)
    minus_di_raw = _wilder_smooth(minus_dm[1:], period, alpha)

    if atr <= 0:
        return 0.0

    plus_di = 100.0 * plus_di_raw / atr
    minus_di = 100.0 * minus_di_raw / atr

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0

    dx = 100.0 * abs(plus_di - minus_di) / di_sum
    return float(dx)


def _wilder_smooth(data: np.ndarray, period: int, alpha: float) -> float:
    """Apply Wilder smoothing and return the final value."""
    if len(data) < period:
        return float(np.mean(data)) if len(data) > 0 else 0.0

    # Seed with SMA
    result = float(np.mean(data[:period]))
    for i in range(period, len(data)):
        result = result * (1 - alpha) + data[i] * alpha
    return result


def compute_ema(data: np.ndarray, period: int) -> float:
    """Compute the final EMA value for the given period."""
    if len(data) < period:
        return float(np.mean(data)) if len(data) > 0 else 0.0

    alpha = 2.0 / (period + 1)
    ema = float(np.mean(data[:period]))
    for i in range(period, len(data)):
        ema = data[i] * alpha + ema * (1 - alpha)
    return ema


def compute_bb_width(closes: np.ndarray, period: int = 20, std_mult: float = 2.0) -> float:
    """Compute Bollinger Band width as a fraction of the middle band.

    BB width = (upper - lower) / middle.
    Narrow width (<0.05) suggests consolidation / sideways market.
    """
    if len(closes) < period:
        return 0.0

    recent = closes[-period:]
    middle = float(np.mean(recent))
    if middle <= 0:
        return 0.0

    std = float(np.std(recent, ddof=1))
    width = (2 * std_mult * std) / middle
    return width


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute the Relative Strength Index.

    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder smooth
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
    """Compute current volume as a ratio of the N-period average."""
    if len(volumes) < period + 1:
        return 1.0

    avg_vol = float(np.mean(volumes[-(period + 1):-1]))
    if avg_vol <= 0:
        return 1.0

    return float(volumes[-1]) / avg_vol


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Classify market regime using ADX, EMA alignment, BB width, RSI, and volume.

    Parameters
    ----------
    config : dict or dataclass
        Configuration with keys/attributes:
        - ema_periods: list[int]  (default [21, 55, 200])
        - adx_period: int         (default 14)
        - adx_trending_threshold: float (default 25)
        - adx_ranging_threshold: float  (default 20)
        - bb_width_threshold: float     (default 0.05)
        - rsi_period: int               (default 14)
        - volume_period: int            (default 20)
    """

    def __init__(self, config: Any = None) -> None:
        self._cfg = config or {}
        self.ema_periods: list[int] = self._get("ema_periods", [21, 55, 200])
        self.adx_period: int = self._get("adx_period", 14)
        self.adx_trending: float = self._get("adx_trending_threshold", 25.0)
        self.adx_ranging: float = self._get("adx_ranging_threshold", 20.0)
        self.bb_width_threshold: float = self._get("bb_width_threshold", 0.05)
        self.rsi_period: int = self._get("rsi_period", 14)
        self.volume_period: int = self._get("volume_period", 20)

    def _get(self, key: str, default: Any) -> Any:
        if isinstance(self._cfg, dict):
            return self._cfg.get(key, default)
        return getattr(self._cfg, key, default)

    def detect(self, df: Any) -> RegimeDetection:
        """Classify regime from a single-timeframe OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.

        Returns
        -------
        RegimeDetection with regime, confidence, indicators, and strategy map.
        """
        closes = np.array(df["close"].values, dtype=float)
        highs = np.array(df["high"].values, dtype=float)
        lows = np.array(df["low"].values, dtype=float)
        volumes = np.array(df["volume"].values, dtype=float)

        # Compute indicators
        adx = compute_adx(highs, lows, closes, self.adx_period)
        adx_trending = adx > self.adx_trending
        adx_ranging = adx < self.adx_ranging

        # EMA alignment
        ema_values = [compute_ema(closes, p) for p in self.ema_periods]
        ema_alignment = self._classify_ema_alignment(ema_values, closes[-1] if len(closes) > 0 else 0)

        # Bollinger Band width
        bb_width = compute_bb_width(closes, period=20, std_mult=2.0)

        # RSI
        rsi = compute_rsi(closes, self.rsi_period)

        # Volume ratio
        vol_ratio = compute_volume_ratio(volumes, self.volume_period)

        # Score each indicator's contribution to regime classification
        scores = self._compute_scores(adx, adx_trending, adx_ranging, ema_alignment, bb_width, rsi, vol_ratio)

        # Classify based on composite scores
        regime, confidence = self._classify(scores, adx_trending, adx_ranging, ema_alignment, bb_width)

        strategy = REGIME_STRATEGY.get(regime, REGIME_STRATEGY[DetailedRegime.UNCERTAIN])

        explanation = (
            f"Regime={regime.value} (conf={confidence:.2f}). "
            f"ADX={adx:.1f} ({'trending' if adx_trending else 'ranging' if adx_ranging else 'transitional'}), "
            f"EMA alignment={ema_alignment}, BB width={bb_width:.4f}, "
            f"RSI={rsi:.1f}, Vol ratio={vol_ratio:.2f}."
        )

        logger.info(
            "regime_detected",
            regime=regime.value,
            confidence=round(confidence, 3),
            adx=round(adx, 1),
            ema_alignment=ema_alignment,
            bb_width=round(bb_width, 4),
            rsi=round(rsi, 1),
            vol_ratio=round(vol_ratio, 2),
        )

        return RegimeDetection(
            regime=regime,
            confidence=round(confidence, 4),
            adx=round(adx, 2),
            adx_trending=adx_trending,
            ema_alignment=ema_alignment,
            bb_width=round(bb_width, 4),
            rsi=round(rsi, 2),
            volume_ratio=round(vol_ratio, 2),
            strategy=strategy,
            scores=scores,
            explanation=explanation,
        )

    def detect_multi_timeframe(self, daily_df: Any, four_hour_df: Any | None = None) -> RegimeDetection:
        """Classify regime using multiple timeframes for higher confidence.

        The daily timeframe provides the primary classification.
        The 4H timeframe provides confirmation — disagreement lowers confidence.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily OHLCV data (primary timeframe).
        four_hour_df : pd.DataFrame, optional
            4-hour OHLCV data (confirmation timeframe).

        Returns
        -------
        RegimeDetection with multi-timeframe context.
        """
        primary = self.detect(daily_df)
        primary.primary_timeframe = "1d"
        primary.primary_regime = primary.regime.value

        if four_hour_df is None or len(four_hour_df) < 20:
            return primary

        confirmation = self.detect(four_hour_df)
        primary.confirmation_timeframe = "4h"
        primary.confirmation_regime = confirmation.regime.value

        # Adjust confidence based on agreement
        if confirmation.regime == primary.regime:
            # Full agreement — boost confidence
            primary.confidence = min(1.0, primary.confidence * 1.15)
        elif self._regimes_compatible(primary.regime, confirmation.regime):
            # Compatible (e.g. BULL + BULL_WEAK) — slight reduction
            primary.confidence *= 0.90
        else:
            # Disagreement — significant confidence reduction
            primary.confidence *= 0.65
            # If disagreement is strong, downgrade to UNCERTAIN
            if primary.confidence < 0.3:
                primary.regime = DetailedRegime.UNCERTAIN
                primary.strategy = REGIME_STRATEGY[DetailedRegime.UNCERTAIN]

        primary.explanation += (
            f" | 4H confirmation: {confirmation.regime.value} "
            f"(ADX={confirmation.adx:.1f}, EMA={confirmation.ema_alignment})."
        )

        return primary

    # ------------------------------------------------------------------
    # Internal classification logic
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_ema_alignment(ema_values: list[float], current_price: float) -> str:
        """Determine EMA alignment direction.

        Bullish: price > EMA_short > EMA_mid > EMA_long (or 2/3 conditions met).
        Bearish: price < EMA_short < EMA_mid < EMA_long.
        Neutral: mixed alignment.
        """
        if len(ema_values) < 3:
            return "neutral"

        short, mid, long_ = ema_values[0], ema_values[1], ema_values[2]

        bull_signals = 0
        bear_signals = 0

        if current_price > short:
            bull_signals += 1
        else:
            bear_signals += 1

        if short > mid:
            bull_signals += 1
        else:
            bear_signals += 1

        if mid > long_:
            bull_signals += 1
        else:
            bear_signals += 1

        if bull_signals >= 3:
            return "bullish"
        if bear_signals >= 3:
            return "bearish"
        if bull_signals >= 2:
            return "bullish_weak"
        if bear_signals >= 2:
            return "bearish_weak"
        return "neutral"

    @staticmethod
    def _compute_scores(
        adx: float,
        adx_trending: bool,
        adx_ranging: bool,
        ema_alignment: str,
        bb_width: float,
        rsi: float,
        vol_ratio: float,
    ) -> dict[str, float]:
        """Compute individual indicator scores contributing to regime classification.

        Returns a dict of score name → value in [-1, +1] where:
        +1 = strongly bullish, -1 = strongly bearish, 0 = neutral.
        """
        scores: dict[str, float] = {}

        # Trend strength score (0 to 1, regardless of direction)
        scores["trend_strength"] = min(1.0, adx / 50.0) if adx_trending else 0.0

        # Direction score from EMA alignment
        ema_map = {
            "bullish": 1.0,
            "bullish_weak": 0.5,
            "neutral": 0.0,
            "bearish_weak": -0.5,
            "bearish": -1.0,
        }
        scores["direction"] = ema_map.get(ema_alignment, 0.0)

        # Volatility contraction (low BB width = sideways/consolidation)
        scores["volatility"] = 0.0 if bb_width < 0.05 else min(1.0, bb_width / 0.15)

        # RSI momentum bias
        if rsi > 60:
            scores["momentum"] = min(1.0, (rsi - 50) / 30)
        elif rsi < 40:
            scores["momentum"] = max(-1.0, (rsi - 50) / 30)
        else:
            scores["momentum"] = 0.0

        # Volume confirmation
        if vol_ratio > 1.3:
            scores["volume_confirm"] = min(1.0, (vol_ratio - 1.0) / 1.0)
        elif vol_ratio < 0.7:
            scores["volume_confirm"] = -0.3  # Low volume = less conviction
        else:
            scores["volume_confirm"] = 0.0

        return scores

    @staticmethod
    def _classify(
        scores: dict[str, float],
        adx_trending: bool,
        adx_ranging: bool,
        ema_alignment: str,
        bb_width: float,
    ) -> tuple[DetailedRegime, float]:
        """Map indicator scores to a regime classification and confidence level."""
        direction = scores.get("direction", 0.0)
        trend_strength = scores.get("trend_strength", 0.0)
        momentum = scores.get("momentum", 0.0)

        # Composite directional score
        composite = direction * 0.40 + momentum * 0.30 + scores.get("volume_confirm", 0) * 0.15 + trend_strength * 0.15 * (1 if direction > 0 else -1 if direction < 0 else 0)

        if adx_ranging and bb_width < 0.05:
            regime = DetailedRegime.SIDEWAYS
            confidence = 0.5 + 0.5 * (1.0 - min(1.0, bb_width / 0.05))
        elif adx_trending and direction >= 0.8 and momentum > 0:
            regime = DetailedRegime.BULL
            confidence = 0.6 + 0.4 * min(1.0, trend_strength)
        elif adx_trending and direction <= -0.8 and momentum < 0:
            regime = DetailedRegime.BEAR
            confidence = 0.6 + 0.4 * min(1.0, trend_strength)
        elif composite > 0.3:
            regime = DetailedRegime.BULL_WEAK
            confidence = 0.4 + 0.3 * min(1.0, composite)
        elif composite < -0.3:
            regime = DetailedRegime.BEAR_WEAK
            confidence = 0.4 + 0.3 * min(1.0, abs(composite))
        else:
            regime = DetailedRegime.UNCERTAIN
            confidence = 0.2

        return regime, round(min(1.0, confidence), 4)

    @staticmethod
    def _regimes_compatible(primary: DetailedRegime, confirmation: DetailedRegime) -> bool:
        """Check if two regimes are compatible (same directional family)."""
        bull_family = {DetailedRegime.BULL, DetailedRegime.BULL_WEAK}
        bear_family = {DetailedRegime.BEAR, DetailedRegime.BEAR_WEAK}

        if primary in bull_family and confirmation in bull_family:
            return True
        if primary in bear_family and confirmation in bear_family:
            return True
        if primary == DetailedRegime.SIDEWAYS and confirmation == DetailedRegime.SIDEWAYS:
            return True
        return False


def should_trade(detection: RegimeDetection) -> bool:
    """Quick check: should we trade given the current regime detection?

    Returns False for UNCERTAIN regime or when aggression is 0.
    """
    strategy = detection.strategy
    return strategy.get("aggression", 0) > 0 and detection.regime != DetailedRegime.UNCERTAIN
