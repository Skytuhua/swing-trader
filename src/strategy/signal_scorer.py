"""
Signal Confidence Scoring System — multi-factor signal quality assessment.

"Only trade when model is confident (>60%)" — reduces noise trades significantly.

Each signal is scored 0-100 across five factors:

1. Trend Alignment (0-25 points)
   - Multi-timeframe trend agreement (daily + 4H)
   - EMA alignment (fast > mid > slow for bullish)
   - Price position relative to key moving averages

2. Volume Confirmation (0-15 points)
   - Volume above/below average
   - Volume trend (increasing on breakouts)
   - OBV confirmation

3. Indicator Confluence (0-25 points)
   - RSI + MACD + EMA alignment agreement
   - Stochastic confirmation
   - Number of indicators confirming vs contradicting

4. Regime Compatibility (0-20 points)
   - How well the signal aligns with the current market regime
   - Bull signal in bull market = high compatibility
   - Counter-trend signals penalised

5. Relative Strength (0-15 points)
   - Performance vs sector / market benchmark
   - Stocks leading their sector get bonus points

Confidence Thresholds
---------------------
Score >= 75 : Full position size (aggressive entry)
Score 60-74 : Half position size (conservative entry)
Score 45-59 : Paper trade only (track for validation)
Score  < 45 : No trade

Self-Calibration
-----------------
The scorer tracks actual win rates per score bucket and adjusts thresholds
over time to maintain optimal signal quality.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SignalScore:
    """Comprehensive signal quality score."""

    total_score: float  # 0-100 composite
    confidence_tier: str  # "full", "half", "paper_only", "no_trade"

    # Component scores
    trend_alignment: float = 0.0  # 0-25
    volume_confirmation: float = 0.0  # 0-15
    indicator_confluence: float = 0.0  # 0-25
    regime_compatibility: float = 0.0  # 0-20
    relative_strength: float = 0.0  # 0-15

    # Position sizing recommendation
    position_scale: float = 0.0  # 0-1 (1 = full position)

    explanation: str = ""
    factors: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationBucket:
    """Tracks actual outcomes for a score bucket for self-calibration."""

    bucket_name: str
    min_score: float
    max_score: float
    total_trades: int = 0
    winning_trades: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def has_sufficient_data(self) -> bool:
        return self.total_trades >= 20


class SignalScorer:
    """Multi-factor signal confidence scoring system.

    Parameters
    ----------
    config : dict or dataclass, optional
        Configuration with:
        - min_score_to_trade: float (default 60)
        - full_position_score: float (default 75)
        - trend_weight: float (default 25)
        - volume_weight: float (default 15)
        - indicator_weight: float (default 25)
        - regime_weight: float (default 20)
        - relative_strength_weight: float (default 15)
    """

    def __init__(self, config: Any = None) -> None:
        cfg = config or {}
        self._get = (
            (lambda k, d: cfg.get(k, d))
            if isinstance(cfg, dict)
            else (lambda k, d: getattr(cfg, k, d))
        )

        self.min_score_to_trade: float = float(self._get("min_score_to_trade", 60.0))
        self.full_position_score: float = float(self._get("full_position_score", 75.0))
        self.trend_max: float = float(self._get("trend_weight", 25.0))
        self.volume_max: float = float(self._get("volume_weight", 15.0))
        self.indicator_max: float = float(self._get("indicator_weight", 25.0))
        self.regime_max: float = float(self._get("regime_weight", 20.0))
        self.rs_max: float = float(self._get("relative_strength_weight", 15.0))

        # Self-calibration buckets
        self._buckets: dict[str, CalibrationBucket] = {
            "high": CalibrationBucket("high", 75, 100),
            "medium": CalibrationBucket("medium", 60, 74.99),
            "low": CalibrationBucket("low", 45, 59.99),
            "reject": CalibrationBucket("reject", 0, 44.99),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        indicators: dict[str, Any],
        regime: Any = None,
        benchmark_return: float = 0.0,
        stock_return: float = 0.0,
        signal_side: str = "long",
    ) -> SignalScore:
        """Score a trading signal across five quality factors.

        Parameters
        ----------
        indicators : dict
            Technical indicators for the stock. Expected keys:
            - ema_fast, ema_slow, ema_200 (or equivalents)
            - rsi_14
            - macd_hist (MACD histogram)
            - stoch_k, stoch_d
            - volume_ratio (current / average)
            - obv_trend (positive / negative / flat)
            - atr_14
            - last_close
            - trend_score (0-100 from technical engine)
        regime : object, optional
            Regime detection result with .regime attribute.
        benchmark_return : float
            Benchmark (e.g. SPY) return over lookback period.
        stock_return : float
            Stock's return over the same period.
        signal_side : str
            "long" or "short" — direction of the proposed trade.

        Returns
        -------
        SignalScore with total score, component scores, and recommendation.
        """
        # 1. Trend Alignment (0-25)
        trend = self._score_trend_alignment(indicators, signal_side)

        # 2. Volume Confirmation (0-15)
        volume = self._score_volume_confirmation(indicators, signal_side)

        # 3. Indicator Confluence (0-25)
        confluence = self._score_indicator_confluence(indicators, signal_side)

        # 4. Regime Compatibility (0-20)
        regime_compat = self._score_regime_compatibility(regime, signal_side)

        # 5. Relative Strength (0-15)
        rs = self._score_relative_strength(stock_return, benchmark_return)

        total = trend + volume + confluence + regime_compat + rs
        total = max(0.0, min(100.0, total))

        # Determine confidence tier and position scale
        if total >= self.full_position_score:
            tier = "full"
            scale = 1.0
        elif total >= self.min_score_to_trade:
            tier = "half"
            # Linear interpolation between min_score and full_position
            scale = 0.5 + 0.5 * (total - self.min_score_to_trade) / (self.full_position_score - self.min_score_to_trade)
        elif total >= 45:
            tier = "paper_only"
            scale = 0.0
        else:
            tier = "no_trade"
            scale = 0.0

        explanation = (
            f"Signal score: {total:.1f}/100 ({tier}). "
            f"Trend={trend:.1f}/{self.trend_max:.0f}, "
            f"Volume={volume:.1f}/{self.volume_max:.0f}, "
            f"Confluence={confluence:.1f}/{self.indicator_max:.0f}, "
            f"Regime={regime_compat:.1f}/{self.regime_max:.0f}, "
            f"RS={rs:.1f}/{self.rs_max:.0f}."
        )

        logger.debug(
            "signal_scored",
            total=round(total, 1),
            tier=tier,
            trend=round(trend, 1),
            volume=round(volume, 1),
            confluence=round(confluence, 1),
            regime=round(regime_compat, 1),
            rs=round(rs, 1),
        )

        return SignalScore(
            total_score=round(total, 2),
            confidence_tier=tier,
            trend_alignment=round(trend, 2),
            volume_confirmation=round(volume, 2),
            indicator_confluence=round(confluence, 2),
            regime_compatibility=round(regime_compat, 2),
            relative_strength=round(rs, 2),
            position_scale=round(scale, 3),
            explanation=explanation,
            factors={
                "trend_details": self._trend_details(indicators),
                "volume_ratio": float(indicators.get("volume_ratio", 1.0) or 1.0),
                "rsi": float(indicators.get("rsi_14", 50) or 50),
                "macd_hist": float(indicators.get("macd_hist", 0) or 0),
            },
        )

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _score_trend_alignment(self, indicators: dict, signal_side: str) -> float:
        """Score trend alignment across multiple indicators (0 to trend_max).

        Checks EMA alignment, price vs key MAs, and trend_score from technical engine.
        """
        score = 0.0
        max_score = self.trend_max

        # EMA alignment check
        ema_fast = float(indicators.get("ema_fast", 0) or indicators.get("ema_8", 0) or 0)
        ema_slow = float(indicators.get("ema_slow", 0) or indicators.get("ema_21", 0) or 0)
        ema_200 = float(indicators.get("ema_200", 0) or 0)
        last_close = float(indicators.get("last_close", 0) or 0)

        if last_close > 0 and ema_fast > 0 and ema_slow > 0:
            if signal_side == "long":
                # Bullish alignment: price > fast > slow > 200
                if last_close > ema_fast:
                    score += max_score * 0.25
                if ema_fast > ema_slow:
                    score += max_score * 0.25
                if ema_200 > 0 and ema_slow > ema_200:
                    score += max_score * 0.25
                if ema_200 > 0 and last_close > ema_200:
                    score += max_score * 0.15
            else:
                # Bearish alignment
                if last_close < ema_fast:
                    score += max_score * 0.25
                if ema_fast < ema_slow:
                    score += max_score * 0.25
                if ema_200 > 0 and ema_slow < ema_200:
                    score += max_score * 0.25
                if ema_200 > 0 and last_close < ema_200:
                    score += max_score * 0.15

        # Technical engine trend score bonus
        trend_score = float(indicators.get("trend_score", 50) or 50)
        if signal_side == "long" and trend_score > 65:
            score += max_score * 0.10 * min(1.0, (trend_score - 65) / 25)
        elif signal_side == "short" and trend_score < 35:
            score += max_score * 0.10 * min(1.0, (35 - trend_score) / 25)

        return min(max_score, score)

    def _score_volume_confirmation(self, indicators: dict, signal_side: str) -> float:
        """Score volume confirmation (0 to volume_max).

        Higher volume on breakouts confirms the move; low volume is suspicious.
        """
        score = 0.0
        max_score = self.volume_max

        vol_ratio = float(indicators.get("volume_ratio", 1.0) or indicators.get("relative_volume", 1.0) or 1.0)

        # Volume ratio scoring
        if vol_ratio >= 2.0:
            score += max_score * 0.60  # Strong volume confirmation
        elif vol_ratio >= 1.5:
            score += max_score * 0.45
        elif vol_ratio >= 1.0:
            score += max_score * 0.25
        elif vol_ratio >= 0.7:
            score += max_score * 0.10
        # Below 0.7 = very low volume, suspicious

        # OBV trend confirmation
        obv_trend = indicators.get("obv_trend", "flat")
        if signal_side == "long" and obv_trend == "positive":
            score += max_score * 0.40
        elif signal_side == "short" and obv_trend == "negative":
            score += max_score * 0.40
        elif obv_trend == "flat":
            score += max_score * 0.10

        return min(max_score, score)

    def _score_indicator_confluence(self, indicators: dict, signal_side: str) -> float:
        """Score indicator confluence (0 to indicator_max).

        More indicators confirming the signal direction = higher score.
        Conflicting indicators reduce the score.
        """
        confirming = 0
        contradicting = 0
        total_indicators = 0

        # RSI
        rsi = float(indicators.get("rsi_14", 50) or 50)
        if signal_side == "long":
            if 30 <= rsi <= 60:  # Oversold recovering = bullish
                confirming += 1
            elif rsi > 75:  # Overbought = contradicting
                contradicting += 1
            else:
                confirming += 0.5
        else:
            if 40 <= rsi <= 70:
                confirming += 1
            elif rsi < 25:
                contradicting += 1
            else:
                confirming += 0.5
        total_indicators += 1

        # MACD histogram
        macd_hist = float(indicators.get("macd_hist", 0) or 0)
        if signal_side == "long":
            if macd_hist > 0:
                confirming += 1
            elif macd_hist < -0.5:
                contradicting += 1
        else:
            if macd_hist < 0:
                confirming += 1
            elif macd_hist > 0.5:
                contradicting += 1
        total_indicators += 1

        # Stochastic
        stoch_k = float(indicators.get("stoch_k", 50) or 50)
        stoch_d = float(indicators.get("stoch_d", 50) or 50)
        if signal_side == "long":
            if stoch_k > stoch_d and stoch_k < 80:
                confirming += 1
            elif stoch_k > 85:
                contradicting += 1
        else:
            if stoch_k < stoch_d and stoch_k > 20:
                confirming += 1
            elif stoch_k < 15:
                contradicting += 1
        total_indicators += 1

        # EMA cross
        ema_fast = float(indicators.get("ema_fast", 0) or indicators.get("ema_8", 0) or 0)
        ema_slow = float(indicators.get("ema_slow", 0) or indicators.get("ema_21", 0) or 0)
        if ema_fast > 0 and ema_slow > 0:
            if signal_side == "long" and ema_fast > ema_slow:
                confirming += 1
            elif signal_side == "short" and ema_fast < ema_slow:
                confirming += 1
            else:
                contradicting += 0.5
            total_indicators += 1

        if total_indicators == 0:
            return self.indicator_max * 0.5  # Neutral if no data

        max_score = self.indicator_max
        confirm_ratio = confirming / total_indicators
        contradict_ratio = contradicting / total_indicators

        score = max_score * confirm_ratio * 0.75
        score -= max_score * contradict_ratio * 0.50  # Contradictions reduce score
        score += max_score * 0.25 * (1 - contradict_ratio)  # Base points for no contradictions

        return max(0.0, min(max_score, score))

    def _score_regime_compatibility(self, regime: Any, signal_side: str) -> float:
        """Score regime compatibility (0 to regime_max).

        Bull regime + long signal = high compatibility.
        Counter-trend signals in strong regimes are heavily penalised.
        """
        max_score = self.regime_max

        if regime is None:
            return max_score * 0.5  # Neutral when no regime data

        regime_value = getattr(regime, "regime", None)
        if regime_value is None:
            regime_value = getattr(regime, "value", None)

        # Handle both string and enum
        regime_str = str(regime_value).lower() if regime_value else "unknown"

        if signal_side == "long":
            if regime_str in ("bull", "favorable"):
                return max_score * 1.0
            if regime_str in ("bull_weak", "mixed"):
                return max_score * 0.65
            if regime_str in ("sideways",):
                return max_score * 0.40
            if regime_str in ("bear_weak",):
                return max_score * 0.20
            if regime_str in ("bear", "unfavorable"):
                return max_score * 0.05  # Almost zero for counter-trend
            if regime_str in ("uncertain",):
                return max_score * 0.25
        else:  # short
            if regime_str in ("bear", "unfavorable"):
                return max_score * 1.0
            if regime_str in ("bear_weak",):
                return max_score * 0.65
            if regime_str in ("sideways",):
                return max_score * 0.40
            if regime_str in ("bull_weak", "mixed"):
                return max_score * 0.20
            if regime_str in ("bull", "favorable"):
                return max_score * 0.05
            if regime_str in ("uncertain",):
                return max_score * 0.25

        return max_score * 0.5

    def _score_relative_strength(self, stock_return: float, benchmark_return: float) -> float:
        """Score relative strength vs benchmark (0 to rs_max).

        Stocks outperforming their benchmark/sector are more likely to continue.
        """
        max_score = self.rs_max

        if benchmark_return == 0 and stock_return == 0:
            return max_score * 0.5  # Neutral

        excess_return = stock_return - benchmark_return

        if excess_return > 5.0:
            return max_score * 1.0  # Strong outperformance
        if excess_return > 2.0:
            return max_score * 0.80
        if excess_return > 0:
            return max_score * 0.60
        if excess_return > -2.0:
            return max_score * 0.35
        return max_score * 0.10  # Significant underperformance

    # ------------------------------------------------------------------
    # Self-calibration
    # ------------------------------------------------------------------

    def record_outcome(self, score: float, is_winner: bool) -> None:
        """Record the outcome of a trade for self-calibration.

        Parameters
        ----------
        score : float
            The signal score when the trade was taken.
        is_winner : bool
            Whether the trade was profitable.
        """
        bucket = self._get_bucket(score)
        if bucket:
            bucket.total_trades += 1
            if is_winner:
                bucket.winning_trades += 1

    def get_calibration_stats(self) -> dict[str, dict[str, Any]]:
        """Get win rate statistics per score bucket for review."""
        return {
            name: {
                "range": f"{b.min_score:.0f}-{b.max_score:.0f}",
                "trades": b.total_trades,
                "wins": b.winning_trades,
                "win_rate": round(b.win_rate, 3),
                "sufficient_data": b.has_sufficient_data,
            }
            for name, b in self._buckets.items()
        }

    def auto_adjust_thresholds(self) -> dict[str, float]:
        """Adjust trading thresholds based on calibration data.

        Only adjusts if sufficient data exists. Returns the new thresholds.
        """
        adjustments = {}

        # If the "medium" bucket has poor win rate, raise the min_score_to_trade
        medium = self._buckets["medium"]
        if medium.has_sufficient_data and medium.win_rate < 0.45:
            self.min_score_to_trade = min(75.0, self.min_score_to_trade + 2.0)
            adjustments["min_score_to_trade"] = self.min_score_to_trade
            logger.info(
                "signal_scorer_threshold_raised",
                new_min_score=self.min_score_to_trade,
                reason=f"Medium bucket win rate {medium.win_rate:.2%} below 45%",
            )

        # If the "high" bucket has very high win rate, we could lower full_position_score
        high = self._buckets["high"]
        if high.has_sufficient_data and high.win_rate > 0.70:
            self.full_position_score = max(65.0, self.full_position_score - 1.0)
            adjustments["full_position_score"] = self.full_position_score

        return adjustments

    def _get_bucket(self, score: float) -> CalibrationBucket | None:
        for bucket in self._buckets.values():
            if bucket.min_score <= score <= bucket.max_score:
                return bucket
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_details(indicators: dict) -> dict[str, Any]:
        return {
            "ema_fast": float(indicators.get("ema_fast", 0) or indicators.get("ema_8", 0) or 0),
            "ema_slow": float(indicators.get("ema_slow", 0) or indicators.get("ema_21", 0) or 0),
            "ema_200": float(indicators.get("ema_200", 0) or 0),
            "last_close": float(indicators.get("last_close", 0) or 0),
        }

    def should_trade(self, score: SignalScore) -> bool:
        """Quick check: should we take this trade?"""
        return score.total_score >= self.min_score_to_trade
