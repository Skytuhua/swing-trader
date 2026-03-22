"""
Scoring engine: transparent, weighted scoring with a multi-factor penalty system.

Each ScreenedCandidate is evaluated against six components (technical, news,
sentiment, liquidity, regime alignment, risk/reward quality), combined with
configurable weights, then reduced by any applicable penalties.  A confidence
score (separate from total_score) quantifies how trustworthy the signal is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import DataQuality, MarketRegime

if TYPE_CHECKING:
    from src.core.config import ScoringConfig
    from src.services.pipeline.screener import RegimeAssessment, ScreenedCandidate
    from src.services.scoring.weights import WeightProfile

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """A candidate enriched with all scoring detail."""

    ticker: str
    total_score: float                      # 0-100 final composite
    confidence: float                       # 0-100 confidence in signal

    # Component scores (each 0-100)
    technical_score: float = 0.0
    news_score: float = 0.0
    sentiment_score: float = 0.0
    liquidity_score: float = 0.0
    regime_score: float = 0.0
    risk_reward_score: float = 0.0

    # Penalty breakdown – key → amount subtracted
    penalties: dict[str, float] = field(default_factory=dict)

    # Human-readable explanation (populated by ScoringEngine)
    explanation: str = ""

    # Forward reference to original candidate for downstream consumers
    screened_candidate: Any = None

    # Extra metadata (e.g. raw R:R ratio)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


class ScoringEngine:
    """Transparent, weighted scoring with configurable weights and penalty system.

    Usage::

        engine = ScoringEngine(weights=MODERATE)
        scored = engine.score(candidate, regime)
    """

    def __init__(self, weights: "WeightProfile | ScoringConfig") -> None:
        # Accept either a WeightProfile or the ScoringConfig from core.config
        self.w = weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        candidate: "ScreenedCandidate",
        regime: "RegimeAssessment",
    ) -> ScoredCandidate:
        """Score a single candidate.

        Args:
            candidate: A ScreenedCandidate from the pipeline screener.
            regime:    Current market regime assessment.

        Returns:
            ScoredCandidate with total_score, confidence, component scores,
            penalties, and explanation.
        """
        # ---- Component scores (0-100 each) ----
        tech = float(candidate.technical.composite_score)
        news = float(candidate.news.score)
        sentiment = float(candidate.sentiment.score)
        liquidity = self._score_liquidity(candidate)
        regime_align = self._score_regime_alignment(candidate.technical, regime)
        rr = self._score_risk_reward(candidate)

        # ---- Weighted composite ----
        tw = self._get_weight("technical_weight", 0.30)
        nw = self._get_weight("news_weight", 0.15)
        sw = self._get_weight("sentiment_weight", 0.10)
        lw = self._get_weight("liquidity_weight", 0.10)
        rw = self._get_weight("regime_weight", 0.15)
        rrw = self._get_weight("risk_reward_weight", 0.20)

        raw_score = (
            tech * tw
            + news * nw
            + sentiment * sw
            + liquidity * lw
            + regime_align * rw
            + rr * rrw
        )

        # ---- Penalty system ----
        penalties: dict[str, float] = {}

        # 1. RSI extension / overheat
        rsi_14 = float(candidate.technical.indicators.get("rsi_14", 50) or 50)
        if rsi_14 > 75:
            penalty = (rsi_14 - 75) * 0.5
            raw_score -= penalty
            penalties["rsi_overextended"] = round(penalty, 2)

        # 2. Earnings proximity (within 5 days)
        if candidate.has_earnings_within(days=5):
            raw_score -= 15.0
            penalties["earnings_proximity"] = 15.0

        # 3. Data quality
        if candidate.data_quality != DataQuality.GOOD:
            raw_score -= 10.0
            penalties["data_quality"] = 10.0

        # 4. Sentiment hype / crowding
        crowding_risk = float(getattr(candidate.sentiment, "crowding_risk", 0) or 0)
        if crowding_risk > 50:
            penalty = crowding_risk * 0.15
            raw_score -= penalty
            penalties["sentiment_crowding"] = round(penalty, 2)

        # 5. Extension penalty – price far above key MAs
        extension_penalty = self._extension_penalty(candidate)
        if extension_penalty > 0:
            raw_score -= extension_penalty
            penalties["price_extension"] = round(extension_penalty, 2)

        total = max(0.0, min(100.0, raw_score))
        confidence = self._compute_confidence(total, penalties, regime)

        # Build explanation
        explanation = self._explain(
            candidate, total, confidence, penalties,
            tech, news, sentiment, liquidity, regime_align, rr,
        )

        scored = ScoredCandidate(
            ticker=candidate.ticker,
            total_score=round(total, 2),
            confidence=round(confidence, 2),
            technical_score=round(tech, 2),
            news_score=round(news, 2),
            sentiment_score=round(sentiment, 2),
            liquidity_score=round(liquidity, 2),
            regime_score=round(regime_align, 2),
            risk_reward_score=round(rr, 2),
            penalties=penalties,
            explanation=explanation,
            screened_candidate=candidate,
            extra={},
        )

        logger.debug(
            "scoring_engine_scored",
            ticker=candidate.ticker,
            total=round(total, 2),
            confidence=round(confidence, 2),
            penalties=penalties,
        )
        return scored

    def score_many(
        self,
        candidates: "list[ScreenedCandidate]",
        regime: "RegimeAssessment",
    ) -> "list[ScoredCandidate]":
        """Score a list of candidates (synchronous, no concurrency needed)."""
        return [self.score(c, regime) for c in candidates]

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _score_liquidity(self, candidate: "ScreenedCandidate") -> float:
        """Score liquidity 0-100 based on available indicators.

        Uses relative_volume and avg_dollar_volume from the technical profile.
        Falls back to 50 (neutral) when data is absent.
        """
        indicators = candidate.technical.indicators

        # Relative volume: > 1.5x average is excellent
        rel_vol = float(indicators.get("relative_volume", 1.0) or 1.0)
        rel_vol_score = min(100.0, max(0.0, (rel_vol - 0.5) / 1.5 * 100.0))

        # ATR percent: low ATR% = low spread/slippage risk
        atr_pct = float(indicators.get("atr_pct", 2.0) or 2.0)
        # < 1% ATR = great (100), > 5% ATR = poor (0)
        atr_score = max(0.0, min(100.0, (5.0 - atr_pct) / 4.0 * 100.0))

        return round((rel_vol_score * 0.6 + atr_score * 0.4), 2)

    def _score_regime_alignment(
        self,
        technical: Any,
        regime: "RegimeAssessment",
    ) -> float:
        """Score how well the stock's trend aligns with the current market regime."""
        trend_score = float(getattr(technical, "trend_score", 50.0) or 50.0)

        multiplier: float
        if regime.regime == MarketRegime.FAVORABLE:
            multiplier = 1.0
        elif regime.regime == MarketRegime.MIXED:
            multiplier = 0.7
        else:  # UNFAVORABLE
            multiplier = 0.3

        # Blend stock's own trend quality with regime fit
        regime_confidence = float(getattr(regime, "confidence", 50.0) or 50.0)
        regime_factor = (regime_confidence / 100.0) * multiplier

        return round(trend_score * multiplier * 0.7 + regime_factor * 100.0 * 0.3, 2)

    def _score_risk_reward(self, candidate: "ScreenedCandidate") -> float:
        """Score 0-100 based on implied R:R from technical structure.

        Uses ATR and distance to support as a proxy for stop distance,
        then checks whether there is room to a resistance level.
        """
        indicators = candidate.technical.indicators
        key_levels = getattr(candidate.technical, "key_levels", {}) or {}

        atr = float(indicators.get("atr_14", 0) or 0)
        last_close = float(indicators.get("last_close", 0) or 0)

        if last_close <= 0 or atr <= 0:
            return 50.0  # Neutral when data is missing

        # Estimated stop distance (1.5-2x ATR typical)
        stop_dist = atr * 1.75

        # Distance to nearest resistance
        resistance = key_levels.get("resistance") or key_levels.get("nearest_resistance")
        if resistance and float(resistance) > last_close:
            reward_dist = float(resistance) - last_close
        else:
            # Assume 3 ATR to target when no resistance level available
            reward_dist = atr * 3.0

        if stop_dist <= 0:
            return 50.0

        rr = reward_dist / stop_dist

        # Map R:R to 0-100:
        # rr ≥ 3   → 100
        # rr = 2   → 75
        # rr = 1.5 → 50
        # rr < 1   → 0
        score = min(100.0, max(0.0, (rr - 1.0) / 2.0 * 100.0))
        return round(score, 2)

    def _extension_penalty(self, candidate: "ScreenedCandidate") -> float:
        """Penalty for price being far extended above key moving averages."""
        indicators = candidate.technical.indicators
        last_close = float(indicators.get("last_close", 0) or 0)
        sma_20 = float(indicators.get("sma_20", 0) or 0)

        if last_close <= 0 or sma_20 <= 0:
            return 0.0

        extension_pct = (last_close / sma_20 - 1.0) * 100.0
        # More than 10% above 20-SMA is extended; penalise linearly beyond
        if extension_pct > 10.0:
            return min(20.0, (extension_pct - 10.0) * 0.8)
        return 0.0

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        total: float,
        penalties: dict[str, float],
        regime: "RegimeAssessment",
    ) -> float:
        """0-100 confidence score – separate from the total score.

        Confidence reflects *certainty* in the signal, not just raw quality.
        It is reduced when many penalties fire or when the regime is uncertain.
        """
        base = total

        # Each penalty type reduces confidence by 3 points
        penalty_count = len(penalties)
        base -= penalty_count * 3.0

        # Regime adjustment
        if regime.regime == MarketRegime.MIXED:
            base *= 0.85
        elif regime.regime == MarketRegime.UNFAVORABLE:
            base *= 0.60

        # Also factor in regime confidence (e.g. 50% confident = less reliable)
        regime_confidence = float(getattr(regime, "confidence", 80.0) or 80.0)
        regime_factor = 0.7 + 0.3 * (regime_confidence / 100.0)
        base *= regime_factor

        return round(max(0.0, min(100.0, base)), 2)

    # ------------------------------------------------------------------
    # Explanation builder
    # ------------------------------------------------------------------

    def _explain(
        self,
        candidate: "ScreenedCandidate",
        total: float,
        confidence: float,
        penalties: dict[str, float],
        tech: float,
        news: float,
        sentiment: float,
        liquidity: float,
        regime_align: float,
        rr: float,
    ) -> str:
        parts: list[str] = [
            f"{candidate.ticker} scored {total:.1f}/100 (confidence {confidence:.1f}/100).",
        ]

        # Strengths
        strong: list[str] = []
        if tech >= 70:
            ts = getattr(candidate.technical, "trend_score", None)
            ms = getattr(candidate.technical, "momentum_score", None)
            desc = f"technical={tech:.0f}"
            if ts is not None:
                desc += f" (trend={ts:.0f}"
            if ms is not None:
                desc += f", momentum={ms:.0f}"
            if ts is not None or ms is not None:
                desc += ")"
            strong.append(desc)
        if news >= 65:
            strong.append(f"positive news={news:.0f}")
        if sentiment >= 65:
            strong.append(f"sentiment={sentiment:.0f}")
        if regime_align >= 70:
            strong.append(f"regime alignment={regime_align:.0f}")
        if rr >= 70:
            strong.append(f"strong R:R={rr:.0f}")

        if strong:
            parts.append("Strengths: " + ", ".join(strong) + ".")

        # Penalties
        if penalties:
            pen_parts = [
                f"{k.replace('_', ' ')} (−{v:.1f})" for k, v in penalties.items()
            ]
            parts.append("Penalties applied: " + "; ".join(pen_parts) + ".")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_weight(self, attr: str, default: float) -> float:
        return float(getattr(self.w, attr, default) or default)
