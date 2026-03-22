"""SentimentScorer — final sentiment score for a ticker.

Takes the SentimentProfile from SentimentAggregator and produces a final
0-100 sentiment score with contextual adjustments:

1. Start from the aggregator's base score.
2. Apply lifecycle-phase modifier:
   - EARLY  : slight bonus (potential early opportunity)
   - RISING : significant bonus (momentum building)
   - PEAKING: slight penalty (risk of reversal)
   - FADING : significant penalty (momentum dying)
3. Apply crowding/hype penalty when risk > threshold.
4. Detect sentiment/price divergence when price data is available:
   - Bearish price action + bullish sentiment → reduce score (false signal risk)
   - Bullish price action + bearish sentiment → minor penalty (selling pressure)
5. Reduce score confidence when noise is high (low mention count or single source).
6. Produce a 0-100 final score with detailed explanation.

The scorer is stateless — all context is passed as arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.core.enums import SentimentPhase
from src.services.sentiment.base import SentimentProfile

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SentimentScore:
    """Final sentiment score output consumed by the ranking pipeline."""

    score: float
    """0-100 composite score (50 = neutral)."""

    base_score: float
    """Raw aggregator score before adjustments."""

    phase: SentimentPhase
    """Detected sentiment lifecycle phase."""

    crowding_risk: float
    """0-100 crowding/hype risk (higher = noisier)."""

    intensity: float
    """Volume z-score vs baseline."""

    confidence: float
    """0-1 confidence in the score given data quality and volume."""

    divergence_detected: bool = False
    """True when sentiment and price trend diverge significantly."""

    trust_adjusted: bool = False
    """True when the score was reduced due to noise/low data quality."""

    adjustments: list[tuple[str, float]] = field(default_factory=list)
    """List of (reason, delta) applied to the base score."""

    explanation: str = ""


# ---------------------------------------------------------------------------
# Phase modifiers
# ---------------------------------------------------------------------------

_PHASE_MODIFIER: dict[SentimentPhase, float] = {
    SentimentPhase.EARLY:   +3.0,   # early discovery: slight bonus
    SentimentPhase.RISING:  +8.0,   # momentum building: good timing
    SentimentPhase.PEAKING: -5.0,   # near top: reversal risk
    SentimentPhase.FADING:  -12.0,  # momentum gone: avoid
}


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------


def _detect_divergence(
    sentiment_polarity: float,
    price_change_pct: Optional[float],
    threshold: float = 0.05,  # 5% price move
) -> bool:
    """Detect meaningful sentiment/price divergence.

    A divergence exists when:
    - Sentiment is clearly bullish (polarity > 0.2) but price has fallen
      more than ``threshold`` recently, OR
    - Sentiment is clearly bearish (polarity < -0.2) but price has risen
      more than ``threshold``.

    Returns True when divergence is detected (reduces our trust in the signal).
    """
    if price_change_pct is None:
        return False
    if sentiment_polarity > 0.2 and price_change_pct < -threshold:
        return True
    if sentiment_polarity < -0.2 and price_change_pct > threshold:
        return True
    return False


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class SentimentScorer:
    """Compute the final sentiment score from an aggregated SentimentProfile.

    Parameters
    ----------
    crowding_threshold:
        Crowding risk above which a penalty is applied.  Default 30.
    max_crowding_penalty:
        Maximum score deduction for high crowding.  Default 15.
    min_mention_count_for_high_confidence:
        Minimum clean mentions needed to give full confidence.  Default 10.
    noise_trust_reduction:
        Score is pulled toward 50 by this fraction when data is noisy.
        Default 0.3 (30% pull toward neutral).
    divergence_penalty:
        Score deduction applied when sentiment/price divergence is detected.
        Default 8.
    """

    def __init__(
        self,
        crowding_threshold: float = 30.0,
        max_crowding_penalty: float = 15.0,
        min_mention_count_for_high_confidence: int = 10,
        noise_trust_reduction: float = 0.30,
        divergence_penalty: float = 8.0,
    ) -> None:
        self.crowding_threshold = crowding_threshold
        self.max_crowding_penalty = max_crowding_penalty
        self.min_mention_count = min_mention_count_for_high_confidence
        self.noise_trust_reduction = noise_trust_reduction
        self.divergence_penalty = divergence_penalty

    def score(
        self,
        profile: SentimentProfile,
        price_change_pct: Optional[float] = None,
    ) -> SentimentScore:
        """Compute the final sentiment score.

        Args:
            profile: Aggregated SentimentProfile from SentimentAggregator.
            price_change_pct: Optional recent price change (e.g. 3-day).
                              Positive = up, negative = down.  Used for divergence
                              detection only.

        Returns:
            SentimentScore with final 0-100 score and full explanation.
        """
        adjustments: list[tuple[str, float]] = []
        working_score = profile.score

        # 1. Phase modifier
        phase_delta = _PHASE_MODIFIER.get(profile.phase, 0.0)
        if phase_delta != 0.0:
            working_score += phase_delta
            adjustments.append((f"phase_{profile.phase.value}", phase_delta))
            logger.debug(
                "sentiment_scorer_phase_adj",
                ticker=profile.ticker,
                phase=profile.phase.value,
                delta=phase_delta,
            )

        # 2. Crowding/hype penalty
        crowding_penalty = 0.0
        if profile.crowding_risk > self.crowding_threshold:
            excess = profile.crowding_risk - self.crowding_threshold
            crowding_penalty = min(
                self.max_crowding_penalty,
                excess * (self.max_crowding_penalty / (100.0 - self.crowding_threshold)),
            )
            working_score -= crowding_penalty
            adjustments.append(("crowding_penalty", -crowding_penalty))
            logger.debug(
                "sentiment_scorer_crowding_penalty",
                ticker=profile.ticker,
                crowding_risk=profile.crowding_risk,
                penalty=crowding_penalty,
            )

        # 3. Sentiment/price divergence
        divergence = _detect_divergence(profile.polarity, price_change_pct)
        if divergence:
            working_score -= self.divergence_penalty
            adjustments.append(("divergence_penalty", -self.divergence_penalty))
            logger.info(
                "sentiment_scorer_divergence",
                ticker=profile.ticker,
                polarity=profile.polarity,
                price_change_pct=price_change_pct,
            )

        # 4. Noise / low-confidence adjustment
        trust_adjusted = False
        confidence = self._compute_confidence(profile)

        if confidence < 0.5:
            # Pull score toward neutral (50) proportionally to noise level
            trust_factor = self.noise_trust_reduction * (1.0 - confidence / 0.5)
            pull = (50.0 - working_score) * trust_factor
            working_score += pull
            adjustments.append(("noise_trust_reduction", round(pull, 2)))
            trust_adjusted = True
            logger.debug(
                "sentiment_scorer_noise_reduction",
                ticker=profile.ticker,
                confidence=round(confidence, 3),
                pull=round(pull, 2),
            )

        # 5. Clamp
        final_score = max(0.0, min(100.0, working_score))

        # 6. Build explanation
        explanation = self._build_explanation(
            profile=profile,
            base_score=profile.score,
            final_score=final_score,
            adjustments=adjustments,
            divergence=divergence,
            confidence=confidence,
            price_change_pct=price_change_pct,
        )

        logger.info(
            "sentiment_scorer_done",
            ticker=profile.ticker,
            base_score=round(profile.score, 2),
            final_score=round(final_score, 2),
            phase=profile.phase.value,
            confidence=round(confidence, 3),
            divergence=divergence,
        )

        return SentimentScore(
            score=round(final_score, 2),
            base_score=round(profile.score, 2),
            phase=profile.phase,
            crowding_risk=profile.crowding_risk,
            intensity=profile.intensity,
            confidence=round(confidence, 3),
            divergence_detected=divergence,
            trust_adjusted=trust_adjusted,
            adjustments=adjustments,
            explanation=explanation,
        )

    def _compute_confidence(self, profile: SentimentProfile) -> float:
        """Return a 0-1 confidence value based on data quality and volume.

        Factors:
        - Mention count vs minimum threshold
        - Number of distinct sources (multi-source = more reliable)
        - Data quality flag from the aggregator
        """
        quality_weights = {"good": 1.0, "degraded": 0.6, "unavailable": 0.1}
        quality_factor = quality_weights.get(profile.data_quality, 0.5)

        # Volume factor: 0 at 0 mentions → 1.0 at min_mention_count
        volume_factor = min(1.0, profile.mention_count / max(self.min_mention_count, 1))

        # Source diversity factor
        source_count = len(profile.sources)
        source_factor = min(1.0, source_count / 2.0)  # full confidence at 2+ sources

        confidence = quality_factor * 0.4 + volume_factor * 0.4 + source_factor * 0.2
        return round(min(1.0, confidence), 4)

    @staticmethod
    def _build_explanation(
        profile: SentimentProfile,
        base_score: float,
        final_score: float,
        adjustments: list[tuple[str, float]],
        divergence: bool,
        confidence: float,
        price_change_pct: Optional[float],
    ) -> str:
        lines: list[str] = [
            f"{profile.ticker} sentiment: {profile.mention_count} clean mentions. "
            f"Base score {base_score:.1f}/100 → Final {final_score:.1f}/100.",
            f"Phase: {profile.phase.value} | Polarity: {profile.polarity:+.2f} | "
            f"Intensity: {profile.intensity:+.1f}σ | Confidence: {confidence:.0%}.",
        ]

        for reason, delta in adjustments:
            sign = "+" if delta >= 0 else ""
            lines.append(f"  Adjustment [{reason}]: {sign}{delta:.1f} pts.")

        if divergence:
            direction = "down" if (price_change_pct or 0) < 0 else "up"
            lines.append(
                f"  ⚠ Divergence: sentiment is {'bullish' if profile.polarity > 0 else 'bearish'} "
                f"but price moved {direction} ({price_change_pct:+.1f}%)."
            )

        if profile.crowding_risk > 50:
            lines.append(
                f"  ⚠ High crowding risk ({profile.crowding_risk:.0f}/100) — "
                "retail frenzy may reverse sharply."
            )

        return " ".join(lines)
