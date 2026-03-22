"""SentimentAggregator — combine social mentions from all providers.

Pipeline
--------
1. Merge all mentions from reddit, stocktwits, etc.
2. Spam/noise removal (duplicate authors, very-low-character posts, etc.)
3. Compute weighted average polarity (upvote-weighted).
4. Compute intensity z-score vs a rolling baseline mention volume.
5. Detect lifecycle phase: EARLY / RISING / PEAKING / FADING.
6. Compute crowding / hype risk.
7. Return SentimentProfile with 0-100 score and metadata.

Phase detection heuristic
--------------------------
- EARLY  : Low volume, moderate positive drift (under the radar).
- RISING : Volume accelerating, polarity positive and increasing.
- PEAKING: High volume (intensity > 2σ), polarity possibly plateauing.
- FADING : Volume declining from peak, polarity reversing or dropping.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import structlog

from src.core.enums import SentimentPhase
from src.services.sentiment.base import SentimentProfile, SentimentProvider, SocialMention

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Spam detection helpers
# ---------------------------------------------------------------------------

_MIN_TEXT_LENGTH = 15
_MAX_AUTHOR_MENTIONS = 5  # one author can't contribute more than this
_SPAM_PATTERNS = [
    re.compile(r"\b(follow|subscribe|dm me|visit|check out)\b", re.IGNORECASE),
    re.compile(r"https?://\S+"),                    # pure-URL posts
    re.compile(r"(.)\1{6,}"),                       # "AAAAAAA..." repetition
    re.compile(r"\$\d+\s*(giveaway|free|win)", re.IGNORECASE),
]


def _is_spam(mention: SocialMention) -> bool:
    """Heuristic spam/noise filter."""
    text = mention.text.strip()
    if len(text) < _MIN_TEXT_LENGTH:
        return True
    for pattern in _SPAM_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _remove_spam(mentions: list[SocialMention]) -> list[SocialMention]:
    """Remove spam posts and cap per-author contribution."""
    author_count: Counter[str] = Counter()
    clean: list[SocialMention] = []
    for m in mentions:
        if _is_spam(m):
            continue
        if author_count[m.author] >= _MAX_AUTHOR_MENTIONS:
            continue
        author_count[m.author] += 1
        clean.append(m)
    return clean


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------


def _detect_phase(
    recent_count: int,
    older_count: int,
    avg_polarity: float,
    intensity: float,
) -> SentimentPhase:
    """Classify the sentiment lifecycle phase.

    Args:
        recent_count: Mentions in the last 6 hours.
        older_count: Mentions from 6–24 hours ago.
        avg_polarity: Average polarity of all clean mentions.
        intensity: Volume z-score.
    """
    total = recent_count + older_count
    if total == 0:
        return SentimentPhase.EARLY

    recent_share = recent_count / max(total, 1)

    if intensity < 0.5:
        # Low volume — early stage regardless of direction
        return SentimentPhase.EARLY

    if intensity >= 2.0:
        # Very high volume — either peaking or fading
        if recent_share >= 0.6:
            return SentimentPhase.PEAKING
        else:
            return SentimentPhase.FADING

    # Moderate volume
    if recent_share >= 0.55 and avg_polarity > 0.1:
        return SentimentPhase.RISING

    if recent_share < 0.40:
        return SentimentPhase.FADING

    return SentimentPhase.RISING


# ---------------------------------------------------------------------------
# Baseline volume estimator (simple rolling window)
# ---------------------------------------------------------------------------


class _VolumeBaseline:
    """Tracks a rolling mean and std-dev of per-ticker mention counts.

    Uses an exponential moving average for efficiency.
    """

    def __init__(
        self,
        initial_mean: float = 10.0,
        initial_std: float = 5.0,
        alpha: float = 0.1,
    ) -> None:
        self._mean = initial_mean
        self._var = initial_std ** 2
        self._alpha = alpha

    def z_score(self, count: float) -> float:
        std = max(self._var ** 0.5, 1.0)
        return (count - self._mean) / std

    def update(self, count: float) -> None:
        self._mean = (1 - self._alpha) * self._mean + self._alpha * count
        diff = count - self._mean
        self._var = (1 - self._alpha) * self._var + self._alpha * diff ** 2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return max(self._var ** 0.5, 1.0)


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------


class SentimentAggregator:
    """Combine social mentions into a SentimentProfile.

    Parameters
    ----------
    providers:
        List of SentimentProvider instances to query.
    initial_baseline_volume:
        Expected mentions per ticker per day under normal conditions.
        Used for intensity z-score calculation.  Default 10.
    initial_baseline_std:
        Standard deviation of the baseline.  Default 5.
    """

    def __init__(
        self,
        providers: list[SentimentProvider] | None = None,
        initial_baseline_volume: float = 10.0,
        initial_baseline_std: float = 5.0,
    ) -> None:
        self._providers = providers or []
        # Per-ticker baselines; new tickers use the global default
        self._baselines: dict[str, _VolumeBaseline] = {}
        self._global_baseline = _VolumeBaseline(
            initial_mean=initial_baseline_volume,
            initial_std=initial_baseline_std,
        )

    def _get_baseline(self, ticker: str) -> _VolumeBaseline:
        if ticker not in self._baselines:
            self._baselines[ticker] = _VolumeBaseline(
                initial_mean=self._global_baseline.mean,
                initial_std=self._global_baseline.std,
            )
        return self._baselines[ticker]

    # ------------------------------------------------------------------
    # Core aggregation (synchronous — takes already-fetched mentions)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        ticker: str,
        mentions: list[SocialMention],
        reference_time: Optional[datetime] = None,
    ) -> SentimentProfile:
        """Produce a SentimentProfile from raw mentions.

        Args:
            ticker: The equity symbol being analysed.
            mentions: All raw mentions (may be unfiltered).
            reference_time: UTC reference for recency calculations (default: now).
        """
        if reference_time is None:
            reference_time = datetime.now(tz=timezone.utc)

        if not mentions:
            logger.info("sentiment_aggregator_no_mentions", ticker=ticker)
            return SentimentProfile(
                ticker=ticker,
                score=50.0,
                phase=SentimentPhase.EARLY,
                data_quality="unavailable",
                explanation=f"No social mentions found for {ticker}.",
            )

        # 1. Spam removal
        clean = _remove_spam(mentions)
        spam_removed = len(mentions) - len(clean)
        if spam_removed:
            logger.debug("sentiment_spam_removed", ticker=ticker, count=spam_removed)

        if not clean:
            return SentimentProfile(
                ticker=ticker,
                score=50.0,
                phase=SentimentPhase.EARLY,
                mention_count=0,
                data_quality="degraded",
                explanation=f"All {len(mentions)} mentions flagged as spam.",
            )

        # 2. Compute weighted polarity (upvote-weight)
        total_weight = sum(m.weight for m in clean)
        if total_weight == 0:
            total_weight = len(clean)

        avg_polarity = sum(m.polarity * m.weight for m in clean) / total_weight

        # 3. Bullish / bearish breakdown
        bullish_count = sum(1 for m in clean if m.is_bullish)
        bearish_count = sum(1 for m in clean if m.is_bearish)
        bullish_pct = bullish_count / len(clean)
        bearish_pct = bearish_count / len(clean)

        # 4. Intensity z-score
        baseline = self._get_baseline(ticker)
        volume_zscore = baseline.z_score(len(clean))
        baseline.update(len(clean))

        # 5. Phase detection
        six_hours_ago = reference_time - timedelta(hours=6)
        recent = [m for m in clean if m.timestamp > six_hours_ago]
        older = [m for m in clean if m.timestamp <= six_hours_ago]
        phase = _detect_phase(
            recent_count=len(recent),
            older_count=len(older),
            avg_polarity=avg_polarity,
            intensity=volume_zscore,
        )

        # 6. Crowding / hype risk
        # Risk increases sharply when volume is >2σ above baseline
        if volume_zscore > 2.0:
            crowding_risk = min(100.0, (volume_zscore - 2.0) * 20.0)
        else:
            crowding_risk = 0.0

        # Extra crowding signal: if a single source dominates (>80% of mentions)
        source_counts: Counter = Counter(m.source for m in clean)
        max_source_pct = max(source_counts.values()) / len(clean)
        if max_source_pct > 0.80:
            crowding_risk = min(100.0, crowding_risk + 15.0)

        # 7. Map polarity to 0-100 score
        score = max(0.0, min(100.0, 50.0 + avg_polarity * 30.0))

        # 8. Data quality
        sources = list({str(m.source) for m in clean})
        if len(sources) >= 2 and len(clean) >= 5:
            quality = "good"
        elif len(clean) >= 2:
            quality = "degraded"
        else:
            quality = "degraded"

        # 9. Explanation
        explanation = self._build_explanation(
            ticker=ticker,
            clean=clean,
            avg_polarity=avg_polarity,
            volume_zscore=volume_zscore,
            phase=phase,
            crowding_risk=crowding_risk,
            score=score,
        )

        logger.info(
            "sentiment_aggregated",
            ticker=ticker,
            mention_count=len(clean),
            avg_polarity=round(avg_polarity, 3),
            intensity_zscore=round(volume_zscore, 2),
            phase=phase.value,
            crowding_risk=round(crowding_risk, 1),
            score=round(score, 2),
        )

        return SentimentProfile(
            ticker=ticker,
            score=round(score, 2),
            polarity=round(avg_polarity, 4),
            intensity=round(volume_zscore, 3),
            phase=phase,
            crowding_risk=round(crowding_risk, 2),
            mention_count=len(clean),
            bullish_pct=round(bullish_pct, 3),
            bearish_pct=round(bearish_pct, 3),
            sources=sources,
            data_quality=quality,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Async fetch + aggregate
    # ------------------------------------------------------------------

    async def fetch_and_aggregate(
        self,
        ticker: str,
        since: datetime,
        reference_time: Optional[datetime] = None,
    ) -> SentimentProfile:
        """Query all registered providers and aggregate the results.

        Args:
            ticker: Equity symbol.
            since: Fetch mentions published after this UTC datetime.
            reference_time: UTC reference for phase detection (default: now).
        """
        if not self._providers:
            logger.warning("sentiment_no_providers", ticker=ticker)
            return SentimentProfile(
                ticker=ticker,
                score=50.0,
                phase=SentimentPhase.EARLY,
                data_quality="unavailable",
                explanation="No sentiment providers configured.",
            )

        import asyncio

        tasks = [provider.fetch_mentions(ticker, since) for provider in self._providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_mentions: list[SocialMention] = []
        for provider, result in zip(self._providers, results):
            provider_name = type(provider).__name__
            if isinstance(result, Exception):
                logger.warning(
                    "sentiment_provider_error",
                    provider=provider_name,
                    error=str(result),
                )
            else:
                logger.debug(
                    "sentiment_provider_fetched",
                    provider=provider_name,
                    count=len(result),
                )
                all_mentions.extend(result)

        return self.aggregate(ticker, all_mentions, reference_time=reference_time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        ticker: str,
        clean: list[SocialMention],
        avg_polarity: float,
        volume_zscore: float,
        phase: SentimentPhase,
        crowding_risk: float,
        score: float,
    ) -> str:
        direction = "positive" if avg_polarity > 0.05 else ("negative" if avg_polarity < -0.05 else "neutral")
        lines = [
            f"{ticker}: {len(clean)} clean mentions analysed. "
            f"Avg polarity {avg_polarity:+.2f} ({direction}). "
            f"Volume intensity {volume_zscore:+.1f}σ vs baseline. "
            f"Phase: {phase.value}. Score: {score:.1f}/100.",
        ]
        if crowding_risk > 20:
            lines.append(
                f"⚠ High crowding/hype risk ({crowding_risk:.0f}/100) — "
                "elevated volume may indicate a crowded trade."
            )
        # Source breakdown
        source_counts: Counter = Counter(str(m.source) for m in clean)
        src_str = ", ".join(f"{s}: {c}" for s, c in source_counts.most_common())
        lines.append(f"Sources: {src_str}.")

        return " ".join(lines)
