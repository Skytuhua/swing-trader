"""NewsScorer — aggregate news signal for a ticker into a 0-100 score.

Scoring formula
---------------
1. Filter news items to those mentioning the ticker.
2. For each item compute a time-decay weight:
       weight = exp(-age_hours / HALF_LIFE_HOURS)
   where HALF_LIFE_HOURS = 72 (configurable).
3. Compute a weighted average of ``sentiment_score × relevance_score``.
4. Map the weighted average to a 0-100 scale: score = 50 + avg * 50.
5. Detect catalyst types (earnings, upgrade, negative catalyst).
6. Detect conflicting signals (both strongly positive and negative items).
7. Apply bonuses/penalties for catalysts and conflicts.
8. Build a human-readable explanation string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.services.news.base import NewsItem, NewsScore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Catalyst keyword sets
# ---------------------------------------------------------------------------

_EARNINGS_KEYWORDS = frozenset({
    "earnings", "revenue", "eps", "beats estimates", "misses estimates",
    "quarterly results", "annual results", "profit", "guidance", "outlook",
    "q1", "q2", "q3", "q4", "full year", "fiscal",
})

_UPGRADE_KEYWORDS = frozenset({
    "upgrade", "raised price target", "raises pt", "initiates buy",
    "outperform", "overweight", "strong buy", "price target increase",
    "bull case", "upgraded",
})

_DOWNGRADE_KEYWORDS = frozenset({
    "downgrade", "cut price target", "cuts pt", "underperform",
    "underweight", "sell", "price target decrease", "downgraded",
})

_NEGATIVE_EVENT_KEYWORDS = frozenset({
    "lawsuit", "fraud", "sec investigation", "recall", "data breach",
    "layoffs", "bankruptcy", "default", "restatement", "class action",
    "warning", "profit warning", "debt", "going concern",
})

_POSITIVE_EVENT_KEYWORDS = frozenset({
    "partnership", "contract win", "new product", "launch",
    "patent", "approval", "fda approval", "record revenue",
    "dividend increase", "buyback", "share repurchase",
})


def _kw_match(text: str, keywords: frozenset[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Individual item analysis
# ---------------------------------------------------------------------------


@dataclass
class _ItemAnalysis:
    """Per-article breakdown used during scoring."""

    item: NewsItem
    age_hours: float
    decay: float
    weighted_score: float  # sentiment_score * relevance_score * decay
    is_earnings: bool
    is_upgrade: bool
    is_downgrade: bool
    is_negative_event: bool
    is_positive_event: bool


def _analyse_item(item: NewsItem, current_time: datetime) -> _ItemAnalysis:
    if item.published_at.tzinfo is None:
        published_at = item.published_at.replace(tzinfo=timezone.utc)
    else:
        published_at = item.published_at

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    age_hours = (current_time - published_at).total_seconds() / 3600.0
    age_hours = max(0.0, age_hours)  # clamp — don't penalise future-dated articles

    # 72-hour half-life: decay = exp(-age / 72)
    decay = math.exp(-age_hours / 72.0)

    # Weighted score: combines model sentiment, relevance, and recency
    weighted = item.sentiment_score * item.relevance_score * decay

    text = f"{item.headline} {item.summary}"
    return _ItemAnalysis(
        item=item,
        age_hours=age_hours,
        decay=decay,
        weighted_score=weighted,
        is_earnings=_kw_match(text, _EARNINGS_KEYWORDS) or "earnings" in item.category,
        is_upgrade=_kw_match(text, _UPGRADE_KEYWORDS),
        is_downgrade=_kw_match(text, _DOWNGRADE_KEYWORDS),
        is_negative_event=_kw_match(text, _NEGATIVE_EVENT_KEYWORDS) or item.sentiment_score < -0.3,
        is_positive_event=_kw_match(text, _POSITIVE_EVENT_KEYWORDS) or item.sentiment_score > 0.3,
    )


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------


def _build_explanation(
    ticker: str,
    analyses: list[_ItemAnalysis],
    base_score: float,
    final_score: float,
    catalyst_types: list[str],
    has_conflict: bool,
) -> str:
    """Produce a human-readable explanation of the news score."""
    lines: list[str] = []
    count = len(analyses)

    if not analyses:
        return f"No recent news found for {ticker}."

    # Summary line
    lines.append(
        f"{ticker}: {count} news item{'s' if count != 1 else ''} analysed. "
        f"Base weighted sentiment → {base_score:.1f}/100. Final score: {final_score:.1f}/100."
    )

    # Catalyst summary
    if catalyst_types:
        lines.append(f"Catalysts detected: {', '.join(catalyst_types)}.")

    if has_conflict:
        lines.append("⚠ Conflicting signals: both strongly positive and negative coverage present.")

    # Top 3 most impactful recent items
    top = sorted(analyses, key=lambda a: abs(a.weighted_score), reverse=True)[:3]
    if top:
        lines.append("Top influential items:")
        for a in top:
            direction = "+" if a.item.sentiment_score >= 0 else ""
            lines.append(
                f"  [{a.item.source}] \"{a.item.headline[:80]}\" | "
                f"sentiment={direction}{a.item.sentiment_score:.2f} | "
                f"age={a.age_hours:.1f}h | decay={a.decay:.2f}"
            )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# NewsScorer
# ---------------------------------------------------------------------------


class NewsScorer:
    """Aggregate news items for a ticker and produce a 0-100 score.

    The score anchors at 50 (neutral). Positive sentiment pushes it higher;
    negative sentiment pushes it lower.  Catalysts apply additional adjustments.

    Parameters
    ----------
    half_life_hours:
        Time-decay half-life in hours.  Default 72 (3 days).
    min_confidence:
        FinBERT confidence threshold; items below this are down-weighted.
        Default 0.4.
    earnings_bonus:
        Score bonus applied when an earnings catalyst with positive sentiment
        is detected.  Default 8.
    negative_penalty:
        Score penalty applied when a confirmed negative catalyst is detected.
        Default 10.
    conflict_penalty:
        Score penalty applied when conflicting signals are detected.
        Default 5.
    """

    def __init__(
        self,
        half_life_hours: float = 72.0,
        min_confidence: float = 0.40,
        earnings_bonus: float = 8.0,
        negative_penalty: float = 10.0,
        conflict_penalty: float = 5.0,
    ) -> None:
        self.half_life_hours = half_life_hours
        self.min_confidence = min_confidence
        self.earnings_bonus = earnings_bonus
        self.negative_penalty = negative_penalty
        self.conflict_penalty = conflict_penalty

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score_for_ticker(
        self,
        ticker: str,
        news_items: list[NewsItem],
        current_time: Optional[datetime] = None,
    ) -> NewsScore:
        """Score aggregated news impact for ``ticker`` over a ~3-5 day horizon.

        Args:
            ticker: Target equity symbol.
            news_items: All enriched news items (may contain multiple tickers).
            current_time: Reference time for age calculations (defaults to UTC now).

        Returns:
            NewsScore with 0-100 composite score and metadata.
        """
        if current_time is None:
            current_time = datetime.now(tz=timezone.utc)

        # 1. Filter to items that mention this ticker
        ticker_upper = ticker.upper()
        relevant = [
            item for item in news_items
            if ticker_upper in [t.upper() for t in item.tickers]
            or ticker_upper in (item.headline + item.summary).upper()
        ]

        if not relevant:
            logger.info("news_scorer_no_items", ticker=ticker)
            return NewsScore(
                score=50.0,
                catalyst_present=False,
                explanation=f"No recent news found for {ticker}.",
                item_count=0,
            )

        # 2. Analyse each item
        analyses = [_analyse_item(item, current_time) for item in relevant]

        # 3. Filter low-confidence items to reduce noise
        high_conf = [a for a in analyses if a.item.confidence >= self.min_confidence]
        effective = high_conf if len(high_conf) >= 2 else analyses

        # 4. Compute weighted scores
        total_decay = sum(a.decay for a in effective)
        if total_decay == 0:
            total_decay = 1.0

        # Normalise by total decay weight for a proper weighted average
        weighted_avg = sum(a.weighted_score for a in effective) / total_decay

        # 5. Map to 0-100 scale: 50 + weighted_avg * 50
        base_score = 50.0 + weighted_avg * 50.0
        base_score = max(0.0, min(100.0, base_score))

        # 6. Detect catalysts
        has_earnings = any(a.is_earnings for a in analyses)
        has_upgrade = any(a.is_upgrade for a in analyses)
        has_downgrade = any(a.is_downgrade for a in analyses)
        has_negative_event = any(a.is_negative_event for a in analyses)
        has_positive_event = any(a.is_positive_event for a in analyses)

        catalyst_present = has_earnings or has_upgrade or has_positive_event
        negative_catalyst = has_negative_event or has_downgrade

        # 7. Detect conflicting signals
        sentiments = [item.sentiment_score for item in relevant]
        has_conflict = (
            len(sentiments) >= 2
            and max(sentiments) > 0.30
            and min(sentiments) < -0.30
        )

        # 8. Apply adjustments
        adjustment = 0.0

        if has_earnings and weighted_avg > 0:
            adjustment += self.earnings_bonus
            logger.debug("news_scorer_earnings_bonus", ticker=ticker, bonus=self.earnings_bonus)

        if has_upgrade and weighted_avg > 0:
            adjustment += self.earnings_bonus * 0.5  # smaller upgrade bonus

        if has_negative_event or has_downgrade:
            adjustment -= self.negative_penalty
            logger.debug("news_scorer_negative_penalty", ticker=ticker, penalty=self.negative_penalty)

        if has_conflict:
            adjustment -= self.conflict_penalty
            logger.debug("news_scorer_conflict_penalty", ticker=ticker)

        final_score = max(0.0, min(100.0, base_score + adjustment))

        # 9. Build catalyst type list for explanation
        catalyst_types: list[str] = []
        if has_earnings:
            catalyst_types.append("earnings")
        if has_upgrade:
            catalyst_types.append("analyst upgrade")
        if has_downgrade:
            catalyst_types.append("analyst downgrade")
        if has_negative_event:
            catalyst_types.append("negative event")
        if has_positive_event:
            catalyst_types.append("positive event")

        # 10. Compute effective decay-weighted item count
        decay_count = total_decay

        explanation = _build_explanation(
            ticker=ticker,
            analyses=effective,
            base_score=base_score,
            final_score=final_score,
            catalyst_types=catalyst_types,
            has_conflict=has_conflict,
        )

        logger.info(
            "news_scorer_scored",
            ticker=ticker,
            item_count=len(relevant),
            base_score=round(base_score, 2),
            final_score=round(final_score, 2),
            catalyst_present=catalyst_present,
            negative_catalyst=negative_catalyst,
            conflicting_signals=has_conflict,
        )

        return NewsScore(
            score=round(final_score, 2),
            catalyst_present=catalyst_present,
            negative_catalyst=negative_catalyst,
            conflicting_signals=has_conflict,
            item_count=len(relevant),
            explanation=explanation,
            weighted_sentiment=round(weighted_avg, 4),
            decay_weighted_count=round(decay_count, 2),
        )

    def score_batch(
        self,
        tickers: list[str],
        news_items: list[NewsItem],
        current_time: Optional[datetime] = None,
    ) -> dict[str, NewsScore]:
        """Score multiple tickers from the same news pool.

        Args:
            tickers: List of equity symbols to score.
            news_items: Shared pool of enriched news items.
            current_time: Reference time for age calculations.

        Returns:
            Dict mapping ticker → NewsScore.
        """
        if current_time is None:
            current_time = datetime.now(tz=timezone.utc)

        return {
            ticker: self.score_for_ticker(ticker, news_items, current_time)
            for ticker in tickers
        }
