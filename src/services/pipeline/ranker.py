"""
Candidate ranker: compute total scores and sort candidates from best to worst.

The CandidateRanker delegates scoring to the ScoringEngine and returns a
list sorted by total_score descending, with ties broken by confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.pipeline.screener import RegimeAssessment, ScreenedCandidate
    from src.services.scoring.engine import ScoredCandidate, ScoringEngine

logger = structlog.get_logger(__name__)


class CandidateRanker:
    """Rank ScreenedCandidates by their total composite score.

    Usage::

        ranker = CandidateRanker(scoring_engine)
        ranked = ranker.rank(candidates, regime)
        best = ranked[0]  # highest total_score
    """

    def __init__(self, scoring_engine: "ScoringEngine") -> None:
        self.scoring_engine = scoring_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: list["ScreenedCandidate"],
        regime: "RegimeAssessment",
    ) -> list["ScoredCandidate"]:
        """Score all candidates and return them sorted best-first.

        Args:
            candidates: Candidates that have passed the MultiStageScreener.
            regime:     Current market regime assessment.

        Returns:
            List of ScoredCandidate sorted by total_score descending.
            Ties are broken by confidence (higher confidence wins).
        """
        if not candidates:
            logger.info("ranker_no_candidates")
            return []

        scored: list["ScoredCandidate"] = []
        for candidate in candidates:
            try:
                scored_candidate = self.scoring_engine.score(candidate, regime)
                scored.append(scored_candidate)
                logger.debug(
                    "ranker_scored",
                    ticker=candidate.ticker,
                    total_score=round(scored_candidate.total_score, 2),
                    confidence=round(scored_candidate.confidence, 2),
                )
            except Exception as exc:
                logger.warning(
                    "ranker_score_failed",
                    ticker=candidate.ticker,
                    error=str(exc),
                )

        # Sort: primary = total_score DESC, secondary = confidence DESC
        scored.sort(
            key=lambda c: (c.total_score, c.confidence),
            reverse=True,
        )

        logger.info(
            "ranker_complete",
            total_scored=len(scored),
            top_ticker=scored[0].ticker if scored else None,
            top_score=round(scored[0].total_score, 2) if scored else None,
        )
        return scored

    def rank_top_n(
        self,
        candidates: list["ScreenedCandidate"],
        regime: "RegimeAssessment",
        n: int = 5,
    ) -> list["ScoredCandidate"]:
        """Return only the top-N candidates after ranking.

        Useful when you want a shortlist rather than the full ranking.
        """
        ranked = self.rank(candidates, regime)
        return ranked[:n]
