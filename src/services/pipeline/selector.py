"""
Final selector: pick the best candidate or decide NO TRADE.

Applies confidence threshold, minimum R:R, and regime alignment checks
to the top-ranked ScoredCandidate and either returns the selected ticker
or a NoTradeDecision with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from src.core.enums import MarketRegime

if TYPE_CHECKING:
    from src.core.config import ScoringConfig
    from src.services.pipeline.screener import RegimeAssessment
    from src.services.scoring.engine import ScoredCandidate

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class NoTradeDecision:
    """Returned when the selector decides not to trade."""

    reason: str
    detail: str = ""
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    top_candidate: str | None = None
    top_score: float | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"NO TRADE – {self.reason}" + (f": {self.detail}" if self.detail else "")


@dataclass
class SelectionResult:
    """Returned when a trade is selected."""

    ticker: str
    scored_candidate: "ScoredCandidate"
    selected_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"SELECT {self.ticker} "
            f"(score={self.scored_candidate.total_score:.1f}, "
            f"confidence={self.scored_candidate.confidence:.1f})"
        )


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

# Default thresholds (overridden by config where available)
_DEFAULT_CONFIDENCE_THRESHOLD = 60.0
_DEFAULT_MIN_RR = 1.5
_DEFAULT_MIN_SCORE = 55.0


class FinalSelector:
    """Select the best trading candidate from a ranked list.

    Decision logic:
      1. If the ranked list is empty → NO TRADE.
      2. If the top candidate's total_score < min_score → NO TRADE.
      3. If confidence < confidence_threshold → NO TRADE.
      4. If regime == UNFAVORABLE → NO TRADE (never trade unfavorable).
      5. If risk_reward_ratio < min_rr → NO TRADE.
      6. Otherwise → SELECT top candidate.
    """

    def __init__(self, config: "ScoringConfig | None" = None) -> None:
        self.config = config
        self._confidence_threshold: float = (
            config.confidence_threshold
            if config and hasattr(config, "confidence_threshold")
            else _DEFAULT_CONFIDENCE_THRESHOLD
        )
        self._min_rr: float = (
            config.min_risk_reward
            if config and hasattr(config, "min_risk_reward")
            else _DEFAULT_MIN_RR
        )
        self._min_score: float = (
            config.no_trade_threshold
            if config and hasattr(config, "no_trade_threshold")
            else _DEFAULT_MIN_SCORE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        ranked_candidates: list["ScoredCandidate"],
        regime: "RegimeAssessment",
    ) -> SelectionResult | NoTradeDecision:
        """Pick the best candidate or return a NoTradeDecision.

        Args:
            ranked_candidates: Candidates sorted best-first by CandidateRanker.
            regime:            Current market regime.

        Returns:
            SelectionResult or NoTradeDecision.
        """
        # ---- Empty list ----
        if not ranked_candidates:
            logger.info("selector_no_trade_empty_list")
            return NoTradeDecision(
                reason="no_candidates",
                detail="No candidates survived the screening pipeline.",
            )

        top = ranked_candidates[0]
        log = logger.bind(ticker=top.ticker, score=round(top.total_score, 2))

        # ---- Unfavorable regime ----
        if regime.regime == MarketRegime.UNFAVORABLE:
            log.info("selector_no_trade_unfavorable_regime")
            return NoTradeDecision(
                reason="unfavorable_regime",
                detail="Market regime is UNFAVORABLE. No new trades allowed.",
                top_candidate=top.ticker,
                top_score=top.total_score,
            )

        # ---- Minimum score threshold ----
        if top.total_score < self._min_score:
            log.info(
                "selector_no_trade_score_too_low",
                score=top.total_score,
                threshold=self._min_score,
            )
            return NoTradeDecision(
                reason="score_below_threshold",
                detail=(
                    f"Top candidate score {top.total_score:.1f} "
                    f"< threshold {self._min_score:.1f}."
                ),
                top_candidate=top.ticker,
                top_score=top.total_score,
            )

        # ---- Confidence threshold ----
        if top.confidence < self._confidence_threshold:
            log.info(
                "selector_no_trade_low_confidence",
                confidence=top.confidence,
                threshold=self._confidence_threshold,
            )
            return NoTradeDecision(
                reason="confidence_below_threshold",
                detail=(
                    f"Confidence {top.confidence:.1f} "
                    f"< threshold {self._confidence_threshold:.1f}."
                ),
                top_candidate=top.ticker,
                top_score=top.total_score,
            )

        # ---- Minimum R:R ----
        rr = getattr(top, "risk_reward_score", None)
        # risk_reward_score is 0-100; map it back to a ratio for the check.
        # The raw R:R ratio is stored in extra or directly on the scored candidate.
        raw_rr: float | None = (
            top.extra.get("risk_reward_ratio") if hasattr(top, "extra") else None
        )
        if raw_rr is not None and raw_rr < self._min_rr:
            log.info(
                "selector_no_trade_rr_too_low",
                rr=raw_rr,
                min_rr=self._min_rr,
            )
            return NoTradeDecision(
                reason="insufficient_risk_reward",
                detail=(
                    f"R:R ratio {raw_rr:.2f} < minimum {self._min_rr:.2f}."
                ),
                top_candidate=top.ticker,
                top_score=top.total_score,
            )

        # ---- Regime alignment check for MIXED ----
        if regime.regime == MarketRegime.MIXED:
            # In mixed regime require higher confidence
            mixed_conf_threshold = min(self._confidence_threshold + 10.0, 80.0)
            if top.confidence < mixed_conf_threshold:
                log.info(
                    "selector_no_trade_mixed_regime_confidence",
                    confidence=top.confidence,
                    mixed_threshold=mixed_conf_threshold,
                )
                return NoTradeDecision(
                    reason="mixed_regime_confidence_too_low",
                    detail=(
                        f"In MIXED regime, confidence {top.confidence:.1f} "
                        f"< adjusted threshold {mixed_conf_threshold:.1f}."
                    ),
                    top_candidate=top.ticker,
                    top_score=top.total_score,
                )

        # ---- All checks passed: SELECT ----
        result = SelectionResult(ticker=top.ticker, scored_candidate=top)
        log.info(
            "selector_trade_selected",
            confidence=round(top.confidence, 2),
            regime=regime.regime,
        )
        return result
