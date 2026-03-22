"""
Score explainer: generate rich human-readable explanations of ranking decisions.

The ScoreExplainer takes a list of ScoredCandidates and produces clear,
actionable prose explaining why the top-ranked stock was selected, which
factors contributed most, and what penalties were applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.pipeline.screener import RegimeAssessment
    from src.services.scoring.engine import ScoredCandidate

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds used for narrative generation
# ---------------------------------------------------------------------------

_STRONG_THRESHOLD = 70.0
_WEAK_THRESHOLD = 40.0

_COMPONENT_LABELS: dict[str, str] = {
    "technical_score": "Technical Analysis",
    "news_score": "News & Catalysts",
    "sentiment_score": "Social Sentiment",
    "liquidity_score": "Liquidity",
    "regime_score": "Regime Alignment",
    "risk_reward_score": "Risk/Reward Quality",
}


class ScoreExplainer:
    """Generate human-readable explanations for scoring decisions.

    Usage::

        explainer = ScoreExplainer()
        report = explainer.explain_top_pick(ranked_candidates, regime)
        print(report)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_top_pick(
        self,
        ranked_candidates: "list[ScoredCandidate]",
        regime: "RegimeAssessment",
    ) -> str:
        """Full narrative for the #1 ranked candidate.

        Args:
            ranked_candidates: List sorted best-first (output of CandidateRanker).
            regime:            Current market regime.

        Returns:
            Multi-paragraph explanation string suitable for logging or alerts.
        """
        if not ranked_candidates:
            return "No candidates were ranked this cycle. No trade identified."

        top = ranked_candidates[0]
        rank_context = self._rank_context(ranked_candidates)
        component_analysis = self._component_analysis(top)
        penalty_summary = self._penalty_summary(top)
        regime_note = self._regime_note(regime)
        confidence_note = self._confidence_note(top)
        action_summary = self._action_summary(top)

        sections = [
            f"=== Ranking Explanation: {top.ticker} ranked #1 ===",
            "",
            f"SCORE: {top.total_score:.1f}/100  |  CONFIDENCE: {top.confidence:.1f}/100",
            "",
            rank_context,
            "",
            component_analysis,
        ]

        if penalty_summary:
            sections += ["", penalty_summary]

        sections += [
            "",
            regime_note,
            "",
            confidence_note,
            "",
            action_summary,
        ]

        return "\n".join(sections)

    def explain_candidate(
        self,
        candidate: "ScoredCandidate",
        rank: int,
    ) -> str:
        """Single-paragraph summary for any ranked candidate."""
        strengths = self._strong_components(candidate)
        weaknesses = self._weak_components(candidate)
        penalties = self._penalty_summary(candidate, brief=True)

        parts: list[str] = [
            f"#{rank} {candidate.ticker}",
            f"(score={candidate.total_score:.1f}, confidence={candidate.confidence:.1f})",
        ]

        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}.")
        if weaknesses:
            parts.append(f"Weak areas: {', '.join(weaknesses)}.")
        if penalties:
            parts.append(penalties)

        return " ".join(parts)

    def explain_no_trade(
        self,
        reason: str,
        detail: str = "",
        top_candidate: "ScoredCandidate | None" = None,
    ) -> str:
        """Explanation for a NO TRADE decision."""
        lines = [f"NO TRADE decision — Reason: {reason.replace('_', ' ').title()}."]
        if detail:
            lines.append(detail)
        if top_candidate is not None:
            lines.append(
                f"Best available candidate was {top_candidate.ticker} "
                f"with score {top_candidate.total_score:.1f} and "
                f"confidence {top_candidate.confidence:.1f}."
            )
        return " ".join(lines)

    def generate_ranking_table(
        self,
        ranked_candidates: "list[ScoredCandidate]",
    ) -> str:
        """Compact ranked table suitable for logs / reports."""
        if not ranked_candidates:
            return "No candidates to display."

        header = (
            f"{'Rank':<5} {'Ticker':<8} {'Score':>6} {'Conf':>6} "
            f"{'Tech':>5} {'News':>5} {'Sent':>5} {'RR':>5} {'Penalties'}"
        )
        divider = "-" * 72
        rows = [header, divider]

        for i, c in enumerate(ranked_candidates, start=1):
            pen_str = (
                ", ".join(f"{k}(−{v:.0f})" for k, v in c.penalties.items())
                if c.penalties
                else "none"
            )
            rows.append(
                f"{i:<5} {c.ticker:<8} {c.total_score:>6.1f} {c.confidence:>6.1f} "
                f"{c.technical_score:>5.0f} {c.news_score:>5.0f} "
                f"{c.sentiment_score:>5.0f} {c.risk_reward_score:>5.0f} {pen_str}"
            )

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_context(ranked: "list[ScoredCandidate]") -> str:
        top = ranked[0]
        n = len(ranked)
        if n == 1:
            return f"{top.ticker} was the only candidate that passed all screening stages."
        second = ranked[1]
        gap = top.total_score - second.total_score
        return (
            f"{top.ticker} led a field of {n} screened candidates. "
            f"It outscored the #2 pick ({second.ticker}, "
            f"score={second.total_score:.1f}) by {gap:.1f} points."
        )

    @staticmethod
    def _component_analysis(candidate: "ScoredCandidate") -> str:
        lines = ["KEY CONTRIBUTING FACTORS:"]
        components = [
            ("Technical Analysis", candidate.technical_score),
            ("News & Catalysts", candidate.news_score),
            ("Social Sentiment", candidate.sentiment_score),
            ("Liquidity", candidate.liquidity_score),
            ("Regime Alignment", candidate.regime_score),
            ("Risk/Reward Quality", candidate.risk_reward_score),
        ]
        for label, score in components:
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            qualifier = (
                "STRONG" if score >= _STRONG_THRESHOLD
                else "WEAK" if score < _WEAK_THRESHOLD
                else "moderate"
            )
            lines.append(f"  {label:<22} {score:>5.1f}/100  [{bar}]  {qualifier}")
        return "\n".join(lines)

    @staticmethod
    def _penalty_summary(
        candidate: "ScoredCandidate",
        brief: bool = False,
    ) -> str:
        if not candidate.penalties:
            return ""
        total_deducted = sum(candidate.penalties.values())
        if brief:
            items = ", ".join(
                f"{k.replace('_', ' ')}(−{v:.1f})" for k, v in candidate.penalties.items()
            )
            return f"Penalties: {items} (total −{total_deducted:.1f})."
        lines = [f"PENALTIES APPLIED (total deduction: −{total_deducted:.1f}):"]
        descriptions = {
            "rsi_overextended": "RSI above 75 – stock may be overextended in the short term.",
            "earnings_proximity": "Earnings report due within 5 days – elevated binary event risk.",
            "data_quality": "Data quality below GOOD – signals may be less reliable.",
            "sentiment_crowding": "Social sentiment crowding risk above threshold – momentum may be peaking.",
            "price_extension": "Price extended far above 20-day SMA – increased mean-reversion risk.",
        }
        for key, amount in candidate.penalties.items():
            desc = descriptions.get(key, key.replace("_", " "))
            lines.append(f"  • {desc}  (−{amount:.1f} points)")
        return "\n".join(lines)

    @staticmethod
    def _regime_note(regime: "RegimeAssessment") -> str:
        from src.core.enums import MarketRegime

        regime_descriptions = {
            MarketRegime.FAVORABLE: (
                "The broad market regime is FAVORABLE. "
                "Trend-following setups have higher base rates in this environment."
            ),
            MarketRegime.MIXED: (
                "The broad market regime is MIXED. "
                "Only high-confidence setups were considered. "
                "Position sizing has been reduced accordingly."
            ),
            MarketRegime.UNFAVORABLE: (
                "The broad market regime is UNFAVORABLE. "
                "This candidate would normally be blocked by the regime filter; "
                "review selector logic if a trade was selected."
            ),
        }
        default = f"Regime: {regime.regime}."
        note = regime_descriptions.get(regime.regime, default)
        conf_note = (
            f" Regime confidence: {regime.confidence:.0f}/100."
            if hasattr(regime, "confidence") and regime.confidence is not None
            else ""
        )
        return "REGIME NOTE: " + note + conf_note

    @staticmethod
    def _confidence_note(candidate: "ScoredCandidate") -> str:
        conf = candidate.confidence
        if conf >= 75:
            level = "HIGH confidence"
            detail = "Multiple independent factors align; signal reliability is high."
        elif conf >= 55:
            level = "MODERATE confidence"
            detail = "Signal is reasonable but not exceptional; standard position sizing applies."
        else:
            level = "LOW confidence"
            detail = (
                "Signal quality is marginal. "
                "Consider reducing position size or waiting for a cleaner setup."
            )
        return f"CONFIDENCE: {level} ({conf:.1f}/100). {detail}"

    @staticmethod
    def _action_summary(candidate: "ScoredCandidate") -> str:
        return (
            f"ACTION SUMMARY: {candidate.ticker} is selected for trade construction. "
            f"Score={candidate.total_score:.1f}, Confidence={candidate.confidence:.1f}. "
            f"Proceed to TradeConstructor for entry zone, stop, and targets."
        )

    @staticmethod
    def _strong_components(candidate: "ScoredCandidate") -> list[str]:
        result = []
        for attr, label in _COMPONENT_LABELS.items():
            val = getattr(candidate, attr, 0.0)
            if val >= _STRONG_THRESHOLD:
                result.append(f"{label}={val:.0f}")
        return result

    @staticmethod
    def _weak_components(candidate: "ScoredCandidate") -> list[str]:
        result = []
        for attr, label in _COMPONENT_LABELS.items():
            val = getattr(candidate, attr, 0.0)
            if val < _WEAK_THRESHOLD:
                result.append(f"{label}={val:.0f}")
        return result
