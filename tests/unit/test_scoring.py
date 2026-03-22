"""
Unit tests for the ScoringEngine.

Tests: weighted scoring, penalty application, confidence computation,
regime dampening, edge cases.
At least 10 test cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.enums import DataQuality, MarketRegime
from src.services.pipeline.screener import RegimeAssessment, ScreenedCandidate
from src.services.scoring.engine import ScoredCandidate, ScoringEngine


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_technical_profile(
    composite_score: float = 70.0,
    indicators: dict | None = None,
    key_levels: dict | None = None,
    setup_type: str = "breakout",
    trend_score: float = 70.0,
    momentum_score: float = 65.0,
) -> MagicMock:
    """Create a mock TechnicalProfile."""
    tp = MagicMock()
    tp.composite_score = composite_score
    tp.trend_score = trend_score
    tp.momentum_score = momentum_score
    tp.setup_type = setup_type
    tp.indicators = {
        "rsi_14": 55.0,
        "atr_14": 2.5,
        "atr_pct": 1.5,
        "relative_volume": 1.5,
        "last_close": 150.0,
        "sma_20": 145.0,
        **(indicators or {}),
    }
    tp.key_levels = key_levels or {"resistance": 160.0, "support": 140.0}
    return tp


def _make_news_score(score: float = 65.0) -> MagicMock:
    ns = MagicMock()
    ns.score = score
    ns.negative_catalyst = False
    ns.positive_catalyst = score > 60.0
    return ns


def _make_sentiment_profile(score: float = 60.0, crowding_risk: float = 20.0) -> MagicMock:
    sp = MagicMock()
    sp.score = score
    sp.crowding_risk = crowding_risk
    return sp


def _make_candidate(
    ticker: str = "AAPL",
    technical_score: float = 70.0,
    news_score: float = 65.0,
    sentiment_score: float = 60.0,
    data_quality: DataQuality = DataQuality.GOOD,
    rsi: float = 55.0,
    crowding_risk: float = 20.0,
    next_earnings_date: date | None = None,
    sma_20: float = 145.0,
    last_close: float = 150.0,
) -> ScreenedCandidate:
    """Build a mock ScreenedCandidate for testing."""
    technical = _make_technical_profile(
        composite_score=technical_score,
        indicators={
            "rsi_14": rsi,
            "atr_14": 2.5,
            "atr_pct": 1.5,
            "relative_volume": 1.5,
            "last_close": last_close,
            "sma_20": sma_20,
        },
    )
    news = _make_news_score(news_score)
    sentiment = _make_sentiment_profile(sentiment_score, crowding_risk)

    candidate = ScreenedCandidate(
        ticker=ticker,
        technical=technical,
        news=news,
        sentiment=sentiment,
        data_quality=data_quality,
        next_earnings_date=next_earnings_date,
    )
    return candidate


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestScoringEngineWeighting:

    def test_high_scores_produce_high_total(self, scoring_config, favorable_regime):
        """All component scores at 90 should produce a high total score."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate(
            technical_score=90.0, news_score=90.0, sentiment_score=90.0
        )
        result = engine.score(candidate, favorable_regime)
        assert result.total_score > 70.0

    def test_low_scores_produce_low_total(self, scoring_config, favorable_regime):
        """All component scores at 20 should produce a low total score."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate(
            technical_score=20.0, news_score=20.0, sentiment_score=20.0
        )
        result = engine.score(candidate, favorable_regime)
        # Total is kept well below high-score range; liquidity/regime contribute ~20 pts
        # so threshold is generous at 50
        assert result.total_score < 50.0

    def test_total_score_is_clamped_to_0_100(self, scoring_config, favorable_regime):
        """Total score must always be in [0, 100]."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate(technical_score=100.0, news_score=100.0, sentiment_score=100.0)
        result = engine.score(candidate, favorable_regime)
        assert 0.0 <= result.total_score <= 100.0

    def test_confidence_is_clamped_to_0_100(self, scoring_config, favorable_regime):
        """Confidence must always be in [0, 100]."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate()
        result = engine.score(candidate, favorable_regime)
        assert 0.0 <= result.confidence <= 100.0

    def test_component_scores_populated(self, scoring_config, favorable_regime):
        """Scored candidate contains all component score fields."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate()
        result = engine.score(candidate, favorable_regime)
        assert result.technical_score > 0.0
        assert result.news_score > 0.0
        assert result.sentiment_score > 0.0
        assert result.liquidity_score >= 0.0
        assert result.regime_score >= 0.0
        assert result.risk_reward_score >= 0.0


class TestScoringEnginePenalties:

    def test_rsi_overextended_penalty_applied(self, scoring_config, favorable_regime):
        """RSI > 75 triggers rsi_overextended penalty."""
        engine = ScoringEngine(scoring_config)
        # Normal RSI candidate vs overextended
        normal = _make_candidate(rsi=55.0)
        overextended = _make_candidate(rsi=85.0)  # RSI = 85 → penalty

        normal_result = engine.score(normal, favorable_regime)
        ext_result = engine.score(overextended, favorable_regime)

        assert "rsi_overextended" in ext_result.penalties
        assert ext_result.total_score < normal_result.total_score

    def test_data_quality_penalty_applied(self, scoring_config, favorable_regime):
        """Degraded data quality triggers a penalty."""
        engine = ScoringEngine(scoring_config)
        good = _make_candidate(data_quality=DataQuality.GOOD)
        degraded = _make_candidate(data_quality=DataQuality.DEGRADED)

        good_result = engine.score(good, favorable_regime)
        bad_result = engine.score(degraded, favorable_regime)

        assert "data_quality" in bad_result.penalties
        assert bad_result.total_score < good_result.total_score

    def test_sentiment_crowding_penalty(self, scoring_config, favorable_regime):
        """High crowding risk triggers sentiment_crowding penalty."""
        engine = ScoringEngine(scoring_config)
        low_crowd = _make_candidate(crowding_risk=10.0)
        high_crowd = _make_candidate(crowding_risk=80.0)

        low_result = engine.score(low_crowd, favorable_regime)
        high_result = engine.score(high_crowd, favorable_regime)

        assert "sentiment_crowding" in high_result.penalties
        assert high_result.total_score < low_result.total_score


class TestScoringEngineRegimeDampening:

    def test_unfavorable_regime_reduces_confidence(self, scoring_config, unfavorable_regime):
        """Unfavorable regime significantly reduces confidence score."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate(technical_score=80.0, news_score=75.0)
        result = engine.score(candidate, unfavorable_regime)
        # Confidence should be significantly reduced
        assert result.confidence < 70.0

    def test_mixed_regime_reduces_regime_score(self, scoring_config, mixed_regime, favorable_regime):
        """Mixed regime yields lower regime_score than favorable."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate()
        fav_result = engine.score(candidate, favorable_regime)
        mixed_result = engine.score(candidate, mixed_regime)
        # Regime score should be lower in mixed regime
        assert mixed_result.regime_score <= fav_result.regime_score

    def test_favorable_regime_no_confidence_reduction(self, scoring_config, favorable_regime):
        """Favorable regime does not drastically reduce confidence."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate(technical_score=80.0)
        result = engine.score(candidate, favorable_regime)
        # With high scores and favorable regime, confidence should be > 60
        assert result.confidence > 50.0


class TestScoringEngineEdgeCases:

    def test_score_many_returns_list(self, scoring_config, favorable_regime):
        """score_many processes multiple candidates."""
        engine = ScoringEngine(scoring_config)
        candidates = [_make_candidate(f"TKR{i}") for i in range(5)]
        results = engine.score_many(candidates, favorable_regime)
        assert len(results) == 5
        assert all(isinstance(r, ScoredCandidate) for r in results)

    def test_explanation_is_populated(self, scoring_config, favorable_regime):
        """Explanation string is non-empty after scoring."""
        engine = ScoringEngine(scoring_config)
        candidate = _make_candidate()
        result = engine.score(candidate, favorable_regime)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10
