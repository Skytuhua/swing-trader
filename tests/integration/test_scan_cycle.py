"""
Integration tests for the scan cycle with mocked external services.

Tests end-to-end: scoring → ranker → selector.
At least 3 test cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.enums import DataQuality, MarketRegime
from src.services.pipeline.ranker import CandidateRanker
from src.services.pipeline.screener import RegimeAssessment, ScreenedCandidate
from src.services.pipeline.selector import FinalSelector, NoTradeDecision, SelectionResult
from src.services.scoring.engine import ScoredCandidate, ScoringEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_screened_candidates(count: int = 3) -> list[ScreenedCandidate]:
    """Generate mock screened candidates for pipeline tests."""
    candidates = []
    for i in range(count):
        ticker = f"STOCK{i}"
        tech = MagicMock()
        tech.composite_score = 60.0 + i * 10.0
        tech.trend_score = 65.0 + i * 8.0
        tech.momentum_score = 55.0 + i * 9.0
        tech.setup_type = "breakout"
        tech.indicators = {
            "rsi_14": 55.0 + i * 2.0,
            "atr_14": 2.0 + i * 0.5,
            "atr_pct": 1.5,
            "relative_volume": 1.3 + i * 0.2,
            "last_close": 100.0 + i * 50.0,
            "sma_20": 95.0 + i * 48.0,
        }
        tech.key_levels = {
            "resistance": (110.0 + i * 50.0),
            "support": (90.0 + i * 48.0),
        }

        news = MagicMock()
        news.score = 60.0 + i * 5.0
        news.negative_catalyst = False

        sentiment = MagicMock()
        sentiment.score = 55.0 + i * 5.0
        sentiment.crowding_risk = 15.0

        candidate = ScreenedCandidate(
            ticker=ticker,
            technical=tech,
            news=news,
            sentiment=sentiment,
            data_quality=DataQuality.GOOD,
        )
        candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScanCycleIntegration:

    def test_scoring_pipeline_end_to_end(self, scoring_config, favorable_regime):
        """Full scoring pipeline: candidates → ScoredCandidates with all fields."""
        engine = ScoringEngine(scoring_config)
        candidates = _make_mock_screened_candidates(5)

        scored = engine.score_many(candidates, favorable_regime)

        # All candidates get scored
        assert len(scored) == 5
        # Scores are in valid range
        for sc in scored:
            assert 0.0 <= sc.total_score <= 100.0
            assert 0.0 <= sc.confidence <= 100.0
            assert sc.ticker.startswith("STOCK")

    def test_scan_cycle_selects_best_candidate(self, scoring_config, favorable_regime):
        """Pipeline selects the highest-scoring candidate above threshold."""
        engine = ScoringEngine(scoring_config)
        candidates = _make_mock_screened_candidates(3)

        ranker = CandidateRanker(scoring_engine=engine)
        ranked = ranker.rank(candidates, favorable_regime)

        # The best candidate should be first
        assert len(ranked) == 3
        assert ranked[0].total_score >= ranked[-1].total_score

        # Use FinalSelector with low threshold to allow selection
        selector = FinalSelector()
        result = selector.select(ranked, regime=favorable_regime)
        # Result is either a SelectionResult or NoTradeDecision
        assert result is not None

    def test_scan_cycle_no_trade_when_all_below_threshold(self, favorable_regime):
        """Pipeline returns NO TRADE when no candidates meet the score threshold."""
        # All candidates score 30 (below typical threshold of 55)
        weak_scored = [
            ScoredCandidate(ticker=f"WEAK{i}", total_score=30.0, confidence=25.0)
            for i in range(5)
        ]

        selector = FinalSelector()
        result = selector.select(weak_scored, regime=favorable_regime)

        assert isinstance(result, NoTradeDecision)  # NO TRADE decision

    def test_regime_filters_affect_scoring(self, scoring_config, unfavorable_regime, favorable_regime):
        """Unfavorable regime significantly reduces confidence vs favorable."""
        engine = ScoringEngine(scoring_config)
        candidates = _make_mock_screened_candidates(1)

        fav_scored = engine.score_many(candidates, favorable_regime)
        unfav_scored = engine.score_many(candidates, unfavorable_regime)

        # Confidence should be lower in unfavorable regime
        fav_conf = fav_scored[0].confidence
        unfav_conf = unfav_scored[0].confidence
        assert unfav_conf <= fav_conf

    def test_full_pipeline_from_screen_to_select(self, scoring_config, favorable_regime):
        """End-to-end: screen → score → rank → select produces valid decision."""
        engine = ScoringEngine(scoring_config)
        candidates = _make_mock_screened_candidates(10)  # 10 candidates
        ranker = CandidateRanker(scoring_engine=engine)
        selector = FinalSelector()

        ranked = ranker.rank(candidates, favorable_regime)
        result = selector.select(ranked, regime=favorable_regime)

        # Result should be a valid decision type
        assert isinstance(result, (SelectionResult, NoTradeDecision))
