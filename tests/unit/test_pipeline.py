"""
Unit tests for the candidate pipeline (universe, screener, ranker, selector).

Tests: universe filtering, screener stages, NO TRADE decision, ranking order.
At least 6 test cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.enums import DataQuality, MarketRegime
from src.services.pipeline.screener import RegimeAssessment, ScreenedCandidate
from src.services.pipeline.ranker import CandidateRanker
from src.services.pipeline.selector import FinalSelector, NoTradeDecision, SelectionResult
from src.services.scoring.engine import ScoredCandidate, ScoringEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_technical_profile(composite_score: float = 70.0) -> MagicMock:
    tp = MagicMock()
    tp.composite_score = composite_score
    tp.trend_score = composite_score * 0.9
    tp.momentum_score = composite_score * 0.8
    tp.setup_type = "breakout"
    tp.indicators = {
        "rsi_14": 55.0, "atr_14": 2.5, "atr_pct": 1.5,
        "relative_volume": 1.5, "last_close": 150.0, "sma_20": 145.0,
    }
    tp.key_levels = {"resistance": 160.0, "support": 140.0}
    return tp


def _make_screened_candidate(
    ticker: str,
    tech_score: float = 70.0,
    news_score: float = 65.0,
    sentiment_score: float = 60.0,
    data_quality: DataQuality = DataQuality.GOOD,
) -> ScreenedCandidate:
    news = MagicMock()
    news.score = news_score
    news.negative_catalyst = False

    sentiment = MagicMock()
    sentiment.score = sentiment_score
    sentiment.crowding_risk = 20.0

    return ScreenedCandidate(
        ticker=ticker,
        technical=_make_technical_profile(tech_score),
        news=news,
        sentiment=sentiment,
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# Universe filtering tests
# ---------------------------------------------------------------------------


class TestUniverseFiltering:

    def test_universe_filter_accepts_valid_ticker_list(self):
        """UniverseFilter exposes a tradable_tickers or ticker list attribute."""
        from src.services.pipeline.universe import UniverseFilter
        # UniverseFilter needs a config and data_manager
        config = MagicMock()
        config.tickers = ["AAPL", "MSFT", "TSLA"]
        config.min_price = 5.0
        config.max_price = 1000.0
        config.min_avg_volume = 100_000
        config.blacklist = []

        data_manager = AsyncMock()
        uf = UniverseFilter(config=config, data_manager=data_manager)
        # UniverseFilter should be instantiated without error
        assert uf is not None

    def test_universe_filter_has_get_tradable_universe(self):
        """UniverseFilter exposes an async get_tradable_universe method."""
        from src.services.pipeline.universe import UniverseFilter
        config = MagicMock()
        config.tickers = ["AAPL"]
        config.min_price = 5.0
        config.max_price = 1000.0
        config.min_avg_volume = 0
        config.blacklist = []
        data_manager = AsyncMock()
        uf = UniverseFilter(config=config, data_manager=data_manager)
        assert hasattr(uf, "get_tradable_universe")
        import asyncio
        assert asyncio.iscoroutinefunction(uf.get_tradable_universe)


# ---------------------------------------------------------------------------
# Screener regime threshold tests
# ---------------------------------------------------------------------------


class TestScreenerRegimeThresholds:

    def test_screener_threshold_higher_in_unfavorable_regime(self):
        """Unfavorable regime has a stricter technical score threshold."""
        from src.services.pipeline.screener import (
            _TECHNICAL_PRESCREEN_MIN,
            _TECHNICAL_UNFAVORABLE_MIN,
            MultiStageScreener,
        )
        # The unfavorable threshold should be higher (stricter) than base
        assert _TECHNICAL_UNFAVORABLE_MIN > _TECHNICAL_PRESCREEN_MIN

    def test_screener_threshold_higher_in_mixed_regime(self):
        """Mixed regime has a stricter threshold than favorable."""
        from src.services.pipeline.screener import (
            _TECHNICAL_MIXED_MIN,
            _TECHNICAL_PRESCREEN_MIN,
        )
        assert _TECHNICAL_MIXED_MIN > _TECHNICAL_PRESCREEN_MIN

    def test_regime_threshold_returns_correct_value(self):
        """_regime_threshold returns the right bar for each regime."""
        from src.services.pipeline.screener import (
            MultiStageScreener,
            _TECHNICAL_MIXED_MIN,
            _TECHNICAL_PRESCREEN_MIN,
            _TECHNICAL_UNFAVORABLE_MIN,
        )
        # Test using the static method directly
        fav_regime = RegimeAssessment(regime=MarketRegime.FAVORABLE)
        mix_regime = RegimeAssessment(regime=MarketRegime.MIXED)
        unf_regime = RegimeAssessment(regime=MarketRegime.UNFAVORABLE)

        assert MultiStageScreener._regime_threshold(fav_regime) == _TECHNICAL_PRESCREEN_MIN
        assert MultiStageScreener._regime_threshold(mix_regime) == _TECHNICAL_MIXED_MIN
        assert MultiStageScreener._regime_threshold(unf_regime) == _TECHNICAL_UNFAVORABLE_MIN


# ---------------------------------------------------------------------------
# Ranker tests
# ---------------------------------------------------------------------------


class TestRanker:

    def test_ranker_sorts_by_score_descending(self, scoring_config, favorable_regime):
        """CandidateRanker ranks pre-scored candidates by total_score DESC."""
        # CandidateRanker takes a ScoringEngine and ranks ScreenedCandidates
        engine = ScoringEngine(scoring_config)
        ranker = CandidateRanker(scoring_engine=engine)

        # Create ScreenedCandidates with different composite scores
        candidates = [
            _make_screened_candidate("LOW",  tech_score=50.0, news_score=50.0),
            _make_screened_candidate("HIGH", tech_score=90.0, news_score=85.0),
            _make_screened_candidate("MID",  tech_score=70.0, news_score=65.0),
        ]
        ranked = ranker.rank(candidates, favorable_regime)
        assert len(ranked) == 3
        # First should have highest total_score
        assert ranked[0].total_score >= ranked[1].total_score
        assert ranked[1].total_score >= ranked[2].total_score

    def test_ranker_returns_empty_for_no_candidates(self, scoring_config, favorable_regime):
        """Ranker returns empty list when no candidates provided."""
        engine = ScoringEngine(scoring_config)
        ranker = CandidateRanker(scoring_engine=engine)
        result = ranker.rank([], favorable_regime)
        assert result == []


# ---------------------------------------------------------------------------
# Selector / NO TRADE tests
# ---------------------------------------------------------------------------


class TestFinalSelector:

    def test_selector_returns_no_trade_when_empty(self, favorable_regime):
        """FinalSelector returns NoTradeDecision when list is empty."""
        selector = FinalSelector()
        result = selector.select([], regime=favorable_regime)
        assert isinstance(result, NoTradeDecision)
        assert "no_candidate" in result.reason

    def test_selector_returns_no_trade_when_all_below_threshold(self, favorable_regime):
        """FinalSelector returns NoTradeDecision when best score < min_score."""
        selector = FinalSelector()  # default min_score = 55.0
        weak_candidates = [
            ScoredCandidate(ticker="WEAK1", total_score=30.0, confidence=25.0),
            ScoredCandidate(ticker="WEAK2", total_score=20.0, confidence=15.0),
        ]
        result = selector.select(weak_candidates, regime=favorable_regime)
        assert isinstance(result, NoTradeDecision)
        assert "threshold" in result.reason or "score" in result.reason

    def test_selector_returns_no_trade_in_unfavorable_regime(self, unfavorable_regime):
        """FinalSelector always returns NoTradeDecision in UNFAVORABLE regime."""
        selector = FinalSelector()
        strong_candidates = [
            ScoredCandidate(ticker="STRONG", total_score=90.0, confidence=85.0),
        ]
        result = selector.select(strong_candidates, regime=unfavorable_regime)
        assert isinstance(result, NoTradeDecision)
        assert "unfavorable" in result.reason.lower()

    def test_selector_picks_best_above_threshold(self, favorable_regime):
        """FinalSelector picks the highest-scoring candidate above all thresholds."""
        # Use default confidence_threshold=60 and min_score=55
        selector = FinalSelector()
        candidates = [
            ScoredCandidate(ticker="BEST",  total_score=85.0, confidence=75.0),
            ScoredCandidate(ticker="SECOND", total_score=70.0, confidence=65.0),
        ]
        result = selector.select(candidates, regime=favorable_regime)
        assert isinstance(result, SelectionResult)
        assert result.ticker == "BEST"
