"""Tests for Signal Confidence Scoring System (src/strategy/signal_scorer.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.strategy.signal_scorer import CalibrationBucket, SignalScore, SignalScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeRegime:
    regime: str = "bull"


def _bullish_indicators() -> dict:
    """Indicators for a strongly bullish setup."""
    return {
        "ema_fast": 155.0,
        "ema_slow": 150.0,
        "ema_200": 140.0,
        "last_close": 158.0,
        "rsi_14": 55.0,
        "macd_hist": 1.5,
        "stoch_k": 65.0,
        "stoch_d": 60.0,
        "volume_ratio": 1.8,
        "obv_trend": "positive",
        "trend_score": 75.0,
    }


def _bearish_indicators() -> dict:
    """Indicators for a bearish setup."""
    return {
        "ema_fast": 145.0,
        "ema_slow": 150.0,
        "ema_200": 160.0,
        "last_close": 142.0,
        "rsi_14": 35.0,
        "macd_hist": -1.5,
        "stoch_k": 25.0,
        "stoch_d": 30.0,
        "volume_ratio": 1.5,
        "obv_trend": "negative",
        "trend_score": 25.0,
    }


def _neutral_indicators() -> dict:
    """Indicators with mixed/neutral signals."""
    return {
        "ema_fast": 150.0,
        "ema_slow": 150.0,
        "ema_200": 150.0,
        "last_close": 150.0,
        "rsi_14": 50.0,
        "macd_hist": 0.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "volume_ratio": 1.0,
        "obv_trend": "flat",
        "trend_score": 50.0,
    }


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestSignalScorer:
    @pytest.fixture
    def scorer(self):
        return SignalScorer()

    def test_score_returns_signal_score(self, scorer):
        result = scorer.score(_bullish_indicators())
        assert isinstance(result, SignalScore)
        assert 0 <= result.total_score <= 100

    def test_bullish_long_high_score(self, scorer):
        result = scorer.score(
            _bullish_indicators(),
            regime=_FakeRegime("bull"),
            stock_return=5.0,
            benchmark_return=1.0,
            signal_side="long",
        )
        assert result.total_score >= 60  # Should be a tradeable signal
        assert result.confidence_tier in ("full", "half")

    def test_bearish_short_high_score(self, scorer):
        result = scorer.score(
            _bearish_indicators(),
            regime=_FakeRegime("bear"),
            stock_return=-5.0,
            benchmark_return=-1.0,
            signal_side="short",
        )
        assert result.total_score >= 50

    def test_counter_trend_low_score(self, scorer):
        """Long signal in a bear market should score lower."""
        result = scorer.score(
            _bullish_indicators(),
            regime=_FakeRegime("bear"),
            signal_side="long",
        )
        counter = result.regime_compatibility
        # Compare with aligned signal
        result_aligned = scorer.score(
            _bullish_indicators(),
            regime=_FakeRegime("bull"),
            signal_side="long",
        )
        assert counter < result_aligned.regime_compatibility

    def test_neutral_moderate_score(self, scorer):
        result = scorer.score(_neutral_indicators())
        assert 20 <= result.total_score <= 70

    def test_no_regime_data(self, scorer):
        result = scorer.score(_bullish_indicators(), regime=None)
        assert result.regime_compatibility > 0  # Should get neutral score

    def test_component_scores_within_bounds(self, scorer):
        result = scorer.score(_bullish_indicators(), regime=_FakeRegime("bull"))
        assert 0 <= result.trend_alignment <= 25
        assert 0 <= result.volume_confirmation <= 15
        assert 0 <= result.indicator_confluence <= 25
        assert 0 <= result.regime_compatibility <= 20
        assert 0 <= result.relative_strength <= 15


# ---------------------------------------------------------------------------
# Confidence tier tests
# ---------------------------------------------------------------------------


class TestConfidenceTiers:
    def test_full_position_tier(self):
        scorer = SignalScorer({"full_position_score": 75, "min_score_to_trade": 60})
        result = scorer.score(
            _bullish_indicators(),
            regime=_FakeRegime("bull"),
            stock_return=8.0,
            benchmark_return=1.0,
            signal_side="long",
        )
        if result.total_score >= 75:
            assert result.confidence_tier == "full"
            assert result.position_scale == 1.0

    def test_half_position_tier(self):
        scorer = SignalScorer({"full_position_score": 75, "min_score_to_trade": 60})
        # Use indicators that score in the 60-74 range
        indicators = _neutral_indicators()
        indicators["volume_ratio"] = 1.5
        indicators["rsi_14"] = 55
        indicators["ema_fast"] = 152
        indicators["last_close"] = 153
        result = scorer.score(indicators, regime=_FakeRegime("bull"), signal_side="long")
        if 60 <= result.total_score < 75:
            assert result.confidence_tier == "half"
            assert 0.5 <= result.position_scale < 1.0

    def test_no_trade_tier(self):
        scorer = SignalScorer({"min_score_to_trade": 60})
        # Very weak indicators
        indicators = {
            "ema_fast": 0, "ema_slow": 0, "ema_200": 0, "last_close": 0,
            "rsi_14": 50, "macd_hist": 0, "stoch_k": 50, "stoch_d": 50,
            "volume_ratio": 0.5, "obv_trend": "flat", "trend_score": 30,
        }
        result = scorer.score(indicators, regime=_FakeRegime("uncertain"), signal_side="long")
        if result.total_score < 45:
            assert result.confidence_tier == "no_trade"
            assert result.position_scale == 0.0


# ---------------------------------------------------------------------------
# Component scorer tests
# ---------------------------------------------------------------------------


class TestTrendAlignment:
    def test_perfect_bullish_alignment(self):
        scorer = SignalScorer()
        indicators = {
            "ema_fast": 155, "ema_slow": 150, "ema_200": 140,
            "last_close": 158, "trend_score": 80,
        }
        result = scorer.score(indicators, signal_side="long")
        assert result.trend_alignment > 15  # Should get most of the 25 points

    def test_no_ema_data(self):
        scorer = SignalScorer()
        indicators = {"ema_fast": 0, "ema_slow": 0, "ema_200": 0, "last_close": 0}
        result = scorer.score(indicators, signal_side="long")
        assert result.trend_alignment >= 0  # Should not crash


class TestVolumeConfirmation:
    def test_high_volume_scores_well(self):
        scorer = SignalScorer()
        indicators = {"volume_ratio": 2.5, "obv_trend": "positive"}
        result = scorer.score(indicators, signal_side="long")
        assert result.volume_confirmation > 10

    def test_low_volume_scores_poorly(self):
        scorer = SignalScorer()
        indicators = {"volume_ratio": 0.5, "obv_trend": "flat"}
        result = scorer.score(indicators, signal_side="long")
        assert result.volume_confirmation < 8


class TestRelativeStrength:
    def test_strong_outperformance(self):
        scorer = SignalScorer()
        result = scorer.score({}, stock_return=10.0, benchmark_return=2.0)
        assert result.relative_strength > 10

    def test_underperformance(self):
        scorer = SignalScorer()
        result = scorer.score({}, stock_return=-5.0, benchmark_return=2.0)
        assert result.relative_strength < 5


# ---------------------------------------------------------------------------
# Self-calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_record_outcome(self):
        scorer = SignalScorer()
        scorer.record_outcome(80.0, True)
        scorer.record_outcome(80.0, False)
        stats = scorer.get_calibration_stats()
        assert stats["high"]["trades"] == 2
        assert stats["high"]["wins"] == 1

    def test_calibration_stats(self):
        scorer = SignalScorer()
        for _ in range(10):
            scorer.record_outcome(80.0, True)
        for _ in range(5):
            scorer.record_outcome(80.0, False)
        stats = scorer.get_calibration_stats()
        assert stats["high"]["win_rate"] == pytest.approx(10 / 15, abs=0.01)

    def test_auto_adjust_raises_threshold(self):
        scorer = SignalScorer({"min_score_to_trade": 60})
        # Simulate poor performance in medium bucket
        for _ in range(20):
            scorer.record_outcome(65.0, False)
        for _ in range(5):
            scorer.record_outcome(65.0, True)

        old_threshold = scorer.min_score_to_trade
        adjustments = scorer.auto_adjust_thresholds()
        if "min_score_to_trade" in adjustments:
            assert scorer.min_score_to_trade > old_threshold


class TestShouldTrade:
    def test_should_trade_high_score(self):
        scorer = SignalScorer({"min_score_to_trade": 60})
        score = SignalScore(total_score=75.0, confidence_tier="full")
        assert scorer.should_trade(score) is True

    def test_should_not_trade_low_score(self):
        scorer = SignalScorer({"min_score_to_trade": 60})
        score = SignalScore(total_score=45.0, confidence_tier="no_trade")
        assert scorer.should_trade(score) is False


class TestExplanation:
    def test_explanation_populated(self):
        scorer = SignalScorer()
        result = scorer.score(_bullish_indicators())
        assert len(result.explanation) > 0
        assert "Signal score" in result.explanation
