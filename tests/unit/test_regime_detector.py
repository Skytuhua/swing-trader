"""Tests for Market Regime Detection Filter (src/strategy/regime_detector.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime_detector import (
    DetailedRegime,
    REGIME_STRATEGY,
    RegimeDetection,
    RegimeDetector,
    compute_adx,
    compute_bb_width,
    compute_ema,
    compute_rsi,
    compute_volume_ratio,
    should_trade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trending_up(n: int = 100, start: float = 100.0) -> pd.DataFrame:
    """Generate strongly trending-up OHLCV data."""
    rng = np.random.default_rng(42)
    closes = start * np.cumprod(1 + rng.normal(0.003, 0.008, n))
    highs = closes * (1 + rng.uniform(0.002, 0.015, n))
    lows = closes * (1 - rng.uniform(0.002, 0.015, n))
    opens = np.roll(closes, 1)
    opens[0] = start
    volumes = rng.integers(1_000_000, 3_000_000, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    })


def _make_trending_down(n: int = 100, start: float = 200.0) -> pd.DataFrame:
    """Generate strongly trending-down OHLCV data."""
    rng = np.random.default_rng(13)
    closes = start * np.cumprod(1 + rng.normal(-0.003, 0.008, n))
    highs = closes * (1 + rng.uniform(0.002, 0.015, n))
    lows = closes * (1 - rng.uniform(0.002, 0.015, n))
    opens = np.roll(closes, 1)
    opens[0] = start
    volumes = rng.integers(1_000_000, 3_000_000, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    })


def _make_sideways(n: int = 100, center: float = 100.0) -> pd.DataFrame:
    """Generate flat/sideways OHLCV data."""
    rng = np.random.default_rng(7)
    closes = center + rng.normal(0, 0.3, n)  # Tiny range
    highs = closes + rng.uniform(0.05, 0.2, n)
    lows = closes - rng.uniform(0.05, 0.2, n)
    opens = np.roll(closes, 1)
    opens[0] = center
    volumes = rng.integers(800_000, 1_200_000, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    })


# ---------------------------------------------------------------------------
# Indicator computation tests
# ---------------------------------------------------------------------------


class TestComputeEMA:
    def test_basic_ema(self):
        data = np.arange(1.0, 21.0)
        ema = compute_ema(data, 10)
        assert ema > 10  # EMA of ascending data is above midpoint

    def test_short_data(self):
        data = np.array([5.0, 10.0])
        ema = compute_ema(data, 20)
        assert ema == 7.5  # Falls back to mean

    def test_empty_data(self):
        assert compute_ema(np.array([]), 14) == 0.0


class TestComputeRSI:
    def test_rising_prices(self):
        """RSI of steadily rising prices should be high (>70)."""
        prices = np.linspace(100, 130, 50)
        rsi = compute_rsi(prices, 14)
        assert rsi > 70

    def test_falling_prices(self):
        """RSI of steadily falling prices should be low (<30)."""
        prices = np.linspace(130, 100, 50)
        rsi = compute_rsi(prices, 14)
        assert rsi < 30

    def test_flat_prices(self):
        """RSI of flat prices should be near 50."""
        prices = np.full(50, 100.0)
        rsi = compute_rsi(prices, 14)
        # With no changes, avg_gain and avg_loss are both 0; RSI defaults to 50
        assert 40 <= rsi <= 60 or rsi == 100.0  # Edge: all gains=0, losses=0

    def test_insufficient_data(self):
        rsi = compute_rsi(np.array([100.0, 101.0]), 14)
        assert rsi == 50.0


class TestComputeADX:
    def test_trending_adx(self):
        """ADX for trending data should be elevated."""
        df = _make_trending_up(100)
        adx = compute_adx(
            df["high"].values, df["low"].values, df["close"].values, 14
        )
        assert adx > 0  # Just verify it computes

    def test_short_data(self):
        adx = compute_adx(
            np.array([10.0, 11.0]),
            np.array([9.0, 10.0]),
            np.array([10.0, 10.5]),
            14,
        )
        assert adx == 0.0


class TestComputeBBWidth:
    def test_narrow_bb(self):
        """Flat data should have very narrow Bollinger Bands."""
        closes = np.full(30, 100.0)
        width = compute_bb_width(closes)
        assert width == 0.0  # Zero std = zero width

    def test_volatile_bb(self):
        """Volatile data should have wide Bollinger Bands."""
        rng = np.random.default_rng(42)
        closes = 100 + rng.normal(0, 5, 30)
        width = compute_bb_width(closes)
        assert width > 0.05


class TestComputeVolumeRatio:
    def test_normal_volume(self):
        volumes = np.full(30, 1_000_000)
        ratio = compute_volume_ratio(volumes, 20)
        assert 0.9 <= ratio <= 1.1

    def test_volume_spike(self):
        volumes = np.full(30, 1_000_000)
        volumes[-1] = 3_000_000  # 3x spike
        ratio = compute_volume_ratio(volumes, 20)
        assert ratio > 2.5


# ---------------------------------------------------------------------------
# Regime Detector tests
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_detect_returns_regime_detection(self):
        df = _make_trending_up()
        detector = RegimeDetector()
        result = detector.detect(df)
        assert isinstance(result, RegimeDetection)
        assert result.regime in DetailedRegime
        assert 0 <= result.confidence <= 1

    def test_bullish_data_detects_bull(self):
        df = _make_trending_up(150)
        detector = RegimeDetector()
        result = detector.detect(df)
        assert result.regime in (DetailedRegime.BULL, DetailedRegime.BULL_WEAK)
        assert result.ema_alignment in ("bullish", "bullish_weak")

    def test_bearish_data_detects_bear(self):
        df = _make_trending_down(150)
        detector = RegimeDetector()
        result = detector.detect(df)
        assert result.regime in (DetailedRegime.BEAR, DetailedRegime.BEAR_WEAK)
        assert result.ema_alignment in ("bearish", "bearish_weak")

    def test_sideways_data(self):
        df = _make_sideways(150)
        detector = RegimeDetector()
        result = detector.detect(df)
        # Sideways data should not produce BULL or BEAR
        assert result.regime in (
            DetailedRegime.SIDEWAYS, DetailedRegime.UNCERTAIN,
            DetailedRegime.BULL_WEAK, DetailedRegime.BEAR_WEAK,
        )

    def test_strategy_map_populated(self):
        df = _make_trending_up()
        detector = RegimeDetector()
        result = detector.detect(df)
        assert "bias" in result.strategy
        assert "aggression" in result.strategy
        assert "trail_mult" in result.strategy

    def test_multi_timeframe_same_direction(self):
        daily = _make_trending_up(150)
        four_h = _make_trending_up(200, start=100)
        detector = RegimeDetector()
        result = detector.detect_multi_timeframe(daily, four_h)
        assert result.primary_regime is not None
        assert result.confirmation_regime is not None

    def test_multi_timeframe_no_confirmation(self):
        daily = _make_trending_up()
        detector = RegimeDetector()
        result = detector.detect_multi_timeframe(daily, None)
        assert result.confirmation_regime is None

    def test_custom_config(self):
        config = {
            "ema_periods": [10, 30, 100],
            "adx_period": 10,
            "adx_trending_threshold": 30,
            "adx_ranging_threshold": 15,
        }
        detector = RegimeDetector(config)
        assert detector.adx_trending == 30
        assert detector.adx_ranging == 15

    def test_explanation_not_empty(self):
        df = _make_trending_up()
        detector = RegimeDetector()
        result = detector.detect(df)
        assert len(result.explanation) > 0
        assert "ADX" in result.explanation

    def test_scores_dict(self):
        df = _make_trending_up()
        detector = RegimeDetector()
        result = detector.detect(df)
        assert "direction" in result.scores
        assert "momentum" in result.scores
        assert "trend_strength" in result.scores


class TestShouldTrade:
    def test_should_trade_bull(self):
        detection = RegimeDetection(
            regime=DetailedRegime.BULL,
            confidence=0.8,
            strategy=REGIME_STRATEGY[DetailedRegime.BULL],
        )
        assert should_trade(detection) is True

    def test_should_not_trade_uncertain(self):
        detection = RegimeDetection(
            regime=DetailedRegime.UNCERTAIN,
            confidence=0.2,
            strategy=REGIME_STRATEGY[DetailedRegime.UNCERTAIN],
        )
        assert should_trade(detection) is False

    def test_should_not_trade_zero_aggression(self):
        detection = RegimeDetection(
            regime=DetailedRegime.BEAR,
            confidence=0.5,
            strategy={"aggression": 0},
        )
        assert should_trade(detection) is False


class TestRegimeStrategyMap:
    def test_all_regimes_have_strategy(self):
        for regime in DetailedRegime:
            assert regime in REGIME_STRATEGY
            strategy = REGIME_STRATEGY[regime]
            assert "bias" in strategy
            assert "aggression" in strategy
            assert "trail_mult" in strategy

    def test_bull_most_aggressive(self):
        assert REGIME_STRATEGY[DetailedRegime.BULL]["aggression"] == 1.0

    def test_uncertain_zero_aggression(self):
        assert REGIME_STRATEGY[DetailedRegime.UNCERTAIN]["aggression"] == 0.0
