"""
Shared pytest fixtures for the SwingTrader test suite.

Provides:
  - sample_ohlcv_df       100-day synthetic OHLCV DataFrame
  - multi_ticker_data     Dict of 3 synthetic OHLCV DataFrames
  - sample_config         Minimal AppConfig-like object
  - mock_data_manager     MagicMock for DataManager
  - mock_broker           MagicMock for BrokerAdapter
  - sample_quote          A Quote object
  - sample_news_items     List of mock news dicts
  - paper_broker          Real PaperBroker instance for integration tests
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.enums import DataQuality, MarketRegime


# ---------------------------------------------------------------------------
# Synthetic OHLCV generation
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n_days: int = 100,
    start_price: float = 100.0,
    seed: int = 42,
    trend: float = 0.0003,  # slight upward drift per day
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with realistic price action."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-03", periods=n_days, freq="B")

    # Geometric random walk
    daily_returns = rng.normal(trend, 0.015, n_days)  # ~1.5% daily vol
    close_prices = start_price * np.cumprod(1 + daily_returns)

    # Realistic OHLCV structure
    opens = np.roll(close_prices, 1)
    opens[0] = start_price

    daily_ranges = close_prices * rng.uniform(0.005, 0.025, n_days)  # 0.5–2.5% range
    highs = np.maximum(opens, close_prices) + daily_ranges * 0.6
    lows = np.minimum(opens, close_prices) - daily_ranges * 0.4

    # Volume with occasional spikes
    base_volume = rng.integers(800_000, 2_000_000, n_days).astype(float)
    spike_mask = rng.random(n_days) < 0.1  # 10% chance of volume spike
    base_volume[spike_mask] *= rng.uniform(2.0, 4.0, spike_mask.sum())

    df = pd.DataFrame(
        {
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(close_prices, 4),
            "volume": base_volume.astype(int),
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_ohlcv_df() -> pd.DataFrame:
    """100 trading days of synthetic OHLCV data with realistic price action."""
    return _make_ohlcv(n_days=100, start_price=100.0, seed=42)


@pytest.fixture(scope="session")
def sample_ohlcv_df_200() -> pd.DataFrame:
    """200 trading days of synthetic OHLCV data (for indicators needing 200 bars)."""
    return _make_ohlcv(n_days=200, start_price=150.0, seed=99)


@pytest.fixture(scope="session")
def bullish_ohlcv_df() -> pd.DataFrame:
    """100 days of a strongly trending upward stock."""
    return _make_ohlcv(n_days=100, start_price=50.0, seed=7, trend=0.002)


@pytest.fixture(scope="session")
def bearish_ohlcv_df() -> pd.DataFrame:
    """100 days of a strongly trending downward stock."""
    return _make_ohlcv(n_days=100, start_price=200.0, seed=13, trend=-0.002)


@pytest.fixture(scope="session")
def multi_ticker_data(sample_ohlcv_df) -> dict[str, pd.DataFrame]:
    """Dict of 3 synthetic tickers for multi-asset backtest tests."""
    return {
        "AAPL": _make_ohlcv(n_days=150, start_price=170.0, seed=10, trend=0.001),
        "MSFT": _make_ohlcv(n_days=150, start_price=300.0, seed=20, trend=0.0005),
        "TSLA": _make_ohlcv(n_days=150, start_price=250.0, seed=30, trend=-0.0003),
    }


@pytest.fixture
def minimal_ohlcv_df() -> pd.DataFrame:
    """5-bar OHLCV – useful for testing edge cases with insufficient data."""
    dates = pd.bdate_range(start="2023-01-03", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open":  [100.0, 101.0, 102.0, 101.5, 103.0],
            "high":  [102.0, 103.0, 104.0, 103.0, 105.0],
            "low":   [99.0,  100.0, 101.0, 100.5, 102.0],
            "close": [101.0, 102.0, 101.5, 102.5, 104.0],
            "volume": [1_000_000] * 5,
        },
        index=dates,
    )


@pytest.fixture
def flat_ohlcv_df() -> pd.DataFrame:
    """50 bars of completely flat price (all same OHLCV) – edge case."""
    dates = pd.bdate_range(start="2023-01-03", periods=50, freq="B")
    return pd.DataFrame(
        {
            "open":   [100.0] * 50,
            "high":   [100.0] * 50,
            "low":    [100.0] * 50,
            "close":  [100.0] * 50,
            "volume": [500_000] * 50,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@dataclass
class _TechnicalConfig:
    sma_windows: list = field(default_factory=lambda: [9, 20, 50, 200])
    ema_windows: list = field(default_factory=lambda: [9, 20, 50])
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    volume_avg_period: int = 20
    breakout_lookback: int = 20
    breakout_threshold: float = 1.005  # 0.5% above prior high


@dataclass
class _ScoringConfig:
    technical_weight: float = 0.30
    news_weight: float = 0.15
    sentiment_weight: float = 0.10
    liquidity_weight: float = 0.10
    regime_weight: float = 0.15
    risk_reward_weight: float = 0.20


@dataclass
class _RiskConfig:
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    max_total_risk_pct: float = 6.0
    max_position_pct: float = 20.0
    max_risk_per_trade_pct: float = 1.5
    max_spread_pct: float = 0.5
    duplicate_signal_cooldown_minutes: float = 60.0


@dataclass
class _AppConfig:
    initial_capital: float = 100_000.0
    technical: _TechnicalConfig = field(default_factory=_TechnicalConfig)
    scoring: _ScoringConfig = field(default_factory=_ScoringConfig)
    risk: _RiskConfig = field(default_factory=_RiskConfig)
    commission_per_share: float = 0.005
    slippage_pct: float = 0.001
    max_hold_days: int = 5
    max_positions: int = 5


@pytest.fixture
def sample_config() -> _AppConfig:
    return _AppConfig()


@pytest.fixture
def technical_config() -> _TechnicalConfig:
    return _TechnicalConfig()


@pytest.fixture
def scoring_config() -> _ScoringConfig:
    return _ScoringConfig()


@pytest.fixture
def risk_config() -> _RiskConfig:
    return _RiskConfig()


# ---------------------------------------------------------------------------
# Mock broker
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_broker():
    """MagicMock for BrokerAdapter with async methods."""
    from src.services.execution.base import AccountInfo, BrokerOrder, BrokerPosition, Quote
    from src.core.enums import OrderSide, OrderStatus, OrderType

    broker = AsyncMock()
    broker.get_account.return_value = AccountInfo(
        account_id="test-001",
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        portfolio_value=100_000.0,
    )
    broker.get_quote.return_value = Quote(
        ticker="AAPL",
        price=150.0,
        bid=149.85,
        ask=150.15,
    )
    broker.get_positions.return_value = []
    broker.get_open_orders.return_value = []
    broker.is_market_open.return_value = True
    broker.health_check.return_value = True

    filled_order = BrokerOrder(
        broker_order_id="test-order-001",
        ticker="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        status=OrderStatus.FILLED,
        filled_quantity=100,
        average_fill_price=150.15,
    )
    broker.submit_order.return_value = filled_order
    return broker


@pytest.fixture
def paper_broker():
    """Real PaperBroker for integration tests."""
    from src.services.execution.paper_broker import PaperBroker
    return PaperBroker(
        initial_cash=100_000.0,
        slippage_pct=0.001,
        commission_per_share=0.005,
    )


# ---------------------------------------------------------------------------
# Mock DataManager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_manager(sample_ohlcv_df):
    """MagicMock for DataManager; returns sample_ohlcv_df for any ticker."""
    manager = AsyncMock()
    manager.get_daily_ohlcv.return_value = sample_ohlcv_df
    manager.get_quote.return_value = MagicMock(
        ticker="AAPL",
        price=sample_ohlcv_df["close"].iloc[-1],
        bid=sample_ohlcv_df["close"].iloc[-1] * 0.999,
        ask=sample_ohlcv_df["close"].iloc[-1] * 1.001,
    )
    manager.get_market_snapshot.return_value = {}
    return manager


# ---------------------------------------------------------------------------
# Sample quote fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_quote():
    """A Quote object for AAPL at $150."""
    from src.services.execution.base import Quote
    return Quote(
        ticker="AAPL",
        price=150.0,
        bid=149.85,
        ask=150.15,
    )


# ---------------------------------------------------------------------------
# Sample news items
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_news_items() -> list[dict]:
    """List of mock news items with sentiment attributes."""
    return [
        {
            "headline": "Apple reports record quarterly revenue",
            "summary": "Apple Inc. beat analyst expectations with strong iPhone sales.",
            "source": "Reuters",
            "published_at": datetime.now(tz=timezone.utc).isoformat(),
            "ticker": "AAPL",
            "sentiment": "positive",
            "score": 0.82,
        },
        {
            "headline": "Tech sector faces headwinds from rising rates",
            "summary": "Rising interest rates put pressure on high-growth tech stocks.",
            "source": "Bloomberg",
            "published_at": datetime.now(tz=timezone.utc).isoformat(),
            "ticker": "AAPL",
            "sentiment": "negative",
            "score": 0.35,
        },
        {
            "headline": "Apple launches new iPhone model",
            "summary": "The new iPhone features improved camera and battery life.",
            "source": "TechCrunch",
            "published_at": datetime.now(tz=timezone.utc).isoformat(),
            "ticker": "AAPL",
            "sentiment": "positive",
            "score": 0.75,
        },
    ]


# ---------------------------------------------------------------------------
# Kill-switch mock fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kill_switch():
    """A KillSwitch mock that is inactive by default."""
    ks = AsyncMock()
    ks.is_active = False  # property-style access
    ks.reason = None
    ks.activate = AsyncMock()
    ks.deactivate = AsyncMock()
    return ks


# ---------------------------------------------------------------------------
# RegimeAssessment helper
# ---------------------------------------------------------------------------


@pytest.fixture
def favorable_regime():
    """RegimeAssessment for a favorable market."""
    from src.services.pipeline.screener import RegimeAssessment
    return RegimeAssessment(
        regime=MarketRegime.FAVORABLE,
        confidence=80.0,
        trend_strength=75.0,
    )


@pytest.fixture
def mixed_regime():
    """RegimeAssessment for a mixed market."""
    from src.services.pipeline.screener import RegimeAssessment
    return RegimeAssessment(
        regime=MarketRegime.MIXED,
        confidence=55.0,
        trend_strength=50.0,
    )


@pytest.fixture
def unfavorable_regime():
    """RegimeAssessment for an unfavorable market."""
    from src.services.pipeline.screener import RegimeAssessment
    return RegimeAssessment(
        regime=MarketRegime.UNFAVORABLE,
        confidence=70.0,
        trend_strength=25.0,
    )
