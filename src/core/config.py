"""
SwingTrader — Application Configuration.

Provides typed configuration dataclasses and a root ``Settings`` class backed
by pydantic-settings.  All sections correspond to top-level keys in
``config/default.yaml`` and its environment-specific overlays.

Usage
-----
    from src.core.config import get_settings, Settings

    settings = get_settings()
    db_url = settings.database.url
    broker_key = settings.broker.api_key

Environment overrides (all optional)
-------------------------------------
    DB_URL                   → settings.database.url
    REDIS_URL                → settings.redis.url
    ALPACA_API_KEY           → settings.broker.api_key
    ALPACA_API_SECRET        → settings.broker.api_secret
    FINNHUB_API_KEY          → settings.news.api_keys["finnhub"]
    LOG_LEVEL                → settings.app.log_level

All names listed in ``__all__`` are importable from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Nested configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatabaseConfig:
    """SQLAlchemy async engine configuration."""

    url: str = field(
        default_factory=lambda: os.environ.get(
            "DB_URL",
            "postgresql+asyncpg://swingtrader:swingtrader@localhost:5432/swingtrader",
        )
    )
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    prefix: str = "swingtrader:"
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


@dataclass
class BrokerConfig:
    """Alpaca (or paper) broker configuration."""

    provider: str = "alpaca"
    api_key: str = field(
        default_factory=lambda: os.environ.get("ALPACA_API_KEY", "")
    )
    api_secret: str = field(
        default_factory=lambda: os.environ.get("ALPACA_API_SECRET", "")
    )
    base_url: str = "https://paper-api.alpaca.markets"
    paper: bool = True
    initial_cash: float = 100_000.0


@dataclass
class MarketDataConfig:
    """Market data provider configuration."""

    provider: str = "alpaca"
    fallback_provider: str = "yfinance"
    cache_ttl_daily: int = 14_400    # 4 hours
    cache_ttl_intraday: int = 300    # 5 minutes
    cache_ttl_quote: int = 15        # 15 seconds
    stale_threshold_minutes: int = 15


@dataclass
class NewsConfig:
    """News provider and FinBERT configuration."""

    providers: list = field(default_factory=lambda: ["finnhub"])
    api_keys: dict = field(default_factory=dict)
    finbert_model: str = "ProsusAI/finbert"
    dedup_similarity_threshold: float = 0.85
    decay_hours: int = 72


@dataclass
class SentimentConfig:
    """Social sentiment provider configuration."""

    providers: list = field(default_factory=lambda: ["reddit", "stocktwits"])
    api_keys: dict = field(default_factory=dict)
    baseline_volume: int = 100
    baseline_std: int = 50


@dataclass
class UniverseConfig:
    """Stock universe filter parameters."""

    min_price: float = 10.0
    min_market_cap: float = 1e9
    min_avg_dollar_volume: float = 5e6
    min_volume: int = 500_000
    max_spread_pct: float = 0.5
    exclusion_list: list = field(default_factory=list)
    earnings_blackout_days: int = 2
    max_universe_size: int = 500


@dataclass
class TechnicalConfig:
    """Technical indicator configuration."""

    sma_windows: list = field(default_factory=lambda: [9, 20, 50, 200])
    ema_windows: list = field(default_factory=lambda: [9, 20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    roc_periods: list = field(default_factory=lambda: [5, 10, 20])


@dataclass
class ScoringConfig:
    """Composite scoring engine configuration."""

    technical_weight: float = 0.30
    news_weight: float = 0.15
    sentiment_weight: float = 0.10
    liquidity_weight: float = 0.10
    regime_weight: float = 0.15
    risk_reward_weight: float = 0.20
    confidence_threshold: float = 55.0
    no_trade_threshold: float = 40.0
    min_risk_reward: float = 1.5
    profile: str = "moderate"


@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_position_pct: float = 20.0
    max_risk_per_trade_pct: float = 2.0
    max_total_risk_pct: float = 6.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_slippage_pct: float = 0.5
    max_spread_pct: float = 0.5
    kill_switch_enabled: bool = True
    duplicate_cooldown_minutes: int = 60
    data_freshness_minutes: int = 15


@dataclass
class PositionConfig:
    """Position lifecycle parameters."""

    max_hold_days: int = 5
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 2.0
    trailing_stop_atr_mult: float = 1.5
    trailing_activation_pct: float = 1.0


@dataclass
class ScheduleConfig:
    """APScheduler cron configuration."""

    scan_cron: str = "30 9,11,13,15 * * 1-5"
    monitor_interval_seconds: int = 30
    report_cron: str = "5 16 * * 1-5"
    health_interval_minutes: int = 5
    reconcile_interval_seconds: int = 60


@dataclass
class AppConfig:
    """Top-level application settings."""

    mode: str = "paper"
    log_level: str = "INFO"
    debug: bool = False
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    config_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Root Settings (pydantic-settings)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root application settings.

    Each attribute is a typed dataclass that can be overridden via
    environment variables or programmatically.

    pydantic-settings does not natively support dataclass nested models,
    so we declare them as ``Any`` with defaults and validate on access.
    The dataclass defaults are always used; env vars that are meaningful
    to specific sections (e.g. DB_URL, REDIS_URL) are applied inside each
    dataclass's ``default_factory`` at construction time so they resolve
    at import time just like any other Python default.
    """

    # Infrastructure
    database: Any = Field(default_factory=DatabaseConfig)
    redis: Any = Field(default_factory=RedisConfig)

    # Broker / trading mode
    broker: Any = Field(default_factory=BrokerConfig)

    # Data & news
    market_data: Any = Field(default_factory=MarketDataConfig)
    news: Any = Field(default_factory=NewsConfig)
    sentiment: Any = Field(default_factory=SentimentConfig)

    # Strategy
    universe: Any = Field(default_factory=UniverseConfig)
    technical: Any = Field(default_factory=TechnicalConfig)
    scoring: Any = Field(default_factory=ScoringConfig)

    # Risk & position management
    risk: Any = Field(default_factory=RiskConfig)
    position: Any = Field(default_factory=PositionConfig)

    # Scheduler
    schedule: Any = Field(default_factory=ScheduleConfig)

    # Application
    app: Any = Field(default_factory=AppConfig)

    model_config = {
        # Allow extra fields coming from YAML-based config injection
        "extra": "allow",
        # Read env vars case-insensitively
        "env_prefix": "",
        "case_sensitive": False,
        # Do not attempt to auto-read a .env file (we load YAML ourselves)
        "env_file": None,
    }

    def apply_yaml(self, raw: dict[str, Any]) -> None:
        """Apply a raw YAML config dict on top of current defaults.

        This is called by ``src/main.py`` after loading the merged YAML to
        push per-environment overrides into the settings instance.

        Parameters
        ----------
        raw:
            Merged YAML dict from ``_load_config()``.
        """
        # App
        if "app" in raw:
            _apply(self.app, raw["app"])
        elif "mode" in raw:
            self.app.mode = raw["mode"]

        # Database
        if "database" in raw:
            _apply(self.database, raw["database"])

        # Redis
        if "redis" in raw:
            _apply(self.redis, raw["redis"])

        # Broker
        if "broker" in raw:
            _apply(self.broker, raw["broker"])

        # Market data (yaml key may be "data" or "market_data")
        for key in ("market_data", "data"):
            if key in raw:
                _apply(self.market_data, raw[key])
                break

        # News
        if "news" in raw:
            _apply(self.news, raw["news"])

        # Sentiment
        if "sentiment" in raw:
            _apply(self.sentiment, raw["sentiment"])

        # Universe
        if "universe" in raw:
            _apply(self.universe, raw["universe"])

        # Technical / indicators
        for key in ("technical", "indicators"):
            if key in raw:
                _apply(self.technical, raw[key])
                break

        # Scoring
        if "scoring" in raw:
            _apply(self.scoring, raw["scoring"])
            # Handle nested weights dict from YAML
            if "weights" in raw["scoring"] and isinstance(raw["scoring"]["weights"], dict):
                w = raw["scoring"]["weights"]
                self.scoring.technical_weight = float(w.get("technical", self.scoring.technical_weight))
                self.scoring.news_weight = float(w.get("news", self.scoring.news_weight))
                self.scoring.sentiment_weight = float(w.get("sentiment", self.scoring.sentiment_weight))
                self.scoring.liquidity_weight = float(w.get("liquidity", self.scoring.liquidity_weight))
                self.scoring.regime_weight = float(w.get("regime", self.scoring.regime_weight))
                self.scoring.risk_reward_weight = float(w.get("risk_reward", self.scoring.risk_reward_weight))

        # Risk
        if "risk" in raw:
            _apply(self.risk, raw["risk"])

        # Position
        if "position" in raw:
            _apply(self.position, raw["position"])

        # Schedule
        if "schedule" in raw:
            _apply(self.schedule, raw["schedule"])

        # Apply env var overrides (highest priority)
        _apply_env_overrides(self)


def _apply(obj: Any, data: dict[str, Any]) -> None:
    """Set attributes on *obj* from *data*, ignoring unknown keys."""
    for k, v in data.items():
        if hasattr(obj, k) and not isinstance(v, dict):
            try:
                setattr(obj, k, v)
            except Exception:  # noqa: BLE001
                pass


def _apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides to the settings instance."""
    if url := os.environ.get("DB_URL"):
        settings.database.url = url
    if url := os.environ.get("REDIS_URL"):
        settings.redis.url = url
    if key := os.environ.get("ALPACA_API_KEY"):
        settings.broker.api_key = key
    if secret := os.environ.get("ALPACA_API_SECRET"):
        settings.broker.api_secret = secret
    if level := os.environ.get("LOG_LEVEL"):
        settings.app.log_level = level
    if key := os.environ.get("FINNHUB_API_KEY"):
        settings.news.api_keys["finnhub"] = key


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the application settings singleton.

    The singleton is created on first call and cached for the lifetime of
    the process.  Use ``reset_settings()`` in tests to force re-creation.

    Returns
    -------
    Settings
        The global ``Settings`` instance.
    """
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
        _apply_env_overrides(_settings_instance)
    return _settings_instance


def get_config() -> Settings:
    """Alias for :func:`get_settings`."""
    return get_settings()


def reset_settings() -> None:
    """Reset the singleton so the next call to ``get_settings()`` rebuilds it.

    Intended for use in unit tests only.
    """
    global _settings_instance  # noqa: PLW0603
    _settings_instance = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Root settings
    "Settings",
    # Sub-configs
    "AppConfig",
    "DatabaseConfig",
    "RedisConfig",
    "BrokerConfig",
    "MarketDataConfig",
    "NewsConfig",
    "SentimentConfig",
    "UniverseConfig",
    "TechnicalConfig",
    "ScoringConfig",
    "RiskConfig",
    "PositionConfig",
    "ScheduleConfig",
    # Accessors
    "get_settings",
    "get_config",
    "reset_settings",
]
