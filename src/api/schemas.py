"""
Pydantic v2 request/response schemas for the SwingTrader Admin API.

All schemas use model_config with populate_by_name=True and from_attributes=True
so they can be constructed from both dicts and SQLAlchemy ORM objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=True,
    )


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class ErrorResponse(_Base):
    """Returned on all 4xx/5xx errors."""

    detail: str = Field(..., description="Human-readable error description.")
    code: str = Field(default="error", description="Machine-readable error code.")


class PaginationParams(_Base):
    """Common query-parameter schema for paginated list endpoints."""

    page: int = Field(default=1, ge=1, description="1-based page number.")
    page_size: int = Field(default=50, ge=1, le=500, description="Items per page.")
    sort_by: str = Field(default="created_at", description="Column to sort by.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(_Base):
    """Basic health check response."""

    status: str = Field(..., description="'ok' or 'degraded'.")
    uptime_seconds: float = Field(..., description="Seconds since process start.")
    version: str = Field(..., description="Application version string.")
    mode: str = Field(..., description="Trading mode: 'paper' or 'live'.")
    checks: dict[str, Any] = Field(
        default_factory=dict,
        description="Connectivity check results keyed by component name.",
    )


# ---------------------------------------------------------------------------
# TradingDecision
# ---------------------------------------------------------------------------

class DecisionResponse(_Base):
    """Full representation of a TradingDecision record."""

    id: uuid.UUID
    cycle_id: str
    timestamp: datetime

    # Regime context
    market_regime: str | None = None
    regime_confidence: float | None = None

    # Candidate summary
    top_candidates_json: list[Any] | None = None

    # Decision outcome
    selected_ticker: str | None = None
    is_no_trade: bool
    no_trade_reason: str | None = None
    reason_summary: str | None = None

    # Component scores
    technical_score: float | None = None
    news_score: float | None = None
    sentiment_score: float | None = None
    risk_reward_score: float | None = None
    liquidity_score: float | None = None
    regime_alignment_score: float | None = None
    confidence_score: float | None = None

    # Trade construction
    entry_price_low: float | None = None
    entry_price_high: float | None = None
    entry_method: str | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    trailing_stop_rule: str | None = None

    # Position sizing
    allocation_pct: float | None = None
    dollar_size: float | None = None
    share_quantity: int | None = None

    # Risk management
    invalidation_conditions: list[Any] | None = None
    data_quality_flags: dict[str, Any] | None = None

    # Metadata
    config_version: str | None = None
    model_version: str | None = None

    # Timestamps from TimestampMixin
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DecisionListResponse(_Base):
    """Paginated list of TradingDecision records."""

    items: list[DecisionResponse]
    total: int = Field(..., description="Total number of matching records.")
    page: int
    page_size: int
    has_next: bool


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class PositionResponse(_Base):
    """Full representation of a Position record."""

    id: uuid.UUID
    decision_id: uuid.UUID
    ticker: str
    status: str

    # Entry
    entry_price: float
    entry_date: datetime
    quantity: int
    original_quantity: int

    # Live pricing
    current_price: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None

    # Risk levels
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    trailing_stop_price: float | None = None
    max_price_since_entry: float | None = None

    # Hold time
    hold_days: int
    last_monitored_at: datetime | None = None

    # Thesis
    thesis_score: float | None = None

    # Exit details
    exit_reason: str | None = None
    exit_price: float | None = None
    exit_date: datetime | None = None
    realized_pnl: float | None = None
    realized_pnl_pct: float | None = None
    commission_total: float

    notes: str | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PositionListResponse(_Base):
    """Paginated list of Position records."""

    items: list[PositionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class OrderResponse(_Base):
    """Full representation of an Order record."""

    id: uuid.UUID
    decision_id: uuid.UUID | None = None
    position_id: uuid.UUID | None = None
    ticker: str
    side: str
    order_type: str
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    trail_amount: float | None = None
    trail_percent: float | None = None
    time_in_force: str
    status: str
    broker_order_id: str | None = None

    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_qty: int
    filled_avg_price: float | None = None
    commission: float

    reject_reason: str | None = None
    is_entry: bool
    is_exit: bool
    idempotency_key: str

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None


class OrderListResponse(_Base):
    """Paginated list of Order records."""

    items: list[OrderResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ---------------------------------------------------------------------------
# CompletedTrade
# ---------------------------------------------------------------------------

class TradeResponse(_Base):
    """Full representation of a CompletedTrade record."""

    id: uuid.UUID
    position_id: uuid.UUID
    decision_id: uuid.UUID
    ticker: str

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int

    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str

    entry_score: float | None = None
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    commission_total: float

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TradeListResponse(_Base):
    """Paginated list of CompletedTrade records."""

    items: list[TradeResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class TradeSummaryResponse(_Base):
    """Aggregate statistics over completed trades."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float = Field(..., description="Win rate as a decimal (0.0–1.0).")
    total_net_pnl: float
    avg_net_pnl: float
    avg_pnl_pct: float
    avg_hold_days: float
    expectancy: float = Field(
        ...,
        description="Expected value per trade in USD (win_rate×avg_win − loss_rate×avg_loss).",
    )
    profit_factor: float = Field(
        ...,
        description="Gross profit / gross loss ratio.",
    )
    sharpe_estimate: float = Field(
        ...,
        description="Annualised Sharpe ratio estimate from daily P&L.",
    )
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    exit_reason_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Count of trades grouped by exit reason.",
    )
    period_start: datetime | None = None
    period_end: datetime | None = None


# ---------------------------------------------------------------------------
# RiskSnapshot
# ---------------------------------------------------------------------------

class RiskSnapshotResponse(_Base):
    """Full representation of a RiskSnapshot record."""

    id: uuid.UUID
    timestamp: datetime

    portfolio_value: float
    cash: float
    equity: float

    total_risk_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float
    peak_portfolio_value: float | None = None
    open_positions_count: int

    kill_switch_active: bool
    kill_switch_reason: str | None = None

    data_quality: str
    position_details_json: list[Any] | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RiskSnapshotListResponse(_Base):
    """Paginated list of RiskSnapshot records."""

    items: list[RiskSnapshotResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------

class KillSwitchRequest(_Base):
    """Request body for kill-switch activate/deactivate."""

    action: str = Field(
        ...,
        pattern="^(activate|deactivate)$",
        description="Must be 'activate' or 'deactivate'.",
    )
    reason: str = Field(
        default="Manual operator action",
        min_length=1,
        max_length=512,
        description="Human-readable reason for the action.",
    )


class KillSwitchResponse(_Base):
    """Current kill-switch state."""

    active: bool
    reason: str | None = None
    activated_at: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigResponse(_Base):
    """Sanitised application configuration (no secrets or API keys)."""

    mode: str
    log_level: str
    debug: bool
    config_version: str
    model_version: str
    api_host: str
    api_port: int
    timezone: str

    # Non-secret section snapshots
    universe: dict[str, Any] = Field(default_factory=dict)
    technical: dict[str, Any] = Field(default_factory=dict)
    scoring: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    news: dict[str, Any] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)

    # Redacted flags
    secrets_redacted: bool = Field(
        default=True,
        description="Always True — API keys and credentials are never returned.",
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class DailyReportPositionSummary(_Base):
    """Per-position summary for daily reports."""

    ticker: str
    status: str
    entry_price: float
    current_price: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    hold_days: int


class ReportResponse(_Base):
    """Daily trading report for a calendar date."""

    report_date: str = Field(..., description="Date in YYYY-MM-DD format.")
    generated_at: datetime

    # Portfolio snapshot
    portfolio_value: float | None = None
    daily_pnl: float | None = None
    daily_pnl_pct: float | None = None
    max_drawdown: float | None = None

    # Activity counts
    decisions_made: int
    trades_entered: int
    trades_exited: int
    open_positions_count: int

    # Trade summary
    realized_pnl_today: float
    commissions_today: float

    # Open positions
    open_positions: list[DailyReportPositionSummary] = Field(default_factory=list)

    # Kill switch
    kill_switch_active: bool

    # Scan decisions overview
    no_trade_decisions: int
    trade_decisions: int

    # Notes / anomalies
    notes: list[str] = Field(default_factory=list)
