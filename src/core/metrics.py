"""
Prometheus metrics definitions for the swing-trader system.

All metrics are module-level singletons registered with the default
Prometheus registry.  Import this module early in the application
lifecycle so that metrics are available before the first collection.

Metrics are grouped by concern:
- Scan cycle counters and latency
- Trade lifecycle counters and gauges
- Portfolio value gauges
- Data quality and error counters
- Risk / safety counters

Usage
-----
    from src.core.metrics import METRICS

    METRICS.scan_cycles_total.inc()
    METRICS.scan_duration_seconds.observe(elapsed)

Or import individual metrics directly:

    from src.core.metrics import scan_cycles_total
    scan_cycles_total.inc()
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, Info


# ---------------------------------------------------------------------------
# Histogram bucket presets
# ---------------------------------------------------------------------------

# Latency buckets tuned for trading operations (seconds)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

# Scan duration buckets (seconds) – scans may take up to a few minutes
_SCAN_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# -- Scan cycle --

scan_cycles_total = Counter(
    "swingtrader_scan_cycles_total",
    "Total number of scan cycles executed.",
    ["status"],  # status: success | error
)

scan_duration_seconds = Histogram(
    "swingtrader_scan_duration_seconds",
    "Wall-clock time for a complete scan cycle in seconds.",
    buckets=_SCAN_BUCKETS,
)

# -- Trade lifecycle --

trades_opened_total = Counter(
    "swingtrader_trades_opened_total",
    "Total number of positions opened.",
    ["ticker"],
)

trades_closed_total = Counter(
    "swingtrader_trades_closed_total",
    "Total number of positions closed.",
    ["ticker", "exit_reason"],
)

no_trade_decisions_total = Counter(
    "swingtrader_no_trade_decisions_total",
    "Total number of scan cycles that produced a NO TRADE decision.",
    ["reason"],
)

# -- Active state gauges --

active_positions = Gauge(
    "swingtrader_active_positions",
    "Number of currently open positions.",
)

portfolio_value = Gauge(
    "swingtrader_portfolio_value_usd",
    "Current total portfolio value in USD.",
)

cash_balance = Gauge(
    "swingtrader_cash_balance_usd",
    "Current uninvested cash balance in USD.",
)

total_pnl = Gauge(
    "swingtrader_total_pnl_usd",
    "Cumulative realised P&L in USD since inception.",
)

unrealized_pnl = Gauge(
    "swingtrader_unrealized_pnl_usd",
    "Current unrealised P&L across all open positions.",
)

win_rate = Gauge(
    "swingtrader_win_rate",
    "Win rate (0.0–1.0) over all closed trades.",
)

daily_pnl = Gauge(
    "swingtrader_daily_pnl_usd",
    "Intraday realised + unrealised P&L in USD.",
)

drawdown_pct = Gauge(
    "swingtrader_drawdown_pct",
    "Current drawdown from portfolio peak as a percentage.",
)

# -- Order execution --

order_latency_seconds = Histogram(
    "swingtrader_order_latency_seconds",
    "Time from order submission to first fill confirmation in seconds.",
    ["order_type", "side"],  # order_type: market|limit|stop; side: buy|sell
    buckets=_LATENCY_BUCKETS,
)

orders_submitted_total = Counter(
    "swingtrader_orders_submitted_total",
    "Total orders submitted to the broker.",
    ["side", "order_type"],
)

orders_rejected_total = Counter(
    "swingtrader_orders_rejected_total",
    "Total orders rejected by the broker.",
    ["reason"],
)

# -- Data quality --

data_fetch_errors_total = Counter(
    "swingtrader_data_fetch_errors_total",
    "Total data-fetch errors by provider and data type.",
    ["provider", "data_type"],  # data_type: ohlcv|quote|news|sentiment
)

data_provider_latency_seconds = Histogram(
    "swingtrader_data_provider_latency_seconds",
    "Latency for external data-provider API calls.",
    ["provider", "data_type"],
    buckets=_LATENCY_BUCKETS,
)

stale_data_events_total = Counter(
    "swingtrader_stale_data_events_total",
    "Total events where data was too stale to use.",
    ["provider"],
)

# -- Risk / safety --

kill_switch_activations_total = Counter(
    "swingtrader_kill_switch_activations_total",
    "Total number of times the kill switch has been activated.",
    ["trigger"],  # trigger: daily_loss|drawdown|manual|stale_data
)

risk_limit_breaches_total = Counter(
    "swingtrader_risk_limit_breaches_total",
    "Total risk-limit breach events.",
    ["limit_name"],
)

# -- Build info --

build_info = Info(
    "swingtrader_build",
    "Static metadata about the running application build.",
)


# ---------------------------------------------------------------------------
# Convenience container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Metrics:
    """Typed container providing dot-access to all metrics."""

    scan_cycles_total: Counter = scan_cycles_total  # type: ignore[assignment]
    scan_duration_seconds: Histogram = scan_duration_seconds  # type: ignore[assignment]
    trades_opened_total: Counter = trades_opened_total  # type: ignore[assignment]
    trades_closed_total: Counter = trades_closed_total  # type: ignore[assignment]
    no_trade_decisions_total: Counter = no_trade_decisions_total  # type: ignore[assignment]
    active_positions: Gauge = active_positions  # type: ignore[assignment]
    portfolio_value: Gauge = portfolio_value  # type: ignore[assignment]
    cash_balance: Gauge = cash_balance  # type: ignore[assignment]
    total_pnl: Gauge = total_pnl  # type: ignore[assignment]
    unrealized_pnl: Gauge = unrealized_pnl  # type: ignore[assignment]
    win_rate: Gauge = win_rate  # type: ignore[assignment]
    daily_pnl: Gauge = daily_pnl  # type: ignore[assignment]
    drawdown_pct: Gauge = drawdown_pct  # type: ignore[assignment]
    order_latency_seconds: Histogram = order_latency_seconds  # type: ignore[assignment]
    orders_submitted_total: Counter = orders_submitted_total  # type: ignore[assignment]
    orders_rejected_total: Counter = orders_rejected_total  # type: ignore[assignment]
    data_fetch_errors_total: Counter = data_fetch_errors_total  # type: ignore[assignment]
    data_provider_latency_seconds: Histogram = data_provider_latency_seconds  # type: ignore[assignment]
    stale_data_events_total: Counter = stale_data_events_total  # type: ignore[assignment]
    kill_switch_activations_total: Counter = kill_switch_activations_total  # type: ignore[assignment]
    risk_limit_breaches_total: Counter = risk_limit_breaches_total  # type: ignore[assignment]
    build_info: Info = build_info  # type: ignore[assignment]


METRICS = _Metrics()


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------


def init_build_info(version: str = "unknown", mode: str = "paper") -> None:
    """Populate the build-info metric with static application metadata.

    Call once at startup after settings are loaded.

    Parameters
    ----------
    version:
        Application / model version string.
    mode:
        Trading mode (``"paper"`` or ``"live"``).
    """
    build_info.info({"version": version, "mode": mode})
