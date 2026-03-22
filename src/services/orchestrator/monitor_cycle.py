"""
Position monitoring cycle.

Called every 30 seconds during market hours by the scheduler's monitor_job.
Delegates to PositionMonitor.monitor_cycle() for the per-position logic and
performs additional orchestration:

  - Fetch all open positions from the broker
  - Update current prices + PnL gauges
  - Run the exit engine via PositionMonitor.monitor_cycle()
  - Execute any triggered exit orders via OrderManager
  - Reconcile open orders
  - Update Prometheus metrics (active_positions, unrealized_pnl, etc.)

Error handling: individual position failures are isolated so that one bad
ticker does not prevent the remaining positions from being monitored.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.metrics import METRICS

if TYPE_CHECKING:
    from src.services.orchestrator.scan_cycle import ServiceContainer

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Monitor cycle
# ---------------------------------------------------------------------------


async def run_monitor_cycle(services: "ServiceContainer") -> None:
    """Execute one position monitoring tick.

    Parameters
    ----------
    services:
        Fully-initialised service container from main.py.
    """
    log = logger.bind(job="monitor_cycle")
    cycle_start = time.monotonic()

    # ---- Gate: kill switch active → skip (positions closed by kill switch itself) ----
    try:
        if await services.kill_switch.is_active:
            log.warning("monitor_cycle_skipped_kill_switch")
            return
    except Exception as exc:
        log.error("monitor_cycle_kill_switch_error", error=str(exc))
        return

    # ---- Gate: market closed → nothing to monitor ----
    try:
        if not services.calendar.is_market_open():
            log.debug("monitor_cycle_skipped_market_closed")
            return
    except Exception as exc:
        log.warning("monitor_cycle_calendar_error", error=str(exc))
        # Do not abort – continue monitoring even if calendar check fails

    # ---- Fetch open positions from broker (source of truth) ----
    open_positions: list[Any] = []
    try:
        open_positions = await services.broker.get_positions()
        log.debug("monitor_cycle_positions_fetched", count=len(open_positions))
    except Exception as exc:
        log.error("monitor_cycle_get_positions_error", error=str(exc))
        return  # Cannot monitor without knowing what is open

    if not open_positions:
        log.debug("monitor_cycle_no_open_positions")
        METRICS.active_positions.set(0)
        return

    # ---- Update active-positions gauge immediately ----
    METRICS.active_positions.set(len(open_positions))

    # ---- Delegate to PositionMonitor for per-position exit logic ----
    try:
        await services.position_monitor.monitor_cycle()
        log.debug("monitor_cycle_position_monitor_complete")
    except Exception as exc:
        log.error("monitor_cycle_position_monitor_error", error=str(exc))
        # Continue to reconcile and update metrics even if monitor partially failed

    # ---- Reconcile orders with broker (ensure fills are reflected) ----
    try:
        await services.order_manager.reconcile()
        log.debug("monitor_cycle_reconcile_complete")
    except Exception as exc:
        log.error("monitor_cycle_reconcile_error", error=str(exc))

    # ---- Update portfolio-level metrics ----
    try:
        await _update_portfolio_metrics(services=services, log=log)
    except Exception as exc:
        log.error("monitor_cycle_metrics_error", error=str(exc))

    elapsed = time.monotonic() - cycle_start
    log.debug("monitor_cycle_complete", duration_seconds=round(elapsed, 3), positions=len(open_positions))


# ---------------------------------------------------------------------------
# Portfolio metrics helper
# ---------------------------------------------------------------------------


async def _update_portfolio_metrics(
    services: "ServiceContainer",
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Refresh Prometheus gauges from latest broker account snapshot."""
    try:
        account = await services.broker.get_account()
    except Exception as exc:
        log.warning("monitor_cycle_account_fetch_error", error=str(exc))
        return

    portfolio_val = float(getattr(account, "portfolio_value", 0) or 0)
    cash_bal = float(getattr(account, "cash", 0) or 0)
    daily_pl = float(getattr(account, "daily_pnl", 0) or 0)
    unrealized_pl = float(getattr(account, "unrealized_pnl", 0) or 0)

    if portfolio_val > 0:
        METRICS.portfolio_value.set(portfolio_val)
    if cash_bal >= 0:
        METRICS.cash_balance.set(cash_bal)
    METRICS.daily_pnl.set(daily_pl)
    METRICS.unrealized_pnl.set(unrealized_pl)

    log.debug(
        "monitor_portfolio_metrics_updated",
        portfolio_value=round(portfolio_val, 2),
        cash=round(cash_bal, 2),
        daily_pnl=round(daily_pl, 2),
        unrealized_pnl=round(unrealized_pl, 2),
    )
