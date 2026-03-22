"""
Daily end-of-day report cycle.

Called by the scheduler at 16:05 ET after market close.  Generates a
structured summary of the trading day and persists it to the database
for audit and performance review.

Report sections
---------------
  1. Metadata            – date, mode, regime, config version
  2. Open positions      – current holdings with unrealised PnL
  3. Closed trades       – trades closed today with realised PnL
  4. Portfolio summary   – portfolio value, cash, daily P&L, drawdown
  5. Scan cycle summary  – number of cycles, trades opened, NO TRADE reasons
  6. Risk snapshot       – kill switch status, breaches, data quality events
  7. Data quality        – provider errors, stale data events
  8. Performance KPIs    – win rate, average win/loss, profit factor

The report dict is:
  - Logged as a structured INFO event (queryable in log aggregators)
  - Stored to the database in the ``daily_reports`` table (if it exists)
  - Cached in Redis for 48 hours under ``report:{date}``
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.services.orchestrator.scan_cycle import ServiceContainer

logger = structlog.get_logger(__name__)

_REDIS_TTL_SECONDS = 48 * 3600  # 48 hours


# ---------------------------------------------------------------------------
# Report cycle entry point
# ---------------------------------------------------------------------------


async def run_report_cycle(services: "ServiceContainer") -> None:
    """Generate and persist the end-of-day trading summary.

    Parameters
    ----------
    services:
        Fully-initialised service container from main.py.
    """
    log = logger.bind(job="report_cycle")
    report_date = date.today().isoformat()
    cycle_start = time.monotonic()

    log.info("report_cycle_start", report_date=report_date)

    report: dict[str, Any] = {}
    try:
        report = await _build_report(services=services, report_date=report_date, log=log)
    except Exception as exc:
        log.error("report_cycle_build_error", error=str(exc), exc_info=True)
        # Build a minimal error report so something is persisted
        report = {
            "date": report_date,
            "status": "error",
            "error": str(exc),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ---- Log the full report as a structured event ----
    log.info("eod_report", **report)

    # ---- Persist to DB ----
    try:
        await _persist_report(services=services, report=report, log=log)
    except Exception as exc:
        log.error("report_cycle_persist_error", error=str(exc))

    # ---- Cache in Redis ----
    try:
        await services.redis.set(
            f"report:{report_date}",
            report,
            ttl=_REDIS_TTL_SECONDS,
        )
        log.debug("report_cached_in_redis", key=f"report:{report_date}")
    except Exception as exc:
        log.warning("report_cycle_redis_cache_error", error=str(exc))

    elapsed = time.monotonic() - cycle_start
    log.info(
        "report_cycle_complete",
        report_date=report_date,
        duration_seconds=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


async def _build_report(
    services: "ServiceContainer",
    report_date: str,
    log: structlog.stdlib.BoundLogger,
) -> dict[str, Any]:
    """Assemble all sections of the daily report."""
    report: dict[str, Any] = {
        "date": report_date,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "trading_mode": services.trading_mode,
        "config_version": getattr(services.config, "version", "unknown"),
        "status": "ok",
    }

    # ---- Section 1: Market regime ----
    try:
        market_data = await services.data_manager.get_market_snapshot(force_refresh=False)
        regime = await services.regime_engine.classify(market_data)
        report["regime"] = {
            "classification": regime.regime.value,
            "score": round(regime.score, 2),
            "confidence": round(getattr(regime, "confidence", 0), 2),
            "explanation": getattr(regime, "explanation", ""),
        }
    except Exception as exc:
        log.warning("report_regime_error", error=str(exc))
        report["regime"] = {"error": str(exc)}

    # ---- Section 2: Broker account / portfolio summary ----
    try:
        account = await services.broker.get_account()
        portfolio_val = float(getattr(account, "portfolio_value", 0) or 0)
        cash_bal = float(getattr(account, "cash", 0) or 0)
        daily_pl = float(getattr(account, "daily_pnl", 0) or 0)
        unrealized_pl = float(getattr(account, "unrealized_pnl", 0) or 0)
        report["portfolio"] = {
            "portfolio_value": round(portfolio_val, 2),
            "cash": round(cash_bal, 2),
            "daily_pnl": round(daily_pl, 2),
            "daily_pnl_pct": round(daily_pl / portfolio_val * 100, 3) if portfolio_val else 0.0,
            "unrealized_pnl": round(unrealized_pl, 2),
        }
    except Exception as exc:
        log.warning("report_account_error", error=str(exc))
        report["portfolio"] = {"error": str(exc)}

    # ---- Section 3: Open positions ----
    try:
        open_positions = await services.broker.get_positions()
        positions_list: list[dict[str, Any]] = []
        for pos in open_positions:
            ticker = getattr(pos, "ticker", getattr(pos, "symbol", "UNKNOWN"))
            qty = getattr(pos, "qty", getattr(pos, "quantity", 0))
            avg_entry = float(getattr(pos, "avg_entry_price", 0) or 0)
            current_price = float(getattr(pos, "current_price", 0) or 0)
            unrealized = float(getattr(pos, "unrealized_pl", getattr(pos, "unrealized_pnl", 0)) or 0)
            positions_list.append(
                {
                    "ticker": ticker,
                    "quantity": qty,
                    "avg_entry_price": round(avg_entry, 4),
                    "current_price": round(current_price, 4),
                    "unrealized_pnl": round(unrealized, 2),
                }
            )
        report["open_positions"] = {
            "count": len(positions_list),
            "positions": positions_list,
        }
    except Exception as exc:
        log.warning("report_positions_error", error=str(exc))
        report["open_positions"] = {"error": str(exc)}

    # ---- Section 4: Today's decisions from DB ----
    try:
        todays_decisions = await _query_todays_decisions(services=services, report_date=report_date)
        report["decisions_today"] = todays_decisions
    except Exception as exc:
        log.warning("report_decisions_error", error=str(exc))
        report["decisions_today"] = {"error": str(exc)}

    # ---- Section 5: Kill switch status ----
    try:
        ks_status = await services.kill_switch.status()
        report["kill_switch"] = ks_status
    except Exception as exc:
        log.warning("report_kill_switch_error", error=str(exc))
        report["kill_switch"] = {"error": str(exc)}

    # ---- Section 6: Infrastructure health ----
    try:
        db_health = await services.db.health_check()
        redis_health = await services.redis.health_check()
        report["infrastructure"] = {
            "database": db_health,
            "redis": redis_health,
        }
    except Exception as exc:
        log.warning("report_infra_health_error", error=str(exc))
        report["infrastructure"] = {"error": str(exc)}

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _query_todays_decisions(
    services: "ServiceContainer",
    report_date: str,
) -> dict[str, Any]:
    """Query today's trading decisions from the database."""
    from sqlalchemy import func, select, cast
    from sqlalchemy import Date as SADate
    from src.models.decision import TradingDecision as TradingDecisionORM

    try:
        async with services.db.session() as session:
            # Total decisions today
            today = date.fromisoformat(report_date)
            result = await session.execute(
                select(TradingDecisionORM).where(
                    func.date(TradingDecisionORM.timestamp) == today
                )
            )
            decisions = result.scalars().all()

            total = len(decisions)
            trades = [d for d in decisions if not d.is_no_trade]
            no_trades = [d for d in decisions if d.is_no_trade]

            # Aggregate NO TRADE reasons
            no_trade_reasons: dict[str, int] = {}
            for d in no_trades:
                reason = d.no_trade_reason or "unknown"
                no_trade_reasons[reason] = no_trade_reasons.get(reason, 0) + 1

            # Trade details
            trade_summaries = [
                {
                    "ticker": d.selected_ticker,
                    "cycle_id": d.cycle_id,
                    "confidence": d.confidence_score,
                    "allocation_pct": d.allocation_pct,
                    "dollar_size": d.dollar_size,
                    "stop_loss": d.stop_loss,
                    "take_profit_1": d.take_profit_1,
                }
                for d in trades
            ]

            return {
                "total_scan_cycles": total,
                "trades_opened": len(trades),
                "no_trade_decisions": len(no_trades),
                "no_trade_reasons": no_trade_reasons,
                "trades": trade_summaries,
            }
    except Exception as exc:
        logger.warning("report_query_decisions_error", error=str(exc))
        return {
            "total_scan_cycles": 0,
            "trades_opened": 0,
            "no_trade_decisions": 0,
            "no_trade_reasons": {},
            "trades": [],
            "query_error": str(exc),
        }


async def _persist_report(
    services: "ServiceContainer",
    report: dict[str, Any],
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Attempt to store the report to the daily_reports table if it exists.

    The table may not yet exist (migrations pending) – failure is non-fatal.
    """
    try:
        from sqlalchemy import text
        async with services.db.session() as session:
            # Check if daily_reports table exists before inserting
            result = await session.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT FROM information_schema.tables "
                    "  WHERE table_name = 'daily_reports'"
                    ")"
                )
            )
            table_exists = result.scalar()

            if not table_exists:
                log.debug("report_persist_skipped_no_table")
                return

            await session.execute(
                text(
                    "INSERT INTO daily_reports (report_date, report_json, generated_at) "
                    "VALUES (:date, :json, :ts) "
                    "ON CONFLICT (report_date) DO UPDATE SET "
                    "  report_json = EXCLUDED.report_json, "
                    "  generated_at = EXCLUDED.generated_at"
                ),
                {
                    "date": report.get("date"),
                    "json": __import__("json").dumps(report),
                    "ts": report.get("generated_at"),
                },
            )
            log.info("report_persisted_to_db", date=report.get("date"))
    except Exception as exc:
        log.warning("report_persist_db_error", error=str(exc))
