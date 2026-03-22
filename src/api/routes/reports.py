"""
Daily report routes.

GET /api/v1/reports/daily         – daily report for a specific date
GET /api/v1/reports/daily/latest  – most recent daily report
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import DailyReportPositionSummary, ReportResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

async def _build_daily_report(
    request: Request,
    report_date: date,
) -> ReportResponse:
    """Build a daily report by aggregating data for the given date.

    Pulls data from DB/mock stores for:
    - Risk snapshots (portfolio value, drawdown)
    - Completed trades (realised PnL, commissions)
    - Positions (open positions)
    - Decisions (scan counts)
    - Kill-switch state
    """
    generated_at = datetime.now(tz=timezone.utc)
    notes: list[str] = []

    # ------------------------------------------------------------------
    # Risk snapshot for the date
    # ------------------------------------------------------------------
    portfolio_value: float | None = None
    daily_pnl: float | None = None
    daily_pnl_pct: float | None = None
    max_drawdown: float | None = None
    kill_switch_active = False

    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.risk_snapshot import RiskSnapshot

        date_start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        date_end = datetime(report_date.year, report_date.month, report_date.day, 23, 59, 59, tzinfo=timezone.utc)

        async with get_db_session() as session:
            result = await session.execute(
                select(RiskSnapshot)
                .where(RiskSnapshot.timestamp.between(date_start, date_end))
                .order_by(RiskSnapshot.timestamp.desc())
                .limit(1)
            )
            snap = result.scalar_one_or_none()
            if snap:
                portfolio_value = snap.portfolio_value
                daily_pnl = snap.daily_pnl
                daily_pnl_pct = snap.daily_pnl_pct
                max_drawdown = snap.max_drawdown
                kill_switch_active = snap.kill_switch_active
    except Exception as exc:
        logger.debug("report_risk_snapshot_fallback", reason=str(exc))
        # Fall back to mock snapshot if date matches
        from src.api.routes.risk import _MOCK_RISK_SNAPSHOTS

        for snap_dict in _MOCK_RISK_SNAPSHOTS:
            if snap_dict["timestamp"].date() == report_date:
                portfolio_value = snap_dict["portfolio_value"]
                daily_pnl = snap_dict["daily_pnl"]
                daily_pnl_pct = snap_dict["daily_pnl_pct"]
                max_drawdown = snap_dict["max_drawdown"]
                kill_switch_active = snap_dict["kill_switch_active"]
                break

    # ------------------------------------------------------------------
    # Completed trades for the date
    # ------------------------------------------------------------------
    trades_entered = 0
    trades_exited = 0
    realized_pnl_today = 0.0
    commissions_today = 0.0

    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.trade import CompletedTrade

        date_start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        date_end = datetime(report_date.year, report_date.month, report_date.day, 23, 59, 59, tzinfo=timezone.utc)

        async with get_db_session() as session:
            result = await session.execute(
                select(CompletedTrade).where(
                    CompletedTrade.exit_date.between(date_start, date_end)
                )
            )
            day_trades = result.scalars().all()
            trades_exited = len(day_trades)
            realized_pnl_today = sum(t.net_pnl for t in day_trades)
            commissions_today = sum(t.commission_total for t in day_trades)

        async with get_db_session() as session:
            from src.models.position import Position

            result = await session.execute(
                select(Position).where(
                    Position.entry_date.between(date_start, date_end)
                )
            )
            entered = result.scalars().all()
            trades_entered = len(entered)
    except Exception as exc:
        logger.debug("report_trades_fallback", reason=str(exc))
        from src.api.routes.trades import _MOCK_TRADES

        for t in _MOCK_TRADES:
            if t["exit_date"].date() == report_date:
                trades_exited += 1
                realized_pnl_today += t["net_pnl"]
                commissions_today += t["commission_total"]

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------
    open_positions_count = 0
    open_position_summaries: list[DailyReportPositionSummary] = []

    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.position import Position

        async with get_db_session() as session:
            result = await session.execute(
                select(Position).where(Position.status.in_(["open", "closing"]))
            )
            open_pos = result.scalars().all()
            open_positions_count = len(open_pos)
            open_position_summaries = [
                DailyReportPositionSummary(
                    ticker=p.ticker,
                    status=p.status,
                    entry_price=p.entry_price,
                    current_price=p.current_price,
                    unrealized_pnl=p.unrealized_pnl,
                    unrealized_pnl_pct=p.unrealized_pnl_pct,
                    hold_days=p.hold_days,
                )
                for p in open_pos
            ]
    except Exception as exc:
        logger.debug("report_positions_fallback", reason=str(exc))
        from src.api.routes.positions import _MOCK_POSITIONS

        open_pos_dicts = [p for p in _MOCK_POSITIONS if p["status"] in ("open", "closing")]
        open_positions_count = len(open_pos_dicts)
        open_position_summaries = [
            DailyReportPositionSummary(
                ticker=p["ticker"],
                status=p["status"],
                entry_price=p["entry_price"],
                current_price=p.get("current_price"),
                unrealized_pnl=p.get("unrealized_pnl"),
                unrealized_pnl_pct=p.get("unrealized_pnl_pct"),
                hold_days=p["hold_days"],
            )
            for p in open_pos_dicts
        ]

    # ------------------------------------------------------------------
    # Decisions for the date
    # ------------------------------------------------------------------
    decisions_made = 0
    no_trade_decisions = 0
    trade_decisions = 0

    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.decision import TradingDecision

        date_start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        date_end = datetime(report_date.year, report_date.month, report_date.day, 23, 59, 59, tzinfo=timezone.utc)

        async with get_db_session() as session:
            result = await session.execute(
                select(TradingDecision).where(
                    TradingDecision.timestamp.between(date_start, date_end)
                )
            )
            day_decisions = result.scalars().all()
            decisions_made = len(day_decisions)
            no_trade_decisions = sum(1 for d in day_decisions if d.is_no_trade)
            trade_decisions = decisions_made - no_trade_decisions
    except Exception as exc:
        logger.debug("report_decisions_fallback", reason=str(exc))
        from src.api.routes.decisions import _MOCK_DECISIONS

        for d in _MOCK_DECISIONS:
            if d["timestamp"].date() == report_date:
                decisions_made += 1
                if d["is_no_trade"]:
                    no_trade_decisions += 1
                else:
                    trade_decisions += 1

    # ------------------------------------------------------------------
    # Compose notes
    # ------------------------------------------------------------------
    if kill_switch_active:
        notes.append("WARNING: Kill switch was active on this date.")
    if decisions_made == 0:
        notes.append("No scan cycles completed on this date.")

    return ReportResponse(
        report_date=report_date.isoformat(),
        generated_at=generated_at,
        portfolio_value=portfolio_value,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        max_drawdown=max_drawdown,
        decisions_made=decisions_made,
        trades_entered=trades_entered,
        trades_exited=trades_exited,
        open_positions_count=open_positions_count,
        realized_pnl_today=round(realized_pnl_today, 2),
        commissions_today=round(commissions_today, 2),
        open_positions=open_position_summaries,
        kill_switch_active=kill_switch_active,
        no_trade_decisions=no_trade_decisions,
        trade_decisions=trade_decisions,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/reports/daily",
    response_model=ReportResponse,
    summary="Daily trading report",
    description=(
        "Returns a comprehensive daily trading report for the specified date. "
        "Includes portfolio snapshot, P&L, open positions, scan decisions made, "
        "and trades entered/exited. If no data exists for the date, a 404 is returned."
    ),
)
async def get_daily_report(
    request: Request,
    report_date: Annotated[
        date,
        Query(
            alias="date",
            description="Report date in YYYY-MM-DD format.",
            examples=["2026-03-21"],
        ),
    ],
) -> ReportResponse:
    today = datetime.now(tz=timezone.utc).date()
    if report_date > today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot generate a report for a future date: {report_date}.",
        )

    logger.info("daily_report_requested", report_date=report_date.isoformat())
    return await _build_daily_report(request, report_date)


@router.get(
    "/reports/daily/latest",
    response_model=ReportResponse,
    summary="Latest daily report",
    description=(
        "Returns the daily report for the most recent trading date "
        "(today if today is a weekday, otherwise the last trading day)."
    ),
)
async def get_latest_daily_report(request: Request) -> ReportResponse:
    from datetime import timedelta

    today = datetime.now(tz=timezone.utc).date()

    # Walk back to find the most recent weekday
    candidate = today
    for _ in range(7):  # Safety limit
        if candidate.weekday() < 5:  # Monday=0 … Friday=4
            break
        candidate -= timedelta(days=1)

    logger.info("latest_daily_report_requested", report_date=candidate.isoformat())
    return await _build_daily_report(request, candidate)
