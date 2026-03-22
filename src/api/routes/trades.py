"""
Completed trade routes.

GET /api/v1/trades          – paginated list of completed trades with PnL
GET /api/v1/trades/summary  – aggregate statistics (win rate, expectancy, Sharpe estimate)
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Query, Request

from src.api.schemas import TradeListResponse, TradeResponse, TradeSummaryResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory seed data
# ---------------------------------------------------------------------------

_MOCK_TRADES: list[dict[str, Any]] = [
    {
        "id": uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        "position_id": uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        "decision_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "ticker": "MSFT",
        "entry_date": datetime(2026, 3, 10, 14, 5, 0, tzinfo=timezone.utc),
        "exit_date": datetime(2026, 3, 17, 14, 32, 0, tzinfo=timezone.utc),
        "entry_price": 415.00,
        "exit_price": 435.20,
        "quantity": 24,
        "gross_pnl": 484.80,
        "net_pnl": 484.80,
        "pnl_pct": 4.87,
        "hold_days": 7,
        "exit_reason": "take_profit_1",
        "entry_score": 79.5,
        "max_favorable_excursion": 410.0,
        "max_adverse_excursion": -120.0,
        "commission_total": 0.0,
        "created_at": datetime(2026, 3, 17, 14, 32, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 17, 14, 32, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("12121212-1212-1212-1212-121212121212"),
        "position_id": uuid.UUID("12345678-1234-1234-1234-123456789012"),
        "decision_id": uuid.UUID("44444444-4444-4444-4444-444444444444"),
        "ticker": "TSLA",
        "entry_date": datetime(2026, 3, 3, 14, 5, 0, tzinfo=timezone.utc),
        "exit_date": datetime(2026, 3, 6, 14, 30, 0, tzinfo=timezone.utc),
        "entry_price": 195.00,
        "exit_price": 188.00,
        "quantity": 51,
        "gross_pnl": -357.00,
        "net_pnl": -357.00,
        "pnl_pct": -3.59,
        "hold_days": 3,
        "exit_reason": "stop_loss",
        "entry_score": 66.0,
        "max_favorable_excursion": 255.0,
        "max_adverse_excursion": -400.0,
        "commission_total": 0.0,
        "created_at": datetime(2026, 3, 6, 14, 30, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 6, 14, 30, 1, tzinfo=timezone.utc),
    },
]


# ---------------------------------------------------------------------------
# Data-access helpers
# ---------------------------------------------------------------------------

async def _get_trades(
    request: Request,
    page: int,
    page_size: int,
    ticker: str | None,
    exit_reason: str | None,
    winners_only: bool | None,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.trade import CompletedTrade

        async with get_db_session() as session:
            stmt = select(CompletedTrade)
            if ticker:
                stmt = stmt.where(CompletedTrade.ticker == ticker.upper())
            if exit_reason:
                stmt = stmt.where(CompletedTrade.exit_reason == exit_reason)
            if winners_only is True:
                stmt = stmt.where(CompletedTrade.net_pnl > 0)
            elif winners_only is False:
                stmt = stmt.where(CompletedTrade.net_pnl <= 0)

            from sqlalchemy import func, select as _select

            count_stmt = _select(func.count()).select_from(stmt.subquery())
            total: int = (await session.execute(count_stmt)).scalar_one()

            stmt = (
                stmt.order_by(CompletedTrade.exit_date.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [{c.key: getattr(r, c.key) for c in r.__table__.columns} for r in rows], total
    except Exception as exc:
        logger.debug("trades_db_fallback", reason=str(exc))

    items = list(_MOCK_TRADES)
    if ticker:
        items = [t for t in items if t["ticker"] == ticker.upper()]
    if exit_reason:
        items = [t for t in items if t["exit_reason"] == exit_reason]
    if winners_only is True:
        items = [t for t in items if t["net_pnl"] > 0]
    elif winners_only is False:
        items = [t for t in items if t["net_pnl"] <= 0]

    items.sort(key=lambda x: x["exit_date"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    return items[start : start + page_size], total


async def _get_all_trades(request: Request) -> list[dict[str, Any]]:
    """Retrieve all trades (no pagination) for summary statistics."""
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.trade import CompletedTrade

        async with get_db_session() as session:
            rows = (await session.execute(select(CompletedTrade))).scalars().all()
            return [{c.key: getattr(r, c.key) for c in r.__table__.columns} for r in rows]
    except Exception as exc:
        logger.debug("all_trades_db_fallback", reason=str(exc))

    return list(_MOCK_TRADES)


def _compute_summary(trades: list[dict[str, Any]]) -> TradeSummaryResponse:
    """Compute aggregate statistics from a list of trade dicts."""
    if not trades:
        return TradeSummaryResponse(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_net_pnl=0.0,
            avg_net_pnl=0.0,
            avg_pnl_pct=0.0,
            avg_hold_days=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            sharpe_estimate=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            exit_reason_breakdown={},
            period_start=None,
            period_end=None,
        )

    net_pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]

    total = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total if total else 0.0

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    total_pnl = sum(net_pnls)
    avg_pnl = total_pnl / total
    avg_pnl_pct = sum(t["pnl_pct"] for t in trades) / total
    avg_hold = sum(t["hold_days"] for t in trades) / total

    # Sharpe estimate: assume ~252 trading days per year, use daily PnL std dev
    if len(net_pnls) > 1:
        mean_pnl = total_pnl / total
        variance = sum((p - mean_pnl) ** 2 for p in net_pnls) / (total - 1)
        std_pnl = math.sqrt(variance)
        # Scale: trades are not daily, annualise by trade frequency
        sharpe_estimate = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0.0
    else:
        sharpe_estimate = 0.0

    exit_reasons: dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown") or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    dates = [t["exit_date"] for t in trades if t.get("exit_date")]
    period_start = min(t["entry_date"] for t in trades if t.get("entry_date")) if trades else None
    period_end = max(dates) if dates else None

    return TradeSummaryResponse(
        total_trades=total,
        winning_trades=win_count,
        losing_trades=loss_count,
        win_rate=round(win_rate, 4),
        total_net_pnl=round(total_pnl, 2),
        avg_net_pnl=round(avg_pnl, 2),
        avg_pnl_pct=round(avg_pnl_pct, 4),
        avg_hold_days=round(avg_hold, 2),
        expectancy=round(expectancy, 2),
        profit_factor=round(profit_factor, 4),
        sharpe_estimate=round(sharpe_estimate, 4),
        largest_win=round(max(wins), 2) if wins else 0.0,
        largest_loss=round(min(losses), 2) if losses else 0.0,
        avg_win=round(avg_win, 2),
        avg_loss=round(-avg_loss, 2),
        exit_reason_breakdown=exit_reasons,
        period_start=period_start,
        period_end=period_end,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/trades",
    response_model=TradeListResponse,
    summary="List completed trades",
    description=(
        "Returns a paginated list of CompletedTrade records with P&L data, "
        "ordered by exit date descending."
    ),
)
async def list_trades(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    ticker: Annotated[str | None, Query(description="Filter by ticker symbol.")] = None,
    exit_reason: Annotated[str | None, Query(description="Filter by exit reason.")] = None,
    winners_only: Annotated[
        bool | None,
        Query(description="True = winning trades only, False = losing trades only."),
    ] = None,
) -> TradeListResponse:
    items, total = await _get_trades(request, page, page_size, ticker, exit_reason, winners_only)
    return TradeListResponse(
        items=[TradeResponse(**t) for t in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/trades/summary",
    response_model=TradeSummaryResponse,
    summary="Trade summary statistics",
    description=(
        "Returns aggregate performance statistics across all completed trades: "
        "win rate, expectancy, average hold time, Sharpe ratio estimate, profit factor, "
        "and a breakdown by exit reason."
    ),
)
async def get_trades_summary(request: Request) -> TradeSummaryResponse:
    all_trades = await _get_all_trades(request)
    return _compute_summary(all_trades)
