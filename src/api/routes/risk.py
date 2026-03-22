"""
Risk snapshot routes.

GET /api/v1/risk/snapshot  – current (most recent) risk snapshot
GET /api/v1/risk/history   – paginated historical risk snapshots
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import RiskSnapshotListResponse, RiskSnapshotResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory seed data
# ---------------------------------------------------------------------------

_MOCK_RISK_SNAPSHOTS: list[dict[str, Any]] = [
    {
        "id": uuid.UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1"),
        "timestamp": datetime(2026, 3, 21, 15, 0, 0, tzinfo=timezone.utc),
        "portfolio_value": 105_096.25,
        "cash": 95_596.25,
        "equity": 9_500.00,
        "total_risk_pct": 0.95,
        "daily_pnl": 96.25,
        "daily_pnl_pct": 0.092,
        "max_drawdown": 0.0,
        "peak_portfolio_value": 105_096.25,
        "open_positions_count": 1,
        "kill_switch_active": False,
        "kill_switch_reason": None,
        "data_quality": "good",
        "position_details_json": [
            {
                "ticker": "AAPL",
                "position_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "risk_pct": 0.95,
                "unrealized_pnl": 96.25,
            }
        ],
        "created_at": datetime(2026, 3, 21, 15, 0, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 21, 15, 0, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2"),
        "timestamp": datetime(2026, 3, 21, 14, 0, 0, tzinfo=timezone.utc),
        "portfolio_value": 105_000.00,
        "cash": 105_000.00,
        "equity": 0.0,
        "total_risk_pct": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "peak_portfolio_value": 105_000.00,
        "open_positions_count": 0,
        "kill_switch_active": False,
        "kill_switch_reason": None,
        "data_quality": "good",
        "position_details_json": [],
        "created_at": datetime(2026, 3, 21, 14, 0, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 21, 14, 0, 1, tzinfo=timezone.utc),
    },
]


# ---------------------------------------------------------------------------
# Data-access helpers
# ---------------------------------------------------------------------------

async def _get_latest_snapshot(request: Request) -> dict[str, Any] | None:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.risk_snapshot import RiskSnapshot

        async with get_db_session() as session:
            result = await session.execute(
                select(RiskSnapshot).order_by(RiskSnapshot.timestamp.desc()).limit(1)
            )
            row = result.scalar_one_or_none()
            if row:
                return {c.key: getattr(row, c.key) for c in row.__table__.columns}
            return None
    except Exception as exc:
        logger.debug("risk_snapshot_db_fallback", reason=str(exc))

    if _MOCK_RISK_SNAPSHOTS:
        return sorted(_MOCK_RISK_SNAPSHOTS, key=lambda x: x["timestamp"], reverse=True)[0]
    return None


async def _get_snapshot_history(
    request: Request,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.risk_snapshot import RiskSnapshot

        async with get_db_session() as session:
            stmt = select(RiskSnapshot)

            from sqlalchemy import func, select as _select

            count_stmt = _select(func.count()).select_from(stmt.subquery())
            total: int = (await session.execute(count_stmt)).scalar_one()

            stmt = (
                stmt.order_by(RiskSnapshot.timestamp.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [{c.key: getattr(r, c.key) for c in r.__table__.columns} for r in rows], total
    except Exception as exc:
        logger.debug("risk_history_db_fallback", reason=str(exc))

    items = sorted(_MOCK_RISK_SNAPSHOTS, key=lambda x: x["timestamp"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    return items[start : start + page_size], total


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/risk/snapshot",
    response_model=RiskSnapshotResponse,
    summary="Current risk snapshot",
    description=(
        "Returns the most recent portfolio risk snapshot, including current portfolio value, "
        "cash balance, equity, total risk percentage, daily P&L, max drawdown, and kill-switch state."
    ),
)
async def get_risk_snapshot(request: Request) -> RiskSnapshotResponse:
    snapshot = await _get_latest_snapshot(request)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No risk snapshots found. The risk engine may not have run yet.",
        )
    return RiskSnapshotResponse(**snapshot)


@router.get(
    "/risk/history",
    response_model=RiskSnapshotListResponse,
    summary="Risk snapshot history",
    description=(
        "Returns a paginated list of historical risk snapshots, ordered by timestamp descending. "
        "Useful for charting portfolio value and drawdown over time."
    ),
)
async def get_risk_history(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 100,
) -> RiskSnapshotListResponse:
    items, total = await _get_snapshot_history(request, page, page_size)
    return RiskSnapshotListResponse(
        items=[RiskSnapshotResponse(**s) for s in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )
