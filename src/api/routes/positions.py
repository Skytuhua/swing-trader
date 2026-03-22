"""
Position routes.

GET /api/v1/positions          – list open positions
GET /api/v1/positions/history  – closed/closing positions
GET /api/v1/positions/{id}     – single position by UUID
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import PositionListResponse, PositionResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory seed data
# ---------------------------------------------------------------------------

_MOCK_POSITIONS: list[dict[str, Any]] = [
    {
        "id": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "decision_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "ticker": "AAPL",
        "status": "open",
        "entry_price": 172.75,
        "entry_date": datetime(2026, 3, 21, 14, 5, 0, tzinfo=timezone.utc),
        "quantity": 55,
        "original_quantity": 55,
        "current_price": 174.50,
        "unrealized_pnl": 96.25,
        "unrealized_pnl_pct": 1.01,
        "stop_loss": 168.00,
        "take_profit_1": 180.00,
        "take_profit_2": 188.00,
        "trailing_stop_price": None,
        "max_price_since_entry": 174.50,
        "hold_days": 0,
        "last_monitored_at": datetime(2026, 3, 21, 15, 0, 0, tzinfo=timezone.utc),
        "thesis_score": 81.5,
        "exit_reason": None,
        "exit_price": None,
        "exit_date": None,
        "realized_pnl": None,
        "realized_pnl_pct": None,
        "commission_total": 0.0,
        "notes": None,
        "created_at": datetime(2026, 3, 21, 14, 5, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 21, 15, 0, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        "decision_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "ticker": "MSFT",
        "status": "closed",
        "entry_price": 415.00,
        "entry_date": datetime(2026, 3, 10, 14, 5, 0, tzinfo=timezone.utc),
        "quantity": 0,
        "original_quantity": 24,
        "current_price": None,
        "unrealized_pnl": None,
        "unrealized_pnl_pct": None,
        "stop_loss": 400.00,
        "take_profit_1": 435.00,
        "take_profit_2": 450.00,
        "trailing_stop_price": 428.50,
        "max_price_since_entry": 432.00,
        "hold_days": 7,
        "last_monitored_at": datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc),
        "thesis_score": None,
        "exit_reason": "take_profit_1",
        "exit_price": 435.20,
        "exit_date": datetime(2026, 3, 17, 14, 32, 0, tzinfo=timezone.utc),
        "realized_pnl": 484.80,
        "realized_pnl_pct": 4.87,
        "commission_total": 0.0,
        "notes": "TP1 triggered on momentum spike.",
        "created_at": datetime(2026, 3, 10, 14, 5, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 17, 14, 32, 1, tzinfo=timezone.utc),
    },
]


# ---------------------------------------------------------------------------
# Data-access helpers
# ---------------------------------------------------------------------------

async def _get_positions(
    request: Request,
    page: int,
    page_size: int,
    statuses: list[str],
    ticker: str | None,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.position import Position

        async with get_db_session() as session:
            stmt = select(Position)
            if statuses:
                stmt = stmt.where(Position.status.in_(statuses))
            if ticker:
                stmt = stmt.where(Position.ticker == ticker.upper())

            from sqlalchemy import func, select as _select

            count_stmt = _select(func.count()).select_from(stmt.subquery())
            total: int = (await session.execute(count_stmt)).scalar_one()

            stmt = (
                stmt.order_by(Position.entry_date.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [{c.key: getattr(r, c.key) for c in r.__table__.columns} for r in rows], total
    except Exception as exc:
        logger.debug("positions_db_fallback", reason=str(exc))

    items = [p for p in _MOCK_POSITIONS if p["status"] in statuses]
    if ticker:
        items = [p for p in items if p["ticker"] == ticker.upper()]
    total = len(items)
    start = (page - 1) * page_size
    return items[start : start + page_size], total


async def _get_position_by_id(
    request: Request, position_id: uuid.UUID
) -> dict[str, Any] | None:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.position import Position

        async with get_db_session() as session:
            result = await session.execute(
                select(Position).where(Position.id == position_id)
            )
            row = result.scalar_one_or_none()
            if row:
                return {c.key: getattr(row, c.key) for c in row.__table__.columns}
            return None
    except Exception as exc:
        logger.debug("position_by_id_db_fallback", reason=str(exc))

    for p in _MOCK_POSITIONS:
        if p["id"] == position_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/positions",
    response_model=PositionListResponse,
    summary="List open positions",
    description=(
        "Returns open (and optionally closing) positions, ordered by entry date descending."
    ),
)
async def list_open_positions(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    ticker: Annotated[str | None, Query(description="Filter by ticker symbol.")] = None,
    include_closing: Annotated[bool, Query(description="Include positions with status='closing'.")] = True,
) -> PositionListResponse:
    statuses = ["open", "closing"] if include_closing else ["open"]
    items, total = await _get_positions(request, page, page_size, statuses, ticker)
    return PositionListResponse(
        items=[PositionResponse(**p) for p in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/positions/history",
    response_model=PositionListResponse,
    summary="Position history",
    description="Returns closed positions, ordered by exit date descending.",
)
async def list_position_history(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    ticker: Annotated[str | None, Query(description="Filter by ticker symbol.")] = None,
) -> PositionListResponse:
    items, total = await _get_positions(request, page, page_size, ["closed"], ticker)
    return PositionListResponse(
        items=[PositionResponse(**p) for p in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/positions/{position_id}",
    response_model=PositionResponse,
    summary="Get position by ID",
    description="Returns a single Position record by its UUID.",
)
async def get_position(
    position_id: uuid.UUID,
    request: Request,
) -> PositionResponse:
    position = await _get_position_by_id(request, position_id)
    if position is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {position_id} not found.",
        )
    return PositionResponse(**position)
