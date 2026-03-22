"""
Order routes.

GET /api/v1/orders       – list orders with optional filters
GET /api/v1/orders/{id}  – single order by UUID
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import OrderListResponse, OrderResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory seed data
# ---------------------------------------------------------------------------

_MOCK_ORDERS: list[dict[str, Any]] = [
    {
        "id": uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        "decision_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "position_id": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "ticker": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "quantity": 55,
        "limit_price": 173.00,
        "stop_price": None,
        "trail_amount": None,
        "trail_percent": None,
        "time_in_force": "day",
        "status": "filled",
        "broker_order_id": "alpaca-order-001",
        "submitted_at": datetime(2026, 3, 21, 14, 4, 0, tzinfo=timezone.utc),
        "filled_at": datetime(2026, 3, 21, 14, 5, 0, tzinfo=timezone.utc),
        "filled_qty": 55,
        "filled_avg_price": 172.75,
        "commission": 0.0,
        "reject_reason": None,
        "is_entry": True,
        "is_exit": False,
        "idempotency_key": "decision-11111111-entry-001",
        "created_at": datetime(2026, 3, 21, 14, 4, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 21, 14, 5, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        "decision_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "position_id": uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        "ticker": "MSFT",
        "side": "buy",
        "order_type": "limit",
        "quantity": 24,
        "limit_price": 415.50,
        "stop_price": None,
        "trail_amount": None,
        "trail_percent": None,
        "time_in_force": "day",
        "status": "filled",
        "broker_order_id": "alpaca-order-002",
        "submitted_at": datetime(2026, 3, 10, 14, 4, 0, tzinfo=timezone.utc),
        "filled_at": datetime(2026, 3, 10, 14, 5, 0, tzinfo=timezone.utc),
        "filled_qty": 24,
        "filled_avg_price": 415.00,
        "commission": 0.0,
        "reject_reason": None,
        "is_entry": True,
        "is_exit": False,
        "idempotency_key": "decision-33333333-entry-001",
        "created_at": datetime(2026, 3, 10, 14, 4, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 10, 14, 5, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        "decision_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "position_id": uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        "ticker": "MSFT",
        "side": "sell",
        "order_type": "limit",
        "quantity": 24,
        "limit_price": 435.00,
        "stop_price": None,
        "trail_amount": None,
        "trail_percent": None,
        "time_in_force": "day",
        "status": "filled",
        "broker_order_id": "alpaca-order-003",
        "submitted_at": datetime(2026, 3, 17, 14, 31, 0, tzinfo=timezone.utc),
        "filled_at": datetime(2026, 3, 17, 14, 32, 0, tzinfo=timezone.utc),
        "filled_qty": 24,
        "filled_avg_price": 435.20,
        "commission": 0.0,
        "reject_reason": None,
        "is_entry": False,
        "is_exit": True,
        "idempotency_key": "position-bbbbbbbb-exit-tp1",
        "created_at": datetime(2026, 3, 17, 14, 31, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 17, 14, 32, 1, tzinfo=timezone.utc),
    },
]


# ---------------------------------------------------------------------------
# Data-access helpers
# ---------------------------------------------------------------------------

async def _get_orders(
    request: Request,
    page: int,
    page_size: int,
    ticker: str | None,
    order_status: str | None,
    side: str | None,
    is_entry: bool | None,
    is_exit: bool | None,
    position_id: uuid.UUID | None,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.order import Order

        async with get_db_session() as session:
            stmt = select(Order)
            if ticker:
                stmt = stmt.where(Order.ticker == ticker.upper())
            if order_status:
                stmt = stmt.where(Order.status == order_status)
            if side:
                stmt = stmt.where(Order.side == side.lower())
            if is_entry is not None:
                stmt = stmt.where(Order.is_entry == is_entry)
            if is_exit is not None:
                stmt = stmt.where(Order.is_exit == is_exit)
            if position_id is not None:
                stmt = stmt.where(Order.position_id == position_id)

            from sqlalchemy import func, select as _select

            count_stmt = _select(func.count()).select_from(stmt.subquery())
            total: int = (await session.execute(count_stmt)).scalar_one()

            stmt = (
                stmt.order_by(Order.submitted_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [{c.key: getattr(r, c.key) for c in r.__table__.columns} for r in rows], total
    except Exception as exc:
        logger.debug("orders_db_fallback", reason=str(exc))

    items = list(_MOCK_ORDERS)
    if ticker:
        items = [o for o in items if o["ticker"] == ticker.upper()]
    if order_status:
        items = [o for o in items if o["status"] == order_status]
    if side:
        items = [o for o in items if o["side"] == side.lower()]
    if is_entry is not None:
        items = [o for o in items if o["is_entry"] == is_entry]
    if is_exit is not None:
        items = [o for o in items if o["is_exit"] == is_exit]
    if position_id is not None:
        items = [o for o in items if o["position_id"] == position_id]

    items.sort(key=lambda x: x["submitted_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    return items[start : start + page_size], total


async def _get_order_by_id(
    request: Request, order_id: uuid.UUID
) -> dict[str, Any] | None:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.order import Order

        async with get_db_session() as session:
            result = await session.execute(select(Order).where(Order.id == order_id))
            row = result.scalar_one_or_none()
            if row:
                return {c.key: getattr(row, c.key) for c in row.__table__.columns}
            return None
    except Exception as exc:
        logger.debug("order_by_id_db_fallback", reason=str(exc))

    for o in _MOCK_ORDERS:
        if o["id"] == order_id:
            return o
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/orders",
    response_model=OrderListResponse,
    summary="List orders",
    description=(
        "Returns a paginated list of broker order records, ordered by submitted_at descending. "
        "Filter by ticker, status, side, is_entry, is_exit, or position_id."
    ),
)
async def list_orders(
    request: Request,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    ticker: Annotated[str | None, Query(description="Filter by ticker symbol.")] = None,
    order_status: Annotated[
        str | None,
        Query(
            alias="status",
            description="Filter by order status: pending, submitted, filled, cancelled, rejected.",
        ),
    ] = None,
    side: Annotated[str | None, Query(description="Filter by order side: buy or sell.")] = None,
    is_entry: Annotated[bool | None, Query(description="Filter entry orders only.")] = None,
    is_exit: Annotated[bool | None, Query(description="Filter exit orders only.")] = None,
    position_id: Annotated[uuid.UUID | None, Query(description="Filter by position UUID.")] = None,
) -> OrderListResponse:
    items, total = await _get_orders(
        request, page, page_size, ticker, order_status, side, is_entry, is_exit, position_id
    )
    return OrderListResponse(
        items=[OrderResponse(**o) for o in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Get order by ID",
    description="Returns a single Order record by its UUID.",
)
async def get_order(
    order_id: uuid.UUID,
    request: Request,
) -> OrderResponse:
    order = await _get_order_by_id(request, order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )
    return OrderResponse(**order)
