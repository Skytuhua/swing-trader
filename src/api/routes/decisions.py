"""
Trading decision routes.

GET /api/v1/decisions          – paginated list with optional date filter
GET /api/v1/decisions/latest   – most recent decision
GET /api/v1/decisions/{id}     – single decision by UUID
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import DecisionListResponse, DecisionResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory store (used when DB is not available)
# ---------------------------------------------------------------------------

# This store is populated with a seed record and acts as the data source when
# the real SQLAlchemy session is unavailable.  The real DB integration simply
# replaces the ``_get_decisions`` / ``_get_decision_by_id`` helpers below.

_MOCK_DECISIONS: list[dict[str, Any]] = [
    {
        "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "cycle_id": "2026-03-21T09:00:00",
        "timestamp": datetime(2026, 3, 21, 14, 0, 0, tzinfo=timezone.utc),
        "market_regime": "favorable",
        "regime_confidence": 78.5,
        "top_candidates_json": [
            {"ticker": "AAPL", "score": 82.1},
            {"ticker": "NVDA", "score": 79.3},
        ],
        "selected_ticker": "AAPL",
        "is_no_trade": False,
        "no_trade_reason": None,
        "reason_summary": "Strong momentum on AAPL with bullish MACD crossover and high relative volume.",
        "technical_score": 84.0,
        "news_score": 76.0,
        "sentiment_score": 71.0,
        "risk_reward_score": 85.0,
        "liquidity_score": 95.0,
        "regime_alignment_score": 80.0,
        "confidence_score": 82.0,
        "entry_price_low": 172.50,
        "entry_price_high": 173.00,
        "entry_method": "limit",
        "stop_loss": 168.00,
        "take_profit_1": 180.00,
        "take_profit_2": 188.00,
        "trailing_stop_rule": "2×ATR14",
        "allocation_pct": 9.5,
        "dollar_size": 9500.0,
        "share_quantity": 55,
        "invalidation_conditions": ["Price closes below 200-day SMA"],
        "data_quality_flags": {},
        "config_version": "0.1.0",
        "model_version": "0.1.0",
        "created_at": datetime(2026, 3, 21, 14, 0, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 21, 14, 0, 1, tzinfo=timezone.utc),
    },
    {
        "id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
        "cycle_id": "2026-03-20T09:00:00",
        "timestamp": datetime(2026, 3, 20, 14, 0, 0, tzinfo=timezone.utc),
        "market_regime": "mixed",
        "regime_confidence": 52.0,
        "top_candidates_json": [],
        "selected_ticker": None,
        "is_no_trade": True,
        "no_trade_reason": "Regime confidence below threshold (52 < 60). No candidates passed scoring threshold.",
        "reason_summary": "Mixed market regime — skipping trade cycle.",
        "technical_score": None,
        "news_score": None,
        "sentiment_score": None,
        "risk_reward_score": None,
        "liquidity_score": None,
        "regime_alignment_score": None,
        "confidence_score": None,
        "entry_price_low": None,
        "entry_price_high": None,
        "entry_method": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "trailing_stop_rule": None,
        "allocation_pct": None,
        "dollar_size": None,
        "share_quantity": None,
        "invalidation_conditions": None,
        "data_quality_flags": {},
        "config_version": "0.1.0",
        "model_version": "0.1.0",
        "created_at": datetime(2026, 3, 20, 14, 0, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 20, 14, 0, 1, tzinfo=timezone.utc),
    },
]


# ---------------------------------------------------------------------------
# Data-access helpers (swap these for real DB queries when ready)
# ---------------------------------------------------------------------------

async def _get_decisions(
    request: Request,
    page: int,
    page_size: int,
    date_from: date | None,
    date_to: date | None,
    ticker: str | None,
    is_no_trade: bool | None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, total) from DB or in-memory store."""
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.decision import TradingDecision

        async with get_db_session() as session:
            stmt = select(TradingDecision)

            if date_from:
                dt_from = datetime(date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc)
                stmt = stmt.where(TradingDecision.timestamp >= dt_from)
            if date_to:
                dt_to = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=timezone.utc)
                stmt = stmt.where(TradingDecision.timestamp <= dt_to)
            if ticker:
                stmt = stmt.where(TradingDecision.selected_ticker == ticker.upper())
            if is_no_trade is not None:
                stmt = stmt.where(TradingDecision.is_no_trade == is_no_trade)

            from sqlalchemy import func, select as _select

            count_stmt = _select(func.count()).select_from(stmt.subquery())
            total_result = await session.execute(count_stmt)
            total: int = total_result.scalar_one()

            stmt = (
                stmt.order_by(TradingDecision.timestamp.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_orm_to_dict(r) for r in rows], total

    except Exception as exc:
        logger.debug("decisions_db_fallback", reason=str(exc))

    # Fall back to in-memory mock
    items = list(_MOCK_DECISIONS)
    if date_from:
        items = [d for d in items if d["timestamp"].date() >= date_from]
    if date_to:
        items = [d for d in items if d["timestamp"].date() <= date_to]
    if ticker:
        items = [d for d in items if d.get("selected_ticker") == ticker.upper()]
    if is_no_trade is not None:
        items = [d for d in items if d["is_no_trade"] == is_no_trade]

    items.sort(key=lambda x: x["timestamp"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    return items[start : start + page_size], total


async def _get_decision_by_id(
    request: Request, decision_id: uuid.UUID
) -> dict[str, Any] | None:
    try:
        from sqlalchemy import select

        from src.core.database import get_db_session
        from src.models.decision import TradingDecision

        async with get_db_session() as session:
            result = await session.execute(
                select(TradingDecision).where(TradingDecision.id == decision_id)
            )
            row = result.scalar_one_or_none()
            return _orm_to_dict(row) if row else None
    except Exception as exc:
        logger.debug("decision_by_id_db_fallback", reason=str(exc))

    for d in _MOCK_DECISIONS:
        if d["id"] == decision_id:
            return d
    return None


def _orm_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy model instance to a dict."""
    return {c.key: getattr(obj, c.key) for c in obj.__table__.columns}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/decisions",
    response_model=DecisionListResponse,
    summary="List trading decisions",
    description=(
        "Returns a paginated list of TradingDecision records, ordered by "
        "timestamp descending (most recent first)."
    ),
)
async def list_decisions(
    request: Request,
    page: Annotated[int, Query(ge=1, description="Page number (1-based).")] = 1,
    page_size: Annotated[int, Query(ge=1, le=500, description="Items per page.")] = 50,
    date_from: Annotated[date | None, Query(description="Filter: timestamp >= this date (YYYY-MM-DD).")] = None,
    date_to: Annotated[date | None, Query(description="Filter: timestamp <= this date (YYYY-MM-DD).")] = None,
    ticker: Annotated[str | None, Query(description="Filter by selected_ticker (case-insensitive).")] = None,
    is_no_trade: Annotated[bool | None, Query(description="Filter by is_no_trade flag.")] = None,
) -> DecisionListResponse:
    items, total = await _get_decisions(
        request, page, page_size, date_from, date_to, ticker, is_no_trade
    )
    return DecisionListResponse(
        items=[DecisionResponse(**d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/decisions/latest",
    response_model=DecisionResponse,
    summary="Latest trading decision",
    description="Returns the most recently created TradingDecision record.",
)
async def get_latest_decision(request: Request) -> DecisionResponse:
    items, total = await _get_decisions(request, 1, 1, None, None, None, None)
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trading decisions found.",
        )
    return DecisionResponse(**items[0])


@router.get(
    "/decisions/{decision_id}",
    response_model=DecisionResponse,
    summary="Get decision by ID",
    description="Returns a single TradingDecision record by its UUID.",
)
async def get_decision(
    decision_id: uuid.UUID,
    request: Request,
) -> DecisionResponse:
    decision = await _get_decision_by_id(request, decision_id)
    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision {decision_id} not found.",
        )
    return DecisionResponse(**decision)
