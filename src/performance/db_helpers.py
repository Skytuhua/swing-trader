"""Database optimization helpers.

Provides batch insert utilities and index recommendations for
PostgreSQL with async SQLAlchemy.

Usage::

    from src.performance.db_helpers import batch_insert

    async with db.session() as session:
        await batch_insert(session, trade_objects, batch_size=100)
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession


async def batch_insert(
    session: AsyncSession,
    objects: Sequence[Any],
    batch_size: int = 100,
) -> int:
    """Insert ORM objects in batches for better throughput.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    objects : Sequence
        ORM model instances to insert.
    batch_size : int
        Number of objects per batch flush.

    Returns
    -------
    int
        Number of objects inserted.
    """
    total = 0
    for i in range(0, len(objects), batch_size):
        batch = objects[i : i + batch_size]
        session.add_all(batch)
        await session.flush()
        total += len(batch)
    return total


# Recommended indexes for performance-critical queries.
# These should be created via Alembic migration or at startup.
RECOMMENDED_INDEXES = [
    # Trade history: fast lookups by symbol and date range
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades (ticker);",
    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);",
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker_timestamp ON trades (ticker, timestamp);",
    # Positions: fast lookups by status and ticker
    "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);",
    "CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions (ticker);",
    "CREATE INDEX IF NOT EXISTS idx_positions_ticker_status ON positions (ticker, status);",
    # Decisions: audit trail queries
    "CREATE INDEX IF NOT EXISTS idx_decisions_cycle_id ON trading_decisions (cycle_id);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON trading_decisions (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON trading_decisions (selected_ticker);",
    # Orders: fast lookup by status and broker ID
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);",
    "CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders (broker_order_id);",
]


async def ensure_indexes(engine) -> list[str]:
    """Create recommended indexes if they don't exist.

    Parameters
    ----------
    engine : AsyncEngine
        SQLAlchemy async engine.

    Returns
    -------
    list[str]
        List of executed CREATE INDEX statements.
    """
    from sqlalchemy import text

    executed = []
    async with engine.begin() as conn:
        for stmt in RECOMMENDED_INDEXES:
            try:
                await conn.execute(text(stmt))
                executed.append(stmt)
            except Exception:
                pass  # Index may already exist or table may not exist yet
    return executed
