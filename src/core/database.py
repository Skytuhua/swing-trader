"""
Async SQLAlchemy 2.0 database engine and session management.

Uses asyncpg as the underlying database driver for PostgreSQL.  The
``DatabaseManager`` class owns the engine lifecycle and exposes a health
check suitable for monitoring endpoints.

Usage
-----
    from src.core.database import DatabaseManager

    db = DatabaseManager()
    await db.init()

    async with db.session() as session:
        result = await session.execute(select(MyModel))

    await db.close()
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from src.core.config import Settings

logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> AsyncEngine:
    """Create and return a configured async SQLAlchemy engine.

    Parameters
    ----------
    settings:
        Application settings object (``get_settings()``).

    Returns
    -------
    AsyncEngine
        Configured engine ready for use.
    """
    db_cfg = settings.database
    return create_async_engine(
        db_cfg.url,
        pool_size=db_cfg.pool_size,
        max_overflow=db_cfg.max_overflow,
        pool_pre_ping=db_cfg.pool_pre_ping,
        pool_recycle=db_cfg.pool_recycle,
        echo=db_cfg.echo,
        future=True,
    )


class DatabaseManager:
    """Manages the async database engine and session factory.

    Attributes
    ----------
    engine:
        The underlying async SQLAlchemy engine (available after ``init``).
    session_factory:
        Callable that creates ``AsyncSession`` instances.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            from src.core.config import get_settings

            settings = get_settings()
        self._settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    async def init(self) -> None:
        """Initialise the engine and session factory.

        Call this once at application startup, before any database
        operations are performed.
        """
        self.engine = build_engine(self._settings)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        logger.info(
            "Database engine initialised",
            extra={"url": self._settings.database.url.split("@")[-1]},
        )

    async def init_db(self) -> None:
        """Create all ORM tables that do not yet exist.

        Imports all models to ensure their metadata is registered before
        the CREATE TABLE statements are issued.
        """
        if self.engine is None:
            await self.init()

        # Import models so their table metadata is registered on Base.metadata
        import src.models  # noqa: F401

        from src.models.base import Base

        async with self.engine.begin() as conn:  # type: ignore[union-attr]
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created / verified.")

    async def close(self) -> None:
        """Dispose the engine and release all pooled connections."""
        if self.engine is not None:
            await self.engine.dispose()
            logger.info("Database engine disposed.")

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Return an async context-manager that yields an ``AsyncSession``.

        The session is committed on clean exit and rolled back on any
        exception.  It is always closed on exit.

        Example
        -------
            async with db.session() as s:
                s.add(my_obj)
        """
        if self.session_factory is None:
            raise RuntimeError(
                "DatabaseManager.init() must be called before acquiring sessions."
            )
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> dict[str, object]:
        """Execute a lightweight query and return a health-status dict.

        Returns
        -------
        dict
            ``{"status": "ok", "latency_ms": float}`` on success, or
            ``{"status": "error", "detail": str}`` on failure.
        """
        import time

        if self.engine is None:
            return {"status": "error", "detail": "Engine not initialised."}

        start = time.monotonic()
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "ok", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Return the module-level database manager singleton."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = DatabaseManager()
    return _manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session.

    Usage
    -----
        @router.get("/")
        async def handler(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with get_db_manager().session() as session:
        yield session
