"""
Alembic migration environment for SwingTrader.

This module configures Alembic to:
  - Read the database URL from the DATABASE_URL environment variable.
  - Import all SQLAlchemy ORM models so that autogenerate can detect
    schema differences between the models and the database.
  - Run migrations using an async engine (asyncpg) via a synchronous
    wrapper, which is the Alembic-recommended pattern for asyncio projects.

Usage:
    # Generate a new migration:
    alembic revision --autogenerate -m "add_positions_table"

    # Apply all pending migrations:
    alembic upgrade head

    # Downgrade one step:
    alembic downgrade -1
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ---------------------------------------------------------------------------
# Import all ORM models so Alembic's autogenerate sees the full metadata.
# ---------------------------------------------------------------------------
# Each model module must be imported here — autogenerate only detects tables
# that have been registered on Base.metadata by the time env.py runs.
from src.models.base import Base  # noqa: F401 — registers DeclarativeBase

# Trigger model registration by importing every model module.
import src.models.market_data   # noqa: F401
import src.models.indicator     # noqa: F401
import src.models.news          # noqa: F401
import src.models.sentiment     # noqa: F401
import src.models.candidate     # noqa: F401
import src.models.decision      # noqa: F401
import src.models.order         # noqa: F401
import src.models.position      # noqa: F401
import src.models.trade         # noqa: F401
import src.models.risk_snapshot # noqa: F401
import src.models.alert         # noqa: F401

# ---------------------------------------------------------------------------
# Alembic config object (wraps alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support.
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
# Read DATABASE_URL from environment; fall back to alembic.ini value.
# Alembic migrations run synchronously, so we swap the asyncpg driver for
# psycopg2. The application itself uses asyncpg at runtime.
def get_sync_url() -> str:
    """Return a synchronous PostgreSQL URL for Alembic migrations."""
    raw_url = os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", ""),
    )
    # Replace the asyncpg driver with psycopg2 for synchronous migration runs.
    # e.g. postgresql+asyncpg://... → postgresql+psycopg2://...
    # If the URL does not contain asyncpg, leave it unchanged.
    sync_url = raw_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    # Also handle bare postgresql:// (no driver specified).
    if sync_url.startswith("postgresql://"):
        sync_url = sync_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return sync_url


def get_async_url() -> str:
    """Return the asyncpg URL for the async engine."""
    raw_url = os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", ""),
    )
    # Ensure asyncpg driver is present for async engine.
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw_url


# ---------------------------------------------------------------------------
# Offline migrations (--sql mode)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL statements without connecting to the database.
    Useful for generating migration scripts to review or apply manually.

    Example:
        alembic upgrade head --sql > migration.sql
    """
    url = get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schema-level objects (sequences, enums, etc.) in comparisons.
        include_schemas=True,
        # Compare column types to catch column type changes.
        compare_type=True,
        # Detect columns that have had their server_default changed.
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations — synchronous wrapper over async engine
# ---------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    """Execute pending migrations using the given synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        compare_type=True,
        compare_server_default=True,
        # Render all ARRAY, ENUM, and other PostgreSQL-specific types correctly.
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via a sync connection proxy."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_async_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No connection pooling for migration runs.
    )

    async with connectable.connect() as connection:
        # Alembic expects a synchronous connection; run_sync bridges async → sync.
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to the live database)."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
