#!/usr/bin/env python3
"""
scripts/init_db.py — Database Initialisation for SwingTrader
=============================================================

This script:
  1. Creates the PostgreSQL database schema (all tables) from the
     SQLAlchemy ORM model metadata.
  2. Seeds an initial system-level configuration record so the application
     can start without requiring a manual DB insert.

Usage:
    # Activate your virtualenv first, then:
    python scripts/init_db.py

    # Or via Make:
    make db-init

Environment variables (set in .env or export before running):
    DATABASE_URL  — asyncpg connection string, e.g.:
                    postgresql+asyncpg://swing:swing@localhost:5432/swingtrader

Notes:
  - This script is idempotent: running it multiple times is safe.
    Tables are created with CREATE TABLE IF NOT EXISTS semantics.
  - For incremental schema changes after initial setup, use Alembic:
        make db-migrate
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env from project root (if present).
load_dotenv(PROJECT_ROOT / ".env")

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Import Base and all models to register their metadata.
from src.models.base import Base  # noqa: F401

# Import every model module so their Table objects are attached to Base.metadata.
import src.models.market_data    # noqa: F401
import src.models.indicator      # noqa: F401
import src.models.news           # noqa: F401
import src.models.sentiment      # noqa: F401
import src.models.candidate      # noqa: F401
import src.models.decision       # noqa: F401
import src.models.order          # noqa: F401
import src.models.position       # noqa: F401
import src.models.trade          # noqa: F401
import src.models.risk_snapshot  # noqa: F401
import src.models.alert          # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    """Read DATABASE_URL from environment, raising a clear error if missing."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print(
            "\n[ERROR] DATABASE_URL environment variable is not set.\n"
            "  Set it in your .env file or export it before running this script.\n"
            "  Example:\n"
            "    DATABASE_URL=postgresql+asyncpg://swing:swing@localhost:5432/swingtrader\n"
        )
        sys.exit(1)
    # Ensure the URL uses the asyncpg driver.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def create_engine() -> AsyncEngine:
    """Create and return an async SQLAlchemy engine."""
    url = get_database_url()
    print(f"[init_db] Connecting to: {url.split('@')[-1]}")  # hide credentials
    engine = create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
    )
    return engine


async def create_all_tables(engine: AsyncEngine) -> None:
    """Create all ORM tables using CREATE TABLE IF NOT EXISTS."""
    print("[init_db] Creating database tables …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"[init_db] ✓  {len(Base.metadata.tables)} table(s) created / already exist:")
    for table_name in sorted(Base.metadata.tables.keys()):
        print(f"           - {table_name}")


async def seed_initial_config(engine: AsyncEngine) -> None:
    """
    Insert an initial system configuration record if one does not exist.

    The 'system_config' table (if present in models) stores the active
    trading config at runtime. This seed record allows the application to
    start without a manual DB insert.
    """
    # Only proceed if the system_config table was generated.
    if "system_configs" not in Base.metadata.tables:
        print("[init_db] No 'system_configs' table found — skipping seed.")
        return

    table = Base.metadata.tables["system_configs"]
    async with engine.begin() as conn:
        # Check whether a config record already exists.
        result = await conn.execute(sa.select(sa.func.count()).select_from(table))
        count = result.scalar_one()
        if count > 0:
            print(f"[init_db] ✓  system_configs already has {count} record(s) — skipping seed.")
            return

        await conn.execute(
            table.insert().values(
                trading_mode="paper",
                config_version="0.1.0",
                created_at=datetime.now(tz=timezone.utc),
                updated_at=datetime.now(tz=timezone.utc),
            )
        )
    print("[init_db] ✓  Initial system_configs record inserted (mode=paper).")


async def verify_connection(engine: AsyncEngine) -> None:
    """Run a trivial query to verify connectivity."""
    async with engine.connect() as conn:
        result = await conn.execute(sa.text("SELECT version()"))
        version = result.scalar_one()
    print(f"[init_db] ✓  Connected to PostgreSQL: {version}")


async def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("  SwingTrader — Database Initialisation")
    print("=" * 60)

    engine = await create_engine()
    try:
        await verify_connection(engine)
        await create_all_tables(engine)
        await seed_initial_config(engine)
    finally:
        await engine.dispose()

    print("=" * 60)
    print("  Database initialisation complete.")
    print("  Next steps:")
    print("    make db-migrate       # Apply Alembic migrations")
    print("    make seed-universe    # Seed the stock universe")
    print("    make run              # Start the bot (paper mode)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
