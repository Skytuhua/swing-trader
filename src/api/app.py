"""
FastAPI application factory for the SwingTrader Admin API.

Usage::

    from src.api.app import create_app

    app = create_app(config)
    # Run with: uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Or for direct module execution (uvicorn target)::

    uvicorn src.api.app:app
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.middleware import (
    ApiKeyMiddleware,
    ErrorHandlerMiddleware,
    RequestLoggingMiddleware,
)
from src.api.routes import (
    config_api,
    decisions,
    health,
    kill_switch,
    orders,
    positions,
    reports,
    risk,
    trades,
)
from src.core.logging_config import configure_logging

logger = structlog.get_logger(__name__)

# Process start time for uptime calculation
_START_TIME: float = time.time()


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events for the FastAPI application.

    On startup:
    - Configure structured logging
    - Attempt to connect to DB and Redis (non-fatal; health endpoint reports status)

    On shutdown:
    - Gracefully close DB connection pool and Redis connections
    """
    config: dict[str, Any] = app.state.config  # type: ignore[attr-defined]
    mode = config.get("app", {}).get("mode", "paper")
    log_level = config.get("app", {}).get("log_level", "INFO")

    configure_logging(log_level=log_level)
    logger.info("api_startup", mode=mode, version=config.get("app", {}).get("config_version", "unknown"))

    # ------------------------------------------------------------------
    # DB initialisation (best-effort; failures are reported via /health)
    # ------------------------------------------------------------------
    db_ok = False
    try:
        from src.core.database import init_db

        await init_db(config)
        db_ok = True
        logger.info("database_connected")
    except Exception as exc:
        logger.warning("database_connect_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Redis initialisation (best-effort)
    # ------------------------------------------------------------------
    redis_ok = False
    try:
        from src.core.redis_client import init_redis

        await init_redis(config)
        redis_ok = True
        logger.info("redis_connected")
    except Exception as exc:
        logger.warning("redis_connect_failed", error=str(exc))

    # Store connectivity state for the health endpoint
    app.state.db_ok = db_ok
    app.state.redis_ok = redis_ok
    app.state.start_time = _START_TIME

    yield  # Application is now running

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("api_shutdown")
    try:
        from src.core.database import close_db

        await close_db()
        logger.info("database_disconnected")
    except Exception as exc:
        logger.warning("database_disconnect_error", error=str(exc))

    try:
        from src.core.redis_client import close_redis

        await close_redis()
        logger.info("redis_disconnected")
    except Exception as exc:
        logger.warning("redis_disconnect_error", error=str(exc))


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the SwingTrader Admin FastAPI application.

    Parameters
    ----------
    config:
        Application configuration dict (loaded from YAML + env overrides).
        If ``None``, a minimal default configuration is used so the app can
        still start for development purposes.

    Returns
    -------
    FastAPI
        Fully configured FastAPI instance ready to be served by uvicorn.
    """
    if config is None:
        config = _default_config()

    app_cfg: dict[str, Any] = config.get("app", {})
    debug: bool = app_cfg.get("debug", False)
    mode: str = app_cfg.get("mode", "paper")
    api_key: str | None = config.get("api", {}).get("api_key") or None
    cors_origins: list[str] = config.get("api", {}).get(
        "cors_origins",
        ["http://localhost:3000", "http://localhost:5173"],
    )

    # ------------------------------------------------------------------
    # FastAPI instance
    # ------------------------------------------------------------------
    app = FastAPI(
        title="SwingTrader Admin API",
        description=(
            "Administrative and monitoring API for the autonomous SwingTrader bot.\n\n"
            "Provides read-only access to trading decisions, positions, orders, completed "
            "trades, risk metrics, and configuration, plus control endpoints for the kill "
            "switch."
        ),
        version=app_cfg.get("config_version", "0.1.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=debug,
        lifespan=_lifespan,
        contact={
            "name": "SwingTrader Ops",
            "email": "ops@swingtrader.local",
        },
        license_info={
            "name": "Proprietary",
        },
        openapi_tags=[
            {"name": "health", "description": "Health and liveness checks."},
            {"name": "decisions", "description": "Trading decision records."},
            {"name": "positions", "description": "Open and historical positions."},
            {"name": "orders", "description": "Broker order records."},
            {"name": "trades", "description": "Completed trade records and statistics."},
            {"name": "risk", "description": "Portfolio risk snapshots."},
            {"name": "config", "description": "Sanitised application configuration."},
            {"name": "kill-switch", "description": "Emergency trading halt controls."},
            {"name": "reports", "description": "Daily trading reports."},
        ],
    )

    # Store config on app state so lifespan and routes can access it
    app.state.config = config
    app.state.mode = mode
    app.state.start_time = _START_TIME

    # ------------------------------------------------------------------
    # Middleware  (applied in reverse order – last added = outermost)
    # ------------------------------------------------------------------

    # 1. CORS (outermost – must handle OPTIONS before any auth check)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # 2. API-key authentication (after CORS, before logging)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    # 3. Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # 4. Error handler (innermost – catches errors from route handlers)
    app.add_middleware(ErrorHandlerMiddleware, debug=debug)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    app.include_router(health.router, tags=["health"])
    app.include_router(decisions.router, prefix="/api/v1", tags=["decisions"])
    app.include_router(positions.router, prefix="/api/v1", tags=["positions"])
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(trades.router, prefix="/api/v1", tags=["trades"])
    app.include_router(risk.router, prefix="/api/v1", tags=["risk"])
    app.include_router(config_api.router, prefix="/api/v1", tags=["config"])
    app.include_router(kill_switch.router, prefix="/api/v1", tags=["kill-switch"])
    app.include_router(reports.router, prefix="/api/v1", tags=["reports"])

    # ------------------------------------------------------------------
    # Prometheus metrics endpoint
    # ------------------------------------------------------------------
    @app.get(
        "/metrics",
        include_in_schema=False,
        summary="Prometheus metrics",
    )
    async def prometheus_metrics() -> Any:
        from fastapi.responses import Response as FastAPIResponse

        data = generate_latest()
        return FastAPIResponse(
            content=data,
            media_type=CONTENT_TYPE_LATEST,
        )

    # ------------------------------------------------------------------
    # Root redirect
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "name": "SwingTrader Admin API",
                "version": app_cfg.get("config_version", "0.1.0"),
                "mode": mode,
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
            }
        )

    logger.info(
        "app_created",
        title="SwingTrader Admin API",
        mode=mode,
        debug=debug,
        cors_origins=cors_origins,
        api_key_enabled=bool(api_key),
    )

    return app


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def _default_config() -> dict[str, Any]:
    """Return a minimal in-memory configuration for development / testing."""
    return {
        "app": {
            "mode": "paper",
            "log_level": "INFO",
            "debug": True,
            "config_version": "0.1.0",
            "model_version": "0.1.0",
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "timezone": "America/New_York",
        },
        "api": {
            "api_key": None,
            "cors_origins": ["*"],
        },
        "database": {
            "url": "postgresql+asyncpg://swing:swing@localhost:5432/swingtrader",
        },
        "redis": {
            "url": "redis://localhost:6379/0",
        },
    }


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn / gunicorn direct usage)
# ---------------------------------------------------------------------------
def _load_config_from_yaml() -> dict[str, Any]:
    """Load configuration from the default YAML file if available."""
    import os

    import yaml

    config_path = os.environ.get(
        "SWING_TRADER_CONFIG",
        "/home/user/workspace/swing-trader/config/default.yaml",
    )
    try:
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        # Override mode from environment
        mode_env = os.environ.get("SWING_TRADER_MODE")
        if mode_env and "app" in cfg:
            cfg["app"]["mode"] = mode_env
        return cfg
    except FileNotFoundError:
        return _default_config()


app = create_app(_load_config_from_yaml())
