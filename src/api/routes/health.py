"""
Health check routes.

GET /health         – basic liveness probe
GET /health/detailed – full connectivity check including DB, Redis, and broker
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: app config / state helpers
# ---------------------------------------------------------------------------

def _get_uptime(request: Request) -> float:
    start: float = getattr(request.app.state, "start_time", time.time())
    return time.time() - start


def _get_version(request: Request) -> str:
    cfg: dict[str, Any] = getattr(request.app.state, "config", {})
    return cfg.get("app", {}).get("config_version", "unknown")


def _get_mode(request: Request) -> str:
    return str(getattr(request.app.state, "mode", "paper"))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description=(
        "Lightweight liveness probe. Returns 200 if the API process is running.\n\n"
        "The ``checks`` dict will be empty for this endpoint. "
        "Use ``GET /health/detailed`` for full connectivity information."
    ),
)
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(_get_uptime(request), 2),
        version=_get_version(request),
        mode=_get_mode(request),
        checks={},
    )


# ---------------------------------------------------------------------------
# GET /health/detailed
# ---------------------------------------------------------------------------

@router.get(
    "/health/detailed",
    response_model=HealthResponse,
    summary="Detailed health check",
    description=(
        "Full connectivity health check. Probes the database, Redis, and (optionally) "
        "the broker. Returns 200 when all critical dependencies are healthy, "
        "or 503 when one or more is unavailable."
    ),
)
async def health_check_detailed(request: Request) -> JSONResponse:
    checks: dict[str, Any] = {}
    all_ok = True

    # ------------------------------------------------------------------
    # Database check
    # ------------------------------------------------------------------
    db_ok: bool = getattr(request.app.state, "db_ok", False)
    if db_ok:
        # Attempt a live ping
        try:
            from src.core.database import get_db_session

            async with get_db_session() as session:
                await session.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["database"] = {"status": "ok"}
        except Exception as exc:
            logger.warning("health_db_check_failed", error=str(exc))
            checks["database"] = {"status": "error", "detail": str(exc)}
            all_ok = False
    else:
        checks["database"] = {"status": "unavailable", "detail": "DB not initialised"}
        all_ok = False

    # ------------------------------------------------------------------
    # Redis check
    # ------------------------------------------------------------------
    redis_ok: bool = getattr(request.app.state, "redis_ok", False)
    if redis_ok:
        try:
            from src.core.redis_client import get_redis

            redis = await get_redis()
            await redis.ping()
            checks["redis"] = {"status": "ok"}
        except Exception as exc:
            logger.warning("health_redis_check_failed", error=str(exc))
            checks["redis"] = {"status": "error", "detail": str(exc)}
            all_ok = False
    else:
        checks["redis"] = {"status": "unavailable", "detail": "Redis not initialised"}
        all_ok = False

    # ------------------------------------------------------------------
    # Broker check (best-effort, non-critical for health status)
    # ------------------------------------------------------------------
    try:
        from src.services.execution.alpaca_broker import AlpacaBroker

        cfg: dict[str, Any] = getattr(request.app.state, "config", {})
        broker_cfg = cfg.get("broker", {})
        if broker_cfg.get("paper", True):
            checks["broker"] = {"status": "paper_mode", "provider": "paper"}
        else:
            checks["broker"] = {"status": "unknown", "provider": broker_cfg.get("provider", "unknown")}
    except Exception as exc:
        checks["broker"] = {"status": "unchecked", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Kill switch state
    # ------------------------------------------------------------------
    try:
        from src.core.redis_client import get_redis as _get_redis
        from src.services.risk.kill_switch import KillSwitch

        _redis = await _get_redis()
        ks = KillSwitch(_redis)
        ks_active = await ks.is_active
        checks["kill_switch"] = {
            "active": ks_active,
            "status": "active" if ks_active else "inactive",
        }
        if ks_active:
            # Kill switch being active is a warning, not a hard failure
            checks["kill_switch"]["reason"] = await ks.reason
    except Exception as exc:
        checks["kill_switch"] = {"status": "unknown", "detail": str(exc)}

    overall_status = "ok" if all_ok else "degraded"
    http_status = 200 if all_ok else 503

    body = HealthResponse(
        status=overall_status,
        uptime_seconds=round(_get_uptime(request), 2),
        version=_get_version(request),
        mode=_get_mode(request),
        checks=checks,
    )

    return JSONResponse(
        status_code=http_status,
        content=body.model_dump(mode="json"),
    )
