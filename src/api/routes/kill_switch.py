"""
Kill-switch control routes.

GET  /api/v1/kill-switch            – current status
POST /api/v1/kill-switch/activate   – activate with a reason
POST /api/v1/kill-switch/deactivate – deactivate
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import KillSwitchRequest, KillSwitchResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory fallback state (used when Redis is unavailable)
# ---------------------------------------------------------------------------

_INMEM_STATE: dict[str, Any] = {
    "active": False,
    "reason": None,
    "activated_at": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_kill_switch_status(request: Request) -> dict[str, Any]:
    """Read kill-switch state from Redis or fall back to in-memory."""
    try:
        from src.core.redis_client import get_redis
        from src.services.risk.kill_switch import KillSwitch

        redis = await get_redis()
        ks = KillSwitch(redis)
        return await ks.status()
    except Exception as exc:
        logger.warning("kill_switch_redis_read_failed", error=str(exc))
        return dict(_INMEM_STATE)


async def _activate_kill_switch(request: Request, reason: str) -> dict[str, Any]:
    """Activate the kill switch and return the new state."""
    from datetime import datetime, timezone

    now_str = datetime.now(tz=timezone.utc).isoformat()

    try:
        from src.core.redis_client import get_redis
        from src.services.risk.kill_switch import KillSwitch

        redis = await get_redis()
        ks = KillSwitch(redis)
        await ks.activate(reason=reason, triggered_by="api_operator")
        return {
            "active": True,
            "reason": reason,
            "activated_at": now_str,
        }
    except Exception as exc:
        logger.error("kill_switch_activate_redis_failed", error=str(exc))
        # Apply to in-memory state as best-effort
        _INMEM_STATE["active"] = True
        _INMEM_STATE["reason"] = reason
        _INMEM_STATE["activated_at"] = now_str
        return dict(_INMEM_STATE)


async def _deactivate_kill_switch(request: Request) -> dict[str, Any]:
    """Deactivate the kill switch and return the new state."""
    try:
        from src.core.redis_client import get_redis
        from src.services.risk.kill_switch import KillSwitch

        redis = await get_redis()
        ks = KillSwitch(redis)
        await ks.deactivate(deactivated_by="api_operator")
        return {"active": False, "reason": None, "activated_at": None}
    except Exception as exc:
        logger.error("kill_switch_deactivate_redis_failed", error=str(exc))
        _INMEM_STATE["active"] = False
        _INMEM_STATE["reason"] = None
        _INMEM_STATE["activated_at"] = None
        return dict(_INMEM_STATE)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/kill-switch",
    response_model=KillSwitchResponse,
    summary="Kill-switch status",
    description=(
        "Returns the current state of the trading kill switch. "
        "When ``active`` is ``true``, all new trading is halted and any open positions "
        "will have been closed automatically."
    ),
)
async def get_kill_switch_status(request: Request) -> KillSwitchResponse:
    state = await _get_kill_switch_status(request)
    return KillSwitchResponse(
        active=state.get("active", False),
        reason=state.get("reason"),
        activated_at=state.get("activated_at"),
    )


@router.post(
    "/kill-switch/activate",
    response_model=KillSwitchResponse,
    status_code=status.HTTP_200_OK,
    summary="Activate kill switch",
    description=(
        "Activates the trading kill switch, immediately halting all new trades. "
        "If a broker adapter is configured, all open positions will be closed automatically. "
        "\n\n"
        "**This is an irreversible action** — you must explicitly call "
        "``POST /api/v1/kill-switch/deactivate`` to resume trading."
    ),
)
async def activate_kill_switch(
    body: KillSwitchRequest,
    request: Request,
) -> KillSwitchResponse:
    if body.action != "activate":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Use this endpoint with action='activate'. For deactivation use /kill-switch/deactivate.",
        )

    current = await _get_kill_switch_status(request)
    if current.get("active"):
        logger.warning(
            "kill_switch_already_active",
            existing_reason=current.get("reason"),
        )
        # Idempotent — return current state rather than erroring
        return KillSwitchResponse(
            active=True,
            reason=current.get("reason"),
            activated_at=current.get("activated_at"),
        )

    logger.critical(
        "kill_switch_activate_requested",
        reason=body.reason,
        source="api",
    )
    new_state = await _activate_kill_switch(request, body.reason)
    return KillSwitchResponse(
        active=new_state["active"],
        reason=new_state.get("reason"),
        activated_at=new_state.get("activated_at"),
    )


@router.post(
    "/kill-switch/deactivate",
    response_model=KillSwitchResponse,
    status_code=status.HTTP_200_OK,
    summary="Deactivate kill switch",
    description=(
        "Deactivates the trading kill switch, allowing new trades to be placed. "
        "\n\n"
        "**The trading engine does not automatically resume** — the operator must "
        "restart the scan cycle after reviewing system state."
    ),
)
async def deactivate_kill_switch(
    request: Request,
) -> KillSwitchResponse:
    current = await _get_kill_switch_status(request)
    if not current.get("active"):
        logger.info("kill_switch_already_inactive")
        return KillSwitchResponse(active=False, reason=None, activated_at=None)

    logger.warning(
        "kill_switch_deactivate_requested",
        previous_reason=current.get("reason"),
        source="api",
    )
    new_state = await _deactivate_kill_switch(request)
    return KillSwitchResponse(
        active=new_state["active"],
        reason=new_state.get("reason"),
        activated_at=new_state.get("activated_at"),
    )
