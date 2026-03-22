"""
Kill switch: Redis-backed emergency stop that closes all positions
and halts all new trading activity until manually deactivated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import KillSwitchActiveError

if TYPE_CHECKING:
    from src.services.execution.base import BrokerAdapter

logger = structlog.get_logger(__name__)

_KEY = "kill_switch"
_REASON_KEY = "kill_switch_reason"
_ACTIVATED_AT_KEY = "kill_switch_activated_at"
_DEACTIVATED_BY_KEY = "kill_switch_deactivated_by"


class KillSwitch:
    """Redis-backed trading kill switch.

    When activated:
    1. Sets ``kill_switch = "active"`` in Redis.
    2. Immediately closes all open positions via the broker.
    3. Logs a CRITICAL event.

    When deactivated:
    1. Removes the Redis key.
    2. Logs a WARNING event (manual override required).

    The ``is_active`` property reads directly from Redis on every call to
    ensure correctness across multiple processes / containers.

    Usage::

        ks = KillSwitch(redis_manager, broker)
        if await ks.is_active:
            raise KillSwitchActiveError("System halted.")

        await ks.activate("Max drawdown exceeded")
        await ks.deactivate()
    """

    def __init__(
        self,
        redis: Any,                                 # RedisManager from src.core.redis_client
        broker: "BrokerAdapter | None" = None,      # Optional – needed to close positions
        namespace: str = "",                        # Allows multiple environments on one Redis
    ) -> None:
        self.redis = redis
        self.broker = broker
        self._ns = f"{namespace}:" if namespace else ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    async def is_active(self) -> bool:
        """Return True if the kill switch is currently active (async)."""
        try:
            value = await self.redis.get(f"{self._ns}{_KEY}")
            return value == "active"
        except Exception as exc:
            # If Redis is unreachable, treat as active (fail-safe)
            logger.error("kill_switch_redis_read_error", error=str(exc))
            return True

    def is_active_sync(self) -> bool:
        """Synchronous fallback – reads from an in-memory cache if available.

        Only accurate if the async is_active has been called recently.
        Prefer ``await ks.is_active`` wherever possible.
        """
        return getattr(self, "_cached_active", False)

    @property
    async def reason(self) -> str | None:
        """Return the reason the kill switch was activated, or None."""
        try:
            return await self.redis.get(f"{self._ns}{_REASON_KEY}")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def activate(self, reason: str, triggered_by: str = "system") -> None:
        """Activate the kill switch.

        1. Write ``kill_switch = active`` to Redis.
        2. Store reason, timestamp, and source.
        3. Close all open positions immediately.
        4. Log CRITICAL.

        Args:
            reason:       Human-readable reason (e.g. "Max drawdown exceeded").
            triggered_by: Identifier of the component that triggered activation.
        """
        now_str = datetime.now(tz=timezone.utc).isoformat()

        try:
            await self.redis.set(f"{self._ns}{_KEY}", "active")
            await self.redis.set(f"{self._ns}{_REASON_KEY}", reason)
            await self.redis.set(f"{self._ns}{_ACTIVATED_AT_KEY}", now_str)
        except Exception as exc:
            logger.error("kill_switch_redis_write_error", error=str(exc))
            # Even if Redis fails, proceed with position closure

        logger.critical(
            "KILL_SWITCH_ACTIVATED",
            reason=reason,
            triggered_by=triggered_by,
            activated_at=now_str,
        )

        # Close all positions
        if self.broker is not None:
            try:
                closed_orders = await self.broker.close_all_positions()
                logger.critical(
                    "kill_switch_positions_closed",
                    count=len(closed_orders),
                    tickers=[o.ticker for o in closed_orders],
                )
            except Exception as exc:
                logger.error(
                    "kill_switch_close_positions_failed",
                    error=str(exc),
                    note="MANUAL INTERVENTION REQUIRED",
                )
        else:
            logger.warning(
                "kill_switch_no_broker",
                note="No broker set; positions NOT automatically closed.",
            )

        self._cached_active = True

    async def deactivate(self, deactivated_by: str = "operator") -> None:
        """Deactivate the kill switch.

        This does NOT automatically resume trading – the operator must
        explicitly restart the trading cycle after reviewing the system.

        Args:
            deactivated_by: Who/what deactivated the switch.
        """
        try:
            await self.redis.delete(f"{self._ns}{_KEY}")
            await self.redis.delete(f"{self._ns}{_REASON_KEY}")
            await self.redis.delete(f"{self._ns}{_ACTIVATED_AT_KEY}")
            await self.redis.set(f"{self._ns}{_DEACTIVATED_BY_KEY}", deactivated_by)
        except Exception as exc:
            logger.error("kill_switch_redis_delete_error", error=str(exc))

        self._cached_active = False
        logger.warning(
            "KILL_SWITCH_DEACTIVATED",
            deactivated_by=deactivated_by,
            note="Manual review required before resuming trading.",
        )

    async def check_or_raise(self) -> None:
        """Raise KillSwitchActiveError if active.

        Convenience wrapper for pre-trade checks.
        """
        if await self.is_active:
            reason = await self.reason
            raise KillSwitchActiveError(
                f"Kill switch is active. Reason: {reason or 'unknown'}. "
                "Deactivate manually after reviewing system state."
            )

    async def status(self) -> dict[str, Any]:
        """Return a status dict for monitoring / health endpoints."""
        active = await self.is_active
        result: dict[str, Any] = {"active": active}

        if active:
            result["reason"] = await self.reason
            try:
                result["activated_at"] = await self.redis.get(
                    f"{self._ns}{_ACTIVATED_AT_KEY}"
                )
            except Exception:
                result["activated_at"] = None

        return result
