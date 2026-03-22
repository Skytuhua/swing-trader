"""
APScheduler-based scheduler for the swing-trader system.

Jobs registered:
  - scan_job       : run_scan_cycle()  – configurable cron (default: 9:30 and every
                     2 hours until 15:30 on trading days)
  - monitor_job    : run_monitor_cycle() – every 30 seconds during market hours
  - report_job     : run_report_cycle() – daily at 16:05 ET (after market close)
  - health_job     : _health_check()   – every 5 minutes
  - reconcile_job  : _reconcile()      – every 1 minute during market hours

All jobs run inside the application's running asyncio event loop via
AsyncIOScheduler.  Missed executions during downtime are *not* re-triggered
(misfire_grace_time = 60 s; max_instances = 1 per job).

Usage::

    scheduler = create_scheduler(services)
    scheduler.start()
    ...
    scheduler.shutdown()
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.core.metrics import METRICS

if TYPE_CHECKING:
    from src.services.orchestrator.scan_cycle import ServiceContainer

logger = structlog.get_logger(__name__)

# Eastern Time zone string understood by APScheduler / zoneinfo
_ET = "America/New_York"

# Grace period: if a job starts more than 60 s late it is skipped
_MISFIRE_GRACE = 60


# ---------------------------------------------------------------------------
# Scheduler wrapper
# ---------------------------------------------------------------------------


class TradingScheduler:
    """Thin wrapper around APScheduler's AsyncIOScheduler.

    Holds a reference to all service dependencies and wires jobs to the
    correct async functions at construction time.

    Parameters
    ----------
    services:
        Populated ``ServiceContainer`` (from main.py DI bootstrap).
    scan_cron:
        APScheduler cron kwargs for the scan job.  Defaults to market open
        (09:30 ET) and every 2 hours until 15:30 ET on weekdays.
    """

    def __init__(
        self,
        services: "ServiceContainer",
        scan_cron: dict[str, Any] | None = None,
    ) -> None:
        self.services = services
        self._scheduler = AsyncIOScheduler(timezone=_ET)

        # Default scan schedule: 09:30, 11:30, 13:30, 15:30 Mon-Fri (ET)
        default_scan_cron: dict[str, Any] = {
            "day_of_week": "mon-fri",
            "hour": "9,11,13,15",
            "minute": "30",
            "timezone": _ET,
        }
        self._scan_cron = scan_cron or default_scan_cron

        self._register_jobs()

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register all scheduled jobs with misfire/concurrency settings."""

        # ---- Scan job ----
        self._scheduler.add_job(
            self._scan_job,
            trigger=CronTrigger(**self._scan_cron),
            id="scan_job",
            name="Candidate Scan Cycle",
            misfire_grace_time=_MISFIRE_GRACE,
            max_instances=1,
            replace_existing=True,
        )

        # ---- Monitor job (every 30 s during market hours) ----
        self._scheduler.add_job(
            self._monitor_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                second="*/30",
                timezone=_ET,
            ),
            id="monitor_job",
            name="Position Monitor Cycle",
            misfire_grace_time=10,
            max_instances=1,
            replace_existing=True,
        )

        # ---- Daily report at market close (16:05 ET) ----
        self._scheduler.add_job(
            self._report_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=5,
                timezone=_ET,
            ),
            id="report_job",
            name="End-of-Day Report",
            misfire_grace_time=300,
            max_instances=1,
            replace_existing=True,
        )

        # ---- Health check every 5 minutes ----
        self._scheduler.add_job(
            self._health_job,
            trigger=IntervalTrigger(minutes=5),
            id="health_job",
            name="System Health Check",
            misfire_grace_time=60,
            max_instances=1,
            replace_existing=True,
        )

        # ---- Reconcile every 1 minute during market hours ----
        self._scheduler.add_job(
            self._reconcile_job,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                minute="*",
                timezone=_ET,
            ),
            id="reconcile_job",
            name="Order Reconciliation",
            misfire_grace_time=30,
            max_instances=1,
            replace_existing=True,
        )

        logger.info(
            "scheduler_jobs_registered",
            jobs=[j.id for j in self._scheduler.get_jobs()],
        )

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    async def _scan_job(self) -> None:
        """Trigger one full scan cycle with a unique cycle_id."""
        from src.services.orchestrator.scan_cycle import run_scan_cycle

        cycle_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "-" + str(uuid.uuid4())[:8]
        logger.info("scheduler_scan_job_start", cycle_id=cycle_id)
        try:
            await run_scan_cycle(cycle_id=cycle_id, services=self.services)
        except Exception as exc:
            logger.error("scheduler_scan_job_error", cycle_id=cycle_id, error=str(exc))

    async def _monitor_job(self) -> None:
        """Trigger one position monitor cycle."""
        from src.services.orchestrator.monitor_cycle import run_monitor_cycle

        logger.debug("scheduler_monitor_job_start")
        try:
            await run_monitor_cycle(services=self.services)
        except Exception as exc:
            logger.error("scheduler_monitor_job_error", error=str(exc))

    async def _report_job(self) -> None:
        """Generate and persist the end-of-day report."""
        from src.services.orchestrator.report_cycle import run_report_cycle

        report_date = datetime.now(tz=timezone.utc).date().isoformat()
        logger.info("scheduler_report_job_start", date=report_date)
        try:
            await run_report_cycle(services=self.services)
        except Exception as exc:
            logger.error("scheduler_report_job_error", date=report_date, error=str(exc))

    async def _health_job(self) -> None:
        """Run lightweight health checks and log status."""
        log = logger.bind(job="health_check")
        try:
            db_status = await self.services.db.health_check()
            redis_status = await self.services.redis.health_check()

            log.info(
                "health_check",
                db=db_status.get("status"),
                db_latency_ms=db_status.get("latency_ms"),
                redis=redis_status.get("status"),
                redis_latency_ms=redis_status.get("latency_ms"),
            )

            if db_status.get("status") != "ok":
                log.error("health_check_db_degraded", detail=db_status.get("detail"))
            if redis_status.get("status") != "ok":
                log.error("health_check_redis_degraded", detail=redis_status.get("detail"))

        except Exception as exc:
            log.error("health_check_error", error=str(exc))

    async def _reconcile_job(self) -> None:
        """Reconcile open orders against broker state."""
        log = logger.bind(job="reconcile")
        try:
            await self.services.order_manager.reconcile()
            log.debug("reconcile_complete")
        except Exception as exc:
            log.error("reconcile_error", error=str(exc))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler (non-blocking)."""
        self._scheduler.start()
        logger.info(
            "scheduler_started",
            job_count=len(self._scheduler.get_jobs()),
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the scheduler, optionally waiting for running jobs."""
        self._scheduler.shutdown(wait=wait)
        logger.info("scheduler_shutdown")

    def pause(self) -> None:
        """Pause all jobs (e.g. during kill-switch activation)."""
        self._scheduler.pause()
        logger.warning("scheduler_paused")

    def resume(self) -> None:
        """Resume all jobs after a pause."""
        self._scheduler.resume()
        logger.info("scheduler_resumed")

    def get_job_states(self) -> list[dict[str, Any]]:
        """Return a summary of all registered jobs and their next run times."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": (
                        job.next_run_time.isoformat() if job.next_run_time else None
                    ),
                    "trigger": str(job.trigger),
                }
            )
        return jobs

    @property
    def running(self) -> bool:
        """True if the scheduler is active (started and not shut down)."""
        return self._scheduler.running


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_scheduler(
    services: "ServiceContainer",
    scan_cron: dict[str, Any] | None = None,
) -> TradingScheduler:
    """Create and return a fully configured TradingScheduler.

    Parameters
    ----------
    services:
        Populated service container (dependency-injected at startup).
    scan_cron:
        Optional override for the scan job cron expression (APScheduler
        CronTrigger kwargs).  Defaults to 09:30, 11:30, 13:30, 15:30 ET
        Mon–Fri.

    Returns
    -------
    TradingScheduler
        Ready to call ``.start()`` on.
    """
    scheduler = TradingScheduler(services=services, scan_cron=scan_cron)
    logger.info("create_scheduler_complete")
    return scheduler
