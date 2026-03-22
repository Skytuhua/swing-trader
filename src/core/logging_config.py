"""
Structured logging configuration using structlog.

Provides:
- JSON output in production (non-TTY / log_level != DEBUG)
- Coloured pretty-print output in development
- Log level driven by ``AppConfig.log_level``
- Context binding helpers for trade_id, cycle_id, request_id

Usage
-----
    from src.core.logging_config import configure_logging, get_logger

    configure_logging()

    logger = get_logger(__name__)
    logger.info("scan_started", cycle_id="abc123", ticker_count=42)

    # Bind context for the duration of a request / trade cycle
    bound_logger = logger.bind(trade_id="t-001", cycle_id="c-002")
    bound_logger.info("order_submitted", order_id="ord-xyz")
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

import structlog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    log_level: str = "INFO",
    force_json: bool | None = None,
) -> None:
    """Configure structlog and stdlib logging for the application.

    This function is idempotent; calling it multiple times is safe.

    Parameters
    ----------
    log_level:
        Root log level string (``"DEBUG"``, ``"INFO"``, etc.).
    force_json:
        If ``True`` always emit JSON.  If ``False`` always emit coloured
        console output.  If ``None`` (default) JSON is used when stdout
        is not a TTY (i.e. in production / containers).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    is_tty = sys.stdout.isatty()
    use_json = not is_tty if force_json is None else force_json

    # ------------------------------------------------------------------
    # Shared processors run on every log record regardless of renderer
    # ------------------------------------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json:
        # Production: render as JSON for log aggregators
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # Development: human-readable coloured output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers in production
    _configure_third_party_levels(level)


def _configure_third_party_levels(level: int) -> None:
    """Suppress overly chatty dependencies in non-debug runs."""
    noisy = [
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "asyncpg",
        "httpx",
        "httpcore",
        "uvicorn.access",
        "apscheduler",
    ]
    suppress_level = max(level, logging.WARNING)
    for name in noisy:
        logging.getLogger(name).setLevel(suppress_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for *name*.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Context binding helpers
# ---------------------------------------------------------------------------


def bind_trade_context(
    trade_id: str | None = None,
    cycle_id: str | None = None,
    request_id: str | None = None,
    ticker: str | None = None,
    **extra: Any,
) -> None:
    """Bind trading context variables for the current async task.

    All subsequent log calls in the same async context will include these
    fields automatically.  Uses ``structlog.contextvars`` which is
    task-local (compatible with asyncio tasks).

    Parameters
    ----------
    trade_id:
        Unique identifier of the current trade / position.
    cycle_id:
        Scan cycle identifier.
    request_id:
        HTTP request ID for tracing.
    ticker:
        Stock ticker symbol for the current operation.
    **extra:
        Any additional key/value pairs to bind.
    """
    ctx: dict[str, Any] = {}
    if trade_id is not None:
        ctx["trade_id"] = trade_id
    if cycle_id is not None:
        ctx["cycle_id"] = cycle_id
    if request_id is not None:
        ctx["request_id"] = request_id
    if ticker is not None:
        ctx["ticker"] = ticker
    ctx.update(extra)
    structlog.contextvars.bind_contextvars(**ctx)


def clear_trade_context() -> None:
    """Clear all bound context variables for the current async task."""
    structlog.contextvars.clear_contextvars()


def configure_from_settings() -> None:
    """Convenience helper: load settings and configure logging in one call."""
    from src.core.config import get_settings

    settings = get_settings()
    configure_logging(log_level=settings.app.log_level)
