"""
Custom ASGI middleware for the SwingTrader Admin API.

Provides:
- ErrorHandlerMiddleware  – catches all unhandled exceptions and returns
  structured JSON error responses.
- RequestLoggingMiddleware – logs each request/response cycle with timing.
- ApiKeyMiddleware         – optional X-API-Key header authentication.
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# Paths that are exempt from API-key authentication
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/detailed",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
})


# ---------------------------------------------------------------------------
# Error handler middleware
# ---------------------------------------------------------------------------

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch all unhandled exceptions and return a structured JSON error.

    This prevents raw tracebacks from leaking to API consumers and ensures
    every error response has the shape::

        {"detail": "<message>", "code": "<code>"}
    """

    def __init__(self, app: ASGIApp, debug: bool = False) -> None:
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001
            request_id = request.state.__dict__.get("request_id", "unknown")
            log = logger.bind(
                request_id=request_id,
                path=request.url.path,
                method=request.method,
            )

            # Determine HTTP status from known exception types
            from src.core.exceptions import (
                BrokerError,
                ConfigurationError,
                DataProviderError,
                SwingTraderError,
            )

            if isinstance(exc, ConfigurationError):
                status_code = 500
                code = "configuration_error"
            elif isinstance(exc, BrokerError):
                status_code = 502
                code = "broker_error"
            elif isinstance(exc, DataProviderError):
                status_code = 503
                code = "data_provider_error"
            elif isinstance(exc, SwingTraderError):
                status_code = 500
                code = "internal_error"
            else:
                status_code = 500
                code = "unexpected_error"

            detail = str(exc) if str(exc) else "An unexpected error occurred."

            log.error(
                "unhandled_exception",
                error_code=code,
                status_code=status_code,
                exc_type=type(exc).__name__,
                exc_detail=detail,
            )

            if self.debug:
                detail = f"{detail}\n\n{traceback.format_exc()}"

            return JSONResponse(
                status_code=status_code,
                content={"detail": detail, "code": code},
            )


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every inbound request and its response status/duration.

    Assigns a unique ``request_id`` (UUID4) to each request and stores it
    on ``request.state`` so downstream handlers and the error middleware can
    reference it.
    """

    # Paths where verbose logging is suppressed to avoid noise
    _QUIET_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind context variables for this async task
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        log = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        quiet = request.url.path in self._QUIET_PATHS

        if not quiet:
            log.info("request_started")

        try:
            response: Response = await call_next(request)
        except Exception:
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            structlog.contextvars.clear_contextvars()

        if not quiet:
            log.info(
                "request_finished",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        else:
            log.debug(
                "request_finished",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

        # Attach request-id to the response for tracing
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# API-key authentication middleware
# ---------------------------------------------------------------------------

class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validate the ``X-API-Key`` header when an API key is configured.

    If ``api_key`` is ``None`` or empty the middleware is a no-op and all
    requests are allowed through (useful for development / paper mode).

    Requests to paths in ``_AUTH_EXEMPT_PATHS`` are always allowed.
    """

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        super().__init__(app)
        self._api_key: str | None = api_key or None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # If no key configured, pass through
        if not self._api_key:
            return await call_next(request)

        # Exempt health/metrics/docs paths
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != self._api_key:
            logger.warning(
                "api_key_auth_failed",
                path=request.url.path,
                method=request.method,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or missing X-API-Key header.",
                    "code": "unauthorized",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
