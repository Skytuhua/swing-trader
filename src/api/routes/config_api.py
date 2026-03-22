"""
Configuration routes.

GET /api/v1/config – read current application configuration (sanitised, no secrets).

API keys, credentials, database URLs, and Redis URLs are always redacted.
"""

from __future__ import annotations

import copy
from typing import Any

import structlog
from fastapi import APIRouter, Request

from src.api.schemas import ConfigResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Sensitive key patterns — any config key matching these will be redacted
# ---------------------------------------------------------------------------

_REDACT_KEYS: frozenset[str] = frozenset({
    "api_key",
    "api_secret",
    "secret",
    "secret_key",
    "password",
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "private_key",
    "url",          # DB/Redis URLs may contain credentials
    "dsn",
})

_REDACT_SECTIONS: frozenset[str] = frozenset({
    "database",
    "redis",
})


def _redact(obj: Any, parent_key: str = "") -> Any:
    """Recursively redact sensitive keys from a config dict."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            lower_k = k.lower()
            if lower_k in _REDACT_KEYS:
                result[k] = "***REDACTED***"
            elif isinstance(v, dict):
                result[k] = _redact(v, parent_key=k)
            elif isinstance(v, list):
                result[k] = [_redact(item) for item in v]
            else:
                result[k] = v
        return result
    return obj


def _build_config_response(raw_config: dict[str, Any]) -> ConfigResponse:
    """Build a ConfigResponse from the raw config dict, excluding secret sections."""
    cfg = copy.deepcopy(raw_config)

    app_cfg = cfg.get("app", {})

    # Completely omit sensitive top-level sections
    sanitised: dict[str, Any] = {}
    for section, values in cfg.items():
        if section in _REDACT_SECTIONS:
            continue
        sanitised[section] = _redact(values)

    return ConfigResponse(
        mode=app_cfg.get("mode", "paper"),
        log_level=app_cfg.get("log_level", "INFO"),
        debug=app_cfg.get("debug", False),
        config_version=app_cfg.get("config_version", "unknown"),
        model_version=app_cfg.get("model_version", "unknown"),
        api_host=app_cfg.get("api_host", "0.0.0.0"),
        api_port=app_cfg.get("api_port", 8000),
        timezone=app_cfg.get("timezone", "America/New_York"),
        universe=sanitised.get("universe", {}),
        technical=sanitised.get("technical", {}),
        scoring=sanitised.get("scoring", {}),
        risk=sanitised.get("risk", {}),
        position=sanitised.get("position", {}),
        schedule=sanitised.get("schedule", {}),
        news=_redact(sanitised.get("news", {})),
        sentiment=_redact(sanitised.get("sentiment", {})),
        secrets_redacted=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get current configuration",
    description=(
        "Returns a sanitised view of the current application configuration. "
        "API keys, secrets, database URLs, and Redis URLs are always redacted. "
        "The ``secrets_redacted`` field will always be ``true``."
    ),
)
async def get_config(request: Request) -> ConfigResponse:
    raw_config: dict[str, Any] = getattr(request.app.state, "config", {})

    if not raw_config:
        # Try loading from YAML file
        try:
            import os

            import yaml

            config_path = os.environ.get(
                "SWING_TRADER_CONFIG",
                "/home/user/workspace/swing-trader/config/default.yaml",
            )
            with open(config_path) as f:
                raw_config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("config_load_failed", error=str(exc))
            raw_config = {}

    logger.info("config_requested", keys=list(raw_config.keys()))
    return _build_config_response(raw_config)
