"""
API route handlers package.

Each module exposes a FastAPI ``APIRouter`` that is registered in
``src.api.app.create_app``.
"""

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

__all__ = [
    "health",
    "decisions",
    "positions",
    "orders",
    "trades",
    "risk",
    "config_api",
    "kill_switch",
    "reports",
]
