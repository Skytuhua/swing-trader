"""Auto-register all technical indicators.

Importing this package (or calling ``register_all()``) is sufficient to
populate IndicatorRegistry with every indicator class defined in the
five sub-modules.

Call order is intentional: later modules can depend on earlier ones for
config validation but not for runtime computation.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def register_all() -> None:
    """Import all indicator modules, triggering @IndicatorRegistry.register decorators."""
    # Import order is alphabetical / logical; all modules self-register on import.
    from src.services.technical.indicators import trend       # noqa: F401
    from src.services.technical.indicators import momentum    # noqa: F401
    from src.services.technical.indicators import volatility  # noqa: F401
    from src.services.technical.indicators import volume      # noqa: F401
    from src.services.technical.indicators import structure   # noqa: F401

    from src.services.technical.registry import IndicatorRegistry
    logger.info(
        "indicator_registry.registered",
        count=len(IndicatorRegistry.list_all()),
        indicators=IndicatorRegistry.list_all(),
    )


# Auto-register when this package is imported
register_all()
