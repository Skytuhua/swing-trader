"""Performance optimization utilities.

Provides:
- Async helpers (parallel fetch, semaphore-limited gather)
- LRU caching decorators for sync and async functions
- Redis-backed indicator caching
- Rolling window data management
- Database batch insert helpers
- Incremental indicator update support
"""

from src.performance.async_helpers import gather_with_semaphore, parallel_fetch
from src.performance.caching import (
    async_lru_cache,
    indicator_cache_key,
    redis_indicator_cache,
)
from src.performance.rolling_window import RollingWindow
from src.performance.db_helpers import batch_insert
from src.performance.incremental import IncrementalIndicatorCache

__all__ = [
    "gather_with_semaphore",
    "parallel_fetch",
    "async_lru_cache",
    "indicator_cache_key",
    "redis_indicator_cache",
    "RollingWindow",
    "batch_insert",
    "IncrementalIndicatorCache",
]
