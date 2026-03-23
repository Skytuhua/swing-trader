"""Caching utilities for performance optimization.

Provides:
- ``async_lru_cache``: LRU cache decorator for async functions.
- ``redis_indicator_cache``: Cache computed indicators in Redis.
- ``indicator_cache_key``: Generate consistent cache keys for indicators.

Usage::

    @async_lru_cache(maxsize=256, ttl_seconds=300)
    async def get_indicator(ticker: str, period: int) -> dict:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import time
from collections import OrderedDict
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def async_lru_cache(
    maxsize: int = 128,
    ttl_seconds: int = 300,
) -> Callable[[F], F]:
    """LRU cache decorator for async functions with TTL expiry.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached results.
    ttl_seconds : int
        Time-to-live for each cache entry in seconds.
        Set to 0 for no expiry.
    """

    def decorator(func: F) -> F:
        cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        lock = asyncio.Lock()

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build cache key from args
            key = _make_key(args, kwargs)

            async with lock:
                if key in cache:
                    ts, val = cache[key]
                    if ttl_seconds <= 0 or (time.monotonic() - ts) < ttl_seconds:
                        cache.move_to_end(key)
                        return val
                    else:
                        del cache[key]

            # Cache miss — compute
            result = await func(*args, **kwargs)

            async with lock:
                cache[key] = (time.monotonic(), result)
                # Evict oldest if over capacity
                while len(cache) > maxsize:
                    cache.popitem(last=False)

            return result

        def cache_clear() -> None:
            cache.clear()

        def cache_info() -> dict[str, int]:
            return {"size": len(cache), "maxsize": maxsize, "ttl": ttl_seconds}

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def indicator_cache_key(ticker: str, indicator_name: str, params: dict[str, Any]) -> str:
    """Generate a consistent, collision-resistant cache key for an indicator.

    Parameters
    ----------
    ticker : str
        Symbol.
    indicator_name : str
        Name of the indicator (e.g. "rsi", "macd").
    params : dict
        Indicator parameters (e.g. {"period": 14}).

    Returns
    -------
    str
        Cache key string.
    """
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    raw = f"ind:{ticker}:{indicator_name}:{param_str}"
    return raw


async def redis_indicator_cache(
    redis_mgr,
    ticker: str,
    indicator_name: str,
    params: dict[str, Any],
    compute_fn: Callable,
    ttl: int = 300,
) -> Any:
    """Cache an indicator result in Redis.

    Parameters
    ----------
    redis_mgr : RedisManager
        Redis manager instance.
    ticker : str
        Symbol.
    indicator_name : str
        Indicator name.
    params : dict
        Indicator parameters.
    compute_fn : callable
        Async or sync function that computes the indicator value.
    ttl : int
        Cache TTL in seconds.

    Returns
    -------
    Any
        The indicator value (from cache or freshly computed).
    """
    key = indicator_cache_key(ticker, indicator_name, params)

    # Try cache first
    cached = await redis_mgr.get(key)
    if cached is not None:
        return cached

    # Compute
    if asyncio.iscoroutinefunction(compute_fn):
        result = await compute_fn()
    else:
        result = compute_fn()

    # Store
    await redis_mgr.set(key, result, ttl=ttl)
    return result


def _make_key(args: tuple, kwargs: dict) -> str:
    """Build a hashable string key from function arguments."""
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    raw = ":".join(parts)
    if len(raw) > 200:
        return hashlib.md5(raw.encode()).hexdigest()
    return raw
