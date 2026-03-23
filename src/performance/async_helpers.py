"""Async optimization helpers.

Provides utilities for efficient concurrent I/O:
- ``gather_with_semaphore``: Run coroutines with bounded concurrency.
- ``parallel_fetch``: Fetch data for multiple symbols in parallel.

Usage::

    results = await gather_with_semaphore(
        [fetch_data(sym) for sym in symbols],
        max_concurrent=10,
    )
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


async def gather_with_semaphore(
    coroutines: list[Coroutine[Any, Any, T]],
    max_concurrent: int = 10,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """Run coroutines concurrently with a semaphore limit.

    Parameters
    ----------
    coroutines : list
        Coroutines to run.
    max_concurrent : int
        Maximum number of concurrent tasks.
    return_exceptions : bool
        If True, exceptions are returned in the results list
        instead of being raised.

    Returns
    -------
    list
        Results in the same order as the input coroutines.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(coro: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await coro

    return await asyncio.gather(
        *[_limited(c) for c in coroutines],
        return_exceptions=return_exceptions,
    )


async def parallel_fetch(
    fetch_fn,
    items: list[Any],
    max_concurrent: int = 10,
) -> dict[Any, Any]:
    """Fetch data for multiple items in parallel.

    Parameters
    ----------
    fetch_fn : callable
        Async function that takes a single item and returns data.
        Signature: ``async def fetch_fn(item) -> (item, result)``
    items : list
        Items to fetch data for.
    max_concurrent : int
        Maximum concurrent fetches.

    Returns
    -------
    dict
        Mapping of item → result (excludes failed fetches).
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(item: Any) -> tuple[Any, Any]:
        async with sem:
            try:
                result = await fetch_fn(item)
                return item, result
            except Exception:
                return item, None

    results = await asyncio.gather(*[_fetch_one(i) for i in items])
    return {k: v for k, v in results if v is not None}
