"""
Async Redis client wrapper for the swing-trader system.

Provides a ``RedisManager`` class that wraps ``redis.asyncio`` with:
- Namespaced key management via a configurable prefix
- get / set / delete with optional TTL
- JSON serialisation / deserialisation using orjson
- publish / subscribe for event broadcasting
- A ``cache`` decorator for caching coroutine results
- Health check for monitoring endpoints

Usage
-----
    from src.core.redis_client import RedisManager

    redis_mgr = RedisManager()
    await redis_mgr.init()

    await redis_mgr.set("my_key", {"foo": "bar"}, ttl=60)
    value = await redis_mgr.get("my_key")

    await redis_mgr.close()
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, TypeVar

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

if TYPE_CHECKING:
    from src.core.config import Settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RedisManager:
    """Async Redis client wrapper with namespaced keys and JSON serialisation.

    Attributes
    ----------
    prefix:
        All Redis keys are stored as ``{prefix}{key}``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            from src.core.config import get_settings

            settings = get_settings()
        self._settings = settings
        self.prefix: str = settings.redis.prefix
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create the Redis connection pool and client.

        Call once at application startup.
        """
        cfg = self._settings.redis
        self._client = aioredis.from_url(
            cfg.url,
            max_connections=cfg.max_connections,
            socket_timeout=cfg.socket_timeout,
            socket_connect_timeout=cfg.socket_connect_timeout,
            decode_responses=False,  # We handle encoding ourselves via orjson
        )
        # Log only the host portion of the URL to avoid leaking credentials
        safe_url = cfg.url.split("@")[-1] if "@" in cfg.url else cfg.url
        logger.info("Redis client initialised.", extra={"url": safe_url})

    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._client is not None:
            await self._client.aclose()
            logger.info("Redis client closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> aioredis.Redis:
        """Return the underlying Redis client; raises if not initialised."""
        if self._client is None:
            raise RuntimeError("RedisManager.init() must be called before use.")
        return self._client

    def _full_key(self, key: str) -> str:
        """Return the namespaced Redis key."""
        return f"{self.prefix}{key}"

    @staticmethod
    def _serialise(value: Any) -> bytes:
        return orjson.dumps(value)

    @staticmethod
    def _deserialise(raw: bytes | None) -> Any:
        if raw is None:
            return None
        return orjson.loads(raw)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any:
        """Retrieve a value from Redis, deserialising from JSON.

        Returns ``None`` if the key does not exist.
        """
        raw = await self.client.get(self._full_key(key))
        return self._deserialise(raw)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a JSON-serialisable *value* at *key*.

        Parameters
        ----------
        key:
            Cache key (without prefix).
        value:
            Any orjson-serialisable Python object.
        ttl:
            Optional time-to-live in seconds.  If omitted the key persists
            indefinitely.
        """
        full_key = self._full_key(key)
        serialised = self._serialise(value)
        if ttl is not None:
            await self.client.setex(full_key, ttl, serialised)
        else:
            await self.client.set(full_key, serialised)

    async def delete(self, key: str) -> int:
        """Delete a key.  Returns the number of keys removed (0 or 1)."""
        return await self.client.delete(self._full_key(key))  # type: ignore[return-value]

    async def delete_many(self, *keys: str) -> int:
        """Delete multiple keys atomically.  Returns the number removed."""
        full_keys = [self._full_key(k) for k in keys]
        return await self.client.delete(*full_keys)  # type: ignore[return-value]

    async def exists(self, key: str) -> bool:
        """Return True if the key exists in Redis."""
        return bool(await self.client.exists(self._full_key(key)))

    async def expire(self, key: str, ttl: int) -> bool:
        """Set or update the TTL of an existing key."""
        return bool(await self.client.expire(self._full_key(key), ttl))

    async def ttl(self, key: str) -> int:
        """Return the remaining TTL in seconds, or -1/-2 if not set / absent."""
        return await self.client.ttl(self._full_key(key))  # type: ignore[return-value]

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter and return the new value."""
        return await self.client.incr(self._full_key(key), amount)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Pub/sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: Any) -> int:
        """Publish a JSON-serialised *message* to *channel*.

        Returns the number of subscribers that received the message.
        """
        payload = self._serialise(message)
        full_channel = self._full_key(channel)
        return await self.client.publish(full_channel, payload)  # type: ignore[return-value]

    def pubsub(self) -> PubSub:
        """Return a raw ``PubSub`` object for manual subscription management."""
        return self.client.pubsub()

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """Async generator that yields deserialised messages from *channel*.

        The generator runs indefinitely until cancelled or the connection
        is closed.  It skips the initial ``subscribe`` confirmation message.

        Example
        -------
            async for msg in redis_mgr.subscribe("scan_completed"):
                handle(msg)
        """
        ps = self.pubsub()
        await ps.subscribe(self._full_key(channel))
        try:
            async for raw_msg in ps.listen():
                if raw_msg["type"] == "message":
                    yield self._deserialise(raw_msg["data"])
        finally:
            await ps.unsubscribe()
            await ps.aclose()

    # ------------------------------------------------------------------
    # Cache decorator
    # ------------------------------------------------------------------

    def cache(self, key_prefix: str, ttl: int = 300) -> Callable[[F], F]:
        """Decorator that caches the result of an async function in Redis.

        The cache key is composed of ``key_prefix`` and the positional /
        keyword arguments of the decorated function.

        Parameters
        ----------
        key_prefix:
            Stable string identifier for this function's cache namespace.
        ttl:
            Cache TTL in seconds.

        Example
        -------
            @redis_mgr.cache("market_data:daily", ttl=300)
            async def fetch_ohlcv(ticker: str, date: str) -> dict: ...
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                args_key = ":".join(str(a) for a in args)
                kwargs_key = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{args_key}:{kwargs_key}"

                cached = await self.get(cache_key)
                if cached is not None:
                    return cached

                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl=ttl)
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, object]:
        """Ping Redis and return a health-status dict.

        Returns
        -------
        dict
            ``{"status": "ok", "latency_ms": float}`` or
            ``{"status": "error", "detail": str}``.
        """
        import time

        if self._client is None:
            return {"status": "error", "detail": "Client not initialised."}

        start = time.monotonic()
        try:
            await self.client.ping()
            latency_ms = (time.monotonic() - start) * 1000
            return {"status": "ok", "latency_ms": round(latency_ms, 2)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: RedisManager | None = None


def get_redis_manager() -> RedisManager:
    """Return the module-level ``RedisManager`` singleton."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = RedisManager()
    return _manager
