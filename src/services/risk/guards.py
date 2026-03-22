"""
Risk guards: independent, composable checks that each expose a check() method.

Guards:
  - MaxSlippageGuard       – reject if realised/expected slippage too high
  - MaxSpreadGuard         – reject if bid-ask spread too wide
  - MaxDrawdownGuard       – activate kill switch when max drawdown breached
  - DailyLossGuard         – halt trading for the day when daily loss exceeded
  - StaleDataGuard         – reject if market data is older than threshold
  - DuplicateSignalGuard   – prevent re-entering the same ticker within cooldown
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from src.core.exceptions import RiskLimitExceededError, StaleDataError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _GuardResult:
    """Lightweight result object."""

    def __init__(self, passed: bool, reason: str = "") -> None:
        self.passed = passed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return f"GuardResult(passed={self.passed}, reason={self.reason!r})"


class BaseGuard:
    """Abstract base for all risk guards."""

    name: str = "base_guard"

    def check(self, *args: Any, **kwargs: Any) -> _GuardResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. MaxSlippageGuard
# ---------------------------------------------------------------------------


class MaxSlippageGuard(BaseGuard):
    """Reject an order if expected slippage exceeds the configured threshold.

    Slippage is computed as:
        |fill_price - limit_price| / limit_price  × 100  (in %)

    Or for market orders, estimated from the current spread.
    """

    name = "max_slippage"

    def __init__(self, max_slippage_pct: float = 0.5) -> None:
        """
        Args:
            max_slippage_pct: Maximum acceptable slippage as a % of price.
        """
        self.max_slippage_pct = max_slippage_pct

    def check(
        self,
        expected_price: float,
        fill_price: float | None = None,
        spread_pct: float | None = None,
    ) -> _GuardResult:
        """Check whether slippage is within limits.

        Args:
            expected_price: The limit/mid price used when deciding to trade.
            fill_price:     Actual fill price (if known post-execution).
            spread_pct:     Current bid-ask spread % (used as a proxy pre-execution).

        Returns:
            _GuardResult with passed status and reason.
        """
        if fill_price is not None and expected_price > 0:
            slippage = abs(fill_price - expected_price) / expected_price * 100.0
        elif spread_pct is not None:
            # Pre-execution: use spread as worst-case slippage estimate
            slippage = spread_pct / 2.0
        else:
            return _GuardResult(True, "No slippage data; allowing.")

        if slippage > self.max_slippage_pct:
            msg = (
                f"Slippage {slippage:.3f}% exceeds max {self.max_slippage_pct:.3f}%."
            )
            logger.warning("guard_slippage_failed", slippage=slippage, max=self.max_slippage_pct)
            return _GuardResult(False, msg)

        return _GuardResult(True)


# ---------------------------------------------------------------------------
# 2. MaxSpreadGuard
# ---------------------------------------------------------------------------


class MaxSpreadGuard(BaseGuard):
    """Reject if the bid-ask spread is too wide (implicit transaction cost)."""

    name = "max_spread"

    def __init__(self, max_spread_pct: float = 0.5) -> None:
        self.max_spread_pct = max_spread_pct

    def check(self, spread_pct: float) -> _GuardResult:
        """
        Args:
            spread_pct: Bid-ask spread as a percentage of mid-price.
        """
        if spread_pct > self.max_spread_pct:
            msg = f"Spread {spread_pct:.3f}% exceeds max {self.max_spread_pct:.3f}%."
            logger.warning("guard_spread_failed", spread_pct=spread_pct, max=self.max_spread_pct)
            return _GuardResult(False, msg)
        return _GuardResult(True)


# ---------------------------------------------------------------------------
# 3. MaxDrawdownGuard
# ---------------------------------------------------------------------------


class MaxDrawdownGuard(BaseGuard):
    """Activate kill switch when portfolio drawdown exceeds threshold."""

    name = "max_drawdown"

    def __init__(
        self,
        max_drawdown_pct: float = 10.0,
        kill_switch: Any = None,       # KillSwitch – injected at runtime
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.kill_switch = kill_switch

    def check(self, current_drawdown_pct: float) -> _GuardResult:
        """
        Args:
            current_drawdown_pct: Current peak-to-trough drawdown as %.
        """
        if current_drawdown_pct > self.max_drawdown_pct:
            msg = (
                f"Drawdown {current_drawdown_pct:.2f}% exceeds max "
                f"{self.max_drawdown_pct:.2f}%."
            )
            logger.critical("guard_drawdown_breached", drawdown=current_drawdown_pct)
            # NOTE: actually activating the kill switch requires an async call;
            # the RiskEngine will call kill_switch.activate() on this failure.
            return _GuardResult(False, msg)
        return _GuardResult(True)


# ---------------------------------------------------------------------------
# 4. DailyLossGuard
# ---------------------------------------------------------------------------


class DailyLossGuard(BaseGuard):
    """Halt trading for the day when the daily loss limit is exceeded."""

    name = "daily_loss"

    def __init__(self, max_daily_loss_pct: float = 2.0) -> None:
        """
        Args:
            max_daily_loss_pct: Maximum allowed loss as % of portfolio for the day.
        """
        self.max_daily_loss_pct = max_daily_loss_pct

    def check(self, daily_pnl_pct: float) -> _GuardResult:
        """
        Args:
            daily_pnl_pct: Today's realised + unrealised P&L as % of portfolio.
                           Negative values indicate a loss.
        """
        if daily_pnl_pct < -self.max_daily_loss_pct:
            msg = (
                f"Daily loss {abs(daily_pnl_pct):.2f}% exceeds max "
                f"{self.max_daily_loss_pct:.2f}%. Halting for the day."
            )
            logger.warning(
                "guard_daily_loss_breached",
                daily_pnl_pct=daily_pnl_pct,
                max=self.max_daily_loss_pct,
            )
            return _GuardResult(False, msg)
        return _GuardResult(True)


# ---------------------------------------------------------------------------
# 5. StaleDataGuard
# ---------------------------------------------------------------------------


class StaleDataGuard(BaseGuard):
    """Reject a trade if market data is older than the configured threshold."""

    name = "stale_data"

    def __init__(self, max_age_minutes: float = 15.0) -> None:
        self.max_age = timedelta(minutes=max_age_minutes)

    def check(
        self,
        data_timestamp: datetime,
        reference: datetime | None = None,
    ) -> _GuardResult:
        """
        Args:
            data_timestamp: When the market data was fetched/published.
            reference:      Reference time (defaults to now UTC).
        """
        now = reference or datetime.now(tz=timezone.utc)

        # Make both timezone-aware for comparison
        if data_timestamp.tzinfo is None:
            data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        age = now - data_timestamp
        if age > self.max_age:
            msg = (
                f"Data is {age.total_seconds() / 60:.1f} minutes old "
                f"(max {self.max_age.total_seconds() / 60:.0f} min)."
            )
            logger.warning(
                "guard_stale_data",
                age_minutes=age.total_seconds() / 60,
                max_minutes=self.max_age.total_seconds() / 60,
            )
            return _GuardResult(False, msg)
        return _GuardResult(True)


# ---------------------------------------------------------------------------
# 6. DuplicateSignalGuard
# ---------------------------------------------------------------------------


class DuplicateSignalGuard(BaseGuard):
    """Prevent entering the same ticker within a cooldown period.

    Maintains an in-memory record of recently traded tickers.
    Reset on system restart (by design – each session is independent).
    """

    name = "duplicate_signal"

    def __init__(self, cooldown_minutes: float = 60.0) -> None:
        self.cooldown = timedelta(minutes=cooldown_minutes)
        # ticker → last_entry_time
        self._last_entries: dict[str, datetime] = {}

    def check(self, ticker: str) -> _GuardResult:
        """
        Args:
            ticker: The ticker to check for duplicate signal.
        """
        last = self._last_entries.get(ticker)
        if last is not None:
            age = datetime.now(tz=timezone.utc) - last
            if age < self.cooldown:
                remaining = (self.cooldown - age).total_seconds() / 60
                msg = (
                    f"Duplicate signal: {ticker} was entered {age.total_seconds() / 60:.0f} min ago. "
                    f"Cooldown: {remaining:.0f} min remaining."
                )
                logger.info(
                    "guard_duplicate_signal",
                    ticker=ticker,
                    last_entry=last.isoformat(),
                    remaining_min=round(remaining, 1),
                )
                return _GuardResult(False, msg)
        return _GuardResult(True)

    def record_entry(self, ticker: str) -> None:
        """Record a successful entry for the given ticker."""
        self._last_entries[ticker] = datetime.now(tz=timezone.utc)
        logger.debug("guard_duplicate_entry_recorded", ticker=ticker)

    def clear_ticker(self, ticker: str) -> None:
        """Remove cooldown for a ticker (e.g. after position is fully closed)."""
        self._last_entries.pop(ticker, None)

    def clear_all(self) -> None:
        """Reset all cooldown records."""
        self._last_entries.clear()

    @property
    def tracked_tickers(self) -> list[str]:
        return list(self._last_entries.keys())
