"""
Circuit Breaker & Risk Controls — defensive safety mechanisms.

"These defensive mechanisms will never generate a single dollar of profit,
but they prevent a single bad trade from becoming a catastrophic loss."

This module implements six independent circuit breakers:

1. Daily Loss Circuit Breaker
   - Monitors net P&L every check cycle.
   - If daily loss exceeds threshold (default 3%), closes all positions and halts trading.
   - Requires manual restart or auto-resumes next trading day.

2. Consecutive Loss Circuit Breaker
   - Tracks sequential losing trades.
   - After N consecutive losses (default 3), pauses trading for a cooldown period (60 min).
   - Counter resets after cooldown expires.

3. Flash Crash Protection
   - Compares current price to a 30-second rolling average.
   - If deviation exceeds 4%, suspends order entry for 2 minutes.
   - Prevents buying into liquidation cascades.

4. Maximum Drawdown Guardrail
   - Tracks equity curve and peak equity.
   - If drawdown from peak exceeds threshold (default 7%), force-closes all positions.
   - Independent of individual stop losses — catches gap scenarios.

5. Position Limits
   - Hard maximum: 5% of account per trade.
   - Maximum sector exposure: 20% of portfolio.
   - Rejects orders that would exceed limits.

6. Fat-Finger Protection
   - Verifies limit price is within 0.5% of current mid before submitting.
   - Catches accidental order entry errors.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class BreakerState(str, Enum):
    """State of a circuit breaker."""

    CLOSED = "closed"  # Normal operation (circuit is "closed" = current flowing)
    OPEN = "open"  # Tripped — trading halted
    COOLDOWN = "cooldown"  # Recovering from trip, temporary pause


class BreakerType(str, Enum):
    """Identifies which breaker tripped."""

    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSS = "consecutive_loss"
    FLASH_CRASH = "flash_crash"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_LIMIT = "position_limit"
    FAT_FINGER = "fat_finger"


@dataclass
class BreakerTrip:
    """Record of a circuit breaker trip event."""

    breaker_type: BreakerType
    triggered_at: datetime
    reason: str
    current_value: float = 0.0
    threshold: float = 0.0
    auto_reset_at: Optional[datetime] = None


@dataclass
class CircuitBreakerStatus:
    """Overall circuit breaker system status."""

    trading_allowed: bool = True
    active_trips: list[BreakerTrip] = field(default_factory=list)
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    flash_crash_suspended: bool = False

    @property
    def is_halted(self) -> bool:
        return not self.trading_allowed


class CircuitBreaker:
    """Centralized circuit breaker system managing all safety mechanisms.

    Parameters
    ----------
    config : dict or dataclass
        Configuration with:
        - max_daily_loss_pct: float (default 3.0)
        - consecutive_loss_limit: int (default 3)
        - cooldown_minutes: float (default 60)
        - flash_crash_deviation_pct: float (default 4.0)
        - flash_crash_suspend_seconds: float (default 120)
        - max_drawdown_pct: float (default 7.0)
        - max_position_pct: float (default 5.0)
        - max_sector_exposure_pct: float (default 20.0)
        - fat_finger_max_deviation_pct: float (default 0.5)
    """

    def __init__(self, config: Any = None) -> None:
        cfg = config or {}
        self._get = (
            (lambda k, d: cfg.get(k, d))
            if isinstance(cfg, dict)
            else (lambda k, d: getattr(cfg, k, d))
        )

        # Daily loss
        self.max_daily_loss_pct: float = float(self._get("max_daily_loss_pct", 3.0))

        # Consecutive loss
        self.consecutive_loss_limit: int = int(self._get("consecutive_loss_limit", 3))
        self.cooldown_minutes: float = float(self._get("cooldown_minutes", 60.0))

        # Flash crash
        self.flash_crash_deviation_pct: float = float(self._get("flash_crash_deviation_pct", 4.0))
        self.flash_crash_suspend_seconds: float = float(self._get("flash_crash_suspend_seconds", 120.0))

        # Maximum drawdown
        self.max_drawdown_pct: float = float(self._get("max_drawdown_pct", 7.0))

        # Position limits
        self.max_position_pct: float = float(self._get("max_position_pct", 5.0))
        self.max_sector_exposure_pct: float = float(self._get("max_sector_exposure_pct", 20.0))

        # Fat-finger
        self.fat_finger_max_deviation_pct: float = float(self._get("fat_finger_max_deviation_pct", 0.5))

        # Internal state
        self._consecutive_losses: int = 0
        self._daily_pnl_pct: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._active_trips: list[BreakerTrip] = []
        self._trading_halted: bool = False
        self._cooldown_until: float = 0.0  # timestamp
        self._flash_suspend_until: float = 0.0  # timestamp

        # Rolling price buffer for flash crash detection (deque of (timestamp, price))
        self._price_buffer: deque[tuple[float, float]] = deque(maxlen=300)

        # Sector exposure tracking: sector -> total allocation pct
        self._sector_exposure: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Daily Loss Circuit Breaker
    # ------------------------------------------------------------------

    def check_daily_loss(self, daily_pnl_pct: float) -> BreakerTrip | None:
        """Check if daily P&L loss exceeds the threshold.

        Parameters
        ----------
        daily_pnl_pct : float
            Today's net P&L as a percentage (negative = loss).

        Returns
        -------
        BreakerTrip if breaker tripped, None if OK.
        """
        self._daily_pnl_pct = daily_pnl_pct

        if daily_pnl_pct < -self.max_daily_loss_pct:
            trip = BreakerTrip(
                breaker_type=BreakerType.DAILY_LOSS,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"Daily loss {abs(daily_pnl_pct):.2f}% exceeds max {self.max_daily_loss_pct:.2f}%. "
                    f"Halting all trading."
                ),
                current_value=daily_pnl_pct,
                threshold=-self.max_daily_loss_pct,
            )
            self._trip(trip)
            return trip
        return None

    # ------------------------------------------------------------------
    # 2. Consecutive Loss Circuit Breaker
    # ------------------------------------------------------------------

    def record_trade_result(self, is_winner: bool) -> BreakerTrip | None:
        """Record a trade result and check for consecutive loss breaker.

        Parameters
        ----------
        is_winner : bool
            True if trade was profitable, False if a loss.

        Returns
        -------
        BreakerTrip if consecutive loss limit hit, None otherwise.
        """
        if is_winner:
            self._consecutive_losses = 0
            return None

        self._consecutive_losses += 1
        logger.info(
            "circuit_breaker_loss_recorded",
            consecutive_losses=self._consecutive_losses,
            limit=self.consecutive_loss_limit,
        )

        if self._consecutive_losses >= self.consecutive_loss_limit:
            cooldown_until = time.time() + self.cooldown_minutes * 60
            self._cooldown_until = cooldown_until

            trip = BreakerTrip(
                breaker_type=BreakerType.CONSECUTIVE_LOSS,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"{self._consecutive_losses} consecutive losses. "
                    f"Cooling down for {self.cooldown_minutes:.0f} minutes."
                ),
                current_value=float(self._consecutive_losses),
                threshold=float(self.consecutive_loss_limit),
                auto_reset_at=datetime.fromtimestamp(cooldown_until, tz=timezone.utc),
            )
            self._active_trips.append(trip)
            logger.warning(
                "circuit_breaker_consecutive_loss",
                losses=self._consecutive_losses,
                cooldown_minutes=self.cooldown_minutes,
            )
            return trip
        return None

    def is_in_cooldown(self) -> bool:
        """Check if the consecutive loss cooldown is still active."""
        if self._cooldown_until <= 0:
            return False
        if time.time() >= self._cooldown_until:
            # Cooldown expired — reset
            self._cooldown_until = 0
            self._consecutive_losses = 0
            logger.info("circuit_breaker_cooldown_expired")
            return False
        return True

    # ------------------------------------------------------------------
    # 3. Flash Crash Protection
    # ------------------------------------------------------------------

    def check_flash_crash(self, symbol: str, current_price: float, timestamp: float | None = None) -> BreakerTrip | None:
        """Check if price deviates too far from 30-second rolling average.

        Call this with every price tick or quote update.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        current_price : float
            Current market price.
        timestamp : float, optional
            Unix timestamp (defaults to now).

        Returns
        -------
        BreakerTrip if flash crash detected, None if OK.
        """
        now = timestamp or time.time()
        self._price_buffer.append((now, current_price))

        # Clean old entries (keep last 30 seconds)
        cutoff = now - 30.0
        while self._price_buffer and self._price_buffer[0][0] < cutoff:
            self._price_buffer.popleft()

        if len(self._price_buffer) < 2:
            return None

        prices = [p for _, p in self._price_buffer]
        rolling_avg = sum(prices) / len(prices)

        if rolling_avg <= 0:
            return None

        deviation_pct = abs(current_price - rolling_avg) / rolling_avg * 100.0

        if deviation_pct > self.flash_crash_deviation_pct:
            self._flash_suspend_until = now + self.flash_crash_suspend_seconds

            trip = BreakerTrip(
                breaker_type=BreakerType.FLASH_CRASH,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"Flash crash detected for {symbol}: price {current_price:.2f} "
                    f"deviated {deviation_pct:.2f}% from 30s avg {rolling_avg:.2f}. "
                    f"Suspending orders for {self.flash_crash_suspend_seconds:.0f}s."
                ),
                current_value=deviation_pct,
                threshold=self.flash_crash_deviation_pct,
                auto_reset_at=datetime.fromtimestamp(
                    self._flash_suspend_until, tz=timezone.utc
                ),
            )
            self._active_trips.append(trip)
            logger.warning(
                "circuit_breaker_flash_crash",
                symbol=symbol,
                price=current_price,
                rolling_avg=round(rolling_avg, 2),
                deviation_pct=round(deviation_pct, 2),
            )
            return trip
        return None

    def is_flash_suspended(self) -> bool:
        """Check if orders are currently suspended due to flash crash detection."""
        if self._flash_suspend_until <= 0:
            return False
        if time.time() >= self._flash_suspend_until:
            self._flash_suspend_until = 0
            return False
        return True

    # ------------------------------------------------------------------
    # 4. Maximum Drawdown Guardrail
    # ------------------------------------------------------------------

    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking for drawdown calculation.

        Call this on every portfolio valuation update.
        """
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def check_max_drawdown(self, current_equity: float | None = None) -> BreakerTrip | None:
        """Check if portfolio drawdown from peak exceeds threshold.

        Parameters
        ----------
        current_equity : float, optional
            Current portfolio equity. If None, uses last updated value.

        Returns
        -------
        BreakerTrip if drawdown breached, None if OK.
        """
        if current_equity is not None:
            self.update_equity(current_equity)

        equity = self._current_equity
        peak = self._peak_equity

        if peak <= 0:
            return None

        drawdown_pct = (peak - equity) / peak * 100.0

        if drawdown_pct > self.max_drawdown_pct:
            trip = BreakerTrip(
                breaker_type=BreakerType.MAX_DRAWDOWN,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"Max drawdown breached: {drawdown_pct:.2f}% from peak "
                    f"${peak:,.2f} to ${equity:,.2f} (limit {self.max_drawdown_pct:.1f}%). "
                    f"Force closing all positions."
                ),
                current_value=drawdown_pct,
                threshold=self.max_drawdown_pct,
            )
            self._trip(trip)
            return trip
        return None

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak equity as a percentage."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity * 100.0)

    # ------------------------------------------------------------------
    # 5. Position Limits
    # ------------------------------------------------------------------

    def check_position_limit(
        self,
        proposed_allocation_pct: float,
        sector: str | None = None,
    ) -> BreakerTrip | None:
        """Check if a proposed trade violates position or sector limits.

        Parameters
        ----------
        proposed_allocation_pct : float
            Proposed trade size as % of portfolio.
        sector : str, optional
            Sector of the proposed trade for sector exposure checks.

        Returns
        -------
        BreakerTrip if limit violated, None if OK.
        """
        # Single position limit
        if proposed_allocation_pct > self.max_position_pct:
            trip = BreakerTrip(
                breaker_type=BreakerType.POSITION_LIMIT,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"Position size {proposed_allocation_pct:.2f}% exceeds max "
                    f"{self.max_position_pct:.1f}% per trade."
                ),
                current_value=proposed_allocation_pct,
                threshold=self.max_position_pct,
            )
            logger.warning("circuit_breaker_position_limit", **{"pct": proposed_allocation_pct})
            return trip

        # Sector exposure limit
        if sector:
            current_sector = self._sector_exposure.get(sector, 0.0)
            projected = current_sector + proposed_allocation_pct
            if projected > self.max_sector_exposure_pct:
                trip = BreakerTrip(
                    breaker_type=BreakerType.POSITION_LIMIT,
                    triggered_at=datetime.now(tz=timezone.utc),
                    reason=(
                        f"Sector '{sector}' exposure would be {projected:.2f}% "
                        f"(max {self.max_sector_exposure_pct:.1f}%)."
                    ),
                    current_value=projected,
                    threshold=self.max_sector_exposure_pct,
                )
                logger.warning("circuit_breaker_sector_limit", sector=sector, projected=projected)
                return trip

        return None

    def update_sector_exposure(self, sector: str, allocation_pct: float) -> None:
        """Update sector exposure after a trade."""
        self._sector_exposure[sector] = self._sector_exposure.get(sector, 0.0) + allocation_pct

    def reduce_sector_exposure(self, sector: str, allocation_pct: float) -> None:
        """Reduce sector exposure after closing a position."""
        current = self._sector_exposure.get(sector, 0.0)
        self._sector_exposure[sector] = max(0.0, current - allocation_pct)

    # ------------------------------------------------------------------
    # 6. Fat-Finger Protection
    # ------------------------------------------------------------------

    def check_fat_finger(
        self,
        limit_price: float,
        current_mid: float,
    ) -> BreakerTrip | None:
        """Verify limit price is within acceptable deviation of current mid.

        Parameters
        ----------
        limit_price : float
            Proposed limit order price.
        current_mid : float
            Current mid-market price (bid+ask)/2.

        Returns
        -------
        BreakerTrip if fat-finger detected, None if OK.
        """
        if current_mid <= 0:
            return None

        deviation_pct = abs(limit_price - current_mid) / current_mid * 100.0

        if deviation_pct > self.fat_finger_max_deviation_pct:
            trip = BreakerTrip(
                breaker_type=BreakerType.FAT_FINGER,
                triggered_at=datetime.now(tz=timezone.utc),
                reason=(
                    f"Fat-finger protection: limit {limit_price:.2f} is "
                    f"{deviation_pct:.2f}% from mid {current_mid:.2f} "
                    f"(max {self.fat_finger_max_deviation_pct:.1f}%)."
                ),
                current_value=deviation_pct,
                threshold=self.fat_finger_max_deviation_pct,
            )
            logger.warning(
                "circuit_breaker_fat_finger",
                limit_price=limit_price,
                current_mid=current_mid,
                deviation_pct=round(deviation_pct, 2),
            )
            return trip
        return None

    # ------------------------------------------------------------------
    # Composite pre-trade check
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        daily_pnl_pct: float = 0.0,
        current_equity: float = 0.0,
        proposed_allocation_pct: float = 0.0,
        limit_price: float = 0.0,
        current_mid: float = 0.0,
        sector: str | None = None,
    ) -> tuple[bool, str]:
        """Run all circuit breaker checks before a trade.

        Returns
        -------
        (allowed, reason) : tuple[bool, str]
            True if trade is allowed, False with reason if blocked.
        """
        # Check halted state
        if self._trading_halted:
            return False, "Trading is halted by circuit breaker."

        # Check cooldown
        if self.is_in_cooldown():
            remaining = (self._cooldown_until - time.time()) / 60
            return False, f"In consecutive-loss cooldown ({remaining:.1f} min remaining)."

        # Check flash crash suspension
        if self.is_flash_suspended():
            remaining = self._flash_suspend_until - time.time()
            return False, f"Flash crash suspension active ({remaining:.0f}s remaining)."

        # Daily loss
        trip = self.check_daily_loss(daily_pnl_pct)
        if trip:
            return False, trip.reason

        # Max drawdown
        if current_equity > 0:
            trip = self.check_max_drawdown(current_equity)
            if trip:
                return False, trip.reason

        # Position limits
        if proposed_allocation_pct > 0:
            trip = self.check_position_limit(proposed_allocation_pct, sector)
            if trip:
                return False, trip.reason

        # Fat-finger
        if limit_price > 0 and current_mid > 0:
            trip = self.check_fat_finger(limit_price, current_mid)
            if trip:
                return False, trip.reason

        return True, "All circuit breaker checks passed."

    # ------------------------------------------------------------------
    # Status & control
    # ------------------------------------------------------------------

    def get_status(self) -> CircuitBreakerStatus:
        """Get the current status of all circuit breakers."""
        return CircuitBreakerStatus(
            trading_allowed=not self._trading_halted and not self.is_in_cooldown() and not self.is_flash_suspended(),
            active_trips=list(self._active_trips),
            daily_pnl_pct=self._daily_pnl_pct,
            consecutive_losses=self._consecutive_losses,
            peak_equity=self._peak_equity,
            current_drawdown_pct=self.current_drawdown_pct,
            flash_crash_suspended=self.is_flash_suspended(),
        )

    def reset(self) -> None:
        """Reset all circuit breakers to normal state.

        Typically called at the start of a new trading day or manually by operator.
        """
        self._trading_halted = False
        self._consecutive_losses = 0
        self._cooldown_until = 0
        self._flash_suspend_until = 0
        self._daily_pnl_pct = 0.0
        self._active_trips.clear()
        self._price_buffer.clear()
        logger.info("circuit_breaker_reset")

    def reset_daily(self) -> None:
        """Reset daily-scoped breakers (called at start of each trading day)."""
        self._daily_pnl_pct = 0.0
        self._trading_halted = False
        # Keep consecutive losses and drawdown tracking across days
        self._active_trips = [
            t for t in self._active_trips if t.breaker_type != BreakerType.DAILY_LOSS
        ]
        logger.info("circuit_breaker_daily_reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trip(self, trip: BreakerTrip) -> None:
        """Record a trip and halt trading."""
        self._trading_halted = True
        self._active_trips.append(trip)
        logger.critical(
            "circuit_breaker_tripped",
            breaker=trip.breaker_type.value,
            reason=trip.reason,
        )
