"""Tests for Circuit Breaker & Risk Controls (src/risk/circuit_breaker.py)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.risk.circuit_breaker import (
    BreakerState,
    BreakerType,
    CircuitBreaker,
    CircuitBreakerStatus,
)


# ---------------------------------------------------------------------------
# Daily loss circuit breaker
# ---------------------------------------------------------------------------


class TestDailyLossBreaker:
    def test_within_limit(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        trip = cb.check_daily_loss(-2.5)
        assert trip is None

    def test_at_limit(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        trip = cb.check_daily_loss(-3.0)
        assert trip is None  # -3.0 is not less than -3.0

    def test_exceeds_limit(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        trip = cb.check_daily_loss(-3.5)
        assert trip is not None
        assert trip.breaker_type == BreakerType.DAILY_LOSS
        assert "3.50%" in trip.reason

    def test_trading_halted_after_trip(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        cb.check_daily_loss(-5.0)
        status = cb.get_status()
        assert status.is_halted is True

    def test_positive_pnl_ok(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        trip = cb.check_daily_loss(5.0)
        assert trip is None


# ---------------------------------------------------------------------------
# Consecutive loss circuit breaker
# ---------------------------------------------------------------------------


class TestConsecutiveLossBreaker:
    def test_single_loss_no_trip(self):
        cb = CircuitBreaker({"consecutive_loss_limit": 3})
        trip = cb.record_trade_result(is_winner=False)
        assert trip is None
        assert cb._consecutive_losses == 1

    def test_consecutive_losses_trip(self):
        cb = CircuitBreaker({"consecutive_loss_limit": 3, "cooldown_minutes": 60})
        cb.record_trade_result(False)
        cb.record_trade_result(False)
        trip = cb.record_trade_result(False)
        assert trip is not None
        assert trip.breaker_type == BreakerType.CONSECUTIVE_LOSS

    def test_win_resets_counter(self):
        cb = CircuitBreaker({"consecutive_loss_limit": 3})
        cb.record_trade_result(False)
        cb.record_trade_result(False)
        cb.record_trade_result(True)
        assert cb._consecutive_losses == 0

    def test_cooldown_active(self):
        cb = CircuitBreaker({"consecutive_loss_limit": 2, "cooldown_minutes": 60})
        cb.record_trade_result(False)
        cb.record_trade_result(False)
        assert cb.is_in_cooldown() is True

    def test_cooldown_expired(self):
        cb = CircuitBreaker({"consecutive_loss_limit": 2, "cooldown_minutes": 0.001})
        cb.record_trade_result(False)
        cb.record_trade_result(False)
        # Cooldown is 0.001 minutes = 0.06 seconds
        time.sleep(0.1)
        assert cb.is_in_cooldown() is False
        assert cb._consecutive_losses == 0  # Reset after cooldown


# ---------------------------------------------------------------------------
# Flash crash protection
# ---------------------------------------------------------------------------


class TestFlashCrashProtection:
    def test_normal_price_no_trip(self):
        cb = CircuitBreaker({"flash_crash_deviation_pct": 4.0})
        now = time.time()
        # Feed 10 normal prices
        for i in range(10):
            cb.check_flash_crash("AAPL", 150.0, now + i * 0.1)
        trip = cb.check_flash_crash("AAPL", 150.5, now + 1.0)
        assert trip is None

    def test_flash_crash_detected(self):
        cb = CircuitBreaker({
            "flash_crash_deviation_pct": 4.0,
            "flash_crash_suspend_seconds": 120,
        })
        now = time.time()
        # Build 30s average around 150
        for i in range(20):
            cb.check_flash_crash("AAPL", 150.0, now + i * 0.5)
        # Sudden 5% drop (>4% deviation)
        trip = cb.check_flash_crash("AAPL", 142.0, now + 10.0)
        assert trip is not None
        assert trip.breaker_type == BreakerType.FLASH_CRASH

    def test_flash_suspend_active(self):
        cb = CircuitBreaker({
            "flash_crash_deviation_pct": 4.0,
            "flash_crash_suspend_seconds": 120,
        })
        now = time.time()
        for i in range(20):
            cb.check_flash_crash("AAPL", 150.0, now + i * 0.5)
        cb.check_flash_crash("AAPL", 140.0, now + 10.0)
        assert cb.is_flash_suspended() is True


# ---------------------------------------------------------------------------
# Maximum drawdown guardrail
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_within_drawdown_limit(self):
        cb = CircuitBreaker({"max_drawdown_pct": 7.0})
        cb.update_equity(100_000)
        trip = cb.check_max_drawdown(96_000)
        assert trip is None

    def test_drawdown_breached(self):
        cb = CircuitBreaker({"max_drawdown_pct": 7.0})
        cb.update_equity(100_000)
        trip = cb.check_max_drawdown(92_000)  # 8% drawdown
        assert trip is not None
        assert trip.breaker_type == BreakerType.MAX_DRAWDOWN

    def test_peak_equity_tracking(self):
        cb = CircuitBreaker()
        cb.update_equity(100_000)
        cb.update_equity(105_000)
        cb.update_equity(103_000)
        assert cb._peak_equity == 105_000

    def test_current_drawdown_pct(self):
        cb = CircuitBreaker()
        cb.update_equity(100_000)
        cb.update_equity(95_000)
        assert abs(cb.current_drawdown_pct - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------


class TestPositionLimits:
    def test_within_limit(self):
        cb = CircuitBreaker({"max_position_pct": 5.0})
        trip = cb.check_position_limit(4.5)
        assert trip is None

    def test_exceeds_single_position_limit(self):
        cb = CircuitBreaker({"max_position_pct": 5.0})
        trip = cb.check_position_limit(6.0)
        assert trip is not None
        assert trip.breaker_type == BreakerType.POSITION_LIMIT

    def test_sector_exposure_within_limit(self):
        cb = CircuitBreaker({"max_sector_exposure_pct": 20.0})
        cb.update_sector_exposure("tech", 10.0)
        trip = cb.check_position_limit(5.0, sector="tech")
        assert trip is None

    def test_sector_exposure_exceeds_limit(self):
        cb = CircuitBreaker({"max_sector_exposure_pct": 20.0})
        cb.update_sector_exposure("tech", 18.0)
        trip = cb.check_position_limit(4.0, sector="tech")
        assert trip is not None

    def test_reduce_sector_exposure(self):
        cb = CircuitBreaker()
        cb.update_sector_exposure("tech", 15.0)
        cb.reduce_sector_exposure("tech", 5.0)
        assert cb._sector_exposure["tech"] == 10.0


# ---------------------------------------------------------------------------
# Fat-finger protection
# ---------------------------------------------------------------------------


class TestFatFingerProtection:
    def test_within_tolerance(self):
        cb = CircuitBreaker({"fat_finger_max_deviation_pct": 0.5})
        trip = cb.check_fat_finger(150.50, 150.00)
        assert trip is None

    def test_fat_finger_detected(self):
        cb = CircuitBreaker({"fat_finger_max_deviation_pct": 0.5})
        trip = cb.check_fat_finger(155.00, 150.00)  # 3.3% deviation
        assert trip is not None
        assert trip.breaker_type == BreakerType.FAT_FINGER

    def test_zero_mid_price(self):
        cb = CircuitBreaker()
        trip = cb.check_fat_finger(150.0, 0.0)
        assert trip is None


# ---------------------------------------------------------------------------
# Composite pre-trade check
# ---------------------------------------------------------------------------


class TestPreTradeCheck:
    def test_all_checks_pass(self):
        cb = CircuitBreaker({
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 7.0,
            "max_position_pct": 5.0,
            "fat_finger_max_deviation_pct": 0.5,
        })
        cb.update_equity(100_000)
        allowed, reason = cb.pre_trade_check(
            daily_pnl_pct=-1.0,
            current_equity=98_000,
            proposed_allocation_pct=4.0,
            limit_price=150.25,
            current_mid=150.00,
        )
        assert allowed is True

    def test_blocked_by_daily_loss(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 3.0})
        cb.update_equity(100_000)
        allowed, reason = cb.pre_trade_check(daily_pnl_pct=-4.0)
        assert allowed is False
        assert "Daily loss" in reason

    def test_blocked_by_halted_state(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 1.0})
        cb.check_daily_loss(-2.0)  # Trip the breaker
        allowed, reason = cb.pre_trade_check(daily_pnl_pct=0.0)
        assert allowed is False
        assert "halted" in reason


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_full_reset(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 1.0, "consecutive_loss_limit": 1})
        cb.check_daily_loss(-2.0)
        cb.record_trade_result(False)
        cb.reset()
        assert cb._trading_halted is False
        assert cb._consecutive_losses == 0
        assert len(cb._active_trips) == 0

    def test_daily_reset(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 1.0})
        cb.check_daily_loss(-2.0)
        cb.reset_daily()
        assert cb._trading_halted is False
        assert cb._daily_pnl_pct == 0.0


class TestCircuitBreakerStatus:
    def test_status_normal(self):
        cb = CircuitBreaker()
        cb.update_equity(100_000)
        status = cb.get_status()
        assert status.trading_allowed is True
        assert status.consecutive_losses == 0

    def test_status_after_trip(self):
        cb = CircuitBreaker({"max_daily_loss_pct": 1.0})
        cb.check_daily_loss(-2.0)
        status = cb.get_status()
        assert status.is_halted is True
        assert len(status.active_trips) > 0
