"""
Market calendar: NYSE trading hours, holidays, early-close dates.

Hardcoded 2025-2026 US market holidays and early-close dates.
Provides is_market_open(), next_market_open(), is_trading_day().
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Time zone
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# NYSE regular session hours (Eastern Time)
# ---------------------------------------------------------------------------

_OPEN_TIME = time(9, 30, 0)     # 09:30 ET
_CLOSE_TIME = time(16, 0, 0)    # 16:00 ET
_EARLY_CLOSE_TIME = time(13, 0, 0)  # 13:00 ET on early-close days

# ---------------------------------------------------------------------------
# US Market Holidays 2025
# ---------------------------------------------------------------------------

_HOLIDAYS_2025: set[date] = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving Day
    date(2025, 12, 25), # Christmas Day
}

# ---------------------------------------------------------------------------
# US Market Holidays 2026
# ---------------------------------------------------------------------------

_HOLIDAYS_2026: set[date] = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed; July 4 is Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving Day
    date(2026, 12, 25), # Christmas Day
}

_ALL_HOLIDAYS: set[date] = _HOLIDAYS_2025 | _HOLIDAYS_2026

# ---------------------------------------------------------------------------
# Early close dates (markets close at 13:00 ET)
# 2025 early closes
# ---------------------------------------------------------------------------

_EARLY_CLOSE_2025: set[date] = {
    date(2025, 7, 3),   # Day before Independence Day
    date(2025, 11, 28), # Day after Thanksgiving (Black Friday)
    date(2025, 12, 24), # Christmas Eve
}

_EARLY_CLOSE_2026: set[date] = {
    date(2026, 7, 2),   # Day before Independence Day observed
    date(2026, 11, 27), # Day after Thanksgiving (Black Friday)
    date(2026, 12, 24), # Christmas Eve
}

_ALL_EARLY_CLOSE: set[date] = _EARLY_CLOSE_2025 | _EARLY_CLOSE_2026


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


class MarketCalendar:
    """NYSE trading calendar with holiday, early-close, and hours support.

    All methods that take a ``dt`` parameter default to "now" in Eastern Time
    when called without arguments.

    Usage::

        cal = MarketCalendar()
        if cal.is_market_open():
            ...
        next_open = cal.next_market_open()
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Return True if the NYSE is currently open for regular session trading.

        Args:
            dt: UTC datetime to check.  Defaults to now.
        """
        et_now = self._to_et(dt)
        today = et_now.date()

        if not self.is_trading_day(today):
            return False

        open_dt = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = self._close_time_for(today)
        close_dt = et_now.replace(
            hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0
        )

        return open_dt <= et_now < close_dt

    def is_trading_day(self, d: date | None = None) -> bool:
        """Return True if the given date is a regular NYSE trading day.

        Args:
            d: Date to check.  Defaults to today in ET.
        """
        if d is None:
            d = datetime.now(tz=_ET).date()

        # Weekend
        if d.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False

        # Holiday
        if d in _ALL_HOLIDAYS:
            return False

        return True

    def is_early_close(self, d: date | None = None) -> bool:
        """Return True if the given date is an early-close day."""
        if d is None:
            d = datetime.now(tz=_ET).date()
        return d in _ALL_EARLY_CLOSE

    def next_market_open(self, dt: datetime | None = None) -> datetime:
        """Return the datetime of the next regular session open (ET).

        If the market is currently open, returns today's open.
        If it's before today's open, returns today's open.
        Otherwise returns the next trading day's open.
        """
        et_now = self._to_et(dt)
        today = et_now.date()

        # Check if market opens later today
        if self.is_trading_day(today):
            today_open = datetime(
                today.year, today.month, today.day,
                _OPEN_TIME.hour, _OPEN_TIME.minute,
                tzinfo=_ET,
            )
            if et_now < today_open:
                return today_open

        # Find next trading day
        candidate = today + timedelta(days=1)
        for _ in range(14):  # Safety limit
            if self.is_trading_day(candidate):
                return datetime(
                    candidate.year, candidate.month, candidate.day,
                    _OPEN_TIME.hour, _OPEN_TIME.minute,
                    tzinfo=_ET,
                )
            candidate += timedelta(days=1)

        raise RuntimeError("Could not find next market open within 14 days.")

    def next_market_close(self, dt: datetime | None = None) -> datetime:
        """Return the datetime of the current or next session close (ET)."""
        et_now = self._to_et(dt)
        today = et_now.date()

        if self.is_trading_day(today):
            close_time = self._close_time_for(today)
            today_close = datetime(
                today.year, today.month, today.day,
                close_time.hour, close_time.minute,
                tzinfo=_ET,
            )
            if et_now < today_close:
                return today_close

        # Find next trading day's close
        candidate = today + timedelta(days=1)
        for _ in range(14):
            if self.is_trading_day(candidate):
                close_time = self._close_time_for(candidate)
                return datetime(
                    candidate.year, candidate.month, candidate.day,
                    close_time.hour, close_time.minute,
                    tzinfo=_ET,
                )
            candidate += timedelta(days=1)

        raise RuntimeError("Could not find next market close within 14 days.")

    def time_until_open(self, dt: datetime | None = None) -> timedelta:
        """Seconds until the next session open.  Zero if market is open."""
        et_now = self._to_et(dt)
        if self.is_market_open(dt):
            return timedelta(0)
        next_open = self.next_market_open(dt)
        return next_open - et_now

    def trading_days_between(self, start: date, end: date) -> int:
        """Count trading days between start (inclusive) and end (exclusive)."""
        count = 0
        current = start
        while current < end:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def previous_trading_day(self, d: date | None = None) -> date:
        """Return the most recent trading day on or before d."""
        if d is None:
            d = datetime.now(tz=_ET).date()
        candidate = d
        for _ in range(14):
            if self.is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise RuntimeError("Could not find previous trading day within 14 days.")

    def session_open(self, d: date | None = None) -> datetime:
        """Return the session open datetime for trading day d."""
        if d is None:
            d = datetime.now(tz=_ET).date()
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day.")
        return datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)

    def session_close(self, d: date | None = None) -> datetime:
        """Return the session close datetime for trading day d."""
        if d is None:
            d = datetime.now(tz=_ET).date()
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day.")
        ct = self._close_time_for(d)
        return datetime(d.year, d.month, d.day, ct.hour, ct.minute, tzinfo=_ET)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_et(dt: datetime | None) -> datetime:
        """Convert UTC datetime to Eastern Time, or return now in ET."""
        if dt is None:
            return datetime.now(tz=_ET)
        if dt.tzinfo is None:
            # Treat naive as UTC
            return dt.replace(tzinfo=timezone.utc).astimezone(_ET)
        return dt.astimezone(_ET)

    @staticmethod
    def _close_time_for(d: date) -> time:
        """Return the close time for the given date (13:00 on early close, else 16:00)."""
        if d in _ALL_EARLY_CLOSE:
            return _EARLY_CLOSE_TIME
        return _CLOSE_TIME

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def is_holiday(cls, d: date) -> bool:
        return d in _ALL_HOLIDAYS

    @classmethod
    def all_holidays(cls) -> set[date]:
        return frozenset(_ALL_HOLIDAYS)  # type: ignore[return-value]

    @classmethod
    def all_early_close_dates(cls) -> set[date]:
        return frozenset(_ALL_EARLY_CLOSE)  # type: ignore[return-value]
