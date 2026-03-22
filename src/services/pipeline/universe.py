"""
Universe filter: identify liquid, tradable US equities meeting all criteria.

Filters applied:
  - Minimum price
  - Minimum market cap
  - Minimum average dollar volume
  - Maximum bid-ask spread %
  - Exclusion list (prohibited tickers)
  - Earnings blackout (within N days of earnings)
  - Halt detection
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import DataProviderError

if TYPE_CHECKING:
    from src.core.config import UniverseConfig
    from src.services.market_data.manager import DataManager

logger = structlog.get_logger(__name__)


class UniverseFilter:
    """Filter tradable universe of liquid US stocks.

    Applies hard filters in sequence; any failure skips the ticker.
    Designed to run before market open each trading day.
    """

    def __init__(self, config: "UniverseConfig", data_manager: "DataManager") -> None:
        self.config = config
        self.data = data_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_tradable_universe(self) -> list[str]:
        """Return filtered list of tickers meeting all criteria.

        Returns:
            Sorted list of ticker symbols that pass all filters.
        """
        all_tickers: list[str] = await self.data.get_universe_tickers()
        logger.info("universe_filter_start", total_candidates=len(all_tickers))

        results: list[str] = []
        failed_counts: dict[str, int] = {
            "market_cap": 0,
            "price": 0,
            "dollar_volume": 0,
            "spread": 0,
            "exclusion": 0,
            "earnings_blackout": 0,
            "halted": 0,
            "data_error": 0,
        }

        # Fetch all data concurrently in batches to respect rate limits
        batch_size = 50
        for batch_start in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_start : batch_start + batch_size]
            tasks = [self._evaluate_ticker(ticker) for ticker in batch]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            for ticker, outcome in zip(batch, outcomes):
                if isinstance(outcome, Exception):
                    logger.warning(
                        "universe_filter_ticker_error",
                        ticker=ticker,
                        error=str(outcome),
                    )
                    failed_counts["data_error"] += 1
                    continue

                passed, reason = outcome  # type: ignore[misc]
                if passed:
                    results.append(ticker)
                elif reason:
                    failed_counts[reason] = failed_counts.get(reason, 0) + 1

        logger.info(
            "universe_filter_complete",
            passed=len(results),
            total=len(all_tickers),
            failed_breakdown=failed_counts,
        )
        return sorted(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _evaluate_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Evaluate a single ticker against all filters.

        Returns:
            (passed, failure_reason) – failure_reason is None when passed.
        """
        # Exclusion list is the cheapest check – do it first
        if ticker in self.config.exclusion_list:
            return False, "exclusion"

        try:
            info = await self.data.get_company_info(ticker)
            quote = await self.data.get_quote(ticker)
        except DataProviderError as exc:
            logger.debug("universe_filter_data_error", ticker=ticker, error=str(exc))
            return False, "data_error"

        # ---- Fundamental filters ----
        market_cap: float = info.get("market_cap", 0.0) or 0.0
        if market_cap < self.config.min_market_cap:
            return False, "market_cap"

        if quote.price < self.config.min_price:
            return False, "price"

        avg_dollar_volume: float = info.get("avg_dollar_volume", 0.0) or 0.0
        if avg_dollar_volume < self.config.min_avg_dollar_volume:
            return False, "dollar_volume"

        # ---- Spread filter ----
        spread_pct: float = getattr(quote, "spread_pct", 0.0) or 0.0
        if spread_pct > self.config.max_spread_pct:
            return False, "spread"

        # ---- Event risk: earnings blackout ----
        if await self._has_upcoming_earnings(ticker, days=2):
            return False, "earnings_blackout"

        # ---- Trading halt ----
        if await self._is_halted(ticker):
            return False, "halted"

        return True, None

    async def _has_upcoming_earnings(self, ticker: str, days: int = 2) -> bool:
        """Return True if the ticker has earnings scheduled within *days* calendar days."""
        try:
            earnings_date: date | None = await self.data.get_next_earnings_date(ticker)
            if earnings_date is None:
                return False
            today = datetime.now(tz=timezone.utc).date()
            return (earnings_date - today).days <= days
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "universe_filter_earnings_check_failed",
                ticker=ticker,
                error=str(exc),
            )
            # When in doubt, treat as blackout to avoid earnings risk
            return True

    async def _is_halted(self, ticker: str) -> bool:
        """Return True if the ticker is currently halted or has no tradable quote."""
        try:
            quote = await self.data.get_quote(ticker)
            # A halted stock typically shows no bid/ask or a zero/NaN price
            if quote.price <= 0:
                return True
            bid = getattr(quote, "bid", None)
            ask = getattr(quote, "ask", None)
            if bid is not None and ask is not None:
                if bid <= 0 and ask <= 0:
                    return True
            return False
        except DataProviderError:
            # Cannot get a quote → treat as halted
            return True
        except Exception as exc:  # pragma: no cover
            logger.debug("universe_filter_halt_check_failed", ticker=ticker, error=str(exc))
            return True

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def refresh_exclusion_list(self, tickers: list[str]) -> None:
        """Dynamically update the exclusion list (e.g. from compliance feed)."""
        self.config.exclusion_list = list(set(self.config.exclusion_list) | set(tickers))
        logger.info(
            "exclusion_list_updated",
            total_excluded=len(self.config.exclusion_list),
        )
