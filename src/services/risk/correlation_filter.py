"""
Correlation-adjusted position sizing to prevent concentrated sector bets.

Checks rolling Pearson correlation of a candidate ticker against all open
positions over a configurable lookback window.  Depending on the highest
pairwise correlation found:

  - corr > block_threshold  →  block trade entirely
  - corr > max_correlation  →  reduce position size proportionally
  - corr ≤ max_correlation  →  full size allowed
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from src.services.market_data.manager import DataManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CorrelationCheckResult:
    """Result of a correlation check for a candidate ticker.

    Attributes
    ----------
    allowed:
        False when the trade is blocked because correlation exceeds
        ``block_threshold``.
    max_correlation_found:
        Highest pairwise correlation found (absolute value) between the
        candidate and any open position.
    correlated_with:
        Ticker of the open position that produced ``max_correlation_found``.
        Empty string when there are no open positions.
    size_adjustment:
        Multiplier to apply to the intended position size.  1.0 = full size;
        <1.0 when correlation warrants a reduction.
    reason:
        Human-readable explanation of the decision.
    pairs:
        Dict mapping each open-position ticker to its correlation with the
        candidate.  Populated even when the trade is allowed.
    """

    allowed: bool
    max_correlation_found: float
    correlated_with: str = ""
    size_adjustment: float = 1.0
    reason: str = ""
    pairs: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CorrelationFilter:
    """Filter and adjust position sizes based on correlation with open positions.

    Parameters
    ----------
    max_correlation:
        Above this threshold the position size starts being reduced.
    block_threshold:
        At or above this threshold the trade is blocked entirely.
    lookback_days:
        Number of calendar days of price history used for the rolling
        Pearson correlation window.
    min_overlapping_bars:
        Minimum number of overlapping daily bars required to compute a
        valid correlation.  Pairs with fewer overlapping bars are skipped
        (treated as uncorrelated).
    """

    def __init__(
        self,
        max_correlation: float = 0.70,
        block_threshold: float = 0.85,
        lookback_days: int = 60,
        min_overlapping_bars: int = 20,
    ) -> None:
        if not (0.0 < max_correlation < block_threshold <= 1.0):
            raise ValueError(
                "Requires 0 < max_correlation < block_threshold ≤ 1.0; "
                f"got max_correlation={max_correlation}, "
                f"block_threshold={block_threshold}"
            )
        self.max_correlation = max_correlation
        self.block_threshold = block_threshold
        self.lookback_days = lookback_days
        self.min_overlapping_bars = min_overlapping_bars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_correlation(
        self,
        candidate_ticker: str,
        open_position_tickers: list[str],
        data_manager: "DataManager",
    ) -> CorrelationCheckResult:
        """Check correlation of the candidate against all open positions.

        Parameters
        ----------
        candidate_ticker:
            The ticker symbol being considered for entry.
        open_position_tickers:
            List of tickers currently in the portfolio.
        data_manager:
            DataManager instance used to fetch price history.

        Returns
        -------
        CorrelationCheckResult
        """
        if not open_position_tickers:
            return CorrelationCheckResult(
                allowed=True,
                max_correlation_found=0.0,
                correlated_with="",
                size_adjustment=1.0,
                reason="No open positions; no correlation check needed.",
                pairs={},
            )

        # Remove the candidate from the open positions list in case it was
        # accidentally included (e.g. partial position already open).
        peers = [t for t in open_position_tickers if t != candidate_ticker]

        if not peers:
            return CorrelationCheckResult(
                allowed=True,
                max_correlation_found=0.0,
                correlated_with="",
                size_adjustment=1.0,
                reason="Only existing position is the candidate itself.",
                pairs={},
            )

        # Fetch price data for all tickers concurrently
        all_tickers = [candidate_ticker] + peers
        returns_map = await self._fetch_returns(all_tickers, data_manager)

        if candidate_ticker not in returns_map or returns_map[candidate_ticker].empty:
            logger.warning(
                "correlation_filter.no_data_for_candidate",
                ticker=candidate_ticker,
            )
            return CorrelationCheckResult(
                allowed=True,
                max_correlation_found=0.0,
                correlated_with="",
                size_adjustment=1.0,
                reason=(
                    f"No price data for {candidate_ticker}; "
                    "correlation check skipped."
                ),
                pairs={},
            )

        candidate_returns = returns_map[candidate_ticker]
        pairs: dict[str, float] = {}

        for peer in peers:
            if peer not in returns_map or returns_map[peer].empty:
                logger.debug(
                    "correlation_filter.no_data_for_peer",
                    peer=peer,
                    candidate=candidate_ticker,
                )
                continue

            corr = self._pearson_correlation(candidate_returns, returns_map[peer])
            if corr is not None:
                pairs[peer] = corr

        if not pairs:
            return CorrelationCheckResult(
                allowed=True,
                max_correlation_found=0.0,
                correlated_with="",
                size_adjustment=1.0,
                reason="Could not compute correlations with any open position (insufficient data).",
                pairs={},
            )

        # Find the maximum absolute correlation
        max_corr_ticker = max(pairs, key=lambda t: abs(pairs[t]))
        max_corr_value = abs(pairs[max_corr_ticker])

        # Decision logic
        if max_corr_value >= self.block_threshold:
            logger.info(
                "correlation_filter.blocked",
                candidate=candidate_ticker,
                correlated_with=max_corr_ticker,
                correlation=round(max_corr_value, 4),
                threshold=self.block_threshold,
            )
            return CorrelationCheckResult(
                allowed=False,
                max_correlation_found=max_corr_value,
                correlated_with=max_corr_ticker,
                size_adjustment=0.0,
                reason=(
                    f"Trade blocked: {candidate_ticker} is {max_corr_value:.2f} correlated "
                    f"with open position {max_corr_ticker} "
                    f"(threshold={self.block_threshold})."
                ),
                pairs=pairs,
            )

        if max_corr_value > self.max_correlation:
            # Linear interpolation: at max_correlation → size_adj = 1.0,
            # at block_threshold → size_adj = 0.0
            # Formula from spec: 1.0 - ((max_corr - max_correlation) / (block_threshold - max_correlation))
            size_adjustment = 1.0 - (
                (max_corr_value - self.max_correlation)
                / (self.block_threshold - self.max_correlation)
            )
            size_adjustment = float(np.clip(size_adjustment, 0.0, 1.0))

            logger.info(
                "correlation_filter.size_reduced",
                candidate=candidate_ticker,
                correlated_with=max_corr_ticker,
                correlation=round(max_corr_value, 4),
                size_adjustment=round(size_adjustment, 4),
            )
            return CorrelationCheckResult(
                allowed=True,
                max_correlation_found=max_corr_value,
                correlated_with=max_corr_ticker,
                size_adjustment=size_adjustment,
                reason=(
                    f"Position size reduced to {size_adjustment:.0%}: "
                    f"{candidate_ticker} is {max_corr_value:.2f} correlated "
                    f"with {max_corr_ticker} (max_correlation={self.max_correlation})."
                ),
                pairs=pairs,
            )

        # All correlations are within acceptable limits
        logger.debug(
            "correlation_filter.passed",
            candidate=candidate_ticker,
            max_correlation=round(max_corr_value, 4),
        )
        return CorrelationCheckResult(
            allowed=True,
            max_correlation_found=max_corr_value,
            correlated_with=max_corr_ticker,
            size_adjustment=1.0,
            reason=(
                f"Correlation within limits: max pairwise correlation "
                f"{max_corr_value:.2f} ≤ {self.max_correlation}."
            ),
            pairs=pairs,
        )

    def compute_portfolio_concentration(self, correlations: dict[str, float]) -> float:
        """Compute an overall portfolio correlation concentration score.

        Parameters
        ----------
        correlations:
            Dict mapping ticker pairs (or any string key) to their pairwise
            absolute correlation values (0–1).

        Returns
        -------
        float
            A score from 0 to 100 where:
              0   = all positions completely uncorrelated
              100 = all positions perfectly correlated (same direction)

        The score is the mean of the squared absolute correlations scaled to
        100.  Squaring emphasises highly-correlated pairs and is equivalent
        to the mean ``R²`` across all pairwise relationships.
        """
        if not correlations:
            return 0.0

        values = np.array(list(correlations.values()), dtype=float)
        # Clamp to [0, 1] in case any out-of-range values slipped through
        values = np.clip(np.abs(values), 0.0, 1.0)
        # Mean R² scaled to 0-100
        score = float(np.mean(values ** 2) * 100.0)
        return round(score, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_returns(
        self,
        tickers: list[str],
        data_manager: "DataManager",
    ) -> dict[str, pd.Series]:
        """Fetch daily log-returns for each ticker concurrently."""
        end_date = date.today()
        # Extra calendar buffer to ensure we get ~lookback_days trading bars
        start_date = end_date - timedelta(days=self.lookback_days + 45)

        async def _one(ticker: str) -> tuple[str, pd.Series]:
            try:
                df = await data_manager.get_daily_ohlcv(ticker, start_date, end_date)
                if df.empty or "close" not in df.columns:
                    return ticker, pd.Series(dtype=float)
                closes = df["close"].dropna()
                if len(closes) < 2:
                    return ticker, pd.Series(dtype=float)
                # Log-returns are better for correlation than simple returns
                log_rets = np.log(closes / closes.shift(1)).dropna()
                return ticker, log_rets
            except Exception as exc:
                logger.warning(
                    "correlation_filter.fetch_failed",
                    ticker=ticker,
                    error=str(exc),
                )
                return ticker, pd.Series(dtype=float)

        results = await asyncio.gather(*[_one(t) for t in tickers])
        return dict(results)

    def _pearson_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> float | None:
        """Compute Pearson correlation over the aligned, overlapping window.

        Returns None if there are fewer than ``min_overlapping_bars`` common
        dates (insufficient data for a reliable estimate).
        """
        # Align on common dates
        aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
        if len(aligned) < self.min_overlapping_bars:
            return None

        # Use only the most recent lookback_days trading bars
        aligned = aligned.iloc[-self.lookback_days :]

        if len(aligned) < self.min_overlapping_bars:
            return None

        a = aligned.iloc[:, 0].values
        b = aligned.iloc[:, 1].values

        # numpy corrcoef returns a 2x2 matrix; [0, 1] is the cross-correlation
        corr_matrix = np.corrcoef(a, b)
        corr = float(corr_matrix[0, 1])

        # Guard against NaN (e.g., zero-variance series)
        if np.isnan(corr):
            return None

        return corr
