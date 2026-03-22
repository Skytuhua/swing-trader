"""
Exit engine: evaluate all exit conditions for a monitored position.

Exit hierarchy (in order of precedence):
  1. Hard stop loss (immediate)
  2. Take-profit 1 – sell 50% (immediate)
  3. Take-profit 2 – sell remainder (immediate)
  4. Trailing stop (immediate)
  5. Time stop – max hold days reached (end-of-day)
  6. Thesis deterioration – re-score below threshold (next_bar)
  7. News shock – high-severity negative catalyst (immediate)
  8. Regime deterioration – market turned unfavorable (next_bar)

Returns ExitSignal on exit, None to hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import ExitReason, MarketRegime

if TYPE_CHECKING:
    from src.services.execution.base import Quote
    from src.services.execution.order_manager import Position
    from src.services.market_data.manager import DataManager
    from src.services.news.scorer import NewsScorer
    from src.services.pipeline.screener import RegimeAssessment
    from src.services.scoring.engine import ScoringEngine
    from src.services.technical.engine import TechnicalEngine

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExitSignal:
    """Signal to exit a position, with urgency level and optional partial size."""

    reason: ExitReason
    urgency: str                          # "immediate" | "next_bar" | "end_of_day"
    partial_pct: float | None = None      # 0-1 fraction to sell; None = 100%
    price_at_signal: float | None = None
    triggered_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    detail: str = ""

    @property
    def is_partial(self) -> bool:
        return self.partial_pct is not None and self.partial_pct < 1.0

    @property
    def is_immediate(self) -> bool:
        return self.urgency == "immediate"


# ---------------------------------------------------------------------------
# Exit engine
# ---------------------------------------------------------------------------

_THESIS_RESCORE_MIN = 30.0      # Below this score → thesis has broken down
_NEWS_SHOCK_SEVERITY = 0.7      # News severity above this is considered a shock
_REGIME_LOSS_PCT_THRESHOLD = 2.0  # Only exit on regime if not deeply profitable


class ExitEngine:
    """Evaluate all exit conditions for a monitored position.

    Dependencies (all optional – engine degrades gracefully):
    - data_manager: for fetching regime snapshots
    - regime_engine: for reclassifying market regime
    - news_scorer:   for detecting news shocks
    - scoring_engine + technical_engine: for thesis re-scoring
    """

    def __init__(
        self,
        data_manager: "DataManager | None" = None,
        regime_engine: Any = None,          # RegimeEngine from services.regime
        news_scorer: "NewsScorer | None" = None,
        scoring_engine: "ScoringEngine | None" = None,
        technical_engine: "TechnicalEngine | None" = None,
        thesis_rescore_min: float = _THESIS_RESCORE_MIN,
        news_shock_severity: float = _NEWS_SHOCK_SEVERITY,
        max_hold_days: int = 5,
    ) -> None:
        self.data = data_manager
        self.regime_engine = regime_engine
        self.news_scorer = news_scorer
        self.scoring_engine = scoring_engine
        self.technical_engine = technical_engine
        self.thesis_rescore_min = thesis_rescore_min
        self.news_shock_severity = news_shock_severity
        self.max_hold_days = max_hold_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        position: "Position",
        quote: "Quote",
    ) -> ExitSignal | None:
        """Evaluate all exit conditions in priority order.

        Args:
            position: The open position to evaluate.
            quote:    Latest market quote.

        Returns:
            ExitSignal if an exit condition is met, None to continue holding.
        """
        price = quote.price
        log = logger.bind(ticker=position.ticker, price=price)

        # 1. HARD STOP LOSS ------------------------------------------------
        if price <= position.stop_loss:
            log.warning(
                "exit_stop_loss",
                current=price,
                stop=position.stop_loss,
            )
            return ExitSignal(
                reason=ExitReason.STOP_LOSS,
                urgency="immediate",
                price_at_signal=price,
                detail=f"Price {price:.2f} ≤ stop {position.stop_loss:.2f}",
            )

        # 2. TAKE-PROFIT 1 (sell 50%) ---------------------------------------
        if price >= position.take_profit_1 and not position.tp1_hit:
            log.info("exit_take_profit_1", current=price, tp1=position.take_profit_1)
            return ExitSignal(
                reason=ExitReason.TAKE_PROFIT_1,
                urgency="immediate",
                partial_pct=0.5,
                price_at_signal=price,
                detail=f"Price {price:.2f} ≥ TP1 {position.take_profit_1:.2f}",
            )

        # 3. TAKE-PROFIT 2 (sell remainder) ---------------------------------
        if position.take_profit_2 and price >= position.take_profit_2:
            log.info("exit_take_profit_2", current=price, tp2=position.take_profit_2)
            return ExitSignal(
                reason=ExitReason.TAKE_PROFIT_2,
                urgency="immediate",
                price_at_signal=price,
                detail=f"Price {price:.2f} ≥ TP2 {position.take_profit_2:.2f}",
            )

        # 4. TRAILING STOP ---------------------------------------------------
        trailing_stop = getattr(position, "trailing_stop_price", None)
        if trailing_stop and price <= trailing_stop:
            log.info("exit_trailing_stop", current=price, trail=trailing_stop)
            return ExitSignal(
                reason=ExitReason.TRAILING_STOP,
                urgency="immediate",
                price_at_signal=price,
                detail=f"Price {price:.2f} ≤ trailing stop {trailing_stop:.2f}",
            )

        # 5. TIME STOP -------------------------------------------------------
        hold_days = getattr(position, "hold_days", 0) or 0
        max_days = getattr(position, "max_hold_days", self.max_hold_days)
        if hold_days >= max_days:
            log.info("exit_time_stop", hold_days=hold_days, max_days=max_days)
            return ExitSignal(
                reason=ExitReason.TIME_STOP,
                urgency="end_of_day",
                price_at_signal=price,
                detail=f"Held {hold_days} days ≥ max {max_days}",
            )

        # 6. THESIS DETERIORATION -------------------------------------------
        thesis_score = await self._rescore_thesis(position)
        if thesis_score is not None and thesis_score < self.thesis_rescore_min:
            log.info("exit_thesis_deterioration", thesis_score=thesis_score)
            return ExitSignal(
                reason=ExitReason.THESIS_DETERIORATION,
                urgency="next_bar",
                price_at_signal=price,
                detail=f"Re-score {thesis_score:.1f} < threshold {self.thesis_rescore_min}",
            )

        # 7. NEWS SHOCK ------------------------------------------------------
        news_shock = await self._check_news_shock(position.ticker)
        if news_shock is not None:
            severity = getattr(news_shock, "severity", 0.0) or 0.0
            if severity > self.news_shock_severity:
                log.warning(
                    "exit_news_shock",
                    severity=severity,
                    ticker=position.ticker,
                )
                return ExitSignal(
                    reason=ExitReason.NEWS_SHOCK,
                    urgency="immediate",
                    price_at_signal=price,
                    detail=f"News shock detected (severity={severity:.2f})",
                )

        # 8. REGIME DETERIORATION --------------------------------------------
        regime = await self._get_current_regime()
        if regime is not None and regime.regime == MarketRegime.UNFAVORABLE:
            unrealised_pct = getattr(position, "unrealized_pnl_pct", 0.0) or 0.0
            if unrealised_pct < _REGIME_LOSS_PCT_THRESHOLD:
                log.info(
                    "exit_regime_deterioration",
                    regime=regime.regime,
                    unrealised_pct=unrealised_pct,
                )
                return ExitSignal(
                    reason=ExitReason.REGIME_DETERIORATION,
                    urgency="next_bar",
                    price_at_signal=price,
                    detail=(
                        f"Regime turned UNFAVORABLE; "
                        f"position only {unrealised_pct:.1f}% profitable."
                    ),
                )

        # 9. UPDATE TRAILING STOP (no exit) ----------------------------------
        self._update_trailing_stop(position, quote)

        return None  # No exit condition met

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def _update_trailing_stop(
        self,
        position: "Position",
        quote: "Quote",
    ) -> None:
        """Ratchet trailing stop upward based on the highest price since entry."""
        max_price = getattr(position, "max_price_since_entry", quote.price) or quote.price
        entry_price = position.entry_price
        atr = getattr(position, "atr_at_entry", 0.0) or 0.0

        # Only activate trailing stop after 1% gain
        if max_price < entry_price * 1.01:
            return
        if atr <= 0:
            return

        new_trail = max_price - (atr * 1.5)
        current_stop = getattr(position, "trailing_stop_price", None) or position.stop_loss

        # Only move stop up, never down
        if new_trail > current_stop:
            position.trailing_stop_price = new_trail
            logger.debug(
                "trailing_stop_updated",
                ticker=position.ticker,
                new_trail=round(new_trail, 4),
                max_price=round(max_price, 4),
            )

    # ------------------------------------------------------------------
    # Optional evaluators
    # ------------------------------------------------------------------

    async def _rescore_thesis(self, position: "Position") -> float | None:
        """Re-score the technical thesis for the position.  Returns None if unavailable."""
        if self.technical_engine is None or self.data is None:
            return None
        try:
            daily_df = await self.data.get_daily_ohlcv(position.ticker, days=60)
            tech = await self.technical_engine.analyze(position.ticker, daily_df)
            return float(tech.composite_score)
        except Exception as exc:
            logger.debug("exit_engine_rescore_failed", ticker=position.ticker, error=str(exc))
            return None

    async def _check_news_shock(self, ticker: str) -> Any | None:
        """Check for high-severity news in the last few hours.  Returns None if unavailable."""
        if self.news_scorer is None:
            return None
        try:
            news_score = await self.news_scorer.score_for_ticker(ticker)
            # Return the news score object for severity inspection
            if news_score.negative_catalyst and news_score.score < 30:
                # Synthesise a severity proxy
                news_score.severity = (30.0 - news_score.score) / 30.0
                return news_score
            return None
        except Exception as exc:
            logger.debug("exit_engine_news_check_failed", ticker=ticker, error=str(exc))
            return None

    async def _get_current_regime(self) -> Any | None:
        """Classify the current market regime.  Returns None if unavailable."""
        if self.regime_engine is None or self.data is None:
            return None
        try:
            snapshot = await self.data.get_market_snapshot()
            return await self.regime_engine.classify(snapshot)
        except Exception as exc:
            logger.debug("exit_engine_regime_check_failed", error=str(exc))
            return None
