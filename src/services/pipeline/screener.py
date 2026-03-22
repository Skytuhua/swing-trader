"""
Multi-stage screener: technical → regime-aware → news → sentiment.

Returns a list of ScreenedCandidate dataclasses representing stocks that
have passed all four screening stages and are ready for scoring/ranking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import DataQuality, MarketRegime
from src.core.exceptions import DataProviderError

if TYPE_CHECKING:
    from src.core.config import TechnicalConfig
    from src.services.market_data.manager import DataManager
    from src.services.news.scorer import NewsScore, NewsScorer
    from src.services.sentiment.aggregator import SentimentProfile, SentimentScorer
    from src.services.technical.engine import TechnicalEngine, TechnicalProfile

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ScreenedCandidate:
    """A stock that has passed all screening stages."""

    ticker: str
    technical: "TechnicalProfile"
    news: "NewsScore"
    sentiment: "SentimentProfile"

    # Optional enrichment fields populated after screening
    screened_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    data_quality: DataQuality = DataQuality.GOOD
    next_earnings_date: date | None = None

    # Raw metadata forwarded to the scorer
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers used by the scoring engine
    # ------------------------------------------------------------------

    def has_earnings_within(self, days: int = 5) -> bool:
        """Return True if earnings are scheduled within *days* calendar days."""
        if self.next_earnings_date is None:
            return False
        today = datetime.now(tz=timezone.utc).date()
        return (self.next_earnings_date - today).days <= days

    @property
    def technical_score(self) -> float:
        return self.technical.composite_score

    @property
    def news_score(self) -> float:
        return self.news.score

    @property
    def sentiment_score(self) -> float:
        return self.sentiment.score


# ---------------------------------------------------------------------------
# Regime assessment type (imported from regime service at runtime)
# ---------------------------------------------------------------------------


@dataclass
class RegimeAssessment:
    """Lightweight DTO – the real object comes from services.regime.engine."""

    regime: MarketRegime = MarketRegime.MIXED
    confidence: float = 50.0
    trend_strength: float = 50.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

_TECHNICAL_PRESCREEN_MIN = 40.0          # Minimum composite technical score
_TECHNICAL_UNFAVORABLE_MIN = 70.0        # Stricter bar in unfavorable regime
_TECHNICAL_MIXED_MIN = 55.0             # Stricter bar in mixed regime
_NEWS_NEGATIVE_CATALYST_MIN = 30.0       # Skip if negative catalyst AND low score
_SENTIMENT_CROWDING_MAX = 80.0          # Skip if over-crowded


class MultiStageScreener:
    """Run candidates through technical, news, and sentiment screens.

    Stages:
      1. Technical pre-screen (composite score threshold)
      2. Regime-aware filter (raise threshold in mixed/unfavorable regimes)
      3. News screen (reject strong negative catalysts)
      4. Sentiment screen (reject overly crowded names)
    """

    def __init__(
        self,
        data_manager: "DataManager",
        technical_engine: "TechnicalEngine",
        news_scorer: "NewsScorer",
        sentiment_scorer: "SentimentScorer",
        concurrency: int = 10,
    ) -> None:
        self.data = data_manager
        self.technical_engine = technical_engine
        self.news_scorer = news_scorer
        self.sentiment_scorer = sentiment_scorer
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def screen(
        self,
        tickers: list[str],
        regime: RegimeAssessment,
    ) -> list[ScreenedCandidate]:
        """Screen tickers through all four stages.

        Args:
            tickers: Universe tickers (from UniverseFilter).
            regime:  Current market regime assessment.

        Returns:
            List of ScreenedCandidate objects that passed all stages.
        """
        logger.info("screener_start", ticker_count=len(tickers), regime=regime.regime)

        tasks = [self._screen_ticker(ticker, regime) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[ScreenedCandidate] = []
        errors = 0
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.warning("screener_ticker_error", ticker=ticker, error=str(result))
                errors += 1
            elif result is not None:
                candidates.append(result)

        logger.info(
            "screener_complete",
            input=len(tickers),
            passed=len(candidates),
            errors=errors,
        )
        return candidates

    # ------------------------------------------------------------------
    # Per-ticker screening pipeline
    # ------------------------------------------------------------------

    async def _screen_ticker(
        self,
        ticker: str,
        regime: RegimeAssessment,
    ) -> ScreenedCandidate | None:
        """Run all four screening stages for a single ticker.

        Returns:
            ScreenedCandidate if it passes all stages, else None.
        """
        async with self._semaphore:
            return await self._run_stages(ticker, regime)

    async def _run_stages(
        self,
        ticker: str,
        regime: RegimeAssessment,
    ) -> ScreenedCandidate | None:
        log = logger.bind(ticker=ticker)

        # ---- Stage 1: Technical pre-screen ----
        try:
            daily_df = await self.data.get_daily_ohlcv(ticker, days=200)
        except DataProviderError as exc:
            log.debug("screener_ohlcv_fetch_failed", error=str(exc))
            return None

        if daily_df is None or daily_df.empty:
            log.debug("screener_no_ohlcv_data")
            return None

        try:
            tech_profile = await self.technical_engine.analyze(ticker, daily_df)
        except Exception as exc:
            log.warning("screener_technical_analysis_failed", error=str(exc))
            return None

        if tech_profile.composite_score < _TECHNICAL_PRESCREEN_MIN:
            log.debug(
                "screener_rejected_technical",
                score=tech_profile.composite_score,
                threshold=_TECHNICAL_PRESCREEN_MIN,
            )
            return None

        # ---- Stage 2: Regime-aware filter ----
        regime_threshold = self._regime_threshold(regime)
        if tech_profile.composite_score < regime_threshold:
            log.debug(
                "screener_rejected_regime",
                score=tech_profile.composite_score,
                threshold=regime_threshold,
                regime=regime.regime,
            )
            return None

        # ---- Stage 3: News screen ----
        try:
            news_score = await self.news_scorer.score_for_ticker(ticker)
        except Exception as exc:
            log.warning("screener_news_score_failed", error=str(exc))
            # Treat news as neutral rather than rejecting on data error
            news_score = self._neutral_news_score()

        if news_score.negative_catalyst and news_score.score < _NEWS_NEGATIVE_CATALYST_MIN:
            log.debug(
                "screener_rejected_news",
                score=news_score.score,
                negative_catalyst=news_score.negative_catalyst,
            )
            return None

        # ---- Stage 4: Sentiment screen ----
        try:
            sent_profile = await self.sentiment_scorer.score_for_ticker(ticker)
        except Exception as exc:
            log.warning("screener_sentiment_score_failed", error=str(exc))
            sent_profile = self._neutral_sentiment_profile(ticker)

        if sent_profile.crowding_risk > _SENTIMENT_CROWDING_MAX:
            log.debug(
                "screener_rejected_sentiment",
                crowding_risk=sent_profile.crowding_risk,
                threshold=_SENTIMENT_CROWDING_MAX,
            )
            return None

        # ---- Build ScreenedCandidate ----
        # Determine data quality
        data_quality = self._assess_data_quality(daily_df)

        # Fetch earnings date (best-effort, non-blocking)
        next_earnings: date | None = None
        try:
            next_earnings = await self.data.get_next_earnings_date(ticker)
        except Exception:
            pass

        candidate = ScreenedCandidate(
            ticker=ticker,
            technical=tech_profile,
            news=news_score,
            sentiment=sent_profile,
            data_quality=data_quality,
            next_earnings_date=next_earnings,
        )

        log.info(
            "screener_candidate_passed",
            technical_score=round(tech_profile.composite_score, 1),
            news_score=round(news_score.score, 1),
            sentiment_score=round(sent_profile.score, 1),
            crowding_risk=round(sent_profile.crowding_risk, 1),
        )
        return candidate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_threshold(regime: RegimeAssessment) -> float:
        if regime.regime == MarketRegime.UNFAVORABLE:
            return _TECHNICAL_UNFAVORABLE_MIN
        if regime.regime == MarketRegime.MIXED:
            return _TECHNICAL_MIXED_MIN
        return _TECHNICAL_PRESCREEN_MIN  # FAVORABLE: use base threshold

    @staticmethod
    def _assess_data_quality(daily_df: Any) -> DataQuality:
        """Heuristically score OHLCV data quality."""
        try:
            if len(daily_df) < 20:
                return DataQuality.DEGRADED
            null_pct = daily_df.isnull().mean().max()
            if null_pct > 0.05:
                return DataQuality.DEGRADED
            return DataQuality.GOOD
        except Exception:
            return DataQuality.DEGRADED

    @staticmethod
    def _neutral_news_score() -> Any:
        """Return a minimal neutral NewsScore when the news service is unavailable."""
        from dataclasses import make_dataclass  # local import to avoid circular deps

        # Build a duck-typed proxy; the real object is NewsScore from news.scorer
        class _NeutralNews:
            score = 50.0
            negative_catalyst = False
            catalyst_present = False
            conflicting_signals = False
            item_count = 0
            explanation = "News data unavailable; using neutral score."

        return _NeutralNews()

    @staticmethod
    def _neutral_sentiment_profile(ticker: str) -> Any:
        """Return a minimal neutral SentimentProfile when sentiment service is down."""

        class _NeutralSentiment:
            score = 50.0
            crowding_risk = 0.0
            phase = None
            explanation = "Sentiment data unavailable; using neutral score."

        return _NeutralSentiment()
