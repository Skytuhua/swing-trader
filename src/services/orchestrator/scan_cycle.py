"""
Main scan cycle: the full autonomous trading decision pipeline.

Called by the scheduler (``scan_job``) on each configured cron tick, and
can also be invoked directly for backtesting or manual runs.

Pipeline
--------
 1.  Kill-switch check          – abort immediately if active
 2.  Market-hours check         – abort if market is closed
 3.  Market regime              – classify FAVORABLE / MIXED / UNFAVORABLE
 4.  Regime gate                – NO TRADE if UNFAVORABLE and config blocks it
 5.  Universe filter            – liquid, tradable tickers
 6.  Multi-stage screener       – technical / regime / news / sentiment passes
 7.  Score + rank               – ScoringEngine + CandidateRanker
 8.  Final selection            – FinalSelector (best or NO TRADE)
 9.  NO TRADE path              – persist decision, publish event, return
10.  Trade construction         – entry zone, stop, TP via TradeConstructor
11.  Position sizing            – PositionSizer
12.  Pre-trade validation       – PreTradeValidator
13.  Pre-trade risk checks      – RiskEngine.pre_trade_check()
14.  Risk fail path             – persist NO TRADE, log, return
15.  Build TradingDecision ORM  – all fields populated
16.  Execute entry order        – OrderManager.place_entry_order()
17.  Persist everything         – DB session
18.  Publish events             – ScanCompletedEvent, TradeOpenedEvent
19.  Update metrics             – Prometheus counters / gauges
20.  Return                     – duration logged

All exceptions inside individual steps are caught and logged; the cycle
tries to persist a record even on partial failure so the audit trail is
complete.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from src.core.enums import MarketRegime, TradingMode
from src.core.events import (
    ScanCompletedEvent,
    TradeOpenedEvent,
    event_bus,
)
from src.core.exceptions import (
    BrokerError,
    ConfigurationError,
    DataProviderError,
    SwingTraderError,
)
from src.core.logging_config import bind_trade_context, clear_trade_context
from src.core.metrics import METRICS

if TYPE_CHECKING:
    from src.core.config import AppConfig
    from src.core.database import DatabaseManager
    from src.core.redis_client import RedisManager
    from src.services.execution.alpaca_broker import AlpacaBroker
    from src.services.execution.order_manager import OrderManager
    from src.services.monitor.exit_engine import ExitEngine
    from src.services.monitor.position_monitor import PositionMonitor
    from src.services.pipeline.ranker import CandidateRanker
    from src.services.pipeline.screener import MultiStageScreener
    from src.services.pipeline.selector import FinalSelector
    from src.services.pipeline.universe import UniverseFilter
    from src.services.regime.engine import RegimeEngine
    from src.services.risk.calendar import MarketCalendar
    from src.services.risk.engine import RiskEngine
    from src.services.risk.kill_switch import KillSwitch
    from src.services.scoring.engine import ScoringEngine
    from src.services.trade.constructor import TradeConstructor
    from src.services.trade.sizer import PositionSizer
    from src.services.trade.validator import PreTradeValidator

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Service container (dependency-injected, populated by main.py)
# ---------------------------------------------------------------------------


@dataclass
class ServiceContainer:
    """All services needed by the scan cycle and scheduler jobs.

    Every field maps 1-to-1 to a service class.  ``main.py`` creates all
    instances and populates this container before handing it to the scheduler.
    """

    # Infrastructure
    config: Any                      # AppConfig
    db: Any                          # DatabaseManager
    redis: Any                       # RedisManager

    # Market data
    data_manager: Any                # DataManager

    # Technical analysis
    technical_engine: Any            # TechnicalEngine

    # News services
    news_provider: Any               # FinnhubNewsProvider
    news_processor: Any              # NewsProcessor
    news_scorer: Any                 # NewsScorer
    news_deduplicator: Any           # NewsDeduplicator

    # Sentiment services
    sentiment_aggregator: Any        # SentimentAggregator
    sentiment_scorer: Any            # SentimentScorer

    # Regime engine
    regime_engine: Any               # RegimeEngine

    # Pipeline
    universe_filter: Any             # UniverseFilter
    screener: Any                    # MultiStageScreener
    ranker: Any                      # CandidateRanker
    selector: Any                    # FinalSelector
    scoring_engine: Any              # ScoringEngine

    # Trade construction
    trade_constructor: Any           # TradeConstructor
    position_sizer: Any              # PositionSizer
    trade_validator: Any             # PreTradeValidator

    # Execution
    order_manager: Any               # OrderManager
    broker: Any                      # AlpacaBroker | PaperBroker

    # Monitor
    position_monitor: Any            # PositionMonitor
    exit_engine: Any                 # ExitEngine

    # Risk
    risk_engine: Any                 # RiskEngine
    kill_switch: Any                 # KillSwitch
    calendar: Any                    # MarketCalendar

    # Optional extras (may be None for paper mode)
    trading_mode: str = "paper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cycle_log(cycle_id: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound with the current cycle_id."""
    return logger.bind(cycle_id=cycle_id)


async def _get_portfolio_state(services: ServiceContainer) -> Any:
    """Fetch broker account state and build a PortfolioState for risk checks."""
    from src.services.risk.engine import RiskEngine

    try:
        account = await services.broker.get_account()
        portfolio_val = float(getattr(account, "portfolio_value", 0) or 0)
        cash = float(getattr(account, "cash", 0) or 0)
        daily_pnl_pct = float(getattr(account, "daily_pnl_pct", 0) or 0)
        peak_value = float(getattr(account, "peak_value", portfolio_val) or portfolio_val)
        # Sum risk pct across open positions (approximation)
        open_positions: list[str] = []
        total_risk_pct = 0.0
        try:
            positions = await services.broker.get_positions()
            open_positions = [p.ticker for p in positions]
            total_risk_pct = len(open_positions) * 1.5  # 1.5% risk per position estimate
        except Exception:
            pass

        return RiskEngine.build_portfolio_state(
            portfolio_value=portfolio_val,
            cash=cash,
            daily_pnl_pct=daily_pnl_pct,
            peak_value=peak_value,
            total_risk_pct=total_risk_pct,
            open_positions=open_positions,
        )
    except Exception as exc:
        log = logger.bind(step="get_portfolio_state")
        log.error("portfolio_state_fetch_error", error=str(exc))
        # Return a conservative fallback so risk checks can still run
        from src.services.risk.engine import RiskEngine
        return RiskEngine.build_portfolio_state(
            portfolio_value=100_000.0,
            cash=100_000.0,
            daily_pnl_pct=0.0,
            peak_value=100_000.0,
            total_risk_pct=0.0,
        )


async def _get_market_data_for_regime(services: ServiceContainer) -> dict[str, Any]:
    """Fetch the market context dict expected by RegimeEngine.classify()."""
    try:
        snapshot = await services.data_manager.get_market_snapshot(force_refresh=True)
        return snapshot
    except Exception as exc:
        logger.warning("regime_market_data_fetch_error", error=str(exc))
        return {}


async def _persist_no_trade(
    services: ServiceContainer,
    cycle_id: str,
    reason: str,
    detail: str = "",
    regime: Any = None,
    candidates_count: int = 0,
    top_candidates: list[Any] | None = None,
) -> None:
    """Persist a NO TRADE decision record to the database."""
    from src.models.decision import TradingDecision as TradingDecisionORM

    try:
        decision = TradingDecisionORM(
            cycle_id=cycle_id,
            timestamp=datetime.now(tz=timezone.utc),
            is_no_trade=True,
            no_trade_reason=reason,
            reason_summary=detail or reason,
            market_regime=regime.regime.value if regime else None,
            regime_confidence=getattr(regime, "confidence", None) if regime else None,
            top_candidates_json=(
                [{"ticker": c.ticker, "score": c.total_score} for c in (top_candidates or [])][:5]
            ),
            config_version=getattr(services.config, "version", "unknown"),
        )
        async with services.db.session() as session:
            session.add(decision)
    except Exception as exc:
        logger.error("persist_no_trade_error", cycle_id=cycle_id, error=str(exc))


# ---------------------------------------------------------------------------
# Main scan cycle
# ---------------------------------------------------------------------------


async def run_scan_cycle(cycle_id: str, services: ServiceContainer) -> None:
    """Execute one full autonomous trading decision pipeline.

    Parameters
    ----------
    cycle_id:
        Unique identifier for this scan run (used for logging + DB audit).
    services:
        Fully-initialised service container from main.py.
    """
    log = _cycle_log(cycle_id)
    bind_trade_context(cycle_id=cycle_id)
    scan_start = time.monotonic()

    log.info("scan_cycle_start", cycle_id=cycle_id, mode=services.trading_mode)

    try:
        await _run_scan_cycle_inner(cycle_id=cycle_id, services=services, log=log)
    except Exception as exc:
        # Top-level safety net: should not normally reach here
        log.error("scan_cycle_unhandled_error", error=str(exc), exc_info=True)
        METRICS.scan_cycles_total.labels(status="error").inc()
    finally:
        elapsed = time.monotonic() - scan_start
        METRICS.scan_duration_seconds.observe(elapsed)
        log.info("scan_cycle_complete", duration_seconds=round(elapsed, 3))
        clear_trade_context()


async def _run_scan_cycle_inner(
    cycle_id: str,
    services: ServiceContainer,
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Inner scan cycle implementation (called by run_scan_cycle)."""
    scan_start = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Kill switch check
    # ------------------------------------------------------------------
    log.debug("scan_step_1_kill_switch_check")
    try:
        if await services.kill_switch.is_active:
            reason = await services.kill_switch.reason
            log.warning(
                "scan_aborted_kill_switch",
                reason=reason,
            )
            METRICS.scan_cycles_total.labels(status="kill_switch").inc()
            METRICS.no_trade_decisions_total.labels(reason="kill_switch").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="kill_switch_active",
                detail=f"Kill switch active: {reason}",
            )
            await event_bus.publish(
                ScanCompletedEvent(
                    cycle_id=cycle_id,
                    candidates_found=0,
                    is_no_trade=True,
                    duration_seconds=time.monotonic() - scan_start,
                )
            )
            return
    except Exception as exc:
        log.error("scan_step_1_error", error=str(exc))
        # Fail-safe: treat any kill-switch check error as active
        METRICS.scan_cycles_total.labels(status="kill_switch_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 2: Market hours check
    # ------------------------------------------------------------------
    log.debug("scan_step_2_market_hours_check")
    try:
        is_open = services.calendar.is_market_open()
        if not is_open:
            log.info("scan_aborted_market_closed")
            METRICS.no_trade_decisions_total.labels(reason="market_closed").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="market_closed",
                detail="NYSE market is not currently open.",
            )
            await event_bus.publish(
                ScanCompletedEvent(
                    cycle_id=cycle_id,
                    candidates_found=0,
                    is_no_trade=True,
                    duration_seconds=time.monotonic() - scan_start,
                )
            )
            return
    except Exception as exc:
        log.error("scan_step_2_error", error=str(exc))
        # Conservative: abort if we can't verify market is open
        return

    # ------------------------------------------------------------------
    # Step 3: Market regime classification
    # ------------------------------------------------------------------
    log.debug("scan_step_3_regime_classification")
    regime = None
    try:
        market_data = await _get_market_data_for_regime(services)
        regime = await services.regime_engine.classify(market_data)
        log.info(
            "scan_regime_classified",
            regime=regime.regime.value,
            score=round(regime.score, 1),
            confidence=round(getattr(regime, "confidence", 0), 1),
        )
    except Exception as exc:
        log.error("scan_step_3_regime_error", error=str(exc))
        # Use MIXED as a safe default if regime classification fails
        from src.services.regime.engine import RegimeAssessment
        from src.core.enums import MarketRegime

        class _FallbackRegime:
            regime = MarketRegime.MIXED
            score = 0.0
            confidence = 0.0
            explanation = "Regime classification failed – defaulting to MIXED."
            signals: dict = {}

        regime = _FallbackRegime()

    # ------------------------------------------------------------------
    # Step 4: Regime gate – no trading in UNFAVORABLE market
    # ------------------------------------------------------------------
    log.debug("scan_step_4_regime_gate")
    try:
        no_trade_in_unfavorable = getattr(
            getattr(services.config, "trading", None),
            "no_trade_in_unfavorable",
            True,
        )
        if regime.regime == MarketRegime.UNFAVORABLE and no_trade_in_unfavorable:
            log.warning(
                "scan_no_trade_unfavorable_regime",
                regime_score=regime.score,
                explanation=getattr(regime, "explanation", ""),
            )
            METRICS.no_trade_decisions_total.labels(reason="unfavorable_regime").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="unfavorable_regime",
                detail=f"Market regime is UNFAVORABLE (score={regime.score:.1f}). No new entries.",
                regime=regime,
            )
            await event_bus.publish(
                ScanCompletedEvent(
                    cycle_id=cycle_id,
                    candidates_found=0,
                    is_no_trade=True,
                    duration_seconds=time.monotonic() - scan_start,
                )
            )
            return
    except Exception as exc:
        log.error("scan_step_4_error", error=str(exc))

    # ------------------------------------------------------------------
    # Step 5: Tradable universe
    # ------------------------------------------------------------------
    log.debug("scan_step_5_universe_filter")
    universe: list[str] = []
    try:
        universe = await services.universe_filter.get_tradable_universe()
        log.info("scan_universe_filtered", ticker_count=len(universe))
        if not universe:
            log.warning("scan_no_trade_empty_universe")
            METRICS.no_trade_decisions_total.labels(reason="empty_universe").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="empty_universe",
                detail="No tickers in tradable universe after filtering.",
                regime=regime,
            )
            await event_bus.publish(
                ScanCompletedEvent(
                    cycle_id=cycle_id,
                    candidates_found=0,
                    is_no_trade=True,
                    duration_seconds=time.monotonic() - scan_start,
                )
            )
            return
    except Exception as exc:
        log.error("scan_step_5_error", error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="universe_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 6: Multi-stage screener
    # ------------------------------------------------------------------
    log.debug("scan_step_6_screening", ticker_count=len(universe))
    screened: list[Any] = []
    try:
        screened = await services.screener.screen(tickers=universe, regime=regime)
        log.info(
            "scan_screening_complete",
            candidates=len(screened),
            input_tickers=len(universe),
        )
    except Exception as exc:
        log.error("scan_step_6_screener_error", error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="screener_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 7: Score and rank candidates
    # ------------------------------------------------------------------
    log.debug("scan_step_7_scoring_ranking", candidates=len(screened))
    ranked: list[Any] = []
    try:
        ranked = services.ranker.rank(candidates=screened, regime=regime)
        log.info(
            "scan_ranking_complete",
            ranked_count=len(ranked),
            top_ticker=ranked[0].ticker if ranked else None,
            top_score=round(ranked[0].total_score, 1) if ranked else None,
        )
    except Exception as exc:
        log.error("scan_step_7_ranking_error", error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="ranking_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 8: Final selection
    # ------------------------------------------------------------------
    log.debug("scan_step_8_selection")
    from src.services.pipeline.selector import NoTradeDecision, SelectionResult

    try:
        selection = services.selector.select(ranked_candidates=ranked, regime=regime)
        log.info(
            "scan_selection_result",
            is_no_trade=isinstance(selection, NoTradeDecision),
            selected=getattr(selection, "ticker", None),
        )
    except Exception as exc:
        log.error("scan_step_8_selection_error", error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="selection_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 9: Handle NO TRADE from selector
    # ------------------------------------------------------------------
    if isinstance(selection, NoTradeDecision):
        log.info(
            "scan_no_trade_decision",
            reason=selection.reason,
            detail=selection.detail,
            top_candidate=selection.top_candidate,
            top_score=selection.top_score,
        )
        METRICS.no_trade_decisions_total.labels(reason=selection.reason).inc()
        METRICS.scan_cycles_total.labels(status="no_trade").inc()

        await _persist_no_trade(
            services, cycle_id,
            reason=selection.reason,
            detail=selection.detail,
            regime=regime,
            top_candidates=ranked[:5],
        )
        await event_bus.publish(
            ScanCompletedEvent(
                cycle_id=cycle_id,
                candidates_found=len(screened),
                is_no_trade=True,
                duration_seconds=time.monotonic() - scan_start,
            )
        )
        return

    # ------------------------------------------------------------------
    # We have a candidate to trade – extract from SelectionResult
    # ------------------------------------------------------------------
    assert isinstance(selection, SelectionResult)
    best = selection.scored_candidate
    ticker = selection.ticker
    bind_trade_context(cycle_id=cycle_id, ticker=ticker)

    log.info(
        "scan_candidate_selected",
        ticker=ticker,
        total_score=round(best.total_score, 1),
        confidence=round(best.confidence, 1),
        technical_score=round(best.technical_score, 1),
        news_score=round(best.news_score, 1),
        sentiment_score=round(best.sentiment_score, 1),
    )

    # ------------------------------------------------------------------
    # Step 10: Construct trade setup (entry, stop, TP)
    # ------------------------------------------------------------------
    log.debug("scan_step_10_trade_construction", ticker=ticker)
    trade_setup = None
    try:
        daily_df = await services.data_manager.get_daily_ohlcv(ticker=ticker, days=200)
        quote = await services.data_manager.get_quote(ticker=ticker)
        trade_setup = services.trade_constructor.construct(
            candidate=best,
            daily_df=daily_df,
            quote=quote,
        )
        log.info(
            "scan_trade_setup",
            ticker=ticker,
            entry_low=round(trade_setup.entry_low, 2),
            entry_high=round(trade_setup.entry_high, 2),
            stop_loss=round(trade_setup.stop_loss, 2),
            take_profit_1=round(trade_setup.take_profit_1, 2),
            take_profit_2=round(trade_setup.take_profit_2, 2),
            rr=round(trade_setup.risk_reward_ratio, 2),
        )
    except Exception as exc:
        log.error("scan_step_10_construction_error", ticker=ticker, error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="construction_error").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason="trade_construction_failed",
            detail=str(exc),
            regime=regime,
            top_candidates=ranked[:5],
        )
        return

    # ------------------------------------------------------------------
    # Step 11: Position sizing
    # ------------------------------------------------------------------
    log.debug("scan_step_11_position_sizing", ticker=ticker)
    position_size = None
    try:
        portfolio_state = await _get_portfolio_state(services)
        position_size = services.position_sizer.size(
            setup=trade_setup,
            confidence=best.confidence,
            regime=regime,
            portfolio_value=portfolio_state.portfolio_value,
            current_risk_pct=portfolio_state.total_risk_pct,
        )
        log.info(
            "scan_position_sized",
            ticker=ticker,
            shares=position_size.shares,
            dollar_size=round(position_size.dollar_size, 2),
            allocation_pct=round(position_size.allocation_pct, 2),
            risk_pct=round(position_size.risk_pct, 2),
            tradeable=position_size.is_tradeable,
        )
        if not position_size.is_tradeable:
            log.warning(
                "scan_no_trade_zero_size",
                ticker=ticker,
                notes=position_size.sizing_notes,
            )
            METRICS.no_trade_decisions_total.labels(reason="zero_position_size").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="zero_position_size",
                detail=position_size.sizing_notes,
                regime=regime,
                top_candidates=ranked[:5],
            )
            return
    except Exception as exc:
        log.error("scan_step_11_sizing_error", ticker=ticker, error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="sizing_error").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason="sizing_error",
            detail=str(exc),
            regime=regime,
        )
        return

    # ------------------------------------------------------------------
    # Step 12: Pre-trade validation
    # ------------------------------------------------------------------
    log.debug("scan_step_12_pre_trade_validation", ticker=ticker)
    try:
        validation = services.trade_validator.validate(setup=trade_setup, size=position_size)
        if not validation.passed:
            log.warning(
                "scan_no_trade_validation_failed",
                ticker=ticker,
                failures=validation.failures,
                warnings=validation.warnings,
            )
            METRICS.no_trade_decisions_total.labels(reason="validation_failed").inc()
            await _persist_no_trade(
                services, cycle_id,
                reason="pre_trade_validation_failed",
                detail="; ".join(validation.failures),
                regime=regime,
            )
            return
        if validation.warnings:
            log.warning(
                "scan_validation_warnings",
                ticker=ticker,
                warnings=validation.warnings,
            )
    except Exception as exc:
        log.error("scan_step_12_validation_error", ticker=ticker, error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="validation_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 13: Pre-trade risk checks
    # ------------------------------------------------------------------
    log.debug("scan_step_13_risk_checks", ticker=ticker)
    try:
        from src.services.risk.engine import TradingDecisionRisk

        risk_decision = TradingDecisionRisk(
            selected_ticker=ticker,
            risk_pct=position_size.risk_pct,
            allocation_pct=position_size.allocation_pct,
            data_timestamp=datetime.now(tz=timezone.utc),
        )
        portfolio_state = await _get_portfolio_state(services)
        risk_result = await services.risk_engine.pre_trade_check(
            decision=risk_decision,
            portfolio=portfolio_state,
        )
    except Exception as exc:
        log.error("scan_step_13_risk_check_error", ticker=ticker, error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="risk_check_error").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason="risk_check_error",
            detail=str(exc),
            regime=regime,
        )
        return

    # ------------------------------------------------------------------
    # Step 14: Risk check failed → NO TRADE
    # ------------------------------------------------------------------
    if not risk_result.passed:
        log.warning(
            "scan_no_trade_risk_check_failed",
            ticker=ticker,
            failed_check=risk_result.failed_check,
            reason=risk_result.reason,
        )
        METRICS.no_trade_decisions_total.labels(reason=risk_result.failed_check or "risk_failed").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason=f"risk_check_failed:{risk_result.failed_check}",
            detail=risk_result.reason,
            regime=regime,
        )
        await event_bus.publish(
            ScanCompletedEvent(
                cycle_id=cycle_id,
                candidates_found=len(screened),
                is_no_trade=True,
                duration_seconds=time.monotonic() - scan_start,
            )
        )
        return

    log.info(
        "scan_risk_checks_passed",
        ticker=ticker,
        checks_run=len(risk_result.checks_run),
    )

    # ------------------------------------------------------------------
    # Step 15: Build TradingDecision ORM object
    # ------------------------------------------------------------------
    log.debug("scan_step_15_build_decision", ticker=ticker)
    from src.models.decision import TradingDecision as TradingDecisionORM

    decision_id = str(uuid.uuid4())
    try:
        orm_decision = TradingDecisionORM(
            id=uuid.UUID(decision_id),
            cycle_id=cycle_id,
            timestamp=datetime.now(tz=timezone.utc),
            is_no_trade=False,
            selected_ticker=ticker,

            # Regime context
            market_regime=regime.regime.value,
            regime_confidence=getattr(regime, "confidence", None),

            # Candidate summary (top-5 for audit)
            top_candidates_json=[
                {"ticker": c.ticker, "score": round(c.total_score, 2), "confidence": round(c.confidence, 2)}
                for c in ranked[:5]
            ],

            # Component scores
            technical_score=round(best.technical_score, 2),
            news_score=round(best.news_score, 2),
            sentiment_score=round(best.sentiment_score, 2),
            risk_reward_score=round(best.risk_reward_score, 2),
            liquidity_score=round(best.liquidity_score, 2),
            regime_alignment_score=round(best.regime_score, 2),
            confidence_score=round(best.confidence, 2),

            # Trade construction
            entry_price_low=round(trade_setup.entry_low, 4),
            entry_price_high=round(trade_setup.entry_high, 4),
            entry_method=trade_setup.entry_method.value,
            stop_loss=round(trade_setup.stop_loss, 4),
            take_profit_1=round(trade_setup.take_profit_1, 4),
            take_profit_2=round(trade_setup.take_profit_2, 4),
            trailing_stop_rule=trade_setup.trailing_stop_rule,

            # Position sizing
            allocation_pct=round(position_size.allocation_pct, 4),
            dollar_size=round(position_size.dollar_size, 2),
            share_quantity=position_size.shares,

            # Risk metadata
            invalidation_conditions=trade_setup.invalidation_conditions,
            data_quality_flags={},

            # Thesis
            reason_summary=best.explanation or f"Selected {ticker} with score {best.total_score:.1f}",

            # Versions
            config_version=getattr(services.config, "version", "unknown"),
            model_version="1.0",
        )
    except Exception as exc:
        log.error("scan_step_15_build_decision_error", ticker=ticker, error=str(exc))
        METRICS.no_trade_decisions_total.labels(reason="decision_build_error").inc()
        return

    # ------------------------------------------------------------------
    # Step 16: Execute entry order
    # ------------------------------------------------------------------
    log.debug("scan_step_16_place_entry_order", ticker=ticker)
    from src.services.execution.order_manager import TradingDecision as OrderManagerDecision

    entry_order = None
    try:
        om_decision = OrderManagerDecision(
            id=decision_id,
            cycle_id=cycle_id,
            selected_ticker=ticker,
            risk_pct=position_size.risk_pct,
            allocation_pct=position_size.allocation_pct,
        )
        entry_order = await services.order_manager.place_entry_order(
            decision=om_decision,
            setup=trade_setup,
            size=position_size,
        )
        log.info(
            "scan_entry_order_placed",
            ticker=ticker,
            order_id=entry_order.id,
            broker_order_id=entry_order.broker_order_id,
            status=entry_order.status.value if entry_order.status else "unknown",
            shares=entry_order.quantity,
        )
        # Record entry in risk engine cooldown tracker
        services.risk_engine.record_trade_entry(ticker)
    except BrokerError as exc:
        log.error(
            "scan_entry_order_broker_error",
            ticker=ticker,
            error=str(exc),
        )
        METRICS.orders_rejected_total.labels(reason="broker_error").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason="entry_order_broker_error",
            detail=str(exc),
            regime=regime,
        )
        return
    except Exception as exc:
        log.error("scan_step_16_order_error", ticker=ticker, error=str(exc))
        METRICS.orders_rejected_total.labels(reason="unknown_error").inc()
        await _persist_no_trade(
            services, cycle_id,
            reason="entry_order_failed",
            detail=str(exc),
            regime=regime,
        )
        return

    # ------------------------------------------------------------------
    # Step 17: Create position record and persist everything
    # ------------------------------------------------------------------
    log.debug("scan_step_17_persist", ticker=ticker)
    from src.models.position import Position as PositionORM

    try:
        position_orm = PositionORM(
            decision_id=uuid.UUID(decision_id),
            ticker=ticker,
            status="open",
            entry_price=trade_setup.entry_high,
            entry_date=datetime.now(tz=timezone.utc),
            quantity=position_size.shares,
            original_quantity=position_size.shares,
            current_price=trade_setup.entry_high,
            stop_loss=trade_setup.stop_loss,
            take_profit_1=trade_setup.take_profit_1,
            take_profit_2=trade_setup.take_profit_2,
            trailing_stop_rule=trade_setup.trailing_stop_rule,
            thesis_score=best.total_score,
            hold_days=0,
            last_monitored_at=datetime.now(tz=timezone.utc),
        )

        async with services.db.session() as session:
            session.add(orm_decision)
            session.add(position_orm)

        log.info(
            "scan_persisted_decision_and_position",
            decision_id=decision_id,
            ticker=ticker,
        )
    except Exception as exc:
        log.error("scan_step_17_persist_error", ticker=ticker, error=str(exc))
        # Non-fatal: order was already placed, log and continue

    # ------------------------------------------------------------------
    # Step 18: Publish events
    # ------------------------------------------------------------------
    log.debug("scan_step_18_publish_events", ticker=ticker)
    try:
        duration = time.monotonic() - scan_start
        await event_bus.publish(
            ScanCompletedEvent(
                cycle_id=cycle_id,
                candidates_found=len(screened),
                selected_ticker=ticker,
                is_no_trade=False,
                duration_seconds=round(duration, 3),
            )
        )
        await event_bus.publish(
            TradeOpenedEvent(
                trade_id=decision_id,
                cycle_id=cycle_id,
                ticker=ticker,
                entry_price=trade_setup.entry_high,
                quantity=position_size.shares,
                stop_loss=trade_setup.stop_loss,
                take_profit_1=trade_setup.take_profit_1,
            )
        )
    except Exception as exc:
        log.error("scan_step_18_events_error", error=str(exc))

    # ------------------------------------------------------------------
    # Step 19 + 20: Update metrics
    # ------------------------------------------------------------------
    log.debug("scan_step_19_20_update_metrics", ticker=ticker)
    try:
        METRICS.scan_cycles_total.labels(status="success").inc()
        METRICS.trades_opened_total.labels(ticker=ticker).inc()
        METRICS.orders_submitted_total.labels(side="buy", order_type="market").inc()

        # Update portfolio gauges from latest broker account state
        try:
            account = await services.broker.get_account()
            portfolio_val = float(getattr(account, "portfolio_value", 0) or 0)
            cash_bal = float(getattr(account, "cash", 0) or 0)
            if portfolio_val > 0:
                METRICS.portfolio_value.set(portfolio_val)
            if cash_bal >= 0:
                METRICS.cash_balance.set(cash_bal)
        except Exception:
            pass  # metric update failure is non-fatal

        # Active positions count
        try:
            positions = await services.broker.get_positions()
            METRICS.active_positions.set(len(positions))
        except Exception:
            pass

        log.info(
            "scan_cycle_trade_opened",
            ticker=ticker,
            shares=position_size.shares,
            dollar_size=round(position_size.dollar_size, 2),
            stop_loss=round(trade_setup.stop_loss, 2),
            take_profit_1=round(trade_setup.take_profit_1, 2),
            cycle_id=cycle_id,
        )
    except Exception as exc:
        log.error("scan_step_20_metrics_error", error=str(exc))
