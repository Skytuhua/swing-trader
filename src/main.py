"""
SwingTrader — application entry point.

Usage
-----
    python -m src.main [--mode paper|live] [--config path/to/config.yaml]

Start-up sequence
-----------------
 1. Parse CLI arguments
 2. Load YAML configuration (merging default + mode-specific overrides)
 3. Configure structlog structured logging
 4. Initialise Prometheus build-info metric
 5. Initialise DatabaseManager (create tables if needed)
 6. Initialise RedisManager
 7. Create all service instances (dependency-injection style)
 8. Register indicator plugins (import triggers @register decorators)
 9. Create TradingScheduler and register all jobs
10. Start FastAPI admin API in background via uvicorn
11. Start APScheduler
12. Register SIGINT / SIGTERM handlers for graceful shutdown
13. Log startup banner
14. Enter asyncio event loop (awaits shutdown signal)

Shutdown sequence
-----------------
 1. Scheduler.shutdown(wait=True)   – let running jobs finish
 2. Uvicorn graceful stop
 3. DatabaseManager.close()
 4. RedisManager.close()
 5. Exit 0

Environment variables (all optional, override config file)
----------------------------------------------------------
    ALPACA_API_KEY       Alpaca API key
    ALPACA_API_SECRET    Alpaca API secret
    DB_URL               Override database.url
    REDIS_URL            Override redis.url
    FINNHUB_API_KEY      Finnhub API key
    LOG_LEVEL            Override app.log_level
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

import structlog
import uvicorn
import yaml

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(mode: str, extra_config: str | None = None) -> dict[str, Any]:
    """Load and merge YAML configuration files.

    Merge order (later overrides earlier):
      1. config/default.yaml
      2. config/{mode}.yaml  (paper.yaml or live.yaml)
      3. extra_config (CLI --config path)
      4. Environment variable overrides

    Returns
    -------
    dict
        Flat merged configuration dict.
    """
    config_dir = Path(__file__).parent.parent / "config"

    # 1. Default config
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    with open(default_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # 2. Mode-specific overrides
    mode_path = config_dir / f"{mode}.yaml"
    if mode_path.exists():
        with open(mode_path) as f:
            mode_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, mode_cfg)

    # 3. Extra config path from CLI
    if extra_config:
        extra_path = Path(extra_config)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra config not found: {extra_config}")
        with open(extra_path) as f:
            extra_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, extra_cfg)

    # 4. Environment variable overrides
    if os.environ.get("DB_URL"):
        cfg.setdefault("database", {})["url"] = os.environ["DB_URL"]
    if os.environ.get("REDIS_URL"):
        cfg.setdefault("redis", {})["url"] = os.environ["REDIS_URL"]
    if os.environ.get("LOG_LEVEL"):
        cfg.setdefault("app", {})["log_level"] = os.environ["LOG_LEVEL"]
    if os.environ.get("ALPACA_API_KEY"):
        cfg.setdefault("broker", {})["api_key"] = os.environ["ALPACA_API_KEY"]
    if os.environ.get("ALPACA_API_SECRET"):
        cfg.setdefault("broker", {})["api_secret"] = os.environ["ALPACA_API_SECRET"]
    if os.environ.get("FINNHUB_API_KEY"):
        cfg.setdefault("news", {}).setdefault("api_keys", {})["finnhub"] = os.environ["FINNHUB_API_KEY"]

    # Override mode from CLI
    cfg.setdefault("app", {})["mode"] = mode

    return cfg


class _Config:
    """Thin object wrapper around the raw config dict providing dot-access."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        # Top-level sub-configs as attribute namespaces
        for key, value in raw.items():
            object.__setattr__(self, key, _ConfigNamespace(value) if isinstance(value, dict) else value)
        self.version: str = raw.get("app", {}).get("config_version", _VERSION)

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)


class _ConfigNamespace:
    """Recursive dot-access namespace over a config dict."""

    def __init__(self, raw: dict[str, Any]) -> None:
        for key, value in raw.items():
            object.__setattr__(
                self,
                key,
                _ConfigNamespace(value) if isinstance(value, dict) else value,
            )

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------


def _build_services(cfg: _Config) -> "Any":  # returns ServiceContainer
    """Instantiate all services using the merged configuration.

    Dependency injection is performed manually: each service receives only
    what it needs, via its ``__init__`` parameters.
    """
    from src.services.orchestrator.scan_cycle import ServiceContainer
    from src.core.database import DatabaseManager
    from src.core.redis_client import RedisManager

    # ---- Infrastructure ----
    # DatabaseManager and RedisManager accept a Settings-like object.
    # We pass the raw namespace since both modules do get_settings() fallback.
    db = DatabaseManager()       # reads DB_URL / settings internally
    redis = RedisManager()       # reads REDIS_URL / settings internally

    # ---- Broker ----
    mode = cfg.app.mode if hasattr(cfg, "app") else "paper"
    broker_cfg = getattr(cfg, "broker", _ConfigNamespace({}))

    if mode == "live":
        from src.services.execution.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(config=broker_cfg)
    else:
        from src.services.execution.paper_broker import PaperBroker
        initial_cash = float(getattr(broker_cfg, "initial_cash", 100_000))
        broker = PaperBroker(initial_cash=initial_cash)

    # ---- Market data ----
    mktdata_cfg = getattr(cfg, "market_data", _ConfigNamespace({}))
    news_cfg = getattr(cfg, "news", _ConfigNamespace({}))

    # Primary: Alpaca data provider
    alpaca_key = getattr(broker_cfg, "api_key", os.environ.get("ALPACA_API_KEY", ""))
    alpaca_secret = getattr(broker_cfg, "api_secret", os.environ.get("ALPACA_API_SECRET", ""))
    paper_mode = mode != "live"

    from src.services.market_data.alpaca_data import AlpacaDataProvider
    primary_data = AlpacaDataProvider(
        api_key=alpaca_key,
        api_secret=alpaca_secret,
        paper=paper_mode,
    )

    # Fallback: yfinance
    from src.services.market_data.yfinance_data import YFinanceDataProvider
    fallback_data = YFinanceDataProvider()

    from src.services.market_data.manager import DataManager
    # Pass the RedisManager instance itself (not redis._client, which is None
    # until redis.init() is called during the startup sequence in _async_main).
    # DataManager should call redis_manager.client lazily on first cache access,
    # by which point init() will have completed.  Passing None here is safe:
    # the _async_main startup sequence replaces services.redis with the
    # already-initialised RedisManager after _build_services() returns.
    data_manager = DataManager(
        primary=primary_data,
        fallback=fallback_data,
        redis_client=None,  # Replaced post-init via services.redis in _async_main
    )

    # ---- Technical analysis ----
    from src.services.technical.registry import IndicatorRegistry
    from src.services.technical.engine import TechnicalEngine

    # Import indicators to trigger @register decorators
    import src.services.technical.indicators  # noqa: F401 – side-effect import

    technical_cfg = getattr(cfg, "technical", _ConfigNamespace({}))
    technical_engine = TechnicalEngine(registry=IndicatorRegistry, config=technical_cfg)

    # ---- News services ----
    finnhub_key = (
        getattr(getattr(news_cfg, "api_keys", _ConfigNamespace({})), "finnhub", None)
        or os.environ.get("FINNHUB_API_KEY", "")
    )
    from src.core.config import NewsConfig as CoreNewsConfig
    core_news_cfg = CoreNewsConfig(
        providers=getattr(news_cfg, "providers", ["finnhub"]),
        api_keys={"finnhub": finnhub_key},
        finbert_model=getattr(news_cfg, "finbert_model", "ProsusAI/finbert"),
    )

    from src.services.news.finnhub_news import FinnhubNewsProvider
    news_provider = FinnhubNewsProvider(config=core_news_cfg)

    from src.services.news.processor import NewsProcessor
    news_processor = NewsProcessor(
        model_name=getattr(news_cfg, "finbert_model", "ProsusAI/finbert"),
    )

    from src.services.news.scorer import NewsScorer
    news_scorer = NewsScorer()

    from src.services.news.deduplicator import NewsDeduplicator
    news_deduplicator = NewsDeduplicator(
        threshold=float(getattr(news_cfg, "dedup_similarity_threshold", 0.85)),
    )

    # ---- Sentiment services ----
    sentiment_cfg = getattr(cfg, "sentiment", _ConfigNamespace({}))

    from src.services.sentiment.aggregator import SentimentAggregator
    sentiment_aggregator = SentimentAggregator(providers=[])  # providers added below

    from src.services.sentiment.scorer import SentimentScorer
    sentiment_scorer = SentimentScorer()

    # ---- Regime engine ----
    from src.services.regime.engine import RegimeEngine
    regime_engine = RegimeEngine()

    # ---- Risk calendar ----
    from src.services.risk.calendar import MarketCalendar
    calendar = MarketCalendar()

    # ---- Kill switch ----
    from src.services.risk.kill_switch import KillSwitch
    kill_switch = KillSwitch(redis=redis, broker=broker)

    # ---- Risk engine ----
    risk_cfg = getattr(cfg, "risk", _ConfigNamespace({}))

    from src.services.risk.engine import RiskEngine
    risk_engine = RiskEngine(
        config=risk_cfg,
        kill_switch=kill_switch,
        calendar=calendar,
        data_manager=data_manager,
        broker=broker,
    )

    # ---- Universe filter ----
    universe_cfg = getattr(cfg, "universe", _ConfigNamespace({}))

    from src.services.pipeline.universe import UniverseFilter
    universe_filter = UniverseFilter(config=universe_cfg, data_manager=data_manager)

    # ---- Screener ----
    from src.services.pipeline.screener import MultiStageScreener
    screener = MultiStageScreener(
        data_manager=data_manager,
        technical_engine=technical_engine,
        news_scorer=news_scorer,
        sentiment_scorer=sentiment_scorer,
    )

    # ---- Scoring & ranking ----
    scoring_cfg = getattr(cfg, "scoring", _ConfigNamespace({}))

    from src.services.scoring.engine import ScoringEngine
    scoring_engine = ScoringEngine(weights=scoring_cfg)

    from src.services.pipeline.ranker import CandidateRanker
    ranker = CandidateRanker(scoring_engine=scoring_engine)

    from src.services.pipeline.selector import FinalSelector
    selector = FinalSelector(config=scoring_cfg)

    # ---- Trade construction ----
    from src.services.trade.constructor import TradeConstructor
    trade_constructor = TradeConstructor()

    position_cfg = getattr(cfg, "position", _ConfigNamespace({}))
    from src.services.trade.sizer import PositionSizer
    position_sizer = PositionSizer(risk_config=risk_cfg)

    from src.services.trade.validator import PreTradeValidator
    trade_validator = PreTradeValidator()

    # ---- Order manager ----
    from src.services.execution.order_manager import OrderManager
    order_manager = OrderManager(broker=broker)

    # ---- Monitor services ----
    from src.services.monitor.alert_manager import AlertManager
    alert_manager = AlertManager()

    from src.services.monitor.exit_engine import ExitEngine
    exit_engine = ExitEngine(
        data_manager=data_manager,
        regime_engine=regime_engine,
        news_scorer=news_scorer,
        scoring_engine=scoring_engine,
        technical_engine=technical_engine,
    )

    from src.services.monitor.position_monitor import PositionMonitor
    position_monitor = PositionMonitor(
        broker=broker,
        data=data_manager,
        exit_engine=exit_engine,
        alert_manager=alert_manager,
    )

    # ---- Assemble ServiceContainer ----
    return ServiceContainer(
        config=cfg,
        db=db,
        redis=redis,
        data_manager=data_manager,
        technical_engine=technical_engine,
        news_provider=news_provider,
        news_processor=news_processor,
        news_scorer=news_scorer,
        news_deduplicator=news_deduplicator,
        sentiment_aggregator=sentiment_aggregator,
        sentiment_scorer=sentiment_scorer,
        regime_engine=regime_engine,
        universe_filter=universe_filter,
        screener=screener,
        ranker=ranker,
        selector=selector,
        scoring_engine=scoring_engine,
        trade_constructor=trade_constructor,
        position_sizer=position_sizer,
        trade_validator=trade_validator,
        order_manager=order_manager,
        broker=broker,
        position_monitor=position_monitor,
        exit_engine=exit_engine,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
        calendar=calendar,
        trading_mode=mode,
    )


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def _print_banner(mode: str, cfg: _Config, log: structlog.stdlib.BoundLogger) -> None:
    """Log a structured startup banner."""
    log.info(
        "SWINGTRADER_STARTUP",
        version=_VERSION,
        mode=mode,
        config_version=getattr(cfg, "version", "unknown"),
        log_level=getattr(getattr(cfg, "app", object()), "log_level", "INFO"),
        api_port=getattr(getattr(cfg, "app", object()), "api_port", 8000),
        python_version=sys.version.split()[0],
    )


# ---------------------------------------------------------------------------
# FastAPI startup
# ---------------------------------------------------------------------------


def _start_api_server(cfg: _Config) -> "uvicorn.Server":
    """Create and return a uvicorn.Server for the FastAPI admin API.

    The caller is responsible for running it in a background task.
    """
    api_host = getattr(getattr(cfg, "app", object()), "api_host", "0.0.0.0")
    api_port = int(getattr(getattr(cfg, "app", object()), "api_port", 8000))
    log_level = getattr(getattr(cfg, "app", object()), "log_level", "INFO").lower()

    try:
        from src.api.app import create_app
        app = create_app()
    except ImportError:
        # API module not yet built – serve a minimal health-check app
        from fastapi import FastAPI
        app = FastAPI(title="SwingTrader", version=_VERSION)

        @app.get("/health")
        async def health():
            return {"status": "ok", "version": _VERSION}

    uv_config = uvicorn.Config(
        app=app,
        host=api_host,
        port=api_port,
        log_level=log_level,
        access_log=False,      # structlog handles logging
        loop="none",           # use the existing asyncio loop
    )
    return uvicorn.Server(config=uv_config)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


async def _graceful_shutdown(
    services: Any,
    scheduler: Any,
    api_server: "uvicorn.Server | None",
    log: structlog.stdlib.BoundLogger,
    shutdown_event: asyncio.Event,
) -> None:
    """Shutdown sequence: scheduler → API → DB → Redis."""
    log.info("graceful_shutdown_start")

    # 1. Signal the main loop to stop
    shutdown_event.set()

    # 2. Pause the scheduler immediately (no new jobs)
    try:
        scheduler.pause()
        log.info("shutdown_scheduler_paused")
    except Exception as exc:
        log.error("shutdown_scheduler_pause_error", error=str(exc))

    # 3. Wait for running jobs (up to 30 s)
    try:
        scheduler.shutdown(wait=True)
        log.info("shutdown_scheduler_stopped")
    except Exception as exc:
        log.error("shutdown_scheduler_stop_error", error=str(exc))

    # 4. Stop uvicorn API server
    if api_server is not None:
        try:
            api_server.should_exit = True
            log.info("shutdown_api_server_stopped")
        except Exception as exc:
            log.error("shutdown_api_stop_error", error=str(exc))

    # 5. Close database
    try:
        await services.db.close()
        log.info("shutdown_database_closed")
    except Exception as exc:
        log.error("shutdown_db_close_error", error=str(exc))

    # 6. Close Redis
    try:
        await services.redis.close()
        log.info("shutdown_redis_closed")
    except Exception as exc:
        log.error("shutdown_redis_close_error", error=str(exc))

    log.info("graceful_shutdown_complete")


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def _async_main(args: argparse.Namespace) -> int:
    """Full async lifecycle: init → run → shutdown."""

    # ---- 1. Load config ----
    try:
        raw_cfg = _load_config(mode=args.mode, extra_config=args.config)
        cfg = _Config(raw_cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ---- 2. Setup logging ----
    from src.core.logging_config import configure_logging
    log_level = getattr(getattr(cfg, "app", object()), "log_level", "INFO")
    configure_logging(log_level=log_level)
    log = structlog.get_logger(__name__)

    # ---- 3. Prometheus build-info ----
    from src.core.metrics import init_build_info
    init_build_info(version=_VERSION, mode=args.mode)

    _print_banner(mode=args.mode, cfg=cfg, log=log)

    # ---- 4. Initialise database ----
    log.info("startup_db_init")
    services = None
    try:
        from src.core.database import DatabaseManager
        db = DatabaseManager()
        await db.init()
        try:
            await db.init_db()   # CREATE TABLE IF NOT EXISTS
        except Exception as exc:
            log.warning("startup_db_create_tables_error", error=str(exc))
            # Non-fatal: tables may already exist
    except Exception as exc:
        log.error("startup_db_init_error", error=str(exc))
        # Non-fatal: allow startup to continue for paper trading without DB

    # ---- 5. Initialise Redis ----
    log.info("startup_redis_init")
    try:
        from src.core.redis_client import RedisManager
        redis_mgr = RedisManager()
        await redis_mgr.init()
    except Exception as exc:
        log.warning("startup_redis_init_error", error=str(exc))
        # Non-fatal: redis is used for caching, not a hard requirement

    # ---- 6. Build all services (DI) ----
    log.info("startup_services_init")
    try:
        services = _build_services(cfg)
        # Re-use the already-initialised DB and Redis managers
        try:
            services.db = db
        except Exception:
            pass
        try:
            services.redis = redis_mgr
        except Exception:
            pass
        # Wire the initialised RedisManager into DataManager now that
        # redis_mgr.init() has completed and redis_mgr._client is valid.
        try:
            if hasattr(services, "data_manager") and hasattr(redis_mgr, "_client"):
                dm = services.data_manager
                if hasattr(dm, "redis_client"):
                    dm.redis_client = redis_mgr.client
                elif hasattr(dm, "_redis"):
                    dm._redis = redis_mgr.client
        except Exception:
            pass  # Non-fatal: DataManager operates without cache if Redis unavailable
        log.info("startup_services_ready")
    except Exception as exc:
        log.error("startup_services_init_error", error=str(exc), exc_info=True)
        return 2

    # ---- 7. Initialise broker ----
    log.info("startup_broker_init")
    try:
        if hasattr(services.broker, "initialize"):
            await services.broker.initialize()
        log.info("startup_broker_ready", mode=args.mode)
    except Exception as exc:
        log.warning("startup_broker_init_error", error=str(exc))
        # Non-fatal for paper trading

    # ---- 8. Register indicator plugins ----
    log.info("startup_indicator_plugins")
    try:
        import src.services.technical.indicators  # noqa: F401
        from src.services.technical.registry import IndicatorRegistry
        registered_count = len(getattr(IndicatorRegistry, "_registry", {}))
        log.info("startup_indicators_registered", count=registered_count)
    except Exception as exc:
        log.warning("startup_indicator_plugins_error", error=str(exc))

    # ---- 9. Create scheduler ----
    log.info("startup_scheduler_init")
    from src.services.orchestrator.scheduler import create_scheduler

    # Parse scan cron from config if provided
    schedule_cfg = getattr(cfg, "schedule", None)
    scan_cron: dict[str, Any] | None = None
    if schedule_cfg:
        scan_cron_str = getattr(schedule_cfg, "scan_cron", None)
        if scan_cron_str:
            # Accept both cron string and pre-built dict
            if isinstance(scan_cron_str, str):
                # Parse "min hour day month dow" -> APScheduler dict
                parts = scan_cron_str.strip().split()
                if len(parts) == 5:
                    scan_cron = {
                        "minute": parts[0],
                        "hour": parts[1],
                        "day": parts[2],
                        "month": parts[3],
                        "day_of_week": parts[4],
                        "timezone": "America/New_York",
                    }
            elif isinstance(scan_cron_str, dict):
                scan_cron = scan_cron_str

    scheduler = create_scheduler(services=services, scan_cron=scan_cron)

    # ---- 10. Start FastAPI server in background ----
    log.info("startup_api_server")
    api_server: uvicorn.Server | None = None
    api_task: asyncio.Task | None = None
    try:
        api_server = _start_api_server(cfg)
        api_task = asyncio.create_task(api_server.serve(), name="api_server")
        log.info(
            "startup_api_server_started",
            host=api_server.config.host,
            port=api_server.config.port,
        )
    except Exception as exc:
        log.warning("startup_api_server_error", error=str(exc))

    # ---- 11. Start APScheduler ----
    log.info("startup_scheduler_start")
    scheduler.start()

    # ---- 12. Signal handlers ----
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals, _frame: FrameType | None = None) -> None:
        log.warning("shutdown_signal_received", signal=sig.name)
        asyncio.ensure_future(
            _graceful_shutdown(
                services=services,
                scheduler=scheduler,
                api_server=api_server,
                log=log,
                shutdown_event=shutdown_event,
            )
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except (ValueError, OSError):
            # Windows / some environments don't support add_signal_handler
            signal.signal(sig, _handle_signal)

    log.info(
        "swingtrader_running",
        mode=args.mode,
        version=_VERSION,
        jobs=scheduler.get_job_states(),
    )

    # ---- 13. Wait for shutdown signal ----
    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        log.info("keyboard_interrupt")
        await _graceful_shutdown(
            services=services,
            scheduler=scheduler,
            api_server=api_server,
            log=log,
            shutdown_event=shutdown_event,
        )

    # Cancel background API task if still running
    if api_task and not api_task.done():
        api_task.cancel()
        try:
            await api_task
        except asyncio.CancelledError:
            pass

    log.info("swingtrader_exited_cleanly")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="swingtrader",
        description="Autonomous swing-trading bot for US equities.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=os.environ.get("TRADING_MODE", "paper"),
        help="Trading mode: paper (default) or live.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to an additional YAML config file to merge on top of defaults.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"swingtrader {_VERSION}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Synchronous entry point called by ``python -m src.main``."""
    args = _parse_args()

    # Warn loudly about live mode
    if args.mode == "live":
        print(
            "\n"
            "  ⚠  WARNING: LIVE TRADING MODE ACTIVE\n"
            "  Real money will be placed at risk.\n"
            "  Press Ctrl-C within 5 seconds to abort.\n",
            file=sys.stderr,
        )
        import time
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("Aborted.", file=sys.stderr)
            sys.exit(0)

    try:
        exit_code = asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
