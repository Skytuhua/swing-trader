# SwingTrader

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Trading Mode: Paper](https://img.shields.io/badge/Trading%20Mode-Paper-yellow)
![Status: Beta](https://img.shields.io/badge/Status-Beta-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)

**SwingTrader** is an autonomous U.S. equity swing trading bot that identifies, enters, manages, and exits medium-term stock positions (3–5 day hold, long-only) without human intervention. It combines multi-source market data, technical analysis, NLP-based news sentiment (FinBERT), social sentiment signals, and a transparent weighted scoring model to make systematic trading decisions.

> ⚠️ **RISK WARNING:** This software is for **educational and research purposes only**. It is **not financial advice**. Trading involves a substantial risk of loss. Always paper-trade first. Read the full disclaimer at the bottom of this file.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Features](#features)
4. [Quick Start](#quick-start)
5. [Configuration Guide](#configuration-guide)
6. [System Architecture Details](#system-architecture-details)
   - [Market Data Pipeline](#market-data-pipeline)
   - [Technical Analysis Engine](#technical-analysis-engine)
   - [News Analysis Engine (FinBERT)](#news-analysis-engine-finbert)
   - [Social Sentiment Engine](#social-sentiment-engine)
   - [Market Regime Classification](#market-regime-classification)
   - [Candidate Pipeline (Multi-Stage)](#candidate-pipeline-multi-stage)
   - [Scoring Model](#scoring-model)
   - [Trade Construction](#trade-construction)
   - [Position Sizing](#position-sizing)
   - [Execution Engine](#execution-engine)
   - [Position Monitoring and Exit Logic](#position-monitoring-and-exit-logic)
   - [Risk Controls and Kill Switch](#risk-controls-and-kill-switch)
7. [Trading Decision Object](#trading-decision-object)
8. [No-Trade Logic](#no-trade-logic)
9. [Paper vs Live Trading](#paper-vs-live-trading)
10. [Backtesting](#backtesting)
11. [API Reference](#api-reference)
12. [Monitoring and Observability](#monitoring-and-observability)
13. [Deployment (Docker Compose)](#deployment-docker-compose)
14. [Development Guide](#development-guide)
15. [Testing](#testing)
16. [Project Structure](#project-structure)
17. [License](#license)
18. [Disclaimer](#disclaimer)

---

## Overview

SwingTrader implements a systematic swing-trading strategy with these core principles:

- **Timeframe:** 3–5 trading day position holds (never intraday scalping)
- **Direction:** Long-only (no short selling or options)
- **Universe:** Top 200 liquid U.S. equities (S&P 500 components by default)
- **Signal generation:** Multi-factor — technical + news catalyst + social sentiment + market regime
- **Decision transparency:** Every trade produces a fully explainable `TradingDecision` object with scores, reasoning, and entry/exit levels
- **Risk-first:** Multiple overlapping risk controls including per-trade risk limits, daily loss limits, drawdown-based kill switch, and an emergency manual kill switch
- **Autonomous:** APScheduler orchestrates all cycles (scan, monitor, daily report) — no human input required during operation

The bot runs 24/7. Trading activity is gated to U.S. market hours (9:30 AM – 4:00 PM ET, NYSE trading days).

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SCHEDULER / ORCHESTRATOR                        │
│   (APScheduler: scan cycle 9:29 ET, monitor every 5 min, reports)    │
├──────────┬───────────┬───────────┬───────────┬───────────┬───────────┤
│ Market   │ Technical │ News      │ Sentiment │ Regime    │ Scoring   │
│ Data     │ Analysis  │ Analysis  │ Analysis  │ Engine    │ & Ranking │
│ Service  │ Engine    │ Engine    │ Engine    │           │ Engine    │
│ (Alpaca  │ (pandas-  │ (Finnhub  │ (Reddit   │ (SPY EMA  │ (Weighted │
│  Finnhub │  ta, TA-  │  NewsAPI  │  StockTwi │  VIX      │  6-factor │
│  yfinance│  Lib)     │  FinBERT) │  ts)      │  Breadth) │  model)   │
├──────────┴───────────┴───────────┴───────────┴───────────┴───────────┤
│                        CANDIDATE PIPELINE                              │
│   Universe → Liquidity Filter → Regime Gate → Technical Screen        │
│   → News Screen → Sentiment Screen → Score → Rank → SELECT or         │
│   → NO TRADE                                                           │
├──────────────────────────────────────────────────────────────────────┤
│                        TRADE CONSTRUCTION                              │
│   Entry zone (ATR band), Stop loss (ATR mult), Take profit (ATR mult) │
│   Position sizing (fixed-fraction risk), Order validation              │
├──────────────────────────────────────────────────────────────────────┤
│                        EXECUTION ENGINE                                │
│   Broker Adapter (Alpaca Paper/Live) → Order Manager                  │
│   → Fill Reconciliation → Position ledger update                       │
├──────────────────────────────────────────────────────────────────────┤
│                        POSITION MONITOR                                │
│   Real-time price → Stop check → TP check → Trailing stop             │
│   → News shock exit → Time stop (day 5/7) → Thesis deterioration      │
│   → Emergency exit on kill switch                                      │
├──────────────────────────────────────────────────────────────────────┤
│                        RISK ENGINE                                     │
│   Kill switch │ Max drawdown │ Daily loss │ Stale data │ Circuit break │
├──────────────────────────────────────────────────────────────────────┤
│  PostgreSQL 16  │  Redis 7  │  FastAPI Admin API  │  Structured Logs  │
│  (ORM models)   │  (cache   │  (port 8000)        │  (structlog +     │
│  (Alembic)      │   pub/sub)│  Prometheus :9090   │   Prometheus)     │
└─────────────────┴───────────┴─────────────────────┴───────────────────┘
```

---

## Features

SwingTrader is built around 20 non-negotiable requirements:

| # | Feature | Implementation |
|---|---------|---------------|
| 1 | Multi-source market data with fallback | Alpaca → Finnhub → yfinance cascade |
| 2 | Full technical indicator suite | EMA, RSI, MACD, Bollinger Bands, ATR, ADX, VWAP, OBV, Stochastic, Ichimoku, Donchian, pivot points |
| 3 | Pluggable indicator registry | `src/services/technical/registry.py` — add indicators without changing engine code |
| 4 | FinBERT NLP news sentiment | `ProsusAI/finbert` model; Hugging Face transformers; per-headline scoring |
| 5 | Multi-source news with deduplication | Finnhub + NewsAPI; cosine-similarity deduplication prevents double-counting |
| 6 | Social sentiment signals | Reddit (wallstreetbets, stocks) + StockTwits; configurable subreddits |
| 7 | Market regime classification | SPY EMA + VIX + market breadth; blocks entries in bear regimes |
| 8 | Multi-stage candidate pipeline | Universe → Liquidity → Regime → Technical → News → Sentiment → Score → Rank |
| 9 | Transparent weighted scoring model | 6 components, configurable weights; `conservative`, `moderate`, `aggressive` profiles |
| 10 | Score explainer | Every decision includes a human-readable explanation of the score breakdown |
| 11 | NO TRADE logic | Explicit `no_trade_reason` when no setup meets the threshold |
| 12 | ATR-based trade construction | Entry, stop, TP, trailing stop all sized relative to ATR for volatility normalisation |
| 13 | Fixed-fraction position sizing | Risk at most N% of portfolio per trade; Kelly fraction optionally applied |
| 14 | Alpaca broker integration | Market + limit orders, fractional shares, fill reconciliation |
| 15 | Paper trading mode | Full paper broker simulation with realistic fill modelling |
| 16 | Autonomous position monitoring | 5-minute cycle; handles stops, TPs, time stops, news shocks, thesis decay |
| 17 | Kill switch | Immediate halt of all new entries; configurable via API and env var |
| 18 | Multi-layer risk controls | Per-trade, daily, total, drawdown, circuit breaker |
| 19 | FastAPI admin API | Full REST API for positions, orders, decisions, risk, config, reports |
| 20 | Full observability | structlog JSON logging + Prometheus metrics + daily reports |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for Docker Compose deployment)
- Alpaca brokerage account — [sign up at alpaca.markets](https://alpaca.markets/) (paper account is free)
- Finnhub API key — [sign up at finnhub.io](https://finnhub.io/) (free tier)
- NewsAPI key — [sign up at newsapi.org](https://newsapi.org/) (free tier)

### 5-Step Setup

**Step 1: Clone the repository and set up the environment**

```bash
git clone https://github.com/example/swing-trader.git
cd swing-trader

# Create virtualenv and install dependencies
make setup
source .venv/bin/activate
```

**Step 2: Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
nano .env
```

Minimum required variables for paper trading:
```bash
TRADING_MODE=paper
ALPACA_API_KEY=your_paper_api_key
ALPACA_API_SECRET=your_paper_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key
DATABASE_URL=postgresql+asyncpg://swing:swing@localhost:5432/swingtrader
REDIS_URL=redis://localhost:6379/0
API_KEY=generate_a_strong_random_key
```

**Step 3: Start infrastructure (PostgreSQL + Redis)**

```bash
# Start only the database services
docker compose up -d postgres redis

# Wait for them to be healthy
docker compose ps
```

**Step 4: Initialise the database and seed the universe**

```bash
make db-init          # Create all tables
make db-migrate       # Apply Alembic migrations
make seed-universe    # Seed S&P 500 stock universe
```

**Step 5: Start the bot**

```bash
# Paper trading mode (recommended for first run)
make run

# Or via Docker Compose (runs everything including app)
make docker-up
```

The FastAPI admin API is now available at `http://localhost:8000`.
API docs (Swagger UI) are at `http://localhost:8000/docs` (paper mode only).

---

## Configuration Guide

SwingTrader uses a layered YAML configuration system:

```
config/default.yaml       Base configuration (all values with defaults)
config/paper.yaml         Paper-trading overrides (applied when TRADING_MODE=paper)
config/live.yaml          Live-trading overrides (applied when TRADING_MODE=live)
config/universe.yaml      Tradable stock universe (generated by seed_universe.py)
```

The active config at runtime is `default.yaml` merged with the environment overlay. Settings in the overlay take precedence.

### Key Configuration Sections

```yaml
# config/default.yaml (excerpt)

universe:
  min_price: 10.0                   # Minimum stock price in USD
  min_market_cap: 1_000_000_000     # $1B minimum market cap
  min_avg_dollar_volume: 5_000_000  # $5M average daily dollar volume

scoring:
  weight_profile: moderate          # conservative | moderate | aggressive
  weights:
    technical: 0.30                 # Technical analysis composite
    news: 0.15                      # News catalyst score
    sentiment: 0.10                 # Social sentiment score
    liquidity: 0.10                 # Liquidity quality
    regime: 0.15                    # Market regime alignment
    risk_reward: 0.20               # Risk-reward quality
  no_trade_threshold: 55.0          # Minimum score to generate a trade
  confidence_threshold: 60.0        # Minimum confidence to submit an order
  min_risk_reward: 1.5              # Minimum R:R ratio required

risk:
  max_position_pct: 0.20            # Max 20% of portfolio in one position
  max_risk_per_trade_pct: 2.0       # Risk at most 2% per trade
  max_total_risk_pct: 6.0           # Total open risk cap: 6%
  max_daily_loss_pct: 3.0           # Halt after 3% intraday loss
  max_drawdown_pct: 10.0            # Kill switch at 10% drawdown

position:
  max_hold_days: 7                  # Close by day 7 (time stop)
  stop_loss_atr_mult: 2.0           # Stop = entry - 2 × ATR(14)
  take_profit_atr_mult: 2.0         # TP = entry + 2 × ATR(14)  → 1:1 R:R
  trailing_stop_atr_mult: 1.5       # Trail at 1.5 × ATR once TP1 hit
```

### Switching Between Paper and Live

Change the `TRADING_MODE` environment variable:

```bash
# Paper (default, safe for testing)
TRADING_MODE=paper

# Live (real money — use extreme caution)
TRADING_MODE=live
```

The corresponding YAML overlay (`paper.yaml` or `live.yaml`) is automatically merged. The live overlay enforces stricter risk controls by default.

### Adding Custom Indicators

The technical analysis engine uses a plugin registry. To add a new indicator:

```python
# src/services/technical/indicators/my_indicator.py
from src.services.technical.registry import IndicatorPlugin, indicator_registry

@indicator_registry.register("my_indicator")
class MyIndicator(IndicatorPlugin):
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        # df has columns: open, high, low, close, volume
        period = kwargs.get("period", 14)
        return df["close"].rolling(period).mean()  # example
```

The registry discovers all plugins at startup — no changes to the engine required.

---

## System Architecture Details

### Market Data Pipeline

**File:** `src/services/market_data/`

The data layer provides OHLCV bars, real-time quotes, and fundamental data through a three-tier fallback system:

1. **Primary: Alpaca** (`alpaca_data.py`) — fetches from the Alpaca Market Data API. Uses the IEX feed (free) or SIP feed (subscription). Supports bars from 1-minute to 1-day timeframes.

2. **Fallback 1: Finnhub** (`finnhub_data.py`) — used when Alpaca is unavailable. Also provides real-time quotes and fundamental metrics (market cap, earnings dates).

3. **Fallback 2: yfinance** (`yfinance_data.py`) — last-resort fallback using Yahoo Finance. Suitable for historical data; not recommended for real-time operation.

The **DataManager** (`manager.py`) orchestrates the cascade, caches results in Redis (configurable TTL), handles retries with exponential backoff (tenacity), and raises `StaleDataError` if no fresh data can be obtained within the configured threshold.

```
Market Data Request
       │
       ▼
   Redis Cache ──hit──▶ return cached data
       │ miss
       ▼
   Alpaca API ──success──▶ update cache ──▶ return
       │ failure
       ▼
   Finnhub API ──success──▶ update cache ──▶ return
       │ failure
       ▼
   yfinance ──success──▶ update cache ──▶ return
       │ failure
       ▼
   raise StaleDataError (trading halted)
```

### Technical Analysis Engine

**File:** `src/services/technical/`

The engine computes a comprehensive indicator suite over 1-day OHLCV bars (252-day lookback by default):

**Trend Indicators** (`indicators/trend.py`)
- EMA(8), EMA(21), EMA(200), SMA(50), SMA(200)
- MACD(12,26,9) with signal and histogram
- Ichimoku Cloud (9, 26, 52 periods)

**Momentum Indicators** (`indicators/momentum.py`)
- RSI(14) with oversold/overbought zones
- Stochastic Oscillator (14,3,3)
- Williams %R(14)
- CCI(20)
- Rate of Change (ROC)

**Volatility Indicators** (`indicators/volatility.py`)
- ATR(14) — used for stop/TP sizing
- Bollinger Bands(20, 2σ)
- Keltner Channels

**Volume Indicators** (`indicators/volume.py`)
- OBV with EMA(20)
- VWAP (intraday, resets daily)
- Volume SMA and surge detection
- Accumulation/Distribution Line

**Market Structure** (`indicators/structure.py`)
- Swing high/low detection
- Support and resistance levels
- Donchian Channels(20)
- Pivot points (prior day high/low/close)

Each indicator module is registered via the plugin registry and can be extended or replaced without modifying the core engine.

### News Analysis Engine (FinBERT)

**Files:** `src/services/news/`

The news engine performs four steps per scan cycle:

1. **Collection** — Fetches recent headlines (48h window) from Finnhub and NewsAPI for each candidate ticker.

2. **Deduplication** (`deduplicator.py`) — Uses cosine similarity of TF-IDF vectors to identify near-duplicate headlines from different sources. Duplicates are merged to avoid double-counting sentiment.

3. **NLP Scoring** (`processor.py`) — Runs each unique headline through `ProsusAI/finbert`, a BERT model fine-tuned on financial texts. FinBERT outputs three classes (positive, neutral, negative) as probability distributions. The impact score is `P(positive) - P(negative)`, ranging from −1 to +1.

4. **Catalyst Scoring** (`scorer.py`) — Aggregates headline-level scores into a ticker-level news score, weighted by headline recency (exponential decay) and source credibility. Identifies specific catalyst types: earnings beats, M&A, analyst upgrades, regulatory approvals, etc.

**News shock exits:** If a breaking headline scores below −0.6 while a position is open, the monitor triggers an immediate market exit regardless of the current price level.

### Social Sentiment Engine

**Files:** `src/services/sentiment/`

Two adapters collect social sentiment data:

- **Reddit** (`reddit.py`) — Reads posts and comments from r/wallstreetbets, r/stocks, r/investing, r/StockMarket. Filters by post age (24h), computes bull/bear ratios from flair and keywords.

- **StockTwits** (`stocktwits.py`) — Reads the StockTwits stream for each ticker symbol. Uses platform-native bull/bear sentiment labels plus keyword analysis.

The **Aggregator** (`aggregator.py`) normalises scores from both sources and produces a combined social sentiment score (−1 to +1) per ticker. Minimum post count thresholds prevent low-data signals from influencing decisions.

### Market Regime Classification

**File:** `src/services/regime/engine.py`

The regime engine classifies the overall market into three states before any candidate is evaluated:

| Regime | Condition | Action |
|--------|-----------|--------|
| BULL | SPY > EMA(200), VIX < 25, breadth > 55% | Trading allowed |
| NEUTRAL | Mixed signals | Trading allowed with reduced sizing |
| BEAR | SPY < EMA(200) OR VIX > 25 OR breadth < 40% | New entries blocked |

Regime is cached in Redis (1-hour TTL) and refreshed hourly during market hours. The regime label is included in every `TradingDecision` object.

### Candidate Pipeline (Multi-Stage)

**Files:** `src/services/pipeline/`

The pipeline reduces the full universe to a ranked shortlist through sequential gates:

```
Full Universe (200 tickers)
        │
        ▼
Stage 1: Liquidity Filter (universe.py)
  - min_price, min_market_cap, min_avg_dollar_volume
  - max_bid_ask_spread_pct, min_atr_pct
        │ ~150 tickers pass
        ▼
Stage 2: Regime Gate (screener.py)
  - Block all if BEAR regime and block_in_bear_regime=true
        │ ~150 tickers (0 if BEAR)
        ▼
Stage 3: Technical Screen (screener.py)
  - RSI not overbought (< 70)
  - Price above EMA(8) and EMA(21)  [trend alignment]
  - ADX > 20 (meaningful trend strength)
  - Volume surge in last 5 days
        │ ~30–50 tickers pass
        ▼
Stage 4: News Screen (screener.py)
  - At least one positive headline in 48h
  - No strongly negative news (score < -0.5)
        │ ~10–20 tickers pass
        ▼
Stage 5: Sentiment Screen (screener.py)
  - Social sentiment ≥ neutral (score ≥ 0)
  - Sufficient post volume (configurable min)
        │ ~5–10 tickers pass
        ▼
Stage 6: Scoring & Ranking (ranker.py)
  - Compute 6-component composite score
  - Sort descending by composite score
        │ top_n_candidates (default: 5)
        ▼
Stage 7: Selection (selector.py)
  - Apply no_trade_threshold gate
  - Apply confidence_threshold gate
  - Apply min_risk_reward gate
  → Selected ticker OR NO_TRADE
```

### Scoring Model

**Files:** `src/services/scoring/`

The composite score (0–100) is a weighted sum of six normalised sub-scores:

| Component | Default Weight | What It Measures |
|-----------|---------------|-----------------|
| `technical` | 30% | EMA alignment, RSI position, MACD momentum, ADX strength, Bollinger expansion |
| `news` | 15% | FinBERT sentiment average, catalyst type quality, headline recency |
| `sentiment` | 10% | Reddit + StockTwits bull/bear ratio, post volume |
| `liquidity` | 10% | Spread tightness, volume depth, dollar volume rank |
| `regime` | 15% | Alignment of the ticker with the broad market regime |
| `risk_reward` | 20% | Computed R:R ratio quality; higher R:R scores higher |

**Three built-in profiles** (`weights.py`):

```python
CONSERVATIVE = WeightProfile(
    name="conservative",
    technical_weight=0.35, news_weight=0.10, sentiment_weight=0.05,
    liquidity_weight=0.15, regime_weight=0.20, risk_reward_weight=0.15,
    confidence_threshold=70.0, no_trade_threshold=65.0, min_risk_reward=2.0,
)

MODERATE = WeightProfile(
    name="moderate",
    technical_weight=0.30, news_weight=0.15, sentiment_weight=0.10,
    liquidity_weight=0.10, regime_weight=0.15, risk_reward_weight=0.20,
    confidence_threshold=60.0, no_trade_threshold=55.0, min_risk_reward=1.5,
)

AGGRESSIVE = WeightProfile(
    name="aggressive",
    technical_weight=0.25, news_weight=0.20, sentiment_weight=0.15,
    liquidity_weight=0.10, regime_weight=0.10, risk_reward_weight=0.20,
    confidence_threshold=50.0, no_trade_threshold=45.0, min_risk_reward=1.2,
)
```

Every decision includes a score **explainer** (`explainer.py`) that produces a human-readable breakdown such as:

```
NVDA scored 73.4/100 (MODERATE profile):
  Technical   54.0 → 16.2  [EMA aligned, RSI=52 neutral, ADX=32 strong trend]
  News        82.0 → 12.3  [Positive earnings surprise, FinBERT avg=+0.71]
  Sentiment   67.0 →  6.7  [Reddit bullish ratio=0.68, 142 posts in 24h]
  Liquidity   91.0 →  9.1  [Spread=0.04%, vol $2.8B/day, rank 4/150]
  Regime      80.0 → 12.0  [BULL regime; SPY above 200d EMA]
  RiskReward  87.5 → 17.5  [R:R = 2.3:1, stop at $118.50, TP at $134.20]
  ──────────────────────────
  COMPOSITE:  73.8  (threshold: 55.0) → TRADE
```

### Trade Construction

**File:** `src/services/trade/constructor.py`

Once a ticker is selected, the trade constructor builds the full order specification:

1. **Entry zone:** Current ask price ± `entry_limit_offset_pct`. Limit orders are placed at `ask × (1 + offset)` to improve fill probability without chasing.

2. **ATR calculation:** `ATR(14)` from daily bars. All price distances are expressed in ATR multiples for volatility-normalised sizing.

3. **Stop loss:** `stop = entry - (atr × stop_loss_atr_mult)`. Default: `entry - 2 × ATR`.

4. **Take profit (TP1):** `tp1 = entry + (atr × take_profit_atr_mult)`. Default: `entry + 2 × ATR` → 1:1 R:R at minimum.

5. **Trailing stop:** Activates once TP1 is hit. Trails at `last_close - (atr × trailing_stop_atr_mult)`.

6. **Partial take profit:** Close `partial_take_profit_fraction` (default 50%) of the position at TP1; trail the remainder.

7. **Pre-trade validation** (`validator.py`): Checks portfolio buying power, existing position count, risk limits, and broker connectivity before submitting.

### Position Sizing

**File:** `src/services/trade/sizer.py`

Position size is calculated using the **fixed-fraction risk method**:

```
risk_per_share = entry_price - stop_price
max_dollar_risk = portfolio_equity × max_risk_per_trade_pct / 100
shares = floor(max_dollar_risk / risk_per_share)
position_value = shares × entry_price
```

An additional check ensures the position does not exceed `max_position_pct` of equity:

```
max_shares_by_pct = floor((portfolio_equity × max_position_pct) / entry_price)
shares = min(shares, max_shares_by_pct)
```

### Execution Engine

**Files:** `src/services/execution/`

Two broker adapters share a common abstract interface (`base.py`):

- **AlpacaBroker** (`alpaca_broker.py`) — wraps `alpaca-py`. Submits market or limit orders, polls for fills, handles partial fills and cancellations. Supports fractional share trading.

- **PaperBroker** (`paper_broker.py`) — internal simulation for paper mode. Models fills at the current mid-price with configurable slippage. Does not require an active brokerage connection.

The **OrderManager** (`order_manager.py`) maintains the lifecycle of each order (submitted → partially filled → filled → cancelled) and updates the position ledger. The **Reconciler** (`reconciler.py`) periodically cross-checks the internal ledger against the broker's reported positions and corrects any discrepancies.

### Position Monitoring and Exit Logic

**Files:** `src/services/monitor/`

The monitor cycle runs every 5 minutes during market hours and evaluates each open position against six exit conditions (in priority order):

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Kill switch active | Market exit immediately |
| 2 | Stop loss breached | Market exit |
| 3 | News shock (score < −0.6) | Market exit |
| 4 | Circuit breaker (5% session loss) | Market exit all positions |
| 5 | Take profit TP1 hit | Close 50%, activate trailing stop |
| 6 | Trailing stop breached | Market exit remainder |
| 7 | Time stop (day 5 or 7) | Market exit at next open |
| 8 | Thesis deterioration (re-score < 35) | Market exit |

Exit decisions are logged with their reason in the `TradingDecision` table for post-trade analysis.

### Risk Controls and Kill Switch

**Files:** `src/services/risk/`

Risk is enforced at multiple layers:

1. **Per-trade risk** (`engine.py`): Each order is checked against `max_risk_per_trade_pct` before submission.

2. **Total portfolio risk** (`engine.py`): Sum of `(entry - stop) × shares` across all open positions must not exceed `max_total_risk_pct` of equity.

3. **Daily loss halt** (`engine.py`): If the portfolio loses more than `max_daily_loss_pct` intraday, new entries are blocked until the next session.

4. **Drawdown kill switch** (`kill_switch.py`): If the portfolio drops more than `max_drawdown_pct` from its high-water mark, the kill switch is activated. All new orders are blocked; existing positions continue to be monitored and exited normally.

5. **Circuit breaker** (`guards.py`): If the portfolio loses more than `circuit_breaker_pct` (default 5%) in a single session, ALL positions are immediately market-exited.

6. **Stale data guard** (`guards.py`): If market data is older than `stale_data_threshold_seconds` (default 120s), all new entries are blocked until fresh data is received.

7. **Manual kill switch** (`kill_switch.py`): Toggleable via the REST API (`POST /api/v1/kill-switch/activate`) or by setting `KILL_SWITCH_ENABLED=true` in the environment. Requires `kill_switch_auto_reset=false` to remain active after restart.

---

## Trading Decision Object

Every scan cycle produces a `TradingDecision` record stored in PostgreSQL. The full schema:

```python
class TradingDecision:
    id: UUID
    created_at: datetime

    # Decision outcome
    action: Literal["BUY", "NO_TRADE"]
    symbol: str | None           # None for NO_TRADE
    no_trade_reason: str | None  # Populated for NO_TRADE

    # Score breakdown
    composite_score: float       # 0–100 weighted composite
    technical_score: float       # 0–100 technical sub-score
    news_score: float            # 0–100 news catalyst sub-score
    sentiment_score: float       # 0–100 social sentiment sub-score
    liquidity_score: float       # 0–100 liquidity sub-score
    regime_score: float          # 0–100 regime alignment sub-score
    risk_reward_score: float     # 0–100 R:R quality sub-score
    confidence: float            # 0–100 model confidence

    # Regime context
    market_regime: Literal["BULL", "NEUTRAL", "BEAR"]

    # Trade specification (populated on BUY)
    entry_price: float | None
    stop_price: float | None
    take_profit_price: float | None
    trailing_stop_price: float | None
    position_size_shares: int | None
    position_value_usd: float | None
    risk_usd: float | None
    risk_reward_ratio: float | None

    # ATR context
    atr_14: float | None

    # Human-readable explanation
    explanation: str             # Full score breakdown text

    # Pipeline stats
    candidates_evaluated: int
    candidates_passed_liquidity: int
    candidates_passed_technical: int
    candidates_passed_news: int
    candidates_passed_sentiment: int
```

---

## No-Trade Logic

SwingTrader is designed to say "no" more often than "yes". A `NO_TRADE` decision is recorded whenever:

- Market regime is BEAR and `block_in_bear_regime=true`
- No candidate passes all pipeline stages
- Best candidate scores below `no_trade_threshold`
- Best candidate confidence is below `confidence_threshold`
- Best candidate R:R ratio is below `min_risk_reward`
- Daily loss limit has been reached
- Kill switch is active
- Stale data detected
- Portfolio is fully allocated (`max_open_positions` reached)

Every `NO_TRADE` record includes a `no_trade_reason` string explaining the specific gate that was not cleared. This data is critical for tuning thresholds post-run.

---

## Paper vs Live Trading

| Aspect | Paper | Live |
|--------|-------|------|
| Broker | Internal simulation / Alpaca paper | Alpaca live brokerage |
| Fill model | Mid-price ± slippage | Real market fills |
| Risk limits | Moderate (configurable) | Strict (configurable) |
| Score threshold | 60.0 (conservative profile) | 65.0 (conservative profile) |
| Confidence threshold | 65.0 | 72.0 |
| Min R:R | 2.0 | 2.0 |
| Max positions | 3 | 4 |
| API docs | Enabled (`/docs`) | Disabled |
| Log format | Console (human-readable) | JSON (structured) |
| Recommended duration before live | ≥ 30 trading days | — |

### Switching to Live Trading

Before enabling live trading:

1. Run paper mode for at least 30 trading days and achieve a positive Sharpe ratio (>1.0 target).
2. Run the walk-forward backtest over at least 2 years of data: `make backtest ARGS="--start 2022-01-01 --end 2023-12-31 --walk-forward"`.
3. Review all positions and decisions manually.
4. Set `TRADING_MODE=live` and use live Alpaca API keys.
5. Start with a small initial capital allocation (e.g., 10% of intended total).
6. Monitor closely for the first 5 trading days.

---

## Backtesting

**Files:** `src/backtest/`, `scripts/run_backtest.py`

SwingTrader includes a historical backtesting engine that replays the full scan-and-monitor loop against historical OHLCV data.

```bash
# Basic backtest over 2023
make backtest ARGS="--start 2023-01-01 --end 2023-12-31"

# Walk-forward validation (12-month train, 3-month test windows)
make backtest ARGS="--start 2021-01-01 --end 2023-12-31 --walk-forward"

# Custom configuration profile
make backtest ARGS="--start 2023-01-01 --end 2023-12-31 --profile conservative"

# Full CLI options
python scripts/run_backtest.py --help
```

**Backtest output includes:**

- Total return, annualised return, Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown (amount and duration)
- Win rate, average win, average loss, profit factor
- Trade log (entry/exit date, reason, P&L per trade)
- Monthly returns heatmap
- Equity curve plot (saved to `backtest_results/`)
- Score distribution histogram
- NO_TRADE reason breakdown

**Walk-forward validation** (`walk_forward.py`) avoids overfitting by training on in-sample windows and testing on out-of-sample periods. Results show how stable the strategy parameters are across different market regimes.

**Backtest limitations:**
- Look-ahead bias is prevented by strict timestamp gating
- News and sentiment data for historical dates are simulated (FinBERT scores are not re-computed)
- Slippage is modelled as a fixed percentage (not bid/ask microsimulation)
- Short selling is not supported

---

## API Reference

The FastAPI admin API runs on port 8000. All endpoints require the `X-API-Key` header set to the value of `API_KEY` in `.env`.

### Authentication

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/v1/health
```

### Endpoints

#### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic liveness probe (no auth required) |
| GET | `/api/v1/health/detailed` | Full health check (DB, Redis, broker, data) |

```bash
curl http://localhost:8000/health
# {"status":"ok","timestamp":"2025-01-15T09:30:00Z"}

curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/detailed
# {"status":"healthy","postgres":"connected","redis":"connected","broker":"connected","stale_data":false}
```

#### Trading Decisions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/decisions` | List all trading decisions (paginated) |
| GET | `/api/v1/decisions/latest` | Most recent decision |
| GET | `/api/v1/decisions/{id}` | Decision by ID (includes score explanation) |

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/decisions/latest
```

#### Positions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/positions` | All open positions |
| GET | `/api/v1/positions/{symbol}` | Position for a specific symbol |
| DELETE | `/api/v1/positions/{symbol}` | Close a position immediately (market order) |

```bash
# Close a position manually
curl -X DELETE -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/positions/NVDA
```

#### Orders

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/orders` | Order history (paginated, filterable) |
| GET | `/api/v1/orders/{order_id}` | Specific order detail |

#### Trades (Closed Positions)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/trades` | Completed trade history with P&L |
| GET | `/api/v1/trades/summary` | Aggregate statistics (win rate, total P&L, Sharpe) |

#### Risk Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/risk` | Current risk snapshot |
| GET | `/api/v1/risk/history` | Risk metric history |

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/risk
# {
#   "total_equity": 105200.00,
#   "total_open_risk_pct": 3.2,
#   "daily_pnl_pct": 0.8,
#   "drawdown_pct": 1.1,
#   "open_positions": 2,
#   "kill_switch_active": false,
#   "regime": "BULL"
# }
```

#### Kill Switch

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/kill-switch` | Current kill switch status |
| POST | `/api/v1/kill-switch/activate` | Activate kill switch (halt new entries) |
| POST | `/api/v1/kill-switch/deactivate` | Deactivate kill switch (resume trading) |

```bash
# Emergency halt
curl -X POST -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/kill-switch/activate

# Resume
curl -X POST -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/kill-switch/deactivate
```

#### Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/config` | Current active configuration (read-only) |
| GET | `/api/v1/config/weights` | Current scoring weight profile |

#### Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/reports/daily` | Daily performance report |
| GET | `/api/v1/reports/weekly` | Weekly performance report |
| GET | `/api/v1/reports/monthly` | Monthly performance report |

---

## Monitoring and Observability

### Structured Logging

All application logs use `structlog` in JSON format (configurable to console format for development). Every log line includes contextual fields: `timestamp`, `level`, `logger`, `event`, `symbol`, `cycle_id`, `trade_id`.

```json
{
  "timestamp": "2025-01-15T09:31:04.123Z",
  "level": "info",
  "logger": "scan_cycle",
  "event": "scan_complete",
  "cycle_id": "c7f3a1b2",
  "candidates_evaluated": 150,
  "candidates_selected": 1,
  "symbol": "NVDA",
  "composite_score": 73.8,
  "action": "BUY"
}
```

Logs are written to stdout (captured by Docker) and optionally to `/app/logs/swingtrader.log`.

### Prometheus Metrics

Metrics are exposed at `http://localhost:9090/metrics` in Prometheus text format.

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `swingtrader_scan_cycles_total` | Counter | Total scan cycles completed |
| `swingtrader_trades_entered_total` | Counter | Total positions opened |
| `swingtrader_trades_exited_total` | Counter | Total positions closed (labelled by exit reason) |
| `swingtrader_portfolio_equity` | Gauge | Current portfolio equity |
| `swingtrader_portfolio_pnl_pct` | Gauge | Unrealised + realised P&L as % of equity |
| `swingtrader_open_positions` | Gauge | Current number of open positions |
| `swingtrader_drawdown_pct` | Gauge | Current drawdown from high-water mark |
| `swingtrader_kill_switch_active` | Gauge | 1 if kill switch is active |
| `swingtrader_data_latency_seconds` | Histogram | Market data fetch latency |
| `swingtrader_scan_duration_seconds` | Histogram | Full scan cycle duration |

Integrate with **Grafana** by adding Prometheus as a data source and importing the included dashboard template.

### Alerts

The `AlertManager` (`src/services/monitor/alert_manager.py`) generates structured alerts for:

- Kill switch activated / deactivated
- Drawdown exceeding configured thresholds
- News shock exit triggered
- Order submission failures
- Stale data detected
- Circuit breaker activated

Alerts are stored in the `alerts` PostgreSQL table and retrievable via `/api/v1/alerts`.

---

## Deployment (Docker Compose)

### Production Deployment

```bash
# 1. Clone the repository on your server
git clone https://github.com/example/swing-trader.git
cd swing-trader

# 2. Set up environment (edit all values carefully)
cp .env.example .env
nano .env

# 3. Build the image
make docker-build

# 4. Start all services
make docker-up

# 5. Initialise the database (first run only)
docker compose exec app python scripts/init_db.py
docker compose exec app python -m alembic upgrade head
docker compose exec app python scripts/seed_universe.py
```

### Monitoring the Deployment

```bash
# Follow application logs
make docker-logs

# Check service health
make docker-ps

# Open a shell inside the container
make docker-shell

# View Prometheus metrics
curl http://localhost:9090/metrics

# View API health
curl http://localhost:8000/health
```

### Updating

```bash
git pull origin main
make docker-build
docker compose up -d app   # Rolling restart of the app only
```

### Backup

The PostgreSQL data volume should be backed up regularly:

```bash
# Dump the database
docker compose exec postgres pg_dump -U swing swingtrader > backup_$(date +%Y%m%d).sql

# Restore
docker compose exec -T postgres psql -U swing swingtrader < backup_20250115.sql
```

---

## Development Guide

### Setting Up the Dev Environment

```bash
# Install all dependencies including dev tools
make dev

# Activate the virtualenv
source .venv/bin/activate

# Start infrastructure
docker compose up -d postgres redis
make db-init
```

### Code Quality

```bash
# Lint
make lint

# Auto-fix lint issues
make lint-fix

# Format code
make format

# Type checking
make typecheck

# Run all quality checks
make check
```

### Adding a New Service

1. Create the module in `src/services/<category>/your_service.py`
2. Define an abstract base class in `base.py` if the service has multiple implementations
3. Register with the appropriate engine/registry
4. Add unit tests in `tests/unit/`
5. Document configuration options in `config/default.yaml`

### Adding a New API Endpoint

1. Create or edit the route file in `src/api/routes/`
2. Add Pydantic schemas to `src/api/schemas.py`
3. Register the router in `src/api/app.py`
4. Add auth middleware as needed in `src/api/middleware.py`

### Database Migrations

```bash
# After modifying ORM models, generate a migration
make db-revision
# Enter a descriptive message when prompted

# Apply the migration
make db-migrate

# Check current state
make db-current
make db-history
```

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run unit tests only (fast, no DB/Redis required)
make test-unit

# Run integration tests (requires running DB and Redis)
make test-integration
```

### Test Structure

```
tests/
├── conftest.py                # Shared fixtures (DB, Redis, mock broker)
├── unit/
│   ├── test_indicators.py     # Technical indicator computations
│   ├── test_scoring.py        # Scoring engine and weight profiles
│   ├── test_sizing.py         # Position sizing calculations
│   ├── test_risk.py           # Risk engine logic
│   ├── test_pipeline.py       # Candidate pipeline stages
│   ├── test_trade_construction.py  # Trade constructor
│   └── test_exit_logic.py    # Exit engine conditions
└── integration/
    ├── test_scan_cycle.py     # Full scan cycle with mocked data
    ├── test_execution.py      # Order management and reconciliation
    └── test_backtest.py       # Backtesting engine
```

### Test Philosophy

- Unit tests use `pytest` and `pytest-asyncio` for async test support.
- All external dependencies (Alpaca, Finnhub, NewsAPI, Redis, PostgreSQL) are mocked in unit tests.
- Integration tests use real PostgreSQL and Redis (via Docker Compose).
- The `factory-boy` library generates realistic test data fixtures.
- Target: ≥70% code coverage.

---

## Project Structure

```
swing-trader/
├── alembic/                        # Database migrations
│   ├── versions/                   # Migration script files
│   └── env.py                      # Alembic async migration env
├── config/
│   ├── default.yaml                # Base configuration (all defaults)
│   ├── paper.yaml                  # Paper trading overrides
│   ├── live.yaml                   # Live trading overrides
│   └── universe.yaml               # Tradable universe (generated)
├── scripts/
│   ├── init_db.py                  # Database initialisation
│   ├── seed_universe.py            # Stock universe seeding
│   └── run_backtest.py             # CLI backtest runner
├── src/
│   ├── main.py                     # Application entry point
│   ├── core/
│   │   ├── config.py               # Configuration loader
│   │   ├── database.py             # Async SQLAlchemy engine + session
│   │   ├── redis_client.py         # Redis connection + helpers
│   │   ├── logging_config.py       # structlog setup
│   │   ├── metrics.py              # Prometheus metrics registry
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   ├── enums.py                # Shared enumerations
│   │   └── events.py               # Internal event bus
│   ├── models/
│   │   ├── base.py                 # SQLAlchemy Base + mixins
│   │   ├── market_data.py          # OHLCV bars and quotes
│   │   ├── indicator.py            # Computed technical indicators
│   │   ├── news.py                 # News items with FinBERT scores
│   │   ├── sentiment.py            # Social sentiment records
│   │   ├── candidate.py            # Candidate pipeline results
│   │   ├── decision.py             # Trading decisions
│   │   ├── order.py                # Order lifecycle records
│   │   ├── position.py             # Open positions
│   │   ├── trade.py                # Closed trades (P&L)
│   │   ├── risk_snapshot.py        # Periodic risk snapshots
│   │   └── alert.py                # System alerts
│   ├── services/
│   │   ├── market_data/            # Data providers (Alpaca, Finnhub, yfinance)
│   │   ├── technical/              # TA engine + indicator plugins
│   │   ├── news/                   # News fetch, FinBERT, dedup, scoring
│   │   ├── sentiment/              # Reddit + StockTwits adapters
│   │   ├── regime/                 # Market regime classifier
│   │   ├── scoring/                # Weighted scoring engine + explainer
│   │   ├── pipeline/               # Universe → rank candidate pipeline
│   │   ├── trade/                  # Constructor, sizer, validator
│   │   ├── execution/              # Broker adapters + order management
│   │   ├── monitor/                # Position monitor + exit engine
│   │   ├── risk/                   # Risk engine + kill switch + guards
│   │   └── orchestrator/           # APScheduler + scan/monitor cycles
│   ├── api/
│   │   ├── app.py                  # FastAPI app factory
│   │   ├── schemas.py              # Pydantic request/response schemas
│   │   ├── middleware.py           # Auth, CORS, error handling
│   │   └── routes/                 # REST endpoints (health, decisions, ...)
│   └── backtest/                   # Backtesting engine
├── tests/
│   ├── conftest.py                 # Shared test fixtures
│   ├── unit/                       # Unit tests (no infrastructure)
│   └── integration/                # Integration tests (DB + Redis)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion patterns
├── alembic.ini                     # Alembic configuration
├── docker-compose.yml              # Full stack: app + postgres + redis
├── Dockerfile                      # Multi-stage Python 3.11 image
├── Makefile                        # Developer convenience targets
├── pyproject.toml                  # Project metadata + dependencies
└── README.md                       # This file
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 SwingTrader Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Disclaimer

**IMPORTANT — PLEASE READ CAREFULLY**

This software is provided solely for **educational and research purposes**. It is NOT financial advice. It is NOT a solicitation to buy or sell any security. The authors and contributors are NOT licensed financial advisors, investment advisors, or broker-dealers.

**Trading involves substantial risk of loss** and is not appropriate for all investors. Past performance — including any backtest results produced by this software — is **not indicative of future results**. Backtests are hypothetical and do not account for all real-world frictions including (but not limited to): slippage, market impact, borrow costs, dividend adjustments, corporate actions, partial fills, and exchange outages.

By using this software:

1. You acknowledge that you may lose some or all of the capital you deploy.
2. You accept full responsibility for all trading decisions made while using this software.
3. You agree that the authors and contributors bear no liability for any financial losses incurred through the use of this software.
4. You confirm that you have read and understood all applicable laws and regulations in your jurisdiction regarding algorithmic trading and electronic order submission.

**Before deploying with real money:**

- Paper-trade for a minimum of 30 trading days.
- Consult a licensed financial advisor.
- Ensure compliance with all applicable securities laws.
- Start with an amount you can afford to lose entirely.
- Never use borrowed money, retirement savings, or funds needed for living expenses.

The kill switch and risk controls provided in this software are best-effort safeguards, not guarantees. Software bugs, infrastructure failures, network outages, and extreme market events can cause losses beyond configured limits.

**USE AT YOUR OWN RISK.**
