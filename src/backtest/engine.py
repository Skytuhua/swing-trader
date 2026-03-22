"""
BacktestEngine: simulate the full decision pipeline day-by-day over historical data.

No-lookahead guarantee: on each simulated trading day, only data up to and
including that day's close is visible to the pipeline.

Architecture:
  - Uses MarketSimulator to serve daily OHLCV slices
  - Uses PaperBroker for execution simulation
  - Mirrors the scan cycle logic: screen → score → size → trade
  - Runs the exit engine on each bar for open positions
  - Returns BacktestResult with full performance metrics
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Full backtest performance report."""

    trades: list[dict] = field(default_factory=list)

    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trade statistics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_hold_days: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_consecutive_losses: int = 0

    # Exposure
    exposure_pct: float = 0.0  # % of trading days where at least one position was open

    # Breakdown
    regime_breakdown: dict = field(default_factory=dict)  # regime → {return, trades, win_rate}
    monthly_returns: dict = field(default_factory=dict)   # "YYYY-MM" → return_pct

    # Equity curve (portfolio value each day)
    equity_curve: list[float] = field(default_factory=list)

    # Metadata
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0
    commission_paid: float = 0.0
    slippage_paid: float = 0.0


# ---------------------------------------------------------------------------
# Simulated position tracker
# ---------------------------------------------------------------------------


@dataclass
class SimPosition:
    ticker: str
    shares: int
    entry_price: float
    entry_date: date
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    atr_at_entry: float
    trailing_stop_price: float | None = None
    tp1_hit: bool = False
    max_price_since_entry: float = 0.0
    max_hold_days: int = 5
    hold_days: int = 0

    @property
    def unrealized_pnl_pct(self) -> float:
        # placeholder; gets computed with live price
        return 0.0


# ---------------------------------------------------------------------------
# BacktestConfig defaults
# ---------------------------------------------------------------------------

_COMMISSION_PER_SHARE = 0.005   # $0.005/share
_SLIPPAGE_PCT = 0.001            # 0.1% per-side slippage
_MAX_HOLD_DAYS = 5
_MAX_POSITIONS = 5
_MAX_RISK_PER_TRADE_PCT = 1.5
_MAX_POSITION_PCT = 20.0
_TECHNICAL_MIN_SCORE = 50.0
_RISK_FREE_RATE_ANNUAL = 0.045   # 4.5% risk-free


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Simulate the full trading pipeline over historical OHLCV data.

    Parameters
    ----------
    data : pd.DataFrame
        Multi-ticker OHLCV data. Either:
        - A single-ticker DataFrame with columns [open, high, low, close, volume]
          and DatetimeIndex, OR
        - A dict[ticker, DataFrame].
    config : dict | object
        Configuration overrides. Supported keys:
        ``initial_capital``, ``commission_per_share``, ``slippage_pct``,
        ``max_hold_days``, ``max_positions``, ``max_risk_per_trade_pct``,
        ``max_position_pct``, ``technical_min_score``.
    start_date : str | None
        Inclusive start date "YYYY-MM-DD". Defaults to first available date.
    end_date : str | None
        Inclusive end date "YYYY-MM-DD". Defaults to last available date.
    universe : list[str] | None
        Restrict simulation to these tickers when data is a dict.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        config: dict | Any | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        universe: list[str] | None = None,
    ) -> None:
        self._raw_data = data
        self._config = config or {}
        self._start_date = start_date
        self._end_date = end_date
        self._universe = universe

        # Parse config values
        cfg = self._config if isinstance(self._config, dict) else {}
        self.initial_capital: float = float(
            cfg.get("initial_capital", getattr(self._config, "initial_capital", 100_000.0))
        )
        self.commission_per_share: float = float(
            cfg.get("commission_per_share", getattr(self._config, "commission_per_share", _COMMISSION_PER_SHARE))
        )
        self.slippage_pct: float = float(
            cfg.get("slippage_pct", getattr(self._config, "slippage_pct", _SLIPPAGE_PCT))
        )
        self.max_hold_days: int = int(
            cfg.get("max_hold_days", getattr(self._config, "max_hold_days", _MAX_HOLD_DAYS))
        )
        self.max_positions: int = int(
            cfg.get("max_positions", getattr(self._config, "max_positions", _MAX_POSITIONS))
        )
        self.max_risk_per_trade_pct: float = float(
            cfg.get("max_risk_per_trade_pct", getattr(self._config, "max_risk_per_trade_pct", _MAX_RISK_PER_TRADE_PCT))
        )
        self.max_position_pct: float = float(
            cfg.get("max_position_pct", getattr(self._config, "max_position_pct", _MAX_POSITION_PCT))
        )
        self.technical_min_score: float = float(
            cfg.get("technical_min_score", getattr(self._config, "technical_min_score", _TECHNICAL_MIN_SCORE))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Run the backtest and return a BacktestResult."""
        from src.backtest.simulator import MarketSimulator

        # Build simulator
        sim = MarketSimulator(
            data=self._raw_data,
            slippage_pct=self.slippage_pct,
            commission_per_share=self.commission_per_share,
        )

        trading_days = sim.get_trading_days(self._start_date, self._end_date)
        if not trading_days:
            logger.warning("backtest_engine: no trading days in range")
            return BacktestResult(
                start_date=self._start_date or "",
                end_date=self._end_date or "",
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
            )

        # State
        cash = self.initial_capital
        positions: dict[str, SimPosition] = {}   # ticker → SimPosition
        completed_trades: list[dict] = []
        equity_curve: list[float] = []
        days_with_positions = 0
        total_commission_paid = 0.0
        total_slippage_paid = 0.0
        regime_trade_buckets: dict[str, list[dict]] = defaultdict(list)

        for day_idx, current_date in enumerate(trading_days):
            # ---- Advance simulator to this day (no lookahead) ----
            sim.advance_to(current_date)

            # ---- Exit checks (morning of this day's bar) ----
            tickers_to_close: list[tuple[str, str, float]] = []  # (ticker, reason, price)

            for ticker, pos in list(positions.items()):
                bar = sim.get_bar(ticker, current_date)
                if bar is None:
                    continue

                pos.hold_days += 1
                open_px = float(bar["open"])
                high_px = float(bar["high"])
                low_px = float(bar["low"])
                close_px = float(bar["close"])

                # Update max price
                pos.max_price_since_entry = max(pos.max_price_since_entry, high_px)

                # Update trailing stop (activate after 1% gain)
                if pos.max_price_since_entry >= pos.entry_price * 1.01 and pos.atr_at_entry > 0:
                    new_trail = pos.max_price_since_entry - (pos.atr_at_entry * 1.5)
                    current_trail = pos.trailing_stop_price or pos.stop_loss
                    if new_trail > current_trail:
                        pos.trailing_stop_price = new_trail

                exit_reason = None
                exit_price = None

                # Gap-down through stop (open ≤ stop)
                if open_px <= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = open_px  # fill at open (worst-case gap)
                elif low_px <= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = pos.stop_loss
                elif pos.trailing_stop_price and low_px <= pos.trailing_stop_price:
                    exit_reason = "trailing_stop"
                    exit_price = pos.trailing_stop_price
                elif not pos.tp1_hit and high_px >= pos.take_profit_1:
                    exit_reason = "take_profit_1"
                    exit_price = pos.take_profit_1
                    pos.tp1_hit = True
                elif pos.tp1_hit and high_px >= pos.take_profit_2:
                    exit_reason = "take_profit_2"
                    exit_price = pos.take_profit_2
                elif pos.hold_days >= pos.max_hold_days:
                    exit_reason = "time_stop"
                    exit_price = close_px

                if exit_reason:
                    tickers_to_close.append((ticker, exit_reason, exit_price))

            # ---- Execute exits ----
            for ticker, reason, exit_price_raw in tickers_to_close:
                pos = positions.pop(ticker, None)
                if pos is None:
                    continue

                fill_price, slippage = sim.fill_sell(ticker, exit_price_raw)
                gross_proceeds = fill_price * pos.shares
                commission = self.commission_per_share * pos.shares
                net_proceeds = gross_proceeds - commission
                cash += net_proceeds

                total_commission_paid += commission
                total_slippage_paid += slippage * pos.shares

                cost_basis = pos.entry_price * pos.shares
                pnl = net_proceeds - cost_basis
                pnl_pct = (pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

                trade = {
                    "ticker": ticker,
                    "entry_date": str(pos.entry_date),
                    "exit_date": str(current_date),
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(fill_price, 4),
                    "shares": pos.shares,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 3),
                    "hold_days": pos.hold_days,
                    "exit_reason": reason,
                    "commission": round(commission, 2),
                    "regime": "unknown",
                }
                completed_trades.append(trade)
                logger.debug(
                    "backtest_exit",
                    ticker=ticker,
                    reason=reason,
                    pnl_pct=round(pnl_pct, 2),
                    hold_days=pos.hold_days,
                )

            # ---- Entry scan ----
            if len(positions) < self.max_positions:
                portfolio_value = cash + self._mark_positions(positions, sim, current_date)
                candidates = sim.get_candidates(
                    current_date,
                    exclude=set(positions.keys()),
                    universe=self._universe,
                )
                for ticker, score, indicators in candidates:
                    if len(positions) >= self.max_positions:
                        break
                    if score < self.technical_min_score:
                        continue

                    bar = sim.get_bar(ticker, current_date)
                    if bar is None:
                        continue

                    atr = float(indicators.get("atr_14", 0) or 0)
                    close_px = float(bar["close"])
                    if atr <= 0:
                        atr = close_px * 0.02

                    # Entry at next-bar open (no lookahead - we use today's close + slippage)
                    entry_price, _ = sim.fill_buy(ticker, close_px)
                    stop_loss = close_px - (atr * 2.0)
                    stop_loss = min(stop_loss, close_px * 0.995)
                    take_profit_1 = close_px + ((close_px - stop_loss) * 2.0)
                    take_profit_2 = close_px + ((close_px - stop_loss) * 3.0)

                    risk_per_share = entry_price - stop_loss
                    if risk_per_share <= 0:
                        continue

                    # Risk-based sizing
                    max_risk_dollars = portfolio_value * (self.max_risk_per_trade_pct / 100.0)
                    risk_based_shares = int(max_risk_dollars / risk_per_share)

                    # Allocation-based sizing
                    alloc_dollars = portfolio_value * (self.max_position_pct / 100.0)
                    alloc_shares = int(alloc_dollars / entry_price)

                    shares = min(risk_based_shares, alloc_shares)
                    if shares <= 0:
                        continue

                    cost = entry_price * shares
                    commission = self.commission_per_share * shares
                    total_cost = cost + commission

                    if total_cost > cash:
                        shares = int((cash - commission) / entry_price)
                        if shares <= 0:
                            continue
                        total_cost = entry_price * shares + commission

                    cash -= total_cost
                    total_commission_paid += commission

                    positions[ticker] = SimPosition(
                        ticker=ticker,
                        shares=shares,
                        entry_price=entry_price,
                        entry_date=current_date,
                        stop_loss=stop_loss,
                        take_profit_1=take_profit_1,
                        take_profit_2=take_profit_2,
                        atr_at_entry=atr,
                        max_price_since_entry=close_px,
                        max_hold_days=self.max_hold_days,
                    )
                    logger.debug(
                        "backtest_entry",
                        ticker=ticker,
                        shares=shares,
                        entry_price=round(entry_price, 4),
                        stop=round(stop_loss, 4),
                        tp1=round(take_profit_1, 4),
                    )

            # ---- Mark-to-market ----
            portfolio_value = cash + self._mark_positions(positions, sim, current_date)
            equity_curve.append(portfolio_value)

            if positions:
                days_with_positions += 1

        # ---- Close any remaining open positions at last close ----
        last_day = trading_days[-1]
        sim.advance_to(last_day)
        for ticker, pos in list(positions.items()):
            bar = sim.get_bar(ticker, last_day)
            close_px = float(bar["close"]) if bar is not None else pos.entry_price
            fill_price, slippage = sim.fill_sell(ticker, close_px)
            gross = fill_price * pos.shares
            comm = self.commission_per_share * pos.shares
            cash += gross - comm
            total_commission_paid += comm
            cost_basis = pos.entry_price * pos.shares
            pnl = (gross - comm) - cost_basis
            pnl_pct = (pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0
            completed_trades.append({
                "ticker": ticker,
                "entry_date": str(pos.entry_date),
                "exit_date": str(last_day),
                "entry_price": round(pos.entry_price, 4),
                "exit_price": round(fill_price, 4),
                "shares": pos.shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 3),
                "hold_days": pos.hold_days,
                "exit_reason": "forced_close",
                "commission": round(comm, 2),
                "regime": "unknown",
            })
        # Update final equity
        if equity_curve:
            equity_curve[-1] = cash

        # ---- Compute metrics ----
        result = self._compute_metrics(
            trades=completed_trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=cash,
            total_days=len(trading_days),
            days_with_positions=days_with_positions,
            start_date=str(trading_days[0]) if trading_days else "",
            end_date=str(trading_days[-1]) if trading_days else "",
            commission_paid=total_commission_paid,
            slippage_paid=total_slippage_paid,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_positions(
        self,
        positions: dict[str, SimPosition],
        sim: "MarketSimulator",  # type: ignore[name-defined]
        current_date: date,
    ) -> float:
        """Return total market value of all open positions."""
        total = 0.0
        for ticker, pos in positions.items():
            bar = sim.get_bar(ticker, current_date)
            price = float(bar["close"]) if bar is not None else pos.entry_price
            total += price * pos.shares
        return total

    def _compute_metrics(
        self,
        trades: list[dict],
        equity_curve: list[float],
        initial_capital: float,
        final_capital: float,
        total_days: int,
        days_with_positions: int,
        start_date: str,
        end_date: str,
        commission_paid: float,
        slippage_paid: float,
    ) -> BacktestResult:
        """Compute all performance metrics from raw trade log and equity curve."""
        result = BacktestResult(
            trades=trades,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            equity_curve=equity_curve,
            commission_paid=round(commission_paid, 2),
            slippage_paid=round(slippage_paid, 2),
        )

        if not equity_curve or initial_capital <= 0:
            return result

        # Total return
        result.total_return_pct = round(
            (final_capital - initial_capital) / initial_capital * 100.0, 3
        )

        # Annualised return (CAGR)
        years = total_days / 252.0
        if years > 0 and initial_capital > 0:
            cagr = ((final_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0
            result.annualized_return_pct = round(cagr, 3)

        # Max drawdown
        result.max_drawdown_pct = round(self._max_drawdown(equity_curve), 3)

        # Sharpe & Sortino
        daily_returns = np.diff(equity_curve) / np.array(equity_curve[:-1], dtype=float)
        rf_daily = (1 + _RISK_FREE_RATE_ANNUAL) ** (1 / 252) - 1
        excess_returns = daily_returns - rf_daily
        if len(excess_returns) > 1:
            std = excess_returns.std(ddof=1)
            result.sharpe_ratio = round(
                float(excess_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0, 3
            )
            downside = excess_returns[excess_returns < 0]
            if len(downside) > 1:
                downside_std = float(np.sqrt((downside ** 2).mean()))
                result.sortino_ratio = round(
                    float(excess_returns.mean() / downside_std * np.sqrt(252))
                    if downside_std > 0 else 0.0,
                    3,
                )

        # Exposure
        result.exposure_pct = round(days_with_positions / total_days * 100.0, 2) if total_days > 0 else 0.0

        # Trade statistics
        if trades:
            pnls = [t["pnl_pct"] for t in trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]

            result.total_trades = len(trades)
            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = round(len(winners) / len(pnls) * 100.0, 2)
            result.avg_win_pct = round(float(np.mean(winners)), 3) if winners else 0.0
            result.avg_loss_pct = round(float(np.mean(losers)), 3) if losers else 0.0
            result.avg_hold_days = round(
                float(np.mean([t.get("hold_days", 0) for t in trades])), 2
            )

            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            result.profit_factor = (
                round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")
            )
            result.expectancy = round(float(np.mean(pnls)), 3)
            result.max_consecutive_losses = self._max_consecutive_losses(pnls)

        # Monthly returns
        if trades:
            monthly: dict[str, list[float]] = defaultdict(list)
            for t in trades:
                try:
                    month_key = t["exit_date"][:7]  # "YYYY-MM"
                    monthly[month_key].append(t["pnl_pct"])
                except (KeyError, TypeError):
                    pass
            result.monthly_returns = {
                k: round(sum(v), 3) for k, v in sorted(monthly.items())
            }

        # Regime breakdown
        regime_buckets: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            regime_buckets[t.get("regime", "unknown")].append(t)
        for regime, regime_trades in regime_buckets.items():
            pnls = [t["pnl_pct"] for t in regime_trades]
            winners = [p for p in pnls if p > 0]
            result.regime_breakdown[regime] = {
                "trades": len(regime_trades),
                "win_rate": round(len(winners) / len(pnls) * 100.0, 2) if pnls else 0.0,
                "avg_return_pct": round(float(np.mean(pnls)), 3) if pnls else 0.0,
                "total_return_pct": round(sum(pnls), 3),
            }

        return result

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        """Maximum peak-to-trough drawdown as a percentage."""
        if not equity_curve:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        peak = arr[0]
        max_dd = 0.0
        for val in arr:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _max_consecutive_losses(pnl_pcts: list[float]) -> int:
        """Count maximum streak of consecutive losing trades."""
        max_streak = 0
        streak = 0
        for p in pnl_pcts:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak
