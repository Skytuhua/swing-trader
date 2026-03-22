"""
Post-trade analysis engine for learning from completed trades.

This module computes performance attribution metrics for individual trades
(MFE, MAE, entry/exit quality, efficiency) and aggregates them across batches
to identify systematic patterns in winners vs losers.  Results feed into
strategy refinement and adaptive scoring weight adjustments.

Trade dict schema
-----------------
Required keys:
  ticker          : str
  entry_price     : float
  exit_price      : float
  quantity        : int
  entry_date      : str | datetime   (ISO-8601 or datetime object)
  exit_date       : str | datetime
  net_pnl         : float
  gross_pnl       : float
  commission_total: float
  exit_reason     : str

Optional keys (significantly improve analysis when present):
  price_series    : list[float]  – daily close prices during hold period
  high_series     : list[float]  – daily highs during hold period
  low_series      : list[float]  – daily lows during hold period
  stop_price      : float
  target_price    : float
  entry_score     : float        – composite signal score at entry (0-100)
  signal_components: dict[str, float]  – individual signal scores
  max_favorable_excursion : float  – pre-computed MFE (USD) if available
  max_adverse_excursion   : float  – pre-computed MAE (USD) if available
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PostTradeReport:
    """Detailed analysis of a single completed trade.

    Attributes
    ----------
    ticker:
        Instrument traded.
    entry_price, exit_price:
        Execution prices.
    quantity:
        Shares traded.
    hold_days:
        Calendar days position was held.
    net_pnl:
        Net profit/loss in USD.
    pnl_pct:
        Net P&L as percentage of entry cost.
    is_winner:
        True when net_pnl > 0.

    MFE / MAE
    ---------
    mfe_usd:
        Max Favorable Excursion – best unrealised profit achieved during
        the trade (USD).  Positive = profit.
    mae_usd:
        Max Adverse Excursion – worst unrealised loss suffered during the
        trade (USD).  Negative = loss.
    mfe_pct:
        MFE as % of entry cost.
    mae_pct:
        MAE as % of entry cost.

    Quality metrics (each 0–100, higher = better)
    -----------------------------------------------
    efficiency:
        actual_pnl / MFE (0–100).  100 = captured the entire favourable move.
        A consistently low efficiency signals premature exits.
    stop_quality:
        How well the stop was placed relative to MAE.  100 = stop was never
        close to being hit before the target was reached.
    entry_quality:
        How close to the low of the trade the entry was.  100 = perfect entry.
    exit_quality:
        How close to the high of the trade the exit was.  100 = perfect exit.
    time_quality:
        Ratio of active profit days to total hold days (0–100).

    Signal accuracy
    ---------------
    signal_components:
        Component scores at entry time (copied from trade dict).
    predictive_components:
        List of signal component names with above-average scores in winners.
    """

    ticker: str
    entry_price: float
    exit_price: float
    quantity: int
    hold_days: int
    net_pnl: float
    pnl_pct: float
    is_winner: bool
    exit_reason: str = ""

    # Excursion
    mfe_usd: float = 0.0
    mae_usd: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0

    # Quality scores 0-100
    efficiency: float = 0.0
    stop_quality: float = 0.0
    entry_quality: float = 0.0
    exit_quality: float = 0.0
    time_quality: float = 0.0

    # Signal context
    entry_score: float | None = None
    signal_components: dict[str, float] = field(default_factory=dict)
    predictive_components: list[str] = field(default_factory=list)

    # Raw series (optional, omitted from repr for brevity)
    price_series: list[float] = field(default_factory=list, repr=False)


@dataclass
class SignalAccuracy:
    """Per-signal-component accuracy statistics."""

    component_name: str
    win_rate: float          # fraction of trades where this signal was present and trade won
    avg_score_winners: float
    avg_score_losers: float
    separation: float        # avg_score_winners - avg_score_losers (higher = more predictive)
    sample_size: int


@dataclass
class BatchTradeAnalysis:
    """Aggregated analysis across a batch of completed trades.

    Attributes
    ----------
    trade_count:
        Number of trades analysed.
    win_count, loss_count:
        Absolute win/loss counts.
    win_rate:
        Fraction of trades that were profitable.
    avg_pnl_pct:
        Mean P&L % across all trades.
    avg_winner_pnl_pct:
        Mean P&L % for winning trades.
    avg_loser_pnl_pct:
        Mean P&L % for losing trades (negative value).
    profit_factor:
        Gross profit / gross loss.  >1.0 = edge exists.
    avg_efficiency:
        Mean efficiency score (0–100) – measures how much of the MFE was captured.
    avg_mfe_pct:
        Mean max favorable excursion %.
    avg_mae_pct:
        Mean max adverse excursion % (absolute value).
    avg_entry_quality:
        Mean entry quality score (0–100).
    avg_exit_quality:
        Mean exit quality score (0–100).
    avg_hold_days:
        Average holding period.
    signal_accuracy:
        List of SignalAccuracy objects, ranked by predictive separation.
    exit_reason_breakdown:
        Dict mapping exit_reason → (count, win_rate, avg_pnl_pct).
    optimal_stop_pct:
        Median MAE% of losing trades – empirical guide for stop placement.
    optimal_tp_pct:
        Median MFE% of winning trades – empirical guide for take-profit placement.
    reports:
        Individual PostTradeReport objects included in this batch.
    """

    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_winner_pnl_pct: float = 0.0
    avg_loser_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    avg_efficiency: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    avg_entry_quality: float = 0.0
    avg_exit_quality: float = 0.0
    avg_hold_days: float = 0.0
    signal_accuracy: list[SignalAccuracy] = field(default_factory=list)
    exit_reason_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    optimal_stop_pct: float = 0.0
    optimal_tp_pct: float = 0.0
    reports: list[PostTradeReport] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


class PostTradeAnalyzer:
    """Analyse completed trades to identify what worked and what didn't.

    Designed to be stateless – pass all required data in each method call.
    """

    # ------------------------------------------------------------------
    # Single-trade analysis
    # ------------------------------------------------------------------

    def analyze_trade(self, trade: dict) -> PostTradeReport:
        """Generate a post-trade analysis report for a single closed trade.

        Parameters
        ----------
        trade:
            Dict conforming to the schema described in the module docstring.

        Returns
        -------
        PostTradeReport
        """
        ticker = str(trade.get("ticker", "UNKNOWN"))
        entry_price = float(trade.get("entry_price", 0.0))
        exit_price = float(trade.get("exit_price", 0.0))
        quantity = int(trade.get("quantity", 0))
        net_pnl = float(trade.get("net_pnl", 0.0))
        gross_pnl = float(trade.get("gross_pnl", net_pnl))
        commission = float(trade.get("commission_total", 0.0))
        exit_reason = str(trade.get("exit_reason", "unknown"))
        entry_score = trade.get("entry_score")
        signal_components: dict[str, float] = trade.get("signal_components", {}) or {}

        # Validate prices
        if entry_price <= 0 or quantity <= 0:
            logger.warning(
                "post_trade.invalid_trade",
                ticker=ticker,
                entry_price=entry_price,
                quantity=quantity,
            )
            return PostTradeReport(
                ticker=ticker,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                hold_days=0,
                net_pnl=net_pnl,
                pnl_pct=0.0,
                is_winner=net_pnl > 0,
                exit_reason=exit_reason,
                entry_score=float(entry_score) if entry_score is not None else None,
                signal_components=signal_components,
            )

        # Hold period
        hold_days = self._compute_hold_days(trade)

        # P&L %
        entry_cost = entry_price * quantity
        pnl_pct = (net_pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0
        is_winner = net_pnl > 0

        # Price series
        high_series: list[float] = list(trade.get("high_series") or [])
        low_series: list[float] = list(trade.get("low_series") or [])
        price_series: list[float] = list(trade.get("price_series") or [])

        # MFE / MAE
        mfe_usd, mae_usd = self._compute_excursions(
            trade, entry_price, exit_price, quantity,
            high_series, low_series, price_series,
        )
        mfe_pct = (mfe_usd / entry_cost) * 100.0 if entry_cost > 0 else 0.0
        mae_pct = (mae_usd / entry_cost) * 100.0 if entry_cost > 0 else 0.0

        # Quality metrics
        efficiency = self._compute_efficiency(net_pnl, mfe_usd)
        stop_quality = self._compute_stop_quality(mae_usd, trade, entry_price, quantity)
        entry_quality = self._compute_entry_quality(entry_price, low_series, price_series)
        exit_quality = self._compute_exit_quality(exit_price, high_series, price_series)
        time_quality = self._compute_time_quality(price_series, entry_price, hold_days)

        report = PostTradeReport(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            hold_days=hold_days,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
            exit_reason=exit_reason,
            mfe_usd=mfe_usd,
            mae_usd=mae_usd,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            efficiency=efficiency,
            stop_quality=stop_quality,
            entry_quality=entry_quality,
            exit_quality=exit_quality,
            time_quality=time_quality,
            entry_score=float(entry_score) if entry_score is not None else None,
            signal_components=signal_components,
            price_series=price_series,
        )

        logger.debug(
            "post_trade.analyzed",
            ticker=ticker,
            is_winner=is_winner,
            pnl_pct=round(pnl_pct, 2),
            efficiency=round(efficiency, 1),
            mfe_pct=round(mfe_pct, 2),
            mae_pct=round(mae_pct, 2),
        )
        return report

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(self, trades: list[dict]) -> BatchTradeAnalysis:
        """Aggregate analysis across multiple trades.

        Parameters
        ----------
        trades:
            List of trade dicts (same schema as ``analyze_trade``).

        Returns
        -------
        BatchTradeAnalysis
        """
        if not trades:
            return BatchTradeAnalysis()

        reports = [self.analyze_trade(t) for t in trades]
        return self._aggregate_reports(reports)

    def analyze_reports(self, reports: list[PostTradeReport]) -> BatchTradeAnalysis:
        """Aggregate pre-computed PostTradeReport objects."""
        if not reports:
            return BatchTradeAnalysis()
        return self._aggregate_reports(reports)

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------

    def generate_improvement_suggestions(
        self, analysis: BatchTradeAnalysis
    ) -> list[str]:
        """Generate actionable improvement suggestions from batch analysis.

        Parameters
        ----------
        analysis:
            BatchTradeAnalysis from ``analyze_batch``.

        Returns
        -------
        list[str]
            Ordered list of human-readable suggestions (most impactful first).
        """
        suggestions: list[tuple[float, str]] = []  # (priority, text)

        if analysis.trade_count == 0:
            return ["Insufficient trade data for analysis."]

        # --- Win rate ---
        if analysis.win_rate < 0.40:
            suggestions.append((
                10.0,
                f"Win rate is low ({analysis.win_rate:.1%}). "
                "Review entry criteria – consider raising minimum signal score threshold."
            ))
        elif analysis.win_rate > 0.65:
            suggestions.append((
                2.0,
                f"Win rate is strong ({analysis.win_rate:.1%}). "
                "Consider whether take-profit targets could be extended to improve avg winner."
            ))

        # --- Efficiency ---
        if analysis.avg_efficiency < 40.0:
            suggestions.append((
                9.0,
                f"Trade efficiency is low ({analysis.avg_efficiency:.1f}/100): "
                "only capturing ~{:.0f}% of the average favourable move. "
                "Consider trailing stops or later take-profit triggers.".format(
                    analysis.avg_efficiency
                )
            ))
        elif analysis.avg_efficiency < 60.0:
            suggestions.append((
                5.0,
                f"Trade efficiency ({analysis.avg_efficiency:.1f}/100) has room to improve. "
                "Review take-profit placement relative to MFE."
            ))

        # --- Entry quality ---
        if analysis.avg_entry_quality < 50.0:
            suggestions.append((
                7.0,
                f"Entry quality is poor ({analysis.avg_entry_quality:.1f}/100). "
                "Entries are occurring significantly above intra-trade lows. "
                "Consider using limit orders closer to support levels."
            ))

        # --- Exit quality ---
        if analysis.avg_exit_quality < 50.0:
            suggestions.append((
                6.0,
                f"Exit quality is poor ({analysis.avg_exit_quality:.1f}/100). "
                "Trades are being closed well below their intra-trade highs. "
                "Consider scaling exits or using trailing stops."
            ))

        # --- Profit factor ---
        if analysis.profit_factor < 1.0:
            suggestions.append((
                10.0,
                f"Profit factor {analysis.profit_factor:.2f} < 1.0: "
                "gross losses exceed gross profits. "
                "The strategy has no positive expectancy. "
                "Re-evaluate the core signal logic."
            ))
        elif analysis.profit_factor < 1.5:
            suggestions.append((
                6.0,
                f"Profit factor {analysis.profit_factor:.2f} is marginal. "
                "Tightening stop placement or improving entry quality may help."
            ))

        # --- Stop placement ---
        if analysis.optimal_stop_pct > 0 and analysis.trade_count >= 10:
            suggestions.append((
                4.0,
                f"Empirical stop suggestion: set stops at ≈{analysis.optimal_stop_pct:.1f}% "
                "below entry (median MAE of losing trades). "
                f"Compare to current average MAE of losers."
            ))

        # --- Signal accuracy ---
        if analysis.signal_accuracy:
            top = analysis.signal_accuracy[0]
            bottom = analysis.signal_accuracy[-1]
            if top.separation > 5.0:
                suggestions.append((
                    5.0,
                    f"Most predictive signal: '{top.component_name}' "
                    f"(avg score {top.avg_score_winners:.1f} in winners vs "
                    f"{top.avg_score_losers:.1f} in losers). "
                    "Consider increasing its weight."
                ))
            if bottom.separation < -2.0 and len(analysis.signal_accuracy) >= 3:
                suggestions.append((
                    4.0,
                    f"Least predictive signal: '{bottom.component_name}' "
                    f"(higher scores in losers). "
                    "Consider reducing or removing this component."
                ))

        # --- Hold period ---
        if analysis.avg_hold_days < 1.5 and analysis.win_rate < 0.55:
            suggestions.append((
                3.0,
                f"Average hold of {analysis.avg_hold_days:.1f} days is very short. "
                "Trades may be exiting too quickly before the thesis develops."
            ))
        elif analysis.avg_hold_days > 15 and analysis.win_rate < 0.55:
            suggestions.append((
                3.0,
                f"Average hold of {analysis.avg_hold_days:.1f} days is long. "
                "Consider a time-stop to limit opportunity cost."
            ))

        # --- Exit reason breakdown ---
        for reason, stats in analysis.exit_reason_breakdown.items():
            count = stats.get("count", 0)
            wr = stats.get("win_rate", 0.0)
            avg_pnl = stats.get("avg_pnl_pct", 0.0)
            if count >= 3 and wr < 0.30:
                suggestions.append((
                    5.0,
                    f"Exit reason '{reason}' has a {wr:.0%} win rate across {count} trades "
                    f"(avg P&L {avg_pnl:+.1f}%). "
                    "Review conditions that trigger this exit."
                ))

        # Sort by priority (descending) and return text only
        suggestions.sort(key=lambda x: -x[0])
        return [s for _, s in suggestions] if suggestions else [
            "No major issues detected. Continue monitoring performance."
        ]

    # ------------------------------------------------------------------
    # Internal: single-trade helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hold_days(trade: dict) -> int:
        """Parse entry/exit dates and return calendar hold days."""
        try:
            raw_entry = trade.get("entry_date")
            raw_exit = trade.get("exit_date")

            def _parse(raw: Any) -> datetime:
                if isinstance(raw, datetime):
                    return raw
                if isinstance(raw, str):
                    return datetime.fromisoformat(raw)
                # Assume unix timestamp
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)

            entry_dt = _parse(raw_entry)
            exit_dt = _parse(raw_exit)
            delta = exit_dt - entry_dt
            return max(1, delta.days)
        except Exception:
            return int(trade.get("hold_days", 1)) or 1

    @staticmethod
    def _compute_excursions(
        trade: dict,
        entry_price: float,
        exit_price: float,
        quantity: int,
        high_series: list[float],
        low_series: list[float],
        price_series: list[float],
    ) -> tuple[float, float]:
        """Return (mfe_usd, mae_usd).

        mfe_usd: best unrealised profit achieved during the trade (always ≥ 0)
        mae_usd: worst unrealised loss suffered (always ≤ 0)

        Preferred source: pre-computed values in the trade dict.
        Fallback: infer from high/low series.
        """
        # Use pre-computed values if present
        pre_mfe = trade.get("max_favorable_excursion")
        pre_mae = trade.get("max_adverse_excursion")
        if pre_mfe is not None and pre_mae is not None:
            return float(pre_mfe), float(pre_mae)

        if high_series and low_series and len(high_series) == len(low_series):
            highs = np.array(high_series, dtype=float)
            lows = np.array(low_series, dtype=float)
            mfe_price = float(highs.max())
            mae_price = float(lows.min())
        elif price_series:
            prices = np.array(price_series, dtype=float)
            mfe_price = float(prices.max())
            mae_price = float(prices.min())
        else:
            # Can only infer from entry/exit – conservative estimate
            mfe_price = max(entry_price, exit_price)
            mae_price = min(entry_price, exit_price)

        mfe_usd = max(0.0, (mfe_price - entry_price) * quantity)
        mae_usd = min(0.0, (mae_price - entry_price) * quantity)
        return mfe_usd, mae_usd

    @staticmethod
    def _compute_efficiency(net_pnl: float, mfe_usd: float) -> float:
        """Compute capture efficiency: how much of MFE was realised.

        Returns 0–100.  100 = captured the entire favourable move.
        Negative when the trade was a loss (exited below entry, MFE was the
        exit itself).
        """
        if mfe_usd <= 0:
            return 0.0
        efficiency = (net_pnl / mfe_usd) * 100.0
        return float(np.clip(efficiency, 0.0, 100.0))

    @staticmethod
    def _compute_stop_quality(
        mae_usd: float,
        trade: dict,
        entry_price: float,
        quantity: int,
    ) -> float:
        """Score how well the stop was placed.

        If the stop was never touched (MAE > stop distance in $ terms):
          100 = stop was perfectly safe throughout the trade.
        If the stop was hit:
          Penalise based on how much adverse excursion occurred before stop.

        Returns 0–100.
        """
        stop_price = trade.get("stop_price")
        if stop_price is None or entry_price <= 0:
            # No stop info; use MAE relative to typical 2% stop
            typical_stop_usd = entry_price * 0.02 * quantity
            if typical_stop_usd <= 0:
                return 50.0
            ratio = abs(mae_usd) / typical_stop_usd
            # Perfect score if MAE never exceeded even a 2% stop
            return float(np.clip(100.0 - ratio * 50.0, 0.0, 100.0))

        stop_distance_usd = abs(entry_price - float(stop_price)) * quantity
        if stop_distance_usd <= 0:
            return 50.0

        mae_abs = abs(mae_usd)
        if mae_abs <= stop_distance_usd:
            # Stop was never breached – score based on how much buffer remained
            buffer_ratio = 1.0 - (mae_abs / stop_distance_usd)
            return float(np.clip(50.0 + buffer_ratio * 50.0, 0.0, 100.0))
        else:
            # Stop was gapped or widened – penalise
            overshoot_ratio = (mae_abs - stop_distance_usd) / stop_distance_usd
            return float(np.clip(50.0 - overshoot_ratio * 50.0, 0.0, 100.0))

    @staticmethod
    def _compute_entry_quality(
        entry_price: float,
        low_series: list[float],
        price_series: list[float],
    ) -> float:
        """Score how close the entry was to the trade's intra-trade low.

        100 = entry was at the exact low; 0 = entry was at the exact high.
        """
        if low_series:
            trade_low = min(low_series)
        elif price_series:
            trade_low = min(price_series)
        else:
            return 50.0  # no data, neutral

        # Need the trade high too
        from_series = price_series or low_series
        if not from_series:
            return 50.0

        trade_high = max(from_series)
        if trade_high <= trade_low:
            return 50.0

        # Lower entry relative to range = better quality
        range_pct = trade_high - trade_low
        entry_pos = (entry_price - trade_low) / range_pct  # 0 = perfect low, 1 = high
        quality = (1.0 - entry_pos) * 100.0
        return float(np.clip(quality, 0.0, 100.0))

    @staticmethod
    def _compute_exit_quality(
        exit_price: float,
        high_series: list[float],
        price_series: list[float],
    ) -> float:
        """Score how close the exit was to the trade's intra-trade high.

        100 = exit was at the exact high; 0 = exit was at the exact low.
        """
        if high_series:
            trade_high = max(high_series)
        elif price_series:
            trade_high = max(price_series)
        else:
            return 50.0

        from_series = price_series or high_series
        if not from_series:
            return 50.0

        trade_low = min(from_series)
        if trade_high <= trade_low:
            return 50.0

        range_pct = trade_high - trade_low
        exit_pos = (exit_price - trade_low) / range_pct  # 1 = perfect high, 0 = low
        quality = exit_pos * 100.0
        return float(np.clip(quality, 0.0, 100.0))

    @staticmethod
    def _compute_time_quality(
        price_series: list[float],
        entry_price: float,
        hold_days: int,
    ) -> float:
        """Score whether the holding period was optimal.

        Measured as the fraction of days the close was above the entry price
        (i.e., days spent in profitable territory).  Higher = better.
        Returns 0–100.
        """
        if not price_series or hold_days <= 0:
            return 50.0

        above_entry = sum(1 for p in price_series if p > entry_price)
        quality = (above_entry / len(price_series)) * 100.0
        return float(np.clip(quality, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Internal: batch aggregation
    # ------------------------------------------------------------------

    def _aggregate_reports(self, reports: list[PostTradeReport]) -> BatchTradeAnalysis:
        """Aggregate a list of PostTradeReport objects into BatchTradeAnalysis."""
        n = len(reports)
        winners = [r for r in reports if r.is_winner]
        losers = [r for r in reports if not r.is_winner]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / n if n > 0 else 0.0

        def _mean(values: list[float]) -> float:
            return float(statistics.mean(values)) if values else 0.0

        def _median(values: list[float]) -> float:
            return float(statistics.median(values)) if values else 0.0

        avg_pnl_pct = _mean([r.pnl_pct for r in reports])
        avg_winner_pnl_pct = _mean([r.pnl_pct for r in winners])
        avg_loser_pnl_pct = _mean([r.pnl_pct for r in losers])

        gross_profit = sum(r.net_pnl for r in winners)
        gross_loss = abs(sum(r.net_pnl for r in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        avg_efficiency = _mean([r.efficiency for r in reports])
        avg_mfe_pct = _mean([r.mfe_pct for r in reports])
        avg_mae_pct = _mean([abs(r.mae_pct) for r in reports])
        avg_entry_quality = _mean([r.entry_quality for r in reports])
        avg_exit_quality = _mean([r.exit_quality for r in reports])
        avg_hold_days = _mean([float(r.hold_days) for r in reports])

        # Optimal stop/TP from data
        # Optimal stop = median MAE% of losers (how far price went against before reverting)
        optimal_stop_pct = _median([abs(r.mae_pct) for r in losers]) if losers else 0.0
        # Optimal TP = median MFE% of winners
        optimal_tp_pct = _median([r.mfe_pct for r in winners]) if winners else 0.0

        # Signal accuracy analysis
        signal_accuracy = self._compute_signal_accuracy(winners, losers)

        # Exit reason breakdown
        exit_breakdown = self._compute_exit_breakdown(reports)

        analysis = BatchTradeAnalysis(
            trade_count=n,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl_pct,
            avg_winner_pnl_pct=avg_winner_pnl_pct,
            avg_loser_pnl_pct=avg_loser_pnl_pct,
            profit_factor=profit_factor,
            avg_efficiency=avg_efficiency,
            avg_mfe_pct=avg_mfe_pct,
            avg_mae_pct=avg_mae_pct,
            avg_entry_quality=avg_entry_quality,
            avg_exit_quality=avg_exit_quality,
            avg_hold_days=avg_hold_days,
            signal_accuracy=signal_accuracy,
            exit_reason_breakdown=exit_breakdown,
            optimal_stop_pct=optimal_stop_pct,
            optimal_tp_pct=optimal_tp_pct,
            reports=reports,
        )

        logger.info(
            "post_trade.batch_analyzed",
            trade_count=n,
            win_rate=round(win_rate, 3),
            avg_pnl_pct=round(avg_pnl_pct, 2),
            profit_factor=round(min(profit_factor, 999.0), 2),
            avg_efficiency=round(avg_efficiency, 1),
        )
        return analysis

    @staticmethod
    def _compute_signal_accuracy(
        winners: list[PostTradeReport],
        losers: list[PostTradeReport],
    ) -> list[SignalAccuracy]:
        """Rank signal components by their predictive separation between winners/losers."""
        # Gather all component names
        all_components: set[str] = set()
        for r in winners + losers:
            all_components.update(r.signal_components.keys())

        if not all_components:
            return []

        results: list[SignalAccuracy] = []

        for comp in all_components:
            winner_scores = [
                r.signal_components[comp] for r in winners if comp in r.signal_components
            ]
            loser_scores = [
                r.signal_components[comp] for r in losers if comp in r.signal_components
            ]

            avg_w = float(statistics.mean(winner_scores)) if winner_scores else 0.0
            avg_l = float(statistics.mean(loser_scores)) if loser_scores else 0.0
            separation = avg_w - avg_l
            sample = len(winner_scores) + len(loser_scores)

            # Win rate when this component was above its median score
            all_scores = [(s, True) for s in winner_scores] + [
                (s, False) for s in loser_scores
            ]
            if all_scores:
                median_score = statistics.median(s for s, _ in all_scores)
                high_score_trades = [(s, w) for s, w in all_scores if s >= median_score]
                wr = (
                    sum(1 for _, w in high_score_trades if w) / len(high_score_trades)
                    if high_score_trades
                    else 0.0
                )
            else:
                wr = 0.0

            results.append(
                SignalAccuracy(
                    component_name=comp,
                    win_rate=wr,
                    avg_score_winners=avg_w,
                    avg_score_losers=avg_l,
                    separation=separation,
                    sample_size=sample,
                )
            )

        # Sort by separation descending (most predictive first)
        results.sort(key=lambda x: -x.separation)
        return results

    @staticmethod
    def _compute_exit_breakdown(
        reports: list[PostTradeReport],
    ) -> dict[str, dict[str, Any]]:
        """Compute per-exit-reason stats."""
        buckets: dict[str, list[PostTradeReport]] = defaultdict(list)
        for r in reports:
            buckets[r.exit_reason].append(r)

        breakdown: dict[str, dict[str, Any]] = {}
        for reason, group in buckets.items():
            n = len(group)
            wins = sum(1 for r in group if r.is_winner)
            avg_pnl = statistics.mean(r.pnl_pct for r in group) if group else 0.0
            breakdown[reason] = {
                "count": n,
                "win_rate": wins / n if n > 0 else 0.0,
                "avg_pnl_pct": round(avg_pnl, 2),
            }
        return breakdown
