"""
BacktestReporter: format and persist BacktestResult objects.

Provides:
- Console summary (print_summary)
- JSON serialization (to_json / save_json)
- Rolling metrics computation (rolling_sharpe, etc.)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

from src.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

_RISK_FREE_RATE_ANNUAL = 0.045


class BacktestReporter:
    """Format and persist a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        The result to report on.
    """

    def __init__(self, result: BacktestResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        r = self.result
        sep = "=" * 60

        print(sep)
        print("  BACKTEST RESULTS SUMMARY")
        print(sep)
        print(f"  Period         : {r.start_date} → {r.end_date}")
        print(f"  Initial Capital: ${r.initial_capital:>12,.2f}")
        print(f"  Final Capital  : ${r.final_capital:>12,.2f}")
        print(sep)
        print("  RETURNS")
        print(f"    Total Return    : {r.total_return_pct:>8.2f}%")
        print(f"    Annualized CAGR : {r.annualized_return_pct:>8.2f}%")
        print(f"    Max Drawdown    : {r.max_drawdown_pct:>8.2f}%")
        print(f"    Sharpe Ratio    : {r.sharpe_ratio:>8.3f}")
        print(f"    Sortino Ratio   : {r.sortino_ratio:>8.3f}")
        print(f"    Exposure        : {r.exposure_pct:>8.2f}%")
        print(sep)
        print("  TRADE STATISTICS")
        print(f"    Total Trades    : {r.total_trades:>8d}")
        print(f"    Win Rate        : {r.win_rate:>8.2f}%")
        print(f"    Profit Factor   : {r.profit_factor:>8.3f}")
        print(f"    Expectancy      : {r.expectancy:>8.3f}%")
        print(f"    Avg Hold (days) : {r.avg_hold_days:>8.2f}")
        print(f"    Winners         : {r.winning_trades:>8d}")
        print(f"    Losers          : {r.losing_trades:>8d}")
        print(f"    Avg Win         : {r.avg_win_pct:>8.3f}%")
        print(f"    Avg Loss        : {r.avg_loss_pct:>8.3f}%")
        print(f"    Max Consec Loss : {r.max_consecutive_losses:>8d}")
        print(sep)
        if r.monthly_returns:
            print("  MONTHLY RETURNS")
            for month, ret in sorted(r.monthly_returns.items()):
                bar = "+" * max(0, int(ret)) if ret >= 0 else "-" * max(0, int(abs(ret)))
                print(f"    {month} : {ret:>7.2f}% {bar}")
            print(sep)
        print(f"  Commission Paid : ${r.commission_paid:>10,.2f}")
        print(f"  Slippage Paid   : ${r.slippage_paid:>10,.2f}")
        print(sep)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return result as a JSON-serializable dictionary."""
        r = self.result
        d: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "period": {"start": r.start_date, "end": r.end_date},
            "capital": {
                "initial": r.initial_capital,
                "final": r.final_capital,
                "commission_paid": r.commission_paid,
                "slippage_paid": r.slippage_paid,
            },
            "returns": {
                "total_return_pct": r.total_return_pct,
                "annualized_return_pct": r.annualized_return_pct,
            },
            "risk": {
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "exposure_pct": r.exposure_pct,
            },
            "trades": {
                "total": r.total_trades,
                "winners": r.winning_trades,
                "losers": r.losing_trades,
                "win_rate": r.win_rate,
                "profit_factor": (
                    r.profit_factor if not math.isinf(r.profit_factor) else 9999.0
                ),
                "expectancy": r.expectancy,
                "avg_hold_days": r.avg_hold_days,
                "avg_win_pct": r.avg_win_pct,
                "avg_loss_pct": r.avg_loss_pct,
                "max_consecutive_losses": r.max_consecutive_losses,
            },
            "monthly_returns": r.monthly_returns,
            "regime_breakdown": r.regime_breakdown,
            "equity_curve": [round(v, 2) for v in r.equity_curve],
            "trade_log": r.trades,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, filepath: str) -> None:
        """Write JSON report to a file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info("backtest_report_saved", path=filepath)

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window_days: int = 30) -> list[float | None]:
        """Compute rolling Sharpe ratio over the equity curve.

        Returns a list of the same length as equity_curve; first
        (window_days - 1) entries are None.
        """
        eq = self.result.equity_curve
        if len(eq) < 2:
            return [None] * len(eq)

        returns = np.diff(eq) / np.array(eq[:-1], dtype=float)
        rf_daily = (1 + _RISK_FREE_RATE_ANNUAL) ** (1 / 252) - 1
        excess = returns - rf_daily

        results: list[float | None] = [None]
        for i in range(len(excess)):
            if i + 1 < window_days:
                results.append(None)
                continue
            window = excess[i + 1 - window_days: i + 1]
            std = float(np.std(window, ddof=1))
            sharpe = (
                float(np.mean(window)) / std * math.sqrt(252) if std > 0 else 0.0
            )
            results.append(round(sharpe, 3))

        return results

    def rolling_return(self, window_days: int = 30) -> list[float | None]:
        """Compute rolling window return (%) over the equity curve."""
        eq = self.result.equity_curve
        if len(eq) < window_days:
            return [None] * len(eq)

        results: list[float | None] = [None] * (window_days - 1)
        for i in range(window_days - 1, len(eq)):
            start_val = eq[i - window_days + 1]
            end_val = eq[i]
            ret = (end_val - start_val) / start_val * 100.0 if start_val > 0 else 0.0
            results.append(round(ret, 3))

        return results

    def rolling_drawdown(self) -> list[float]:
        """Compute rolling drawdown from peak for each point in equity curve."""
        eq = self.result.equity_curve
        if not eq:
            return []
        arr = np.array(eq, dtype=float)
        peak = arr[0]
        drawdowns: list[float] = []
        for val in arr:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
            drawdowns.append(round(dd, 3))
        return drawdowns

    # ------------------------------------------------------------------
    # Quick stats helpers
    # ------------------------------------------------------------------

    def win_loss_breakdown(self) -> dict[str, Any]:
        """Return a breakdown table by exit reason."""
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)
        for t in self.result.trades:
            buckets[t.get("exit_reason", "unknown")].append(t.get("pnl_pct", 0.0))
        breakdown: dict[str, Any] = {}
        for reason, pnls in buckets.items():
            winners = [p for p in pnls if p > 0]
            breakdown[reason] = {
                "count": len(pnls),
                "win_rate": round(len(winners) / len(pnls) * 100.0, 2) if pnls else 0.0,
                "avg_pnl_pct": round(float(np.mean(pnls)), 3) if pnls else 0.0,
            }
        return breakdown
