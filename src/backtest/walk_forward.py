"""
WalkForwardValidator: out-of-sample validation via rolling train/test windows.

Process
-------
1. Split data into overlapping train/test windows.
2. For each window, run BacktestEngine on the train split (in-sample)
   and again on the test split (out-of-sample).
3. Compare IS vs OOS metrics to detect overfitting.
4. Support ablation testing: disable individual signal groups.

Example
-------
::

    wfv = WalkForwardValidator(
        data=price_data,
        train_size=252,   # 1 year training
        test_size=63,     # 1 quarter testing
        step_size=21,     # roll forward 1 month at a time
    )
    report = wfv.run()
    print(report.overfitting_detected)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """A single train/test window."""

    window_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None

    @property
    def efficiency_ratio(self) -> float:
        """OOS return / IS return. > 0.5 is acceptable; < 0.3 suggests overfitting."""
        if self.train_result is None or self.test_result is None:
            return 0.0
        is_ret = self.train_result.total_return_pct
        oos_ret = self.test_result.total_return_pct
        if abs(is_ret) < 0.001:
            return 0.0
        return round(oos_ret / is_ret, 4)


@dataclass
class WalkForwardReport:
    """Aggregate results across all walk-forward windows."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    ablation_results: dict[str, list[BacktestResult]] = field(default_factory=dict)

    # Summary metrics
    is_avg_return: float = 0.0     # Average in-sample return per window
    oos_avg_return: float = 0.0    # Average out-of-sample return per window
    is_avg_sharpe: float = 0.0
    oos_avg_sharpe: float = 0.0
    is_win_rate: float = 0.0
    oos_win_rate: float = 0.0
    avg_efficiency_ratio: float = 0.0

    # Overfitting flag
    overfitting_detected: bool = False
    overfitting_reason: str = ""

    # Concatenated OOS equity curve (stitched)
    oos_equity_curve: list[float] = field(default_factory=list)
    oos_combined_return_pct: float = 0.0
    oos_max_drawdown_pct: float = 0.0


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Run walk-forward validation on historical data.

    Parameters
    ----------
    data : pd.DataFrame | dict[str, pd.DataFrame]
        Historical OHLCV data (same format as BacktestEngine).
    train_size : int
        Number of trading days in each training window. Default 252 (1 year).
    test_size : int
        Number of trading days in each test window. Default 63 (1 quarter).
    step_size : int
        Number of days to advance the window each step. Default 21 (1 month).
    config : dict | object | None
        BacktestEngine config passed to each run.
    universe : list[str] | None
        Ticker universe to restrict simulation to.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        config: dict | Any | None = None,
        universe: list[str] | None = None,
    ) -> None:
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.config = config or {}
        self.universe = universe

        # Import MarketSimulator to resolve trading days
        from src.backtest.simulator import MarketSimulator
        sim = MarketSimulator(data)
        self._all_days: list[date] = sim.get_trading_days()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ablation_groups: list[str] | None = None) -> WalkForwardReport:
        """Run all walk-forward windows and return aggregate report.

        Parameters
        ----------
        ablation_groups : list[str] | None
            Optional list of signal group names to test individually disabled.
            Each group is run as a separate backtest with that signal zeroed out.
            Example: ["technical", "news", "sentiment"]
        """
        windows = self._build_windows()
        if not windows:
            logger.warning("walk_forward: no windows could be built")
            return WalkForwardReport()

        completed: list[WalkForwardWindow] = []
        for w in windows:
            logger.info(
                "walk_forward_window",
                idx=w.window_idx,
                train=f"{w.train_start} → {w.train_end}",
                test=f"{w.test_start} → {w.test_end}",
            )
            # In-sample
            w.train_result = self._run_window(w.train_start, w.train_end)
            # Out-of-sample
            w.test_result = self._run_window(w.test_start, w.test_end)
            completed.append(w)

        report = self._aggregate(completed)

        # Ablation testing
        if ablation_groups:
            for group in ablation_groups:
                ablation_results = self._run_ablation(windows, group)
                report.ablation_results[group] = ablation_results

        return report

    def build_windows(self) -> list[WalkForwardWindow]:
        """Expose window list for inspection without running backtests."""
        return self._build_windows()

    # ------------------------------------------------------------------
    # Window building
    # ------------------------------------------------------------------

    def _build_windows(self) -> list[WalkForwardWindow]:
        """Build train/test date windows from available trading days."""
        days = self._all_days
        total_needed = self.train_size + self.test_size

        if len(days) < total_needed:
            logger.warning(
                "walk_forward: insufficient data",
                available=len(days),
                needed=total_needed,
            )
            return []

        windows: list[WalkForwardWindow] = []
        idx = 0
        window_idx = 0

        while idx + total_needed <= len(days):
            train_days = days[idx: idx + self.train_size]
            test_days = days[idx + self.train_size: idx + self.train_size + self.test_size]

            if not train_days or not test_days:
                break

            windows.append(WalkForwardWindow(
                window_idx=window_idx,
                train_start=train_days[0],
                train_end=train_days[-1],
                test_start=test_days[0],
                test_end=test_days[-1],
            ))
            idx += self.step_size
            window_idx += 1

        return windows

    # ------------------------------------------------------------------
    # Running a single window
    # ------------------------------------------------------------------

    def _run_window(self, start: date, end: date) -> BacktestResult:
        """Run BacktestEngine for [start, end] and return result."""
        engine = BacktestEngine(
            data=self.data,
            config=self.config,
            start_date=str(start),
            end_date=str(end),
            universe=self.universe,
        )
        try:
            return engine.run()
        except Exception as exc:
            logger.warning("walk_forward_window_failed", error=str(exc))
            return BacktestResult(
                start_date=str(start),
                end_date=str(end),
                initial_capital=float(
                    self.config.get("initial_capital", 100_000)
                    if isinstance(self.config, dict)
                    else getattr(self.config, "initial_capital", 100_000)
                ),
            )

    # ------------------------------------------------------------------
    # Ablation testing
    # ------------------------------------------------------------------

    def _run_ablation(
        self,
        windows: list[WalkForwardWindow],
        disabled_group: str,
    ) -> list[BacktestResult]:
        """Run OOS windows with one signal group disabled.

        The ablation is implemented by temporarily modifying the config to
        set the relevant weight to zero, so that signal group contributes
        nothing to the final score.
        """
        # Build an ablated config
        ablated_config = dict(self.config) if isinstance(self.config, dict) else {
            k: getattr(self.config, k)
            for k in dir(self.config)
            if not k.startswith("_")
        }

        weight_map = {
            "technical": "technical_min_score",
            "news": "news_weight",
            "sentiment": "sentiment_weight",
        }
        if disabled_group in weight_map:
            key = weight_map[disabled_group]
            # For technical: raise the min score to block all entries
            if disabled_group == "technical":
                ablated_config["technical_min_score"] = 999.0
            else:
                ablated_config[key] = 0.0

        results: list[BacktestResult] = []
        for w in windows:
            engine = BacktestEngine(
                data=self.data,
                config=ablated_config,
                start_date=str(w.test_start),
                end_date=str(w.test_end),
                universe=self.universe,
            )
            try:
                results.append(engine.run())
            except Exception as exc:
                logger.warning("ablation_run_failed", group=disabled_group, error=str(exc))

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, windows: list[WalkForwardWindow]) -> WalkForwardReport:
        """Aggregate per-window metrics into a single report."""
        report = WalkForwardReport(windows=windows)

        is_returns = [w.train_result.total_return_pct for w in windows if w.train_result]
        oos_returns = [w.test_result.total_return_pct for w in windows if w.test_result]
        is_sharpes = [w.train_result.sharpe_ratio for w in windows if w.train_result]
        oos_sharpes = [w.test_result.sharpe_ratio for w in windows if w.test_result]
        is_win_rates = [w.train_result.win_rate for w in windows if w.train_result]
        oos_win_rates = [w.test_result.win_rate for w in windows if w.test_result]
        efficiency_ratios = [w.efficiency_ratio for w in windows]

        if is_returns:
            report.is_avg_return = round(float(np.mean(is_returns)), 3)
        if oos_returns:
            report.oos_avg_return = round(float(np.mean(oos_returns)), 3)
        if is_sharpes:
            report.is_avg_sharpe = round(float(np.mean(is_sharpes)), 3)
        if oos_sharpes:
            report.oos_avg_sharpe = round(float(np.mean(oos_sharpes)), 3)
        if is_win_rates:
            report.is_win_rate = round(float(np.mean(is_win_rates)), 2)
        if oos_win_rates:
            report.oos_win_rate = round(float(np.mean(oos_win_rates)), 2)
        if efficiency_ratios:
            report.avg_efficiency_ratio = round(float(np.mean(efficiency_ratios)), 4)

        # Stitch OOS equity curves
        combined_curve: list[float] = []
        for w in windows:
            if w.test_result and w.test_result.equity_curve:
                if combined_curve:
                    # Scale to continue from last equity value
                    scale = combined_curve[-1] / w.test_result.equity_curve[0] if w.test_result.equity_curve[0] > 0 else 1.0
                    combined_curve.extend([v * scale for v in w.test_result.equity_curve[1:]])
                else:
                    combined_curve.extend(w.test_result.equity_curve)
        report.oos_equity_curve = combined_curve

        if combined_curve and len(combined_curve) >= 2:
            report.oos_combined_return_pct = round(
                (combined_curve[-1] - combined_curve[0]) / combined_curve[0] * 100.0, 3
            )
            report.oos_max_drawdown_pct = round(
                BacktestEngine._max_drawdown(combined_curve), 3
            )

        # Overfitting detection heuristics
        if len(is_returns) >= 2 and len(oos_returns) >= 2:
            if report.avg_efficiency_ratio < 0.3 and report.is_avg_return > 5.0:
                report.overfitting_detected = True
                report.overfitting_reason = (
                    f"Efficiency ratio {report.avg_efficiency_ratio:.3f} < 0.30 "
                    f"while IS avg return is {report.is_avg_return:.2f}%."
                )
            elif report.is_avg_return > 10.0 and report.oos_avg_return < 0.0:
                report.overfitting_detected = True
                report.overfitting_reason = (
                    f"IS avg return {report.is_avg_return:.2f}% but OOS avg return "
                    f"{report.oos_avg_return:.2f}% is negative."
                )
            elif (report.is_win_rate - report.oos_win_rate) > 20.0:
                report.overfitting_detected = True
                report.overfitting_reason = (
                    f"IS win rate {report.is_win_rate:.1f}% vs OOS {report.oos_win_rate:.1f}%: "
                    f"gap > 20pp suggests overfitting."
                )

        return report
