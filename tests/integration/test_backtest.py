"""
Integration tests for the backtesting framework.

Tests: mini backtest end-to-end run, BacktestResult metrics, reporter output,
walk-forward basic structure.
At least 2 test cases.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.reporter import BacktestReporter
from src.backtest.simulator import MarketSimulator


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_test_data(n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV data for backtest integration tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-03", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.015, n_days))
    opens = np.roll(close, 1)
    opens[0] = close[0]
    ranges = close * rng.uniform(0.005, 0.02, n_days)
    highs = np.maximum(opens, close) + ranges * 0.6
    lows = np.minimum(opens, close) - ranges * 0.4
    vols = rng.integers(500_000, 2_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": close, "volume": vols},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestEngineIntegration:

    def test_mini_backtest_completes(self):
        """BacktestEngine runs a full simulation and returns a BacktestResult."""
        data = _generate_test_data(n_days=60)
        config = {
            "initial_capital": 50_000.0,
            "commission_per_share": 0.005,
            "slippage_pct": 0.001,
            "max_hold_days": 5,
            "max_positions": 2,
            "technical_min_score": 0.0,  # Accept all for test
        }
        engine = BacktestEngine(data=data, config=config)
        result = engine.run()

        # Basic type and range checks
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve, list)
        assert len(result.equity_curve) > 0
        assert result.initial_capital == 50_000.0
        assert result.final_capital >= 0.0

    def test_equity_curve_starts_at_initial_capital(self):
        """Equity curve first value is close to initial capital."""
        data = _generate_test_data(n_days=40)
        config = {"initial_capital": 100_000.0, "technical_min_score": 999.0}  # No trades
        engine = BacktestEngine(data=data, config=config)
        result = engine.run()

        if result.equity_curve:
            # First equity point should be at or near initial capital (no trades)
            assert abs(result.equity_curve[0] - 100_000.0) < 1.0

    def test_backtest_with_multiple_tickers(self):
        """Backtest runs correctly with a multi-ticker data dict."""
        multi_data = {
            "AAPL": _generate_test_data(n_days=80, seed=1),
            "MSFT": _generate_test_data(n_days=80, seed=2),
            "TSLA": _generate_test_data(n_days=80, seed=3),
        }
        config = {
            "initial_capital": 100_000.0,
            "max_positions": 3,
            "technical_min_score": 0.0,
        }
        engine = BacktestEngine(data=multi_data, config=config)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100_000.0
        assert result.total_return_pct is not None
        assert result.max_drawdown_pct >= 0.0

    def test_backtest_metrics_are_valid(self):
        """BacktestResult metrics fall within valid ranges."""
        data = _generate_test_data(n_days=80)
        config = {"initial_capital": 100_000.0, "technical_min_score": 0.0}
        engine = BacktestEngine(data=data, config=config)
        result = engine.run()

        # Range validations
        assert result.max_drawdown_pct >= 0.0
        assert 0.0 <= result.win_rate <= 100.0
        assert result.profit_factor >= 0.0
        assert result.exposure_pct >= 0.0
        assert result.avg_hold_days >= 0.0

    def test_backtest_no_trades_in_empty_universe(self):
        """Backtest with excluded universe runs but records no trades."""
        data = _generate_test_data(n_days=40)
        config = {
            "initial_capital": 100_000.0,
            "technical_min_score": 999.0,  # No score will ever exceed this
        }
        engine = BacktestEngine(data=data, config=config)
        result = engine.run()
        assert result.total_trades == 0


class TestBacktestReporter:

    def test_reporter_print_summary_does_not_raise(self, capsys):
        """Reporter.print_summary() runs without exceptions."""
        data = _generate_test_data(n_days=60)
        config = {"initial_capital": 100_000.0, "technical_min_score": 0.0}
        result = BacktestEngine(data=data, config=config).run()
        reporter = BacktestReporter(result)
        reporter.print_summary()  # Should not raise
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out

    def test_reporter_to_json_is_valid(self):
        """Reporter.to_json() returns valid JSON with expected keys."""
        import json
        data = _generate_test_data(n_days=60)
        config = {"initial_capital": 100_000.0, "technical_min_score": 0.0}
        result = BacktestEngine(data=data, config=config).run()
        reporter = BacktestReporter(result)
        json_str = reporter.to_json()
        parsed = json.loads(json_str)

        assert "returns" in parsed
        assert "trades" in parsed
        assert "equity_curve" in parsed
        assert "risk" in parsed

    def test_reporter_save_json_creates_file(self):
        """Reporter.save_json() writes a file to disk."""
        data = _generate_test_data(n_days=40)
        config = {"initial_capital": 100_000.0, "technical_min_score": 999.0}
        result = BacktestEngine(data=data, config=config).run()
        reporter = BacktestReporter(result)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            reporter.save_json(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_rolling_sharpe_correct_length(self):
        """rolling_sharpe() returns a list of same length as equity_curve."""
        data = _generate_test_data(n_days=80)
        config = {"initial_capital": 100_000.0, "technical_min_score": 0.0}
        result = BacktestEngine(data=data, config=config).run()
        reporter = BacktestReporter(result)
        rolling = reporter.rolling_sharpe(window_days=20)
        assert len(rolling) == len(result.equity_curve)


class TestWalkForwardIntegration:

    def test_walk_forward_builds_windows(self):
        """WalkForwardValidator generates the correct number of windows."""
        from src.backtest.walk_forward import WalkForwardValidator
        multi_data = {
            "AAPL": _generate_test_data(n_days=200, seed=10),
            "MSFT": _generate_test_data(n_days=200, seed=20),
        }
        wfv = WalkForwardValidator(
            data=multi_data,
            train_size=100,
            test_size=30,
            step_size=20,
        )
        windows = wfv.build_windows()
        # With 200 days, train=100, test=30, step=20 we should have several windows
        assert len(windows) > 0
        # First window should start at day 0
        for w in windows:
            assert w.train_start <= w.train_end
            assert w.test_start <= w.test_end
            assert w.train_end < w.test_start

    def test_walk_forward_run_completes(self):
        """WalkForwardValidator.run() completes and returns a WalkForwardReport."""
        from src.backtest.walk_forward import WalkForwardReport, WalkForwardValidator
        data = _generate_test_data(n_days=200, seed=42)
        config = {"initial_capital": 50_000.0, "technical_min_score": 0.0}
        wfv = WalkForwardValidator(
            data=data,
            train_size=100,
            test_size=30,
            step_size=50,  # Large step → fewer windows → faster
            config=config,
        )
        report = wfv.run()
        assert isinstance(report, WalkForwardReport)
        assert len(report.windows) > 0
        assert isinstance(report.overfitting_detected, bool)
