#!/usr/bin/env python3
"""
CLI script to run a backtest on historical OHLCV data.

Usage
-----
::

    python scripts/run_backtest.py --ticker AAPL --start 2022-01-01 --end 2023-12-31
    python scripts/run_backtest.py --ticker MSFT GOOG --start 2023-01-01 --initial-capital 200000
    python scripts/run_backtest.py --csv data/prices.csv --start 2022-01-01 --end 2023-12-31

Options
-------
--ticker         One or more ticker symbols (requires yfinance)
--csv            Path to a CSV file with columns [date, open, high, low, close, volume]
--start          Start date YYYY-MM-DD
--end            End date YYYY-MM-DD (defaults to today)
--initial-capital  Starting capital in USD (default: 100000)
--commission     Commission per share in dollars (default: 0.005)
--slippage       Slippage fraction per side (default: 0.001)
--max-hold-days  Maximum days to hold a position (default: 5)
--max-positions  Maximum simultaneous positions (default: 5)
--output         Path to save JSON report (optional)
--walk-forward   Run walk-forward validation instead of single backtest
--train-size     Walk-forward training window in days (default: 252)
--test-size      Walk-forward test window in days (default: 63)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_yfinance(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download historical data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is not installed. Run: pip install yfinance")
        sys.exit(1)

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"  Downloading {ticker} ...")
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}")
            continue
        # yfinance returns multi-level columns when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        # Map yfinance columns to standard names
        col_map = {"adj_close": "close", "adj close": "close"}
        raw.rename(columns=col_map, inplace=True)
        needed = {"open", "high", "low", "close", "volume"}
        missing = needed - set(raw.columns)
        if missing:
            print(f"  WARNING: {ticker} missing columns {missing}")
            continue
        data[ticker] = raw[["open", "high", "low", "close", "volume"]]
    return data


def _load_csv(csv_path: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV data from a CSV file.

    The CSV can optionally have a 'ticker' column; if so, it is treated as
    a multi-ticker file. Otherwise it is treated as a single-ticker file.
    """
    df = pd.read_csv(csv_path, parse_dates=True)

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if "ticker" in df.columns:
        data: dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker"):
            needed = ["open", "high", "low", "close", "volume"]
            group = group[[c for c in needed if c in group.columns]]
            data[str(ticker)] = group
        return data

    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]]
    name = os.path.splitext(os.path.basename(csv_path))[0].upper()
    return {name: df}


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SwingTrader Backtest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source (mutually exclusive)
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--ticker", nargs="+", metavar="TICKER",
        help="One or more ticker symbols (requires yfinance)",
    )
    data_group.add_argument(
        "--csv", metavar="PATH",
        help="Path to CSV file with OHLCV data",
    )

    # Date range
    p.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    p.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )

    # Capital & costs
    p.add_argument("--initial-capital", type=float, default=100_000.0,
                   help="Starting capital in USD (default: 100000)")
    p.add_argument("--commission", type=float, default=0.005,
                   help="Commission per share in dollars (default: 0.005)")
    p.add_argument("--slippage", type=float, default=0.001,
                   help="Slippage fraction per side (default: 0.001)")

    # Position management
    p.add_argument("--max-hold-days", type=int, default=5,
                   help="Maximum days to hold a position (default: 5)")
    p.add_argument("--max-positions", type=int, default=5,
                   help="Maximum simultaneous positions (default: 5)")

    # Output
    p.add_argument("--output", metavar="PATH",
                   help="Path to save JSON report")

    # Walk-forward
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward validation")
    p.add_argument("--train-size", type=int, default=252,
                   help="Walk-forward training window in days (default: 252)")
    p.add_argument("--test-size", type=int, default=63,
                   help="Walk-forward test window in days (default: 63)")
    p.add_argument("--step-size", type=int, default=21,
                   help="Walk-forward step size in days (default: 21)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---- Load data ----
    print("\n[1/3] Loading market data...")
    if args.ticker:
        data = _load_yfinance(args.ticker, args.start, args.end)
    else:
        data = _load_csv(args.csv)

    if not data:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print(f"  Loaded {len(data)} ticker(s): {', '.join(sorted(data.keys()))}")

    # ---- Build config ----
    config = {
        "initial_capital": args.initial_capital,
        "commission_per_share": args.commission,
        "slippage_pct": args.slippage,
        "max_hold_days": args.max_hold_days,
        "max_positions": args.max_positions,
    }

    # ---- Run backtest ----
    from src.backtest.engine import BacktestEngine
    from src.backtest.reporter import BacktestReporter

    if args.walk_forward:
        from src.backtest.walk_forward import WalkForwardValidator

        print("\n[2/3] Running walk-forward validation...")
        wfv = WalkForwardValidator(
            data=data,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            config=config,
        )
        report = wfv.run()
        print(f"\n[3/3] Walk-Forward Summary")
        print(f"  Windows         : {len(report.windows)}")
        print(f"  IS Avg Return   : {report.is_avg_return:.2f}%")
        print(f"  OOS Avg Return  : {report.oos_avg_return:.2f}%")
        print(f"  IS Avg Sharpe   : {report.is_avg_sharpe:.3f}")
        print(f"  OOS Avg Sharpe  : {report.oos_avg_sharpe:.3f}")
        print(f"  IS Win Rate     : {report.is_win_rate:.2f}%")
        print(f"  OOS Win Rate    : {report.oos_win_rate:.2f}%")
        print(f"  Efficiency Ratio: {report.avg_efficiency_ratio:.4f}")
        print(f"  Overfitting     : {'YES ⚠️' if report.overfitting_detected else 'No'}")
        if report.overfitting_detected:
            print(f"    Reason: {report.overfitting_reason}")
        print(f"  OOS Return      : {report.oos_combined_return_pct:.2f}%")
        print(f"  OOS Max DD      : {report.oos_max_drawdown_pct:.2f}%")

        if args.output:
            out = {
                "type": "walk_forward",
                "is_avg_return": report.is_avg_return,
                "oos_avg_return": report.oos_avg_return,
                "is_avg_sharpe": report.is_avg_sharpe,
                "oos_avg_sharpe": report.oos_avg_sharpe,
                "is_win_rate": report.is_win_rate,
                "oos_win_rate": report.oos_win_rate,
                "avg_efficiency_ratio": report.avg_efficiency_ratio,
                "overfitting_detected": report.overfitting_detected,
                "overfitting_reason": report.overfitting_reason,
                "oos_combined_return_pct": report.oos_combined_return_pct,
                "oos_max_drawdown_pct": report.oos_max_drawdown_pct,
                "windows": [
                    {
                        "idx": w.window_idx,
                        "train": f"{w.train_start} → {w.train_end}",
                        "test": f"{w.test_start} → {w.test_end}",
                        "is_return": w.train_result.total_return_pct if w.train_result else 0.0,
                        "oos_return": w.test_result.total_return_pct if w.test_result else 0.0,
                        "efficiency": w.efficiency_ratio,
                    }
                    for w in report.windows
                ],
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"\n  Report saved to: {args.output}")
    else:
        print("\n[2/3] Running backtest...")
        engine = BacktestEngine(
            data=data,
            config=config,
            start_date=args.start,
            end_date=args.end,
        )
        result = engine.run()
        reporter = BacktestReporter(result)

        print()
        reporter.print_summary()

        if args.output:
            reporter.save_json(args.output)
            print(f"\n  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
