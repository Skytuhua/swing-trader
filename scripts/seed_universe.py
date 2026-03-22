#!/usr/bin/env python3
"""
scripts/seed_universe.py — Stock Universe Seeding for SwingTrader
==================================================================

Seeds the tradable stock universe with the top 100 S&P 500 components
(by market cap, as of early 2025) hardcoded in this script.

Optionally, if --fetch is passed and ALPACA_API_KEY is set, the script
fetches a fresh asset list from the Alpaca Assets API and merges it.

The seeded universe is written to config/universe.yaml, which the
UniverseService reads at startup.

Usage:
    # Seed with hardcoded top-100 S&P 500 list:
    python scripts/seed_universe.py

    # Seed with hardcoded list AND refresh from Alpaca API:
    python scripts/seed_universe.py --fetch

    # Dry-run: print tickers without writing the file:
    python scripts/seed_universe.py --dry-run

    # Or via Make:
    make seed-universe

Environment variables (read from .env):
    ALPACA_API_KEY    — required only when --fetch is used
    ALPACA_API_SECRET — required only when --fetch is used
    ALPACA_BASE_URL   — defaults to paper endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

try:
    import yaml
except ImportError:
    print("[seed_universe] ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Top-100 S&P 500 components (by market cap, approximate as of early 2025)
# ---------------------------------------------------------------------------
# These tickers are used as the default tradable universe when no live
# API fetch is performed. Update this list periodically (quarterly
# S&P 500 rebalances typically add/remove <10 tickers).
#
# Source: S&P Dow Jones Indices (https://www.spglobal.com/spdji/en/)
# ---------------------------------------------------------------------------
SP500_TOP_100: list[str] = [
    # Mega-cap technology / communication
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "ORCL",
    # Large-cap technology
    "AMD", "QCOM", "INTC", "TXN", "MU", "AMAT", "KLAC", "LRCX", "ADI", "MRVL",
    "NOW", "CRM", "ADBE", "INTU", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "ZS",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
    "SCHW", "C", "USB", "PNC", "TFC",
    # Healthcare / Biotech
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN", "ISRG",
    "BMY", "GILD", "VRTX", "REGN", "BSX",
    # Consumer discretionary
    "MCD", "SBUX", "NKE", "HD", "LOW", "TGT", "BKNG", "CMG", "YUM", "DRI",
    # Consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "GIS", "KMB",
    # Industrials
    "CAT", "DE", "HON", "RTX", "LMT", "GE", "UPS", "FDX", "MMM", "EMR",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities / Real Estate
    "NEE", "DUK", "SO", "AMT", "PLD",
    # Communication services
    "DIS", "NFLX", "T", "VZ",
    # Materials
    "LIN", "APD", "ECL", "DD",
]


# ---------------------------------------------------------------------------
# Alpaca fetch (optional)
# ---------------------------------------------------------------------------

async def fetch_alpaca_tradable_assets() -> list[str]:
    """
    Fetch US equity assets from the Alpaca Assets API.

    Returns a list of ticker symbols that are:
      - Active
      - Tradable
      - Fractionable (better fill quality)
      - Listed on NYSE or NASDAQ

    Requires ALPACA_API_KEY and ALPACA_API_SECRET in the environment.
    """
    try:
        import httpx
    except ImportError:
        print("[seed_universe] WARNING: httpx not installed — skipping Alpaca fetch.")
        return []

    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        print(
            "[seed_universe] WARNING: ALPACA_API_KEY / ALPACA_API_SECRET not set.\n"
            "  Skipping Alpaca asset fetch. Using hardcoded list only."
        )
        return []

    # Use broker (data) API endpoint.
    data_url = "https://data.alpaca.markets"
    broker_url = base_url.replace("paper-api", "api").rstrip("/")
    assets_url = f"{broker_url}/v2/assets"

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params = {
        "status": "active",
        "asset_class": "us_equity",
    }

    print(f"[seed_universe] Fetching assets from: {assets_url}")
    tickers: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(assets_url, headers=headers, params=params)
            response.raise_for_status()
            assets = response.json()

        for asset in assets:
            if (
                asset.get("tradable")
                and asset.get("fractionable")
                and asset.get("exchange") in ("NYSE", "NASDAQ", "ARCA")
                and asset.get("status") == "active"
            ):
                tickers.append(asset["symbol"])

        print(f"[seed_universe] ✓  Fetched {len(tickers)} tradable assets from Alpaca.")
    except Exception as exc:
        print(f"[seed_universe] WARNING: Alpaca fetch failed: {exc}")

    return tickers


# ---------------------------------------------------------------------------
# Universe composition
# ---------------------------------------------------------------------------

def compose_universe(
    hardcoded: list[str],
    fetched: list[str],
    limit: int = 200,
) -> list[str]:
    """
    Merge the hardcoded top-100 list with optionally fetched tickers.

    Deduplication is performed; hardcoded tickers are always included first.
    The final list is capped at `limit` symbols.
    """
    seen: set[str] = set()
    combined: list[str] = []

    for ticker in hardcoded + fetched:
        clean = ticker.strip().upper()
        # Skip empty strings, tickers with '/' (dual-class identifiers), or
        # symbols longer than 5 chars (likely warrants / rights).
        if not clean or "/" in clean or len(clean) > 5:
            continue
        if clean not in seen:
            seen.add(clean)
            combined.append(clean)

    return combined[:limit]


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_universe_yaml(tickers: list[str], output_path: Path) -> None:
    """Write the universe ticker list to a YAML file."""
    data = {
        "universe": {
            "description": (
                "Tradable stock universe — seeded by scripts/seed_universe.py.\n"
                "Top S&P 500 components filtered for liquidity and tradability."
            ),
            "tickers": tickers,
            "count": len(tickers),
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[seed_universe] ✓  Universe written to: {output_path}  ({len(tickers)} tickers)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(fetch: bool = False, dry_run: bool = False, limit: int = 200) -> None:
    print("=" * 60)
    print("  SwingTrader — Stock Universe Seeding")
    print("=" * 60)

    hardcoded = SP500_TOP_100.copy()
    print(f"[seed_universe] Hardcoded tickers: {len(hardcoded)}")

    fetched: list[str] = []
    if fetch:
        fetched = await fetch_alpaca_tradable_assets()

    universe = compose_universe(hardcoded, fetched, limit=limit)
    print(f"[seed_universe] Final universe size: {len(universe)} tickers")

    if dry_run:
        print("[seed_universe] DRY RUN — not writing any files.")
        print(f"  Tickers: {', '.join(universe[:20])} {'...' if len(universe) > 20 else ''}")
        return

    output_path = PROJECT_ROOT / "config" / "universe.yaml"
    write_universe_yaml(universe, output_path)

    print("=" * 60)
    print("  Universe seeding complete.")
    print("  Next step: make run")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the SwingTrader stock universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch a fresh tradable asset list from the Alpaca API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the universe without writing any files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of tickers in the universe.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(fetch=args.fetch, dry_run=args.dry_run, limit=args.limit))
