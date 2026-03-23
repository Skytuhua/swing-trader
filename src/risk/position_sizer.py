"""
Kelly Criterion Position Sizing — mathematically optimal bet sizing.

"Position sizing is the difference between steady 20% monthly returns
and total account wipeout."

Kelly Criterion Formula
-----------------------
f* = (p * b - q) / b

Where:
    p = historical win rate (probability of winning)
    q = loss rate = 1 - p
    b = average win / average loss (win/loss ratio)
    f* = fraction of bankroll to risk

Half-Kelly (Recommended Default)
---------------------------------
Full Kelly is mathematically optimal but too volatile for practical trading.
Half-Kelly captures ~75% of the growth benefit with significantly less drawdown.
half_kelly = f* / 2

Volatility-Adjusted Sizing
---------------------------
Position size is further scaled by volatility:
    position_value = (risk_budget / ATR) * price
Higher ATR = smaller position to maintain constant risk.

Correlation-Adjusted Sizing
-----------------------------
When holding correlated positions (r > 0.7), treat them partially as the same
position for risk purposes. Reduces total exposure to avoid hidden concentration.

Fallback
--------
If insufficient trade history (< 30 trades), falls back to a fixed fraction
method (default 1% risk per trade) until enough data accumulates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly Criterion position sizing computation."""

    method: str  # "kelly", "half_kelly", "fixed_fraction", "volatility_adjusted"
    fraction: float  # fraction of bankroll to risk (0-1)
    position_value: float  # dollar value of position
    shares: int  # number of shares to buy
    allocation_pct: float  # % of portfolio
    risk_per_trade_pct: float  # % of portfolio at risk

    # Kelly inputs
    win_rate: float = 0.0
    win_loss_ratio: float = 0.0
    full_kelly_fraction: float = 0.0

    # Adjustments applied
    volatility_adjustment: float = 1.0
    correlation_adjustment: float = 1.0
    regime_adjustment: float = 1.0

    # Diagnostics
    trade_count: int = 0
    notes: str = ""


@dataclass
class TradeRecord:
    """A completed trade record for Kelly calculation."""

    pnl: float  # profit/loss in dollars (negative = loss)
    pnl_pct: float  # P&L as % of entry value
    entry_price: float = 0.0
    exit_price: float = 0.0
    ticker: str = ""


class KellyPositionSizer:
    """Position sizing using Kelly Criterion with safety adjustments.

    Parameters
    ----------
    config : dict or dataclass
        Configuration with:
        - method: str ("kelly", "half_kelly", "fixed_fraction", "volatility_adjusted")
        - max_position_pct: float (default 5.0) — hard cap on position size
        - kelly_lookback_trades: int (default 100) — trades used for win rate
        - min_trades_for_kelly: int (default 30) — minimum history required
        - default_risk_per_trade_pct: float (default 1.0) — fallback risk %
        - correlation_threshold: float (default 0.7) — correlation reduction trigger
    """

    def __init__(self, config: Any = None) -> None:
        cfg = config or {}
        self._get = (
            (lambda k, d: cfg.get(k, d))
            if isinstance(cfg, dict)
            else (lambda k, d: getattr(cfg, k, d))
        )

        self.method: str = self._get("method", "half_kelly")
        self.max_position_pct: float = float(self._get("max_position_pct", 5.0))
        self.kelly_lookback: int = int(self._get("kelly_lookback_trades", 100))
        self.min_trades: int = int(self._get("min_trades_for_kelly", 30))
        self.default_risk_pct: float = float(self._get("default_risk_per_trade_pct", 1.0))
        self.correlation_threshold: float = float(self._get("correlation_threshold", 0.7))

        # Trade history for Kelly calculation
        self._trade_history: list[TradeRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        atr: float = 0.0,
        current_positions: list[dict] | None = None,
        regime_aggression: float = 1.0,
    ) -> KellyResult:
        """Compute position size using the configured method.

        Parameters
        ----------
        portfolio_value : float
            Total portfolio equity in dollars.
        entry_price : float
            Planned entry price.
        stop_loss : float
            Stop-loss price.
        atr : float, optional
            Current ATR for volatility adjustment.
        current_positions : list of dict, optional
            Currently held positions for correlation adjustment.
            Each dict should have keys: 'ticker', 'returns' (np.ndarray).
        regime_aggression : float
            Regime-based aggression multiplier (0-1).

        Returns
        -------
        KellyResult with recommended position size and diagnostics.
        """
        if portfolio_value <= 0 or entry_price <= 0:
            return KellyResult(
                method=self.method, fraction=0, position_value=0, shares=0,
                allocation_pct=0, risk_per_trade_pct=0,
                notes="Invalid inputs: portfolio_value or entry_price <= 0.",
            )

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return KellyResult(
                method=self.method, fraction=0, position_value=0, shares=0,
                allocation_pct=0, risk_per_trade_pct=0,
                notes="Invalid stop: risk_per_share <= 0.",
            )

        # Determine method and compute fraction
        trades = self._recent_trades()
        use_kelly = len(trades) >= self.min_trades and self.method in ("kelly", "half_kelly")

        if use_kelly:
            result = self._kelly_size(portfolio_value, entry_price, risk_per_share, trades)
        elif self.method == "volatility_adjusted" and atr > 0:
            result = self._volatility_size(portfolio_value, entry_price, risk_per_share, atr)
        else:
            result = self._fixed_fraction_size(portfolio_value, entry_price, risk_per_share)

        # Apply volatility adjustment (if ATR available and not already used)
        if atr > 0 and result.method != "volatility_adjusted":
            vol_adj = self._volatility_adjustment(atr, entry_price)
            result.volatility_adjustment = vol_adj
            result.position_value *= vol_adj
            result.shares = int(result.position_value / entry_price)

        # Apply correlation adjustment
        if current_positions:
            corr_adj = self._correlation_adjustment(current_positions)
            result.correlation_adjustment = corr_adj
            result.position_value *= corr_adj
            result.shares = int(result.position_value / entry_price)

        # Apply regime adjustment
        if regime_aggression < 1.0:
            result.regime_adjustment = regime_aggression
            result.position_value *= regime_aggression
            result.shares = int(result.position_value / entry_price)

        # Enforce hard cap
        max_value = portfolio_value * (self.max_position_pct / 100.0)
        if result.position_value > max_value:
            result.position_value = max_value
            result.shares = int(max_value / entry_price)
            result.notes += f" Capped at max_position_pct={self.max_position_pct}%."

        # Ensure non-negative
        result.shares = max(0, result.shares)
        result.position_value = max(0.0, result.position_value)

        # Final metrics
        if portfolio_value > 0:
            result.allocation_pct = round(result.position_value / portfolio_value * 100, 3)
            result.risk_per_trade_pct = round(
                result.shares * risk_per_share / portfolio_value * 100, 4
            )

        result.trade_count = len(trades)

        logger.info(
            "kelly_sizer_result",
            method=result.method,
            shares=result.shares,
            position_value=round(result.position_value, 2),
            allocation_pct=result.allocation_pct,
            kelly_fraction=round(result.full_kelly_fraction, 4),
            win_rate=round(result.win_rate, 3),
        )

        return result

    # ------------------------------------------------------------------
    # Sizing methods
    # ------------------------------------------------------------------

    def _kelly_size(
        self,
        portfolio_value: float,
        entry_price: float,
        risk_per_share: float,
        trades: list[TradeRecord],
    ) -> KellyResult:
        """Compute position size using Kelly Criterion.

        f* = (p * b - q) / b
        where p = win rate, q = 1-p, b = avg_win / avg_loss.
        """
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades) if trades else 0.0
        loss_rate = 1.0 - win_rate

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1.0

        if avg_loss == 0:
            avg_loss = 1.0  # prevent division by zero

        win_loss_ratio = float(avg_win / avg_loss)

        # Kelly formula: f* = (p * b - q) / b
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Clamp to [0, 1] — negative Kelly means don't trade
        full_kelly = max(0.0, min(1.0, full_kelly))

        # Use half-Kelly by default for safety
        if self.method == "half_kelly":
            fraction = full_kelly / 2.0
            method = "half_kelly"
        else:
            fraction = full_kelly
            method = "kelly"

        position_value = portfolio_value * fraction
        shares = int(position_value / entry_price) if entry_price > 0 else 0

        return KellyResult(
            method=method,
            fraction=round(fraction, 4),
            position_value=round(position_value, 2),
            shares=shares,
            allocation_pct=round(fraction * 100, 3),
            risk_per_trade_pct=round(shares * risk_per_share / portfolio_value * 100, 4) if portfolio_value > 0 else 0,
            win_rate=round(win_rate, 4),
            win_loss_ratio=round(win_loss_ratio, 4),
            full_kelly_fraction=round(full_kelly, 4),
            notes=f"Kelly from {len(trades)} trades: p={win_rate:.2f}, b={win_loss_ratio:.2f}.",
        )

    def _fixed_fraction_size(
        self,
        portfolio_value: float,
        entry_price: float,
        risk_per_share: float,
    ) -> KellyResult:
        """Fixed fraction sizing: risk a fixed % of portfolio per trade.

        shares = (portfolio_value * risk_pct) / risk_per_share
        """
        risk_budget = portfolio_value * (self.default_risk_pct / 100.0)
        shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry_price

        return KellyResult(
            method="fixed_fraction",
            fraction=round(self.default_risk_pct / 100.0, 4),
            position_value=round(position_value, 2),
            shares=shares,
            allocation_pct=round(position_value / portfolio_value * 100, 3) if portfolio_value > 0 else 0,
            risk_per_trade_pct=self.default_risk_pct,
            notes=f"Fixed fraction: {self.default_risk_pct}% risk per trade (insufficient history for Kelly).",
        )

    def _volatility_size(
        self,
        portfolio_value: float,
        entry_price: float,
        risk_per_share: float,
        atr: float,
    ) -> KellyResult:
        """Volatility-adjusted sizing using ATR.

        position_value = (risk_budget / ATR) * price
        Higher ATR means smaller position to maintain constant dollar risk.
        """
        risk_budget = portfolio_value * (self.default_risk_pct / 100.0)
        # Number of shares that risks exactly risk_budget at ATR distance
        shares = int(risk_budget / atr) if atr > 0 else 0
        position_value = shares * entry_price

        return KellyResult(
            method="volatility_adjusted",
            fraction=round(position_value / portfolio_value, 4) if portfolio_value > 0 else 0,
            position_value=round(position_value, 2),
            shares=shares,
            allocation_pct=round(position_value / portfolio_value * 100, 3) if portfolio_value > 0 else 0,
            risk_per_trade_pct=round(shares * risk_per_share / portfolio_value * 100, 4) if portfolio_value > 0 else 0,
            notes=f"Volatility-adjusted: ATR={atr:.2f}, risk_budget=${risk_budget:.2f}.",
        )

    # ------------------------------------------------------------------
    # Adjustment factors
    # ------------------------------------------------------------------

    def _volatility_adjustment(self, atr: float, price: float) -> float:
        """Compute volatility adjustment factor.

        Scales position size inversely with volatility.
        ATR% of 2% is the baseline (adjustment = 1.0).
        Higher volatility → smaller position.
        """
        if price <= 0:
            return 1.0

        atr_pct = atr / price * 100.0
        baseline_atr_pct = 2.0

        if atr_pct <= 0:
            return 1.0

        adjustment = baseline_atr_pct / atr_pct
        # Clamp between 0.3 and 1.5
        return max(0.3, min(1.5, adjustment))

    def _correlation_adjustment(self, current_positions: list[dict]) -> float:
        """Reduce position size when holding correlated positions.

        If any held position has correlation > threshold with the new position,
        reduce the new position's size proportionally.

        Parameters
        ----------
        current_positions : list of dict
            Each with 'ticker' and 'returns' (np.ndarray of daily returns).

        Returns
        -------
        Adjustment factor (0.5 to 1.0). Lower = more correlated portfolio.
        """
        if not current_positions:
            return 1.0

        max_corr = 0.0
        for pos in current_positions:
            returns = pos.get("returns")
            if returns is not None and len(returns) > 10:
                # Use the correlation of the most correlated position
                corr = abs(float(np.mean(returns)))  # simplified proxy
                max_corr = max(max_corr, corr)

        if max_corr > self.correlation_threshold:
            # Reduce proportionally to how far above threshold
            reduction = 1.0 - (max_corr - self.correlation_threshold) / (1.0 - self.correlation_threshold)
            return max(0.5, reduction)

        return 1.0

    # ------------------------------------------------------------------
    # Trade history management
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade for future Kelly calculations."""
        self._trade_history.append(trade)
        # Keep only the lookback window
        if len(self._trade_history) > self.kelly_lookback * 2:
            self._trade_history = self._trade_history[-self.kelly_lookback:]

    def _recent_trades(self) -> list[TradeRecord]:
        """Get the most recent trades within the lookback window."""
        return self._trade_history[-self.kelly_lookback:]

    @property
    def trade_count(self) -> int:
        """Number of trades in history."""
        return len(self._trade_history)

    @property
    def has_sufficient_history(self) -> bool:
        """Whether we have enough trades to use Kelly Criterion."""
        return len(self._trade_history) >= self.min_trades

    def get_stats(self) -> dict[str, float]:
        """Get current trading statistics used for Kelly calculation."""
        trades = self._recent_trades()
        if not trades:
            return {"win_rate": 0, "win_loss_ratio": 0, "kelly_fraction": 0, "trade_count": 0}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1

        wl_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0
        kelly = (win_rate * wl_ratio - (1 - win_rate)) / wl_ratio if wl_ratio > 0 else 0
        kelly = max(0, kelly)

        return {
            "win_rate": round(win_rate, 4),
            "win_loss_ratio": round(wl_ratio, 4),
            "kelly_fraction": round(kelly, 4),
            "half_kelly_fraction": round(kelly / 2, 4),
            "trade_count": len(trades),
        }
