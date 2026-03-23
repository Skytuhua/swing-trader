"""Fee and commission models for paper trading simulation.

Supports multiple fee structures:
- **per_share**: Fixed dollar amount per share (e.g. $0.005/share).
- **flat**: Fixed dollar fee per order (e.g. $4.95/trade).
- **percentage**: Percentage of trade value (e.g. 0.1%).
- **tiered**: Maker/taker distinction for limit vs market orders.

Also models regulatory fees (SEC fee on sells, TAF fee).

Usage::

    from src.simulation.fees import FeeCalculator
    from src.core.config import SimulationConfig

    calc = FeeCalculator(SimulationConfig())
    result = calc.calculate(
        quantity=100,
        fill_price=150.0,
        side="buy",
        order_type="market",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import SimulationConfig


@dataclass(frozen=True)
class FeeResult:
    """Breakdown of fees for a single fill."""

    commission: float       # Broker commission
    exchange_fee: float     # Maker/taker exchange fee
    sec_fee: float          # SEC fee (sells only)
    taf_fee: float          # TAF fee (sells only)
    total_fee: float        # Sum of all fees
    fee_per_share: float    # Total fee / quantity
    fee_bps: float          # Total fee as bps of trade value


class FeeCalculator:
    """Configurable fee calculator for paper trading.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with fee parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._model = config.fee_model
        self._per_share = config.commission_per_share
        self._flat_fee = config.commission_flat_fee
        self._comm_pct = config.commission_pct / 100.0  # Convert % to fraction
        self._maker_fee = config.maker_fee_pct / 100.0
        self._taker_fee = config.taker_fee_pct / 100.0
        self._sec_per_million = config.sec_fee_per_million
        self._taf_per_share = config.taf_fee_per_share

    def calculate(
        self,
        quantity: int,
        fill_price: float,
        side: str = "buy",
        order_type: str = "market",
    ) -> FeeResult:
        """Calculate all fees for a fill.

        Parameters
        ----------
        quantity : int
            Number of shares filled.
        fill_price : float
            Average fill price.
        side : str
            "buy" or "sell".
        order_type : str
            "market", "limit", etc.

        Returns
        -------
        FeeResult
            Complete fee breakdown.
        """
        trade_value = quantity * fill_price

        # Commission
        commission = self._calculate_commission(quantity, trade_value, order_type)

        # Exchange fee (maker/taker)
        exchange_fee = self._calculate_exchange_fee(trade_value, order_type)

        # Regulatory fees (sells only)
        sec_fee = 0.0
        taf_fee = 0.0
        if side == "sell":
            sec_fee = trade_value * self._sec_per_million / 1_000_000.0
            taf_fee = quantity * self._taf_per_share

        total = commission + exchange_fee + sec_fee + taf_fee
        fee_per_share = total / quantity if quantity > 0 else 0.0
        fee_bps = (total / trade_value * 10_000) if trade_value > 0 else 0.0

        return FeeResult(
            commission=round(commission, 4),
            exchange_fee=round(exchange_fee, 4),
            sec_fee=round(sec_fee, 4),
            taf_fee=round(taf_fee, 6),
            total_fee=round(total, 4),
            fee_per_share=round(fee_per_share, 6),
            fee_bps=round(fee_bps, 3),
        )

    def _calculate_commission(
        self, quantity: int, trade_value: float, order_type: str
    ) -> float:
        """Calculate broker commission based on fee model."""
        if self._model == "per_share":
            return self._per_share * quantity
        elif self._model == "flat":
            return self._flat_fee
        elif self._model == "percentage":
            return self._comm_pct * trade_value
        elif self._model == "tiered":
            # Use maker/taker rates
            if order_type in ("limit", "stop_limit"):
                return self._maker_fee * trade_value
            else:
                return self._taker_fee * trade_value
        return 0.0

    def _calculate_exchange_fee(
        self, trade_value: float, order_type: str
    ) -> float:
        """Calculate exchange maker/taker fee."""
        if self._model == "tiered":
            return 0.0  # Already accounted for in commission for tiered

        if order_type in ("limit", "stop_limit"):
            return self._maker_fee * trade_value
        return self._taker_fee * trade_value
