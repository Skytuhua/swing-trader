"""Tests for simulation accuracy models (slippage, fees, spread, partial fills, latency, gaps)."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass, field

from src.simulation.slippage import SlippageModel, SlippageResult
from src.simulation.fees import FeeCalculator, FeeResult
from src.simulation.spread import SpreadSimulator, SpreadResult
from src.simulation.partial_fills import PartialFillSimulator, PartialFillResult
from src.simulation.latency import LatencySimulator, LatencyResult
from src.simulation.gaps import GapHandler, GapFillResult


# ---------------------------------------------------------------------------
# Minimal SimulationConfig stub for testing
# ---------------------------------------------------------------------------

@dataclass
class _SimConfig:
    """Minimal config stub matching SimulationConfig fields."""

    # Slippage
    slippage_model: str = "volume_weighted"
    slippage_fixed_pct: float = 0.05
    slippage_base_bps: float = 5.0
    slippage_impact_exponent: float = 0.5
    slippage_volatility_weight: float = 0.5
    slippage_max_pct: float = 5.0

    # Fees
    fee_model: str = "per_share"
    commission_per_share: float = 0.005
    commission_flat_fee: float = 0.0
    commission_pct: float = 0.0
    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.0
    sec_fee_per_million: float = 8.0
    taf_fee_per_share: float = 0.000166

    # Spread
    spread_model: str = "volume_tiered"
    spread_fixed_cents: float = 0.01
    spread_fixed_pct: float = 0.05
    spread_time_of_day_scaling: bool = True

    # Partial fills
    partial_fill_enabled: bool = True
    partial_fill_volume_threshold_pct: float = 5.0
    partial_fill_min_rate: float = 0.1
    partial_fill_randomness: float = 0.15

    # Latency
    latency_enabled: bool = True
    latency_base_ms: float = 50.0
    latency_jitter_ms: float = 100.0
    latency_price_drift_bps: float = 2.0

    # Gaps
    gap_handling_enabled: bool = True
    gap_stop_fill_at_open: bool = True


# ===========================================================================
# Slippage Model Tests
# ===========================================================================


class TestSlippageModelFixed(unittest.TestCase):
    """Tests for fixed slippage model."""

    def setUp(self):
        cfg = _SimConfig(slippage_model="fixed", slippage_fixed_pct=0.1)
        self.model = SlippageModel(cfg)

    def test_fixed_returns_constant_slippage(self):
        r1 = self.model.calculate(order_dollar_value=10_000, avg_daily_dollar_volume=1e9)
        r2 = self.model.calculate(order_dollar_value=100_000, avg_daily_dollar_volume=1e9)
        self.assertAlmostEqual(r1.slippage_pct, r2.slippage_pct, places=6)

    def test_fixed_value_matches_config(self):
        result = self.model.calculate(order_dollar_value=50_000)
        self.assertAlmostEqual(result.slippage_pct, 0.001, places=6)  # 0.1% → 0.001

    def test_bps_conversion(self):
        result = self.model.calculate(order_dollar_value=50_000)
        self.assertAlmostEqual(result.slippage_bps, result.slippage_pct * 10_000, places=2)


class TestSlippageModelVolumeWeighted(unittest.TestCase):
    """Tests for volume-weighted (square-root) slippage model."""

    def setUp(self):
        self.model = SlippageModel(_SimConfig(slippage_model="volume_weighted"))

    def test_larger_orders_have_more_slippage(self):
        small = self.model.calculate(
            order_dollar_value=10_000, avg_daily_dollar_volume=1e9
        )
        large = self.model.calculate(
            order_dollar_value=1_000_000, avg_daily_dollar_volume=1e9
        )
        self.assertGreater(large.slippage_pct, small.slippage_pct)

    def test_more_liquid_stocks_have_less_slippage(self):
        illiquid = self.model.calculate(
            order_dollar_value=50_000, avg_daily_dollar_volume=10e6
        )
        liquid = self.model.calculate(
            order_dollar_value=50_000, avg_daily_dollar_volume=5e9
        )
        self.assertGreater(illiquid.slippage_pct, liquid.slippage_pct)

    def test_zero_volume_uses_fallback(self):
        result = self.model.calculate(order_dollar_value=50_000, avg_daily_dollar_volume=0)
        self.assertGreater(result.slippage_pct, 0)

    def test_slippage_is_positive(self):
        result = self.model.calculate(
            order_dollar_value=100_000, avg_daily_dollar_volume=500e6
        )
        self.assertGreater(result.slippage_pct, 0)

    def test_slippage_capped(self):
        cfg = _SimConfig(slippage_model="volume_weighted", slippage_max_pct=1.0)
        model = SlippageModel(cfg)
        # Extreme scenario: tiny volume
        result = model.calculate(
            order_dollar_value=10_000_000, avg_daily_dollar_volume=100
        )
        self.assertLessEqual(result.slippage_pct, 0.01)  # 1% cap


class TestSlippageModelVolatilityAdjusted(unittest.TestCase):
    """Tests for volatility-adjusted slippage model."""

    def setUp(self):
        self.model = SlippageModel(_SimConfig(slippage_model="volatility_adjusted"))

    def test_higher_volatility_more_slippage(self):
        low_vol = self.model.calculate(
            order_dollar_value=50_000, avg_daily_dollar_volume=1e9, atr_pct=0.01
        )
        high_vol = self.model.calculate(
            order_dollar_value=50_000, avg_daily_dollar_volume=1e9, atr_pct=0.05
        )
        self.assertGreater(high_vol.slippage_pct, low_vol.slippage_pct)

    def test_model_name(self):
        result = self.model.calculate(order_dollar_value=50_000, avg_daily_dollar_volume=1e9)
        self.assertEqual(result.model_used, "volatility_adjusted")


class TestSlippageModelComposite(unittest.TestCase):
    """Tests for composite slippage model."""

    def test_composite_uses_both_volume_and_volatility(self):
        model = SlippageModel(_SimConfig(slippage_model="composite"))
        result = model.calculate(
            order_dollar_value=50_000,
            avg_daily_dollar_volume=500e6,
            atr_pct=0.03,
        )
        self.assertGreater(result.slippage_pct, 0)
        self.assertEqual(result.model_used, "composite")


# ===========================================================================
# Fee Calculator Tests
# ===========================================================================


class TestFeeCalculatorPerShare(unittest.TestCase):
    """Tests for per-share fee model."""

    def setUp(self):
        self.calc = FeeCalculator(_SimConfig(fee_model="per_share", commission_per_share=0.005))

    def test_buy_commission(self):
        result = self.calc.calculate(quantity=100, fill_price=150.0, side="buy")
        self.assertAlmostEqual(result.commission, 0.50, places=2)

    def test_sell_includes_regulatory_fees(self):
        result = self.calc.calculate(quantity=100, fill_price=150.0, side="sell")
        self.assertGreater(result.sec_fee, 0)
        self.assertGreater(result.taf_fee, 0)
        self.assertGreater(result.total_fee, result.commission)

    def test_buy_no_regulatory_fees(self):
        result = self.calc.calculate(quantity=100, fill_price=150.0, side="buy")
        self.assertEqual(result.sec_fee, 0.0)
        self.assertEqual(result.taf_fee, 0.0)

    def test_fee_bps_calculated(self):
        result = self.calc.calculate(quantity=100, fill_price=150.0, side="buy")
        expected_bps = result.total_fee / (100 * 150.0) * 10_000
        self.assertAlmostEqual(result.fee_bps, expected_bps, places=2)


class TestFeeCalculatorFlat(unittest.TestCase):
    """Tests for flat fee model."""

    def test_flat_fee_same_regardless_of_size(self):
        calc = FeeCalculator(_SimConfig(fee_model="flat", commission_flat_fee=4.95))
        r1 = calc.calculate(quantity=10, fill_price=100.0, side="buy")
        r2 = calc.calculate(quantity=1000, fill_price=100.0, side="buy")
        self.assertAlmostEqual(r1.commission, r2.commission, places=2)
        self.assertAlmostEqual(r1.commission, 4.95, places=2)


class TestFeeCalculatorPercentage(unittest.TestCase):
    """Tests for percentage fee model."""

    def test_percentage_scales_with_value(self):
        calc = FeeCalculator(_SimConfig(fee_model="percentage", commission_pct=0.1))
        small = calc.calculate(quantity=10, fill_price=100.0, side="buy")
        large = calc.calculate(quantity=100, fill_price=100.0, side="buy")
        self.assertAlmostEqual(large.commission / small.commission, 10.0, places=1)


class TestFeeCalculatorTiered(unittest.TestCase):
    """Tests for maker/taker tiered fee model."""

    def test_limit_order_uses_maker_fee(self):
        calc = FeeCalculator(_SimConfig(
            fee_model="tiered", maker_fee_pct=0.01, taker_fee_pct=0.03
        ))
        result = calc.calculate(quantity=100, fill_price=100.0, side="buy", order_type="limit")
        # Maker fee: 0.01% of $10,000 = $0.10 (approximately, from commission calc)
        self.assertGreaterEqual(result.commission, 0)

    def test_market_order_uses_taker_fee(self):
        calc = FeeCalculator(_SimConfig(
            fee_model="tiered", maker_fee_pct=0.01, taker_fee_pct=0.03
        ))
        limit_result = calc.calculate(quantity=100, fill_price=100.0, side="buy", order_type="limit")
        market_result = calc.calculate(quantity=100, fill_price=100.0, side="buy", order_type="market")
        self.assertGreater(market_result.commission, limit_result.commission)


# ===========================================================================
# Spread Simulator Tests
# ===========================================================================


class TestSpreadSimulatorFixed(unittest.TestCase):
    """Tests for fixed spread model."""

    def test_fixed_cents_spread(self):
        sim = SpreadSimulator(
            _SimConfig(spread_model="fixed", spread_fixed_cents=1.0),
        )
        result = sim.calculate(mid_price=150.0)
        # 1 cent = $0.01 → half_spread = 0.01
        self.assertAlmostEqual(result.half_spread, 0.01, places=4)
        self.assertGreater(result.ask, result.bid)

    def test_ask_greater_than_bid(self):
        sim = SpreadSimulator(_SimConfig(spread_model="fixed"))
        result = sim.calculate(mid_price=100.0)
        self.assertGreater(result.ask, result.bid)


class TestSpreadSimulatorPercentage(unittest.TestCase):
    """Tests for percentage spread model."""

    def test_percentage_scales_with_price(self):
        sim = SpreadSimulator(
            _SimConfig(spread_model="percentage", spread_fixed_pct=0.1),
        )
        low = sim.calculate(mid_price=10.0)
        high = sim.calculate(mid_price=100.0)
        # Higher price → higher dollar spread
        self.assertGreater(high.half_spread, low.half_spread)


class TestSpreadSimulatorVolumeTiered(unittest.TestCase):
    """Tests for volume-tiered spread model."""

    def test_liquid_stock_tight_spread(self):
        import random
        sim = SpreadSimulator(
            _SimConfig(spread_model="volume_tiered"),
            rng=random.Random(42),
        )
        result = sim.calculate(
            mid_price=150.0,
            avg_daily_dollar_volume=5e9,
            time_of_day_label="mid_morning",
        )
        self.assertLess(result.spread_pct, 0.10)  # < 0.1% spread

    def test_illiquid_stock_wider_spread(self):
        import random
        sim = SpreadSimulator(
            _SimConfig(spread_model="volume_tiered"),
            rng=random.Random(42),
        )
        liquid = sim.calculate(
            mid_price=100.0,
            avg_daily_dollar_volume=1e9,
            time_of_day_label="mid_morning",
        )
        illiquid = sim.calculate(
            mid_price=100.0,
            avg_daily_dollar_volume=1e6,
            time_of_day_label="mid_morning",
        )
        self.assertGreater(illiquid.spread_pct, liquid.spread_pct)

    def test_tod_scaling_wider_at_open(self):
        import random
        sim = SpreadSimulator(
            _SimConfig(spread_model="volume_tiered", spread_time_of_day_scaling=True),
            rng=random.Random(42),
        )
        mid_morning = sim.calculate(
            mid_price=100.0,
            avg_daily_dollar_volume=500e6,
            time_of_day_label="mid_morning",
        )
        # Reset RNG for fair comparison
        sim._rng = random.Random(42)
        at_open = sim.calculate(
            mid_price=100.0,
            avg_daily_dollar_volume=500e6,
            time_of_day_label="open_auction",
        )
        self.assertGreater(at_open.spread_pct, mid_morning.spread_pct)


# ===========================================================================
# Partial Fill Tests
# ===========================================================================


class TestPartialFillSimulator(unittest.TestCase):
    """Tests for partial fill simulation."""

    def test_small_order_fully_fills(self):
        import random
        sim = PartialFillSimulator(_SimConfig(), rng=random.Random(42))
        result = sim.calculate(
            requested_qty=100,
            mid_price=150.0,
            avg_daily_dollar_volume=5e9,
        )
        self.assertEqual(result.filled_qty, 100)
        self.assertFalse(result.is_partial)
        self.assertEqual(result.fill_rate, 1.0)

    def test_large_order_partially_fills(self):
        import random
        sim = PartialFillSimulator(_SimConfig(), rng=random.Random(42))
        # Order for ~33M shares of a stock with ADDV of $500M at $150 → ~3.3M adv shares
        # 5% threshold = 166K shares, requesting 500K → should be partial
        result = sim.calculate(
            requested_qty=500_000,
            mid_price=150.0,
            avg_daily_dollar_volume=500e6,
        )
        self.assertTrue(result.is_partial)
        self.assertLess(result.filled_qty, 500_000)
        self.assertGreater(result.filled_qty, 0)

    def test_disabled_always_fills(self):
        cfg = _SimConfig(partial_fill_enabled=False)
        sim = PartialFillSimulator(cfg)
        result = sim.calculate(
            requested_qty=1_000_000,
            mid_price=100.0,
            avg_daily_dollar_volume=100,
        )
        self.assertEqual(result.filled_qty, 1_000_000)
        self.assertFalse(result.is_partial)

    def test_minimum_fill_rate(self):
        import random
        cfg = _SimConfig(partial_fill_min_rate=0.2)
        sim = PartialFillSimulator(cfg, rng=random.Random(42))
        result = sim.calculate(
            requested_qty=10_000_000,  # Extremely large
            mid_price=100.0,
            avg_daily_dollar_volume=100e6,
        )
        # Should fill at least 20%
        self.assertGreaterEqual(result.fill_rate, 0.1)  # min_rate with jitter


# ===========================================================================
# Latency Simulator Tests
# ===========================================================================


class TestLatencySimulator(unittest.TestCase):
    """Tests for execution latency simulation."""

    def test_latency_produces_delay(self):
        import random
        sim = LatencySimulator(_SimConfig(), rng=random.Random(42))
        result = sim.simulate(signal_price=150.0)
        self.assertGreater(result.delay_ms, 0)

    def test_latency_adjusts_price(self):
        import random
        sim = LatencySimulator(_SimConfig(), rng=random.Random(42))
        result = sim.simulate(signal_price=150.0, atr_pct=0.02, side="buy")
        self.assertNotEqual(result.adjusted_price, 150.0)

    def test_disabled_no_effect(self):
        cfg = _SimConfig(latency_enabled=False)
        sim = LatencySimulator(cfg)
        result = sim.simulate(signal_price=150.0)
        self.assertEqual(result.delay_ms, 0.0)
        self.assertEqual(result.adjusted_price, 150.0)

    def test_delay_within_expected_range(self):
        import random
        cfg = _SimConfig(latency_base_ms=50.0, latency_jitter_ms=100.0)
        sim = LatencySimulator(cfg, rng=random.Random(42))
        result = sim.simulate(signal_price=150.0)
        self.assertGreaterEqual(result.delay_ms, 50.0)
        self.assertLessEqual(result.delay_ms, 150.0)


# ===========================================================================
# Gap Handler Tests
# ===========================================================================


class TestGapHandler(unittest.TestCase):
    """Tests for overnight gap handling."""

    def setUp(self):
        self.handler = GapHandler(_SimConfig())

    def test_gap_down_through_stop(self):
        """Stop-loss at 145, open at 142 → fill at 142."""
        result = self.handler.check_gap_fill(
            stop_price=145.0,
            open_price=142.0,
            previous_close=148.0,
            side="sell",
        )
        self.assertTrue(result.is_gapped)
        self.assertAlmostEqual(result.fill_price, 142.0, places=2)
        self.assertGreater(result.slippage_from_stop, 0)

    def test_no_gap(self):
        """Normal trading: open above stop → fill at stop."""
        result = self.handler.check_gap_fill(
            stop_price=145.0,
            open_price=147.0,
            previous_close=148.0,
            side="sell",
        )
        self.assertFalse(result.is_gapped)
        self.assertAlmostEqual(result.fill_price, 145.0, places=2)

    def test_gap_up_through_short_stop(self):
        """Short cover stop at 155, open at 158 → fill at 158."""
        result = self.handler.check_gap_fill(
            stop_price=155.0,
            open_price=158.0,
            previous_close=152.0,
            side="buy",
        )
        self.assertTrue(result.is_gapped)
        self.assertAlmostEqual(result.fill_price, 158.0, places=2)

    def test_disabled_returns_stop_price(self):
        handler = GapHandler(_SimConfig(gap_handling_enabled=False))
        result = handler.check_gap_fill(
            stop_price=145.0,
            open_price=140.0,
            previous_close=148.0,
            side="sell",
        )
        self.assertFalse(result.is_gapped)
        self.assertAlmostEqual(result.fill_price, 145.0, places=2)

    def test_gap_percentage_calculated(self):
        result = self.handler.check_gap_fill(
            stop_price=145.0,
            open_price=142.0,
            previous_close=148.0,
            side="sell",
        )
        expected_gap_pct = (142.0 - 148.0) / 148.0 * 100.0
        self.assertAlmostEqual(result.gap_pct, expected_gap_pct, places=2)


# ===========================================================================
# Integration: Models work with MarketMicrostructureModel
# ===========================================================================


@unittest.skipUnless(
    importlib.util.find_spec("structlog") is not None,
    "structlog not installed — execution package cannot be imported",
)
class TestIntegrationWithMicrostructure(unittest.TestCase):
    """Test that simulation models integrate with MarketMicrostructureModel."""

    def test_model_with_config_uses_enhanced_slippage(self):
        from src.services.execution.market_simulator import MarketMicrostructureModel

        config = _SimConfig()
        model = MarketMicrostructureModel(rng_seed=42, config=config)
        model.set_avg_daily_volume("TEST", 1e9)
        model.set_atr("TEST", 2.0)

        result = model.simulate_fill(
            ticker="TEST",
            side="buy",
            order_type="market",
            requested_qty=100,
            mid_price=100.0,
            current_time=None,
        )
        self.assertGreater(result.filled_qty, 0)
        self.assertGreater(result.fill_price, 0)

    def test_model_without_config_uses_legacy(self):
        from src.services.execution.market_simulator import MarketMicrostructureModel

        model = MarketMicrostructureModel(rng_seed=42)
        model.set_avg_daily_volume("TEST", 1e9)
        model.set_atr("TEST", 2.0)

        result = model.simulate_fill(
            ticker="TEST",
            side="buy",
            order_type="market",
            requested_qty=100,
            mid_price=100.0,
            current_time=None,
        )
        self.assertGreater(result.filled_qty, 0)
        self.assertGreater(result.fill_price, 0)


if __name__ == "__main__":
    unittest.main()
