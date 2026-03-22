"""Tests for realistic market microstructure simulation."""

import unittest
from datetime import time

from src.services.execution.market_simulator import MarketMicrostructureModel


class TestSpreadSimulation(unittest.TestCase):
    """Test spread calculation based on liquidity tiers."""

    def setUp(self):
        self.model = MarketMicrostructureModel(rng_seed=42)

    def test_liquid_stock_has_tight_spread(self):
        """Stocks with >$500M avg daily dollar volume should have tight spread."""
        self.model.set_avg_daily_volume("AAPL", 1e9)
        result = self.model.compute_spread("AAPL", mid_price=150.0, current_time=time(10, 30))
        spread_pct = (result.ask - result.bid) / result.mid * 100
        self.assertLess(spread_pct, 0.10)

    def test_mid_cap_wider_spread(self):
        """Mid-cap stocks should have wider spread than liquid stocks."""
        self.model.set_avg_daily_volume("AAPL", 1e9)
        self.model.set_avg_daily_volume("MIDCAP", 100e6)
        liquid = self.model.compute_spread("AAPL", mid_price=150.0, current_time=time(10, 30))
        mid_cap = self.model.compute_spread("MIDCAP", mid_price=50.0, current_time=time(10, 30))
        liquid_pct = (liquid.ask - liquid.bid) / liquid.mid
        mid_pct = (mid_cap.ask - mid_cap.bid) / mid_cap.mid
        self.assertGreater(mid_pct, liquid_pct)

    def test_spread_is_positive(self):
        """Spread should always be positive (ask > bid)."""
        for ticker, addv in [("A", 1e6), ("B", 10e6), ("C", 100e6), ("D", 1e9)]:
            self.model.set_avg_daily_volume(ticker, addv)
            result = self.model.compute_spread(ticker, mid_price=100.0, current_time=time(10, 30))
            self.assertGreater(result.ask, result.bid)


class TestSimulateFill(unittest.TestCase):
    """Test the fill simulation."""

    def setUp(self):
        self.model = MarketMicrostructureModel(rng_seed=42)
        self.model.set_avg_daily_volume("AAPL", 5e9)
        self.model.set_atr("AAPL", 2.0)

    def test_market_buy_fills(self):
        """Market buy should fill."""
        result = self.model.simulate_fill(
            ticker="AAPL",
            side="buy",
            order_type="market",
            requested_qty=100,
            mid_price=150.0,
            limit_price=None,
            stop_price=None,
            current_time=time(10, 30),
        )
        self.assertGreater(result.filled_qty, 0)
        self.assertGreater(result.fill_price, 0)

    def test_large_order_more_impact(self):
        """Larger orders should have more market impact."""
        self.model.set_avg_daily_volume("SMOL", 50e6)
        self.model.set_atr("SMOL", 3.0)
        small_result = self.model.simulate_fill(
            ticker="SMOL", side="buy", order_type="market",
            requested_qty=100, mid_price=50.0,
            limit_price=None, stop_price=None, current_time=time(10, 30),
        )
        large_result = self.model.simulate_fill(
            ticker="SMOL", side="buy", order_type="market",
            requested_qty=50000, mid_price=50.0,
            limit_price=None, stop_price=None, current_time=time(10, 30),
        )
        self.assertGreater(small_result.filled_qty, 0)
        self.assertGreater(large_result.filled_qty, 0)
        # Large order should fill at same or worse price
        self.assertGreaterEqual(large_result.fill_price, small_result.fill_price)


class TestLimitFillProbability(unittest.TestCase):
    """Test limit order fill probability."""

    def setUp(self):
        pass

    def test_limit_at_ask_mostly_fills(self):
        """Limit at ask should have high probability of filling."""
        fills = 0
        trials = 50
        for i in range(trials):
            model = MarketMicrostructureModel(rng_seed=i)
            model.set_avg_daily_volume("AAPL", 5e9)
            model.set_atr("AAPL", 2.0)
            result = model.simulate_fill(
                ticker="AAPL", side="buy", order_type="limit",
                requested_qty=100, mid_price=150.0, limit_price=150.10,
                stop_price=None, current_time=time(10, 30),
            )
            if result.filled_qty > 0:
                fills += 1
        fill_rate = fills / trials
        self.assertGreater(fill_rate, 0.5)  # Should mostly fill


class TestExecutionTracker(unittest.TestCase):
    """Test execution quality tracking."""

    def test_record_and_retrieve_metrics(self):
        from datetime import datetime
        from src.services.execution.execution_tracker import ExecutionTracker

        tracker = ExecutionTracker()
        tracker.record_fill(
            ticker="AAPL",
            expected_price=150.00,
            actual_price=150.05,
            quantity=100,
            side="buy",
            order_type="market",
            timestamp=datetime.now(),
        )
        metrics = tracker.get_metrics()
        self.assertGreater(metrics.avg_slippage_bps, 0)

    def test_no_fills_returns_zero_metrics(self):
        from src.services.execution.execution_tracker import ExecutionTracker

        tracker = ExecutionTracker()
        metrics = tracker.get_metrics()
        self.assertEqual(metrics.avg_slippage_bps, 0.0)

    def test_multiple_fills_aggregate(self):
        from datetime import datetime
        from src.services.execution.execution_tracker import ExecutionTracker

        tracker = ExecutionTracker()
        for i in range(10):
            tracker.record_fill(
                ticker="AAPL", expected_price=150.0,
                actual_price=150.0 + (i * 0.01), quantity=100,
                side="buy", order_type="market", timestamp=datetime.now(),
            )
        metrics = tracker.get_metrics()
        self.assertEqual(metrics.sample_size, 10)
        self.assertGreater(metrics.total_slippage_usd, 0)


class TestPostTradeAnalyzer(unittest.TestCase):
    """Test post-trade analysis."""

    def test_analyze_winning_trade(self):
        from src.services.analysis.post_trade import PostTradeAnalyzer

        analyzer = PostTradeAnalyzer()
        trade = {
            "ticker": "AAPL",
            "entry_price": 150.0,
            "exit_price": 160.0,
            "quantity": 100,
            "entry_date": "2026-01-10",
            "exit_date": "2026-01-14",
            "stop_loss": 145.0,
            "take_profit_1": 160.0,
            "exit_reason": "take_profit_1",
            "high_prices": [151, 153, 157, 160, 161],
            "low_prices": [149, 150, 153, 156, 158],
            "confidence_score": 75,
            "technical_score": 80,
            "news_score": 65,
            "sentiment_score": 70,
        }
        report = analyzer.analyze_trade(trade)
        self.assertGreater(report.mfe_pct, 0)

    def test_analyze_losing_trade(self):
        from src.services.analysis.post_trade import PostTradeAnalyzer

        analyzer = PostTradeAnalyzer()
        trade = {
            "ticker": "TSLA",
            "entry_price": 200.0,
            "exit_price": 190.0,
            "quantity": 50,
            "entry_date": "2026-01-10",
            "exit_date": "2026-01-12",
            "stop_loss": 190.0,
            "take_profit_1": 220.0,
            "exit_reason": "stop_loss",
            "high_prices": [201, 198],
            "low_prices": [197, 189],
            "confidence_score": 60,
            "technical_score": 55,
            "news_score": 50,
            "sentiment_score": 45,
        }
        report = analyzer.analyze_trade(trade)
        self.assertLessEqual(report.pnl_pct, 0)
        self.assertLess(report.mae_pct, 0)


if __name__ == "__main__":
    unittest.main()
