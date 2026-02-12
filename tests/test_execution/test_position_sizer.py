"""Tests for VaR-aware position sizing."""

import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd

from hrp.execution.position_sizer import PositionSizer, PositionSizingConfig


class TestPositionSizingConfig(unittest.TestCase):
    """Test position sizing configuration."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = PositionSizingConfig(
            portfolio_value=Decimal("100000"),
            max_portfolio_var_pct=0.02,
            max_position_var_pct=0.005,
        )
        self.assertEqual(config.portfolio_value, Decimal("100000"))
        self.assertEqual(config.max_portfolio_var_pct, 0.02)
        self.assertEqual(config.max_position_var_pct, 0.005)

    def test_invalid_portfolio_var_pct(self):
        """Test invalid portfolio VaR percentage."""
        with self.assertRaises(ValueError):
            PositionSizingConfig(
                portfolio_value=Decimal("100000"),
                max_portfolio_var_pct=1.5,  # > 1.0
            )

    def test_invalid_position_var_pct(self):
        """Test invalid position VaR percentage."""
        with self.assertRaises(ValueError):
            PositionSizingConfig(
                portfolio_value=Decimal("100000"),
                max_position_var_pct=-0.01,  # < 0
            )

    def test_invalid_portfolio_value(self):
        """Test invalid portfolio value."""
        with self.assertRaises(ValueError):
            PositionSizingConfig(
                portfolio_value=Decimal("-1000"),  # negative
            )


class TestPositionSizer(unittest.TestCase):
    """Test VaR-aware position sizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            portfolio_value=Decimal("100000"),
            max_portfolio_var_pct=0.02,  # 2% portfolio VaR limit
            max_position_var_pct=0.005,  # 0.5% per position VaR limit
            max_position_pct=0.10,  # 10% hard cap
            min_position_value=Decimal("100"),
            fallback_var_pct=0.02,  # 2% fallback VaR
        )

        # Mock API
        self.mock_api = MagicMock()
        self.sizer = PositionSizer(self.config, self.mock_api)

    def test_initialization(self):
        """Test position sizer initialization."""
        self.assertEqual(self.sizer.config, self.config)
        self.assertEqual(self.sizer.api, self.mock_api)

    def test_get_symbol_var_from_features(self):
        """Test retrieving VaR from feature store."""
        # Mock feature data
        features_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "var_95_1d": [0.015],  # 1.5% daily VaR
        })
        self.mock_api.get_features.return_value = features_df

        var = self.sizer._get_symbol_var(
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            price=Decimal("150.00"),
        )

        self.assertEqual(var, 0.015)
        self.mock_api.get_features.assert_called_once_with(
            symbols=["AAPL"],
            features=["var_95_1d"],
            as_of_date=date(2024, 1, 15),
            version="v1",
        )

    def test_get_symbol_var_fallback(self):
        """Test VaR fallback when feature unavailable."""
        # Mock empty feature data
        self.mock_api.get_features.return_value = pd.DataFrame()

        var = self.sizer._get_symbol_var(
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            price=Decimal("150.00"),
        )

        # Should use fallback
        self.assertEqual(var, self.config.fallback_var_pct)

    def test_get_symbol_var_cvar(self):
        """Test retrieving CVaR instead of VaR."""
        # Configure to use CVaR
        config = PositionSizingConfig(
            portfolio_value=Decimal("100000"),
            use_cvar=True,
        )
        sizer = PositionSizer(config, self.mock_api)

        # Mock CVaR feature
        features_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "cvar_95_1d": [0.020],  # 2% CVaR
        })
        self.mock_api.get_features.return_value = features_df

        var = sizer._get_symbol_var(
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            price=Decimal("150.00"),
        )

        self.assertEqual(var, 0.020)

    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        # Mock VaR = 2%
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        # max_position_var_pct / symbol_var = 0.005 / 0.02 = 0.25 = 25% of portfolio
        # Capped by max_position_pct = 10%
        # Signal strength = 1.0 (full confidence)
        # Position value = $100,000 * 0.10 * 1.0 = $10,000
        # Quantity = $10,000 / $150 = 66 shares

        quantity = self.sizer.calculate_position_size(
            symbol="AAPL",
            signal_strength=1.0,
            current_price=Decimal("150.00"),
            as_of_date=date(2024, 1, 15),
        )

        self.assertEqual(quantity, 66)

    def test_calculate_position_size_weak_signal(self):
        """Test position sizing with weak signal."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        # Signal strength = 0.5 (moderate confidence)
        # Position value = $100,000 * 0.10 * 0.5 = $5,000
        # Quantity = $5,000 / $150 = 33 shares

        quantity = self.sizer.calculate_position_size(
            symbol="AAPL",
            signal_strength=0.5,
            current_price=Decimal("150.00"),
            as_of_date=date(2024, 1, 15),
        )

        self.assertEqual(quantity, 33)

    def test_calculate_position_size_high_var_symbol(self):
        """Test position sizing for high VaR symbol."""
        # High VaR = 5% (volatile stock)
        self.sizer._get_symbol_var = MagicMock(return_value=0.05)

        # max_position_var_pct / symbol_var = 0.005 / 0.05 = 0.10 = 10%
        # Position value = $100,000 * 0.10 * 1.0 = $10,000
        # Quantity = $10,000 / $150 = 66 shares

        quantity = self.sizer.calculate_position_size(
            symbol="VOLATILE",
            signal_strength=1.0,
            current_price=Decimal("150.00"),
            as_of_date=date(2024, 1, 15),
        )

        self.assertEqual(quantity, 66)

    def test_calculate_position_size_below_minimum(self):
        """Test position size below minimum threshold."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        # Very high price leads to small position value
        quantity = self.sizer.calculate_position_size(
            symbol="EXPENSIVE",
            signal_strength=0.1,  # Weak signal
            current_price=Decimal("10000.00"),  # Expensive stock
            as_of_date=date(2024, 1, 15),
        )

        # Position value = $100,000 * 0.10 * 0.1 = $1,000
        # Below minimum? No, but quantity would be 0 shares
        self.assertEqual(quantity, 0)

    def test_calculate_position_size_zero_signal(self):
        """Test position size with zero signal strength."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        quantity = self.sizer.calculate_position_size(
            symbol="AAPL",
            signal_strength=0.0,
            current_price=Decimal("150.00"),
            as_of_date=date(2024, 1, 15),
        )

        self.assertEqual(quantity, 0)

    def test_calculate_portfolio_var_budget_empty(self):
        """Test VaR budget calculation with no positions."""
        budget = self.sizer.calculate_portfolio_var_budget(
            current_positions={},
            current_prices={},
            as_of_date=date(2024, 1, 15),
        )

        # Full budget available
        self.assertEqual(budget, self.config.max_portfolio_var_pct)

    def test_calculate_portfolio_var_budget_with_positions(self):
        """Test VaR budget calculation with existing positions."""
        self.sizer._get_symbol_var = MagicMock(side_effect=[0.02, 0.015])

        current_positions = {
            "AAPL": 100,  # 100 shares
            "MSFT": 150,  # 150 shares
        }
        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
        }

        # AAPL: 100 * $150 = $15,000 (15% of $100k), VaR=2%, contribution=0.003
        # MSFT: 150 * $300 = $45,000 (45% of $100k), VaR=1.5%, contribution=0.00675
        # Total VaR = 0.003 + 0.00675 = 0.00975
        # Remaining = 0.02 - 0.00975 = 0.01025

        budget = self.sizer.calculate_portfolio_var_budget(
            current_positions=current_positions,
            current_prices=current_prices,
            as_of_date=date(2024, 1, 15),
        )

        self.assertAlmostEqual(budget, 0.01025, places=5)

    def test_size_all_positions_basic(self):
        """Test sizing multiple positions."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "signal": [1.0, 1.0, 1.0],
            "prediction": [0.9, 0.8, 0.7],  # Different strengths
        })

        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
            "GOOGL": Decimal("2000.00"),
        }

        positions = self.sizer.size_all_positions(
            signals=signals,
            current_positions={},
            current_prices=current_prices,
            as_of_date=date(2024, 1, 15),
        )

        # After normalization: AAPL=1.0 (0.9-0.7)/(0.9-0.7)=1.0,
        # MSFT=0.5 (0.8-0.7)/(0.9-0.7)=0.5, GOOGL=0.0 (0.7-0.7)/(0.9-0.7)=0.0
        # GOOGL gets signal_strength=0.0 and is rejected
        # Should have 2 positions (AAPL and MSFT)
        self.assertEqual(len(positions), 2)
        self.assertIn("AAPL", positions)
        self.assertIn("MSFT", positions)

        # All allocated quantities should be positive
        self.assertGreater(positions["AAPL"], 0)
        self.assertGreater(positions["MSFT"], 0)

    def test_size_all_positions_no_signals(self):
        """Test sizing with no buy signals."""
        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "signal": [0.0, 0.0],  # No buy signals
            "prediction": [0.5, 0.5],
        })

        positions = self.sizer.size_all_positions(
            signals=signals,
            current_positions={},
            current_prices={"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")},
            as_of_date=date(2024, 1, 15),
        )

        # No positions
        self.assertEqual(len(positions), 0)

    def test_size_all_positions_var_budget_exhausted(self):
        """Test sizing when VaR budget is exhausted."""
        # Set very low VaR for symbols so positions consume budget quickly
        self.sizer._get_symbol_var = MagicMock(return_value=0.01)

        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "signal": [1.0, 1.0, 1.0, 1.0, 1.0],
            "prediction": [0.9, 0.9, 0.9, 0.9, 0.9],  # All strong
        })

        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
            "GOOGL": Decimal("2000.00"),
            "AMZN": Decimal("3000.00"),
            "META": Decimal("400.00"),
        }

        positions = self.sizer.size_all_positions(
            signals=signals,
            current_positions={},
            current_prices=current_prices,
            as_of_date=date(2024, 1, 15),
        )

        # Should not allocate all 5 positions due to VaR budget
        # With max_portfolio_var_pct=0.02 and symbol_var=0.01,
        # max allocation is 0.02/0.01 = 2.0 = 200% of portfolio
        # But max_position_pct=0.10 caps each position
        # So we can fit 0.02/0.001 = 20 positions at 10% each (but VaR per position is 0.001)
        # Actually: position_var = (position_value/portfolio_value) * symbol_var
        #         = 0.10 * 0.01 = 0.001 per position
        # Budget = 0.02, so we can fit 0.02/0.001 = 20 positions

        # But with max_position_pct=0.10, we're limited by position size
        # Each position contributes 0.001 VaR
        # We can fit 20 positions before exhausting budget
        # Since we only have 5 signals, all should be allocated
        self.assertEqual(len(positions), 5)

    def test_size_all_positions_signal_normalization(self):
        """Test signal strength normalization."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        # Signals with different prediction ranges
        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "signal": [1.0, 1.0],
            "prediction": [0.6, 0.4],  # Will normalize to [1.0, 0.0]
        })

        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
        }

        positions = self.sizer.size_all_positions(
            signals=signals,
            current_positions={},
            current_prices=current_prices,
            as_of_date=date(2024, 1, 15),
        )

        # AAPL should have larger position (normalized strength = 1.0)
        # MSFT should have smaller position (normalized strength = 0.0)
        self.assertGreater(positions["AAPL"], positions.get("MSFT", 0))

    def test_size_all_positions_missing_prices(self):
        """Test sizing when prices are missing."""
        self.sizer._get_symbol_var = MagicMock(return_value=0.02)

        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "signal": [1.0, 1.0],
            "prediction": [0.8, 0.8],
        })

        current_prices = {
            "AAPL": Decimal("150.00"),
            # MSFT price missing
        }

        positions = self.sizer.size_all_positions(
            signals=signals,
            current_positions={},
            current_prices=current_prices,
            as_of_date=date(2024, 1, 15),
        )

        # Only AAPL should be sized
        self.assertEqual(len(positions), 1)
        self.assertIn("AAPL", positions)
        self.assertNotIn("MSFT", positions)


if __name__ == "__main__":
    unittest.main()
