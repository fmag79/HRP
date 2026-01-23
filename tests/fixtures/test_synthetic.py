"""Tests for synthetic data generators."""

from datetime import date

import pandas as pd
import pytest

from tests.fixtures.synthetic import (
    generate_prices,
    generate_features,
    generate_corporate_actions,
    generate_universe,
)


class TestGeneratePrices:
    """Tests for price data generator."""

    def test_generates_correct_columns(self):
        """Generated prices have all required columns."""
        df = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        expected = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        assert list(df.columns) == expected

    def test_deterministic_with_seed(self):
        """Same seed produces same data."""
        df1 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        df2 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different data."""
        df1 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=42)
        df2 = generate_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10), seed=99)
        assert not df1["close"].equals(df2["close"])

    def test_high_low_relationship(self):
        """High is always >= low."""
        df = generate_prices(["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 12, 31))
        assert (df["high"] >= df["low"]).all()

    def test_multiple_symbols(self):
        """Generates data for multiple symbols."""
        df = generate_prices(["AAPL", "MSFT", "GOOGL"], date(2023, 1, 1), date(2023, 1, 10))
        assert set(df["symbol"].unique()) == {"AAPL", "MSFT", "GOOGL"}


class TestGenerateFeatures:
    """Tests for feature data generator."""

    def test_generates_correct_columns(self):
        """Generated features have all required columns."""
        df = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5))
        expected = ["symbol", "date", "feature_name", "value", "version"]
        assert list(df.columns) == expected

    def test_deterministic_with_seed(self):
        """Same seed produces same data."""
        df1 = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5), seed=42)
        df2 = generate_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5), seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_multiple_features(self):
        """Generates all requested features."""
        features = ["momentum_20d", "volatility_60d", "rsi_14d"]
        df = generate_features(["AAPL"], features, date(2023, 1, 5))
        assert set(df["feature_name"].unique()) == set(features)


class TestGenerateCorporateActions:
    """Tests for corporate actions generator."""

    def test_generates_correct_columns(self):
        """Generated actions have all required columns."""
        df = generate_corporate_actions(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))
        if not df.empty:
            expected = ["symbol", "date", "action_type", "factor", "source"]
            assert list(df.columns) == expected

    def test_split_factors_valid(self):
        """Split factors are valid ratios."""
        df = generate_corporate_actions(
            ["AAPL", "MSFT", "GOOGL"],
            date(2020, 1, 1),
            date(2023, 12, 31),
            action_types=["split"],
            seed=42,
        )
        if not df.empty:
            assert (df["factor"] > 0).all()


class TestGenerateUniverse:
    """Tests for universe generator."""

    def test_generates_correct_columns(self):
        """Generated universe has all required columns."""
        df = generate_universe(["AAPL"], [date(2023, 1, 1)])
        expected = ["symbol", "date", "in_universe", "sector", "market_cap"]
        assert list(df.columns) == expected

    def test_all_symbols_included(self):
        """All symbols appear in universe."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        df = generate_universe(symbols, [date(2023, 1, 1)])
        assert set(df["symbol"].unique()) == set(symbols)
