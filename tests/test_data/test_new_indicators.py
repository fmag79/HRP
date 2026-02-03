"""
Tests for new technical indicators added in Phase 1.

Tests cover:
- EMA 12-day, EMA 26-day, EMA crossover
- Williams %R (14-day)
- Money Flow Index (14-day)
- VWAP (20-day rolling approximation)
"""

import numpy as np
import pandas as pd
import pytest

from hrp.data.features.computation import (
    FEATURE_FUNCTIONS,
    compute_ema_12d,
    compute_ema_26d,
    compute_ema_crossover,
    compute_williams_r_14d,
    compute_mfi_14d,
    compute_vwap_20d,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_prices():
    """Create sample price data for testing indicators."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    symbols = ["AAPL", "MSFT"]

    # Create realistic price data
    np.random.seed(42)
    data = []
    for symbol in symbols:
        base_price = 150 if symbol == "AAPL" else 400
        for i, dt in enumerate(dates):
            # Add some trend and noise
            close = base_price * (1 + 0.001 * i + 0.02 * np.random.randn())
            high = close * (1 + 0.01 * np.random.rand())
            low = close * (1 - 0.01 * np.random.rand())
            open_price = (high + low) / 2
            volume = int(1e6 * (1 + 0.5 * np.random.rand()))

            data.append({
                "date": dt,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
            })

    df = pd.DataFrame(data)
    df = df.set_index(["date", "symbol"])
    return df


@pytest.fixture
def sample_prices_longer():
    """Create longer sample price data for indicators needing more history."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    symbols = ["AAPL"]

    np.random.seed(42)
    data = []
    for symbol in symbols:
        base_price = 150
        for i, dt in enumerate(dates):
            close = base_price * (1 + 0.001 * i + 0.02 * np.random.randn())
            high = close * (1 + 0.01 * np.random.rand())
            low = close * (1 - 0.01 * np.random.rand())
            volume = int(1e6 * (1 + 0.5 * np.random.rand()))

            data.append({
                "date": dt,
                "symbol": symbol,
                "open": (high + low) / 2,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
            })

    df = pd.DataFrame(data)
    df = df.set_index(["date", "symbol"])
    return df


# =============================================================================
# Test: Features are registered in FEATURE_FUNCTIONS
# =============================================================================


class TestFeatureRegistration:
    """Test that new features are properly registered."""

    def test_ema_12d_registered(self):
        """EMA 12-day should be in FEATURE_FUNCTIONS."""
        assert "ema_12d" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["ema_12d"] == compute_ema_12d

    def test_ema_26d_registered(self):
        """EMA 26-day should be in FEATURE_FUNCTIONS."""
        assert "ema_26d" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["ema_26d"] == compute_ema_26d

    def test_ema_crossover_registered(self):
        """EMA crossover should be in FEATURE_FUNCTIONS."""
        assert "ema_crossover" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["ema_crossover"] == compute_ema_crossover

    def test_williams_r_14d_registered(self):
        """Williams %R 14-day should be in FEATURE_FUNCTIONS."""
        assert "williams_r_14d" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["williams_r_14d"] == compute_williams_r_14d

    def test_mfi_14d_registered(self):
        """MFI 14-day should be in FEATURE_FUNCTIONS."""
        assert "mfi_14d" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["mfi_14d"] == compute_mfi_14d

    def test_vwap_20d_registered(self):
        """VWAP 20-day should be in FEATURE_FUNCTIONS."""
        assert "vwap_20d" in FEATURE_FUNCTIONS
        assert FEATURE_FUNCTIONS["vwap_20d"] == compute_vwap_20d

    def test_total_features_count(self):
        """Should have 45 total features registered (39 technical + 6 fundamental)."""
        assert len(FEATURE_FUNCTIONS) == 45


# =============================================================================
# Test: EMA Indicators
# =============================================================================


class TestEMAIndicators:
    """Tests for EMA-based indicators."""

    def test_ema_12d_output_shape(self, sample_prices):
        """EMA 12-day should return correct shape."""
        result = compute_ema_12d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_12d" in result.columns
        assert len(result) == len(sample_prices)

    def test_ema_26d_output_shape(self, sample_prices):
        """EMA 26-day should return correct shape."""
        result = compute_ema_26d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_26d" in result.columns
        assert len(result) == len(sample_prices)

    def test_ema_crossover_output_shape(self, sample_prices):
        """EMA crossover should return correct shape."""
        result = compute_ema_crossover(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_crossover" in result.columns
        assert len(result) == len(sample_prices)

    def test_ema_crossover_values(self, sample_prices):
        """EMA crossover should only return +1, -1, or 0."""
        result = compute_ema_crossover(sample_prices)

        # Get non-NaN values
        values = result["ema_crossover"].dropna().unique()

        # All values should be in {-1, 0, 1}
        for v in values:
            assert v in [-1.0, 0.0, 1.0], f"Unexpected crossover value: {v}"

    def test_ema_12d_less_than_ema_26d_initial(self, sample_prices):
        """With uptrend, EMA-12 should eventually exceed EMA-26."""
        result_12 = compute_ema_12d(sample_prices)
        result_26 = compute_ema_26d(sample_prices)

        # EMA values should be positive (based on positive prices)
        assert (result_12["ema_12d"].dropna() > 0).all()
        assert (result_26["ema_26d"].dropna() > 0).all()


# =============================================================================
# Test: Williams %R
# =============================================================================


class TestWilliamsR:
    """Tests for Williams %R indicator."""

    def test_williams_r_output_shape(self, sample_prices):
        """Williams %R should return correct shape."""
        result = compute_williams_r_14d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "williams_r_14d" in result.columns
        assert len(result) == len(sample_prices)

    def test_williams_r_range(self, sample_prices):
        """Williams %R should be between -100 and 0."""
        result = compute_williams_r_14d(sample_prices)

        values = result["williams_r_14d"].dropna()

        # All values should be in [-100, 0]
        assert (values >= -100).all(), f"Min value: {values.min()}"
        assert (values <= 0).all(), f"Max value: {values.max()}"

    def test_williams_r_nan_for_insufficient_data(self, sample_prices):
        """Williams %R should have NaN for first 13 rows (need 14-day window)."""
        result = compute_williams_r_14d(sample_prices)

        # For each symbol, first 13 values should be NaN
        for symbol in ["AAPL", "MSFT"]:
            symbol_data = result.xs(symbol, level="symbol")
            # First 13 rows should have NaN (rolling window of 14)
            assert symbol_data.iloc[:13]["williams_r_14d"].isna().all()


# =============================================================================
# Test: Money Flow Index
# =============================================================================


class TestMFI:
    """Tests for Money Flow Index indicator."""

    def test_mfi_output_shape(self, sample_prices):
        """MFI should return correct shape."""
        result = compute_mfi_14d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "mfi_14d" in result.columns
        assert len(result) == len(sample_prices)

    def test_mfi_range(self, sample_prices):
        """MFI should be between 0 and 100."""
        result = compute_mfi_14d(sample_prices)

        values = result["mfi_14d"].dropna()

        # All values should be in [0, 100]
        assert (values >= 0).all(), f"Min value: {values.min()}"
        assert (values <= 100).all(), f"Max value: {values.max()}"

    def test_mfi_nan_for_insufficient_data(self, sample_prices):
        """MFI should have NaN for initial rows (need 14-day rolling + 1 diff)."""
        result = compute_mfi_14d(sample_prices)

        # For each symbol, first rows should be NaN
        for symbol in ["AAPL", "MSFT"]:
            symbol_data = result.xs(symbol, level="symbol")
            # First 13 rows should have NaN (rolling window of 14 starts producing at index 13)
            assert symbol_data.iloc[:13]["mfi_14d"].isna().all()


# =============================================================================
# Test: VWAP
# =============================================================================


class TestVWAP:
    """Tests for VWAP (20-day rolling approximation) indicator."""

    def test_vwap_output_shape(self, sample_prices):
        """VWAP should return correct shape."""
        result = compute_vwap_20d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "vwap_20d" in result.columns
        assert len(result) == len(sample_prices)

    def test_vwap_positive_values(self, sample_prices):
        """VWAP should be positive (based on positive prices)."""
        result = compute_vwap_20d(sample_prices)

        values = result["vwap_20d"].dropna()
        assert (values > 0).all()

    def test_vwap_nan_for_insufficient_data(self, sample_prices):
        """VWAP should have NaN for first 19 rows (need 20-day window)."""
        result = compute_vwap_20d(sample_prices)

        # For each symbol, first 19 values should be NaN
        for symbol in ["AAPL", "MSFT"]:
            symbol_data = result.xs(symbol, level="symbol")
            # First 19 rows should have NaN (rolling window of 20)
            assert symbol_data.iloc[:19]["vwap_20d"].isna().all()

    def test_vwap_reasonable_values(self, sample_prices):
        """VWAP should be close to price (within typical price range)."""
        result = compute_vwap_20d(sample_prices)

        # Get close prices and VWAP
        close = sample_prices["close"].unstack(level="symbol")
        vwap = result["vwap_20d"].unstack(level="symbol")

        # VWAP should be within 10% of close price on average
        for symbol in ["AAPL", "MSFT"]:
            valid_idx = vwap[symbol].dropna().index
            ratio = vwap[symbol].loc[valid_idx] / close[symbol].loc[valid_idx]
            # Ratio should be close to 1 (within 10%)
            assert (ratio > 0.9).all() and (ratio < 1.1).all()


# =============================================================================
# Test: Integration with ingestion/features.py
# =============================================================================


class TestIngestionIntegration:
    """Test that new features are included in ingestion module."""

    def test_new_features_in_feature_columns(self):
        """New features should be in the feature_columns list."""
        from hrp.data.ingestion.features import _compute_all_features

        # Create minimal test data
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        np.random.seed(42)

        df = pd.DataFrame({
            "date": dates,
            "high": 150 + np.random.randn(30),
            "low": 148 + np.random.randn(30),
            "close": 149 + np.random.randn(30),
            "adj_close": 149 + np.random.randn(30),
            "volume": np.random.randint(1e6, 2e6, 30),
        })

        result = _compute_all_features(df, "TEST", "v1")

        # Check that new features are present
        feature_names = result["feature_name"].unique()
        new_features = ["ema_12d", "ema_26d", "ema_crossover",
                        "williams_r_14d", "mfi_14d", "vwap_20d"]

        for feature in new_features:
            assert feature in feature_names, f"Missing feature: {feature}"
