"""
Tests for risk feature computation functions.

Tests the integration of VaR/CVaR calculator with the feature system.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from hrp.data.features.risk_features import (
    compute_var_95_1d,
    compute_cvar_95_1d,
    compute_var_99_1d,
    compute_mc_var_95_1d,
    compute_var_95_10d,
)


@pytest.fixture
def sample_prices():
    """
    Create sample price data for testing.

    Returns DataFrame with MultiIndex (date, symbol) and OHLCV columns.
    """
    symbols = ["AAPL", "GOOGL"]
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    # Create realistic price data
    np.random.seed(42)
    data = []

    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 150.0
        prices = []

        for i, date in enumerate(dates):
            # Random walk with drift
            drift = 0.0001  # Slight upward drift
            volatility = 0.02  # 2% daily volatility
            return_val = np.random.normal(drift, volatility)

            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + return_val)

            prices.append(price)

            data.append({
                "date": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "adj_close": price,
                "volume": int(np.random.uniform(1e6, 1e7))
            })

    df = pd.DataFrame(data)
    df = df.set_index(["date", "symbol"])
    return df


class TestVaRFeatureComputation:
    """Tests for VaR feature computation functions."""

    def test_compute_var_95_1d(self, sample_prices):
        """Test 1-day VaR at 95% confidence (parametric method)."""
        result = compute_var_95_1d(sample_prices)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert "var_95_1d" in result.columns
        assert result.index.names == ["date", "symbol"]

        # Check values are reasonable
        var_values = result["var_95_1d"].dropna()
        assert len(var_values) > 0
        assert (var_values > 0).all()  # VaR should be positive (loss magnitude)
        assert (var_values < 0.10).all()  # Shouldn't exceed 10% for 1-day

    def test_compute_cvar_95_1d(self, sample_prices):
        """Test 1-day CVaR at 95% confidence."""
        result = compute_cvar_95_1d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "cvar_95_1d" in result.columns

        cvar_values = result["cvar_95_1d"].dropna()
        assert len(cvar_values) > 0
        assert (cvar_values > 0).all()

        # CVaR should be greater than VaR
        var_result = compute_var_95_1d(sample_prices)
        var_values = var_result["var_95_1d"]
        cvar_values_aligned = result["cvar_95_1d"]

        # At least 80% of CVaR values should be >= corresponding VaR
        comparison = cvar_values_aligned >= var_values
        assert comparison.dropna().mean() >= 0.8

    def test_compute_var_99_1d(self, sample_prices):
        """Test 1-day VaR at 99% confidence."""
        result = compute_var_99_1d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "var_99_1d" in result.columns

        var_99_values = result["var_99_1d"].dropna()
        var_95_values = compute_var_95_1d(sample_prices)["var_95_1d"].dropna()

        # 99% VaR should be higher than 95% VaR
        assert var_99_values.mean() > var_95_values.mean()

    def test_compute_mc_var_95_1d(self, sample_prices):
        """Test Monte Carlo VaR at 95% confidence."""
        result = compute_mc_var_95_1d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "mc_var_95_1d" in result.columns

        mc_var_values = result["mc_var_95_1d"].dropna()
        assert len(mc_var_values) > 0
        assert (mc_var_values > 0).all()

        # MC VaR should be in similar range to parametric VaR
        parametric_var = compute_var_95_1d(sample_prices)["var_95_1d"]

        # Mean MC VaR should be within 50% of parametric VaR mean
        mc_mean = mc_var_values.mean()
        param_mean = parametric_var.dropna().mean()
        assert abs(mc_mean - param_mean) / param_mean < 0.5

    def test_compute_var_95_10d(self, sample_prices):
        """Test 10-day VaR at 95% confidence."""
        result = compute_var_95_10d(sample_prices)

        assert isinstance(result, pd.DataFrame)
        assert "var_95_10d" in result.columns

        var_10d_values = result["var_95_10d"].dropna()
        var_1d_values = compute_var_95_1d(sample_prices)["var_95_1d"].dropna()

        # 10-day VaR should be larger than 1-day VaR
        # (approximately sqrt(10) times larger under normal assumptions)
        assert var_10d_values.mean() > var_1d_values.mean()

        # But not more than 5x larger (sqrt(10) ~= 3.16)
        assert var_10d_values.mean() < var_1d_values.mean() * 5

    def test_insufficient_data(self):
        """Test that features return NaN when insufficient data."""
        # Create tiny dataset with < 30 observations
        symbols = ["AAPL"]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        data = []
        for date in dates:
            data.append({
                "date": date,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adj_close": 100.0,
                "volume": 1000000
            })

        df = pd.DataFrame(data).set_index(["date", "symbol"])

        # Should return all NaN values (calculator requires 30+ observations)
        result = compute_var_95_1d(df)
        assert result["var_95_1d"].isna().all()

    def test_rolling_window_computation(self, sample_prices):
        """Test that VaR is computed as a rolling window, not all at once."""
        result = compute_var_95_1d(sample_prices)

        # Extract AAPL values
        aapl_var = result.xs("AAPL", level="symbol")["var_95_1d"]

        # VaR should vary over time (not constant)
        assert aapl_var.std() > 0

        # First 60 values should be NaN (need 60 days for rolling calculation)
        assert aapl_var.iloc[:60].isna().all()

        # Later values should be present
        assert aapl_var.iloc[60:].notna().sum() > 0
