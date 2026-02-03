"""
Comprehensive tests for the feature computation engine.

Tests cover:
- Individual feature computation functions (momentum, volatility, RSI, etc.)
- FEATURE_FUNCTIONS registry mapping
- FeatureComputer class methods
- register_default_features function
"""

import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.data.features.computation import (
    # Feature computation functions
    compute_momentum_20d,
    compute_momentum_60d,
    compute_momentum_252d,
    compute_volatility_20d,
    compute_volatility_60d,
    compute_returns_1d,
    compute_returns_5d,
    compute_returns_20d,
    compute_returns_60d,
    compute_returns_252d,
    compute_volume_20d,
    compute_volume_ratio,
    compute_obv,
    compute_rsi_14d,
    compute_atr_14d,
    compute_adx_14d,
    compute_macd_line,
    compute_macd_signal,
    compute_macd_histogram,
    compute_cci_20d,
    compute_roc_10d,
    compute_trend,
    compute_bb_upper_20d,
    compute_bb_lower_20d,
    compute_bb_width_20d,
    compute_stoch_k_14d,
    compute_stoch_d_14d,
    compute_sma_20d,
    compute_sma_50d,
    compute_sma_200d,
    compute_price_to_sma_20d,
    compute_price_to_sma_50d,
    compute_price_to_sma_200d,
    compute_ema_12d,
    compute_ema_26d,
    compute_ema_crossover,
    compute_williams_r_14d,
    compute_mfi_14d,
    compute_vwap_20d,
    compute_market_cap,
    compute_pe_ratio,
    compute_pb_ratio,
    compute_dividend_yield,
    compute_ev_ebitda,
    # Registry and classes
    FEATURE_FUNCTIONS,
    FeatureComputer,
    register_default_features,
)
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_prices():
    """
    Create synthetic price data for testing feature computations.

    Returns DataFrame with MultiIndex (date, symbol) and OHLCV columns.
    Covers 300 trading days for two symbols.
    """
    np.random.seed(42)

    symbols = ["AAPL", "MSFT"]
    n_days = 300

    # Generate dates (trading days only, roughly)
    start_date = date(2023, 1, 3)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    rows = []
    for symbol in symbols:
        # Starting price
        base_price = 150.0 if symbol == "AAPL" else 250.0

        # Generate random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.cumprod(1 + returns)

        for i, dt in enumerate(dates):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.uniform(1e6, 5e6))

            rows.append({
                "date": dt,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "symbol"])
    return df


@pytest.fixture
def short_prices():
    """
    Create minimal price data (25 days) for basic computation tests.
    """
    np.random.seed(123)

    symbols = ["TEST"]
    n_days = 25

    dates = pd.bdate_range(start=date(2023, 6, 1), periods=n_days)

    rows = []
    base_price = 100.0
    prices = base_price * np.cumprod(1 + np.random.normal(0.001, 0.015, n_days))

    for i, dt in enumerate(dates):
        close = prices[i]
        rows.append({
            "date": dt,
            "symbol": "TEST",
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": 1000000,
        })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "symbol"])
    return df


@pytest.fixture
def computation_test_db():
    """Create a temporary database for FeatureComputer tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# Return Feature Tests
# =============================================================================


class TestReturnFeatures:
    """Tests for return computation functions."""

    def test_returns_1d_basic(self, short_prices):
        """compute_returns_1d should calculate 1-day returns."""
        result = compute_returns_1d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "returns_1d" in result.columns
        # First day should be NaN
        assert pd.isna(result.iloc[0]["returns_1d"])
        # Should have some non-NaN values
        assert result["returns_1d"].notna().sum() > 0

    def test_returns_5d_basic(self, short_prices):
        """compute_returns_5d should calculate 5-day returns."""
        result = compute_returns_5d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "returns_5d" in result.columns
        # First 5 days should be NaN
        assert result["returns_5d"].isna().sum() >= 5

    def test_returns_20d_basic(self, short_prices):
        """compute_returns_20d should calculate 20-day returns."""
        result = compute_returns_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "returns_20d" in result.columns
        # First 20 days should be NaN
        assert result["returns_20d"].isna().sum() >= 20

    def test_returns_60d_basic(self, synthetic_prices):
        """compute_returns_60d should calculate 60-day returns."""
        result = compute_returns_60d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "returns_60d" in result.columns

    def test_returns_252d_basic(self, synthetic_prices):
        """compute_returns_252d should calculate 252-day returns."""
        result = compute_returns_252d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "returns_252d" in result.columns


# =============================================================================
# Momentum Feature Tests
# =============================================================================


class TestMomentumFeatures:
    """Tests for momentum computation functions."""

    def test_momentum_20d_basic(self, short_prices):
        """compute_momentum_20d should calculate 20-day momentum."""
        result = compute_momentum_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "momentum_20d" in result.columns

    def test_momentum_60d_basic(self, synthetic_prices):
        """compute_momentum_60d should calculate 60-day momentum."""
        result = compute_momentum_60d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "momentum_60d" in result.columns

    def test_momentum_252d_basic(self, synthetic_prices):
        """compute_momentum_252d should calculate 252-day momentum."""
        result = compute_momentum_252d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "momentum_252d" in result.columns

    def test_momentum_matches_returns(self, short_prices):
        """Momentum should equal returns over same period."""
        momentum = compute_momentum_20d(short_prices)
        returns = compute_returns_20d(short_prices)

        # They should be the same (momentum is just trailing return)
        pd.testing.assert_frame_equal(
            momentum.rename(columns={"momentum_20d": "value"}),
            returns.rename(columns={"returns_20d": "value"}),
        )


# =============================================================================
# Volatility Feature Tests
# =============================================================================


class TestVolatilityFeatures:
    """Tests for volatility computation functions."""

    def test_volatility_20d_basic(self, short_prices):
        """compute_volatility_20d should calculate 20-day volatility."""
        result = compute_volatility_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "volatility_20d" in result.columns
        # Volatility should be non-negative
        assert (result["volatility_20d"].dropna() >= 0).all()

    def test_volatility_60d_basic(self, synthetic_prices):
        """compute_volatility_60d should calculate 60-day volatility."""
        result = compute_volatility_60d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "volatility_60d" in result.columns
        # Volatility should be non-negative
        assert (result["volatility_60d"].dropna() >= 0).all()

    def test_volatility_is_annualized(self, synthetic_prices):
        """Volatility should be annualized (scaled by sqrt(252))."""
        result = compute_volatility_60d(synthetic_prices)

        # Annualized vol for stocks is typically 0.15-0.50 (15%-50%)
        # Our synthetic data has 2% daily vol -> ~32% annual
        valid_values = result["volatility_60d"].dropna()
        assert valid_values.mean() > 0.1  # Should be > 10% annualized
        assert valid_values.mean() < 1.0  # Should be < 100% annualized


# =============================================================================
# Volume Feature Tests
# =============================================================================


class TestVolumeFeatures:
    """Tests for volume computation functions."""

    def test_volume_20d_basic(self, short_prices):
        """compute_volume_20d should calculate 20-day average volume."""
        result = compute_volume_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "volume_20d" in result.columns
        # Volume should be positive
        assert (result["volume_20d"].dropna() > 0).all()

    def test_volume_ratio_basic(self, short_prices):
        """compute_volume_ratio should calculate volume/avg_volume ratio."""
        result = compute_volume_ratio(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "volume_ratio" in result.columns
        # Ratio should be around 1.0 for consistent volume
        valid_values = result["volume_ratio"].dropna()
        assert valid_values.mean() > 0.5
        assert valid_values.mean() < 2.0

    def test_obv_basic(self, short_prices):
        """compute_obv should calculate On-Balance Volume."""
        result = compute_obv(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "obv" in result.columns


# =============================================================================
# RSI Feature Tests
# =============================================================================


class TestRSIFeature:
    """Tests for RSI computation."""

    def test_rsi_14d_basic(self, synthetic_prices):
        """compute_rsi_14d should calculate 14-day RSI."""
        result = compute_rsi_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "rsi_14d" in result.columns

    def test_rsi_bounded_0_100(self, synthetic_prices):
        """RSI should be bounded between 0 and 100."""
        result = compute_rsi_14d(synthetic_prices)

        valid_values = result["rsi_14d"].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_averages_around_50(self, synthetic_prices):
        """RSI should average around 50 for random walk."""
        result = compute_rsi_14d(synthetic_prices)

        valid_values = result["rsi_14d"].dropna()
        # Should be between 30-70 on average for random data
        assert 30 < valid_values.mean() < 70


# =============================================================================
# Moving Average Feature Tests
# =============================================================================


class TestMovingAverageFeatures:
    """Tests for SMA and EMA computation functions."""

    def test_sma_20d_basic(self, short_prices):
        """compute_sma_20d should calculate 20-day SMA."""
        result = compute_sma_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "sma_20d" in result.columns

    def test_sma_50d_basic(self, synthetic_prices):
        """compute_sma_50d should calculate 50-day SMA."""
        result = compute_sma_50d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "sma_50d" in result.columns

    def test_sma_200d_basic(self, synthetic_prices):
        """compute_sma_200d should calculate 200-day SMA."""
        result = compute_sma_200d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "sma_200d" in result.columns

    def test_ema_12d_basic(self, short_prices):
        """compute_ema_12d should calculate 12-day EMA."""
        result = compute_ema_12d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_12d" in result.columns

    def test_ema_26d_basic(self, short_prices):
        """compute_ema_26d should calculate 26-day EMA."""
        result = compute_ema_26d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_26d" in result.columns

    def test_ema_crossover_basic(self, synthetic_prices):
        """compute_ema_crossover should return +1 or -1."""
        result = compute_ema_crossover(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ema_crossover" in result.columns
        # Values should be -1, 0, or +1
        valid_values = result["ema_crossover"].dropna()
        assert set(valid_values.unique()).issubset({-1.0, 0.0, 1.0})

    def test_price_to_sma_ratios(self, short_prices):
        """Price-to-SMA ratios should be around 1.0."""
        result = compute_price_to_sma_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "price_to_sma_20d" in result.columns
        valid_values = result["price_to_sma_20d"].dropna()
        # Ratio should be roughly around 1.0
        assert 0.8 < valid_values.mean() < 1.2


# =============================================================================
# MACD Feature Tests
# =============================================================================


class TestMACDFeatures:
    """Tests for MACD computation functions."""

    def test_macd_line_basic(self, synthetic_prices):
        """compute_macd_line should calculate MACD line."""
        result = compute_macd_line(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "macd_line" in result.columns

    def test_macd_signal_basic(self, synthetic_prices):
        """compute_macd_signal should calculate MACD signal."""
        result = compute_macd_signal(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "macd_signal" in result.columns

    def test_macd_histogram_basic(self, synthetic_prices):
        """compute_macd_histogram should calculate MACD histogram."""
        result = compute_macd_histogram(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "macd_histogram" in result.columns

    def test_macd_histogram_equals_line_minus_signal(self, synthetic_prices):
        """MACD histogram should equal line minus signal."""
        line = compute_macd_line(synthetic_prices)
        signal = compute_macd_signal(synthetic_prices)
        histogram = compute_macd_histogram(synthetic_prices)

        # Calculate expected histogram
        expected = line["macd_line"] - signal["macd_signal"]

        # Should be approximately equal
        diff = (histogram["macd_histogram"] - expected).dropna().abs()
        assert diff.max() < 0.01


# =============================================================================
# ATR and ADX Feature Tests
# =============================================================================


class TestATRADXFeatures:
    """Tests for ATR and ADX computation functions."""

    def test_atr_14d_basic(self, synthetic_prices):
        """compute_atr_14d should calculate 14-day ATR."""
        result = compute_atr_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "atr_14d" in result.columns
        # ATR should be positive
        valid_values = result["atr_14d"].dropna()
        assert (valid_values > 0).all()

    def test_adx_14d_basic(self, synthetic_prices):
        """compute_adx_14d should calculate 14-day ADX."""
        result = compute_adx_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "adx_14d" in result.columns

    def test_adx_bounded_0_100(self, synthetic_prices):
        """ADX should be bounded between 0 and 100."""
        result = compute_adx_14d(synthetic_prices)

        valid_values = result["adx_14d"].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()


# =============================================================================
# Bollinger Bands Feature Tests
# =============================================================================


class TestBollingerBandsFeatures:
    """Tests for Bollinger Bands computation functions."""

    def test_bb_upper_20d_basic(self, short_prices):
        """compute_bb_upper_20d should calculate upper band."""
        result = compute_bb_upper_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "bb_upper_20d" in result.columns

    def test_bb_lower_20d_basic(self, short_prices):
        """compute_bb_lower_20d should calculate lower band."""
        result = compute_bb_lower_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "bb_lower_20d" in result.columns

    def test_bb_width_20d_basic(self, short_prices):
        """compute_bb_width_20d should calculate band width."""
        result = compute_bb_width_20d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "bb_width_20d" in result.columns
        # Width should be positive
        valid_values = result["bb_width_20d"].dropna()
        assert (valid_values > 0).all()

    def test_bb_upper_greater_than_lower(self, short_prices):
        """Upper band should always be greater than lower band."""
        upper = compute_bb_upper_20d(short_prices)
        lower = compute_bb_lower_20d(short_prices)

        # Align indices
        combined = pd.concat([upper, lower], axis=1).dropna()
        assert (combined["bb_upper_20d"] > combined["bb_lower_20d"]).all()


# =============================================================================
# Stochastic Oscillator Feature Tests
# =============================================================================


class TestStochasticFeatures:
    """Tests for Stochastic oscillator computation functions."""

    def test_stoch_k_14d_basic(self, synthetic_prices):
        """compute_stoch_k_14d should calculate %K."""
        result = compute_stoch_k_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "stoch_k_14d" in result.columns

    def test_stoch_d_14d_basic(self, synthetic_prices):
        """compute_stoch_d_14d should calculate %D."""
        result = compute_stoch_d_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "stoch_d_14d" in result.columns

    def test_stochastic_bounded_0_100(self, synthetic_prices):
        """Stochastic %K and %D should be bounded 0-100."""
        k = compute_stoch_k_14d(synthetic_prices)
        d = compute_stoch_d_14d(synthetic_prices)

        k_valid = k["stoch_k_14d"].dropna()
        d_valid = d["stoch_d_14d"].dropna()

        assert (k_valid >= 0).all() and (k_valid <= 100).all()
        assert (d_valid >= 0).all() and (d_valid <= 100).all()


# =============================================================================
# Other Oscillator Feature Tests
# =============================================================================


class TestOtherOscillators:
    """Tests for CCI, ROC, Williams %R, MFI computation functions."""

    def test_cci_20d_basic(self, synthetic_prices):
        """compute_cci_20d should calculate CCI."""
        result = compute_cci_20d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "cci_20d" in result.columns

    def test_roc_10d_basic(self, short_prices):
        """compute_roc_10d should calculate Rate of Change."""
        result = compute_roc_10d(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "roc_10d" in result.columns

    def test_williams_r_14d_basic(self, synthetic_prices):
        """compute_williams_r_14d should calculate Williams %R."""
        result = compute_williams_r_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "williams_r_14d" in result.columns

    def test_williams_r_bounded(self, synthetic_prices):
        """Williams %R should be bounded between -100 and 0."""
        result = compute_williams_r_14d(synthetic_prices)

        valid_values = result["williams_r_14d"].dropna()
        assert (valid_values >= -100).all()
        assert (valid_values <= 0).all()

    def test_mfi_14d_basic(self, synthetic_prices):
        """compute_mfi_14d should calculate MFI."""
        result = compute_mfi_14d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "mfi_14d" in result.columns

    def test_mfi_bounded_0_100(self, synthetic_prices):
        """MFI should be bounded 0-100."""
        result = compute_mfi_14d(synthetic_prices)

        valid_values = result["mfi_14d"].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()


# =============================================================================
# Trend and VWAP Feature Tests
# =============================================================================


class TestTrendVWAPFeatures:
    """Tests for trend and VWAP computation functions."""

    def test_trend_basic(self, synthetic_prices):
        """compute_trend should return +1 or -1."""
        result = compute_trend(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "trend" in result.columns
        # Values should be -1, 0, or +1
        valid_values = result["trend"].dropna()
        assert set(valid_values.unique()).issubset({-1.0, 0.0, 1.0})

    def test_vwap_20d_basic(self, synthetic_prices):
        """compute_vwap_20d should calculate VWAP."""
        result = compute_vwap_20d(synthetic_prices)

        assert isinstance(result, pd.DataFrame)
        assert "vwap_20d" in result.columns
        # VWAP should be positive
        valid_values = result["vwap_20d"].dropna()
        assert (valid_values > 0).all()


# =============================================================================
# Fundamental Passthrough Feature Tests
# =============================================================================


class TestFundamentalPassthroughs:
    """Tests for fundamental passthrough features."""

    def test_market_cap_returns_nan(self, short_prices):
        """compute_market_cap should return NaN (passthrough)."""
        result = compute_market_cap(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "market_cap" in result.columns
        # All values should be NaN (values come from ingestion)
        assert result["market_cap"].isna().all()

    def test_pe_ratio_returns_nan(self, short_prices):
        """compute_pe_ratio should return NaN (passthrough)."""
        result = compute_pe_ratio(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "pe_ratio" in result.columns
        assert result["pe_ratio"].isna().all()

    def test_pb_ratio_returns_nan(self, short_prices):
        """compute_pb_ratio should return NaN (passthrough)."""
        result = compute_pb_ratio(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "pb_ratio" in result.columns
        assert result["pb_ratio"].isna().all()

    def test_dividend_yield_returns_nan(self, short_prices):
        """compute_dividend_yield should return NaN (passthrough)."""
        result = compute_dividend_yield(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "dividend_yield" in result.columns
        assert result["dividend_yield"].isna().all()

    def test_ev_ebitda_returns_nan(self, short_prices):
        """compute_ev_ebitda should return NaN (passthrough)."""
        result = compute_ev_ebitda(short_prices)

        assert isinstance(result, pd.DataFrame)
        assert "ev_ebitda" in result.columns
        assert result["ev_ebitda"].isna().all()


# =============================================================================
# FEATURE_FUNCTIONS Registry Tests
# =============================================================================


class TestFeatureFunctionsRegistry:
    """Tests for the FEATURE_FUNCTIONS mapping."""

    def test_registry_has_expected_features(self):
        """FEATURE_FUNCTIONS should contain all expected features."""
        expected_features = [
            "momentum_20d", "momentum_60d", "momentum_252d",
            "volatility_20d", "volatility_60d",
            "returns_1d", "returns_5d", "returns_20d", "returns_60d", "returns_252d",
            "volume_20d", "volume_ratio", "obv",
            "rsi_14d", "atr_14d", "adx_14d",
            "macd_line", "macd_signal", "macd_histogram",
            "cci_20d", "roc_10d", "trend",
            "bb_upper_20d", "bb_lower_20d", "bb_width_20d",
            "stoch_k_14d", "stoch_d_14d",
            "sma_20d", "sma_50d", "sma_200d",
            "price_to_sma_20d", "price_to_sma_50d", "price_to_sma_200d",
            "ema_12d", "ema_26d", "ema_crossover",
            "williams_r_14d", "mfi_14d", "vwap_20d",
            "market_cap", "pe_ratio", "pb_ratio", "dividend_yield", "ev_ebitda",
        ]

        for feature in expected_features:
            assert feature in FEATURE_FUNCTIONS, f"Missing feature: {feature}"

    def test_registry_functions_are_callable(self):
        """All functions in FEATURE_FUNCTIONS should be callable."""
        for name, func in FEATURE_FUNCTIONS.items():
            assert callable(func), f"{name} is not callable"

    def test_registry_function_count(self):
        """FEATURE_FUNCTIONS should have expected count."""
        # 39 technical + 6 fundamental = 45 features
        assert len(FEATURE_FUNCTIONS) == 45


# =============================================================================
# FeatureComputer Class Tests
# =============================================================================


class TestFeatureComputerInit:
    """Tests for FeatureComputer initialization."""

    def test_init_creates_instance(self, computation_test_db):
        """FeatureComputer should initialize with database."""
        computer = FeatureComputer(computation_test_db)

        assert computer.db is not None
        assert computer.registry is not None


class TestFeatureComputerLogLineage:
    """Tests for FeatureComputer lineage logging."""

    def test_log_lineage_event_returns_id(self, computation_test_db):
        """_log_lineage_event should return lineage_id."""
        computer = FeatureComputer(computation_test_db)

        lineage_id = computer._log_lineage_event(
            event_type="feature_computed",  # Valid event type
            details={"test": "data"},
            actor="test",
        )

        assert isinstance(lineage_id, int)
        assert lineage_id > 0

    def test_log_lineage_event_stores_in_db(self, computation_test_db):
        """_log_lineage_event should store event in database."""
        computer = FeatureComputer(computation_test_db)

        lineage_id = computer._log_lineage_event(
            event_type="feature_computed",  # Valid event type
            details={"key": "value"},
            actor="test_actor",
        )

        # Verify stored
        row = computer.db.fetchone(
            "SELECT event_type, actor FROM lineage WHERE lineage_id = ?",
            (lineage_id,)
        )
        assert row is not None
        assert row[0] == "feature_computed"
        assert row[1] == "test_actor"


class TestFeatureComputerLoadPriceData:
    """Tests for FeatureComputer._load_price_data method."""

    def test_load_price_data_empty_db(self, computation_test_db):
        """_load_price_data should return empty DataFrame if no data."""
        computer = FeatureComputer(computation_test_db)

        result = computer._load_price_data(
            symbols=["AAPL"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_price_data_with_data(self, computation_test_db):
        """_load_price_data should load price data from database."""
        computer = FeatureComputer(computation_test_db)

        # Insert symbol into symbols table first (FK constraint)
        computer.db.execute("""
            INSERT INTO symbols (symbol, name, exchange, asset_type)
            VALUES ('AAPL', 'Apple Inc.', 'NASDAQ', 'equity')
        """)

        # Insert into universe (FK constraint for prices)
        computer.db.execute("""
            INSERT INTO universe (symbol, date, in_universe, sector)
            VALUES
                ('AAPL', '2023-06-01', TRUE, 'Technology'),
                ('AAPL', '2023-06-02', TRUE, 'Technology')
        """)

        # Insert test data
        computer.db.execute("""
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES
                ('AAPL', '2023-06-01', 100, 102, 99, 101, 101, 1000000, 'test'),
                ('AAPL', '2023-06-02', 101, 103, 100, 102, 102, 1100000, 'test')
        """)

        result = computer._load_price_data(
            symbols=["AAPL"],
            start=date(2023, 6, 1),
            end=date(2023, 6, 2),
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "close" in result.columns


class TestFeatureComputerComputeFeature:
    """Tests for FeatureComputer._compute_feature method."""

    def test_compute_feature_unknown_returns_nan(self, computation_test_db, short_prices):
        """_compute_feature should return NaN for unknown features."""
        computer = FeatureComputer(computation_test_db)

        feature_def = {"version": "v1"}
        result = computer._compute_feature(
            prices=short_prices,
            feature_name="unknown_feature",
            feature_def=feature_def,
        )

        assert isinstance(result, pd.DataFrame)
        assert "unknown_feature" in result.columns
        assert result["unknown_feature"].isna().all()

    def test_compute_feature_known_feature(self, computation_test_db, short_prices):
        """_compute_feature should compute known features."""
        computer = FeatureComputer(computation_test_db)

        feature_def = {"version": "v1"}
        result = computer._compute_feature(
            prices=short_prices,
            feature_name="returns_1d",
            feature_def=feature_def,
        )

        assert isinstance(result, pd.DataFrame)
        assert "returns_1d" in result.columns
        # Should have some valid values
        assert result["returns_1d"].notna().sum() > 0


# =============================================================================
# Multi-Symbol Tests
# =============================================================================


class TestMultiSymbolComputation:
    """Tests for computing features across multiple symbols."""

    def test_features_computed_for_all_symbols(self, synthetic_prices):
        """Features should be computed for each symbol independently."""
        result = compute_momentum_20d(synthetic_prices)

        # Should have both symbols in result
        symbols = result.index.get_level_values("symbol").unique()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_feature_values_differ_by_symbol(self, synthetic_prices):
        """Feature values should differ between symbols."""
        result = compute_sma_20d(synthetic_prices)

        # Get a date that both symbols have data for
        result_reset = result.reset_index()
        valid_dates = result_reset.groupby("date").size()
        valid_date = valid_dates[valid_dates == 2].index[10]  # Pick a date with both

        aapl_val = result_reset[
            (result_reset["date"] == valid_date) &
            (result_reset["symbol"] == "AAPL")
        ]["sma_20d"].iloc[0]

        msft_val = result_reset[
            (result_reset["date"] == valid_date) &
            (result_reset["symbol"] == "MSFT")
        ]["sma_20d"].iloc[0]

        # Values should be different (AAPL ~150, MSFT ~250)
        assert aapl_val != msft_val
