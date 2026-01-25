"""Tests for hrp/data/ingestion/features.py."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.data.ingestion.features import (
    _compute_all_features,
    _fetch_prices,
    _upsert_features,
    compute_features,
    compute_features_batch,
    get_feature_stats,
)


class TestComputeFeaturesBatch:
    """Tests for compute_features_batch function."""

    def test_basic(self):
        """Test that vectorized computation works."""
        mock_result = {
            "features_computed": 32,
            "rows_stored": 1000,
        }

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.features.computation.FeatureComputer") as mock_computer_class:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            mock_computer = MagicMock()
            mock_computer.compute_and_store_features.return_value = mock_result
            mock_computer_class.return_value = mock_computer

            result = compute_features_batch(
                symbols=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
            )

        assert result["symbols_requested"] == 2
        assert result["symbols_success"] == 2
        assert result["features_computed"] == 32
        assert result["rows_inserted"] == 1000
        mock_computer.compute_and_store_features.assert_called_once()

    def test_no_symbols(self):
        """Test that returns empty stats when no symbols."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []  # No symbols
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            result = compute_features_batch(symbols=None)

        assert result["symbols_requested"] == 0
        assert result["symbols_success"] == 0
        assert result["features_computed"] == 0

    def test_fallback_on_error(self):
        """Test that falls back to compute_features on error."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.features.computation.FeatureComputer") as mock_computer_class, \
             patch("hrp.data.ingestion.features.compute_features") as mock_compute:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            # Make FeatureComputer raise an error
            mock_computer_class.side_effect = Exception("Batch computation failed")

            mock_compute.return_value = {
                "symbols_requested": 2,
                "symbols_success": 2,
                "symbols_failed": 0,
                "features_computed": 64,
                "rows_inserted": 500,
                "failed_symbols": [],
            }

            result = compute_features_batch(
                symbols=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
            )

        # Should have called fallback
        mock_compute.assert_called_once()
        assert result["symbols_success"] == 2

    def test_auto_discover_symbols(self):
        """Test that symbols are auto-discovered from database when None."""
        mock_result = {
            "features_computed": 32,
            "rows_stored": 500,
        }

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.features.computation.FeatureComputer") as mock_computer_class:
            mock_db = MagicMock()
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = [
                ("AAPL",), ("MSFT",), ("GOOGL",)
            ]
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            mock_computer = MagicMock()
            mock_computer.compute_and_store_features.return_value = mock_result
            mock_computer_class.return_value = mock_computer

            result = compute_features_batch(symbols=None)

        # Should have discovered 3 symbols
        assert result["symbols_requested"] == 3

    def test_date_defaults(self):
        """Test that default dates work (end=today, start=30 days ago)."""
        mock_result = {
            "features_computed": 32,
            "rows_stored": 100,
        }

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.features.computation.FeatureComputer") as mock_computer_class:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            mock_computer = MagicMock()
            mock_computer.compute_and_store_features.return_value = mock_result
            mock_computer_class.return_value = mock_computer

            # Call without start/end
            result = compute_features_batch(symbols=["AAPL"])

        # Should complete without error
        assert result["symbols_requested"] == 1
        mock_computer.compute_and_store_features.assert_called_once()


class TestComputeAllFeatures:
    """Tests for _compute_all_features function."""

    @pytest.fixture
    def sample_price_df(self):
        """Create sample price data with enough history."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.02)
        return pd.DataFrame({
            "date": dates,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.randint(1000000, 10000000, 300),
        })

    def test_returns_long_format(self, sample_price_df):
        """Test that output is in long format with correct columns."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        assert not result.empty
        assert "symbol" in result.columns
        assert "date" in result.columns
        assert "feature_name" in result.columns
        assert "value" in result.columns
        assert "version" in result.columns

        # Check symbol is correct
        assert (result["symbol"] == "AAPL").all()
        assert (result["version"] == "v1").all()

    def test_computes_all_32_features(self, sample_price_df):
        """Test that all 32 features are computed."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        feature_names = result["feature_name"].unique()
        expected_features = [
            "returns_1d", "returns_5d", "returns_20d", "returns_60d", "returns_252d",
            "momentum_20d", "momentum_60d", "momentum_252d",
            "volatility_20d", "volatility_60d",
            "volume_20d", "volume_ratio", "obv",
            "rsi_14d", "atr_14d", "adx_14d", "cci_20d", "roc_10d",
            "macd_line", "macd_signal", "macd_histogram",
            "sma_20d", "sma_50d", "sma_200d",
            "price_to_sma_20d", "price_to_sma_50d", "price_to_sma_200d",
            "trend",
            "bb_upper_20d", "bb_lower_20d", "bb_width_20d",
            "stoch_k_14d", "stoch_d_14d",
        ]

        for feature in expected_features:
            assert feature in feature_names, f"Missing feature: {feature}"

    def test_rsi_calculation(self, sample_price_df):
        """Test that RSI is computed correctly (0-100 range)."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        rsi_values = result[result["feature_name"] == "rsi_14d"]["value"]
        assert len(rsi_values) > 0
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100

    def test_macd_calculation(self, sample_price_df):
        """Test that MACD components are computed."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        macd_line = result[result["feature_name"] == "macd_line"]["value"]
        macd_signal = result[result["feature_name"] == "macd_signal"]["value"]
        macd_hist = result[result["feature_name"] == "macd_histogram"]["value"]

        assert len(macd_line) > 0
        assert len(macd_signal) > 0
        assert len(macd_hist) > 0

    def test_bollinger_bands(self, sample_price_df):
        """Test that Bollinger Bands are computed correctly."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        bb_upper = result[result["feature_name"] == "bb_upper_20d"]["value"]
        bb_lower = result[result["feature_name"] == "bb_lower_20d"]["value"]
        bb_width = result[result["feature_name"] == "bb_width_20d"]["value"]

        assert len(bb_upper) > 0
        assert len(bb_lower) > 0
        assert len(bb_width) > 0

        # Upper should be > lower
        # (compare same dates)
        merged = result[result["feature_name"].isin(["bb_upper_20d", "bb_lower_20d"])]
        pivot = merged.pivot(index="date", columns="feature_name", values="value")
        assert (pivot["bb_upper_20d"] > pivot["bb_lower_20d"]).all()

    def test_handles_nan(self, sample_price_df):
        """Test that NaN values are dropped from output."""
        result = _compute_all_features(sample_price_df, "AAPL", "v1")

        # No NaN values should be in the value column
        assert not result["value"].isna().any()

    def test_empty_input(self):
        """Test that empty DataFrame returns empty result."""
        empty_df = pd.DataFrame()
        result = _compute_all_features(empty_df, "AAPL", "v1")

        assert result.empty

    def test_uses_adj_close(self, sample_price_df):
        """Test that adj_close is used when available."""
        # Modify adj_close to be different from close
        df = sample_price_df.copy()
        df["adj_close"] = df["close"] * 0.9  # 10% split adjustment

        result = _compute_all_features(df, "AAPL", "v1")

        # Should have computed features based on adj_close
        assert not result.empty


class TestComputeFeatures:
    """Tests for compute_features function."""

    def test_insufficient_data_skipped(self):
        """Test that symbols with <60 days of data are skipped."""
        # Only 30 days of data
        short_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=30),
            "high": [102] * 30,
            "low": [98] * 30,
            "close": [100] * 30,
            "adj_close": [100] * 30,
            "volume": [1000000] * 30,
        })

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.features._fetch_prices") as mock_fetch:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_fetch.return_value = short_df

            result = compute_features(
                symbols=["AAPL"],
                start=date(2020, 1, 1),
                end=date(2020, 1, 31),
            )

        assert result["symbols_failed"] == 1
        assert result["symbols_success"] == 0
        assert "AAPL" in result["failed_symbols"]

    def test_error_handling(self):
        """Test that errors are logged and symbol tracked."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.features._fetch_prices") as mock_fetch:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_fetch.side_effect = Exception("Database error")

            result = compute_features(
                symbols=["AAPL"],
                start=date(2020, 1, 1),
                end=date(2020, 12, 31),
            )

        assert result["symbols_failed"] == 1
        assert result["symbols_success"] == 0
        # Failed symbol should have error info
        assert len(result["failed_symbols"]) == 1
        assert result["failed_symbols"][0]["symbol"] == "AAPL"
        assert "Database error" in result["failed_symbols"][0]["error"]

    def test_date_filtering(self):
        """Test that output is filtered to requested date range."""
        # Create data spanning multiple months
        dates = pd.date_range("2019-01-01", periods=500, freq="B")
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)
        price_df = pd.DataFrame({
            "date": dates,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.randint(1000000, 10000000, 500),
        })

        features_df = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "date": pd.date_range("2020-01-01", periods=100, freq="B"),
            "feature_name": ["returns_1d"] * 100,
            "value": np.random.randn(100) * 0.02,
            "version": ["v1"] * 100,
        })

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db, \
             patch("hrp.data.ingestion.features._fetch_prices") as mock_fetch, \
             patch("hrp.data.ingestion.features._compute_all_features") as mock_compute, \
             patch("hrp.data.ingestion.features._upsert_features") as mock_upsert:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_fetch.return_value = price_df
            mock_compute.return_value = features_df
            mock_upsert.return_value = 100

            result = compute_features(
                symbols=["AAPL"],
                start=date(2020, 1, 1),
                end=date(2020, 3, 31),
            )

        assert result["symbols_success"] == 1

    def test_no_symbols_returns_empty(self):
        """Test that empty symbol list returns empty stats."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            result = compute_features(symbols=None)

        assert result["symbols_requested"] == 0
        assert result["features_computed"] == 0


class TestUpsertFeatures:
    """Tests for _upsert_features function."""

    def test_insert_new(self):
        """Test that new features are inserted."""
        features_df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": [date(2020, 1, 2), date(2020, 1, 3)],
            "feature_name": ["returns_1d", "returns_1d"],
            "value": [0.01, 0.02],
            "version": ["v1", "v1"],
        })

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_conn = MagicMock()
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            rows = _upsert_features(mock_db, features_df)

        assert rows == 2
        # Verify execute was called for inserts
        assert mock_conn.execute.call_count > 0

    def test_empty_df_returns_zero(self):
        """Test that empty DataFrame returns 0."""
        empty_df = pd.DataFrame()

        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            rows = _upsert_features(mock_db, empty_df)

        assert rows == 0


class TestGetFeatureStats:
    """Tests for get_feature_stats function."""

    def test_basic(self):
        """Test that expected structure is returned."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_conn = MagicMock()

            # Setup mock return values
            mock_conn.execute.return_value.fetchone.side_effect = [
                (10000,),  # total rows
                (5,),      # unique symbols
                (32,),     # unique features
                (date(2020, 1, 1), date(2023, 12, 31)),  # date range
            ]
            mock_conn.execute.return_value.fetchall.side_effect = [
                [  # per_symbol
                    ("AAPL", 2000, 32, date(2020, 1, 1), date(2023, 12, 31)),
                    ("MSFT", 2000, 32, date(2020, 1, 1), date(2023, 12, 31)),
                ],
                [  # feature_coverage
                    ("returns_1d", 5, 2500),
                    ("momentum_20d", 5, 2500),
                ],
            ]
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            result = get_feature_stats()

        assert "total_rows" in result
        assert "unique_symbols" in result
        assert "unique_features" in result
        assert "date_range" in result
        assert "per_symbol" in result
        assert "feature_coverage" in result

        assert result["total_rows"] == 10000
        assert result["unique_symbols"] == 5
        assert result["unique_features"] == 32

    def test_empty_db(self):
        """Test that handles empty database gracefully."""
        with patch("hrp.data.ingestion.features.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_conn = MagicMock()

            # Empty database returns
            mock_conn.execute.return_value.fetchone.side_effect = [
                (0,),       # total rows
                (0,),       # unique symbols
                (0,),       # unique features
                (None, None),  # date range
            ]
            mock_conn.execute.return_value.fetchall.side_effect = [
                [],  # per_symbol
                [],  # feature_coverage
            ]
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db

            result = get_feature_stats()

        assert result["total_rows"] == 0
        assert result["unique_symbols"] == 0
        assert len(result["per_symbol"]) == 0
        assert len(result["feature_coverage"]) == 0


class TestFetchPrices:
    """Tests for _fetch_prices function."""

    def test_basic(self):
        """Test that prices are fetched correctly."""
        mock_df = pd.DataFrame({
            "date": pd.date_range("2020-01-02", periods=5),
            "high": [102, 103, 104, 103, 105],
            "low": [98, 99, 100, 99, 101],
            "close": [100, 101, 102, 101, 103],
            "adj_close": [100, 101, 102, 101, 103],
            "volume": [1000000] * 5,
        })

        with patch("hrp.utils.calendar.get_trading_days") as mock_trading_days:
            mock_trading_days.return_value = pd.date_range("2020-01-02", periods=5)

            mock_db = MagicMock()
            mock_conn = MagicMock()
            mock_conn.execute.return_value.df.return_value = mock_df
            mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

            result = _fetch_prices(mock_db, "AAPL", date(2020, 1, 1), date(2020, 1, 10))

        assert len(result) == 5
        assert "close" in result.columns
        assert "adj_close" in result.columns

    def test_no_trading_days(self):
        """Test that empty DataFrame returned when no trading days."""
        with patch("hrp.utils.calendar.get_trading_days") as mock_trading_days:
            mock_trading_days.return_value = []  # No trading days (e.g., weekend range)

            mock_db = MagicMock()

            result = _fetch_prices(mock_db, "AAPL", date(2020, 1, 4), date(2020, 1, 5))

        assert result.empty
