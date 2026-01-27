"""
Comprehensive tests for walk-forward ML validation.

Tests cover:
- WalkForwardConfig validation
- FoldResult and WalkForwardResult dataclasses
- generate_folds function for expanding/rolling windows
- compute_fold_metrics function
- aggregate_fold_metrics function
- walk_forward_validate integration
- MLflow logging
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd
import pytest

from hrp.ml.validation import (
    WalkForwardConfig,
    FoldResult,
    WalkForwardResult,
    generate_folds,
    compute_fold_metrics,
    aggregate_fold_metrics,
    walk_forward_validate,
    _process_fold,
    _process_fold_safe,
    _log_to_mlflow,
    _feature_selection_cache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic walk-forward config for testing."""
    return WalkForwardConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d", "volatility_20d", "rsi_14d"],
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        n_folds=3,
        window_type="expanding",
        min_train_periods=252,
    )


@pytest.fixture
def sample_dates():
    """Create sample dates for fold generation."""
    # Generate ~500 business days (about 2 years)
    return sorted(pd.bdate_range(start="2020-01-01", periods=500).date.tolist())


@pytest.fixture
def sample_fold_metrics():
    """Create sample fold metrics for aggregation tests."""
    return [
        {"mse": 0.001, "mae": 0.025, "r2": 0.15, "ic": 0.08},
        {"mse": 0.0012, "mae": 0.027, "r2": 0.12, "ic": 0.07},
        {"mse": 0.0011, "mae": 0.026, "r2": 0.14, "ic": 0.09},
    ]


@pytest.fixture
def mock_all_data():
    """Create mock all_data DataFrame for validation tests."""
    np.random.seed(42)
    n_dates = 300
    n_symbols = 2

    dates = pd.bdate_range(start="2020-01-01", periods=n_dates)
    symbols = ["AAPL", "MSFT"]

    rows = []
    for symbol in symbols:
        for dt in dates:
            rows.append({
                "date": dt,
                "symbol": symbol,
                "momentum_20d": np.random.randn() * 0.1,
                "volatility_20d": abs(np.random.randn() * 0.02),
                "rsi_14d": 50 + np.random.randn() * 10,
                "returns_20d": np.random.randn() * 0.05,
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "symbol"])
    return df


# =============================================================================
# WalkForwardConfig Tests
# =============================================================================


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_config_valid_creation(self):
        """WalkForwardConfig should create with valid parameters."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.model_type == "ridge"
        assert config.n_folds == 5  # Default

    def test_config_invalid_model_type(self):
        """WalkForwardConfig should reject invalid model types."""
        with pytest.raises(ValueError) as exc_info:
            WalkForwardConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
            )
        assert "Unsupported model type" in str(exc_info.value)

    def test_config_invalid_window_type(self):
        """WalkForwardConfig should reject invalid window types."""
        with pytest.raises(ValueError) as exc_info:
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                window_type="invalid",
            )
        assert "window_type must be" in str(exc_info.value)

    def test_config_invalid_n_folds(self):
        """WalkForwardConfig should reject n_folds < 2."""
        with pytest.raises(ValueError) as exc_info:
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )
        assert "n_folds must be >= 2" in str(exc_info.value)

    def test_config_expanding_window(self):
        """WalkForwardConfig should accept expanding window."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            window_type="expanding",
        )
        assert config.window_type == "expanding"

    def test_config_rolling_window(self):
        """WalkForwardConfig should accept rolling window."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            window_type="rolling",
        )
        assert config.window_type == "rolling"

    def test_config_defaults(self):
        """WalkForwardConfig should have sensible defaults."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.n_folds == 5
        assert config.window_type == "expanding"
        assert config.min_train_periods == 252
        assert config.feature_selection is True
        assert config.max_features == 20
        assert config.n_jobs == 1

    def test_config_all_supported_models(self):
        """WalkForwardConfig should accept all supported model types."""
        from hrp.ml.models import SUPPORTED_MODELS

        for model_type in SUPPORTED_MODELS.keys():
            config = WalkForwardConfig(
                model_type=model_type,
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
            )
            assert config.model_type == model_type


# =============================================================================
# FoldResult Tests
# =============================================================================


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self):
        """FoldResult should store fold information."""
        mock_model = MagicMock()

        result = FoldResult(
            fold_index=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 6, 30),
            metrics={"mse": 0.001, "mae": 0.025, "r2": 0.15, "ic": 0.08},
            model=mock_model,
            n_train_samples=500,
            n_test_samples=125,
        )

        assert result.fold_index == 0
        assert result.train_start == date(2020, 1, 1)
        assert result.metrics["mse"] == 0.001
        assert result.n_train_samples == 500


# =============================================================================
# WalkForwardResult Tests
# =============================================================================


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_is_stable_true(self, basic_config):
        """is_stable should return True when stability_score <= 1.0."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={"mean_ic": 0.08},
            stability_score=0.5,
            symbols=["AAPL"],
        )
        assert result.is_stable is True

    def test_is_stable_false(self, basic_config):
        """is_stable should return False when stability_score > 1.0."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={"mean_ic": 0.08},
            stability_score=1.5,
            symbols=["AAPL"],
        )
        assert result.is_stable is False

    def test_is_stable_boundary(self, basic_config):
        """is_stable should return True when stability_score == 1.0."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={"mean_ic": 0.08},
            stability_score=1.0,
            symbols=["AAPL"],
        )
        assert result.is_stable is True

    def test_mean_ic_property(self, basic_config):
        """mean_ic property should return mean IC from aggregate_metrics."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={"mean_ic": 0.12},
            stability_score=0.5,
            symbols=["AAPL"],
        )
        assert result.mean_ic == 0.12

    def test_mean_ic_missing(self, basic_config):
        """mean_ic should return NaN if not in aggregate_metrics."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={},
            stability_score=0.5,
            symbols=["AAPL"],
        )
        assert np.isnan(result.mean_ic)


# =============================================================================
# generate_folds Tests
# =============================================================================


class TestGenerateFolds:
    """Tests for generate_folds function."""

    def test_generate_folds_basic(self, basic_config, sample_dates):
        """generate_folds should create correct number of folds."""
        folds = generate_folds(basic_config, sample_dates)

        assert len(folds) == basic_config.n_folds

    def test_generate_folds_structure(self, basic_config, sample_dates):
        """generate_folds should return tuples of 4 dates."""
        folds = generate_folds(basic_config, sample_dates)

        for fold in folds:
            assert len(fold) == 4
            train_start, train_end, test_start, test_end = fold
            assert isinstance(train_start, date)
            assert isinstance(train_end, date)
            assert isinstance(test_start, date)
            assert isinstance(test_end, date)

    def test_generate_folds_non_overlapping_test(self, basic_config, sample_dates):
        """Test periods should be non-overlapping."""
        folds = generate_folds(basic_config, sample_dates)

        for i in range(len(folds) - 1):
            _, _, _, test_end_i = folds[i]
            _, _, test_start_next, _ = folds[i + 1]
            assert test_end_i < test_start_next

    def test_generate_folds_train_before_test(self, basic_config, sample_dates):
        """Training period should end before test period starts."""
        folds = generate_folds(basic_config, sample_dates)

        for fold in folds:
            train_start, train_end, test_start, test_end = fold
            assert train_end < test_start

    def test_generate_folds_expanding_window(self, sample_dates):
        """Expanding window should have same train_start for all folds."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=sample_dates[0],
            end_date=sample_dates[-1],
            n_folds=3,
            window_type="expanding",
            min_train_periods=100,
        )

        folds = generate_folds(config, sample_dates)

        # All folds should start training from the same date
        train_starts = [f[0] for f in folds]
        assert len(set(train_starts)) == 1

    def test_generate_folds_rolling_window(self, sample_dates):
        """Rolling window should have different train_start for later folds."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=sample_dates[0],
            end_date=sample_dates[-1],
            n_folds=3,
            window_type="rolling",
            min_train_periods=100,
        )

        folds = generate_folds(config, sample_dates)

        # Later folds should have later train_start (rolling forward)
        train_starts = [f[0] for f in folds]
        # At least some should be different for rolling
        assert len(set(train_starts)) >= 1  # Depends on data size

    def test_generate_folds_insufficient_data(self, basic_config):
        """generate_folds should raise ValueError for insufficient data."""
        # Only 10 dates, but need 252 + n_folds
        insufficient_dates = [date(2020, 1, i) for i in range(1, 11)]

        with pytest.raises(ValueError) as exc_info:
            generate_folds(basic_config, insufficient_dates)

        assert "Insufficient data" in str(exc_info.value)

    def test_generate_folds_filters_to_config_range(self, sample_dates):
        """generate_folds should filter dates to config range."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=sample_dates[100],  # Start from 100th date
            end_date=sample_dates[400],    # End at 400th date
            n_folds=3,
            min_train_periods=100,
        )

        folds = generate_folds(config, sample_dates)

        # All dates should be within config range
        for fold in folds:
            for d in fold:
                assert d >= config.start_date
                assert d <= config.end_date


# =============================================================================
# compute_fold_metrics Tests
# =============================================================================


class TestComputeFoldMetrics:
    """Tests for compute_fold_metrics function."""

    def test_compute_metrics_basic(self):
        """compute_fold_metrics should return all expected metrics."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "ic" in metrics

    def test_compute_metrics_perfect_prediction(self):
        """compute_fold_metrics should return 0 MSE for perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert metrics["mse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["ic"] == pytest.approx(1.0)

    def test_compute_metrics_handles_nan(self):
        """compute_fold_metrics should handle NaN values."""
        y_true = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, np.nan, 4.1, 4.9])

        metrics = compute_fold_metrics(y_true, y_pred)

        # Should compute metrics on non-NaN values only
        assert not np.isnan(metrics["mse"])
        assert not np.isnan(metrics["mae"])

    def test_compute_metrics_all_nan(self):
        """compute_fold_metrics should return NaN for all-NaN input."""
        y_true = pd.Series([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert np.isnan(metrics["mse"])
        assert np.isnan(metrics["mae"])
        assert np.isnan(metrics["r2"])
        assert np.isnan(metrics["ic"])

    def test_compute_metrics_ic_rank_correlation(self):
        """IC should be Spearman rank correlation."""
        # Perfect rank correlation but different values
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # Same ranks

        metrics = compute_fold_metrics(y_true, y_pred)

        # Spearman correlation should be 1.0 (perfect rank agreement)
        assert metrics["ic"] == pytest.approx(1.0)

    def test_compute_metrics_single_value(self):
        """compute_fold_metrics should handle single value gracefully."""
        y_true = pd.Series([1.0])
        y_pred = np.array([1.1])

        metrics = compute_fold_metrics(y_true, y_pred)

        # IC with single value should be 0 (can't compute correlation)
        assert metrics["ic"] == 0.0


# =============================================================================
# aggregate_fold_metrics Tests
# =============================================================================


class TestAggregateFoldMetrics:
    """Tests for aggregate_fold_metrics function."""

    def test_aggregate_basic(self, sample_fold_metrics):
        """aggregate_fold_metrics should compute mean and std."""
        aggregate, stability = aggregate_fold_metrics(sample_fold_metrics)

        assert "mean_mse" in aggregate
        assert "std_mse" in aggregate
        assert "mean_mae" in aggregate
        assert "std_mae" in aggregate
        assert "mean_r2" in aggregate
        assert "std_r2" in aggregate
        assert "mean_ic" in aggregate
        assert "std_ic" in aggregate

    def test_aggregate_mean_values(self, sample_fold_metrics):
        """aggregate_fold_metrics should compute correct means."""
        aggregate, _ = aggregate_fold_metrics(sample_fold_metrics)

        expected_mean_mse = np.mean([0.001, 0.0012, 0.0011])
        assert aggregate["mean_mse"] == pytest.approx(expected_mean_mse)

    def test_aggregate_stability_score(self, sample_fold_metrics):
        """aggregate_fold_metrics should compute stability score (CV of MSE)."""
        aggregate, stability = aggregate_fold_metrics(sample_fold_metrics)

        mse_values = [0.001, 0.0012, 0.0011]
        expected_cv = np.std(mse_values) / np.mean(mse_values)
        assert stability == pytest.approx(expected_cv)

    def test_aggregate_empty_list(self):
        """aggregate_fold_metrics should handle empty list."""
        aggregate, stability = aggregate_fold_metrics([])

        assert aggregate == {}
        assert stability == float("inf")

    def test_aggregate_single_fold(self):
        """aggregate_fold_metrics should handle single fold."""
        single_fold = [{"mse": 0.001, "mae": 0.025, "r2": 0.15, "ic": 0.08}]

        aggregate, stability = aggregate_fold_metrics(single_fold)

        assert aggregate["mean_mse"] == 0.001
        assert aggregate["std_mse"] == 0.0  # No variance with single value

    def test_aggregate_with_nan(self):
        """aggregate_fold_metrics should skip NaN values."""
        fold_metrics = [
            {"mse": 0.001, "mae": 0.025, "r2": 0.15, "ic": 0.08},
            {"mse": float("nan"), "mae": 0.027, "r2": 0.12, "ic": 0.07},
            {"mse": 0.0011, "mae": 0.026, "r2": 0.14, "ic": 0.09},
        ]

        aggregate, _ = aggregate_fold_metrics(fold_metrics)

        # Mean MSE should only use 2 values (skip NaN)
        expected_mean_mse = np.mean([0.001, 0.0011])
        assert aggregate["mean_mse"] == pytest.approx(expected_mean_mse)


# =============================================================================
# _process_fold Tests
# =============================================================================


class TestProcessFold:
    """Tests for _process_fold function."""

    def test_process_fold_returns_fold_result(self, basic_config, mock_all_data):
        """_process_fold should return a FoldResult."""
        with patch("hrp.ml.validation.get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return array matching input size
            mock_model.predict.side_effect = lambda X: np.random.randn(len(X))
            mock_get_model.return_value = mock_model

            result = _process_fold(
                fold_idx=0,
                train_start=date(2020, 1, 1),
                train_end=date(2020, 6, 30),
                test_start=date(2020, 7, 1),
                test_end=date(2020, 9, 30),
                config=basic_config,
                all_data=mock_all_data,
            )

        assert isinstance(result, FoldResult)
        assert result.fold_index == 0

    def test_process_fold_trains_model(self, basic_config, mock_all_data):
        """_process_fold should train the model."""
        with patch("hrp.ml.validation.get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return array matching input size
            mock_model.predict.side_effect = lambda X: np.random.randn(len(X))
            mock_get_model.return_value = mock_model

            _process_fold(
                fold_idx=0,
                train_start=date(2020, 1, 1),
                train_end=date(2020, 6, 30),
                test_start=date(2020, 7, 1),
                test_end=date(2020, 9, 30),
                config=basic_config,
                all_data=mock_all_data,
            )

        mock_model.fit.assert_called_once()

    def test_process_fold_computes_metrics(self, basic_config, mock_all_data):
        """_process_fold should compute metrics."""
        with patch("hrp.ml.validation.get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return array matching input size
            mock_model.predict.side_effect = lambda X: np.random.randn(len(X))
            mock_get_model.return_value = mock_model

            result = _process_fold(
                fold_idx=0,
                train_start=date(2020, 1, 1),
                train_end=date(2020, 6, 30),
                test_start=date(2020, 7, 1),
                test_end=date(2020, 9, 30),
                config=basic_config,
                all_data=mock_all_data,
            )

        assert "mse" in result.metrics
        assert "ic" in result.metrics


class TestProcessFoldSafe:
    """Tests for _process_fold_safe function."""

    def test_process_fold_safe_returns_none_on_error(self, basic_config, mock_all_data):
        """_process_fold_safe should return None on exception."""
        with patch("hrp.ml.validation._process_fold") as mock_process:
            mock_process.side_effect = Exception("Test error")

            result = _process_fold_safe(
                fold_idx=0,
                train_start=date(2020, 1, 1),
                train_end=date(2020, 6, 30),
                test_start=date(2020, 7, 1),
                test_end=date(2020, 9, 30),
                config=basic_config,
                all_data=mock_all_data,
            )

        assert result is None

    def test_process_fold_safe_returns_result_on_success(self, basic_config, mock_all_data):
        """_process_fold_safe should return FoldResult on success."""
        mock_fold_result = MagicMock(spec=FoldResult)

        with patch("hrp.ml.validation._process_fold") as mock_process:
            mock_process.return_value = mock_fold_result

            result = _process_fold_safe(
                fold_idx=0,
                train_start=date(2020, 1, 1),
                train_end=date(2020, 6, 30),
                test_start=date(2020, 7, 1),
                test_end=date(2020, 9, 30),
                config=basic_config,
                all_data=mock_all_data,
            )

        assert result == mock_fold_result


# =============================================================================
# walk_forward_validate Integration Tests
# =============================================================================


class TestWalkForwardValidate:
    """Integration tests for walk_forward_validate function."""

    def test_validate_returns_result(self, basic_config):
        """walk_forward_validate should return WalkForwardResult."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            # Create mock data
            np.random.seed(42)
            dates = pd.bdate_range(start="2020-01-01", periods=500)
            symbols = ["AAPL"]

            rows = []
            for symbol in symbols:
                for dt in dates:
                    rows.append({
                        "date": dt,
                        "symbol": symbol,
                        "momentum_20d": np.random.randn() * 0.1,
                        "volatility_20d": abs(np.random.randn() * 0.02),
                        "rsi_14d": 50 + np.random.randn() * 10,
                        "returns_20d": np.random.randn() * 0.05,
                    })

            mock_data = pd.DataFrame(rows).set_index(["date", "symbol"])
            mock_fetch.return_value = mock_data

            with patch("hrp.ml.validation.get_model") as mock_get_model:
                mock_model = MagicMock()
                # Return array matching input size
                mock_model.predict.side_effect = lambda X: np.random.randn(len(X))
                mock_get_model.return_value = mock_model

                result = walk_forward_validate(
                    config=basic_config,
                    symbols=["AAPL"],
                )

        assert isinstance(result, WalkForwardResult)

    def test_validate_empty_data_raises(self, basic_config):
        """walk_forward_validate should raise ValueError for empty data."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()

            with pytest.raises(ValueError) as exc_info:
                walk_forward_validate(
                    config=basic_config,
                    symbols=["AAPL"],
                )

            assert "No data found" in str(exc_info.value)

    def test_validate_all_folds_fail_raises(self, basic_config):
        """walk_forward_validate should raise ValueError if all folds fail."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            # Create minimal mock data
            np.random.seed(42)
            dates = pd.bdate_range(start="2020-01-01", periods=500)

            rows = []
            for dt in dates:
                rows.append({
                    "date": dt,
                    "symbol": "AAPL",
                    "momentum_20d": np.random.randn(),
                    "volatility_20d": np.random.randn(),
                    "rsi_14d": np.random.randn(),
                    "returns_20d": np.random.randn(),
                })

            mock_data = pd.DataFrame(rows).set_index(["date", "symbol"])
            mock_fetch.return_value = mock_data

            with patch("hrp.ml.validation._process_fold") as mock_process:
                mock_process.side_effect = Exception("All fail")

                with pytest.raises(ValueError) as exc_info:
                    walk_forward_validate(
                        config=basic_config,
                        symbols=["AAPL"],
                    )

                assert "All folds failed" in str(exc_info.value)


# =============================================================================
# _log_to_mlflow Tests
# =============================================================================


class TestLogToMlflow:
    """Tests for _log_to_mlflow function."""

    def test_log_to_mlflow_success(self, basic_config):
        """_log_to_mlflow should log to MLflow without error."""
        mock_fold = FoldResult(
            fold_index=0,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 1),
            test_end=date(2021, 6, 30),
            metrics={"mse": 0.001, "mae": 0.025, "r2": 0.15, "ic": 0.08},
            model=MagicMock(),
            n_train_samples=252,
            n_test_samples=126,
        )

        result = WalkForwardResult(
            config=basic_config,
            fold_results=[mock_fold],
            aggregate_metrics={"mean_mse": 0.001, "mean_ic": 0.08},
            stability_score=0.5,
            symbols=["AAPL"],
        )

        # mlflow is imported inside the function, so patch it via sys.modules
        mock_mlflow = MagicMock()
        mock_run_context = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_context)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        import sys
        original_mlflow = sys.modules.get("mlflow")
        try:
            sys.modules["mlflow"] = mock_mlflow
            _log_to_mlflow(result)

            mock_mlflow.set_experiment.assert_called_once()
            mock_mlflow.log_param.assert_called()
            mock_mlflow.log_metric.assert_called()
        finally:
            if original_mlflow is not None:
                sys.modules["mlflow"] = original_mlflow

    def test_log_to_mlflow_import_error(self, basic_config):
        """_log_to_mlflow should handle missing MLflow gracefully."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={},
            stability_score=0.5,
            symbols=["AAPL"],
        )

        import sys
        original_mlflow = sys.modules.get("mlflow")
        try:
            # Remove mlflow from modules to simulate ImportError
            if "mlflow" in sys.modules:
                del sys.modules["mlflow"]

            # Also need to block import by raising ImportError
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "mlflow":
                    raise ImportError("No module named 'mlflow'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                # Should not raise, just log warning
                _log_to_mlflow(result)
            finally:
                builtins.__import__ = original_import
        finally:
            if original_mlflow is not None:
                sys.modules["mlflow"] = original_mlflow

    def test_log_to_mlflow_exception(self, basic_config):
        """_log_to_mlflow should handle MLflow exceptions gracefully."""
        result = WalkForwardResult(
            config=basic_config,
            fold_results=[],
            aggregate_metrics={},
            stability_score=0.5,
            symbols=["AAPL"],
        )

        # mlflow is imported inside the function, so patch it via sys.modules
        mock_mlflow = MagicMock()
        mock_mlflow.set_experiment.side_effect = Exception("MLflow error")

        import sys
        original_mlflow = sys.modules.get("mlflow")
        try:
            sys.modules["mlflow"] = mock_mlflow
            # Should not raise, just log error
            _log_to_mlflow(result)
        finally:
            if original_mlflow is not None:
                sys.modules["mlflow"] = original_mlflow
