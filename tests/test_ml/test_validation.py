"""Tests for walk-forward validation."""

from datetime import date
from unittest.mock import patch

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
)


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.model_type == "ridge"
        assert config.n_folds == 5
        assert config.window_type == "expanding"
        assert config.min_train_periods == 252
        assert config.feature_selection is True
        assert config.max_features == 20
        assert config.hyperparameters == {}

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = WalkForwardConfig(
            model_type="random_forest",
            target="returns_5d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=10,
            window_type="rolling",
            min_train_periods=504,
            hyperparameters={"n_estimators": 100},
        )
        assert config.n_folds == 10
        assert config.window_type == "rolling"
        assert config.min_train_periods == 504
        assert config.hyperparameters == {"n_estimators": 100}

    def test_config_invalid_model_type(self):
        """Test config rejects invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            WalkForwardConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_window_type(self):
        """Test config rejects invalid window type."""
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                window_type="invalid",
            )

    def test_config_invalid_n_folds(self):
        """Test config rejects n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self):
        """Test creating FoldResult."""
        result = FoldResult(
            fold_index=0,
            train_start=date(2015, 1, 1),
            train_end=date(2018, 12, 31),
            test_start=date(2019, 1, 1),
            test_end=date(2019, 12, 31),
            metrics={"mse": 0.001, "mae": 0.02, "r2": 0.15, "ic": 0.05},
            model=None,  # Mock model
            n_train_samples=1000,
            n_test_samples=250,
        )
        assert result.fold_index == 0
        assert result.metrics["mse"] == 0.001
        assert result.n_train_samples == 1000


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_walk_forward_result_creation(self):
        """Test creating WalkForwardResult."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        fold_results = [
            FoldResult(
                fold_index=i,
                train_start=date(2015, 1, 1),
                train_end=date(2018, 12, 31),
                test_start=date(2019, 1, 1),
                test_end=date(2019, 12, 31),
                metrics={"mse": 0.001 + i * 0.0001},
                model=None,
                n_train_samples=1000,
                n_test_samples=250,
            )
            for i in range(3)
        ]
        result = WalkForwardResult(
            config=config,
            fold_results=fold_results,
            aggregate_metrics={"mean_mse": 0.0011, "std_mse": 0.0001},
            stability_score=0.09,
            symbols=["AAPL", "MSFT"],
        )
        assert len(result.fold_results) == 3
        assert result.stability_score == 0.09
        assert result.aggregate_metrics["mean_mse"] == 0.0011


class TestGenerateFolds:
    """Tests for generate_folds function."""

    @pytest.fixture
    def sample_dates(self):
        """Generate sample business dates."""
        return pd.date_range("2015-01-01", "2023-12-31", freq="B").date.tolist()

    def test_generate_folds_count(self, sample_dates):
        """Test that correct number of folds are generated."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)
        assert len(folds) == 5

    def test_generate_folds_no_overlap(self, sample_dates):
        """Test that test periods do not overlap."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)

        # Check test periods don't overlap
        for i in range(len(folds) - 1):
            _, _, _, test_end_i = folds[i]
            _, _, test_start_next, _ = folds[i + 1]
            assert test_end_i < test_start_next

    def test_generate_folds_expanding_window(self, sample_dates):
        """Test expanding window: train_start is always the same."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=3,
            window_type="expanding",
        )
        folds = generate_folds(config, sample_dates)

        # All folds should have the same train_start
        train_starts = [f[0] for f in folds]
        assert all(ts == train_starts[0] for ts in train_starts)

        # train_end should increase with each fold
        train_ends = [f[1] for f in folds]
        assert train_ends == sorted(train_ends)

    def test_generate_folds_rolling_window(self, sample_dates):
        """Test rolling window: train window size is constant."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=3,
            window_type="rolling",
        )
        folds = generate_folds(config, sample_dates)

        # Calculate training period lengths (in days)
        train_lengths = []
        for train_start, train_end, _, _ in folds:
            length = (train_end - train_start).days
            train_lengths.append(length)

        # All training periods should be approximately the same length
        # (allow 15% variance due to business days)
        avg_length = sum(train_lengths) / len(train_lengths)
        for length in train_lengths:
            assert abs(length - avg_length) / avg_length < 0.15

    def test_generate_folds_train_before_test(self, sample_dates):
        """Test that train_end < test_start for all folds."""
        config = WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=5,
        )
        folds = generate_folds(config, sample_dates)

        for train_start, train_end, test_start, test_end in folds:
            assert train_end < test_start


class TestComputeFoldMetrics:
    """Tests for compute_fold_metrics function."""

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert abs(metrics["ic"] - 1.0) < 0.0001  # Allow for floating point precision

    def test_compute_metrics_known_values(self):
        """Test metrics with known values."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])

        metrics = compute_fold_metrics(y_true, y_pred)

        # MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
        assert abs(metrics["mse"] - 0.25) < 0.001
        # MAE = mean(0.5 + 0.5 + 0.5) = 0.5
        assert abs(metrics["mae"] - 0.5) < 0.001

    def test_compute_metrics_returns_all_keys(self):
        """Test that all expected metrics are returned."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        metrics = compute_fold_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "ic" in metrics

    def test_compute_metrics_ic_range(self):
        """Test that IC is in valid range [-1, 1]."""
        y_true = pd.Series(np.random.randn(100))
        y_pred = np.random.randn(100)

        metrics = compute_fold_metrics(y_true, y_pred)

        assert -1.0 <= metrics["ic"] <= 1.0


class TestAggregateFoldMetrics:
    """Tests for aggregate_fold_metrics function."""

    def test_aggregate_metrics_mean_std(self):
        """Test aggregation computes mean and std."""
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.002, "mae": 0.03, "r2": 0.2, "ic": 0.06},
            {"mse": 0.003, "mae": 0.04, "r2": 0.3, "ic": 0.07},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        assert "mean_mse" in agg
        assert "std_mse" in agg
        assert "mean_mae" in agg
        assert "std_mae" in agg
        assert "mean_r2" in agg
        assert "std_r2" in agg
        assert "mean_ic" in agg
        assert "std_ic" in agg

        # Check mean_mse = mean([0.001, 0.002, 0.003]) = 0.002
        assert abs(agg["mean_mse"] - 0.002) < 0.0001

    def test_aggregate_metrics_stability_score(self):
        """Test stability score calculation."""
        # High variance case
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.010, "mae": 0.03, "r2": 0.2, "ic": 0.06},
            {"mse": 0.001, "mae": 0.04, "r2": 0.3, "ic": 0.07},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        # stability = std_mse / mean_mse
        expected_stability = agg["std_mse"] / agg["mean_mse"]
        assert abs(stability - expected_stability) < 0.0001

    def test_aggregate_metrics_stable_model(self):
        """Test stability score for consistent model."""
        # Low variance case - stable
        fold_metrics = [
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
            {"mse": 0.001, "mae": 0.02, "r2": 0.1, "ic": 0.05},
        ]

        agg, stability = aggregate_fold_metrics(fold_metrics)

        # Zero variance = zero stability score
        assert stability == 0.0


class TestWalkForwardValidate:
    """Tests for walk_forward_validate function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample WalkForwardConfig."""
        return WalkForwardConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            feature_selection=False,
        )

    @pytest.fixture
    def mock_features_df(self):
        """Create mock features DataFrame."""
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        np.random.seed(42)
        n = len(index)

        # Create features with weak signal
        momentum = np.random.randn(n) * 0.1
        volatility = np.abs(np.random.randn(n)) * 0.2
        target = 0.1 * momentum + np.random.randn(n) * 0.05

        return pd.DataFrame(
            {
                "momentum_20d": momentum,
                "volatility_20d": volatility,
                "returns_20d": target,
            },
            index=index,
        )

    def test_walk_forward_validate_returns_result(self, sample_config, mock_features_df):
        """Test walk_forward_validate returns WalkForwardResult."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 3
        assert result.config == sample_config

    def test_walk_forward_validate_fold_metrics(self, sample_config, mock_features_df):
        """Test that each fold has valid metrics."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        for fold in result.fold_results:
            assert "mse" in fold.metrics
            assert "ic" in fold.metrics
            assert fold.n_train_samples > 0
            assert fold.n_test_samples > 0

    def test_walk_forward_validate_aggregate_metrics(self, sample_config, mock_features_df):
        """Test aggregate metrics are computed."""
        with patch("hrp.ml.validation._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            result = walk_forward_validate(
                config=sample_config,
                symbols=["AAPL", "MSFT"],
            )

        assert "mean_mse" in result.aggregate_metrics
        assert "std_mse" in result.aggregate_metrics
        assert result.stability_score >= 0

    def test_walk_forward_validate_expanding_vs_rolling(self, mock_features_df):
        """Test both window types produce results."""
        for window_type in ["expanding", "rolling"]:
            config = WalkForwardConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d", "volatility_20d"],
                start_date=date(2015, 1, 1),
                end_date=date(2020, 12, 31),
                n_folds=3,
                window_type=window_type,
                feature_selection=False,
            )

            with patch("hrp.ml.validation._fetch_features") as mock_fetch:
                mock_fetch.return_value = mock_features_df

                result = walk_forward_validate(config, symbols=["AAPL", "MSFT"])

            assert len(result.fold_results) == 3


class TestModuleExports:
    """Test that validation module is properly exported."""

    def test_import_from_ml_module(self):
        """Test importing from hrp.ml."""
        from hrp.ml import (
            WalkForwardConfig,
            WalkForwardResult,
            FoldResult,
            walk_forward_validate,
        )

        assert WalkForwardConfig is not None
        assert WalkForwardResult is not None
        assert FoldResult is not None
        assert walk_forward_validate is not None
