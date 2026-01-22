"""Tests for walk-forward validation."""

from datetime import date

import pytest

from hrp.ml.validation import WalkForwardConfig, FoldResult, WalkForwardResult


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
