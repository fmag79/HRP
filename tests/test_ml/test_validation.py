"""Tests for walk-forward validation."""

from datetime import date

import pandas as pd
import pytest

from hrp.ml.validation import WalkForwardConfig, FoldResult, WalkForwardResult, generate_folds


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
