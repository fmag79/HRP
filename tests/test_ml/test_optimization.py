"""Tests for cross-validated optimization."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

import optuna

from hrp.ml.optimization import (
    OptimizationConfig,
    OptimizationResult,
    cross_validated_optimize,
    _evaluate_with_pruning,
    _get_sampler,
    SCORING_METRICS,
)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_result_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            best_params={"alpha": 1.0},
            best_score=0.05,
            cv_results=pd.DataFrame({"mean_score": [0.03, 0.05, 0.04]}),
            fold_results=[],
            all_trials=[{"params": {"alpha": 1.0}, "mean_score": 0.05}],
            hypothesis_id="HYP-2025-001",
            n_trials_completed=3,
            early_stopped=False,
            execution_time_seconds=10.5,
        )
        assert result.best_params == {"alpha": 1.0}
        assert result.best_score == 0.05
        assert result.n_trials_completed == 3
        assert result.early_stopped is False
        assert result.execution_time_seconds == 10.5


class TestScoringMetrics:
    """Tests for scoring metric definitions."""

    def test_all_metrics_defined(self):
        """Test all expected metrics are defined."""
        expected = ["ic", "r2", "mse", "mae", "sharpe"]
        for metric in expected:
            assert metric in SCORING_METRICS

    def test_higher_is_better_correct(self):
        """Test higher_is_better flags are correct."""
        assert SCORING_METRICS["ic"] is True  # Higher IC is better
        assert SCORING_METRICS["r2"] is True  # Higher R2 is better
        assert SCORING_METRICS["mse"] is False  # Lower MSE is better
        assert SCORING_METRICS["mae"] is False  # Lower MAE is better


class TestModuleExports:
    """Test that optimization module is properly exported."""

    def test_import_from_ml_module(self):
        """Test importing from hrp.ml."""
        from hrp.ml.optimization import (
            OptimizationConfig,
            OptimizationResult,
            cross_validated_optimize,
        )
        from optuna.distributions import FloatDistribution

        assert OptimizationConfig is not None
        assert OptimizationResult is not None
        assert cross_validated_optimize is not None
        assert FloatDistribution is not None


class TestOptimizationConfigNew:
    """Tests for new Optuna-based OptimizationConfig."""

    def test_config_with_param_space(self):
        """Test creating config with param_space distributions."""
        config = OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            param_space={
                "alpha": FloatDistribution(0.01, 100.0, log=True),
            },
            start_date=date(2015, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert "alpha" in config.param_space
        assert config.sampler == "tpe"
        assert config.n_trials == 50
        assert config.enable_pruning is True

    def test_config_rejects_invalid_sampler(self):
        """Test config rejects invalid sampler."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                sampler="invalid",
            )

    def test_config_rejects_empty_param_space(self):
        """Test config rejects empty param_space."""
        with pytest.raises(ValueError, match="param_space cannot be empty"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_model_type(self):
        """Test config rejects invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            OptimizationConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_window_type(self):
        """Test config rejects invalid window type."""
        with pytest.raises(ValueError, match="window_type must be"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                window_type="invalid",
            )

    def test_config_invalid_n_folds(self):
        """Test config rejects n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )

    def test_config_invalid_scoring_metric(self):
        """Test config rejects invalid scoring metric."""
        with pytest.raises(ValueError, match="Unsupported scoring_metric"):
            OptimizationConfig(
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                param_space={"alpha": FloatDistribution(0.1, 10.0)},
                start_date=date(2015, 1, 1),
                end_date=date(2023, 12, 31),
                scoring_metric="invalid_metric",
            )


class TestGetSampler:
    """Tests for _get_sampler function."""

    def test_tpe_sampler(self):
        """Test TPE sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("tpe", param_space)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_random_sampler(self):
        """Test random sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("random", param_space)
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_grid_sampler_with_categorical(self):
        """Test grid sampler with categorical distribution."""
        param_space = {"solver": CategoricalDistribution(["lbfgs", "saga"])}
        sampler = _get_sampler("grid", param_space)
        assert isinstance(sampler, optuna.samplers.GridSampler)

    def test_grid_sampler_requires_step_for_float(self):
        """Test grid sampler requires step for float distributions."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}  # No step
        with pytest.raises(ValueError, match="requires step"):
            _get_sampler("grid", param_space)

    def test_cmaes_sampler(self):
        """Test CMA-ES sampler creation."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        sampler = _get_sampler("cmaes", param_space)
        assert isinstance(sampler, optuna.samplers.CmaEsSampler)

    def test_invalid_sampler_raises(self):
        """Test invalid sampler raises ValueError."""
        param_space = {"alpha": FloatDistribution(0.1, 10.0)}
        with pytest.raises(ValueError, match="Unknown sampler"):
            _get_sampler("invalid", param_space)


class TestEvaluateWithPruning:
    """Tests for _evaluate_with_pruning function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample OptimizationConfig for pruning tests."""
        return OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            param_space={"alpha": FloatDistribution(0.1, 10.0)},
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            feature_selection=False,
            enable_pruning=True,
            early_stop_decay_threshold=0.5,
        )

    @pytest.fixture
    def mock_features_df(self):
        """Create mock features DataFrame."""
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        np.random.seed(42)
        n = len(index)

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

    @pytest.fixture
    def sample_folds(self):
        """Create sample folds for testing."""
        return [
            (date(2015, 1, 1), date(2016, 12, 31), date(2017, 1, 1), date(2017, 12, 31)),
            (date(2015, 1, 1), date(2017, 12, 31), date(2018, 1, 1), date(2018, 12, 31)),
            (date(2015, 1, 1), date(2018, 12, 31), date(2019, 1, 1), date(2019, 12, 31)),
        ]

    def test_reports_intermediate_values(
        self, sample_config, mock_features_df, sample_folds
    ):
        """Verify trial.report is called once per fold."""
        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False

        params = {"alpha": 1.0}

        mean_score, fold_results = _evaluate_with_pruning(
            trial=mock_trial,
            params=params,
            config=sample_config,
            all_data=mock_features_df,
            folds=sample_folds,
        )

        # Verify trial.report was called once per fold
        assert mock_trial.report.call_count == len(sample_folds)

        # Verify call arguments: (score, fold_index)
        for call_idx, call in enumerate(mock_trial.report.call_args_list):
            args, kwargs = call
            assert len(args) == 2
            assert args[1] == call_idx  # fold_idx

    def test_prunes_on_should_prune(
        self, sample_config, mock_features_df, sample_folds
    ):
        """Verify TrialPruned raised when should_prune returns True."""
        mock_trial = MagicMock()
        # Return False for first fold, True for second fold
        mock_trial.should_prune.side_effect = [False, True]

        params = {"alpha": 1.0}

        with pytest.raises(optuna.TrialPruned):
            _evaluate_with_pruning(
                trial=mock_trial,
                params=params,
                config=sample_config,
                all_data=mock_features_df,
                folds=sample_folds,
            )

        # Should have pruned after second fold
        assert mock_trial.report.call_count == 2
        assert mock_trial.should_prune.call_count == 2


class TestCrossValidatedOptimizeNew:
    """Tests for Optuna-based cross_validated_optimize."""

    @pytest.fixture
    def sample_config(self):
        return OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            param_space={"alpha": FloatDistribution(0.1, 10.0, log=True)},
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            sampler="tpe",
            n_trials=5,
            feature_selection=False,
            enable_pruning=False,
        )

    @pytest.fixture
    def mock_features_df(self):
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        np.random.seed(42)
        n = len(index)
        return pd.DataFrame(
            {
                "momentum_20d": np.random.randn(n) * 0.1,
                "volatility_20d": np.abs(np.random.randn(n)) * 0.2,
                "returns_20d": 0.1 * np.random.randn(n) * 0.1 + np.random.randn(n) * 0.05,
            },
            index=index,
        )

    def test_returns_optimization_result(self, sample_config, mock_features_df):
        with patch("hrp.ml.optimization._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df
            result = cross_validated_optimize(
                config=sample_config, symbols=["AAPL", "MSFT"], log_to_mlflow=False
            )
        assert isinstance(result, OptimizationResult)
        assert "alpha" in result.best_params
        assert result.n_trials_completed > 0

    def test_respects_trial_counter(self, mock_features_df):
        config = OptimizationConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            param_space={"alpha": FloatDistribution(0.1, 10.0)},
            start_date=date(2015, 1, 1),
            end_date=date(2020, 12, 31),
            n_folds=3,
            hypothesis_id="HYP-TEST-001",
            n_trials=10,
            feature_selection=False,
            enable_pruning=False,
        )
        with patch("hrp.ml.optimization._fetch_features") as mock_fetch, patch(
            "hrp.ml.optimization.HyperparameterTrialCounter"
        ) as mock_counter:
            mock_fetch.return_value = mock_features_df
            mock_instance = MagicMock()
            mock_instance.remaining_trials = 3
            mock_instance.can_try.return_value = True
            mock_counter.return_value = mock_instance
            cross_validated_optimize(config=config, symbols=["AAPL"], log_to_mlflow=False)
        mock_counter.assert_called_once_with(hypothesis_id="HYP-TEST-001", max_trials=10)
