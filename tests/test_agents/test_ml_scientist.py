"""
Tests for ML Scientist research agent.

Tests cover:
- Agent initialization
- Feature extraction from hypotheses
- Feature combination generation
- Model scoring
- Status determination
- Trial budget management
- Walk-forward validation integration
- Hypothesis updates
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.research_agents import (
    MLScientist,
    ModelExperimentResult,
    MLScientistReport,
)


class TestMLScientistInit:
    """Tests for ML Scientist initialization."""

    def test_default_initialization(self):
        """MLScientist initializes with default model types."""
        agent = MLScientist()
        assert agent.model_types == ["ridge", "lasso", "lightgbm"]
        assert agent.target == "returns_20d"
        assert agent.n_folds == 5
        assert agent.window_type == "expanding"

    def test_custom_model_types(self):
        """MLScientist accepts custom model types."""
        agent = MLScientist(model_types=["ridge", "xgboost"])
        assert agent.model_types == ["ridge", "xgboost"]

    def test_hypothesis_filter(self):
        """MLScientist can filter to specific hypothesis IDs."""
        agent = MLScientist(hypothesis_ids=["HYP-2026-001", "HYP-2026-002"])
        assert agent.hypothesis_ids == ["HYP-2026-001", "HYP-2026-002"]

    def test_custom_target(self):
        """MLScientist accepts custom target variable."""
        agent = MLScientist(target="returns_5d")
        assert agent.target == "returns_5d"

    def test_custom_date_range(self):
        """MLScientist accepts custom date range."""
        agent = MLScientist(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert agent.start_date == date(2020, 1, 1)
        assert agent.end_date == date(2023, 12, 31)

    def test_custom_trial_limit(self):
        """MLScientist accepts custom trial limit."""
        agent = MLScientist(max_trials_per_hypothesis=100)
        assert agent.max_trials == 100

    def test_skip_hyperparameter_search_flag(self):
        """MLScientist accepts skip_hyperparameter_search flag."""
        agent = MLScientist(skip_hyperparameter_search=True)
        assert agent.skip_hyperparameter_search is True

    def test_actor_identity(self):
        """MLScientist has correct actor identity."""
        agent = MLScientist()
        assert agent.actor == "agent:ml-scientist"
        assert agent.ACTOR == "agent:ml-scientist"

    def test_job_id(self):
        """MLScientist has correct job ID."""
        agent = MLScientist()
        assert agent.job_id == "ml_scientist_training"


class TestFeatureExtraction:
    """Tests for feature extraction from hypotheses."""

    def test_extract_features_from_metadata(self):
        """Features extracted from hypothesis metadata."""
        agent = MLScientist()
        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Some thesis about momentum",
            "metadata": {"features": ["momentum_20d", "volatility_60d"]},
        }
        features = agent._extract_features_from_hypothesis(hypothesis)
        assert features == ["momentum_20d", "volatility_60d"]

    def test_extract_features_from_thesis(self):
        """Features parsed from hypothesis thesis text."""
        agent = MLScientist()
        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "The momentum_20d feature is positively correlated with returns",
            "metadata": {},
        }
        features = agent._extract_features_from_hypothesis(hypothesis)
        assert "momentum_20d" in features

    def test_extract_multiple_features_from_thesis(self):
        """Multiple features parsed from thesis text."""
        agent = MLScientist()
        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Testing momentum_20d and volatility_60d as predictors",
            "metadata": {},
        }
        features = agent._extract_features_from_hypothesis(hypothesis)
        assert "momentum_20d" in features
        assert "volatility_60d" in features

    def test_fallback_to_default(self):
        """Falls back to momentum_20d if no features found."""
        agent = MLScientist()
        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Some vague thesis without feature names",
            "metadata": {},
        }
        features = agent._extract_features_from_hypothesis(hypothesis)
        # Default fallback is now volatility_60d (safe feature, not leaky like momentum_20d)
        assert features == ["volatility_60d"]

    def test_empty_metadata_uses_thesis(self):
        """If metadata has no features, uses thesis parsing."""
        agent = MLScientist()
        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Testing rsi_14d as a mean-reversion signal",
            "metadata": {"other_key": "value"},
        }
        features = agent._extract_features_from_hypothesis(hypothesis)
        assert "rsi_14d" in features


class TestFeatureCombinations:
    """Tests for feature combination generation."""

    def test_generates_base_combination(self):
        """Base features included as first combination (after leakage filtering)."""
        agent = MLScientist()
        # Use a safe feature - momentum_20d is filtered out as leaky
        combinations = agent._generate_feature_combinations(["volatility_60d"])
        assert ["volatility_60d"] in combinations
        assert combinations[0] == ["volatility_60d"]

    def test_adds_complementary_features(self):
        """Complementary features added to combinations."""
        agent = MLScientist()
        # Use a safe feature with defined complements
        combinations = agent._generate_feature_combinations(["volatility_60d"])
        # volatility_60d has safe complements: atr_14d, volume_ratio, adx_14d, bb_width_20d
        combo_features = [set(c) for c in combinations]
        assert {"volatility_60d", "atr_14d"} in combo_features or \
               {"volatility_60d", "volume_ratio"} in combo_features

    def test_respects_max_features(self):
        """Combinations limited to MAX_FEATURES_PER_MODEL."""
        agent = MLScientist()
        combinations = agent._generate_feature_combinations(["momentum_20d"])
        for combo in combinations:
            assert len(combo) <= agent.MAX_FEATURES_PER_MODEL

    def test_limits_total_combinations(self):
        """Total combinations limited to MAX_FEATURE_COMBINATIONS."""
        agent = MLScientist()
        combinations = agent._generate_feature_combinations(["momentum_20d"])
        assert len(combinations) <= agent.MAX_FEATURE_COMBINATIONS

    def test_deduplicates_combinations(self):
        """Duplicate combinations are removed."""
        agent = MLScientist()
        combinations = agent._generate_feature_combinations(["momentum_20d"])
        combo_sets = [frozenset(c) for c in combinations]
        assert len(combo_sets) == len(set(combo_sets))

    def test_unknown_feature_returns_base_only(self):
        """Unknown feature without complements returns base only."""
        agent = MLScientist()
        combinations = agent._generate_feature_combinations(["unknown_feature"])
        assert ["unknown_feature"] in combinations


class TestModelScoring:
    """Tests for model scoring calculations."""

    def test_higher_ic_scores_higher(self):
        """Higher IC produces higher score."""
        agent = MLScientist()
        result_high = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.05,
            ic_std=0.02,
            stability_score=0.8,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.05}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        result_low = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.02,
            ic_std=0.02,
            stability_score=0.8,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.02}] * 5,
            mlflow_run_id="run2",
            training_time_seconds=10.0,
        )
        score_high = agent._calculate_model_score(result_high)
        score_low = agent._calculate_model_score(result_low)
        assert score_high > score_low

    def test_lower_stability_scores_higher(self):
        """Lower stability score produces higher overall score."""
        agent = MLScientist()
        result_stable = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.04,
            ic_std=0.02,
            stability_score=0.5,  # More stable
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.04}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        result_unstable = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.04,
            ic_std=0.02,
            stability_score=1.5,  # Less stable
            is_stable=False,
            n_folds=5,
            fold_results=[{"ic": 0.04}] * 5,
            mlflow_run_id="run2",
            training_time_seconds=10.0,
        )
        score_stable = agent._calculate_model_score(result_stable)
        score_unstable = agent._calculate_model_score(result_unstable)
        assert score_stable > score_unstable

    def test_consistency_bonus(self):
        """All positive folds get consistency bonus."""
        agent = MLScientist()
        result_consistent = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.04,
            ic_std=0.02,
            stability_score=0.8,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.04}, {"ic": 0.03}, {"ic": 0.05}, {"ic": 0.04}, {"ic": 0.04}],
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        result_inconsistent = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.04,
            ic_std=0.02,
            stability_score=0.8,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.04}, {"ic": -0.01}, {"ic": 0.05}, {"ic": 0.04}, {"ic": 0.04}],
            mlflow_run_id="run2",
            training_time_seconds=10.0,
        )
        score_consistent = agent._calculate_model_score(result_consistent)
        score_inconsistent = agent._calculate_model_score(result_inconsistent)
        assert score_consistent > score_inconsistent


class TestStatusDetermination:
    """Tests for hypothesis status determination."""

    def test_validated_status(self):
        """High IC + low stability = validated."""
        agent = MLScientist()
        result = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.04,  # Above IC_THRESHOLD_VALIDATED (0.03)
            ic_std=0.02,
            stability_score=0.8,  # Below STABILITY_THRESHOLD_VALIDATED (1.0)
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.04}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        status = agent._determine_status(result)
        assert status == "validated"

    def test_rejected_status_low_ic(self):
        """Low IC = rejected."""
        agent = MLScientist()
        result = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.01,  # Below IC_THRESHOLD_PROMISING (0.02)
            ic_std=0.02,
            stability_score=0.8,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.01}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        status = agent._determine_status(result)
        assert status == "rejected"

    def test_rejected_status_high_instability(self):
        """High instability = rejected."""
        agent = MLScientist()
        result = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.03,
            ic_std=0.02,
            stability_score=2.5,  # Above STABILITY_THRESHOLD_PROMISING (1.5)
            is_stable=False,
            n_folds=5,
            fold_results=[{"ic": 0.03}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        status = agent._determine_status(result)
        assert status == "rejected"

    def test_promising_stays_testing(self):
        """Moderate results keep hypothesis in testing."""
        agent = MLScientist()
        result = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d"],
            model_params={},
            mean_ic=0.025,  # Between PROMISING (0.02) and VALIDATED (0.03)
            ic_std=0.02,
            stability_score=1.2,  # Between VALIDATED (1.0) and PROMISING (1.5)
            is_stable=False,
            n_folds=5,
            fold_results=[{"ic": 0.025}] * 5,
            mlflow_run_id="run1",
            training_time_seconds=10.0,
        )
        status = agent._determine_status(result)
        assert status == "testing"


class TestHyperparameterGrid:
    """Tests for hyperparameter grid generation."""

    def test_ridge_param_grid(self):
        """Ridge has alpha hyperparameter grid."""
        agent = MLScientist()
        grid = agent._get_param_grid("ridge")
        assert len(grid) > 0
        # All entries should have alpha
        for params in grid:
            if params:  # Non-empty params
                assert "alpha" in params

    def test_lightgbm_param_grid(self):
        """LightGBM has num_leaves, learning_rate grid."""
        agent = MLScientist()
        grid = agent._get_param_grid("lightgbm")
        assert len(grid) > 0

    def test_unknown_model_returns_empty(self):
        """Unknown model returns empty params only."""
        agent = MLScientist()
        grid = agent._get_param_grid("unknown_model")
        assert grid == [{}]

    def test_grid_limited_to_max(self):
        """Grid is limited to prevent explosion."""
        agent = MLScientist()
        grid = agent._get_param_grid("lightgbm")
        assert len(grid) <= 10  # Should be limited


class TestModelExperimentResult:
    """Tests for ModelExperimentResult dataclass."""

    def test_dataclass_creation(self):
        """ModelExperimentResult can be created with all fields."""
        result = ModelExperimentResult(
            hypothesis_id="HYP-001",
            model_type="ridge",
            features=["momentum_20d", "volatility_60d"],
            model_params={"alpha": 1.0},
            mean_ic=0.042,
            ic_std=0.015,
            stability_score=0.71,
            is_stable=True,
            n_folds=5,
            fold_results=[{"ic": 0.04, "mse": 0.001}],
            mlflow_run_id="abc123",
            training_time_seconds=45.5,
        )
        assert result.hypothesis_id == "HYP-001"
        assert result.model_type == "ridge"
        assert result.features == ["momentum_20d", "volatility_60d"]
        assert result.mean_ic == 0.042
        assert result.is_stable is True


class TestMLScientistReport:
    """Tests for MLScientistReport dataclass."""

    def test_dataclass_creation(self):
        """MLScientistReport can be created with all fields."""
        report = MLScientistReport(
            run_date=date(2026, 1, 26),
            hypotheses_processed=3,
            hypotheses_validated=2,
            hypotheses_rejected=1,
            total_trials=45,
            total_training_time_seconds=1234.5,
            best_models=[],
            mlflow_experiment_id="exp123",
        )
        assert report.hypotheses_processed == 3
        assert report.hypotheses_validated == 2


class TestMLScientistIntegration:
    """Integration tests for ML Scientist."""

    @patch("hrp.ml.walk_forward_validate")
    @patch("hrp.risk.overfitting.HyperparameterTrialCounter")
    def test_process_hypothesis_runs_experiments(
        self, mock_counter_class, mock_wf_validate
    ):
        """Processing a hypothesis runs walk-forward experiments."""
        # Setup mocks
        mock_counter = MagicMock()
        mock_counter.remaining_trials = 50
        mock_counter.can_try.return_value = True
        mock_counter_class.return_value = mock_counter

        mock_result = MagicMock()
        mock_result.mean_ic = 0.04
        mock_result.aggregate_metrics = {"std_ic": 0.02}
        mock_result.stability_score = 0.8
        mock_result.is_stable = True
        mock_result.fold_results = [MagicMock(metrics={"ic": 0.04})] * 5
        mock_result.mlflow_run_id = None
        mock_wf_validate.return_value = mock_result

        agent = MLScientist(
            hypothesis_ids=["HYP-2026-001"],
            model_types=["ridge"],
            skip_hyperparameter_search=True,
        )

        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Testing momentum_20d signal",
            "metadata": {"features": ["momentum_20d"]},
        }

        results = agent._process_hypothesis(hypothesis, ["AAPL", "MSFT"])

        # Should have run at least one experiment
        assert len(results) > 0
        mock_wf_validate.assert_called()

    @patch("hrp.ml.walk_forward_validate")
    @patch("hrp.risk.overfitting.HyperparameterTrialCounter")
    def test_trial_budget_enforced(self, mock_counter_class, mock_wf_validate):
        """Trial budget is enforced during processing."""
        # Setup mock counter to run out of trials
        mock_counter = MagicMock()
        mock_counter.remaining_trials = 0
        mock_counter.can_try.return_value = False
        mock_counter_class.return_value = mock_counter

        agent = MLScientist(
            hypothesis_ids=["HYP-2026-001"],
            model_types=["ridge", "lasso", "lightgbm"],
        )

        hypothesis = {
            "id": "HYP-2026-001",
            "thesis": "Testing momentum_20d",
            "metadata": {"features": ["momentum_20d"]},
        }

        results = agent._process_hypothesis(hypothesis, ["AAPL"])

        # Should return empty when no trials remaining
        assert len(results) == 0
        mock_wf_validate.assert_not_called()


class TestMLScientistConstants:
    """Tests for ML Scientist constants."""

    def test_ic_thresholds(self):
        """IC thresholds are properly defined."""
        assert MLScientist.IC_THRESHOLD_VALIDATED == 0.03
        assert MLScientist.IC_THRESHOLD_PROMISING == 0.02

    def test_stability_thresholds(self):
        """Stability thresholds are properly defined."""
        assert MLScientist.STABILITY_THRESHOLD_VALIDATED == 1.0
        assert MLScientist.STABILITY_THRESHOLD_PROMISING == 1.5

    def test_trial_limits(self):
        """Trial limits are properly defined."""
        assert MLScientist.MAX_TRIALS_PER_HYPOTHESIS == 50
        assert MLScientist.MAX_FEATURE_COMBINATIONS == 10
        assert MLScientist.MAX_FEATURES_PER_MODEL == 3

    def test_default_model_types(self):
        """Default model types are defined."""
        assert MLScientist.DEFAULT_MODEL_TYPES == ["ridge", "lasso", "lightgbm"]

    def test_hyperparameter_grids_defined(self):
        """Hyperparameter grids are defined for common models."""
        assert "ridge" in MLScientist.HYPERPARAMETER_GRIDS
        assert "lasso" in MLScientist.HYPERPARAMETER_GRIDS
        assert "lightgbm" in MLScientist.HYPERPARAMETER_GRIDS

    def test_complementary_features_defined(self):
        """Complementary features are defined for safe (non-leaky) features."""
        # Safe features should have complements defined
        assert "volatility_60d" in MLScientist.COMPLEMENTARY_FEATURES
        assert "atr_14d" in MLScientist.COMPLEMENTARY_FEATURES
        assert "volume_ratio" in MLScientist.COMPLEMENTARY_FEATURES
        # Leaky features like momentum_20d should NOT be in complements
        assert "momentum_20d" not in MLScientist.COMPLEMENTARY_FEATURES

    def test_leaky_features_defined(self):
        """Leaky features are defined for target-based filtering."""
        assert "returns_20d" in MLScientist.LEAKY_FEATURES_BY_TARGET
        leaky = MLScientist.LEAKY_FEATURES_BY_TARGET["returns_20d"]
        # These have high correlation with returns_20d
        assert "momentum_20d" in leaky  # corr = 1.0
        assert "rsi_14d" in leaky  # corr = 0.71
        assert "price_to_sma_20d" in leaky  # corr = 0.84
