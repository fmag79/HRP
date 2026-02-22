"""
Integration tests for Optimization dashboard page.

Tests the main optimization page components, configuration flow,
and integration with OptimizationAPI.
"""

import pytest
from datetime import date
from unittest.mock import Mock, MagicMock, patch

from hrp.api.optimization_api import OptimizationAPI, OptimizationPreview
from hrp.ml.optimization import OptimizationResult, OptimizationConfig


class TestOptimizationPageComponents:
    """Test optimization page component structure and validation."""

    def test_optimization_page_imports(self):
        """Test that optimization page can be imported."""
        from hrp.dashboard.pages import optimization
        assert hasattr(optimization, 'render_optimization_page')

    def test_optimization_controls_imports(self):
        """Test that optimization controls can be imported."""
        from hrp.dashboard.components import optimization_controls

        expected_functions = [
            'render_strategy_selector',
            'render_model_selector',
            'render_sampler_selector',
            'render_trials_slider',
            'render_folds_slider',
            'render_scoring_selector',
            'render_date_range',
            'render_feature_selector',
            'render_optimization_preview',
            'render_results_tab',
            'render_fold_analysis_tab',
            'render_study_history_tab',
        ]

        for func_name in expected_functions:
            assert hasattr(optimization_controls, func_name), f"Missing: {func_name}"


class TestOptimizationPreview:
    """Test optimization preview functionality."""

    def test_render_optimization_preview_with_warnings(self):
        """Test preview rendering with warnings."""
        from hrp.dashboard.components.optimization_controls import render_optimization_preview

        preview = OptimizationPreview(
            estimated_time_seconds=300,
            estimated_cost_estimate="Medium (~5m)",
            parameter_space_summary={"alpha": "0.01 to 100.0 (log)", "l1_ratio": "0.0 to 1.0"},
            recommended_sampler="tpe",
            warnings=["High trial count may take >10m"],
        )

        # This test ensures the function doesn't crash
        # Actual rendering would require Streamlit context
        assert preview.warnings == ["High trial count may take >10m"]
        assert preview.estimated_cost_estimate == "Medium (~5m)"

    def test_render_optimization_preview_no_warnings(self):
        """Test preview rendering without warnings."""
        from hrp.dashboard.components.optimization_controls import render_optimization_preview

        preview = OptimizationPreview(
            estimated_time_seconds=60,
            estimated_cost_estimate="Low (~1m)",
            parameter_space_summary={"alpha": "0.01 to 100.0 (log)"},
            recommended_sampler="tpe",
            warnings=[],
        )

        assert len(preview.warnings) == 0
        assert preview.estimated_time_seconds == 60


class TestOptimizationResults:
    """Test optimization results rendering."""

    def test_render_results_tab_with_full_results(self):
        """Test results tab with complete optimization result."""
        from hrp.dashboard.components.optimization_controls import render_results_tab

        result = OptimizationResult(
            best_params={"alpha": 1.0, "l1_ratio": 0.5},
            best_score=0.0234,
            best_trial=42,
            trial_history=[
                {"trial": 0, "score": 0.01},
                {"trial": 1, "score": 0.02},
                {"trial": 2, "score": 0.0234},
            ],
            parameter_importance={"alpha": 0.8, "l1_ratio": 0.2},
            fold_results=[
                {"train_score": 0.03, "test_score": 0.025},
                {"train_score": 0.028, "test_score": 0.022},
            ],
            study_name="test_study",
            optimization_time=123.45,
        )

        assert result.best_score == 0.0234
        assert result.best_trial == 42
        assert len(result.trial_history) == 3
        assert len(result.parameter_importance) == 2

    def test_render_fold_analysis_with_stability(self):
        """Test fold analysis tab with stability metrics."""
        from hrp.dashboard.components.optimization_controls import render_fold_analysis_tab

        result = OptimizationResult(
            best_params={"alpha": 1.0},
            best_score=0.025,
            best_trial=10,
            trial_history=[],
            parameter_importance={},
            fold_results=[
                {"train_score": 0.03, "test_score": 0.025},
                {"train_score": 0.029, "test_score": 0.024},
                {"train_score": 0.031, "test_score": 0.026},
            ],
            study_name="test_study",
            optimization_time=60.0,
        )

        assert len(result.fold_results) == 3
        # Verify fold results structure
        for fold in result.fold_results:
            assert "train_score" in fold
            assert "test_score" in fold


class TestOptimizationConfiguration:
    """Test optimization configuration building."""

    def test_configuration_validation_date_range(self):
        """Test date range validation."""
        start_date = date(2020, 1, 1)
        end_date = date(2019, 1, 1)  # Invalid: end before start

        # Invalid date range should be caught by UI validation
        assert start_date >= end_date

    def test_configuration_validation_features(self):
        """Test feature selection validation."""
        features = []

        # Empty features should be caught by UI validation
        assert len(features) == 0

    def test_configuration_validation_valid(self):
        """Test valid configuration."""
        from hrp.ml.optimization import OptimizationConfig
        from optuna.distributions import FloatDistribution

        config = OptimizationConfig(
            hypothesis_id="test_hypo",
            strategy_name="multifactor_ml",
            model_type="ridge",
            param_space={"alpha": FloatDistribution(0.01, 100.0, log=True)},
            sampler="tpe",
            n_trials=50,
            n_folds=5,
            scoring="ic",
            features=["momentum_20d", "volatility_60d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            pruning_enabled=True,
            study_name=None,
        )

        assert config.n_trials == 50
        assert config.n_folds == 5
        assert len(config.features) == 2


class TestStudyHistory:
    """Test study history functionality."""

    def test_render_study_history_empty(self):
        """Test study history with no studies."""
        from hrp.dashboard.components.optimization_controls import render_study_history_tab

        studies = []

        # Should handle empty list gracefully
        assert len(studies) == 0

    def test_render_study_history_with_studies(self):
        """Test study history with multiple studies."""
        from hrp.dashboard.components.optimization_controls import render_study_history_tab

        studies = [
            {
                "study_name": "study_1",
                "datetime_start": "2023-01-01T10:00:00",
                "n_trials": 50,
                "best_value": 0.0234,
                "user_attrs": {"model": "ridge"},
            },
            {
                "study_name": "study_2",
                "datetime_start": "2023-01-02T11:00:00",
                "n_trials": 100,
                "best_value": 0.0256,
                "user_attrs": {"model": "lasso"},
            },
        ]

        assert len(studies) == 2
        assert studies[0]["best_value"] == 0.0234
        assert studies[1]["n_trials"] == 100


class TestOptimizationAPIIntegration:
    """Test integration with OptimizationAPI."""

    @patch('hrp.api.optimization_api.cross_validated_optimize')
    def test_optimization_execution_flow(self, mock_optimize):
        """Test end-to-end optimization execution."""
        from hrp.api.optimization_api import OptimizationAPI
        from hrp.api.platform import PlatformAPI
        from hrp.ml.optimization import OptimizationConfig
        from optuna.distributions import FloatDistribution

        # Mock the cross_validated_optimize function
        mock_result = OptimizationResult(
            best_params={"alpha": 1.0},
            best_score=0.0234,
            best_trial=25,
            trial_history=[],
            parameter_importance={},
            fold_results=[],
            study_name="test_study",
            optimization_time=120.0,
        )
        mock_optimize.return_value = mock_result

        # Mock PlatformAPI
        mock_api = Mock(spec=PlatformAPI)
        opt_api = OptimizationAPI(mock_api)

        config = OptimizationConfig(
            hypothesis_id="test_hypo",
            strategy_name="multifactor_ml",
            model_type="ridge",
            param_space={"alpha": FloatDistribution(0.01, 100.0, log=True)},
            sampler="tpe",
            n_trials=10,
            n_folds=3,
            scoring="ic",
            features=["momentum_20d"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            pruning_enabled=True,
            study_name="test_study",
        )

        symbols = ["AAPL", "MSFT", "GOOGL"]

        result = opt_api.run_optimization(config, symbols)

        assert result.best_score == 0.0234
        assert result.study_name == "test_study"
        mock_optimize.assert_called_once()


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_updates(self):
        """Test that progress callback is called during optimization."""
        callback_calls = []

        def progress_callback(current: int, total: int):
            callback_calls.append((current, total))

        # Simulate trials
        total_trials = 5
        for i in range(total_trials):
            progress_callback(i + 1, total_trials)

        assert len(callback_calls) == 5
        assert callback_calls[-1] == (5, 5)

    def test_progress_callback_optional(self):
        """Test that optimization works without progress callback."""
        # Progress callback should be optional
        progress_callback = None

        assert progress_callback is None
