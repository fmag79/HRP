"""
Tests for OptimizationAPI.
"""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from hrp.api.optimization_api import (
    OptimizationAPI,
    OptimizationPreview,
    DEFAULT_PARAM_SPACES,
)
from hrp.ml.optimization import OptimizationConfig, OptimizationResult
from hrp.api.platform import PlatformAPI


@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = Mock()
    db.fetchdf = Mock(return_value=pd.DataFrame())
    db.fetchone = Mock(return_value=None)
    db.execute = Mock()
    db.fetchall = Mock(return_value=[])
    return db


@pytest.fixture
def api_with_mock_db(mock_db):
    """Create PlatformAPI with mocked database."""
    with patch("hrp.api.platform.get_db", return_value=mock_db):
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        return api


@pytest.fixture
def optimization_api(api_with_mock_db):
    """Create OptimizationAPI with mocked PlatformAPI."""
    return OptimizationAPI(api_with_mock_db)


@pytest.fixture
def sample_optimization_config():
    """Create a sample OptimizationConfig for testing."""
    return OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d", "volatility_60d"],
        param_space={"alpha": MagicMock()},
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        n_folds=5,
        n_trials=50,
        scoring_metric="ic",
        sampler="tpe",
    )


@pytest.fixture
def sample_optimization_result():
    """Create a sample OptimizationResult for testing."""
    return OptimizationResult(
        best_params={"alpha": 1.0},
        best_score=0.05,
        cv_results=pd.DataFrame({"mean_score": [0.03, 0.04, 0.05]}),
        fold_results=[],
        all_trials=[],
        hypothesis_id="test-hypothesis",
        n_trials_completed=50,
        early_stopped=False,
        early_stop_reason=None,
        execution_time_seconds=300.0,
    )


# =========================================================================
# Test get_default_param_space
# =========================================================================

def test_get_default_param_space_ridge(optimization_api):
    """Test getting default param space for ridge model."""
    from optuna.distributions import FloatDistribution
    param_space = optimization_api.get_default_param_space("ridge")

    assert "alpha" in param_space
    assert isinstance(param_space["alpha"], FloatDistribution)


def test_get_default_param_space_random_forest(optimization_api):
    """Test getting default param space for random forest model."""
    param_space = optimization_api.get_default_param_space("random_forest")

    assert "n_estimators" in param_space
    assert "max_depth" in param_space
    assert "min_samples_leaf" in param_space


def test_get_default_param_space_lightgbm(optimization_api):
    """Test getting default param space for LightGBM model."""
    param_space = optimization_api.get_default_param_space("lightgbm")

    assert "n_estimators" in param_space
    assert "learning_rate" in param_space
    assert "max_depth" in param_space
    assert "num_leaves" in param_space


def test_get_default_param_space_invalid_model(optimization_api):
    """Test getting default param space for invalid model type."""
    with pytest.raises(ValueError, match="Unsupported model type"):
        optimization_api.get_default_param_space("invalid_model")


def test_get_default_param_space_returns_copy(optimization_api):
    """Test that get_default_param_space returns a copy, not reference."""
    param_space1 = optimization_api.get_default_param_space("ridge")
    param_space2 = optimization_api.get_default_param_space("ridge")

    # Modifying one should not affect the other
    # (This test ensures the method returns copies, not references)
    assert param_space1 is not param_space2


# =========================================================================
# Test get_available_strategies
# =========================================================================

def test_get_available_strategies(optimization_api):
    """Test getting available strategies."""
    # Mock get_deployed_strategies
    optimization_api.api.get_deployed_strategies = Mock(
        return_value=[
            {"hypothesis_id": "strategy-1"},
            {"hypothesis_id": "strategy-2"},
        ]
    )

    strategies = optimization_api.get_available_strategies()

    assert strategies == ["strategy-1", "strategy-2"]
    optimization_api.api.get_deployed_strategies.assert_called_once()


def test_get_available_strategies_empty(optimization_api):
    """Test getting available strategies when none deployed."""
    # Mock get_deployed_strategies
    optimization_api.api.get_deployed_strategies = Mock(return_value=[])

    strategies = optimization_api.get_available_strategies()

    assert strategies == []


def test_get_available_strategies_error(optimization_api):
    """Test getting available strategies when API fails."""
    # Mock get_deployed_strategies to raise error
    optimization_api.api.get_deployed_strategies = Mock(
        side_effect=Exception("Database error")
    )

    strategies = optimization_api.get_available_strategies()

    # Should return empty list on error
    assert strategies == []


# =========================================================================
# Test estimate_execution_time
# =========================================================================

def test_estimate_execution_time_ridge(optimization_api, sample_optimization_config):
    """Test estimating execution time for ridge model."""
    # Ridge has complexity 1.0, TPE has overhead 1.0
    # 50 trials * 5 folds * 10s * 1.0 * 1.0 = 2500 seconds
    estimated = optimization_api.estimate_execution_time(sample_optimization_config)

    assert estimated > 0
    # Base calculation: trials * folds * base_time
    expected_base = 50 * 5 * optimization_api.BASE_TIME_PER_FOLD
    assert estimated == pytest.approx(expected_base, rel=0.1)


def test_estimate_execution_time_random_forest(optimization_api):
    """Test estimating execution time for random forest (slower)."""
    config = OptimizationConfig(
        model_type="random_forest",
        target="returns_20d",
        features=["momentum_20d"],
        param_space={"n_estimators": MagicMock()},
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        n_folds=5,
        n_trials=50,
        scoring_metric="ic",
        sampler="tpe",
    )

    # Random forest has complexity 2.0
    estimated = optimization_api.estimate_execution_time(config)

    # Should be 2x the base time
    expected_base = 50 * 5 * optimization_api.BASE_TIME_PER_FOLD
    assert estimated == pytest.approx(expected_base * 2.0, rel=0.1)


def test_estimate_execution_time_grid_sampler(optimization_api):
    """Test estimating execution time with grid sampler (slower)."""
    config = OptimizationConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d"],
        param_space={"alpha": MagicMock()},
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        n_folds=5,
        n_trials=50,
        scoring_metric="ic",
        sampler="grid",
    )

    # Grid sampler has overhead 2.0
    estimated = optimization_api.estimate_execution_time(config)

    # Should be 2x the base time
    expected_base = 50 * 5 * optimization_api.BASE_TIME_PER_FOLD
    assert estimated == pytest.approx(expected_base * 2.0, rel=0.1)


# =========================================================================
# Test preview_configuration
# =========================================================================

def test_preview_configuration_low_cost(optimization_api, sample_optimization_config):
    """Test preview configuration with low estimated cost."""
    # Small config for low cost (need < 120 seconds = 2 min)
    # With base_time=10s, we need: trials * folds * 10 < 120
    # Let's use 3 trials * 3 folds = 90 seconds (1.5 min)
    sample_optimization_config.n_trials = 3
    sample_optimization_config.n_folds = 3

    preview = optimization_api.preview_configuration(sample_optimization_config)

    assert isinstance(preview, OptimizationPreview)
    assert preview.estimated_time_seconds > 0
    assert "Low" in preview.estimated_cost_estimate
    assert preview.recommended_sampler == "tpe"
    assert len(preview.warnings) > 0
    assert "Configuration looks good!" in preview.warnings[0]


def test_preview_configuration_high_cost(optimization_api, sample_optimization_config):
    """Test preview configuration with high estimated cost."""
    # Large config for high cost
    sample_optimization_config.n_trials = 200
    sample_optimization_config.n_folds = 10

    preview = optimization_api.preview_configuration(sample_optimization_config)

    assert isinstance(preview, OptimizationPreview)
    assert "High" in preview.estimated_cost_estimate
    # Should have warning about high estimated time
    assert any("High estimated time" in w for w in preview.warnings)


def test_preview_configuration_grid_warning(optimization_api, sample_optimization_config):
    """Test preview configuration with grid sampler warning."""
    sample_optimization_config.n_trials = 100
    sample_optimization_config.sampler = "grid"

    preview = optimization_api.preview_configuration(sample_optimization_config)

    # Should have warning about grid sampler
    assert any("Grid sampler" in w for w in preview.warnings)


def test_preview_configuration_many_features(optimization_api, sample_optimization_config):
    """Test preview configuration with many features warning."""
    # Add many features
    sample_optimization_config.features = [f"feature_{i}" for i in range(35)]

    preview = optimization_api.preview_configuration(sample_optimization_config)

    # Should have warning about large feature set
    assert any("Large feature set" in w for w in preview.warnings)


def test_preview_configuration_low_folds(optimization_api, sample_optimization_config):
    """Test preview configuration with low fold count warning."""
    sample_optimization_config.n_folds = 2

    preview = optimization_api.preview_configuration(sample_optimization_config)

    # Should have warning about low fold count
    assert any("Low fold count" in w for w in preview.warnings)


def test_preview_configuration_parameter_summary(optimization_api, sample_optimization_config):
    """Test that parameter space summary is generated correctly."""
    # Create a real distribution
    from optuna.distributions import FloatDistribution
    sample_optimization_config.param_space = {
        "alpha": FloatDistribution(0.01, 100.0, log=True),
        "beta": FloatDistribution(0.1, 10.0, step=0.1),
    }

    preview = optimization_api.preview_configuration(sample_optimization_config)

    assert "alpha" in preview.parameter_space_summary
    assert "beta" in preview.parameter_space_summary
    assert "log" in preview.parameter_space_summary["alpha"]
    assert "step" in preview.parameter_space_summary["beta"]


# =========================================================================
# Test run_optimization
# =========================================================================

@patch("hrp.api.optimization_api.cross_validated_optimize")
def test_run_optimization(mock_optimize, optimization_api, sample_optimization_config):
    """Test running optimization."""
    # Mock the optimization function
    mock_optimize.return_value = OptimizationResult(
        best_params={"alpha": 1.0},
        best_score=0.05,
        cv_results=pd.DataFrame(),
        fold_results=[],
        all_trials=[],
        hypothesis_id="test",
        n_trials_completed=50,
        execution_time_seconds=300.0,
    )

    result = optimization_api.run_optimization(
        config=sample_optimization_config,
        symbols=["AAPL", "MSFT"],
    )

    assert result.n_trials_completed == 50
    mock_optimize.assert_called_once()


@patch("hrp.api.optimization_api.cross_validated_optimize")
def test_run_optimization_with_progress_callback(
    mock_optimize, optimization_api, sample_optimization_config
):
    """Test running optimization with progress callback."""
    # Mock the optimization function
    mock_optimize.return_value = OptimizationResult(
        best_params={"alpha": 1.0},
        best_score=0.05,
        cv_results=pd.DataFrame(),
        fold_results=[],
        all_trials=[],
        hypothesis_id="test",
        n_trials_completed=50,
        execution_time_seconds=300.0,
    )

    # Mock progress callback
    progress_callback = Mock()

    result = optimization_api.run_optimization(
        config=sample_optimization_config,
        symbols=["AAPL", "MSFT"],
        progress_callback=progress_callback,
    )

    # Progress callback should be called with completion
    progress_callback.assert_called_once_with(50, 50)


# =========================================================================
# Test list_studies
# =========================================================================

@patch("hrp.api.optimization_api.get_config")
def test_list_studies_empty(mock_get_config, optimization_api, tmp_path):
    """Test listing studies when storage is empty."""
    # Mock config to use temp directory
    mock_config = Mock()
    mock_config.data.optuna_dir = tmp_path
    mock_get_config.return_value = mock_config

    studies = optimization_api.list_studies()

    assert studies == []


@patch("hrp.api.optimization_api.get_config")
@patch("optuna.load_study")
def test_list_studies_with_filter(
    mock_load_study, mock_get_config, optimization_api, tmp_path
):
    """Test listing studies filtered by hypothesis_id."""
    # Mock config to use temp directory
    mock_config = Mock()
    mock_config.data.optuna_dir = tmp_path
    mock_get_config.return_value = mock_config

    # Create a fake study database file
    study_db = tmp_path / "test-hypothesis.db"
    study_db.touch()

    # Mock the loaded study
    mock_study = Mock()
    mock_study.direction = Mock()
    mock_study.direction.name = "maximize"
    mock_study.trials = []
    mock_load_study.return_value = mock_study

    studies = optimization_api.list_studies(hypothesis_id="test-hypothesis")

    assert len(studies) == 1
    assert studies[0]["study_name"] == "test-hypothesis"


# =========================================================================
# Test get_study_details
# =========================================================================

def test_get_study_details_not_found(optimization_api, tmp_path):
    """Test getting study details when study doesn't exist."""
    # Create temp directory
    with patch("hrp.api.optimization_api.get_config") as mock_get_config:
        mock_config = Mock()
        mock_config.data.optuna_dir = tmp_path
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="Study not found"):
            optimization_api.get_study_details("nonexistent")


@patch("hrp.api.optimization_api.get_config")
@patch("optuna.load_study")
def test_get_study_details_success(
    mock_load_study, mock_get_config, optimization_api, tmp_path
):
    """Test getting study details successfully."""
    # Mock config to use temp directory
    mock_config = Mock()
    mock_config.data.optuna_dir = tmp_path
    mock_get_config.return_value = mock_config

    # Create a fake study database file
    study_db = tmp_path / "test-study.db"
    study_db.touch()

    # Mock the loaded study
    mock_study = Mock()
    mock_study.direction = Mock()
    mock_study.direction.name = "maximize"
    mock_study.trials = []

    # Mock best trial
    mock_best_trial = Mock()
    mock_best_trial.params = {"alpha": 1.0}
    mock_best_trial.value = 0.05
    mock_best_trial.number = 0
    mock_study.best_trial = mock_best_trial

    # Mock trials dataframe
    mock_study.trials_dataframe = Mock(return_value=pd.DataFrame())

    mock_load_study.return_value = mock_study

    details = optimization_api.get_study_details("test-study")

    assert details["study_name"] == "test-study"
    assert details["direction"] == "maximize"
    assert details["best_params"] == {"alpha": 1.0}
    assert details["best_value"] == 0.05


# =========================================================================
# Test delete_study
# =========================================================================

def test_delete_study_not_found(optimization_api, tmp_path):
    """Test deleting a study that doesn't exist."""
    with patch("hrp.api.optimization_api.get_config") as mock_get_config:
        mock_config = Mock()
        mock_config.data.optuna_dir = tmp_path
        mock_get_config.return_value = mock_config

        result = optimization_api.delete_study("nonexistent")

        assert result is False


def test_delete_study_success(optimization_api, tmp_path):
    """Test deleting a study successfully."""
    # Create temp directory and study file
    with patch("hrp.api.optimization_api.get_config") as mock_get_config:
        mock_config = Mock()
        mock_config.data.optuna_dir = tmp_path
        mock_get_config.return_value = mock_config

        # Create a fake study database file
        study_db = tmp_path / "test-study.db"
        study_db.touch()

        # Verify file exists
        assert study_db.exists()

        result = optimization_api.delete_study("test-study")

        assert result is True
        # Verify file is deleted
        assert not study_db.exists()
