"""Tests for daily prediction job."""
import pytest
from datetime import date
from unittest.mock import Mock, patch
from hrp.agents.prediction_job import DailyPredictionJob


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.get_universe.return_value = ["AAPL", "MSFT"]
    api.log_event = Mock()
    return api


def test_prediction_job_requires_deployed_strategies(mock_api):
    """Test job checks for deployed strategies."""
    mock_api.get_deployed_strategies.return_value = []

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "no_deployed_strategies"
    assert result["predictions_generated"] == 0


def test_prediction_job_generates_predictions(mock_api):
    """Test job generates predictions for deployed strategies."""
    # Mock deployed strategy
    mock_strategy = {
        "hypothesis_id": "HYP-2026-001",
        "metadata": {"model_name": "momentum_v1"},
    }

    mock_predictions = Mock()
    mock_predictions.__len__ = Mock(return_value=3)

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]
    mock_api.predict_model.return_value = mock_predictions

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "success"
    assert result["predictions_generated"] == 3
    assert result["strategies_processed"] == 1
    mock_api.predict_model.assert_called_once()


def test_prediction_job_handles_prediction_errors(mock_api):
    """Test job continues on prediction errors."""
    mock_strategy = {
        "hypothesis_id": "HYP-2026-001",
        "metadata": {"model_name": "broken_model"},
    }

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_universe.return_value = ["AAPL"]
    mock_api.predict_model.side_effect = ValueError("Model not found")

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "partial_failure"
    assert result["predictions_generated"] == 0
    assert result["errors"] == 1


def test_prediction_job_skips_missing_model_name(mock_api):
    """Test job skips strategies without model_name."""
    mock_strategy = {
        "hypothesis_id": "HYP-2026-001",
        "metadata": {},  # No model_name
    }

    mock_api.get_deployed_strategies.return_value = [mock_strategy]
    mock_api.get_universe.return_value = ["AAPL"]

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["errors"] == 1
    mock_api.predict_model.assert_not_called()


def test_prediction_job_processes_multiple_strategies(mock_api):
    """Test job processes all deployed strategies."""
    strategies = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_a"}},
        {"hypothesis_id": "HYP-2026-002", "metadata": {"model_name": "model_b"}},
    ]

    mock_predictions = Mock()
    mock_predictions.__len__ = Mock(return_value=2)

    mock_api.get_deployed_strategies.return_value = strategies
    mock_api.get_universe.return_value = ["AAPL", "MSFT"]
    mock_api.predict_model.return_value = mock_predictions

    job = DailyPredictionJob(api=mock_api)
    result = job.execute()

    assert result["status"] == "success"
    assert result["strategies_processed"] == 2
    assert result["predictions_generated"] == 4  # 2 predictions x 2 strategies
    assert mock_api.predict_model.call_count == 2
