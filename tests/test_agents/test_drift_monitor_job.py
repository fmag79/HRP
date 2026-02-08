"""Tests for drift monitor job."""
import pytest
from unittest.mock import Mock
from hrp.agents.drift_monitor_job import DriftMonitorJob, DriftConfig


@pytest.fixture
def mock_api():
    """Mock PlatformAPI for testing."""
    api = Mock()
    api.log_event = Mock()
    api.execute_write = Mock()
    return api


@pytest.fixture
def drift_config():
    """Default drift config for tests."""
    return DriftConfig(
        prediction_drift_threshold=0.20,
        feature_drift_threshold=0.15,
        lookback_days=30,
        auto_rollback=False,
    )


def test_drift_monitor_no_deployed_models(mock_api, drift_config):
    """Test job handles no deployed models."""
    mock_api.get_deployed_strategies.return_value = []

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    assert result["status"] == "no_deployed_models"
    assert result["models_checked"] == 0


def test_drift_monitor_no_drift_detected(mock_api, drift_config):
    """Test job when no drift detected."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.check_model_drift.return_value = {
        "drift_detected": False,
        "drift_score": 0.05,
    }

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    assert result["status"] == "success"
    assert result["models_checked"] == 1
    assert len(result["drift_detected"]) == 0


def test_drift_monitor_detects_drift(mock_api, drift_config):
    """Test job detects drift."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.check_model_drift.return_value = {
        "drift_detected": True,
        "drift_type": "prediction",
        "drift_score": 0.25,
    }

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    assert result["status"] == "drift_detected"
    assert len(result["drift_detected"]) == 1
    assert result["drift_detected"][0]["model_name"] == "model_v1"


def test_drift_monitor_auto_rollback(mock_api, drift_config):
    """Test auto-rollback on drift detection."""
    drift_config.auto_rollback = True

    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.check_model_drift.return_value = {
        "drift_detected": True,
        "drift_type": "prediction",
        "drift_score": 0.30,
    }
    mock_api.rollback_deployment = Mock()

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    assert result["rollbacks_triggered"] == 1
    mock_api.rollback_deployment.assert_called_once()


def test_drift_monitor_skips_model_without_name(mock_api, drift_config):
    """Test job skips models without model_name."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {}},  # No model_name
    ]

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    mock_api.check_model_drift.assert_not_called()


def test_drift_monitor_handles_check_error(mock_api, drift_config):
    """Test job handles drift check errors gracefully."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.check_model_drift.side_effect = ValueError("Check failed")

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    result = job.execute()

    # Should not fail, just skip this model
    assert result["status"] == "success"
    assert len(result["drift_detected"]) == 0


def test_drift_monitor_records_checks(mock_api, drift_config):
    """Test job records drift checks to database."""
    mock_api.get_deployed_strategies.return_value = [
        {"hypothesis_id": "HYP-2026-001", "metadata": {"model_name": "model_v1"}}
    ]
    mock_api.check_model_drift.return_value = {
        "drift_detected": False,
        "drift_score": 0.05,
    }

    job = DriftMonitorJob(drift_config=drift_config, api=mock_api)
    job.execute()

    # Should have recorded the check
    mock_api.execute_write.assert_called()
