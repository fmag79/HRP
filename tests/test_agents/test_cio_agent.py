"""Tests for CIOAgent class."""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock, call

from hrp.agents.cio import CIOAgent


class TestCIOAgentInit:
    """Test CIOAgent initialization."""

    def test_init_with_defaults(self):
        """Test CIOAgent can be initialized with defaults."""
        with patch("hrp.agents.cio.PlatformAPI"):
            agent = CIOAgent(
                job_id="test-job-001",
                actor="agent:cio-test",
            )

            assert agent.agent_name == "cio"
            assert agent.agent_version == "1.0.0"
            assert agent.api is not None
            assert agent.thresholds["min_sharpe"] == 1.0
            assert agent.thresholds["max_drawdown"] == 0.20

    def test_init_with_custom_thresholds(self):
        """Test CIOAgent accepts custom thresholds."""
        with patch("hrp.agents.cio.PlatformAPI"):
            custom_thresholds = {
                "min_sharpe": 1.5,
                "max_drawdown": 0.15,
            }
            agent = CIOAgent(
                job_id="test-job-002",
                actor="agent:cio-test",
                thresholds=custom_thresholds,
            )

            assert agent.thresholds["min_sharpe"] == 1.5
            assert agent.thresholds["max_drawdown"] == 0.15
            # Defaults still present
            assert agent.thresholds["sharpe_decay_limit"] == 0.50

    def test_init_with_passed_api(self):
        """Test CIOAgent accepts a PlatformAPI instance."""
        with patch("hrp.agents.cio.PlatformAPI") as mock_api_class:
            mock_api = Mock()
            agent = CIOAgent(
                job_id="test-job-003",
                actor="agent:cio-test",
                api=mock_api,
            )

            assert agent.api == mock_api
            # PlatformAPI class not called again since we passed instance
            mock_api_class.return_value.assert_not_called()


class TestCIOModelStaging:
    """Test CIO Agent model staging on CONTINUE decisions."""

    def _make_agent(self) -> tuple[CIOAgent, Mock]:
        """Create a CIOAgent with a mocked PlatformAPI."""
        mock_api = Mock()
        mock_api.register_model.return_value = "1"
        mock_api.deploy_model.return_value = {"status": "deployed"}
        with patch("hrp.agents.cio.PlatformAPI"):
            agent = CIOAgent(
                job_id="test-staging-001",
                actor="agent:cio-test",
                api=mock_api,
            )
        return agent, mock_api

    def test_stages_model_on_continue_decision(self):
        """CIO Agent should call _stage_model_for_deployment for CONTINUE decisions."""
        agent, mock_api = self._make_agent()

        experiment_data = {
            "experiment_id": "exp-001",
            "model_type": "ridge",
            "features": ["momentum_20d", "volatility_60d"],
            "target": "returns_20d",
            "metrics": {"sharpe": 1.5, "mean_ic": 0.05},
            "hyperparameters": {"alpha": 0.1},
        }

        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            experiment_data=experiment_data,
        )

        mock_api.register_model.assert_called_once()
        mock_api.deploy_model.assert_called_once()
        mock_api.log_event.assert_called_once()

        # Verify register_model received correct args
        reg_call = mock_api.register_model.call_args
        assert reg_call.kwargs["model_name"] == "hyp_HYP-2026-001_ridge"
        assert reg_call.kwargs["model_type"] == "ridge"
        assert reg_call.kwargs["features"] == ["momentum_20d", "volatility_60d"]
        assert reg_call.kwargs["hypothesis_id"] == "HYP-2026-001"
        assert reg_call.kwargs["experiment_id"] == "exp-001"

        # Verify deploy_model received correct args
        dep_call = mock_api.deploy_model.call_args
        assert dep_call.kwargs["model_name"] == "hyp_HYP-2026-001_ridge"
        assert dep_call.kwargs["environment"] == "staging"
        assert dep_call.kwargs["actor"] == "agent:cio"

    def test_does_not_stage_on_kill_decision(self):
        """CIO Agent should NOT stage model for KILL decisions."""
        agent, mock_api = self._make_agent()

        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-002",
            decision="KILL",
            experiment_data={"experiment_id": "exp-002"},
        )

        mock_api.register_model.assert_not_called()
        mock_api.deploy_model.assert_not_called()
        mock_api.log_event.assert_not_called()

    def test_does_not_stage_on_conditional_decision(self):
        """CIO Agent should NOT stage model for CONDITIONAL decisions."""
        agent, mock_api = self._make_agent()

        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-003",
            decision="CONDITIONAL",
            experiment_data={"experiment_id": "exp-003"},
        )

        mock_api.register_model.assert_not_called()

    def test_does_not_stage_on_pivot_decision(self):
        """CIO Agent should NOT stage model for PIVOT decisions."""
        agent, mock_api = self._make_agent()

        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-004",
            decision="PIVOT",
            experiment_data={"experiment_id": "exp-004"},
        )

        mock_api.register_model.assert_not_called()

    def test_stage_model_handles_api_error_gracefully(self):
        """Staging should log error but not raise on API failure."""
        agent, mock_api = self._make_agent()
        mock_api.register_model.side_effect = RuntimeError("DB connection lost")

        # Should not raise
        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-005",
            decision="CONTINUE",
            experiment_data={"experiment_id": "exp-005", "model_type": "lasso"},
        )

        mock_api.register_model.assert_called_once()
        mock_api.deploy_model.assert_not_called()

    def test_stage_model_defaults_for_missing_fields(self):
        """Staging should use defaults when experiment_data has missing fields."""
        agent, mock_api = self._make_agent()

        agent._maybe_stage_model(
            hypothesis_id="HYP-2026-006",
            decision="CONTINUE",
            experiment_data={"experiment_id": "exp-006"},
        )

        reg_call = mock_api.register_model.call_args
        assert reg_call.kwargs["model_type"] == "unknown"
        assert reg_call.kwargs["features"] == []
        assert reg_call.kwargs["target"] == "returns_20d"
        assert reg_call.kwargs["metrics"] == {}
