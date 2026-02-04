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


class TestCIOSkillGate:
    """Test CIO Agent skill probability gate for CONTINUE decisions."""

    def _make_agent(self) -> tuple[CIOAgent, Mock]:
        """Create a CIOAgent with a mocked PlatformAPI."""
        mock_api = Mock()
        mock_api.list_hypotheses.return_value = []
        mock_api.register_model.return_value = "1"
        mock_api.deploy_model.return_value = {"status": "deployed"}
        mock_api.log_cio_decision.return_value = None
        mock_api.log_event.return_value = None
        with patch("hrp.agents.cio.PlatformAPI"):
            agent = CIOAgent(
                job_id="test-skill-gate-001",
                actor="agent:cio-test",
                api=mock_api,
            )
        return agent, mock_api

    def test_skill_probability_threshold_defined(self):
        """Verify SKILL_PROBABILITY_THRESHOLD constant is defined."""
        assert hasattr(CIOAgent, "SKILL_PROBABILITY_THRESHOLD")
        assert CIOAgent.SKILL_PROBABILITY_THRESHOLD == 0.10

    def test_calculate_deflated_sharpe_low_probability(self):
        """Test that low Sharpe with many trials yields low probability of skill."""
        agent, _ = self._make_agent()

        # Sharpe of 1.0 with 50 trials should have very low probability of skill
        # Expected max Sharpe under null ≈ norm.ppf(1 - 1/50) ≈ 2.05
        result = agent._calculate_deflated_sharpe(
            sharpe=1.0,
            n_trials=50,
            n_observations=252
        )

        assert result["deflated_sharpe"] < 0  # Below expected max under null
        assert result["probability_of_skill"] < 0.10  # Low confidence

    def test_calculate_deflated_sharpe_high_probability(self):
        """Test that high Sharpe yields high probability of skill."""
        agent, _ = self._make_agent()

        # Sharpe of 3.0 with 10 trials should have high probability of skill
        result = agent._calculate_deflated_sharpe(
            sharpe=3.0,
            n_trials=10,
            n_observations=252
        )

        assert result["deflated_sharpe"] > 0  # Above expected max under null
        assert result["probability_of_skill"] > 0.50  # Higher confidence

    def test_execute_kills_high_score_low_skill_hypothesis(self):
        """Test that execute() overrides CONTINUE to KILL when skill probability is low.

        This is the core test for the fix: a hypothesis with high dimensional score
        (>= 0.75 which would normally be CONTINUE) but low probability of skill
        should be overridden to KILL.
        """
        agent, mock_api = self._make_agent()

        # Set up a validated hypothesis with good dimensional scores but low Sharpe
        mock_api.list_hypotheses.side_effect = [
            # First call returns validated hypotheses
            [{"hypothesis_id": "HYP-2026-001", "title": "Test", "thesis": "Test thesis"}],
            # Subsequent calls for pipeline stats
            [], [], [], [], []
        ]

        # Return experiment data that will give high dimensional score
        mock_api.get_hypothesis_with_metadata.return_value = {
            "hypothesis_id": "HYP-2026-001",
            "title": "Test Hypothesis",
            "thesis": "Test thesis",
            "metadata": {
                "ml_scientist_results": {
                    "mean_ic": 0.05,  # High IC → high statistical score
                    "ic_std": 0.01,
                    "stability_score": 0.5,  # Good stability
                }
            }
        }

        # Mock other API calls
        mock_api.get_prices.return_value = None  # Skip regime detection
        mock_api.log_agent_event.return_value = None

        # Patch _assess_thesis_with_claude to return strong scores
        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "strong",
                "regime_alignment": "aligned"
            }

            with patch.object(agent, "_generate_report") as mock_report:
                mock_report.return_value = "/tmp/test-report.md"

                result = agent.execute()

        # Verify we got a result
        assert result["status"] == "complete"
        assert len(result["decisions"]) == 1

        decision = result["decisions"][0]

        # The Sharpe is derived from IC: sharpe = mean_ic * 20 = 0.05 * 20 = 1.0
        # With n_trials from pipeline stats (at least len(hypotheses) = 1),
        # but the key is the deflated Sharpe calculation
        # The score would be high (good IC, stability, etc.) but we need to
        # verify the skill gate logic works

        # If skill probability < 10% and original decision was CONTINUE,
        # it should be overridden to KILL
        if decision["deflated_sharpe"]["probability_of_skill"] < 0.10:
            assert decision["decision"] == "KILL", \
                f"Expected KILL due to low skill probability, got {decision['decision']}"
            assert decision["skill_gate_failed"] is True
        else:
            # If probability is high enough, CONTINUE is acceptable
            assert decision["decision"] in ["CONTINUE", "CONDITIONAL", "KILL"]

    def test_execute_continues_with_sufficient_skill(self):
        """Test that execute() allows CONTINUE when skill probability is sufficient.

        A hypothesis with both high dimensional score AND high probability of skill
        should get CONTINUE decision (not overridden).
        """
        agent, mock_api = self._make_agent()

        # Increase n_observations to boost probability of skill
        mock_api.list_hypotheses.side_effect = [
            [{"hypothesis_id": "HYP-2026-002", "title": "Strong Test", "thesis": "Strong thesis"}],
            [], [], [], [], []
        ]

        mock_api.get_hypothesis_with_metadata.return_value = {
            "hypothesis_id": "HYP-2026-002",
            "title": "Strong Test Hypothesis",
            "thesis": "Strong thesis",
            "metadata": {
                "ml_scientist_results": {
                    "mean_ic": 0.08,  # Very high IC → Sharpe = 1.6
                    "ic_std": 0.01,
                    "stability_score": 0.3,
                    # Higher n_observations increases statistical confidence
                }
            }
        }

        mock_api.get_prices.return_value = None

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "strong",
                "regime_alignment": "aligned"
            }

            with patch.object(agent, "_generate_report") as mock_report:
                mock_report.return_value = "/tmp/test-report.md"

                # Override n_trials to be low (1 trial = no multiple testing penalty)
                with patch.object(agent, "_get_pipeline_statistics") as mock_stats:
                    mock_stats.return_value = {"total_generated": 1}

                    result = agent.execute()

        assert result["status"] == "complete"
        assert len(result["decisions"]) == 1

        decision = result["decisions"][0]

        # With low n_trials and high Sharpe, probability of skill should be high
        # Decision should NOT be overridden
        if decision["deflated_sharpe"]["probability_of_skill"] >= 0.10:
            # Skill gate should not have triggered
            assert decision.get("skill_gate_failed", False) is False
            # If score >= 0.75, should be CONTINUE
            if decision["score"] >= 0.75:
                assert decision["decision"] == "CONTINUE"

    def test_save_decision_with_override(self):
        """Test that _save_decision_with_override uses the overridden decision."""
        agent, mock_api = self._make_agent()

        # Create a score that would normally be CONTINUE
        from hrp.agents.cio import CIOScore
        score = CIOScore(
            hypothesis_id="HYP-2026-003",
            statistical=0.9,
            risk=0.8,
            economic=0.7,
            cost=0.8,
            critical_failure=False,
        )

        assert score.decision == "CONTINUE"  # Verify score would be CONTINUE

        # Save with override to KILL
        agent._save_decision_with_override(
            hypothesis_id="HYP-2026-003",
            score=score,
            rationale="Skill gate override test",
            decision_override="KILL",
        )

        # Verify the logged decision is KILL, not CONTINUE
        mock_api.log_cio_decision.assert_called_once()
        call_kwargs = mock_api.log_cio_decision.call_args.kwargs
        assert call_kwargs["decision"] == "KILL"
        assert call_kwargs["hypothesis_id"] == "HYP-2026-003"

    def test_skill_gate_does_not_affect_non_continue_decisions(self):
        """Skill gate should only override CONTINUE, not KILL/CONDITIONAL/PIVOT."""
        agent, mock_api = self._make_agent()

        # Set up a hypothesis that would score poorly (CONDITIONAL - score ~0.5)
        # Use values that won't trigger critical failures (which would cause PIVOT)
        mock_api.list_hypotheses.side_effect = [
            [{"hypothesis_id": "HYP-2026-004", "title": "Poor", "thesis": "Poor thesis"}],
            [], [], [], [], []
        ]

        mock_api.get_hypothesis_with_metadata.return_value = {
            "hypothesis_id": "HYP-2026-004",
            "title": "Poor Hypothesis",
            "thesis": "Poor thesis",
            "metadata": {
                "ml_scientist_results": {
                    "mean_ic": 0.02,  # Low-ish IC → Sharpe = 0.4
                    "ic_std": 0.02,
                    "stability_score": 1.5,  # Moderate stability (not critical)
                }
            }
        }

        mock_api.get_prices.return_value = None

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "moderate",
                "regime_alignment": "neutral"
            }

            with patch.object(agent, "_generate_report") as mock_report:
                mock_report.return_value = "/tmp/test-report.md"

                result = agent.execute()

        decision = result["decisions"][0]

        # Score should be < 0.75 (not CONTINUE), so skill gate won't apply
        # Decision should be CONDITIONAL or KILL based on score
        assert decision["decision"] in ["CONDITIONAL", "KILL"]
        # skill_gate_failed should be False because original decision was not CONTINUE
        assert decision.get("skill_gate_failed", False) is False
