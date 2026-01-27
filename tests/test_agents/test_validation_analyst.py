"""
Tests for Validation Analyst research agent.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.research_agents import (
    ValidationAnalyst,
    ValidationCheck,
    ValidationSeverity,
    HypothesisValidation,
    ValidationAnalystReport,
)


class TestValidationDataclasses:
    """Tests for Validation Analyst dataclasses."""

    def test_validation_severity_enum(self):
        """ValidationSeverity has expected values."""
        assert ValidationSeverity.NONE.value == "none"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_validation_check_creation(self):
        """ValidationCheck can be created with all fields."""
        check = ValidationCheck(
            name="parameter_sensitivity",
            passed=True,
            severity=ValidationSeverity.NONE,
            details={"baseline_sharpe": 1.2, "variations": {}},
            message="Parameters are stable",
        )
        assert check.name == "parameter_sensitivity"
        assert check.passed is True
        assert check.severity == ValidationSeverity.NONE

    def test_hypothesis_validation_properties(self):
        """HypothesisValidation computes properties correctly."""
        validation = HypothesisValidation(
            hypothesis_id="HYP-2026-001",
            experiment_id="exp_123",
            validation_date=date(2026, 1, 26),
        )

        # Add checks
        validation.add_check(ValidationCheck(
            name="test1",
            passed=True,
            severity=ValidationSeverity.NONE,
            details={},
            message="OK",
        ))
        validation.add_check(ValidationCheck(
            name="test2",
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            details={},
            message="Failed",
        ))

        assert validation.overall_passed is False
        assert validation.critical_count == 1
        assert validation.warning_count == 0
        assert validation.has_critical_issues is True

    def test_validation_analyst_report(self):
        """ValidationAnalystReport aggregates correctly."""
        report = ValidationAnalystReport(
            report_date=date(2026, 1, 26),
            hypotheses_validated=5,
            hypotheses_passed=3,
            hypotheses_failed=2,
            validations=[],
            duration_seconds=45.2,
        )
        assert report.hypotheses_validated == 5
        assert report.hypotheses_passed == 3


class TestValidationAnalystInit:
    """Tests for ValidationAnalyst initialization."""

    def test_default_initialization(self):
        """ValidationAnalyst initializes with defaults."""
        agent = ValidationAnalyst()
        assert agent.ACTOR == "agent:validation-analyst"
        assert agent.actor == "agent:validation-analyst"
        assert agent.job_id == "validation_analyst_review"

    def test_custom_hypothesis_ids(self):
        """ValidationAnalyst accepts hypothesis filter."""
        agent = ValidationAnalyst(hypothesis_ids=["HYP-2026-001"])
        assert agent.hypothesis_ids == ["HYP-2026-001"]

    def test_custom_thresholds(self):
        """ValidationAnalyst accepts custom thresholds."""
        agent = ValidationAnalyst(
            param_sensitivity_threshold=0.6,
            min_profitable_periods=0.5,
            min_profitable_regimes=1,
        )
        assert agent.param_sensitivity_threshold == 0.6
        assert agent.min_profitable_periods == 0.5
        assert agent.min_profitable_regimes == 1

    def test_send_alerts_flag(self):
        """ValidationAnalyst accepts send_alerts flag."""
        agent = ValidationAnalyst(send_alerts=False)
        assert agent.send_alerts is False


class TestParameterSensitivity:
    """Tests for parameter sensitivity check."""

    def test_stable_parameters_pass(self):
        """Stable parameters pass the check."""
        agent = ValidationAnalyst()
        experiments = {
            "baseline": {"sharpe": 1.0, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.8, "params": {"lookback": 16}},
            "var_2": {"sharpe": 0.9, "params": {"lookback": 24}},
        }
        check = agent._check_parameter_sensitivity(experiments, "baseline")
        assert check.passed is True
        assert check.severity == ValidationSeverity.NONE

    def test_unstable_parameters_fail(self):
        """Unstable parameters fail the check."""
        agent = ValidationAnalyst()
        experiments = {
            "baseline": {"sharpe": 1.0, "params": {"lookback": 20}},
            "var_1": {"sharpe": 0.2, "params": {"lookback": 16}},  # 20% of baseline
        }
        check = agent._check_parameter_sensitivity(experiments, "baseline")
        assert check.passed is False
        assert check.severity == ValidationSeverity.CRITICAL


class TestTimeStability:
    """Tests for time stability check."""

    def test_stable_periods_pass(self):
        """Profitable across periods passes."""
        agent = ValidationAnalyst()
        period_metrics = [
            {"period": "2020", "sharpe": 1.0, "profitable": True},
            {"period": "2021", "sharpe": 0.8, "profitable": True},
            {"period": "2022", "sharpe": 0.5, "profitable": True},
        ]
        check = agent._check_time_stability(period_metrics)
        assert check.passed is True

    def test_unstable_periods_fail(self):
        """Unprofitable in most periods fails."""
        agent = ValidationAnalyst()
        period_metrics = [
            {"period": "2020", "sharpe": 1.0, "profitable": True},
            {"period": "2021", "sharpe": -0.5, "profitable": False},
            {"period": "2022", "sharpe": -0.3, "profitable": False},
        ]
        check = agent._check_time_stability(period_metrics)
        assert check.passed is False
        assert check.severity == ValidationSeverity.CRITICAL


class TestRegimeStability:
    """Tests for regime stability check."""

    def test_regime_stable_pass(self):
        """Profitable in all regimes passes."""
        agent = ValidationAnalyst()
        regime_metrics = {
            "bull": {"sharpe": 1.5, "profitable": True},
            "bear": {"sharpe": 0.3, "profitable": True},
            "sideways": {"sharpe": 0.2, "profitable": True},
        }
        check = agent._check_regime_stability(regime_metrics)
        assert check.passed is True

    def test_regime_unstable_fail(self):
        """Profitable in only 1 regime fails."""
        agent = ValidationAnalyst()
        regime_metrics = {
            "bull": {"sharpe": 1.5, "profitable": True},
            "bear": {"sharpe": -0.5, "profitable": False},
            "sideways": {"sharpe": -0.3, "profitable": False},
        }
        check = agent._check_regime_stability(regime_metrics)
        assert check.passed is False


class TestExecutionCost:
    """Tests for execution cost estimation."""

    def test_execution_cost_calculation(self):
        """Execution costs calculated correctly."""
        agent = ValidationAnalyst(commission_bps=5, slippage_bps=10)

        # 100 trades, $10,000 average trade size
        check = agent._estimate_execution_costs(
            num_trades=100,
            avg_trade_value=10000,
            gross_return=0.15,  # 15% gross return
        )

        assert check.passed is True
        assert "net_return" in check.details
        assert "total_cost_bps" in check.details
        # Total cost: 15 bps * 100 trades = 1500 bps = 15%
        # Net return should be ~0% (gross 15% - costs 15%)

    def test_high_costs_warning(self):
        """High costs relative to return trigger warning."""
        agent = ValidationAnalyst(commission_bps=50, slippage_bps=50)

        check = agent._estimate_execution_costs(
            num_trades=100,
            avg_trade_value=10000,
            gross_return=0.10,  # 10% gross
        )

        # Costs exceed 50% of return
        assert check.severity in [ValidationSeverity.WARNING, ValidationSeverity.CRITICAL]


class TestValidationAnalystExecute:
    """Tests for ValidationAnalyst execute method."""

    @patch.object(ValidationAnalyst, "_get_hypotheses_to_validate")
    @patch.object(ValidationAnalyst, "_validate_hypothesis")
    @patch.object(ValidationAnalyst, "_write_research_note")
    @patch.object(ValidationAnalyst, "_log_agent_event")
    def test_execute_processes_hypotheses(
        self,
        mock_log,
        mock_write_note,
        mock_validate,
        mock_get_hypotheses,
    ):
        """Execute processes all hypotheses and returns report."""
        # Setup mocks
        mock_get_hypotheses.return_value = [
            {"hypothesis_id": "HYP-2026-001", "experiment_id": "exp_1"},
            {"hypothesis_id": "HYP-2026-002", "experiment_id": "exp_2"},
        ]
        mock_validate.return_value = HypothesisValidation(
            hypothesis_id="HYP-2026-001",
            experiment_id="exp_1",
            validation_date=date.today(),
            checks=[ValidationCheck(
                name="test",
                passed=True,
                severity=ValidationSeverity.NONE,
                details={},
                message="OK",
            )],
        )

        agent = ValidationAnalyst(send_alerts=False)
        result = agent.execute()

        assert result["hypotheses_validated"] == 2
        assert mock_validate.call_count == 2
        assert mock_write_note.called
        assert mock_log.called

    @patch.object(ValidationAnalyst, "_get_hypotheses_to_validate")
    @patch.object(ValidationAnalyst, "_log_agent_event")
    @patch.object(ValidationAnalyst, "_write_research_note")
    def test_execute_no_hypotheses(
        self, mock_write_note, mock_log, mock_get_hypotheses
    ):
        """Execute handles no hypotheses gracefully."""
        mock_get_hypotheses.return_value = []

        agent = ValidationAnalyst(send_alerts=False)
        result = agent.execute()

        assert result["hypotheses_validated"] == 0
        assert result["hypotheses_passed"] == 0


def test_validation_analyst_exported():
    """ValidationAnalyst is exported from hrp.agents module."""
    from hrp.agents import (
        ValidationAnalyst,
        ValidationCheck,
        ValidationSeverity,
        HypothesisValidation,
        ValidationAnalystReport,
    )

    assert ValidationAnalyst is not None
    assert ValidationCheck is not None
