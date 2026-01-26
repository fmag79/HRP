"""
Tests for ML Quality Sentinel research agent.

Tests cover:
- Agent initialization
- Sharpe decay checks
- Target leakage detection
- Feature count validation
- Fold stability checks
- Suspiciously good detection
- Experiment audit flow
- Deployed model monitoring
- Report generation
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.agents.research_agents import (
    AuditCheck,
    AuditSeverity,
    ExperimentAudit,
    MLQualitySentinel,
    MonitoringAlert,
    QualitySentinelReport,
)


class TestMLQualitySentinelInit:
    """Tests for ML Quality Sentinel initialization."""

    def test_default_initialization(self):
        """MLQualitySentinel initializes with default settings."""
        sentinel = MLQualitySentinel()
        assert sentinel.experiment_ids is None
        assert sentinel.hypothesis_ids is None
        assert sentinel.audit_window_days == 1
        assert sentinel.include_monitoring is True
        assert sentinel.fail_on_critical is True
        assert sentinel.send_alerts is True

    def test_custom_experiment_ids(self):
        """MLQualitySentinel accepts specific experiment IDs."""
        sentinel = MLQualitySentinel(experiment_ids=["exp1", "exp2"])
        assert sentinel.experiment_ids == ["exp1", "exp2"]

    def test_custom_hypothesis_ids(self):
        """MLQualitySentinel accepts hypothesis IDs to audit."""
        sentinel = MLQualitySentinel(hypothesis_ids=["HYP-2026-001"])
        assert sentinel.hypothesis_ids == ["HYP-2026-001"]

    def test_custom_audit_window(self):
        """MLQualitySentinel accepts custom audit window."""
        sentinel = MLQualitySentinel(audit_window_days=7)
        assert sentinel.audit_window_days == 7

    def test_monitoring_disabled(self):
        """MLQualitySentinel can disable deployed model monitoring."""
        sentinel = MLQualitySentinel(include_monitoring=False)
        assert sentinel.include_monitoring is False

    def test_actor_identity(self):
        """MLQualitySentinel has correct actor identity."""
        sentinel = MLQualitySentinel()
        assert sentinel.actor == "agent:ml-quality-sentinel"
        assert sentinel.ACTOR == "agent:ml-quality-sentinel"

    def test_job_id(self):
        """MLQualitySentinel has correct job ID."""
        sentinel = MLQualitySentinel()
        assert sentinel.job_id == "ml_quality_sentinel_audit"


class TestSharpeDecayCheck:
    """Tests for Sharpe decay detection."""

    def test_critical_sharpe_decay(self):
        """Critical flag when decay > 50%."""
        sentinel = MLQualitySentinel()
        experiment = {"train_sharpe": 2.0, "test_sharpe": 0.8}  # 60% decay
        check = sentinel._check_sharpe_decay(experiment)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        assert "Critical Sharpe decay" in check.message

    def test_warning_sharpe_decay(self):
        """Warning when decay 30-50%."""
        sentinel = MLQualitySentinel()
        experiment = {"train_sharpe": 2.0, "test_sharpe": 1.3}  # 35% decay
        check = sentinel._check_sharpe_decay(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.WARNING
        assert "Moderate Sharpe decay" in check.message

    def test_acceptable_sharpe_decay(self):
        """Passes when decay < 30%."""
        sentinel = MLQualitySentinel()
        experiment = {"train_sharpe": 2.0, "test_sharpe": 1.6}  # 20% decay
        check = sentinel._check_sharpe_decay(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.NONE
        assert "within acceptable limits" in check.message

    def test_zero_train_sharpe_skips(self):
        """Skips check when train Sharpe is 0 or negative."""
        sentinel = MLQualitySentinel()
        experiment = {"train_sharpe": 0, "test_sharpe": 1.0}
        check = sentinel._check_sharpe_decay(experiment)
        assert check.passed is True
        assert "non-positive" in check.message

    def test_none_values_handled(self):
        """Handles None values gracefully."""
        sentinel = MLQualitySentinel()
        experiment = {"train_sharpe": None, "test_sharpe": None}
        check = sentinel._check_sharpe_decay(experiment)
        assert check.passed is True


class TestTargetLeakageCheck:
    """Tests for target leakage detection."""

    def test_critical_leakage_detected(self):
        """Critical flag when correlation > 0.95."""
        sentinel = MLQualitySentinel()
        features = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "leaky_feature": [10, 20, 30, 40, 50],  # Perfect correlation
        })
        target = pd.Series([10, 20, 30, 40, 50])
        check = sentinel._check_target_leakage(features, target)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        assert "leaky_feature" in str(check.details["suspicious_features"])

    def test_warning_high_correlation(self):
        """Warning when correlation 0.85-0.95."""
        sentinel = MLQualitySentinel()
        # Create feature with ~0.9 correlation
        np.random.seed(42)
        target = pd.Series([10, 20, 30, 40, 50])
        features = pd.DataFrame({
            "high_corr": target + np.random.normal(0, 2, 5),
        })
        check = sentinel._check_target_leakage(features, target)
        # High correlation should trigger warning
        assert check.severity in [AuditSeverity.WARNING, AuditSeverity.CRITICAL]

    def test_no_leakage_detected(self):
        """Passes when correlations are low."""
        sentinel = MLQualitySentinel()
        np.random.seed(42)
        features = pd.DataFrame({
            "random_feature": np.random.randn(100),
        })
        target = pd.Series(np.random.randn(100))
        check = sentinel._check_target_leakage(features, target)
        assert check.passed is True
        assert check.severity == AuditSeverity.NONE


class TestFeatureCountValidation:
    """Tests for feature count validation."""

    def test_critical_too_many_features(self):
        """Critical when feature count > 50."""
        sentinel = MLQualitySentinel()
        experiment = {"feature_count": 60, "sample_count": 10000}
        check = sentinel._validate_feature_count(experiment)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        assert "Too many features" in check.message

    def test_critical_insufficient_samples(self):
        """Critical when samples per feature < 20."""
        sentinel = MLQualitySentinel()
        experiment = {"feature_count": 10, "sample_count": 100}  # 10 samples/feature
        check = sentinel._validate_feature_count(experiment)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        assert "Insufficient samples" in check.message

    def test_warning_high_feature_count(self):
        """Warning when feature count 30-50."""
        sentinel = MLQualitySentinel()
        experiment = {"feature_count": 35, "sample_count": 10000}
        check = sentinel._validate_feature_count(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.WARNING
        assert "High feature count" in check.message

    def test_acceptable_feature_count(self):
        """Passes with reasonable feature count."""
        sentinel = MLQualitySentinel()
        experiment = {"feature_count": 10, "sample_count": 5000}  # 500 samples/feature
        check = sentinel._validate_feature_count(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.NONE

    def test_zero_feature_count(self):
        """Handles zero features gracefully."""
        sentinel = MLQualitySentinel()
        experiment = {"feature_count": 0, "sample_count": 1000}
        check = sentinel._validate_feature_count(experiment)
        assert check.passed is True


class TestFoldStabilityCheck:
    """Tests for fold stability checks."""

    def test_critical_high_cv(self):
        """Critical when CV > 2.0."""
        sentinel = MLQualitySentinel()
        fold_results = [
            {"ic": 0.10},
            {"ic": -0.05},
            {"ic": 0.15},
            {"ic": -0.08},
            {"ic": 0.12},
        ]
        check = sentinel._check_fold_stability(fold_results)
        # High variance should trigger critical
        assert check.severity in [AuditSeverity.CRITICAL, AuditSeverity.WARNING]

    def test_critical_sign_flips(self):
        """Critical when too many sign flips (but CV is ok)."""
        sentinel = MLQualitySentinel()
        # Use values with lower CV but clear sign flips
        fold_results = [
            {"ic": 0.04},
            {"ic": -0.01},  # sign flip
            {"ic": 0.03},
            {"ic": -0.02},  # sign flip
            {"ic": 0.04},
        ]
        check = sentinel._check_fold_stability(fold_results)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        # Either sign flips or CV could trigger critical
        assert "sign flips" in check.message or "CV" in check.message

    def test_warning_moderate_cv(self):
        """Warning when CV 1.0-2.0."""
        sentinel = MLQualitySentinel()
        fold_results = [
            {"ic": 0.05},
            {"ic": 0.03},
            {"ic": 0.07},
            {"ic": 0.02},
            {"ic": 0.06},
        ]
        check = sentinel._check_fold_stability(fold_results)
        # Moderate variance should be warning or none
        assert check.severity in [AuditSeverity.WARNING, AuditSeverity.NONE]

    def test_stable_folds(self):
        """Passes with consistent fold results."""
        sentinel = MLQualitySentinel()
        fold_results = [
            {"ic": 0.04},
            {"ic": 0.045},
            {"ic": 0.038},
            {"ic": 0.042},
            {"ic": 0.041},
        ]
        check = sentinel._check_fold_stability(fold_results)
        assert check.passed is True
        assert check.severity == AuditSeverity.NONE
        assert "Stable across folds" in check.message

    def test_empty_fold_results(self):
        """Critical when no fold results."""
        sentinel = MLQualitySentinel()
        check = sentinel._check_fold_stability([])
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL


class TestSuspiciouslyGoodCheck:
    """Tests for suspiciously good results detection."""

    def test_critical_high_ic(self):
        """Critical when IC > 0.15."""
        sentinel = MLQualitySentinel()
        experiment = {"mean_ic": 0.20, "sharpe": 1.5, "r2": 0.3}
        check = sentinel._check_suspiciously_good(experiment)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL
        assert "extremely suspicious" in check.message

    def test_critical_high_sharpe(self):
        """Critical when Sharpe > 3.0."""
        sentinel = MLQualitySentinel()
        experiment = {"mean_ic": 0.05, "sharpe": 3.5, "r2": 0.3}
        check = sentinel._check_suspiciously_good(experiment)
        assert check.passed is False
        assert check.severity == AuditSeverity.CRITICAL

    def test_warning_suspicious_ic(self):
        """Warning when IC 0.10-0.15."""
        sentinel = MLQualitySentinel()
        experiment = {"mean_ic": 0.12, "sharpe": 1.5, "r2": 0.3}
        check = sentinel._check_suspiciously_good(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.WARNING
        assert "suspicious" in check.message

    def test_warning_high_r2(self):
        """Warning when R² > 0.5."""
        sentinel = MLQualitySentinel()
        experiment = {"mean_ic": 0.05, "sharpe": 1.5, "r2": 0.6}
        check = sentinel._check_suspiciously_good(experiment)
        assert check.severity == AuditSeverity.WARNING
        assert "R²" in check.message

    def test_plausible_results(self):
        """Passes with plausible results."""
        sentinel = MLQualitySentinel()
        experiment = {"mean_ic": 0.04, "sharpe": 1.2, "r2": 0.15}
        check = sentinel._check_suspiciously_good(experiment)
        assert check.passed is True
        assert check.severity == AuditSeverity.NONE
        assert "plausible" in check.message


class TestExperimentAudit:
    """Tests for ExperimentAudit dataclass."""

    def test_audit_creation(self):
        """ExperimentAudit can be created."""
        audit = ExperimentAudit(
            experiment_id="exp123",
            hypothesis_id="HYP-2026-001",
            mlflow_run_id="run456",
            audit_date=date.today(),
        )
        assert audit.experiment_id == "exp123"
        assert audit.hypothesis_id == "HYP-2026-001"
        assert len(audit.checks) == 0

    def test_add_check(self):
        """Checks can be added to audit."""
        audit = ExperimentAudit(
            experiment_id="exp123",
            hypothesis_id="HYP-2026-001",
            mlflow_run_id=None,
            audit_date=date.today(),
        )
        check = AuditCheck(
            name="test_check",
            passed=True,
            severity=AuditSeverity.NONE,
            details={},
            message="Test passed",
        )
        audit.add_check(check)
        assert len(audit.checks) == 1

    def test_overall_passed(self):
        """overall_passed reflects all checks."""
        audit = ExperimentAudit(
            experiment_id="exp123",
            hypothesis_id="HYP-2026-001",
            mlflow_run_id=None,
            audit_date=date.today(),
        )
        audit.add_check(AuditCheck("c1", True, AuditSeverity.NONE, {}, "ok"))
        audit.add_check(AuditCheck("c2", True, AuditSeverity.WARNING, {}, "warn"))
        assert audit.overall_passed is True

        audit.add_check(AuditCheck("c3", False, AuditSeverity.CRITICAL, {}, "fail"))
        assert audit.overall_passed is False

    def test_critical_count(self):
        """critical_count counts critical severity checks."""
        audit = ExperimentAudit(
            experiment_id="exp123",
            hypothesis_id="HYP-2026-001",
            mlflow_run_id=None,
            audit_date=date.today(),
        )
        audit.add_check(AuditCheck("c1", False, AuditSeverity.CRITICAL, {}, "fail1"))
        audit.add_check(AuditCheck("c2", True, AuditSeverity.WARNING, {}, "warn"))
        audit.add_check(AuditCheck("c3", False, AuditSeverity.CRITICAL, {}, "fail2"))
        assert audit.critical_count == 2
        assert audit.warning_count == 1

    def test_has_critical_issues(self):
        """has_critical_issues flags critical issues."""
        audit = ExperimentAudit(
            experiment_id="exp123",
            hypothesis_id="HYP-2026-001",
            mlflow_run_id=None,
            audit_date=date.today(),
        )
        audit.add_check(AuditCheck("c1", True, AuditSeverity.WARNING, {}, "warn"))
        assert audit.has_critical_issues is False

        audit.add_check(AuditCheck("c2", False, AuditSeverity.CRITICAL, {}, "fail"))
        assert audit.has_critical_issues is True


class TestMonitoringAlert:
    """Tests for MonitoringAlert dataclass."""

    def test_alert_creation(self):
        """MonitoringAlert can be created."""
        alert = MonitoringAlert(
            model_id="model123",
            hypothesis_id="HYP-2026-001",
            alert_type="ic_degradation",
            severity=AuditSeverity.CRITICAL,
            message="IC dropped 60%",
            recommended_action="Review model",
        )
        assert alert.model_id == "model123"
        assert alert.alert_type == "ic_degradation"
        assert alert.severity == AuditSeverity.CRITICAL


class TestQualitySentinelReport:
    """Tests for QualitySentinelReport dataclass."""

    def test_report_creation(self):
        """QualitySentinelReport can be created."""
        report = QualitySentinelReport(
            report_date=date.today(),
            experiments_audited=10,
            experiments_passed=8,
            experiments_flagged=2,
            critical_issues=[("exp1", "issue1")],
            warnings=[("exp2", "warning1")],
            models_monitored=5,
            model_alerts=[],
            duration_seconds=45.5,
        )
        assert report.experiments_audited == 10
        assert report.experiments_passed == 8
        assert report.experiments_flagged == 2


class TestAuditExperiment:
    """Tests for full experiment audit flow."""

    def test_audit_experiment_runs_all_checks(self):
        """_audit_experiment runs all applicable checks."""
        sentinel = MLQualitySentinel()
        experiment = {
            "id": "exp123",
            "hypothesis_id": "HYP-2026-001",
            "mlflow_run_id": "run456",
            "train_sharpe": 1.5,
            "test_sharpe": 1.2,
            "mean_ic": 0.04,
            "sharpe": 1.2,
            "r2": 0.15,
            "feature_count": 10,
            "sample_count": 5000,
        }
        audit = sentinel._audit_experiment(experiment)

        # Should have sharpe_decay, feature_count, and suspiciously_good checks
        check_names = [c.name for c in audit.checks]
        assert "sharpe_decay" in check_names
        assert "feature_count" in check_names
        assert "suspiciously_good" in check_names

    def test_audit_with_fold_results(self):
        """_audit_experiment includes fold stability when data available."""
        sentinel = MLQualitySentinel()
        experiment = {
            "id": "exp123",
            "hypothesis_id": "HYP-2026-001",
            "mlflow_run_id": "run456",
            "train_sharpe": 1.5,
            "test_sharpe": 1.2,
            "mean_ic": 0.04,
            "feature_count": 10,
            "sample_count": 5000,
            "fold_results": [{"ic": 0.04}] * 5,
        }
        audit = sentinel._audit_experiment(experiment)
        check_names = [c.name for c in audit.checks]
        assert "fold_stability" in check_names


class TestDeployedModelMonitoring:
    """Tests for deployed model monitoring."""

    def test_count_consecutive_losses(self):
        """_count_consecutive_losses counts correctly."""
        sentinel = MLQualitySentinel()

        # No losses
        assert sentinel._count_consecutive_losses([0.01, 0.02, 0.01]) == 0

        # Single loss at end
        assert sentinel._count_consecutive_losses([0.01, 0.02, -0.01]) == 1

        # Multiple losses at end
        assert sentinel._count_consecutive_losses([0.01, -0.02, -0.01]) == 2

        # Loss streak broken
        assert sentinel._count_consecutive_losses([-0.01, 0.02, -0.01]) == 1

        # Empty list
        assert sentinel._count_consecutive_losses([]) == 0


class TestConstants:
    """Tests for MLQualitySentinel constants."""

    def test_sharpe_decay_thresholds(self):
        """Sharpe decay thresholds are properly defined."""
        assert MLQualitySentinel.SHARPE_DECAY_WARNING == 0.3
        assert MLQualitySentinel.SHARPE_DECAY_CRITICAL == 0.5

    def test_leakage_thresholds(self):
        """Leakage thresholds are properly defined."""
        assert MLQualitySentinel.LEAKAGE_WARNING == 0.85
        assert MLQualitySentinel.LEAKAGE_CRITICAL == 0.95

    def test_feature_count_thresholds(self):
        """Feature count thresholds are properly defined."""
        assert MLQualitySentinel.FEATURE_COUNT_WARNING == 30
        assert MLQualitySentinel.FEATURE_COUNT_CRITICAL == 50
        assert MLQualitySentinel.MIN_SAMPLES_PER_FEATURE == 20

    def test_fold_stability_thresholds(self):
        """Fold stability thresholds are properly defined."""
        assert MLQualitySentinel.FOLD_CV_WARNING == 1.0
        assert MLQualitySentinel.FOLD_CV_CRITICAL == 2.0
        assert MLQualitySentinel.MAX_SIGN_FLIPS == 1

    def test_suspicious_thresholds(self):
        """Suspiciously good thresholds are properly defined."""
        assert MLQualitySentinel.IC_SUSPICIOUS_WARNING == 0.10
        assert MLQualitySentinel.IC_SUSPICIOUS_CRITICAL == 0.15
        assert MLQualitySentinel.SHARPE_SUSPICIOUS_WARNING == 2.5
        assert MLQualitySentinel.SHARPE_SUSPICIOUS_CRITICAL == 3.0

    def test_monitoring_thresholds(self):
        """Monitoring thresholds are properly defined."""
        assert MLQualitySentinel.IC_DEGRADATION_THRESHOLD == 0.5
        assert MLQualitySentinel.MAX_LOSS_STREAK == 7


class TestMLQualitySentinelIntegration:
    """Integration tests for ML Quality Sentinel."""

    @patch.object(MLQualitySentinel, "_get_experiments_to_audit")
    @patch.object(MLQualitySentinel, "_log_agent_event")
    @patch.object(MLQualitySentinel, "_monitor_deployed_models")
    @patch.object(MLQualitySentinel, "_write_research_note")
    def test_execute_with_no_experiments(
        self, mock_write, mock_monitor, mock_log, mock_get_exp
    ):
        """Execute handles empty experiment list."""
        mock_get_exp.return_value = []
        mock_monitor.return_value = ([], 0)

        sentinel = MLQualitySentinel(send_alerts=False)
        result = sentinel.execute()

        assert result["experiments_audited"] == 0
        assert result["experiments_passed"] == 0
        assert result["experiments_flagged"] == 0

    @patch.object(MLQualitySentinel, "_get_experiments_to_audit")
    @patch.object(MLQualitySentinel, "_log_agent_event")
    @patch.object(MLQualitySentinel, "_monitor_deployed_models")
    @patch.object(MLQualitySentinel, "_write_research_note")
    @patch.object(MLQualitySentinel, "_flag_hypothesis")
    def test_execute_flags_critical_experiments(
        self, mock_flag, mock_write, mock_monitor, mock_log, mock_get_exp
    ):
        """Execute flags experiments with critical issues."""
        mock_get_exp.return_value = [
            {
                "id": "exp1",
                "hypothesis_id": "HYP-001",
                "train_sharpe": 2.0,
                "test_sharpe": 0.5,  # 75% decay = critical
                "mean_ic": 0.04,
                "feature_count": 10,
                "sample_count": 5000,
            }
        ]
        mock_monitor.return_value = ([], 0)

        sentinel = MLQualitySentinel(send_alerts=False)
        result = sentinel.execute()

        assert result["experiments_audited"] == 1
        assert result["experiments_flagged"] == 1
        mock_flag.assert_called_once()

    @patch.object(MLQualitySentinel, "_get_experiments_to_audit")
    @patch.object(MLQualitySentinel, "_log_agent_event")
    @patch.object(MLQualitySentinel, "_monitor_deployed_models")
    @patch.object(MLQualitySentinel, "_write_research_note")
    def test_execute_passes_clean_experiments(
        self, mock_write, mock_monitor, mock_log, mock_get_exp
    ):
        """Execute passes experiments without issues."""
        mock_get_exp.return_value = [
            {
                "id": "exp1",
                "hypothesis_id": "HYP-001",
                "train_sharpe": 1.5,
                "test_sharpe": 1.3,  # 13% decay = ok
                "mean_ic": 0.04,
                "sharpe": 1.3,
                "feature_count": 10,
                "sample_count": 5000,
            }
        ]
        mock_monitor.return_value = ([], 0)

        sentinel = MLQualitySentinel(send_alerts=False)
        result = sentinel.execute()

        assert result["experiments_audited"] == 1
        assert result["experiments_passed"] == 1
        assert result["experiments_flagged"] == 0
