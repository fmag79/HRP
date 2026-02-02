"""
ML Quality Sentinel agent for experiment auditing and model monitoring.

Independent quality auditor that detects overfitting, leakage, and
model degradation. Acts as an impartial watchdog.
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType


class AuditSeverity(Enum):
    """Severity level for audit checks."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    severity: AuditSeverity
    details: dict[str, Any]
    message: str


@dataclass
class ExperimentAudit:
    """Complete audit of a single experiment."""

    experiment_id: str
    hypothesis_id: str
    mlflow_run_id: str | None
    audit_date: date
    checks: list[AuditCheck] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        """Count of critical severity checks."""
        return sum(1 for c in self.checks if c.severity == AuditSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count of warning severity checks."""
        return sum(1 for c in self.checks if c.severity == AuditSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found."""
        return self.critical_count > 0

    def add_check(self, check: AuditCheck) -> None:
        """Add a check result to the audit."""
        self.checks.append(check)


@dataclass
class MonitoringAlert:
    """Alert from deployed model monitoring."""

    model_id: str
    hypothesis_id: str
    alert_type: str  # ic_degradation, feature_drift, loss_streak
    severity: AuditSeverity
    message: str
    recommended_action: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualitySentinelReport:
    """Complete Sentinel run report."""

    report_date: date
    experiments_audited: int
    experiments_passed: int
    experiments_flagged: int
    critical_issues: list[tuple[str, str]]  # (experiment_id, issue)
    warnings: list[tuple[str, str]]
    models_monitored: int
    model_alerts: list[MonitoringAlert]
    duration_seconds: float


class MLQualitySentinel(ResearchAgent):
    """
    Independent quality auditor for ML experiments and deployed models.

    Detects overfitting, leakage, and model degradation. Acts as an
    impartial watchdog that prevents bad models from propagating.

    Quality Checks:
    1. Sharpe decay (train vs test) - critical if >50%
    2. Target leakage - critical if correlation >0.95
    3. Feature count - critical if >50 features
    4. Fold stability - critical if CV >2.0 or sign flips
    5. Suspiciously good - critical if IC >0.15 or Sharpe >3.0
    """

    DEFAULT_JOB_ID = "ml_quality_sentinel_audit"
    ACTOR = "agent:ml-quality-sentinel"

    # Sharpe decay thresholds
    SHARPE_DECAY_WARNING = 0.3
    SHARPE_DECAY_CRITICAL = 0.5

    # Leakage thresholds
    LEAKAGE_WARNING = 0.85
    LEAKAGE_CRITICAL = 0.95

    # Feature count thresholds
    FEATURE_COUNT_WARNING = 30
    FEATURE_COUNT_CRITICAL = 50
    MIN_SAMPLES_PER_FEATURE = 20

    # Fold stability thresholds
    FOLD_CV_WARNING = 1.0
    FOLD_CV_CRITICAL = 2.0
    MAX_SIGN_FLIPS = 1

    # Suspiciously good thresholds
    IC_SUSPICIOUS_WARNING = 0.10
    IC_SUSPICIOUS_CRITICAL = 0.15
    SHARPE_SUSPICIOUS_WARNING = 2.5
    SHARPE_SUSPICIOUS_CRITICAL = 3.0

    # Model monitoring thresholds
    IC_DEGRADATION_THRESHOLD = 0.5  # 50% drop from baseline
    MAX_LOSS_STREAK = 7

    def __init__(
        self,
        experiment_ids: list[str] | None = None,
        hypothesis_ids: list[str] | None = None,
        audit_window_days: int = 1,
        include_monitoring: bool = True,
        fail_on_critical: bool = True,
        send_alerts: bool = True,
    ):
        """
        Initialize the ML Quality Sentinel.

        Args:
            experiment_ids: Specific experiments to audit
            hypothesis_ids: Audit experiments for these hypotheses
            audit_window_days: Days of recent experiments to audit
            include_monitoring: Whether to monitor deployed models
            fail_on_critical: Whether to fail job on critical issues
            send_alerts: Whether to send email alerts
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.experiment_ids = experiment_ids
        self.hypothesis_ids = hypothesis_ids
        self.audit_window_days = audit_window_days
        self.include_monitoring = include_monitoring
        self.fail_on_critical = fail_on_critical
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """Run quality audit on experiments and deployed models."""
        start_time = time.time()

        # 1. Get experiments to audit
        experiments = self._get_experiments_to_audit()

        # 2. Audit each experiment
        audits: list[ExperimentAudit] = []
        critical_issues: list[tuple[str, str]] = []
        warnings: list[tuple[str, str]] = []

        for experiment in experiments:
            audit = self._audit_experiment(experiment)
            audits.append(audit)

            # Collect issues
            for check in audit.checks:
                if check.severity == AuditSeverity.CRITICAL:
                    critical_issues.append((audit.experiment_id, check.message))
                elif check.severity == AuditSeverity.WARNING:
                    warnings.append((audit.experiment_id, check.message))

            # Log per-experiment audit event
            self._log_agent_event(
                event_type=EventType.ML_QUALITY_SENTINEL_AUDIT,
                hypothesis_id=audit.hypothesis_id,
                details={
                    "experiment_id": audit.experiment_id,
                    "mlflow_run_id": audit.mlflow_run_id,
                    "overall_passed": audit.overall_passed,
                    "critical_count": audit.critical_count,
                    "warning_count": audit.warning_count,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "severity": c.severity.value}
                        for c in audit.checks
                    ],
                },
            )

            # Flag hypothesis if critical issues found
            if audit.has_critical_issues:
                self._flag_hypothesis(audit)

        # 3. Monitor deployed models
        model_alerts: list[MonitoringAlert] = []
        models_monitored = 0

        if self.include_monitoring:
            model_alerts, models_monitored = self._monitor_deployed_models()

        # 4. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "experiments_audited": len(audits),
                "experiments_passed": sum(1 for a in audits if a.overall_passed),
                "experiments_flagged": sum(1 for a in audits if a.has_critical_issues),
                "critical_issues": len(critical_issues),
                "warnings": len(warnings),
                "models_monitored": models_monitored,
                "model_alerts": len(model_alerts),
                "duration_seconds": duration,
            },
        )

        # 5. Write research note
        self._write_research_note(audits, model_alerts, critical_issues, warnings, duration)

        # 6. Send email notification
        if self.send_alerts and (critical_issues or model_alerts):
            self._send_alert_email(audits, model_alerts, critical_issues, warnings)

        # 7. Build report
        report = QualitySentinelReport(
            report_date=date.today(),
            experiments_audited=len(audits),
            experiments_passed=sum(1 for a in audits if a.overall_passed),
            experiments_flagged=sum(1 for a in audits if a.has_critical_issues),
            critical_issues=critical_issues,
            warnings=warnings,
            models_monitored=models_monitored,
            model_alerts=model_alerts,
            duration_seconds=duration,
        )

        return {
            "report_date": report.report_date.isoformat(),
            "experiments_audited": report.experiments_audited,
            "experiments_passed": report.experiments_passed,
            "experiments_flagged": report.experiments_flagged,
            "critical_issues_count": len(report.critical_issues),
            "warnings_count": len(report.warnings),
            "models_monitored": report.models_monitored,
            "model_alerts_count": len(report.model_alerts),
            "duration_seconds": report.duration_seconds,
        }

    def _get_experiments_to_audit(self) -> list[dict]:
        """Get experiments from the audit window."""
        if self.experiment_ids:
            return [self._get_experiment(eid) for eid in self.experiment_ids if eid]

        if self.hypothesis_ids:
            experiments = []
            for hid in self.hypothesis_ids:
                experiments.extend(self._get_experiments_for_hypothesis(hid))
            return experiments

        # Default: get experiments from last N days
        return self._get_recent_experiments(days=self.audit_window_days)

    def _audit_experiment(self, experiment: dict) -> ExperimentAudit:
        """Run all quality checks on an experiment."""
        audit = ExperimentAudit(
            experiment_id=experiment.get("id", "unknown"),
            hypothesis_id=experiment.get("hypothesis_id", "unknown"),
            mlflow_run_id=experiment.get("mlflow_run_id"),
            audit_date=date.today(),
        )

        # 1. Sharpe Decay Check
        sharpe_check = self._check_sharpe_decay(experiment)
        audit.add_check(sharpe_check)

        # 2. Target Leakage Check (if feature data available)
        if "features_df" in experiment and "target" in experiment:
            leakage_check = self._check_target_leakage(
                experiment["features_df"],
                experiment["target"],
            )
            audit.add_check(leakage_check)

        # 3. Feature Count Validation
        feature_check = self._validate_feature_count(experiment)
        audit.add_check(feature_check)

        # 4. Fold Stability Check
        if "fold_results" in experiment and experiment["fold_results"]:
            stability_check = self._check_fold_stability(experiment["fold_results"])
            audit.add_check(stability_check)

        # 5. Suspiciously Good Check
        suspicion_check = self._check_suspiciously_good(experiment)
        audit.add_check(suspicion_check)

        return audit

    def _check_sharpe_decay(self, experiment: dict) -> AuditCheck:
        """Check for excessive Sharpe ratio decay."""
        train_sharpe = experiment.get("train_sharpe", 0)
        test_sharpe = experiment.get("test_sharpe", 0)

        if train_sharpe is None:
            train_sharpe = 0
        if test_sharpe is None:
            test_sharpe = 0

        if train_sharpe <= 0:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.NONE,
                details={"train_sharpe": train_sharpe, "test_sharpe": test_sharpe},
                message="Train Sharpe non-positive, skip decay check",
            )

        decay_ratio = (train_sharpe - test_sharpe) / train_sharpe

        if decay_ratio >= self.SHARPE_DECAY_CRITICAL:
            return AuditCheck(
                name="sharpe_decay",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message=f"Critical Sharpe decay {decay_ratio:.1%}: {train_sharpe:.2f} → {test_sharpe:.2f}",
            )
        elif decay_ratio >= self.SHARPE_DECAY_WARNING:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message=f"Moderate Sharpe decay {decay_ratio:.1%}",
            )
        else:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message="Sharpe decay within acceptable limits",
            )

    def _check_target_leakage(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> AuditCheck:
        """Check for target leakage via correlation."""
        correlations = features.corrwith(target).abs()
        max_corr = float(correlations.max()) if not correlations.empty else 0.0

        critical_features = correlations[correlations >= self.LEAKAGE_CRITICAL].index.tolist()
        warning_features = correlations[
            (correlations >= self.LEAKAGE_WARNING) & (correlations < self.LEAKAGE_CRITICAL)
        ].index.tolist()

        if critical_features:
            return AuditCheck(
                name="target_leakage",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "suspicious_features": critical_features,
                    "max_correlation": max_corr,
                },
                message=f"Likely leakage in: {critical_features}",
            )
        elif warning_features:
            return AuditCheck(
                name="target_leakage",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "suspicious_features": warning_features,
                    "max_correlation": max_corr,
                },
                message=f"High correlation (may be legitimate): {warning_features}",
            )
        else:
            return AuditCheck(
                name="target_leakage",
                passed=True,
                severity=AuditSeverity.NONE,
                details={"max_correlation": max_corr},
                message="No leakage detected",
            )

    def _validate_feature_count(self, experiment: dict) -> AuditCheck:
        """Validate feature count relative to samples."""
        feature_count = experiment.get("feature_count", 0)
        sample_count = experiment.get("sample_count", 1)

        if feature_count is None:
            feature_count = 0
        if sample_count is None or sample_count == 0:
            sample_count = 1

        ratio = sample_count / feature_count if feature_count > 0 else float("inf")

        if feature_count > self.FEATURE_COUNT_CRITICAL:
            return AuditCheck(
                name="feature_count",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"Too many features: {feature_count} > {self.FEATURE_COUNT_CRITICAL}",
            )
        elif ratio < self.MIN_SAMPLES_PER_FEATURE and feature_count > 0:
            return AuditCheck(
                name="feature_count",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"Insufficient samples per feature: {ratio:.1f} < {self.MIN_SAMPLES_PER_FEATURE}",
            )
        elif feature_count > self.FEATURE_COUNT_WARNING:
            return AuditCheck(
                name="feature_count",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"High feature count: {feature_count} (consider reduction)",
            )
        else:
            return AuditCheck(
                name="feature_count",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message="Feature count acceptable",
            )

    def _check_fold_stability(self, fold_results: list[dict]) -> AuditCheck:
        """Check for consistent performance across folds."""
        fold_ics = [f.get("ic", 0) for f in fold_results if f.get("ic") is not None]

        if not fold_ics:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={},
                message="No fold results to analyze",
            )

        mean_ic = float(np.mean(fold_ics))
        std_ic = float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0
        cv = std_ic / abs(mean_ic) if mean_ic != 0 else float("inf")

        positive = sum(1 for ic in fold_ics if ic > 0)
        negative = sum(1 for ic in fold_ics if ic < 0)
        sign_flips = min(positive, negative)

        if cv > self.FOLD_CV_CRITICAL:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"Unstable across folds: CV={cv:.2f} > {self.FOLD_CV_CRITICAL}",
            )
        elif sign_flips > self.MAX_SIGN_FLIPS:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"IC sign flips: {sign_flips} folds have opposite sign",
            )
        elif cv > self.FOLD_CV_WARNING:
            return AuditCheck(
                name="fold_stability",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"Moderate instability: CV={cv:.2f}",
            )
        else:
            return AuditCheck(
                name="fold_stability",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message="Stable across folds",
            )

    def _check_suspiciously_good(self, experiment: dict) -> AuditCheck:
        """Flag results that are too good to be true."""
        mean_ic = experiment.get("mean_ic", 0)
        sharpe = experiment.get("sharpe")
        r2 = experiment.get("r2")

        if mean_ic is None:
            mean_ic = 0

        flags = []

        if mean_ic > self.IC_SUSPICIOUS_CRITICAL:
            flags.append(
                f"IC={mean_ic:.4f} exceeds {self.IC_SUSPICIOUS_CRITICAL} (extremely suspicious)"
            )
        elif mean_ic > self.IC_SUSPICIOUS_WARNING:
            flags.append(f"IC={mean_ic:.4f} exceeds {self.IC_SUSPICIOUS_WARNING} (suspicious)")

        if sharpe is not None and sharpe > self.SHARPE_SUSPICIOUS_CRITICAL:
            flags.append(f"Sharpe={sharpe:.2f} exceeds {self.SHARPE_SUSPICIOUS_CRITICAL}")
        elif sharpe is not None and sharpe > self.SHARPE_SUSPICIOUS_WARNING:
            flags.append(f"Sharpe={sharpe:.2f} exceeds {self.SHARPE_SUSPICIOUS_WARNING}")

        if r2 is not None and r2 > 0.5:
            flags.append(f"R²={r2:.4f} exceeds 0.5")

        # Determine severity based on flags
        has_critical = any(
            "extremely suspicious" in f or f"exceeds {self.SHARPE_SUSPICIOUS_CRITICAL}" in f
            for f in flags
        )

        if flags and has_critical:
            return AuditCheck(
                name="suspiciously_good",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                    "flags": flags,
                },
                message="Results too good to be true: " + "; ".join(flags),
            )
        elif flags:
            return AuditCheck(
                name="suspiciously_good",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                    "flags": flags,
                },
                message="Results warrant review: " + "; ".join(flags),
            )
        else:
            return AuditCheck(
                name="suspiciously_good",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                },
                message="Results within plausible range",
            )

    def _monitor_deployed_models(self) -> tuple[list[MonitoringAlert], int]:
        """Monitor deployed models for degradation."""
        alerts: list[MonitoringAlert] = []
        deployed_models = self._get_deployed_models()

        for model in deployed_models:
            # IC degradation check
            baseline_ic = model.get("validation_ic", 0)
            recent_ic = self._calculate_recent_ic(model, window=20)

            if (
                baseline_ic is not None
                and baseline_ic > 0
                and recent_ic < baseline_ic * (1 - self.IC_DEGRADATION_THRESHOLD)
            ):
                alerts.append(
                    MonitoringAlert(
                        model_id=model.get("id", "unknown"),
                        hypothesis_id=model.get("hypothesis_id", "unknown"),
                        alert_type="ic_degradation",
                        severity=AuditSeverity.CRITICAL,
                        message=f"IC degraded: {baseline_ic:.4f} → {recent_ic:.4f}",
                        recommended_action="Review model, consider suspension",
                    )
                )

            # Loss streak check
            recent_returns = self._get_model_returns(model, days=10)
            loss_streak = self._count_consecutive_losses(recent_returns)

            if loss_streak >= self.MAX_LOSS_STREAK:
                alerts.append(
                    MonitoringAlert(
                        model_id=model.get("id", "unknown"),
                        hypothesis_id=model.get("hypothesis_id", "unknown"),
                        alert_type="loss_streak",
                        severity=AuditSeverity.WARNING,
                        message=f"{loss_streak} consecutive losing days",
                        recommended_action="Review market regime compatibility",
                    )
                )

        return alerts, len(deployed_models)

    def _flag_hypothesis(self, audit: ExperimentAudit) -> None:
        """Flag hypothesis with quality issues."""
        critical_checks = [c for c in audit.checks if c.severity == AuditSeverity.CRITICAL]

        try:
            self.api.update_hypothesis(
                hypothesis_id=audit.hypothesis_id,
                metadata={
                    "quality_flags": {
                        "flagged_at": datetime.now().isoformat(),
                        "flagged_by": self.ACTOR,
                        "critical_issues": [c.message for c in critical_checks],
                        "audit_id": audit.experiment_id,
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to flag hypothesis {audit.hypothesis_id}: {e}")

        # Log to lineage
        self._log_agent_event(
            event_type=EventType.HYPOTHESIS_FLAGGED,
            hypothesis_id=audit.hypothesis_id,
            details={
                "experiment_id": audit.experiment_id,
                "critical_count": audit.critical_count,
                "issues": [c.message for c in critical_checks],
            },
        )

    # ==========================================================================
    # Helper methods (data access)
    # ==========================================================================

    def _get_experiment(self, experiment_id: str) -> dict:
        """Get experiment by ID from MLflow."""
        try:
            return self.api.get_experiment(experiment_id) or {}
        except Exception as e:
            logger.warning(f"Failed to get experiment {experiment_id}: {e}")
            return {"id": experiment_id}

    def _get_experiments_for_hypothesis(self, hypothesis_id: str) -> list[dict]:
        """Get all experiments for a hypothesis."""
        try:
            experiment_ids = self.api.get_experiments_for_hypothesis(hypothesis_id)
            return [self._get_experiment(eid) for eid in experiment_ids]
        except Exception as e:
            logger.warning(f"Failed to get experiments for {hypothesis_id}: {e}")
            return []

    def _get_recent_experiments(self, days: int) -> list[dict]:
        """Get experiments from the last N days."""
        # Query MLflow for recent runs
        try:
            from hrp.research.mlflow_utils import setup_mlflow

            setup_mlflow()
            client = mlflow.tracking.MlflowClient()

            # Get runs from the last N days
            cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Search all walk-forward experiments plus Default
            all_experiments = client.search_experiments()
            all_exp_ids = [exp.experiment_id for exp in all_experiments]

            runs = client.search_runs(
                experiment_ids=all_exp_ids,
                filter_string=f"attributes.start_time > {cutoff_time}",
                max_results=100,
            )

            experiments = []
            for run in runs:
                # Skip nested fold runs (only audit parent walk-forward runs)
                if run.data.tags.get("mlflow.parentRunId"):
                    continue

                exp_dict = {
                    "id": run.info.run_id,
                    "mlflow_run_id": run.info.run_id,
                    "hypothesis_id": run.data.tags.get("hypothesis_id", "unknown"),
                    "train_sharpe": run.data.metrics.get("train_sharpe"),
                    "test_sharpe": run.data.metrics.get("test_sharpe"),
                    "mean_ic": run.data.metrics.get("mean_ic"),
                    "sharpe": run.data.metrics.get("sharpe_ratio"),
                    "r2": run.data.metrics.get("r2"),
                    "feature_count": int(run.data.params.get("feature_count", 0) or 0),
                    "sample_count": int(run.data.params.get("sample_count", 1000) or 1000),
                }

                # Extract fold results if available
                fold_ics = []
                for i in range(10):  # Max 10 folds
                    fold_ic = run.data.metrics.get(f"fold_{i}_ic")
                    if fold_ic is not None:
                        fold_ics.append({"ic": fold_ic})
                if fold_ics:
                    exp_dict["fold_results"] = fold_ics

                experiments.append(exp_dict)

            return experiments

        except Exception as e:
            logger.warning(f"Failed to get recent experiments: {e}")
            return []

    def _get_deployed_models(self) -> list[dict]:
        """Get all deployed/active models."""
        try:
            strategies = self.api.get_deployed_strategies()
            return [
                {
                    "id": s.get("id", "unknown"),
                    "hypothesis_id": s.get("id", "unknown"),
                    "validation_ic": s.get("metadata", {}).get("validation_ic"),
                }
                for s in strategies
            ]
        except Exception as e:
            logger.warning(f"Failed to get deployed models: {e}")
            return []

    def _calculate_recent_ic(self, model: dict, window: int) -> float:
        """Calculate IC over recent window."""
        # Placeholder - would need actual signal/return correlation calculation
        return model.get("validation_ic", 0) or 0

    def _get_model_returns(self, model: dict, days: int) -> list[float]:
        """Get recent returns for a model."""
        # Placeholder - would need actual returns data
        return []

    def _count_consecutive_losses(self, returns: list[float]) -> int:
        """Count consecutive negative returns from end."""
        count = 0
        for r in reversed(returns):
            if r < 0:
                count += 1
            else:
                break
        return count

    def _write_research_note(
        self,
        audits: list[ExperimentAudit],
        model_alerts: list[MonitoringAlert],
        critical_issues: list[tuple[str, str]],
        warnings: list[tuple[str, str]],
        duration: float,
    ) -> None:
        """Write per-run audit report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges,
            render_section_divider,
        )

        from hrp.agents.output_paths import research_note_path

        report_date = date.today().isoformat()
        filepath = research_note_path("04-ml-quality-sentinel")

        passed_count = sum(1 for a in audits if a.overall_passed)
        flagged_count = sum(1 for a in audits if a.has_critical_issues)

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="ML Quality Sentinel Report",
            report_type="ml-quality-sentinel",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "🔬", "label": "Audited", "value": len(audits), "detail": "experiments"},
            {"icon": "✅", "label": "Passed", "value": passed_count, "detail": "clean"},
            {"icon": "🚨", "label": "Flagged", "value": flagged_count, "detail": "issues"},
            {"icon": "📊", "label": "Alerts", "value": len(model_alerts), "detail": "models"},
        ]))

        # ── Alert banner for critical issues ──
        if critical_issues:
            parts.append(render_alert_banner(
                [f"{len(critical_issues)} CRITICAL issues detected — immediate review required",
                 f"{len(warnings)} additional warnings flagged"],
                severity="critical",
            ))
        elif len(audits) > 0 and flagged_count == 0:
            parts.append(render_alert_banner(
                [f"All {len(audits)} experiments passed quality checks ✅"],
                severity="info",
            ))

        # ── Health Gauges ──
        pass_rate = (passed_count / max(len(audits), 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Experiment Quality", "value": pass_rate, "max_val": 100,
             "trend": "up" if flagged_count == 0 else "down"},
            {"label": "Model Monitoring", "value": max(100 - len(model_alerts) * 25, 0), "max_val": 100,
             "trend": "stable" if len(model_alerts) == 0 else "down"},
        ]))

        # ── Critical Issues ──
        if critical_issues:
            parts.append(render_section_divider("🚨 Critical Issues (Require Attention)"))
            parts.append("| Experiment | Issue |")
            parts.append("|------------|-------|")
            for exp_id, issue in critical_issues:
                parts.append(f"| 🔴 {exp_id} | {issue} |")
            parts.append("")

        # ── Warnings ──
        if warnings:
            parts.append(render_section_divider("⚠️ Warnings"))
            parts.append("| Experiment | Warning |")
            parts.append("|------------|---------|")
            for exp_id, warning in warnings:
                parts.append(f"| 🟡 {exp_id} | {warning} |")
            parts.append("")

        # ── Model Monitoring Alerts ──
        if model_alerts:
            parts.append(render_section_divider("📊 Model Monitoring Alerts"))
            for alert in model_alerts:
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    alert.severity.value, "⚪"
                )
                parts.extend([
                    f"### {severity_emoji} {alert.model_id}",
                    "",
                    f"| Field | Detail |",
                    f"|-------|--------|",
                    f"| **Type** | {alert.alert_type} |",
                    f"| **Severity** | {severity_emoji} {alert.severity.value} |",
                    f"| **Message** | {alert.message} |",
                    f"| **Action** | {alert.recommended_action} |",
                    "",
                ])

        # ── Footer ──
        parts.append(render_footer(
            agent_name="ml-quality-sentinel",
            duration_seconds=duration,
        ))

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text("\n".join(parts))
            logger.info(f"Wrote research note to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to write research note: {e}")

    def _send_alert_email(
        self,
        audits: list[ExperimentAudit],
        model_alerts: list[MonitoringAlert],
        critical_issues: list[tuple[str, str]],
        warnings: list[tuple[str, str]],
    ) -> None:
        """Send alert email for quality issues."""
        try:
            notifier = EmailNotifier()

            summary_data = {
                "report_date": date.today().isoformat(),
                "experiments_audited": len(audits),
                "experiments_passed": sum(1 for a in audits if a.overall_passed),
                "experiments_flagged": sum(1 for a in audits if a.has_critical_issues),
                "critical_issues": len(critical_issues),
                "warnings": len(warnings),
                "model_alerts": len(model_alerts),
            }

            # Add critical issues
            for i, (exp_id, issue) in enumerate(critical_issues[:5]):
                summary_data[f"critical_{i+1}"] = f"{exp_id}: {issue}"

            subject = f"[HRP] ML Quality Sentinel - {len(critical_issues)} Critical Issues Found"

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
