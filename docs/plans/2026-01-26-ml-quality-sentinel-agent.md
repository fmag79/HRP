# Plan: ML Quality Sentinel Agent Implementation

## Overview

Build the ML Quality Sentinel - responsible for automated quality auditing of ML experiments, detecting overfitting, data leakage, and model degradation. Acts as an independent safety net that prevents flawed models from propagating through the pipeline.

**Updated:** 2026-01-26 - Fully aligned with Alpha Researcher infrastructure decisions

---

## Agent Specification: ML Quality Sentinel

### Identity

| Attribute | Value |
|-----------|-------|
| **Name** | ML Quality Sentinel |
| **Actor ID** | `agent:ml-quality-sentinel` |
| **Type** | Custom (deterministic - extends `ResearchAgent`) |
| **Role** | Quality assurance, overfitting detection, leakage detection, model monitoring |
| **Trigger** | Lineage event (after ML Scientist) + MCP on-demand |
| **Upstream** | ML Scientist (produces experiments to audit) |
| **Downstream** | Risk Manager (receives quality flags), Report Generator (summarizes issues) |

### Purpose

Independently audit all ML experiments and deployed models to catch quality issues before they cause harm. The Sentinel operates as an impartial watchdog that:

1. **Prevents bad models** from being validated
2. **Detects degradation** in deployed models
3. **Maintains audit trail** of all quality checks
4. **Alerts immediately** when critical issues found

---

## Core Capabilities

### 1. Experiment Audit Pipeline

**What it does:** Reviews ML experiments completed by ML Scientist for quality issues.

```python
# ML Quality Sentinel audit loop
for experiment in get_recent_experiments():
    audit_result = AuditResult(experiment_id=experiment.id)

    # 1. Sharpe Decay Check
    decay_check = check_sharpe_decay(
        train_sharpe=experiment.train_sharpe,
        test_sharpe=experiment.test_sharpe,
        max_decay_ratio=0.5,
    )
    audit_result.add_check("sharpe_decay", decay_check)

    # 2. Target Leakage Check
    leakage_check = check_target_leakage(
        features=experiment.feature_data,
        target=experiment.target_data,
        correlation_threshold=0.95,
    )
    audit_result.add_check("target_leakage", leakage_check)

    # 3. Feature Count Validation
    feature_check = validate_feature_count(
        feature_count=len(experiment.features),
        sample_count=experiment.sample_count,
        max_threshold=50,
    )
    audit_result.add_check("feature_count", feature_check)

    # 4. Stability Analysis
    stability_check = check_fold_stability(
        fold_results=experiment.fold_results,
        max_cv=2.0,  # Coefficient of variation
    )
    audit_result.add_check("fold_stability", stability_check)

    # 5. Suspiciously Good Check
    suspicion_check = check_suspiciously_good(
        mean_ic=experiment.mean_ic,
        sharpe=experiment.sharpe,
        ic_threshold=0.15,  # IC > 0.15 is suspicious
        sharpe_threshold=3.0,  # Sharpe > 3.0 is suspicious
    )
    audit_result.add_check("suspiciously_good", suspicion_check)

    # Log and alert
    log_audit_result(audit_result)
    if audit_result.has_critical_issues:
        send_alert(audit_result)
        flag_hypothesis(experiment.hypothesis_id, audit_result)
```

**Inputs:**
- MLflow experiment runs from ML Scientist
- Walk-forward validation results
- Feature and target data for leakage checks
- Hypothesis metadata

**Outputs:**
- Audit reports per experiment
- Quality flags on hypotheses
- Alert emails for critical issues
- Lineage events for audit trail

### 2. Quality Checks

#### 2.1 Sharpe Decay Detection

Catches overfitting by comparing train vs test performance.

```python
@dataclass
class SharpeDecayResult:
    passed: bool
    train_sharpe: float
    test_sharpe: float
    decay_ratio: float  # (train - test) / train
    severity: str  # "none", "warning", "critical"
    message: str

def check_sharpe_decay(
    train_sharpe: float,
    test_sharpe: float,
    max_decay_ratio: float = 0.5,
) -> SharpeDecayResult:
    """
    Check for excessive performance degradation from train to test.

    Thresholds:
    - decay_ratio < 0.3: OK (minor expected)
    - 0.3 <= decay_ratio < 0.5: WARNING
    - decay_ratio >= 0.5: CRITICAL (likely overfit)
    """
    if train_sharpe <= 0:
        return SharpeDecayResult(
            passed=True,
            decay_ratio=0.0,
            severity="none",
            message="Train Sharpe non-positive, skip decay check",
        )

    decay_ratio = (train_sharpe - test_sharpe) / train_sharpe

    if decay_ratio >= max_decay_ratio:
        return SharpeDecayResult(
            passed=False,
            decay_ratio=decay_ratio,
            severity="critical",
            message=f"Sharpe decayed {decay_ratio:.1%}: {train_sharpe:.2f} → {test_sharpe:.2f}",
        )
    elif decay_ratio >= 0.3:
        return SharpeDecayResult(
            passed=True,  # Warning but not blocking
            decay_ratio=decay_ratio,
            severity="warning",
            message=f"Moderate Sharpe decay {decay_ratio:.1%}",
        )
    else:
        return SharpeDecayResult(
            passed=True,
            decay_ratio=decay_ratio,
            severity="none",
            message="Sharpe decay within acceptable limits",
        )
```

#### 2.2 Target Leakage Detection

Catches data leakage that would invalidate results.

```python
@dataclass
class LeakageResult:
    passed: bool
    suspicious_features: list[str]
    max_correlation: float
    severity: str
    message: str

def check_target_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    correlation_threshold: float = 0.95,
    warn_threshold: float = 0.85,
) -> LeakageResult:
    """
    Detect features with suspiciously high correlation to target.

    High correlation suggests:
    - Feature contains future information
    - Feature is derived from target
    - Data processing error
    """
    correlations = features.corrwith(target).abs()

    critical_features = correlations[correlations >= correlation_threshold].index.tolist()
    warning_features = correlations[
        (correlations >= warn_threshold) & (correlations < correlation_threshold)
    ].index.tolist()

    if critical_features:
        return LeakageResult(
            passed=False,
            suspicious_features=critical_features,
            max_correlation=correlations.max(),
            severity="critical",
            message=f"Likely leakage in: {critical_features}",
        )
    elif warning_features:
        return LeakageResult(
            passed=True,
            suspicious_features=warning_features,
            max_correlation=correlations.max(),
            severity="warning",
            message=f"High correlation (may be legitimate): {warning_features}",
        )
    else:
        return LeakageResult(
            passed=True,
            suspicious_features=[],
            max_correlation=correlations.max(),
            severity="none",
            message="No leakage detected",
        )
```

#### 2.3 Feature Count Validation

Prevents curse of dimensionality.

```python
@dataclass
class FeatureCountResult:
    passed: bool
    feature_count: int
    sample_count: int
    ratio: float  # samples per feature
    severity: str
    message: str

def validate_feature_count(
    feature_count: int,
    sample_count: int,
    warn_threshold: int = 30,
    max_threshold: int = 50,
    min_samples_per_feature: int = 20,
) -> FeatureCountResult:
    """
    Validate feature count relative to samples.

    Rules:
    - Max 50 features (hard limit)
    - At least 20 samples per feature (rule of thumb)
    - Warning if > 30 features
    """
    ratio = sample_count / feature_count if feature_count > 0 else float('inf')

    if feature_count > max_threshold:
        return FeatureCountResult(
            passed=False,
            feature_count=feature_count,
            sample_count=sample_count,
            ratio=ratio,
            severity="critical",
            message=f"Too many features: {feature_count} > {max_threshold}",
        )
    elif ratio < min_samples_per_feature:
        return FeatureCountResult(
            passed=False,
            feature_count=feature_count,
            sample_count=sample_count,
            ratio=ratio,
            severity="critical",
            message=f"Insufficient samples per feature: {ratio:.1f} < {min_samples_per_feature}",
        )
    elif feature_count > warn_threshold:
        return FeatureCountResult(
            passed=True,
            feature_count=feature_count,
            sample_count=sample_count,
            ratio=ratio,
            severity="warning",
            message=f"High feature count: {feature_count} (consider reduction)",
        )
    else:
        return FeatureCountResult(
            passed=True,
            feature_count=feature_count,
            sample_count=sample_count,
            ratio=ratio,
            severity="none",
            message="Feature count acceptable",
        )
```

#### 2.4 Fold Stability Check

Ensures consistent performance across validation folds.

```python
@dataclass
class FoldStabilityResult:
    passed: bool
    fold_ics: list[float]
    mean_ic: float
    std_ic: float
    cv: float  # Coefficient of variation
    sign_flips: int  # Folds with opposite sign IC
    severity: str
    message: str

def check_fold_stability(
    fold_results: list[dict],
    max_cv: float = 2.0,
    max_sign_flips: int = 1,
) -> FoldStabilityResult:
    """
    Check if model performs consistently across folds.

    Instability signals:
    - High CV: performance varies wildly
    - Sign flips: model sometimes predicts opposite direction
    """
    fold_ics = [f.get('ic', 0) for f in fold_results]

    if not fold_ics:
        return FoldStabilityResult(
            passed=False,
            fold_ics=[],
            mean_ic=0,
            std_ic=0,
            cv=float('inf'),
            sign_flips=0,
            severity="critical",
            message="No fold results to analyze",
        )

    mean_ic = np.mean(fold_ics)
    std_ic = np.std(fold_ics)
    cv = std_ic / abs(mean_ic) if mean_ic != 0 else float('inf')

    # Count sign flips
    positive = sum(1 for ic in fold_ics if ic > 0)
    negative = sum(1 for ic in fold_ics if ic < 0)
    sign_flips = min(positive, negative)

    if cv > max_cv:
        return FoldStabilityResult(
            passed=False,
            fold_ics=fold_ics,
            mean_ic=mean_ic,
            std_ic=std_ic,
            cv=cv,
            sign_flips=sign_flips,
            severity="critical",
            message=f"Unstable across folds: CV={cv:.2f} > {max_cv}",
        )
    elif sign_flips > max_sign_flips:
        return FoldStabilityResult(
            passed=False,
            fold_ics=fold_ics,
            mean_ic=mean_ic,
            std_ic=std_ic,
            cv=cv,
            sign_flips=sign_flips,
            severity="critical",
            message=f"IC sign flips: {sign_flips} folds have opposite sign",
        )
    elif cv > 1.0:
        return FoldStabilityResult(
            passed=True,
            fold_ics=fold_ics,
            mean_ic=mean_ic,
            std_ic=std_ic,
            cv=cv,
            sign_flips=sign_flips,
            severity="warning",
            message=f"Moderate instability: CV={cv:.2f}",
        )
    else:
        return FoldStabilityResult(
            passed=True,
            fold_ics=fold_ics,
            mean_ic=mean_ic,
            std_ic=std_ic,
            cv=cv,
            sign_flips=sign_flips,
            severity="none",
            message="Stable across folds",
        )
```

#### 2.5 Suspiciously Good Detection

Flags results that are "too good to be true."

```python
@dataclass
class SuspicionResult:
    passed: bool
    flags: list[str]
    severity: str
    message: str

def check_suspiciously_good(
    mean_ic: float,
    sharpe: float | None = None,
    r2: float | None = None,
    ic_threshold: float = 0.15,
    sharpe_threshold: float = 3.0,
    r2_threshold: float = 0.5,
) -> SuspicionResult:
    """
    Flag results that exceed reasonable expectations.

    In quantitative finance:
    - IC > 0.15 is extremely rare (Nobel Prize territory)
    - Sharpe > 3.0 sustained is suspicious
    - R² > 0.5 in cross-sectional returns is unlikely

    These usually indicate:
    - Data leakage (most common)
    - Look-ahead bias
    - Survivorship bias
    - Implementation error
    """
    flags = []

    if mean_ic > ic_threshold:
        flags.append(f"IC={mean_ic:.4f} exceeds {ic_threshold} (extremely suspicious)")

    if sharpe is not None and sharpe > sharpe_threshold:
        flags.append(f"Sharpe={sharpe:.2f} exceeds {sharpe_threshold}")

    if r2 is not None and r2 > r2_threshold:
        flags.append(f"R²={r2:.4f} exceeds {r2_threshold}")

    if flags:
        return SuspicionResult(
            passed=False,
            flags=flags,
            severity="critical",
            message="Results too good to be true: " + "; ".join(flags),
        )
    else:
        return SuspicionResult(
            passed=True,
            flags=[],
            severity="none",
            message="Results within plausible range",
        )
```

### 3. Model Monitoring (Deployed Models)

**What it does:** Daily check on models that have been validated and are in paper/live trading.

```python
def monitor_deployed_models(self) -> list[MonitoringAlert]:
    """
    Daily check on deployed model performance.

    Monitors:
    1. IC decay over rolling windows
    2. Regime shifts that invalidate model assumptions
    3. Feature distribution drift
    """
    alerts = []

    for model in get_deployed_models():
        # 1. Rolling IC degradation
        recent_ic = calculate_rolling_ic(model, window=20)  # Last 20 days
        baseline_ic = model.validation_ic

        if recent_ic < baseline_ic * 0.5:  # IC dropped by 50%
            alerts.append(MonitoringAlert(
                model_id=model.id,
                alert_type="ic_degradation",
                severity="critical",
                message=f"IC degraded: {baseline_ic:.4f} → {recent_ic:.4f}",
                recommended_action="Review model, consider suspension",
            ))

        # 2. Feature drift detection
        for feature in model.features:
            drift = calculate_distribution_drift(
                baseline=model.training_distribution[feature],
                current=get_current_distribution(feature),
            )
            if drift > DRIFT_THRESHOLD:
                alerts.append(MonitoringAlert(
                    model_id=model.id,
                    alert_type="feature_drift",
                    severity="warning",
                    message=f"Feature {feature} distribution shifted",
                    recommended_action="Retrain model with recent data",
                ))

        # 3. Consecutive loss days
        recent_returns = get_model_returns(model, days=10)
        loss_streak = count_consecutive_losses(recent_returns)
        if loss_streak >= 7:
            alerts.append(MonitoringAlert(
                model_id=model.id,
                alert_type="loss_streak",
                severity="warning",
                message=f"{loss_streak} consecutive losing days",
                recommended_action="Review market regime compatibility",
            ))

    return alerts
```

### 4. Audit Thresholds Summary

| Check | Warning | Critical | Action |
|-------|---------|----------|--------|
| **Sharpe Decay** | > 30% | > 50% | Flag hypothesis, require review |
| **Target Leakage** | corr > 0.85 | corr > 0.95 | Block validation, investigate |
| **Feature Count** | > 30 | > 50 | Require feature reduction |
| **Samples per Feature** | < 30 | < 20 | Require more data or fewer features |
| **Fold CV** | > 1.0 | > 2.0 | Flag instability |
| **Sign Flips** | > 0 | > 1 | Critical - model unreliable |
| **Suspiciously Good IC** | > 0.10 | > 0.15 | Mandatory leakage investigation |
| **Suspiciously Good Sharpe** | > 2.5 | > 3.0 | Mandatory review |

### 5. Audit Report Structure

```python
@dataclass
class AuditCheck:
    name: str
    passed: bool
    severity: str  # "none", "warning", "critical"
    details: dict
    message: str

@dataclass
class ExperimentAudit:
    experiment_id: str
    hypothesis_id: str
    audit_date: date
    checks: list[AuditCheck]
    overall_passed: bool
    critical_count: int
    warning_count: int
    recommendations: list[str]

    @property
    def has_critical_issues(self) -> bool:
        return self.critical_count > 0

@dataclass
class DailyAuditReport:
    report_date: date
    experiments_audited: int
    experiments_passed: int
    experiments_flagged: int
    critical_issues: list[tuple[str, str]]  # (experiment_id, issue)
    warnings: list[tuple[str, str]]
    models_monitored: int
    model_alerts: list[MonitoringAlert]
```

### 6. Email Notification

```
Subject: [HRP] ML Quality Sentinel - 2 Critical Issues Found

ML Quality Sentinel Daily Report
================================
Date: 2026-01-26
Experiments Audited: 12
Models Monitored: 3

CRITICAL ISSUES (Require Immediate Attention)
┌────────────────┬──────────────────────────────────────────────────┐
│ Experiment     │ Issue                                            │
├────────────────┼──────────────────────────────────────────────────┤
│ HYP-2026-005   │ Target leakage detected: pe_ratio (corr=0.97)   │
│ exp_abc123     │ Sharpe decay 65%: 1.8 → 0.6                     │
└────────────────┴──────────────────────────────────────────────────┘

WARNINGS
┌────────────────┬──────────────────────────────────────────────────┐
│ Experiment     │ Warning                                          │
├────────────────┼──────────────────────────────────────────────────┤
│ HYP-2026-003   │ High feature count (35) - consider reduction    │
│ HYP-2026-004   │ Moderate fold instability (CV=1.3)              │
└────────────────┴──────────────────────────────────────────────────┘

DEPLOYED MODEL MONITORING
┌────────────────┬────────┬────────────────────────────────────────┐
│ Model          │ Status │ Notes                                  │
├────────────────┼────────┼────────────────────────────────────────┤
│ momentum_v2    │ ✅ OK  │ IC=0.031, within expected range       │
│ lowvol_factor  │ ⚠️ WARN│ IC dropped 35% vs baseline            │
│ rsi_reversion  │ ✅ OK  │ 5-day win streak                       │
└────────────────┴────────┴────────────────────────────────────────┘

RECOMMENDATIONS
1. Investigate pe_ratio leakage in HYP-2026-005 before proceeding
2. Consider retraining lowvol_factor with recent data
3. Review exp_abc123 - likely overfit to training data

---
HRP ML Quality Sentinel | Automated Research Agent
```

---

## Implementation Design

### Infrastructure Dependencies

This agent requires infrastructure built for Alpha Researcher (build order: Alpha Researcher → ML Quality Sentinel):

| Component | Location | Purpose |
|-----------|----------|---------|
| Lineage Event Watcher | `hrp/agents/scheduler.py` | Triggers agents on lineage events |
| `register_lineage_trigger()` | `hrp/agents/scheduler.py` | Register event-driven callbacks |
| `EventType.AGENT_RUN_COMPLETE` | `hrp/research/lineage.py` | Agents log when done |
| `EventType.ML_QUALITY_SENTINEL_AUDIT` | `hrp/research/lineage.py` | Per-experiment audit events (new) |
| `EventType.HYPOTHESIS_FLAGGED` | `hrp/research/lineage.py` | Flag quality issues (new) |

**Note:** The lineage event watcher is shared infrastructure. Alpha Researcher listens for Signal Scientist events; ML Quality Sentinel listens for ML Scientist events. Same mechanism, different actor filters.

### Idempotent Execution

As a Custom (deterministic) agent, ML Quality Sentinel doesn't need Claude API checkpointing.
However, it is **idempotent**:
- Re-auditing same experiment produces same results
- Duplicate flags not created if already flagged (check before flag)
- Safe to re-run after partial failure

### Class Structure

```python
# In hrp/agents/research_agents.py (extend existing file)

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any
import time
import numpy as np
import pandas as pd

from hrp.agents.jobs import JobStatus
from hrp.risk.overfitting import (
    SharpeDecayMonitor,
    TargetLeakageValidator,
    FeatureCountValidator,
)
from hrp.research.lineage import log_event, EventType


class AuditSeverity(Enum):
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
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == AuditSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == AuditSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        return self.critical_count > 0

    def add_check(self, check: AuditCheck) -> None:
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
    Detects overfitting, leakage, and model degradation.
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
    DRIFT_THRESHOLD = 0.3  # KL divergence threshold
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
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # No job dependency - triggered by lineage events
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

            # Log per-experiment audit event (matches ALPHA_RESEARCHER_REVIEW pattern)
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

        # 5. Write research note (per-run audit report)
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
            return [self._get_experiment(eid) for eid in self.experiment_ids]

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
            experiment_id=experiment.get('id', 'unknown'),
            hypothesis_id=experiment.get('hypothesis_id', 'unknown'),
            mlflow_run_id=experiment.get('mlflow_run_id'),
            audit_date=date.today(),
        )

        # 1. Sharpe Decay Check
        sharpe_check = self._check_sharpe_decay(experiment)
        audit.add_check(sharpe_check)

        # 2. Target Leakage Check (if feature data available)
        if 'features_df' in experiment and 'target' in experiment:
            leakage_check = self._check_target_leakage(
                experiment['features_df'],
                experiment['target'],
            )
            audit.add_check(leakage_check)

        # 3. Feature Count Validation
        feature_check = self._validate_feature_count(experiment)
        audit.add_check(feature_check)

        # 4. Fold Stability Check
        if 'fold_results' in experiment:
            stability_check = self._check_fold_stability(experiment['fold_results'])
            audit.add_check(stability_check)

        # 5. Suspiciously Good Check
        suspicion_check = self._check_suspiciously_good(experiment)
        audit.add_check(suspicion_check)

        return audit

    def _check_sharpe_decay(self, experiment: dict) -> AuditCheck:
        """Check for excessive Sharpe ratio decay."""
        train_sharpe = experiment.get('train_sharpe', 0)
        test_sharpe = experiment.get('test_sharpe', 0)

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
        max_corr = correlations.max()

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
        feature_count = experiment.get('feature_count', 0)
        sample_count = experiment.get('sample_count', 1)
        ratio = sample_count / feature_count if feature_count > 0 else float('inf')

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
        elif ratio < self.MIN_SAMPLES_PER_FEATURE:
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
        fold_ics = [f.get('ic', 0) for f in fold_results]

        if not fold_ics:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={},
                message="No fold results to analyze",
            )

        mean_ic = np.mean(fold_ics)
        std_ic = np.std(fold_ics)
        cv = std_ic / abs(mean_ic) if mean_ic != 0 else float('inf')

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
        mean_ic = experiment.get('mean_ic', 0)
        sharpe = experiment.get('sharpe')
        r2 = experiment.get('r2')

        flags = []

        if mean_ic > self.IC_SUSPICIOUS_CRITICAL:
            flags.append(f"IC={mean_ic:.4f} exceeds {self.IC_SUSPICIOUS_CRITICAL} (extremely suspicious)")
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
            'extremely suspicious' in f or
            f'exceeds {self.SHARPE_SUSPICIOUS_CRITICAL}' in f
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
        alerts = []
        deployed_models = self._get_deployed_models()

        for model in deployed_models:
            # IC degradation check
            baseline_ic = model.get('validation_ic', 0)
            recent_ic = self._calculate_recent_ic(model, window=20)

            if baseline_ic > 0 and recent_ic < baseline_ic * (1 - self.IC_DEGRADATION_THRESHOLD):
                alerts.append(MonitoringAlert(
                    model_id=model['id'],
                    hypothesis_id=model.get('hypothesis_id', 'unknown'),
                    alert_type="ic_degradation",
                    severity=AuditSeverity.CRITICAL,
                    message=f"IC degraded: {baseline_ic:.4f} → {recent_ic:.4f}",
                    recommended_action="Review model, consider suspension",
                ))

            # Loss streak check
            recent_returns = self._get_model_returns(model, days=10)
            loss_streak = self._count_consecutive_losses(recent_returns)

            if loss_streak >= self.MAX_LOSS_STREAK:
                alerts.append(MonitoringAlert(
                    model_id=model['id'],
                    hypothesis_id=model.get('hypothesis_id', 'unknown'),
                    alert_type="loss_streak",
                    severity=AuditSeverity.WARNING,
                    message=f"{loss_streak} consecutive losing days",
                    recommended_action="Review market regime compatibility",
                ))

        return alerts, len(deployed_models)

    def _flag_hypothesis(self, audit: ExperimentAudit) -> None:
        """Flag hypothesis with quality issues."""
        critical_checks = [c for c in audit.checks if c.severity == AuditSeverity.CRITICAL]

        self.api.update_hypothesis(
            hypothesis_id=audit.hypothesis_id,
            metadata={
                'quality_flags': {
                    'flagged_at': datetime.now().isoformat(),
                    'flagged_by': self.ACTOR,
                    'critical_issues': [c.message for c in critical_checks],
                    'audit_id': audit.experiment_id,
                }
            },
            actor=self.ACTOR,
        )

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

    # Helper methods (implementations depend on data access patterns)
    def _get_experiment(self, experiment_id: str) -> dict:
        """Get experiment by ID from MLflow."""
        # Implementation depends on MLflow integration
        pass

    def _get_experiments_for_hypothesis(self, hypothesis_id: str) -> list[dict]:
        """Get all experiments for a hypothesis."""
        pass

    def _get_recent_experiments(self, days: int) -> list[dict]:
        """Get experiments from the last N days."""
        pass

    def _get_deployed_models(self) -> list[dict]:
        """Get all deployed/active models."""
        pass

    def _calculate_recent_ic(self, model: dict, window: int) -> float:
        """Calculate IC over recent window."""
        pass

    def _get_model_returns(self, model: dict, days: int) -> list[float]:
        """Get recent returns for a model."""
        pass

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
        """Write per-run audit report to docs/research/."""
        from pathlib import Path

        report_date = date.today().isoformat()
        filename = f"{report_date}-ml-quality-sentinel.md"
        filepath = Path("docs/research") / filename

        # Build markdown report
        lines = [
            f"# ML Quality Sentinel Report - {report_date}",
            "",
            "## Summary",
            f"- Experiments audited: {len(audits)}",
            f"- Experiments passed: {sum(1 for a in audits if a.overall_passed)}",
            f"- Experiments flagged: {sum(1 for a in audits if a.has_critical_issues)}",
            f"- Critical issues: {len(critical_issues)}",
            f"- Warnings: {len(warnings)}",
            f"- Models monitored: {len(model_alerts)}",
            f"- Duration: {duration:.1f}s",
            "",
        ]

        if critical_issues:
            lines.extend([
                "## Critical Issues (Require Attention)",
                "",
                "| Experiment | Issue |",
                "|------------|-------|",
            ])
            for exp_id, issue in critical_issues:
                lines.append(f"| {exp_id} | {issue} |")
            lines.append("")

        if warnings:
            lines.extend([
                "## Warnings",
                "",
                "| Experiment | Warning |",
                "|------------|---------|",
            ])
            for exp_id, warning in warnings:
                lines.append(f"| {exp_id} | {warning} |")
            lines.append("")

        if model_alerts:
            lines.extend([
                "## Model Monitoring Alerts",
                "",
            ])
            for alert in model_alerts:
                lines.extend([
                    f"### {alert.model_id}",
                    f"- **Type:** {alert.alert_type}",
                    f"- **Severity:** {alert.severity.value}",
                    f"- **Message:** {alert.message}",
                    f"- **Recommended Action:** {alert.recommended_action}",
                    "",
                ])

        lines.extend([
            "---",
            f"*Generated by ML Quality Sentinel ({self.ACTOR})*",
        ])

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(lines))

    def _send_alert_email(
        self,
        audits: list[ExperimentAudit],
        model_alerts: list[MonitoringAlert],
        critical_issues: list[tuple[str, str]],
        warnings: list[tuple[str, str]],
    ) -> None:
        """Send alert email for quality issues."""
        # Implementation uses hrp/notifications/email.py
        pass
```

### Trigger Model

**Primary: Lineage Event Trigger**
- ML Scientist logs `AGENT_RUN_COMPLETE` event to lineage
- Scheduler's event watcher detects new event with `actor='agent:ml-scientist'`
- Triggers ML Quality Sentinel automatically
- Audits experiments from the completed ML Scientist run

**Secondary: MCP On-Demand**
- `run_ml_quality_sentinel(experiment_id?, hypothesis_id?, include_monitoring?)`
- For ad-hoc audits or re-audits

**Tertiary: Scheduled Model Monitoring**
- Daily cron job for deployed model monitoring (independent of ML Scientist)

### Scheduler Integration

```python
# In hrp/agents/scheduler.py

def setup_quality_sentinel(self) -> None:
    """
    Configure ML Quality Sentinel triggers.

    Two trigger modes:
    1. Lineage event: Runs after ML Scientist completes
    2. Scheduled: Daily model monitoring
    """
    from hrp.agents.research_agents import MLQualitySentinel

    # Register for lineage event trigger
    # (Uses shared lineage event watcher infrastructure)
    self.register_lineage_trigger(
        event_type="AGENT_RUN_COMPLETE",
        actor_filter="agent:ml-scientist",
        callback=self._trigger_quality_sentinel_audit,
    )

    # Scheduled model monitoring (6:30 PM ET daily)
    monitor_agent = MLQualitySentinel(
        audit_window_days=0,  # Skip experiment audit
        include_monitoring=True,
        send_alerts=True,
    )

    self.scheduler.add_job(
        monitor_agent.run,
        trigger=CronTrigger(
            hour=18,
            minute=30,
            timezone=self.timezone,
        ),
        id="ml_quality_sentinel_monitoring",
        name="ML Quality Sentinel - Model Monitoring",
        replace_existing=True,
    )

def _trigger_quality_sentinel_audit(self, event: dict) -> None:
    """Callback when ML Scientist completes."""
    from hrp.agents.research_agents import MLQualitySentinel

    # Extract hypothesis IDs from ML Scientist's completion event
    hypothesis_ids = event.get('details', {}).get('hypotheses_processed', [])

    agent = MLQualitySentinel(
        hypothesis_ids=hypothesis_ids if hypothesis_ids else None,
        audit_window_days=1,
        include_monitoring=False,
        send_alerts=True,
    )

    agent.run()
```

### MCP Integration

```python
# In hrp/mcp/research_server.py

@mcp.tool()
def run_ml_quality_sentinel(
    experiment_id: str | None = None,
    hypothesis_id: str | None = None,
    include_monitoring: bool = False,
) -> dict[str, Any]:
    """
    Run ML Quality Sentinel audit.

    Args:
        experiment_id: Optional specific experiment to audit
        hypothesis_id: Optional hypothesis (audits all its experiments)
        include_monitoring: Whether to include deployed model monitoring

    Returns:
        Audit results including all checks and any issues found

    Examples:
        # Audit specific experiment
        run_ml_quality_sentinel(experiment_id="exp_abc123")

        # Audit all experiments for a hypothesis
        run_ml_quality_sentinel(hypothesis_id="HYP-2026-042")

        # Run full audit with model monitoring
        run_ml_quality_sentinel(include_monitoring=True)

        # Just check deployed models
        run_ml_quality_sentinel(include_monitoring=True)
    """
    from hrp.agents.research_agents import MLQualitySentinel

    agent = MLQualitySentinel(
        experiment_ids=[experiment_id] if experiment_id else None,
        hypothesis_ids=[hypothesis_id] if hypothesis_id else None,
        audit_window_days=0 if (experiment_id or hypothesis_id) else 1,
        include_monitoring=include_monitoring,
        send_alerts=True,
    )

    return agent.run()
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `hrp/agents/research_agents.py` | MODIFY | Add `MLQualitySentinel`, `ExperimentAudit`, `AuditCheck`, etc. |
| `hrp/agents/scheduler.py` | MODIFY | Add `setup_quality_sentinel()` method (uses shared lineage event watcher) |
| `hrp/agents/__init__.py` | MODIFY | Export `MLQualitySentinel` and related classes |
| `hrp/mcp/research_server.py` | MODIFY | Add `run_ml_quality_sentinel()` tool (consolidated - consistent with Alpha Researcher) |
| `hrp/research/lineage.py` | MODIFY | Add `ML_QUALITY_SENTINEL_AUDIT` to EventType enum |
| `tests/test_agents/test_ml_quality_sentinel.py` | CREATE | Unit tests for all quality checks |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_agents/test_ml_quality_sentinel.py

class TestMLQualitySentinelInit:
    def test_default_initialization(self):
        """MLQualitySentinel initializes with default thresholds."""

    def test_custom_thresholds(self):
        """Can override default thresholds."""

    def test_audit_window_configuration(self):
        """Audit window properly filters experiments."""


class TestSharpeDecayCheck:
    def test_no_decay_passes(self):
        """No decay passes check."""

    def test_moderate_decay_warning(self):
        """30-50% decay triggers warning."""

    def test_critical_decay_fails(self):
        """50%+ decay triggers critical failure."""

    def test_negative_train_sharpe_skipped(self):
        """Non-positive train Sharpe skips check."""


class TestTargetLeakageCheck:
    def test_no_leakage_passes(self):
        """Low correlations pass check."""

    def test_warning_correlation(self):
        """85-95% correlation triggers warning."""

    def test_critical_leakage_fails(self):
        """95%+ correlation triggers critical failure."""

    def test_identifies_suspicious_features(self):
        """Returns list of suspicious feature names."""


class TestFeatureCountValidation:
    def test_acceptable_count_passes(self):
        """Reasonable feature count passes."""

    def test_high_count_warning(self):
        """30-50 features triggers warning."""

    def test_excessive_count_fails(self):
        """50+ features triggers critical failure."""

    def test_insufficient_samples_fails(self):
        """Less than 20 samples per feature fails."""


class TestFoldStabilityCheck:
    def test_stable_folds_pass(self):
        """Consistent IC across folds passes."""

    def test_high_cv_fails(self):
        """CV > 2.0 triggers critical failure."""

    def test_sign_flips_fail(self):
        """Multiple sign flips trigger critical failure."""

    def test_moderate_cv_warning(self):
        """CV 1.0-2.0 triggers warning."""


class TestSuspiciouslyGoodCheck:
    def test_normal_results_pass(self):
        """Normal IC/Sharpe pass check."""

    def test_suspicious_ic_warning(self):
        """IC 0.10-0.15 triggers warning."""

    def test_suspicious_ic_critical(self):
        """IC > 0.15 triggers critical failure."""

    def test_suspicious_sharpe_critical(self):
        """Sharpe > 3.0 triggers critical failure."""


class TestModelMonitoring:
    def test_healthy_model_no_alerts(self):
        """Healthy model generates no alerts."""

    def test_ic_degradation_alert(self):
        """50%+ IC drop generates alert."""

    def test_loss_streak_alert(self):
        """7+ day loss streak generates alert."""


class TestHypothesisFlagging:
    def test_flags_on_critical_issues(self):
        """Hypothesis flagged when critical issues found."""

    def test_no_flag_on_warnings_only(self):
        """Hypothesis not flagged for warnings only."""

    def test_flag_includes_issue_details(self):
        """Flag metadata includes issue descriptions."""


class TestAuditReport:
    def test_report_aggregates_correctly(self):
        """Report counts match individual audits."""

    def test_critical_issues_collected(self):
        """All critical issues appear in report."""


class TestEmailNotification:
    def test_alert_sent_on_critical(self):
        """Email sent when critical issues found."""

    def test_no_alert_when_all_pass(self):
        """No email when all checks pass."""
```

### Integration Tests

```python
class TestMLQualitySentinelIntegration:
    def test_full_audit_pipeline(self):
        """End-to-end: fetch experiments -> audit -> flag."""

    def test_scheduler_triggers_correctly(self):
        """Sentinel runs on schedule."""

    def test_mcp_tool_invocation(self):
        """MCP tools trigger audits correctly."""

    def test_lineage_tracking(self):
        """Agent events logged to lineage table."""

    def test_works_with_ml_scientist_output(self):
        """Can audit experiments from ML Scientist."""
```

---

## Verification Plan

1. **Unit tests pass:** `pytest tests/test_agents/test_ml_quality_sentinel.py -v`
2. **Create test experiments:** Create experiments with known issues
3. **Run Sentinel:** Execute on test experiments, verify issues detected
4. **Verify flagging:** Check hypothesis flagged correctly
5. **Verify alerts:** Check email sent for critical issues
6. **Verify lineage:** Check agent events in lineage table
7. **Full test suite:** `pytest tests/ -v` (all tests pass)

---

## Integration with Existing Infrastructure

### Overfitting Guards

Leverages existing `hrp/risk/overfitting.py`:
- `SharpeDecayMonitor` - Existing implementation
- `TargetLeakageValidator` - Existing implementation
- `FeatureCountValidator` - Existing implementation

### MLflow Integration

Uses existing `hrp/research/mlflow_utils.py`:
- Query experiments and runs
- Extract metrics and parameters
- Access artifacts

### Email Notifications

Uses existing `hrp/notifications/email.py`:
- Send alert emails
- Format HTML tables

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Trigger mechanism | Lineage event | Consistent with Alpha Researcher; uses shared event watcher |
| Run after ML Scientist | Listens for `AGENT_RUN_COMPLETE` from `agent:ml-scientist` | Audit what ML Scientist produces |
| MCP tool | Single consolidated tool | Consistent with `run_alpha_researcher()` pattern |
| Per-experiment audit event | `ML_QUALITY_SENTINEL_AUDIT` | Matches `ALPHA_RESEARCHER_REVIEW` pattern |
| Research notes output | `docs/research/YYYY-MM-DD-ml-quality-sentinel.md` | Consistent with Alpha Researcher |
| Separate audit vs monitoring | Two trigger modes | Different triggers and purposes |
| Flag hypothesis (not reject) | Conservative | Human review for final decisions |
| Critical = blocking | Fail fast | Prevent bad models from propagating |
| Warning = non-blocking | Informative | Awareness without halting pipeline |
| Email on critical only | Reduce noise | Only actionable alerts |

---

## Future Enhancements

1. **Automated remediation:** Suggest specific fixes for common issues
2. **Historical trend analysis:** Track quality metrics over time
3. **Comparative audit:** Compare experiments across hypotheses
4. **Custom check plugins:** Allow user-defined quality checks
5. **Dashboard integration:** Visual quality monitoring panel
6. **Slack integration:** Real-time alerts to Slack channel
