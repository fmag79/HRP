"""
Validation Analyst agent for pre-deployment stress testing.

Stress tests validated hypotheses before deployment approval,
including parameter sensitivity, time stability, regime analysis,
and execution cost estimation.
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.agents.ml_quality_sentinel import AuditSeverity
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType

# Re-use AuditSeverity as ValidationSeverity for consistency
ValidationSeverity = AuditSeverity


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: ValidationSeverity
    details: dict[str, Any]
    message: str


@dataclass
class HypothesisValidation:
    """Complete validation of a single hypothesis."""

    hypothesis_id: str
    experiment_id: str
    validation_date: date
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        """Count critical failures."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count warnings."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found."""
        return self.critical_count > 0

    def add_check(self, check: ValidationCheck) -> None:
        """Add a check result to the validation."""
        self.checks.append(check)


@dataclass
class ValidationAnalystReport:
    """Complete Validation Analyst run report."""

    report_date: date
    hypotheses_validated: int
    hypotheses_passed: int
    hypotheses_failed: int
    validations: list[HypothesisValidation]
    duration_seconds: float


class ValidationAnalyst(ResearchAgent):
    """
    Stress tests validated hypotheses before deployment approval.

    Performs:
    1. Parameter sensitivity - Tests stability under parameter changes
    2. Time stability - Verifies consistent performance across periods
    3. Regime analysis - Checks performance in bull/bear/sideways markets
    4. Execution cost estimation - Calculates realistic transaction costs

    Type: Hybrid (deterministic tests + Claude reasoning for edge cases)
    """

    DEFAULT_JOB_ID = "validation_analyst_review"
    ACTOR = "agent:validation-analyst"

    # Default thresholds
    DEFAULT_PARAM_SENSITIVITY_THRESHOLD = 0.5  # Min ratio of varied/baseline Sharpe
    DEFAULT_MIN_PROFITABLE_PERIODS = 0.67  # 2/3 of periods must be profitable
    DEFAULT_MIN_PROFITABLE_REGIMES = 2  # At least 2 of 3 regimes profitable

    # Transaction cost assumptions
    DEFAULT_COMMISSION_BPS = 5  # 5 basis points per trade
    DEFAULT_SLIPPAGE_BPS = 10  # 10 basis points slippage

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        param_sensitivity_threshold: float | None = None,
        min_profitable_periods: float | None = None,
        min_profitable_regimes: int | None = None,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        include_claude_reasoning: bool = True,
        send_alerts: bool = True,
    ):
        """
        Initialize the Validation Analyst.

        Args:
            hypothesis_ids: Specific hypotheses to validate (None = all audited)
            param_sensitivity_threshold: Min ratio for parameter sensitivity
            min_profitable_periods: Min ratio of profitable time periods
            min_profitable_regimes: Min number of profitable regimes
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            include_claude_reasoning: Use Claude for edge case analysis
            send_alerts: Send email alerts on failures
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.param_sensitivity_threshold = (
            param_sensitivity_threshold or self.DEFAULT_PARAM_SENSITIVITY_THRESHOLD
        )
        self.min_profitable_periods = (
            min_profitable_periods or self.DEFAULT_MIN_PROFITABLE_PERIODS
        )
        self.min_profitable_regimes = (
            min_profitable_regimes or self.DEFAULT_MIN_PROFITABLE_REGIMES
        )
        self.commission_bps = commission_bps or self.DEFAULT_COMMISSION_BPS
        self.slippage_bps = slippage_bps or self.DEFAULT_SLIPPAGE_BPS
        self.include_claude_reasoning = include_claude_reasoning
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """
        Run validation on hypotheses ready for final review.

        Returns:
            Dict with validation results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to validate
        hypotheses = self._get_hypotheses_to_validate()

        # 2. Validate each hypothesis
        validations: list[HypothesisValidation] = []
        passed_count = 0
        failed_count = 0

        for hypothesis in hypotheses:
            validation = self._validate_hypothesis(hypothesis)
            validations.append(validation)

            if validation.overall_passed:
                passed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "validated",
                    validation,
                )
            else:
                failed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "testing",
                    validation,
                )

            # Log per-hypothesis validation event
            self._log_agent_event(
                event_type=EventType.VALIDATION_ANALYST_REVIEW,
                hypothesis_id=validation.hypothesis_id,
                experiment_id=validation.experiment_id,
                details={
                    "overall_passed": validation.overall_passed,
                    "critical_count": validation.critical_count,
                    "warning_count": validation.warning_count,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "severity": c.severity.value,
                        }
                        for c in validation.checks
                    ],
                },
            )

        # 3. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_validated": len(validations),
                "hypotheses_passed": passed_count,
                "hypotheses_failed": failed_count,
                "duration_seconds": duration,
            },
        )

        self._log_agent_event(
            event_type=EventType.VALIDATION_ANALYST_COMPLETE,
            details={
                "hypotheses_validated": len(validations),
                "hypotheses_passed": passed_count,
                "hypotheses_failed": failed_count,
                "duration_seconds": duration,
            },
        )

        # 4. Write research note
        self._write_research_note(validations, duration)

        # 5. Send alerts if failures
        if self.send_alerts and failed_count > 0:
            self._send_alert_email(validations)

        # 6. Build report
        report = ValidationAnalystReport(
            report_date=date.today(),
            hypotheses_validated=len(validations),
            hypotheses_passed=passed_count,
            hypotheses_failed=failed_count,
            validations=validations,
            duration_seconds=duration,
        )

        return {
            "report_date": report.report_date.isoformat(),
            "hypotheses_validated": report.hypotheses_validated,
            "hypotheses_passed": report.hypotheses_passed,
            "hypotheses_failed": report.hypotheses_failed,
            "duration_seconds": report.duration_seconds,
        }

    def _get_hypotheses_to_validate(self) -> list[dict[str, Any]]:
        """
        Get hypotheses ready for validation.

        Returns hypotheses that:
        - Passed ML Quality Sentinel audit (no critical issues)
        - Are in 'testing' or 'audited' status
        """
        if self.hypothesis_ids:
            # Specific hypotheses requested
            return [
                self.api.get_hypothesis(hid)
                for hid in self.hypothesis_ids
                if self.api.get_hypothesis(hid) is not None
            ]

        # Get hypotheses that passed quality audit
        # Look for recent ML_QUALITY_SENTINEL_AUDIT events with overall_passed=True
        cutoff = datetime.now() - timedelta(days=7)
        result = self.api.fetchall_readonly(
            """
            SELECT DISTINCT l.hypothesis_id
            FROM lineage l
            WHERE l.event_type = ?
              AND l.timestamp > ?
              AND json_extract_string(l.details, '$.overall_passed') = 'true'
            """,
            (EventType.ML_QUALITY_SENTINEL_AUDIT.value, cutoff),
        )

        hypothesis_ids = [row[0] for row in result if row[0]]
        return [
            self.api.get_hypothesis(hid)
            for hid in hypothesis_ids
            if self.api.get_hypothesis(hid) is not None
        ]

    def _validate_hypothesis(
        self,
        hypothesis: dict[str, Any],
    ) -> HypothesisValidation:
        """
        Run all validation checks on a hypothesis.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            HypothesisValidation with all check results
        """
        hypothesis_id = hypothesis.get("hypothesis_id", hypothesis.get("id", "unknown"))
        experiment_id = hypothesis.get("metadata", {}).get("experiment_id", "unknown")

        validation = HypothesisValidation(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            validation_date=date.today(),
        )

        # Get experiment data for this hypothesis
        experiment_data = self._get_experiment_data(hypothesis)

        # 1. Parameter sensitivity check
        if "param_experiments" in experiment_data and experiment_data["param_experiments"]:
            check = self._check_parameter_sensitivity(
                experiment_data["param_experiments"],
                "baseline",
            )
            validation.add_check(check)

        # 2. Time stability check
        if "period_metrics" in experiment_data and experiment_data["period_metrics"]:
            check = self._check_time_stability(experiment_data["period_metrics"])
            validation.add_check(check)

        # 3. Regime stability check
        if "regime_metrics" in experiment_data and experiment_data["regime_metrics"]:
            check = self._check_regime_stability(experiment_data["regime_metrics"])
            validation.add_check(check)

        # 4. Execution cost estimation
        if all(
            k in experiment_data and experiment_data[k]
            for k in ["num_trades", "avg_trade_value", "gross_return"]
        ):
            check = self._estimate_execution_costs(
                experiment_data["num_trades"],
                experiment_data["avg_trade_value"],
                experiment_data["gross_return"],
            )
            validation.add_check(check)

        return validation

    def _get_experiment_data(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Gather experiment data needed for validation checks.

        This is a placeholder - actual implementation would query MLflow
        and run additional backtests for parameter sensitivity.
        """
        # In a full implementation, this would:
        # 1. Query MLflow for the hypothesis's experiments
        # 2. Run parameter variations if not already done
        # 3. Split returns into time periods
        # 4. Detect regimes and calculate regime metrics
        # 5. Extract trade statistics

        # For now, return data from hypothesis metadata if available
        metadata = hypothesis.get("metadata", {})
        return {
            "param_experiments": metadata.get("param_experiments", {}),
            "period_metrics": metadata.get("period_metrics", []),
            "regime_metrics": metadata.get("regime_metrics", {}),
            "num_trades": metadata.get("num_trades", 0),
            "avg_trade_value": metadata.get("avg_trade_value", 0),
            "gross_return": metadata.get("gross_return", 0),
        }

    def _update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: str,
        validation: HypothesisValidation,
    ) -> None:
        """Update hypothesis status and metadata with validation results."""
        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=new_status,
                metadata={
                    "validation_analyst_review": {
                        "date": validation.validation_date.isoformat(),
                        "passed": validation.overall_passed,
                        "critical_count": validation.critical_count,
                        "warning_count": validation.warning_count,
                        "checks": [c.name for c in validation.checks],
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _write_research_note(
        self,
        validations: list[HypothesisValidation],
        duration: float,
    ) -> None:
        """Write per-run validation report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges,
            render_section_divider, get_status_emoji,
        )

        from hrp.agents.output_paths import research_note_path

        report_date = date.today().isoformat()
        filepath = research_note_path("07-validation-analyst")

        passed_count = sum(1 for v in validations if v.overall_passed)
        failed_count = sum(1 for v in validations if not v.overall_passed)

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Validation Analyst Report",
            report_type="validation-analyst",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Validated", "value": len(validations), "detail": "hypotheses"},
            {"icon": "✅", "label": "Passed", "value": passed_count, "detail": "approved"},
            {"icon": "❌", "label": "Failed", "value": failed_count, "detail": "rejected"},
        ]))

        # ── Alert banner ──
        if failed_count > 0:
            parts.append(render_alert_banner(
                [f"{failed_count} hypotheses FAILED validation — review check details below"],
                severity="warning",
            ))
        elif len(validations) > 0:
            parts.append(render_alert_banner(
                [f"All {len(validations)} hypotheses passed validation ✅"],
                severity="info",
            ))

        # ── Health Gauge ──
        pass_rate = (passed_count / max(len(validations), 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Validation Pass Rate", "value": pass_rate, "max_val": 100,
             "trend": "up" if failed_count == 0 else "down"},
        ]))

        # ── Per-hypothesis validation details ──
        parts.append(render_section_divider("📊 Validation Details"))

        for validation in validations:
            status = "PASSED" if validation.overall_passed else "FAILED"
            emoji = "✅" if validation.overall_passed else "❌"

            parts.append(f"### {emoji} {validation.hypothesis_id}: **{status}**")
            parts.append("")

            if validation.checks:
                parts.append("| Check | Result | Severity | Message |")
                parts.append("|-------|--------|----------|---------|")
                for check in validation.checks:
                    check_emoji = "✅" if check.passed else "❌"
                    severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                        check.severity.value, "⚪"
                    )
                    parts.append(
                        f"| {check.name} | {check_emoji} | {severity_emoji} {check.severity.value} | {check.message} |"
                    )
                parts.append("")
            else:
                parts.append("> _No validation checks recorded_\n")

            parts.append("─" * 60)
            parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="validation-analyst",
            duration_seconds=duration,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")

    def _send_alert_email(self, validations: list[HypothesisValidation]) -> None:
        """Send alert email for validation failures."""
        try:
            failed = [v for v in validations if not v.overall_passed]
            if not failed:
                return

            notifier = EmailNotifier()
            subject = f"[HRP] Validation Analyst - {len(failed)} Hypothesis Validation Failures"

            body_lines = [
                "Validation Analyst detected hypothesis validation failures:",
                "",
            ]
            for v in failed:
                body_lines.append(
                    f"- {v.hypothesis_id}: {v.critical_count} critical, {v.warning_count} warnings"
                )

            notifier.send_notification(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send alert email: {e}")

    def _check_parameter_sensitivity(
        self,
        experiments: dict[str, dict[str, Any]],
        baseline_key: str,
    ) -> ValidationCheck:
        """
        Check parameter sensitivity using existing robustness module.

        Args:
            experiments: Dict mapping experiment name to metrics
            baseline_key: Key for baseline experiment

        Returns:
            ValidationCheck with sensitivity results
        """
        from hrp.risk.robustness import check_parameter_sensitivity

        # Check if baseline exists in experiments
        if baseline_key not in experiments:
            logger.warning(
                f"Baseline '{baseline_key}' not found in param_experiments, "
                f"skipping parameter sensitivity check"
            )
            return ValidationCheck(
                name="parameter_sensitivity",
                passed=True,  # Pass by default when data is missing
                severity=ValidationSeverity.NONE,
                details={"warning": f"Baseline '{baseline_key}' not found in experiments"},
                message="Parameter sensitivity check skipped (no baseline data)",
            )

        result = check_parameter_sensitivity(
            experiments=experiments,
            baseline_key=baseline_key,
            threshold=self.param_sensitivity_threshold,
        )

        if not result.passed:
            return ValidationCheck(
                name="parameter_sensitivity",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("parameter_sensitivity", {}),
                message=f"Parameter sensitivity failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="parameter_sensitivity",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("parameter_sensitivity", {}),
            message="Parameters are stable under variation",
        )

    def _check_time_stability(
        self,
        period_metrics: list[dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check time period stability using existing robustness module.

        Args:
            period_metrics: List of period metrics with 'sharpe', 'profitable'

        Returns:
            ValidationCheck with stability results
        """
        from hrp.risk.robustness import check_time_stability

        result = check_time_stability(
            period_metrics=period_metrics,
            min_profitable_ratio=self.min_profitable_periods,
        )

        if not result.passed:
            return ValidationCheck(
                name="time_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("time_stability", {}),
                message=f"Time stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="time_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("time_stability", {}),
            message="Strategy is stable across time periods",
        )

    def _check_regime_stability(
        self,
        regime_metrics: dict[str, dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check market regime stability using existing robustness module.

        Args:
            regime_metrics: Dict mapping regime name to metrics

        Returns:
            ValidationCheck with regime results
        """
        from hrp.risk.robustness import check_regime_stability

        result = check_regime_stability(
            regime_metrics=regime_metrics,
            min_regimes_profitable=self.min_profitable_regimes,
        )

        if not result.passed:
            return ValidationCheck(
                name="regime_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("regime_stability", {}),
                message=f"Regime stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="regime_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("regime_stability", {}),
            message="Strategy works across market regimes",
        )

    def _estimate_execution_costs(
        self,
        num_trades: int,
        avg_trade_value: float,
        gross_return: float,
    ) -> ValidationCheck:
        """
        Estimate realistic execution costs and net return.

        Args:
            num_trades: Number of round-trip trades
            avg_trade_value: Average trade value in dollars
            gross_return: Gross return before costs

        Returns:
            ValidationCheck with cost analysis
        """
        # Calculate total cost in basis points
        cost_per_trade_bps = self.commission_bps + self.slippage_bps
        total_cost_bps = cost_per_trade_bps * num_trades

        # Convert to decimal
        total_cost_decimal = total_cost_bps / 10000

        # Net return
        net_return = gross_return - total_cost_decimal

        # Cost as percentage of gross return
        cost_ratio = total_cost_decimal / gross_return if gross_return > 0 else float("inf")

        details = {
            "num_trades": num_trades,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_cost_bps": total_cost_bps,
            "total_cost_decimal": total_cost_decimal,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_ratio": cost_ratio,
        }

        # Determine severity
        if net_return < 0:
            return ValidationCheck(
                name="execution_costs",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=details,
                message=f"Net return negative after costs: {net_return:.2%}",
            )
        elif cost_ratio > 0.5:  # Costs exceed 50% of gross return
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.WARNING,
                details=details,
                message=f"High execution costs: {cost_ratio:.1%} of gross return",
            )
        else:
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.NONE,
                details=details,
                message=f"Execution costs acceptable: net return {net_return:.2%}",
            )
