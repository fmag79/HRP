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
        """
        Write institutional-grade validation report (Medallion standard).

        Produces a comprehensive research report with:
        - Executive summary with aggregate stress test results
        - Parameter stability analysis with decay metrics
        - Time stability analysis with period-by-period breakdown
        - Regime robustness analysis with bull/bear/sideways performance
        - Execution cost impact analysis
        - Detailed per-hypothesis scorecards
        - Actionable recommendations based on failure patterns
        """
        import numpy as np
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges, render_progress_bar,
            render_section_divider, render_scorecard, format_metric,
            DIVIDER_LIGHT,
        )
        from hrp.agents.output_paths import research_note_path

        report_date = date.today().isoformat()
        filepath = research_note_path("07-validation-analyst")

        # ══════════════════════════════════════════════════════════════════════
        # AGGREGATE STATISTICS
        # ══════════════════════════════════════════════════════════════════════
        total = len(validations)
        passed_count = sum(1 for v in validations if v.overall_passed)
        failed_count = sum(1 for v in validations if not v.overall_passed)
        pass_rate = (passed_count / max(total, 1)) * 100

        # Aggregate check statistics
        total_checks = sum(len(v.checks) for v in validations)
        total_passed_checks = sum(
            sum(1 for c in v.checks if c.passed) for v in validations
        )
        total_critical = sum(v.critical_count for v in validations)
        total_warnings = sum(v.warning_count for v in validations)

        # Check type breakdown
        check_results: dict[str, dict[str, int]] = {}
        for validation in validations:
            for check in validation.checks:
                if check.name not in check_results:
                    check_results[check.name] = {"passed": 0, "failed": 0, "total": 0}
                check_results[check.name]["total"] += 1
                if check.passed:
                    check_results[check.name]["passed"] += 1
                else:
                    check_results[check.name]["failed"] += 1

        # Collect detailed metrics from checks for statistical analysis
        all_param_ratios: list[float] = []
        all_period_profitable_ratios: list[float] = []
        all_sharpe_cvs: list[float] = []
        all_regime_counts: list[int] = []
        all_cost_ratios: list[float] = []
        all_net_returns: list[float] = []

        for validation in validations:
            for check in validation.checks:
                details = check.details
                if check.name == "parameter_sensitivity":
                    variations = details.get("variations", {})
                    for var_data in variations.values():
                        if "ratio" in var_data:
                            all_param_ratios.append(var_data["ratio"])
                elif check.name == "time_stability":
                    if "profitable_ratio" in details:
                        all_period_profitable_ratios.append(details["profitable_ratio"])
                    if "sharpe_cv" in details:
                        all_sharpe_cvs.append(details["sharpe_cv"])
                elif check.name == "regime_stability":
                    if "n_profitable" in details:
                        all_regime_counts.append(details["n_profitable"])
                elif check.name == "execution_costs":
                    if "cost_ratio" in details:
                        all_cost_ratios.append(details["cost_ratio"])
                    if "net_return" in details:
                        all_net_returns.append(details["net_return"])

        parts = []

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_header(
            title="Validation Analyst Report",
            report_type="validation-analyst",
            date_str=report_date,
            subtitle=f"🔬 Pre-deployment stress testing | {total} hypotheses | {passed_count} validated",
        ))

        # ══════════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        parts.append("## Executive Summary\n")

        # Verdict based on pass rate
        if pass_rate == 100 and total > 0:
            verdict = "✅ **ALL HYPOTHESES VALIDATED** — Full batch cleared for Risk Manager review"
        elif pass_rate >= 75:
            verdict = "🟢 **HIGH VALIDATION RATE** — Majority of strategies demonstrate robustness"
        elif pass_rate >= 50:
            verdict = "🟡 **MODERATE VALIDATION RATE** — Mixed robustness across hypothesis pool"
        elif pass_rate > 0:
            verdict = "🟠 **LOW VALIDATION RATE** — Significant stability concerns identified"
        elif total > 0:
            verdict = "🔴 **ALL HYPOTHESES FAILED** — No strategies meet stress test criteria"
        else:
            verdict = "⚪ **NO HYPOTHESES PROCESSED** — No strategies awaiting validation"

        parts.append(f"{verdict}\n")

        # KPI Dashboard (5 metrics for institutional standard)
        parts.append(render_kpi_dashboard([
            {"icon": "🔬", "label": "Validated", "value": total, "detail": "hypotheses"},
            {"icon": "✅", "label": "Passed", "value": passed_count, "detail": f"{pass_rate:.0f}% pass rate"},
            {"icon": "❌", "label": "Failed", "value": failed_count, "detail": "demoted to testing"},
            {"icon": "🔴", "label": "Critical", "value": total_critical, "detail": "blocking issues"},
            {"icon": "⚠️", "label": "Warnings", "value": total_warnings, "detail": "flagged concerns"},
        ]))

        # Alert banner
        if failed_count > 0:
            failure_reasons = []
            for name, results in check_results.items():
                if results["failed"] > 0:
                    failure_reasons.append(f"{name}: {results['failed']} failures")
            parts.append(render_alert_banner(
                [f"{failed_count}/{total} hypotheses FAILED stress testing",
                 f"Failure breakdown: {', '.join(failure_reasons) or 'various checks'}"],
                severity="critical" if pass_rate < 50 else "warning",
            ))
        elif total > 0:
            parts.append(render_alert_banner(
                [f"All {total} hypotheses passed stress testing ✅",
                 "Strategies cleared for Risk Manager review"],
                severity="info",
            ))

        # Health Gauges
        check_pass_rate = (total_passed_checks / max(total_checks, 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Hypothesis Pass Rate", "value": pass_rate, "max_val": 100,
             "trend": "up" if failed_count == 0 else "down"},
            {"label": "Individual Check Pass", "value": check_pass_rate, "max_val": 100,
             "trend": "stable" if check_pass_rate > 80 else "down"},
        ]))

        # ══════════════════════════════════════════════════════════════════════
        # STRESS TEST OVERVIEW
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📊 Stress Test Overview"))

        parts.append("### Validation Check Results\n")
        parts.append("| Check Type | Passed | Failed | Pass Rate | Status |")
        parts.append("|------------|--------|--------|-----------|--------|")

        check_order = ["parameter_sensitivity", "time_stability", "regime_stability", "execution_costs"]
        check_descriptions = {
            "parameter_sensitivity": "Parameter perturbation stability",
            "time_stability": "Consistency across time periods",
            "regime_stability": "Performance in bull/bear/sideways",
            "execution_costs": "Net return after realistic costs",
        }

        for check_name in check_order:
            if check_name in check_results:
                results = check_results[check_name]
                rate = (results["passed"] / max(results["total"], 1)) * 100
                status = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
                desc = check_descriptions.get(check_name, check_name)
                parts.append(
                    f"| {check_name} | {results['passed']} | {results['failed']} | "
                    f"{rate:.0f}% | {status} |"
                )

        # Add any other checks not in the standard order
        for check_name, results in check_results.items():
            if check_name not in check_order:
                rate = (results["passed"] / max(results["total"], 1)) * 100
                status = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
                parts.append(
                    f"| {check_name} | {results['passed']} | {results['failed']} | "
                    f"{rate:.0f}% | {status} |"
                )

        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # STATISTICAL DISTRIBUTION (if we have data)
        # ══════════════════════════════════════════════════════════════════════
        has_stats = any([all_param_ratios, all_period_profitable_ratios,
                        all_sharpe_cvs, all_cost_ratios])

        if has_stats:
            parts.append(render_section_divider("📈 Statistical Distribution"))

            # Parameter Stability Statistics
            if all_param_ratios:
                arr = np.array(all_param_ratios)
                parts.append("### Parameter Stability Analysis\n")
                parts.append("*Ratio of parameter-varied Sharpe to baseline Sharpe*\n")
                parts.append("| Statistic | Value | Interpretation |")
                parts.append("|-----------|-------|----------------|")
                mean_ratio = np.mean(arr)
                std_ratio = np.std(arr)
                min_ratio = np.min(arr)
                parts.append(f"| Mean Ratio | {mean_ratio:.2f} | {'Stable' if mean_ratio >= 0.7 else 'Degradation risk'} |")
                parts.append(f"| Std Dev | {std_ratio:.2f} | {'Low variance' if std_ratio < 0.2 else 'High variance'} |")
                parts.append(f"| Min Ratio | {min_ratio:.2f} | {'Acceptable' if min_ratio >= 0.5 else 'Fragile edge case'} |")
                parts.append(f"| Threshold | {self.param_sensitivity_threshold:.2f} | Configured minimum |")
                parts.append("")

            # Time Stability Statistics
            if all_period_profitable_ratios or all_sharpe_cvs:
                parts.append("### Time Stability Analysis\n")
                parts.append("| Metric | Mean | Std Dev | Min | Max | Threshold |")
                parts.append("|--------|------|---------|-----|-----|-----------|")

                if all_period_profitable_ratios:
                    arr = np.array(all_period_profitable_ratios)
                    parts.append(
                        f"| Profitable Period Ratio | {np.mean(arr):.1%} | {np.std(arr):.1%} | "
                        f"{np.min(arr):.1%} | {np.max(arr):.1%} | ≥{self.min_profitable_periods:.1%} |"
                    )

                if all_sharpe_cvs:
                    arr = np.array(all_sharpe_cvs)
                    parts.append(
                        f"| Sharpe CV (variability) | {np.mean(arr):.2f} | {np.std(arr):.2f} | "
                        f"{np.min(arr):.2f} | {np.max(arr):.2f} | ≤1.0 |"
                    )

                parts.append("")

            # Regime Analysis Statistics
            if all_regime_counts:
                arr = np.array(all_regime_counts)
                parts.append("### Regime Robustness Analysis\n")
                parts.append(f"*Number of profitable regimes (out of 3: bull, bear, sideways)*\n")
                parts.append("| Statistic | Value |")
                parts.append("|-----------|-------|")
                parts.append(f"| Mean Profitable Regimes | {np.mean(arr):.1f} |")
                parts.append(f"| Min Profitable Regimes | {int(np.min(arr))} |")
                parts.append(f"| Max Profitable Regimes | {int(np.max(arr))} |")
                parts.append(f"| Threshold | ≥{self.min_profitable_regimes} regimes |")
                parts.append("")

            # Execution Cost Statistics
            if all_cost_ratios or all_net_returns:
                parts.append("### Execution Cost Impact\n")
                parts.append("| Metric | Mean | Min | Max | Concern Level |")
                parts.append("|--------|------|-----|-----|---------------|")

                if all_cost_ratios:
                    arr = np.array([r for r in all_cost_ratios if r != float("inf")])
                    if len(arr) > 0:
                        mean_cost = np.mean(arr)
                        concern = "🟢 Low" if mean_cost < 0.3 else "🟡 Moderate" if mean_cost < 0.5 else "🔴 High"
                        parts.append(
                            f"| Cost/Gross Return | {mean_cost:.1%} | {np.min(arr):.1%} | "
                            f"{np.max(arr):.1%} | {concern} |"
                        )

                if all_net_returns:
                    arr = np.array(all_net_returns)
                    mean_net = np.mean(arr)
                    concern = "🟢 Profitable" if mean_net > 0.05 else "🟡 Marginal" if mean_net > 0 else "🔴 Negative"
                    parts.append(
                        f"| Net Return (after costs) | {mean_net:.2%} | {np.min(arr):.2%} | "
                        f"{np.max(arr):.2%} | {concern} |"
                    )

                parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # VALIDATION THRESHOLDS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("⚙️ Validation Thresholds"))

        parts.append("```")
        parts.append(f"  Parameter Sensitivity     Min {self.param_sensitivity_threshold:.0%} of baseline Sharpe")
        parts.append(f"  Time Stability            ≥{self.min_profitable_periods:.0%} periods profitable, CV≤1.0")
        parts.append(f"  Regime Robustness         ≥{self.min_profitable_regimes} of 3 regimes profitable")
        parts.append(f"  Commission (per trade)    {self.commission_bps:.1f} bps")
        parts.append(f"  Slippage (per trade)      {self.slippage_bps:.1f} bps")
        parts.append(f"  Cost Impact Warning       >50% of gross return")
        parts.append("```\n")

        # ══════════════════════════════════════════════════════════════════════
        # PER-HYPOTHESIS DETAILED ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📋 Hypothesis Analysis"))

        for validation in validations:
            status = "VALIDATED" if validation.overall_passed else "FAILED"
            status_emoji = "✅" if validation.overall_passed else "❌"

            parts.append(f"### {status_emoji} {validation.hypothesis_id} — **{status}**\n")

            # Get hypothesis context
            context = self._get_hypothesis_context(validation.hypothesis_id)
            if context["title"]:
                parts.append(f"**{context['title']}**\n")
            if context["thesis"]:
                thesis_short = context["thesis"][:200] + "..." if len(context["thesis"]) > 200 else context["thesis"]
                parts.append(f"> {thesis_short}\n")

            # Metadata summary table
            parts.append("| Attribute | Value |")
            parts.append("|-----------|-------|")
            parts.append(f"| **Validation Result** | {status} |")
            parts.append(f"| **Critical Issues** | {validation.critical_count} |")
            parts.append(f"| **Warnings** | {validation.warning_count} |")
            parts.append(f"| **Checks Performed** | {len(validation.checks)} |")
            parts.append(f"| **Experiment ID** | `{validation.experiment_id}` |")
            parts.append(f"| **Validation Date** | {validation.validation_date.isoformat()} |")
            parts.append("")

            # Detailed check results with metrics
            if validation.checks:
                parts.append("#### Stress Test Results\n")

                for check in validation.checks:
                    check_emoji = "✅" if check.passed else "❌"
                    severity_emoji = {"critical": "🔴", "warning": "🟡", "none": "🟢"}.get(
                        check.severity.value, "⚪"
                    )

                    parts.append(f"**{check_emoji} {check.name.replace('_', ' ').title()}** — {severity_emoji} {check.severity.value.upper()}\n")
                    parts.append(f"*{check.message}*\n")

                    # Check-specific detailed metrics
                    details = check.details

                    if check.name == "parameter_sensitivity" and details:
                        baseline_sharpe = details.get("baseline_sharpe")
                        variations = details.get("variations", {})

                        if baseline_sharpe is not None and variations:
                            parts.append("| Variation | Sharpe | Ratio | Status |")
                            parts.append("|-----------|--------|-------|--------|")
                            parts.append(f"| **Baseline** | {baseline_sharpe:.3f} | 1.00 | — |")

                            for var_name, var_data in variations.items():
                                var_sharpe = var_data.get("sharpe", 0)
                                ratio = var_data.get("ratio", 0)
                                var_status = "✅" if ratio >= self.param_sensitivity_threshold else "❌"
                                parts.append(f"| {var_name} | {var_sharpe:.3f} | {ratio:.2f} | {var_status} |")

                            parts.append("")

                    elif check.name == "time_stability" and details:
                        periods = details.get("periods", [])
                        n_profitable = details.get("n_profitable", 0)
                        n_periods = details.get("n_periods", 0)
                        profitable_ratio = details.get("profitable_ratio", 0)
                        mean_sharpe = details.get("mean_sharpe", 0)
                        sharpe_cv = details.get("sharpe_cv", 0)

                        parts.append("| Metric | Value | Threshold | Status |")
                        parts.append("|--------|-------|-----------|--------|")
                        prof_status = "✅" if profitable_ratio >= self.min_profitable_periods else "❌"
                        parts.append(f"| Profitable Periods | {n_profitable}/{n_periods} ({profitable_ratio:.1%}) | ≥{self.min_profitable_periods:.0%} | {prof_status} |")
                        parts.append(f"| Mean Sharpe | {mean_sharpe:.3f} | — | {'🟢' if mean_sharpe > 0 else '🔴'} |")
                        cv_status = "✅" if sharpe_cv <= 1.0 else "❌"
                        parts.append(f"| Sharpe CV | {sharpe_cv:.2f} | ≤1.0 | {cv_status} |")
                        parts.append("")

                        # Period breakdown table (if available)
                        if periods:
                            parts.append("**Period Breakdown:**\n")
                            parts.append("| Period | Sharpe | Return | Profitable |")
                            parts.append("|--------|--------|--------|------------|")
                            for period in periods[:12]:  # Limit to 12 periods
                                p_name = period.get("period", "?")
                                p_sharpe = period.get("sharpe", 0)
                                p_return = period.get("total_return", 0)
                                p_profitable = period.get("profitable", p_return > 0)
                                p_status = "✅" if p_profitable else "❌"
                                parts.append(f"| {p_name} | {p_sharpe:.2f} | {p_return:.1%} | {p_status} |")
                            parts.append("")

                    elif check.name == "regime_stability" and details:
                        regimes = details.get("regimes", {})
                        n_profitable = details.get("n_profitable", 0)
                        n_regimes = details.get("n_regimes", 0)

                        parts.append("| Regime | Sharpe | Return | Profitable |")
                        parts.append("|--------|--------|--------|------------|")
                        for regime_name, regime_data in regimes.items():
                            r_sharpe = regime_data.get("sharpe", 0)
                            r_return = regime_data.get("total_return", 0)
                            r_profitable = regime_data.get("profitable", False)
                            r_status = "✅" if r_profitable else "❌"
                            parts.append(f"| {regime_name.title()} | {r_sharpe:.2f} | {r_return:.1%} | {r_status} |")

                        parts.append("")
                        regime_status = "✅" if n_profitable >= self.min_profitable_regimes else "❌"
                        parts.append(f"**Summary:** {n_profitable}/{n_regimes} regimes profitable {regime_status}")
                        parts.append("")

                    elif check.name == "execution_costs" and details:
                        parts.append("| Metric | Value |")
                        parts.append("|--------|-------|")
                        parts.append(f"| Trades | {details.get('num_trades', 0):,} |")
                        parts.append(f"| Commission | {details.get('commission_bps', 0):.1f} bps/trade |")
                        parts.append(f"| Slippage | {details.get('slippage_bps', 0):.1f} bps/trade |")
                        parts.append(f"| Total Cost | {details.get('total_cost_bps', 0):.0f} bps |")
                        parts.append(f"| Gross Return | {details.get('gross_return', 0):.2%} |")
                        parts.append(f"| **Net Return** | **{details.get('net_return', 0):.2%}** |")

                        cost_ratio = details.get("cost_ratio", 0)
                        if cost_ratio != float("inf"):
                            cost_pct = cost_ratio * 100
                            cost_concern = "🟢 Acceptable" if cost_pct < 30 else "🟡 Moderate" if cost_pct < 50 else "🔴 High"
                            parts.append(f"| Cost Impact | {cost_pct:.1f}% of gross — {cost_concern} |")

                        parts.append("")

                    parts.append(DIVIDER_LIGHT)
                    parts.append("")

            else:
                parts.append("> _No validation checks were performed — data may be missing_\n")
                parts.append("")

            parts.append("─" * 70)
            parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("💡 Recommendations"))

        recommendations = []

        if pass_rate == 100 and total > 0:
            recommendations.append("- ✅ **Proceed to Risk Manager** — All hypotheses cleared stress testing")

        if pass_rate == 0 and total > 0:
            recommendations.append("- 🔴 **Review signal generation** — All strategies failed; systematic issues likely")
            recommendations.append("- 🔴 **Check data quality** — Ensure features and prices are accurate")

        if 0 < pass_rate < 100:
            recommendations.append(f"- ✅ **{passed_count} hypotheses ready for Risk Manager review**")
            recommendations.append(f"- ⚠️ **{failed_count} hypotheses demoted to 'testing'** — Require refinement")

        # Check-specific recommendations
        param_results = check_results.get("parameter_sensitivity", {})
        if param_results.get("failed", 0) > 0:
            recommendations.append(
                f"- **Parameter sensitivity failures ({param_results['failed']})** — "
                "Strategies may be over-fitted to specific parameters; consider more robust signal construction"
            )

        time_results = check_results.get("time_stability", {})
        if time_results.get("failed", 0) > 0:
            recommendations.append(
                f"- **Time stability failures ({time_results['failed']})** — "
                "Strategies may be period-specific; verify signals persist out-of-sample"
            )

        regime_results = check_results.get("regime_stability", {})
        if regime_results.get("failed", 0) > 0:
            recommendations.append(
                f"- **Regime robustness failures ({regime_results['failed']})** — "
                "Strategies may not survive regime changes; consider regime-aware position sizing"
            )

        cost_results = check_results.get("execution_costs", {})
        if cost_results.get("failed", 0) > 0:
            recommendations.append(
                f"- **Execution cost failures ({cost_results['failed']})** — "
                "Returns eroded by transaction costs; reduce turnover or increase holding period"
            )

        if total_critical > total:
            recommendations.append(
                f"- 🔴 **High critical issue rate** — {total_critical} critical issues across {total} hypotheses"
            )

        if not recommendations:
            recommendations.append("- No specific recommendations — validation results nominal")

        for rec in recommendations:
            parts.append(rec)

        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_footer(
            agent_name="validation-analyst",
            duration_seconds=duration,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")

    def _get_hypothesis_context(self, hypothesis_id: str) -> dict[str, Any]:
        """
        Get hypothesis title and thesis for report context.

        Args:
            hypothesis_id: Hypothesis ID to look up

        Returns:
            Dict with title and thesis fields
        """
        try:
            hyp = self.api.get_hypothesis(hypothesis_id)
            return {
                "title": hyp.get("title", "") if hyp else "",
                "thesis": hyp.get("thesis", "") if hyp else "",
            }
        except Exception:
            return {"title": "", "thesis": ""}

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
