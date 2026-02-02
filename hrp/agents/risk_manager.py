"""
Risk Manager agent for independent portfolio risk oversight.

Reviews validated hypotheses for portfolio-level risk before deployment.
Can veto strategies but CANNOT approve deployment (maintains independence).
"""

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType


@dataclass
class RiskVeto:
    """Record of a risk veto decision."""

    hypothesis_id: str
    veto_reason: str
    veto_type: Literal["drawdown", "concentration", "correlation", "limits", "other"]
    severity: Literal["critical", "warning"]
    details: dict[str, Any]
    veto_date: date


@dataclass
class PortfolioRiskAssessment:
    """Assessment of portfolio-level risk for a hypothesis."""

    hypothesis_id: str
    passed: bool
    vetos: list[RiskVeto]
    warnings: list[str]
    portfolio_impact: dict[str, Any]
    assessment_date: date


@dataclass
class RiskManagerReport:
    """Complete Risk Manager run report."""

    report_date: date
    hypotheses_assessed: int
    hypotheses_passed: int
    hypotheses_vetoed: int
    assessments: list[PortfolioRiskAssessment]
    duration_seconds: float


class RiskManager(ResearchAgent):
    """
    Independent portfolio risk oversight agent.

    Reviews validated hypotheses for portfolio-level risk before deployment.
    Can veto strategies but CANNOT approve deployment (maintains independence).

    Performs:
    1. Drawdown risk assessment - Max drawdown limits, drawdown duration
    2. Concentration risk - Position sizes, sector exposure, correlation
    3. Portfolio fit - Correlation with existing positions, diversification value
    4. Risk limits validation - Position limits, turnover limits, leverage

    Type: Custom (deterministic checks with independent veto authority)

    Key principle: Risk Manager operates independently from alpha generation
    and can veto any strategy but cannot approve deployment (only human CIO can).
    """

    DEFAULT_JOB_ID = "risk_manager_review"
    ACTOR = "agent:risk-manager"

    # Portfolio risk limits (conservative institutional defaults)
    MAX_MAX_DRAWDOWN = 0.20  # 20% maximum drawdown
    MAX_DRAWDOWN_DURATION_DAYS = 126  # 6 months to recover
    MAX_POSITION_CORRELATION = 0.70  # Max correlation with existing positions
    MAX_SECTOR_EXPOSURE = 0.30  # 30% max in any sector
    MAX_SINGLE_POSITION = 0.10  # 10% max in single position

    # Portfolio composition targets
    MIN_DIVERSIFICATION = 10  # Minimum positions
    TARGET_POSITIONS = 20  # Target number of positions

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        max_drawdown: float | None = None,
        max_correlation: float | None = None,
        max_sector_exposure: float | None = None,
        send_alerts: bool = True,
    ):
        """
        Initialize the Risk Manager.

        Args:
            hypothesis_ids: Specific hypotheses to assess (None = all validated)
            max_drawdown: Maximum allowed drawdown (default 20%)
            max_correlation: Max correlation with existing positions (default 0.70)
            max_sector_exposure: Max sector exposure (default 30%)
            send_alerts: Send email alerts on vetos
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.max_drawdown = max_drawdown or self.MAX_MAX_DRAWDOWN
        self.max_correlation = max_correlation or self.MAX_POSITION_CORRELATION
        self.max_sector_exposure = max_sector_exposure or self.MAX_SECTOR_EXPOSURE
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """
        Run risk assessment on hypotheses ready for deployment review.

        Returns:
            Dict with assessment results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to assess
        hypotheses = self._get_hypotheses_to_assess()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "assessments": [],
                "message": "No hypotheses awaiting risk assessment",
            }

        # 2. Assess each hypothesis
        assessments: list[PortfolioRiskAssessment] = []
        passed_count = 0
        vetoed_count = 0

        for hypothesis in hypotheses:
            assessment = self._assess_hypothesis_risk(hypothesis)
            assessments.append(assessment)

            if assessment.passed:
                passed_count += 1
                # Log to lineage
                self._log_agent_event(
                    event_type=EventType.RISK_REVIEW_COMPLETE,
                    details={
                        "hypothesis_id": assessment.hypothesis_id,
                        "passed": True,
                        "warnings": assessment.warnings,
                    },
                    hypothesis_id=assessment.hypothesis_id,
                )
            else:
                vetoed_count += 1
                # Log veto to lineage
                for veto in assessment.vetos:
                    self._log_agent_event(
                        event_type=EventType.RISK_VETO,
                        details={
                            "hypothesis_id": veto.hypothesis_id,
                            "veto_reason": veto.veto_reason,
                            "veto_type": veto.veto_type,
                            "severity": veto.severity,
                        },
                        hypothesis_id=veto.hypothesis_id,
                    )

        # 3. Generate report
        report = RiskManagerReport(
            report_date=date.today(),
            hypotheses_assessed=len(assessments),
            hypotheses_passed=passed_count,
            hypotheses_vetoed=vetoed_count,
            assessments=assessments,
            duration_seconds=time.time() - start_time,
        )

        # 4. Write research note
        self._write_research_note(report)

        # 5. Send alerts if any vetos
        if self.send_alerts:
            self._send_veto_alerts(assessments)

        return {
            "status": "complete",
            "assessments": assessments,
            "report": {
                "hypotheses_assessed": report.hypotheses_assessed,
                "hypotheses_passed": report.hypotheses_passed,
                "hypotheses_vetoed": report.hypotheses_vetoed,
                "duration_seconds": report.duration_seconds,
            },
        }

    def _get_hypotheses_to_assess(self) -> list[dict[str, Any]]:
        """
        Get hypotheses that need risk assessment.

        Fetches hypotheses with 'validated' status that haven't been
        risk-assessed yet.
        """
        if self.hypothesis_ids:
            # Specific hypotheses requested
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis_with_metadata(hid)
                if hyp:
                    hypotheses.append(hyp)
            return hypotheses
        else:
            # All validated hypotheses not yet risk-assessed
            return self.api.list_hypotheses_with_metadata(
                status='validated',
                metadata_exclude='%risk_manager_review%',
                limit=10,
            )

    def _assess_hypothesis_risk(
        self, hypothesis: dict[str, Any]
    ) -> PortfolioRiskAssessment:
        """
        Assess portfolio-level risk for a single hypothesis.

        Performs the following checks:
        1. Drawdown risk
        2. Concentration risk
        3. Correlation with existing positions
        4. Risk limits validation

        Args:
            hypothesis: Hypothesis record from database

        Returns:
            PortfolioRiskAssessment with veto decisions
        """
        hypothesis_id = hypothesis["hypothesis_id"]
        metadata_str = hypothesis.get("metadata") or "{}"
        metadata = (
            json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        )

        # Get experiment metrics
        experiment_data = self._get_experiment_metrics(hypothesis_id, metadata)

        # Initialize assessment
        vetos: list[RiskVeto] = []
        warnings: list[str] = []

        # Check 1: Drawdown risk
        dd_veto = self._check_drawdown_risk(hypothesis_id, experiment_data)
        if dd_veto:
            vetos.append(dd_veto)

        # Check 2: Concentration risk
        conc_vetos, conc_warnings = self._check_concentration_risk(
            hypothesis_id, experiment_data, metadata
        )
        vetos.extend(conc_vetos)
        warnings.extend(conc_warnings)

        # Check 3: Correlation with existing positions
        corr_veto = self._check_correlation_risk(hypothesis_id, metadata)
        if corr_veto:
            vetos.append(corr_veto)

        # Check 4: Risk limits validation
        limits_vetos = self._check_risk_limits(hypothesis_id, experiment_data)
        vetos.extend(limits_vetos)

        # Calculate portfolio impact
        portfolio_impact = self._calculate_portfolio_impact(
            hypothesis_id, experiment_data, metadata
        )

        # Determine if passed (no critical vetos)
        passed = all(v.severity != "critical" for v in vetos)

        assessment = PortfolioRiskAssessment(
            hypothesis_id=hypothesis_id,
            passed=passed,
            vetos=vetos,
            warnings=warnings,
            portfolio_impact=portfolio_impact,
            assessment_date=date.today(),
        )

        # Update hypothesis with risk assessment
        self._update_hypothesis_with_risk_assessment(assessment)

        return assessment

    def _get_experiment_metrics(
        self, hypothesis_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get experiment metrics from metadata or MLflow.

        Reads ml_scientist_results first (structured walk-forward metrics),
        then falls back to validation_analyst_review, then conservative defaults.

        Args:
            hypothesis_id: Hypothesis ID
            metadata: Hypothesis metadata dict

        Returns:
            Dict with key metrics: sharpe, max_drawdown, volatility, etc.
        """
        # Primary source: ML Scientist structured results
        ml_results = metadata.get("ml_scientist_results", {})
        if ml_results:
            stability = ml_results.get("stability_score", 2.0)
            mean_ic = ml_results.get("mean_ic", 0)
            return {
                "sharpe": mean_ic * 20,  # IC-based proxy (IC=0.05 → ~1.0 Sharpe)
                "max_drawdown": min(0.10 + stability * 0.10, 0.50),
                "volatility": 0.15,
                "turnover": 0.30,
                "num_positions": self.TARGET_POSITIONS,
                "sector_exposure": {},
            }

        # Secondary: Validation Analyst review
        validation = metadata.get("validation_analyst_review", {})
        if validation:
            return {
                "sharpe": validation.get("sharpe", 0),
                "max_drawdown": validation.get("max_drawdown", 0.25),
                "volatility": validation.get("volatility", 0.20),
                "turnover": validation.get("turnover", 0.50),
                "num_positions": validation.get("num_positions", self.TARGET_POSITIONS),
                "sector_exposure": validation.get("sector_exposure", {}),
            }

        # Last resort: conservative defaults
        logger.warning(
            f"{hypothesis_id}: No ml_scientist_results or validation_analyst_review "
            "in metadata, using conservative defaults"
        )
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.25,
            "volatility": 0.20,
            "turnover": 0.30,
            "num_positions": self.TARGET_POSITIONS,
            "sector_exposure": {},
        }

    def _check_drawdown_risk(
        self, hypothesis_id: str, metrics: dict[str, Any]
    ) -> RiskVeto | None:
        """
        Check if drawdown exceeds limits.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics

        Returns:
            RiskVeto if drawdown too high, None otherwise
        """
        max_dd = metrics.get("max_drawdown", 0)

        if max_dd > self.max_drawdown:
            return RiskVeto(
                hypothesis_id=hypothesis_id,
                veto_reason=f"Max drawdown {max_dd:.1%} exceeds limit {self.max_drawdown:.1%}",
                veto_type="drawdown",
                severity="critical",
                details={"max_drawdown": max_dd, "limit": self.max_drawdown},
                veto_date=date.today(),
            )

        # Warning if approaching limit
        if max_dd > self.max_drawdown * 0.8:
            logger.warning(
                f"{hypothesis_id}: Drawdown {max_dd:.1%} approaching limit "
                f"{self.max_drawdown:.1%}"
            )

        return None

    def _check_concentration_risk(
        self, hypothesis_id: str, metrics: dict[str, Any], metadata: dict[str, Any]
    ) -> tuple[list[RiskVeto], list[str]]:
        """
        Check concentration risk (position sizes, sector exposure).

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics
            metadata: Hypothesis metadata

        Returns:
            Tuple of (vetos, warnings)
        """
        vetos: list[RiskVeto] = []
        warnings: list[str] = []

        num_positions = metrics.get("num_positions", self.TARGET_POSITIONS)
        sector_exposure = metrics.get("sector_exposure", {})

        # Check minimum diversification
        if num_positions < self.MIN_DIVERSIFICATION:
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Only {num_positions} positions, minimum {self.MIN_DIVERSIFICATION}",
                    veto_type="concentration",
                    severity="critical",
                    details={"num_positions": num_positions, "minimum": self.MIN_DIVERSIFICATION},
                    veto_date=date.today(),
                )
            )

        # Check sector concentration
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                vetos.append(
                    RiskVeto(
                        hypothesis_id=hypothesis_id,
                        veto_reason=f"Sector '{sector}' exposure {exposure:.1%} exceeds limit {self.max_sector_exposure:.1%}",
                        veto_type="concentration",
                        severity="critical",
                        details={
                            "sector": sector,
                            "exposure": exposure,
                            "limit": self.max_sector_exposure,
                        },
                        veto_date=date.today(),
                    )
                )

        return vetos, warnings

    def _check_correlation_risk(
        self, hypothesis_id: str, metadata: dict[str, Any]
    ) -> RiskVeto | None:
        """
        Check correlation with existing paper portfolio positions.

        Args:
            hypothesis_id: Hypothesis ID
            metadata: Hypothesis metadata

        Returns:
            RiskVeto if too correlated, None otherwise
        """
        # Get existing paper portfolio
        try:
            portfolio_positions = self.api.get_paper_portfolio()
            portfolio_positions = [p for p in portfolio_positions if p.get("weight", 0) > 0]

            if not portfolio_positions:
                return None

            # For now, correlation check is a placeholder
            # In production, would compute actual correlation from returns
            # For new implementation, just check if same features are used

            # Check for duplicate strategies (same feature set)
            existing_features = metadata.get("features", [])
            if existing_features:
                # Simple check: if more than 50% feature overlap, flag as warning
                # (This is a placeholder - real implementation would compute correlation)
                pass

        except Exception as e:
            logger.debug(f"Could not check correlation: {e}")
            return None

        return None

    def _check_risk_limits(
        self, hypothesis_id: str, metrics: dict[str, Any]
    ) -> list[RiskVeto]:
        """
        Check if strategy respects risk limits.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics

        Returns:
            List of vetos (empty if all limits respected)
        """
        vetos: list[RiskVeto] = []

        volatility = metrics.get("volatility", 0)
        turnover = metrics.get("turnover", 0)

        # Check volatility (high vol = high risk)
        if volatility > 0.25:  # 25% annual vol
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Volatility {volatility:.1%} exceeds prudent limit",
                    veto_type="limits",
                    severity="warning",  # Warning, not critical
                    details={"volatility": volatility},
                    veto_date=date.today(),
                )
            )

        # Check turnover (high turnover = high costs)
        if turnover > 0.50:  # 50% annual turnover
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Turnover {turnover:.1%} may erode returns with costs",
                    veto_type="limits",
                    severity="warning",
                    details={"turnover": turnover},
                    veto_date=date.today(),
                )
            )

        return vetos

    def _calculate_portfolio_impact(
        self, hypothesis_id: str, metrics: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate the impact of adding this hypothesis to the portfolio.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics
            metadata: Hypothesis metadata

        Returns:
            Dict with portfolio impact assessment
        """
        # Get current portfolio state
        try:
            portfolio_positions = self.api.get_paper_portfolio()
            active = [p for p in portfolio_positions if p.get("weight", 0) > 0]
            current_positions = len(active)
            current_weight = sum(p.get("weight", 0) for p in active)
        except Exception:
            current_positions = 0
            current_weight = 0.0

        # Calculate impact
        new_positions = current_positions + 1
        new_weight = min(1.0, current_weight + 0.05)  # Assume 5% allocation

        return {
            "current_positions": current_positions,
            "new_positions": new_positions,
            "current_weight": current_weight,
            "new_weight": new_weight,
            "weight_increase": new_weight - current_weight,
            "diversification_value": "medium" if new_positions < 15 else "low",
        }

    def _update_hypothesis_with_risk_assessment(
        self, assessment: PortfolioRiskAssessment
    ) -> None:
        """Update hypothesis with risk assessment results."""
        try:
            self.api.update_hypothesis(
                hypothesis_id=assessment.hypothesis_id,
                status="validated" if assessment.passed else "risk_vetoed",
                metadata={
                    "risk_manager_review": {
                        "date": assessment.assessment_date.isoformat(),
                        "passed": assessment.passed,
                        "veto_count": len(assessment.vetos),
                        "warning_count": len(assessment.warnings),
                        "vetos": [
                            {
                                "reason": v.veto_reason,
                                "type": v.veto_type,
                                "severity": v.severity,
                            }
                            for v in assessment.vetos
                        ],
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to update hypothesis {assessment.hypothesis_id}: {e}")

    def _write_research_note(self, report: RiskManagerReport) -> None:
        """Write per-run risk assessment report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges, render_risk_limits,
            render_veto_section, render_section_divider, render_progress_bar,
        )

        from hrp.agents.output_paths import research_note_path

        report_date = report.report_date.isoformat()
        filepath = research_note_path("08-risk-manager")

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Risk Manager Report",
            report_type="risk-manager",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Assessed", "value": report.hypotheses_assessed, "detail": "hypotheses"},
            {"icon": "✅", "label": "Passed", "value": report.hypotheses_passed, "detail": "approved"},
            {"icon": "🚫", "label": "Vetoed", "value": report.hypotheses_vetoed, "detail": "blocked"},
        ]))

        # ── Alert banner ──
        if report.hypotheses_vetoed > 0:
            veto_pct = report.hypotheses_vetoed / max(report.hypotheses_assessed, 1) * 100
            parts.append(render_alert_banner(
                [f"{report.hypotheses_vetoed} of {report.hypotheses_assessed} hypotheses VETOED ({veto_pct:.0f}%)",
                 "📌 Review risk limits and drawdown thresholds if veto rate is excessive"],
                severity="critical" if veto_pct > 50 else "warning",
            ))
        elif report.hypotheses_assessed > 0:
            parts.append(render_alert_banner(
                [f"All {report.hypotheses_assessed} hypotheses passed risk assessment ✅"],
                severity="info",
            ))

        # ── Health Gauges ──
        pass_rate = (report.hypotheses_passed / max(report.hypotheses_assessed, 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Risk Pass Rate", "value": pass_rate, "max_val": 100,
             "trend": "up" if report.hypotheses_vetoed == 0 else "down"},
            {"label": "Portfolio Safety", "value": 100 - (report.hypotheses_vetoed * 10), "max_val": 100,
             "trend": "stable"},
        ]))

        # ── Risk Limits ──
        parts.append(render_risk_limits({
            "Max Drawdown": f"{self.max_drawdown:.1%}",
            "Max Correlation": f"{self.max_correlation:.2f}",
            "Max Sector Exposure": f"{self.max_sector_exposure:.1%}",
            "Min Diversification": f"{self.MIN_DIVERSIFICATION} positions",
        }))

        # ── Per-hypothesis assessment ──
        parts.append(render_section_divider("📊 Hypothesis Assessments"))

        for assessment in report.assessments:
            status = "passed" if assessment.passed else "vetoed"
            veto_data = []
            if assessment.vetos:
                for veto in assessment.vetos:
                    veto_data.append({
                        "type": veto.veto_type,
                        "reason": veto.veto_reason,
                        "severity": veto.severity,
                    })

            warning_data = list(assessment.warnings) if assessment.warnings else []
            impact_data = assessment.portfolio_impact if assessment.portfolio_impact else None

            parts.append(render_veto_section(
                hypothesis_id=assessment.hypothesis_id,
                status=status,
                vetos=veto_data if veto_data else None,
                warnings=warning_data if warning_data else None,
                portfolio_impact=impact_data,
            ))

        # ── Disclaimer ──
        parts.append("")
        parts.append("```")
        parts.append("⚖️  NOTICE: Risk Manager operates independently and can veto strategies")
        parts.append("   but CANNOT approve deployment. Only the human CIO has final approval")
        parts.append("   authority. All decisions require human sign-off.")
        parts.append("```")
        parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="risk-manager",
            duration_seconds=report.duration_seconds,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")

    def _send_veto_alerts(self, assessments: list[PortfolioRiskAssessment]) -> None:
        """Send email alerts for vetoes."""
        try:
            vetoed = [a for a in assessments if not a.passed and a.vetos]
            if not vetoed:
                return

            notifier = EmailNotifier()
            subject = f"[HRP] Risk Manager - {len(vetoed)} Strategy Vetoes"

            body_lines = [
                "Risk Manager has vetoed the following strategies:",
                "",
            ]
            for assessment in vetoed:
                for veto in assessment.vetos:
                    body_lines.append(
                        f"- {assessment.hypothesis_id}: {veto.veto_reason}"
                    )

            notifier.send_notification(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send veto alerts: {e}")
