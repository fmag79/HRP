"""
CIO Agent - Chief Investment Officer Agent.

Makes strategic decisions about research lines and manages paper portfolio.
Advisory mode: presents recommendations, awaits user approval.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from hrp.agents.sdk_agent import SDKAgent
from hrp.research.lineage import EventType
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anthropic import Anthropic


@dataclass
class CIOScore:
    """
    Balanced score across 4 dimensions for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis being scored
        statistical: Statistical quality score (0-1)
        risk: Risk profile score (0-1)
        economic: Economic rationale score (0-1)
        cost: Cost realism score (0-1)
        critical_failure: Whether a critical failure was detected
    """

    hypothesis_id: str
    statistical: float
    risk: float
    economic: float
    cost: float
    critical_failure: bool = False

    @property
    def total(self) -> float:
        """Calculate total score as average of 4 dimensions."""
        return (self.statistical + self.risk + self.economic + self.cost) / 4

    @property
    def decision(self) -> Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]:
        """
        Map score to decision.

        Returns:
            CONTINUE: Score >= 0.75, no critical failure
            CONDITIONAL: Score 0.50-0.74, no critical failure
            KILL: Score < 0.50, no critical failure
            PIVOT: Critical failure detected (overrides score)
        """
        if self.critical_failure:
            return "PIVOT"

        if self.total >= 0.75:
            return "CONTINUE"
        if self.total >= 0.50:
            return "CONDITIONAL"
        return "KILL"


@dataclass
class CIODecision:
    """
    Single decision for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis this decision is for
        decision: One of CONTINUE, CONDITIONAL, KILL, PIVOT
        score: The CIOScore that led to this decision
        rationale: Human-readable explanation
        evidence: Supporting data (MLflow runs, reports, metrics)
        paper_allocation: For CONTINUE decisions, portfolio weight (0-1)
        pivot_direction: For PIVOT decisions, suggested redirect
    """

    hypothesis_id: str
    decision: Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]
    score: CIOScore
    rationale: str
    evidence: dict
    paper_allocation: Optional[float] = None
    pivot_direction: Optional[str] = None


@dataclass
class CIOReport:
    """
    Complete weekly CIO report.

    Attributes:
        report_date: When the report was generated
        decisions: All decisions made in this review cycle
        portfolio_state: Current paper portfolio state
        market_regime: Current market regime context
        next_actions: Prioritized action items
        report_path: Path to the generated markdown report
    """

    report_date: date
    decisions: list[CIODecision]
    portfolio_state: dict
    market_regime: str
    next_actions: list[dict]
    report_path: str


class CIOAgent(SDKAgent):
    """
    Chief Investment Officer Agent.

    Makes strategic decisions about research lines and manages paper portfolio.
    Advisory mode: presents recommendations, awaits user approval.
    """

    agent_name = "cio"
    agent_version = "1.0.0"

    # Minimum probability of skill required for CONTINUE decision
    # Strategies below this threshold are KILL even if score >= 0.75
    SKILL_PROBABILITY_THRESHOLD = 0.10  # 10% minimum confidence

    DEFAULT_THRESHOLDS = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.20,
        "sharpe_decay_limit": 0.50,
        "min_ic": 0.03,
        "max_turnover": 0.50,
        "critical_sharpe_decay": 0.75,
        "critical_target_leakage": 0.95,
        "critical_max_drawdown": 0.35,
        "min_profitable_regimes": 2,
    }

    def __init__(
        self,
        job_id: str,
        actor: str,
        api: PlatformAPI | None = None,
        thresholds: dict | None = None,
        dependencies: list[str] | None = None,
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize CIO Agent.

        Args:
            job_id: Unique job identifier
            actor: Actor identity (e.g., "agent:cio")
            api: PlatformAPI instance (created if None)
            thresholds: Custom decision thresholds
            dependencies: List of data requirements
            anthropic_api_key: Optional Anthropic API key (reads from env if None)
        """
        super().__init__(
            job_id=job_id,
            actor=actor,
            dependencies=dependencies or [],
        )
        self.api = api or PlatformAPI()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.anthropic_client = self._init_anthropic_client(anthropic_api_key)

    def _init_anthropic_client(self, api_key: str | None) -> "Anthropic | None":
        """
        Initialize Anthropic client for economic rationale assessment.

        Args:
            api_key: Optional API key (reads from ANTHROPIC_API_KEY env if None)

        Returns:
            Anthropic client instance or None if no API key available
        """
        import os

        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        return Anthropic(api_key=key)

    def execute(self) -> dict[str, any]:
        """
        Execute CIO Agent weekly review.

        Reviews all validated hypotheses, scores them across 4 dimensions,
        persists decisions to database, and generates a report.

        Returns:
            Dict with decisions summary and report path
        """
        from datetime import date
        import json

        # Fetch validated hypotheses awaiting CIO review
        hypotheses = self._fetch_validated_hypotheses()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "decisions": [],
                "message": "No validated hypotheses awaiting CIO review",
            }

        # Get pipeline statistics for multiple testing adjustment
        pipeline_stats = self._get_pipeline_statistics()
        n_trials = max(pipeline_stats.get("total_generated", 10), len(hypotheses))

        decisions = []
        for hyp in hypotheses:
            hypothesis_id = hyp["hypothesis_id"]

            # Gather data for scoring
            experiment_data = self._get_experiment_data(hypothesis_id)
            if not experiment_data:
                continue

            risk_data = self._get_risk_data(experiment_data)
            economic_data = self._get_economic_data(hyp)
            cost_data = self._get_cost_data(hyp)

            # Calculate deflated Sharpe BEFORE scoring for kill gate check
            sharpe = experiment_data.get("sharpe", 0)
            n_observations = experiment_data.get("n_observations", 252)
            stat_sig = self._calculate_statistical_significance(sharpe, n_observations)
            deflated = self._calculate_deflated_sharpe(sharpe, n_trials, n_observations)
            probability_of_skill = deflated.get("probability_of_skill", 0.5)

            # Score the hypothesis
            score = self.score_hypothesis(
                hypothesis_id=hypothesis_id,
                experiment_data=experiment_data,
                risk_data=risk_data,
                economic_data=economic_data,
                cost_data=cost_data,
            )

            # HARD GATE: Override decision if probability of skill is too low
            # Even if the dimensional score is high, we cannot approve strategies
            # that are statistically indistinguishable from luck after multiple testing
            decision = score.decision
            skill_gate_failed = False
            if probability_of_skill < self.SKILL_PROBABILITY_THRESHOLD and decision == "CONTINUE":
                decision = "KILL"
                skill_gate_failed = True
                logger.info(
                    f"Skill gate triggered for {hypothesis_id}: "
                    f"prob_skill={probability_of_skill:.1%} < {self.SKILL_PROBABILITY_THRESHOLD:.0%}"
                )

            # Build rationale (may be augmented if skill gate failed)
            rationale = self._build_rationale(score, experiment_data, risk_data)
            if skill_gate_failed:
                rationale = (
                    f"**SKILL GATE OVERRIDE**: Insufficient statistical evidence after multiple "
                    f"testing adjustment. Probability of skill: {probability_of_skill:.1%} "
                    f"(threshold: {self.SKILL_PROBABILITY_THRESHOLD:.0%}). "
                    f"Deflated Sharpe: {deflated.get('deflated_sharpe', 0):.2f}\n\n"
                    f"Original score ({score.total:.2f}) would have been CONTINUE, "
                    f"but overridden to KILL due to statistical insignificance.\n\n"
                    + rationale
                )

            # Save decision to database (use overridden decision, not score.decision)
            self._save_decision_with_override(
                hypothesis_id=hypothesis_id,
                score=score,
                rationale=rationale,
                decision_override=decision if skill_gate_failed else None,
            )

            decisions.append({
                "hypothesis_id": hypothesis_id,
                "title": hyp.get("title", ""),
                "thesis": hyp.get("thesis", ""),
                "decision": decision,  # Use overridden decision
                "score": score.total,
                "score_breakdown": {
                    "statistical": score.statistical,
                    "risk": score.risk,
                    "economic": score.economic,
                    "cost": score.cost,
                },
                "rationale": rationale,
                "experiment_data": experiment_data,
                "risk_data": risk_data,
                "cost_data": cost_data,
                "statistical_significance": stat_sig,
                "deflated_sharpe": deflated,
                "skill_gate_failed": skill_gate_failed,
            })

            # Stage model if CONTINUE (use overridden decision)
            self._maybe_stage_model(
                hypothesis_id=hypothesis_id,
                decision=decision,  # Use overridden decision, not score.decision
                experiment_data=experiment_data,
            )

        # Generate report with pipeline context
        report_path = self._generate_report(decisions, pipeline_stats, n_trials)

        self._log_agent_event(
            event_type=EventType.CIO_AGENT_DECISION,
            details={
                "decision_count": len(decisions),
                "decisions": [
                    {"hypothesis_id": d["hypothesis_id"], "decision": d["decision"]}
                    for d in decisions
                ],
            },
        )

        return {
            "status": "complete",
            "decisions": decisions,
            "report_path": str(report_path),
            "decision_count": len(decisions),
        }

    def _fetch_validated_hypotheses(self) -> list[dict]:
        """Fetch hypotheses with 'validated' status."""
        return self.api.list_hypotheses(status='validated')

    def _get_experiment_data(self, hypothesis_id: str) -> dict | None:
        """Get experiment metrics for a hypothesis from ML Scientist results."""
        hyp_with_meta = self.api.get_hypothesis_with_metadata(hypothesis_id)
        if not hyp_with_meta:
            return None

        metadata = hyp_with_meta.get("metadata") or {}
        ml_results = metadata.get("ml_scientist_results", {})
        if not ml_results:
            return None

        mean_ic = ml_results.get("mean_ic", 0)
        ic_std = ml_results.get("ic_std", 0)

        return {
            "sharpe": mean_ic * 20,  # IC-based proxy (IC=0.05 → ~1.0 Sharpe)
            "stability_score": ml_results.get("stability_score", 2.0),
            "mean_ic": mean_ic,
            "fold_cv": ic_std / max(mean_ic, 0.001),
        }

    def _get_risk_data(self, experiment_data: dict) -> dict:
        """Derive risk metrics from experiment data."""
        stability = experiment_data.get("stability_score", 2.0)
        return {
            "max_drawdown": min(0.10 + stability * 0.10, 0.50),
            "volatility": 0.15,  # conservative default for long-only equities
            "regime_stable": stability <= 1.0,
            "sharpe_decay": experiment_data.get("fold_cv", 2.0) / 3.0,
        }

    def _get_economic_data(self, hypothesis: dict) -> dict:
        """Get economic rationale data from hypothesis."""
        import json

        metadata = hypothesis.get("metadata")
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        return {
            "thesis": hypothesis.get("thesis", ""),
            "current_regime": "Bull Market",  # Would detect from market data
            "black_box_count": metadata.get("black_box_count", 1),
            "uniqueness": metadata.get("uniqueness", "unknown"),
            "agent_reports": {},
        }

    def _get_cost_data(self, hypothesis: dict) -> dict:
        """Get cost realism data."""
        return {
            "slippage_survival": "stable",
            "turnover": 0.30,
            "capacity": "high",
            "execution_complexity": "low",
        }

    def _build_rationale(
        self, score: "CIOScore", experiment_data: dict, risk_data: dict
    ) -> str:
        """Build human-readable rationale for the decision."""
        parts = []

        # Decision explanation
        if score.decision == "CONTINUE":
            parts.append(f"Strong overall performance (score: {score.total:.2f}).")
        elif score.decision == "CONDITIONAL":
            parts.append(f"Moderate performance (score: {score.total:.2f}) requires additional validation.")
        elif score.decision == "KILL":
            parts.append(f"Poor performance (score: {score.total:.2f}) does not meet deployment criteria.")
        else:  # PIVOT
            parts.append("Critical failure detected; approach requires fundamental revision.")

        # Dimension breakdown
        parts.append(f"\nDimension Scores:")
        parts.append(f"  Statistical: {score.statistical:.2f} (Sharpe: {experiment_data.get('sharpe', 0):.2f})")
        parts.append(f"  Risk: {score.risk:.2f} (Max DD: {risk_data.get('max_drawdown', 0):.1%})")
        parts.append(f"  Economic: {score.economic:.2f}")
        parts.append(f"  Cost: {score.cost:.2f}")

        return "\n".join(parts)

    def _save_decision(
        self, hypothesis_id: str, score: "CIOScore", rationale: str
    ) -> None:
        """Save CIO decision to database."""
        self.api.log_cio_decision(
            hypothesis_id=hypothesis_id,
            decision=score.decision,
            score_total=score.total,
            score_statistical=score.statistical,
            score_risk=score.risk,
            score_economic=score.economic,
            score_cost=score.cost,
            rationale=rationale,
        )

    def _save_decision_with_override(
        self,
        hypothesis_id: str,
        score: "CIOScore",
        rationale: str,
        decision_override: str | None = None,
    ) -> None:
        """
        Save CIO decision to database, with optional decision override.

        Args:
            hypothesis_id: The hypothesis this decision is for
            score: The CIOScore from dimensional scoring
            rationale: Human-readable explanation
            decision_override: If provided, use this decision instead of score.decision
                             (e.g., when skill gate overrides CONTINUE to KILL)
        """
        self.api.log_cio_decision(
            hypothesis_id=hypothesis_id,
            decision=decision_override if decision_override else score.decision,
            score_total=score.total,
            score_statistical=score.statistical,
            score_risk=score.risk,
            score_economic=score.economic,
            score_cost=score.cost,
            rationale=rationale,
        )

    def _maybe_stage_model(
        self,
        hypothesis_id: str,
        decision: str,
        experiment_data: dict,
    ) -> None:
        """Stage model for deployment if decision is CONTINUE."""
        if decision != "CONTINUE":
            return
        self._stage_model_for_deployment(hypothesis_id, experiment_data)

    def _stage_model_for_deployment(
        self,
        hypothesis_id: str,
        experiment_data: dict,
    ) -> None:
        """Register model and deploy to staging."""
        try:
            experiment_id = experiment_data.get("experiment_id")
            model_type = experiment_data.get("model_type", "unknown")
            model_name = f"hyp_{hypothesis_id}_{model_type}"

            model_version = self.api.register_model(
                model=None,
                model_name=model_name,
                model_type=model_type,
                features=experiment_data.get("features", []),
                target=experiment_data.get("target", "returns_20d"),
                metrics=experiment_data.get("metrics", {}),
                hyperparameters=experiment_data.get("hyperparameters", {}),
                training_date=date.today(),
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
            )

            self.api.deploy_model(
                model_name=model_name,
                model_version=model_version,
                validation_data=pd.DataFrame(),
                environment="staging",
                actor="agent:cio",
            )

            self.api.log_event(
                event_type="model_deployed",
                actor="agent:cio",
                hypothesis_id=hypothesis_id,
                details={
                    "model_name": model_name,
                    "environment": "staging",
                    "experiment_id": experiment_id,
                },
            )

            logger.info(f"Staged model {model_name} for hypothesis {hypothesis_id}")

        except Exception as e:
            logger.error(f"Failed to stage model for {hypothesis_id}: {e}")

    def _generate_report(
        self,
        decisions: list[dict],
        pipeline_stats: dict[str, Any] | None = None,
        n_trials: int = 10,
    ) -> "Path":
        """
        Generate Medallion-grade CIO report with comprehensive quantitative analysis.

        Creates an institutional-quality research document with:
        - Statistical significance analysis (p-values, confidence intervals)
        - Multiple testing adjustment (Bailey & Lopez de Prado deflated Sharpe)
        - Market regime context and strategy-regime fit
        - Portfolio-level correlation and diversification analysis
        - Dimensional scoring breakdown
        - Risk decomposition and tail risk metrics
        - Research pipeline funnel statistics
        - Actionable recommendations with governance matrix

        Args:
            decisions: List of decision dicts with comprehensive data
            pipeline_stats: Hypothesis pipeline funnel statistics
            n_trials: Number of hypotheses tested (for multiple testing adjustment)

        Returns:
            Path to generated report file
        """
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_scorecard, render_alert_banner, render_insights,
            render_section_divider, render_progress_bar, DECISION_EMOJI,
            format_metric,
        )

        pipeline_stats = pipeline_stats or {}
        report_date = date.today()
        report_dir = get_config().data.reports_dir / report_date.strftime("%Y-%m-%d")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}-09-cio-review.md"

        # ══════════════════════════════════════════════════════════════════════
        # AGGREGATE STATISTICS
        # ══════════════════════════════════════════════════════════════════════
        total = len(decisions)
        decision_counts = {}
        for d in decisions:
            decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1

        continue_count = decision_counts.get("CONTINUE", 0)
        conditional_count = decision_counts.get("CONDITIONAL", 0)
        kill_count = decision_counts.get("KILL", 0)
        pivot_count = decision_counts.get("PIVOT", 0)

        # Collect scores and metrics
        all_scores = [d.get("score", 0) for d in decisions]
        all_sharpes = []
        all_deflated_sharpes = []
        significant_count = 0

        for d in decisions:
            exp_data = d.get("experiment_data", {})
            stat_sig = d.get("statistical_significance", {})
            deflated = d.get("deflated_sharpe", {})

            if exp_data.get("sharpe") is not None:
                all_sharpes.append(exp_data["sharpe"])
            if deflated.get("deflated_sharpe") is not None:
                all_deflated_sharpes.append(deflated["deflated_sharpe"])
            if stat_sig.get("significant_5pct"):
                significant_count += 1

        # Collect dimensional scores
        dim_scores = {"statistical": [], "risk": [], "economic": [], "cost": []}
        for d in decisions:
            breakdown = d.get("score_breakdown", {})
            for dim in dim_scores:
                if dim in breakdown:
                    dim_scores[dim].append(breakdown[dim])

        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(decisions)

        # Regime context
        regime_context = self._get_regime_context()

        parts = []

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_header(
            title="CIO Weekly Review",
            report_type="cio-review",
            date_str=report_date.strftime("%Y-%m-%d"),
            subtitle=f"🎯 {total} hypotheses | {significant_count} statistically significant | Medallion-Grade Analysis",
        ))

        # ══════════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        parts.append("## Executive Summary\n")

        # Determine verdict based on statistical significance, deflated Sharpe, and decisions
        # IMPORTANT: "STRONG ALPHA" requires deflated Sharpe > 0, meaning alpha beyond
        # what's expected from data snooping / multiple testing
        avg_deflated_sharpe = np.mean(all_deflated_sharpes) if all_deflated_sharpes else 0
        has_positive_deflated_alpha = avg_deflated_sharpe > 0.1  # Meaningful positive alpha

        if total == 0:
            verdict = "⚪ **NO HYPOTHESES TO REVIEW** — Pipeline empty"
        elif significant_count == total and continue_count == total and has_positive_deflated_alpha:
            verdict = "🟢 **STRONG ALPHA DETECTED** — All strategies show statistically significant edge after multiple testing adjustment"
        elif significant_count == total and continue_count == total:
            # All significant but deflated Sharpe near zero - likely data snooping
            verdict = "🟡 **STATISTICAL SIGNIFICANCE BUT NO DEFLATED ALPHA** — Strategies appear significant but expected under null hypothesis given multiple testing"
        elif significant_count >= total * 0.6 and continue_count >= total * 0.5:
            verdict = "🟢 **POSITIVE OUTLOOK** — Majority of strategies show real signal"
        elif continue_count >= total * 0.5:
            verdict = "🟡 **MIXED RESULTS** — Some strategies approved but significance varies"
        elif continue_count > 0:
            verdict = "🟠 **LIMITED APPROVAL** — Few strategies meet deployment criteria"
        elif kill_count == total:
            verdict = "🔴 **ALL STRATEGIES REJECTED** — No hypotheses meet CIO standards"
        else:
            verdict = "🟡 **UNDER REVIEW** — Decisions require additional analysis"

        parts.append(f"{verdict}\n")

        # Executive metrics table
        avg_score = np.mean(all_scores) if all_scores else 0
        avg_deflated = np.mean(all_deflated_sharpes) if all_deflated_sharpes else 0
        prob_skill_avg = np.mean([
            d.get("deflated_sharpe", {}).get("probability_of_skill", 0.5)
            for d in decisions
        ]) if decisions else 0

        parts.append("| Metric | Value | Interpretation |")
        parts.append("|--------|-------|----------------|")
        parts.append(f"| Strategies with p < 0.05 | {significant_count}/{total} ({significant_count/max(total,1)*100:.0f}%) | {'Majority significant' if significant_count > total/2 else 'Minority significant'} |")
        parts.append(f"| Average Deflated Sharpe | {avg_deflated:.2f} | {'Positive after adjustment' if avg_deflated > 0 else 'Negative after adjustment'} |")
        parts.append(f"| Portfolio Expected Sharpe | {portfolio_metrics.get('estimated_portfolio_sharpe', 0):.2f} | Combined strategies |")
        parts.append(f"| Avg Probability of Skill | {prob_skill_avg*100:.0f}% | {'High confidence' if prob_skill_avg > 0.7 else 'Moderate' if prob_skill_avg > 0.5 else 'Low confidence'} |")
        parts.append(f"| Multiple Testing Haircut | ~{n_trials} trials | Bailey & Lopez de Prado adjustment |")
        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # KPI DASHBOARD
        # ══════════════════════════════════════════════════════════════════════
        kpi_metrics = [
            {"icon": "📋", "label": "Reviewed", "value": total, "detail": "hypotheses"},
            {"icon": "✅", "label": "Continue", "value": continue_count, "detail": f"{continue_count/max(total,1)*100:.0f}% approved"},
            {"icon": "⚠️", "label": "Conditional", "value": conditional_count, "detail": "needs work"},
            {"icon": "❌", "label": "Kill/Pivot", "value": kill_count + pivot_count, "detail": "rejected"},
            {"icon": "📊", "label": "Significant", "value": significant_count, "detail": "p < 0.05"},
        ]
        parts.append(render_kpi_dashboard(kpi_metrics[:5]))

        # Alert banners
        if kill_count + pivot_count > total / 2:
            parts.append(render_alert_banner(
                [f"High rejection rate: {kill_count + pivot_count}/{total} hypotheses rejected",
                 "Review hypothesis generation process and signal quality"],
                severity="critical",
            ))
        elif continue_count > 0 and significant_count >= continue_count:
            parts.append(render_alert_banner(
                [f"{continue_count} strategies approved with statistical significance",
                 "Ready for paper portfolio allocation pending human approval"],
                severity="info",
            ))

        # ══════════════════════════════════════════════════════════════════════
        # STATISTICAL SIGNIFICANCE PANEL
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📊 Statistical Significance Analysis"))

        parts.append("### Multiple Testing Adjustment (Bailey & Lopez de Prado)\n")
        parts.append(f"⚠️ **Data Snooping Warning**: {n_trials} hypotheses tested in pipeline\n")

        parts.append("| Hypothesis | Raw Sharpe | Deflated Sharpe | p-value | 95% CI | Significant? |")
        parts.append("|------------|------------|-----------------|---------|--------|--------------|")

        for d in decisions:
            hyp_id = d.get("hypothesis_id", "Unknown")
            stat_sig = d.get("statistical_significance", {})
            deflated = d.get("deflated_sharpe", {})

            raw_sharpe = stat_sig.get("sharpe", 0)
            defl_sharpe = deflated.get("deflated_sharpe", 0)
            haircut = deflated.get("haircut_pct", 0)
            p_val = stat_sig.get("p_value", 1)
            ci_lower = stat_sig.get("ci_95_lower", 0)
            ci_upper = stat_sig.get("ci_95_upper", 0)
            sig_5pct = stat_sig.get("significant_5pct", False)
            sig_1pct = stat_sig.get("significant_1pct", False)

            if sig_1pct:
                sig_str = "✅ p < 0.01"
            elif sig_5pct:
                sig_str = "✅ p < 0.05"
            else:
                sig_str = "❌ NS"

            parts.append(f"| {hyp_id} | {raw_sharpe:.2f} | {defl_sharpe:.2f} ({-haircut:.0f}%) | {p_val:.3f} | [{ci_lower:.2f}, {ci_upper:.2f}] | {sig_str} |")

        parts.append("")

        # Interpretation
        expected_false_pos = n_trials * 0.05
        parts.append("### Interpretation")
        parts.append(f"- **Expected Max Sharpe under null**: {stats.norm.ppf(1 - 1/max(n_trials, 2)):.2f} (given {n_trials} trials)")
        parts.append(f"- **Strategies beating null**: {len([d for d in all_deflated_sharpes if d > 0])}/{total}")
        parts.append(f"- **Expected false positives (5% level)**: {expected_false_pos:.1f}")
        parts.append(f"- **Actual positives (p < 0.05)**: {significant_count}")
        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # MARKET REGIME CONTEXT
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("🌡️ Market Regime Context"))

        parts.append("### Current Regime Detection\n")
        parts.append("| Indicator | Value | Signal |")
        parts.append("|-----------|-------|--------|")

        if regime_context.get("data_available"):
            vol_20d = regime_context.get("volatility_20d", 0)
            spy_vs_sma = regime_context.get("spy_vs_sma50", 0)
            parts.append(f"| SPY 20d Volatility | {vol_20d:.1%} | {'🟢 Low' if vol_20d < 0.18 else '🔴 High'} |")
            parts.append(f"| SPY vs 50-SMA | {spy_vs_sma:+.1f}% | {'🟢 Above' if spy_vs_sma > 0 else '🔴 Below'} |")
            parts.append(f"| **Overall Regime** | **{regime_context.get('regime_label', 'Unknown')}** | |")
        else:
            parts.append("| Data | Unavailable | ⚠️ Cannot determine regime |")

        parts.append("")

        # Strategy-Regime fit (simplified)
        parts.append("### Strategy-Regime Considerations")
        parts.append("| Strategy Type | Low-Vol Bull | High-Vol Bull | Bear Markets |")
        parts.append("|---------------|--------------|---------------|--------------|")
        parts.append("| Momentum | ✅ Strong | ⚠️ Weak | ❌ Poor |")
        parts.append("| Value | ⚠️ Moderate | ✅ Strong | ✅ Resilient |")
        parts.append("| Quality | ✅ Strong | ✅ Strong | ✅ Defensive |")
        parts.append("| Low Vol | ✅ Strong | ⚠️ Moderate | ✅ Strong |")
        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # PORTFOLIO-LEVEL ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        if continue_count > 0:
            parts.append(render_section_divider("📈 Portfolio-Level Analysis"))

            parts.append("### Combined Portfolio Metrics\n")
            parts.append("| Metric | Individual Avg | Combined Portfolio | Diversification Benefit |")
            parts.append("|--------|----------------|--------------------|-----------------------|")

            avg_indiv = portfolio_metrics.get("avg_individual_sharpe", 0)
            port_sharpe = portfolio_metrics.get("estimated_portfolio_sharpe", 0)
            improvement = portfolio_metrics.get("sharpe_improvement_pct", 0)
            div_ratio = portfolio_metrics.get("diversification_ratio", 1)

            parts.append(f"| Expected Sharpe | {avg_indiv:.2f} | {port_sharpe:.2f} | +{improvement:.0f}% |")
            parts.append(f"| Diversification Ratio | 1.00 | {div_ratio:.2f} | — |")
            parts.append(f"| Strategy Count | — | {portfolio_metrics.get('n_strategies', 0)} | — |")
            parts.append(f"| Avg Correlation (est.) | — | {portfolio_metrics.get('avg_pairwise_correlation', 0.25):.2f} | Target < 0.40 |")
            parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # AGGREGATE SCORING DISTRIBUTION
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📊 Aggregate Scoring Distribution"))

        parts.append("### Score Statistics by Dimension\n")
        parts.append("| Statistic | Total | Statistical | Risk | Economic | Cost |")
        parts.append("|-----------|-------|-------------|------|----------|------|")

        def safe_stat(arr, func):
            return func(arr) if arr else 0

        parts.append(f"| **Mean** | {safe_stat(all_scores, np.mean):.2f} | {safe_stat(dim_scores['statistical'], np.mean):.2f} | {safe_stat(dim_scores['risk'], np.mean):.2f} | {safe_stat(dim_scores['economic'], np.mean):.2f} | {safe_stat(dim_scores['cost'], np.mean):.2f} |")
        parts.append(f"| **Std** | {safe_stat(all_scores, np.std):.2f} | {safe_stat(dim_scores['statistical'], np.std):.2f} | {safe_stat(dim_scores['risk'], np.std):.2f} | {safe_stat(dim_scores['economic'], np.std):.2f} | {safe_stat(dim_scores['cost'], np.std):.2f} |")
        parts.append(f"| **Min** | {safe_stat(all_scores, np.min):.2f} | {safe_stat(dim_scores['statistical'], np.min):.2f} | {safe_stat(dim_scores['risk'], np.min):.2f} | {safe_stat(dim_scores['economic'], np.min):.2f} | {safe_stat(dim_scores['cost'], np.min):.2f} |")
        parts.append(f"| **Max** | {safe_stat(all_scores, np.max):.2f} | {safe_stat(dim_scores['statistical'], np.max):.2f} | {safe_stat(dim_scores['risk'], np.max):.2f} | {safe_stat(dim_scores['economic'], np.max):.2f} | {safe_stat(dim_scores['cost'], np.max):.2f} |")
        parts.append("")

        parts.append("### Decision Distribution\n")
        parts.append("| Decision | Count | % | Avg Score | Avg Deflated Sharpe |")
        parts.append("|----------|-------|---|-----------|---------------------|")

        for dec_type in ["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]:
            count = decision_counts.get(dec_type, 0)
            if count > 0:
                dec_decisions = [d for d in decisions if d.get("decision") == dec_type]
                avg_sc = np.mean([d.get("score", 0) for d in dec_decisions])
                avg_defl = np.mean([d.get("deflated_sharpe", {}).get("deflated_sharpe", 0) for d in dec_decisions])
                emoji = DECISION_EMOJI.get(dec_type, "")
                parts.append(f"| {emoji} {dec_type} | {count} | {count/max(total,1)*100:.0f}% | {avg_sc:.2f} | {avg_defl:.2f} |")

        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # DIMENSIONAL DEEP DIVE
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📐 Dimensional Analysis"))

        dimension_info = [
            ("Statistical Quality", "statistical", "Sharpe ratio, IC, stability, fold consistency"),
            ("Risk Profile", "risk", "Max drawdown, volatility, regime stability, Sharpe decay"),
            ("Economic Rationale", "economic", "Thesis strength, regime alignment, interpretability"),
            ("Cost Realism", "cost", "Slippage survival, turnover, capacity, execution"),
        ]

        for dim_name, dim_key, components in dimension_info:
            scores = dim_scores.get(dim_key, [])
            if scores:
                avg = np.mean(scores)
                min_s, max_s = np.min(scores), np.max(scores)
                interpretation = "Strong" if avg >= 0.75 else "Acceptable" if avg >= 0.50 else "Needs improvement"

                parts.append(f"### {dim_name} (Weight: 25%)")
                parts.append(f"- **Average Score**: {avg:.2f}/1.00")
                parts.append(f"- **Range**: [{min_s:.2f}, {max_s:.2f}]")
                parts.append(f"- **Interpretation**: {interpretation}")
                parts.append(f"- **Components**: {components}")
                parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # RESEARCH PIPELINE STATISTICS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("🔬 Research Pipeline Statistics"))

        parts.append("### Hypothesis Funnel\n")
        parts.append("| Stage | Count | Conversion |")
        parts.append("|-------|-------|------------|")

        total_gen = pipeline_stats.get("total_generated", n_trials)
        parts.append(f"| Generated (all time) | {total_gen} | 100% |")
        parts.append(f"| Draft | {pipeline_stats.get('draft', 0)} | — |")
        parts.append(f"| Testing | {pipeline_stats.get('testing', 0)} | — |")
        parts.append(f"| Validated | {pipeline_stats.get('validated', 0)} | {pipeline_stats.get('conversion_to_validated', 0):.0f}% |")
        parts.append(f"| **CIO Review (this cycle)** | **{total}** | — |")
        parts.append(f"| CIO CONTINUE | {continue_count} | {continue_count/max(total,1)*100:.0f}% of reviewed |")
        parts.append(f"| Deployed | {pipeline_stats.get('deployed', 0)} | {pipeline_stats.get('conversion_to_deployed', 0):.0f}% |")
        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # PER-HYPOTHESIS DETAILED ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📋 Hypothesis Analysis"))

        for d in decisions:
            hyp_id = d.get("hypothesis_id", "Unknown")
            title = d.get("title", "")
            thesis = d.get("thesis", "")
            decision = d.get("decision", "")
            score = d.get("score", 0)
            breakdown = d.get("score_breakdown", {})
            rationale = d.get("rationale", "")
            exp_data = d.get("experiment_data", {})
            risk_data = d.get("risk_data", {})
            cost_data = d.get("cost_data", {})
            stat_sig = d.get("statistical_significance", {})
            deflated = d.get("deflated_sharpe", {})

            dec_emoji = DECISION_EMOJI.get(decision, "❓")
            parts.append(f"### {dec_emoji} {hyp_id} — **{decision}** (Score: {score:.2f})\n")

            if title:
                parts.append(f"**{title}**\n")
            if thesis:
                thesis_short = thesis[:200] + "..." if len(thesis) > 200 else thesis
                parts.append(f"> {thesis_short}\n")

            # Statistical Significance
            parts.append("#### Statistical Significance\n")
            parts.append("| Metric | Value | Interpretation |")
            parts.append("|--------|-------|----------------|")

            raw_sharpe = stat_sig.get("sharpe", 0)
            defl_sharpe = deflated.get("deflated_sharpe", 0)
            p_val = stat_sig.get("p_value", 1)
            ci_lower = stat_sig.get("ci_95_lower", 0)
            ci_upper = stat_sig.get("ci_95_upper", 0)
            prob_skill = deflated.get("probability_of_skill", 0.5)

            parts.append(f"| Raw Sharpe | {raw_sharpe:.2f} | {'Above' if raw_sharpe > 1.0 else 'Below'} 1.0 threshold |")
            parts.append(f"| Deflated Sharpe | {defl_sharpe:.2f} | After multiple testing |")
            parts.append(f"| p-value | {p_val:.3f} | {'Significant' if p_val < 0.05 else 'Not significant'} |")
            parts.append(f"| 95% CI | [{ci_lower:.2f}, {ci_upper:.2f}] | {'Excludes' if ci_lower > 0 else 'Includes'} zero |")
            parts.append(f"| Prob. of Skill | {prob_skill*100:.0f}% | {'High' if prob_skill > 0.7 else 'Moderate' if prob_skill > 0.5 else 'Low'} confidence |")
            parts.append("")

            # Dimensional Scorecard
            parts.append("#### Dimensional Scorecard\n")
            parts.append("| Dimension | Score | Rating |")
            parts.append("|-----------|-------|--------|")

            for dim_name in ["statistical", "risk", "economic", "cost"]:
                dim_score = breakdown.get(dim_name, 0)
                bar = render_progress_bar(dim_score, 1.0, width=10, show_pct=False)
                status = "✅" if dim_score >= 0.70 else "⚠️" if dim_score >= 0.50 else "❌"
                parts.append(f"| {dim_name.title()} | {dim_score:.2f} | {bar} {status} |")

            overall_bar = render_progress_bar(score, 1.0, width=10, show_pct=False)
            overall_status = "✅" if score >= 0.75 else "⚠️" if score >= 0.50 else "❌"
            parts.append(f"| **OVERALL** | **{score:.2f}** | {overall_bar} {overall_status} |")
            parts.append("")

            # Experiment Metrics
            parts.append("#### Key Metrics\n")
            parts.append("| Category | Metric | Value | Status |")
            parts.append("|----------|--------|-------|--------|")

            sharpe = exp_data.get("sharpe", 0)
            ic = exp_data.get("mean_ic", 0)
            stability = exp_data.get("stability_score", 2.0)
            fold_cv = exp_data.get("fold_cv", 3.0)
            max_dd = risk_data.get("max_drawdown", 0)
            vol = risk_data.get("volatility", 0)
            decay = risk_data.get("sharpe_decay", 0)
            turnover = cost_data.get("turnover", 0)

            parts.append(f"| Statistical | Sharpe | {sharpe:.2f} | {'✅' if sharpe >= 1.0 else '⚠️' if sharpe >= 0.5 else '❌'} |")
            parts.append(f"| Statistical | Mean IC | {ic:.3f} | {'✅' if ic >= 0.03 else '⚠️' if ic >= 0.01 else '❌'} |")
            parts.append(f"| Statistical | Stability | {stability:.2f} | {'✅' if stability <= 1.0 else '⚠️' if stability <= 2.0 else '❌'} |")
            parts.append(f"| Risk | Max Drawdown | {max_dd:.1%} | {'✅' if max_dd <= 0.20 else '⚠️' if max_dd <= 0.30 else '❌'} |")
            parts.append(f"| Risk | Volatility | {vol:.1%} | {'✅' if vol <= 0.15 else '⚠️' if vol <= 0.25 else '❌'} |")
            parts.append(f"| Risk | Sharpe Decay | {decay:.0%} | {'✅' if decay <= 0.50 else '❌'} |")
            parts.append(f"| Cost | Turnover | {turnover:.0%} | {'✅' if turnover <= 0.50 else '⚠️' if turnover <= 1.0 else '❌'} |")
            parts.append("")

            # Rationale
            if rationale:
                parts.append("#### Rationale\n")
                parts.append(f"```\n{rationale}\n```\n")

            # Recommendation
            if decision == "CONTINUE":
                parts.append("#### Recommendation\n")
                parts.append("**ADD TO PAPER PORTFOLIO**")
                parts.append(f"- Suggested weight: {self.MAX_WEIGHT_PER_POSITION*100:.1f}% (equal risk contribution)")
                parts.append(f"- Statistical confidence: {prob_skill*100:.0f}%")
                parts.append("- Requires human CIO approval before deployment")
                parts.append("")

            parts.append("─" * 70)
            parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("💡 Recommendations"))

        action_items = []

        # Immediate actions
        if continue_count > 0:
            action_items.append({
                "priority": "high", "category": "deployment",
                "action": f"Review {continue_count} CONTINUE decisions for paper portfolio allocation",
            })
            action_items.append({
                "priority": "high", "category": "monitoring",
                "action": "Set up regime monitoring alerts for approved strategies",
            })

        if conditional_count > 0:
            action_items.append({
                "priority": "medium", "category": "research",
                "action": f"Address {conditional_count} CONDITIONAL items with additional walk-forward validation",
            })

        # Risk-based recommendations
        if dim_scores.get("risk") and np.mean(dim_scores["risk"]) < 0.60:
            action_items.append({
                "priority": "medium", "category": "risk",
                "action": "Investigate low risk dimension scores — review drawdown and Sharpe decay",
            })

        # Statistical recommendations
        if significant_count < total / 2:
            action_items.append({
                "priority": "medium", "category": "research",
                "action": "Low statistical significance rate — review hypothesis generation process",
            })

        # Archive recommendations
        if kill_count + pivot_count > 0:
            action_items.append({
                "priority": "low", "category": "archive",
                "action": f"Archive {kill_count + pivot_count} KILL/PIVOT hypotheses with lessons learned",
            })

        parts.append(render_insights("Action Items", action_items))

        # ══════════════════════════════════════════════════════════════════════
        # GOVERNANCE & DISCLAIMER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("⚖️ Governance & Disclaimer"))

        parts.append("### Decision Authority Matrix\n")
        parts.append("| Action | Agent Authority | Human Required |")
        parts.append("|--------|-----------------|----------------|")
        parts.append("| Score hypotheses | ✅ Yes | No |")
        parts.append("| Recommend CONTINUE | ✅ Yes | No |")
        parts.append("| Recommend KILL | ✅ Yes | No |")
        parts.append("| **Execute deployment** | ❌ No | **Yes** |")
        parts.append("| **Allocate capital** | ❌ No | **Yes** |")
        parts.append("| **Override decisions** | ❌ No | **Yes** |")
        parts.append("")

        parts.append("### Audit Trail\n")
        parts.append(f"- **Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET")
        parts.append(f"- **Agent version**: cio-agent v{self.agent_version}")
        parts.append(f"- **Hypotheses in scope**: {total}")
        parts.append(f"- **Multiple testing trials**: {n_trials}")
        parts.append(f"- **Deflation method**: Bailey & Lopez de Prado (2014)")
        parts.append("")

        parts.append("### Disclaimer\n")
        parts.append("```")
        parts.append("This report is generated by an automated agent system and is advisory")
        parts.append("only. All deployment decisions require human approval. Past performance")
        parts.append("does not guarantee future results. Paper portfolio positions do not")
        parts.append("constitute investment advice. Statistical significance does not imply")
        parts.append("economic significance. Multiple testing adjustments are estimates.")
        parts.append("```")
        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_footer(
            agent_name="cio-agent",
            timestamp=datetime.now(),
        ))

        # Write report
        report_path.write_text("\n".join(parts))
        logger.info(f"CIO report written to {report_path}")

        return report_path

    def _score_sharpe(self, sharpe: float) -> float:
        """
        Score Sharpe ratio: 0.5->0, 1.0->0.5, 1.5->1.0.

        Args:
            sharpe: Walk-forward Sharpe ratio

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.5, 1.0, 1.5
        if sharpe <= bad:
            return 0.0
        if sharpe >= good:
            return 1.0
        return (sharpe - bad) / (good - bad)

    def _score_stability(self, stability: float) -> float:
        """
        Score stability score: 2.0->0, 1.0->0.5, 0.5->1.0.

        Lower is better for stability.

        Args:
            stability: Stability score from walk-forward validation

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 2.0, 1.0, 0.5
        if stability >= bad:
            return 0.0
        if stability <= good:
            return 1.0
        if stability >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - stability) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - stability) / (target - good)

    def _score_ic(self, ic: float) -> float:
        """
        Score Information Coefficient: 0.01->0, 0.03->0.5, 0.05->1.0.

        Args:
            ic: Mean Information Coefficient

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.01, 0.03, 0.05
        if ic <= bad:
            return 0.0
        if ic >= good:
            return 1.0
        return (ic - bad) / (good - bad)

    def _score_fold_cv(self, cv: float) -> float:
        """
        Score fold coefficient of variation: 3.0->0, 2.0->0.5, 1.0->1.0.

        Lower is better for CV (less variability across folds).

        Args:
            cv: Coefficient of variation across folds

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 3.0, 2.0, 1.0
        if cv >= bad:
            return 0.0
        if cv <= good:
            return 1.0
        return (bad - cv) / (bad - good)

    def _score_statistical_dimension(
        self,
        hypothesis_id: str,
        experiment_data: dict,
    ) -> float:
        """
        Score statistical quality dimension.

        Averages scores for: Sharpe, stability, IC, fold CV.

        Args:
            hypothesis_id: The hypothesis being scored
            experiment_data: Dict with sharpe, stability_score, mean_ic, fold_cv

        Returns:
            Score 0-1
        """
        sharpe_score = self._score_sharpe(experiment_data.get("sharpe", 0))
        stability_score = self._score_stability(experiment_data.get("stability_score", 2.0))
        ic_score = self._score_ic(experiment_data.get("mean_ic", 0))
        fold_cv_score = self._score_fold_cv(experiment_data.get("fold_cv", 3.0))

        return (sharpe_score + stability_score + ic_score + fold_cv_score) / 4

    def _score_max_drawdown(self, max_dd: float) -> float:
        """
        Score max drawdown: 30%->0, 20%->0.5, 10%->1.0.

        Lower is better for drawdown.

        Args:
            max_dd: Maximum drawdown (as decimal, e.g., 0.20 for 20%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.30, 0.20, 0.10
        if max_dd >= bad:
            return 0.0
        if max_dd <= good:
            return 1.0
        return (bad - max_dd) / (bad - good)

    def _score_volatility(self, vol: float) -> float:
        """
        Score annual volatility: 25%->0, 15%->0.5, 10%->1.0.

        Lower is better for volatility.

        Args:
            vol: Annualized volatility (as decimal)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.25, 0.15, 0.10
        if vol >= bad:
            return 0.0
        if vol <= good:
            return 1.0
        if vol >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - vol) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - vol) / (target - good)

    def _score_regime_stability(self, stable: bool) -> float:
        """
        Score regime stability (binary).

        Args:
            stable: Whether >= 2/3 regimes are profitable

        Returns:
            1.0 if stable, 0.0 if not
        """
        return 1.0 if stable else 0.0

    def _score_sharpe_decay(self, decay: float) -> float:
        """
        Score Sharpe decay (binary).

        Decay <= 50% is good (no overfitting).

        Args:
            decay: Sharpe decay ratio (train_sharpe - test_sharpe) / train_sharpe

        Returns:
            1.0 if decay <= 0.50, 0.0 if > 0.50
        """
        limit = self.thresholds["sharpe_decay_limit"]
        return 1.0 if decay <= limit else 0.0

    def _check_critical_failures_risk(self, risk_data: dict) -> bool:
        """
        Check for critical failures in risk dimension.

        Critical:
        - Max drawdown > 35%
        - Sharpe decay > 75%

        Args:
            risk_data: Dict with max_drawdown, sharpe_decay

        Returns:
            True if critical failure detected
        """
        max_dd = risk_data.get("max_drawdown", 0)
        sharpe_decay = risk_data.get("sharpe_decay", 0)

        if max_dd > self.thresholds["critical_max_drawdown"]:
            return True
        if sharpe_decay > self.thresholds["critical_sharpe_decay"]:
            return True

        return False

    def _score_risk_dimension(self, hypothesis_id: str, risk_data: dict) -> float:
        """
        Score risk profile dimension.

        Averages scores for: Max DD, volatility, regime stability, Sharpe decay.

        Args:
            hypothesis_id: The hypothesis being scored
            risk_data: Dict with max_drawdown, volatility, regime_stable, sharpe_decay_ok

        Returns:
            Score 0-1
        """
        max_dd_score = self._score_max_drawdown(risk_data.get("max_drawdown", 0.30))
        vol_score = self._score_volatility(risk_data.get("volatility", 0.25))
        regime_score = self._score_regime_stability(risk_data.get("regime_stable", False))

        # Handle both sharpe_decay_ok (boolean) and sharpe_decay (float)
        if "sharpe_decay_ok" in risk_data:
            decay_score = 1.0 if risk_data["sharpe_decay_ok"] else 0.0
        else:
            decay_score = self._score_sharpe_decay(risk_data.get("sharpe_decay", 0.60))

        return (max_dd_score + vol_score + regime_score + decay_score) / 4

    def _score_turnover(self, turnover: float) -> float:
        """
        Score annual turnover: 100%->0, 50%->0.5, 20%->1.0.

        Lower is better for turnover (less trading = lower costs).

        Args:
            turnover: Annual turnover rate (as decimal, e.g., 0.50 for 50%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 1.00, 0.50, 0.20
        if turnover >= bad:
            return 0.0
        if turnover <= good:
            return 1.0
        if turnover >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - turnover) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - turnover) / (target - good)

    def _score_capacity(self, capacity: str) -> float:
        """
        Score capacity estimate (ordinal).

        Args:
            capacity: One of "low" (<$1M), "medium" ($1-10M), "high" (>$10M)

        Returns:
            Score 0-1
        """
        scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return scores.get(capacity.lower(), 0.5)

    def _score_slippage_survival(self, survival: str) -> float:
        """
        Score slippage survival (ordinal).

        Args:
            survival: One of "collapse", "degraded", "stable"

        Returns:
            Score 0-1
        """
        scores = {"collapse": 0.0, "degraded": 0.5, "stable": 1.0}
        return scores.get(survival.lower(), 0.5)

    def _score_execution_complexity(self, complexity: str) -> float:
        """
        Score execution complexity (ordinal).

        Args:
            complexity: One of "high", "medium", "low"

        Returns:
            Score 0-1
        """
        scores = {"high": 0.0, "medium": 0.5, "low": 1.0}
        return scores.get(complexity.lower(), 0.5)

    def _score_cost_dimension(self, hypothesis_id: str, cost_data: dict) -> float:
        """
        Score cost realism dimension.

        Averages scores for: Slippage survival, turnover, capacity, complexity.

        Args:
            hypothesis_id: The hypothesis being scored
            cost_data: Dict with slippage_survival, turnover, capacity, execution_complexity

        Returns:
            Score 0-1
        """
        slippage_score = self._score_slippage_survival(
            cost_data.get("slippage_survival", "degraded")
        )
        turnover_score = self._score_turnover(cost_data.get("turnover", 1.00))
        capacity_score = self._score_capacity(cost_data.get("capacity", "medium"))
        complexity_score = self._score_execution_complexity(
            cost_data.get("execution_complexity", "medium")
        )

        return (slippage_score + turnover_score + capacity_score + complexity_score) / 4

    def _score_thesis_strength(self, strength: str) -> float:
        """
        Score thesis strength (ordinal).

        Args:
            strength: One of "weak", "moderate", "strong"

        Returns:
            Score 0-1
        """
        scores = {"weak": 0.0, "moderate": 0.5, "strong": 1.0}
        return scores.get(strength.lower(), 0.5)

    def _score_regime_alignment(self, alignment: str) -> float:
        """
        Score regime alignment (ordinal).

        Args:
            alignment: One of "mismatch", "neutral", "aligned"

        Returns:
            Score 0-1
        """
        scores = {"mismatch": 0.0, "neutral": 0.5, "aligned": 1.0}
        return scores.get(alignment.lower(), 0.5)

    def _score_feature_interpretability(self, black_box_count: int) -> float:
        """
        Score feature interpretability: >5->0, 3-5->0.5, <3->1.0.

        Fewer black-box features = more interpretable = better score.

        Args:
            black_box_count: Number of black-box features (e.g., neural net outputs)

        Returns:
            Score 0-1
        """
        if black_box_count > 5:
            return 0.0
        if black_box_count < 3:
            return 1.0
        return 0.5  # 3-5 features

    def _score_uniqueness(self, uniqueness: str) -> float:
        """
        Score uniqueness (ordinal).

        Args:
            uniqueness: One of "duplicate", "related", "novel"

        Returns:
            Score 0-1
        """
        scores = {"duplicate": 0.0, "related": 0.5, "novel": 1.0}
        return scores.get(uniqueness.lower(), 0.5)

    def _assess_thesis_with_claude(
        self,
        hypothesis_id: str,
        thesis: str,
        agent_reports: dict,
        current_regime: str,
    ) -> dict:
        """
        Use Claude API to assess thesis strength and regime alignment.

        Args:
            hypothesis_id: The hypothesis being assessed
            thesis: The thesis statement
            agent_reports: Dict of agent report content
            current_regime: Current market regime (e.g., "Bull Market")

        Returns:
            Dict with thesis_strength and regime_alignment
        """
        if self.anthropic_client is None:
            # Fallback to moderate scores if no Claude client
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

        # Build prompt for Claude with more specific guidance
        prompt = f"""Assess this trading hypothesis for economic rationale:

Hypothesis: {thesis}

Current Market Regime: {current_regime}

Agent Reports:
{chr(10).join(f'- {k}: {v[:200]}...' for k, v in agent_reports.items()) if agent_reports else 'No agent reports yet'}

Respond ONLY with valid JSON:
{{
    "thesis_strength": "weak" | "moderate" | "strong",
    "regime_alignment": "mismatch" | "neutral" | "aligned"
}}

Scoring Guide:
- thesis_strength:
  - "strong": Well-documented market anomaly (e.g., momentum, value, low volatility), clear economic rationale, academic support
  - "moderate": Plausible but less established effect, or mixed evidence
  - "weak": No clear economic logic, contradicts established research

- regime_alignment:
  - "aligned": Strategy performs well in current regime (e.g., momentum works in bull markets)
  - "neutral": Regime-independent or unclear relationship
  - "mismatch": Strategy historically underperforms in current regime

Be generous - most quant strategies have at least "moderate" strength if they have any backtesting evidence.
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-latest",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()
            logger = logging.getLogger(__name__)
            logger.debug(f"Claude raw response for {hypothesis_id}: {repr(content)}")

            # Extract JSON from markdown code blocks if present
            # Handle: ```json\n{...}\n``` or ```\n{...}\n```
            import re
            # First try to find content between triple backticks
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if code_block_match:
                content = code_block_match.group(1).strip()
            # Otherwise, look for JSON-like content
            else:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                if json_match:
                    content = json_match.group(0)

            result = json.loads(content)

            # Validate result has expected keys
            if "thesis_strength" not in result or "regime_alignment" not in result:
                raise ValueError(f"Invalid Claude response: {result}")

            logger.info(f"Claude assessment for {hypothesis_id}: thesis={result['thesis_strength']}, regime={result['regime_alignment']}")
            return result

        except Exception as e:
            # Log error and fallback
            logger = logging.getLogger(__name__)
            logger.warning(f"Claude API call failed for {hypothesis_id}: {e}. Using fallback scores.")
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

    def _score_economic_dimension(
        self,
        hypothesis_id: str,
        economic_data: dict,
    ) -> float:
        """
        Score economic rationale dimension.

        Averages scores for: Thesis strength, regime alignment, interpretability, uniqueness.

        Args:
            hypothesis_id: The hypothesis being scored
            economic_data: Dict with thesis, regime, black_box_count, uniqueness, agent_reports

        Returns:
            Score 0-1
        """
        # Use Claude to assess thesis and regime if not provided
        if "thesis_strength" not in economic_data or "regime_alignment" not in economic_data:
            claude_assessment = self._assess_thesis_with_claude(
                hypothesis_id=hypothesis_id,
                thesis=economic_data.get("thesis", ""),
                agent_reports=economic_data.get("agent_reports", {}),
                current_regime=economic_data.get("current_regime", "Unknown"),
            )
            economic_data = {**economic_data, **claude_assessment}

        thesis_score = self._score_thesis_strength(economic_data.get("thesis_strength", "moderate"))
        regime_score = self._score_regime_alignment(economic_data.get("regime_alignment", "neutral"))
        interpretability_score = self._score_feature_interpretability(
            economic_data.get("black_box_count", 4)
        )
        uniqueness_score = self._score_uniqueness(economic_data.get("uniqueness", "related"))

        return (thesis_score + regime_score + interpretability_score + uniqueness_score) / 4

    def score_hypothesis(
        self,
        hypothesis_id: str,
        experiment_data: dict,
        risk_data: dict,
        economic_data: dict,
        cost_data: dict,
    ) -> CIOScore:
        """
        Score a hypothesis across all 4 dimensions.

        Args:
            hypothesis_id: The hypothesis to score
            experiment_data: Statistical metrics from MLflow
            risk_data: Risk metrics (max_dd, volatility, etc.)
            economic_data: Economic rationale data (thesis, regime, etc.)
            cost_data: Cost realism data (turnover, capacity, etc.)

        Returns:
            CIOScore with all dimension scores and decision
        """
        # Score each dimension
        statistical = self._score_statistical_dimension(hypothesis_id, experiment_data)
        risk = self._score_risk_dimension(hypothesis_id, risk_data)
        economic = self._score_economic_dimension(hypothesis_id, economic_data)
        cost = self._score_cost_dimension(hypothesis_id, cost_data)

        # Check for critical failures
        critical_failure = self._check_critical_failures_risk(risk_data)

        # Create score object
        return CIOScore(
            hypothesis_id=hypothesis_id,
            statistical=statistical,
            risk=risk,
            economic=economic,
            cost=cost,
            critical_failure=critical_failure,
        )

    # Portfolio constants
    PORTFOLIO_CAPITAL = 1_000_000  # $1M hypothetical
    MAX_POSITIONS = 20
    MAX_WEIGHT_PER_POSITION = 0.05  # 5%
    MIN_WEIGHT_THRESHOLD = 0.01  # 1%

    # Portfolio constraints
    MAX_GROSS_EXPOSURE = 1.0  # 100% long-only
    MAX_SECTOR_CONCENTRATION = 0.30  # 30%
    MAX_TURNOVER = 0.50  # 50% annual
    MAX_DRAWDOWN_LIMIT = 0.15  # 15%

    def _calculate_position_weights(
        self,
        strategies: list[dict],
        target_risk_contribution: float,
        max_weight_cap: float,
    ) -> dict[str, float]:
        """
        Calculate equal-risk position weights.

        weight = target_risk_contribution / volatility, then scaled if needed.

        Args:
            strategies: List of dicts with hypothesis_id and volatility
            target_risk_contribution: Target risk per position (e.g., 0.03 for 3%)
            max_weight_cap: Maximum weight per position (e.g., 0.05 for 5%)

        Returns:
            Dict mapping hypothesis_id to weight (0-1)
        """
        weights = {}
        for strategy in strategies:
            hypothesis_id = strategy["hypothesis_id"]
            volatility = strategy.get("volatility", 0.15)  # Default 15%

            # Equal-risk: weight = target_risk / vol
            weight = target_risk_contribution / volatility
            weights[hypothesis_id] = weight

        # Check if any weights exceed cap
        max_weight = max(weights.values()) if weights else 0
        if max_weight > max_weight_cap:
            # Scale all weights proportionally to fit under cap
            scale_factor = max_weight_cap / max_weight
            weights = {k: v * scale_factor for k, v in weights.items()}

        return weights

    def _check_portfolio_constraints(self, portfolio_state: dict) -> list[str]:
        """
        Check portfolio state against constraints.

        Args:
            portfolio_state: Dict with total_weight, max_sector_weight, turnover, max_drawdown

        Returns:
            List of constraint violation messages (empty if all pass)
        """
        violations = []

        if portfolio_state.get("total_weight", 0) > self.MAX_GROSS_EXPOSURE:
            violations.append(
                f"total_weight {portfolio_state['total_weight']:.1%} > {self.MAX_GROSS_EXPOSURE:.0%}"
            )

        if portfolio_state.get("max_sector_weight", 0) > self.MAX_SECTOR_CONCENTRATION:
            violations.append(
                f"sector concentration {portfolio_state['max_sector_weight']:.1%} "
                f"> {self.MAX_SECTOR_CONCENTRATION:.0%}"
            )

        if portfolio_state.get("turnover", 0) > self.MAX_TURNOVER:
            violations.append(
                f"turnover {portfolio_state['turnover']:.1%} > {self.MAX_TURNOVER:.0%}"
            )

        if portfolio_state.get("max_drawdown", 0) > self.MAX_DRAWDOWN_LIMIT:
            violations.append(
                f"drawdown {portfolio_state['max_drawdown']:.1%} > {self.MAX_DRAWDOWN_LIMIT:.0%}"
            )

        return violations

    def _add_paper_position(
        self,
        hypothesis_id: str,
        weight: float,
        entry_price: float,
    ):
        """
        Add a position to the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being added
            weight: Position weight (0-1)
            entry_price: Entry price per share
        """
        self.api.add_paper_position(hypothesis_id, weight, entry_price)

    def _remove_paper_position(self, hypothesis_id: str):
        """
        Remove a position from the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being removed
        """
        self.api.remove_paper_position(hypothesis_id)

    def _log_paper_trade(
        self,
        hypothesis_id: str,
        action: str,
        weight_before: float,
        weight_after: float,
        price: float,
    ):
        """
        Log a simulated paper trade.

        Args:
            hypothesis_id: The hypothesis being traded
            action: One of 'ADD', 'REMOVE', 'REBALANCE'
            weight_before: Weight before trade
            weight_after: Weight after trade
            price: Execution price
        """
        self.api.log_paper_trade(hypothesis_id, action, weight_before, weight_after, price)

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICAL SIGNIFICANCE METHODS (Medallion-Grade Analysis)
    # ══════════════════════════════════════════════════════════════════════════

    def _calculate_statistical_significance(
        self, sharpe: float, n_observations: int = 252
    ) -> dict[str, Any]:
        """
        Calculate statistical significance of Sharpe ratio.

        Uses the standard error formula: SE = sqrt((1 + 0.5*SR^2) / n)
        which accounts for non-normality of returns.

        Args:
            sharpe: Observed Sharpe ratio
            n_observations: Number of observations (default 252 for 1 year daily)

        Returns:
            Dict with t_stat, p_value, confidence intervals, significance flags
        """
        n = max(n_observations, 2)
        # Standard error of Sharpe ratio (Lo 2002)
        se = np.sqrt((1 + 0.5 * sharpe**2) / n)
        t_stat = sharpe / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        ci_95 = (sharpe - 1.96 * se, sharpe + 1.96 * se)

        return {
            "sharpe": sharpe,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1],
            "significant_5pct": p_value < 0.05,
            "significant_1pct": p_value < 0.01,
            "n_observations": n,
        }

    def _calculate_deflated_sharpe(
        self, sharpe: float, n_trials: int = 10, n_observations: int = 252
    ) -> dict[str, Any]:
        """
        Calculate deflated Sharpe ratio (Bailey & Lopez de Prado 2014).

        Accounts for multiple testing / data snooping by computing the
        expected maximum Sharpe ratio under the null hypothesis.

        Args:
            sharpe: Observed Sharpe ratio
            n_trials: Number of strategies/hypotheses tested
            n_observations: Number of observations

        Returns:
            Dict with deflated Sharpe, probability of skill, haircut percentage
        """
        n = max(n_observations, 2)
        n_trials = max(n_trials, 1)

        # Expected max Sharpe under null hypothesis with n_trials independent tests
        # E[max(Z_1, ..., Z_n)] ≈ Φ^(-1)(1 - 1/n) for large n
        if n_trials > 1:
            expected_max_sharpe = stats.norm.ppf(1 - 1 / n_trials)
        else:
            expected_max_sharpe = 0

        # Deflated Sharpe = observed - expected_max
        deflated = sharpe - expected_max_sharpe

        # Standard error for probability calculation
        se = np.sqrt((1 + 0.5 * sharpe**2) / n)

        # Probability of skill (not luck) = P(true Sharpe > 0 | observed)
        if se > 0:
            prob_skill = stats.norm.cdf((sharpe - expected_max_sharpe) / se)
        else:
            prob_skill = 0.5

        # Haircut percentage (how much of observed Sharpe is likely due to luck)
        haircut_pct = (expected_max_sharpe / sharpe * 100) if sharpe > 0 else 100

        return {
            "observed_sharpe": sharpe,
            "expected_max_sharpe": expected_max_sharpe,
            "deflated_sharpe": deflated,
            "probability_of_skill": prob_skill,
            "n_trials": n_trials,
            "haircut_pct": min(haircut_pct, 100),
            "passes_deflation": deflated > 0,
        }

    def _calculate_tail_risk_metrics(
        self, returns: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Calculate tail risk metrics (VaR, CVaR, higher moments).

        Args:
            returns: List of returns (daily or periodic)

        Returns:
            Dict with VaR, CVaR, skewness, kurtosis, max loss
        """
        if not returns or len(returns) < 20:
            return {
                "var_95": None,
                "cvar_95": None,
                "var_99": None,
                "cvar_99": None,
                "skewness": None,
                "kurtosis": None,
                "max_daily_loss": None,
                "positive_days_pct": None,
            }

        arr = np.array(returns)

        # Value at Risk (5th and 1st percentile)
        var_95 = float(np.percentile(arr, 5))
        var_99 = float(np.percentile(arr, 1))

        # Conditional VaR (Expected Shortfall)
        cvar_95 = float(arr[arr <= var_95].mean()) if len(arr[arr <= var_95]) > 0 else var_95
        cvar_99 = float(arr[arr <= var_99].mean()) if len(arr[arr <= var_99]) > 0 else var_99

        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "var_99": var_99,
            "cvar_99": cvar_99,
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr)),
            "max_daily_loss": float(arr.min()),
            "positive_days_pct": float((arr > 0).mean() * 100),
        }

    def _calculate_portfolio_metrics(
        self, decisions: list[dict]
    ) -> dict[str, Any]:
        """
        Calculate combined portfolio metrics for CONTINUE decisions.

        Estimates portfolio Sharpe, correlation structure, and
        diversification benefit from combining strategies.

        Args:
            decisions: List of decision dicts with score_breakdown, experiment_data

        Returns:
            Dict with portfolio-level metrics
        """
        continue_decisions = [d for d in decisions if d.get("decision") == "CONTINUE"]

        if not continue_decisions:
            return {
                "n_strategies": 0,
                "avg_individual_sharpe": None,
                "estimated_portfolio_sharpe": None,
                "diversification_ratio": None,
                "avg_pairwise_correlation": None,
            }

        # Extract individual Sharpes
        sharpes = []
        for d in continue_decisions:
            exp_data = d.get("experiment_data", {})
            sharpe = exp_data.get("sharpe", 0)
            sharpes.append(sharpe)

        avg_sharpe = np.mean(sharpes) if sharpes else 0
        n = len(sharpes)

        # Estimate average pairwise correlation (conservative assumption)
        # In practice, we'd compute from actual returns
        avg_corr = 0.25  # Conservative default for diversified quant strategies

        # Portfolio Sharpe estimate: SR_p = SR_avg * sqrt(n) / sqrt(1 + (n-1)*rho)
        if n > 1 and avg_corr < 1:
            diversification_factor = np.sqrt(n) / np.sqrt(1 + (n - 1) * avg_corr)
        else:
            diversification_factor = 1.0

        portfolio_sharpe = avg_sharpe * diversification_factor

        return {
            "n_strategies": n,
            "avg_individual_sharpe": avg_sharpe,
            "estimated_portfolio_sharpe": portfolio_sharpe,
            "diversification_ratio": diversification_factor,
            "avg_pairwise_correlation": avg_corr,
            "sharpe_improvement_pct": ((portfolio_sharpe / avg_sharpe) - 1) * 100 if avg_sharpe > 0 else 0,
        }

    def _get_regime_context(self) -> dict[str, Any]:
        """
        Detect current market regime from available data.

        Returns regime classification based on volatility and trend.

        Returns:
            Dict with regime indicators and overall classification
        """
        # In production, this would fetch real market data
        # For now, return conservative estimates
        try:
            # Attempt to get recent SPY data for regime detection
            end_date = date.today()
            start_date = date(end_date.year, end_date.month - 2, end_date.day) if end_date.month > 2 else date(end_date.year - 1, end_date.month + 10, end_date.day)

            spy_prices = self.api.get_prices(
                symbols=["SPY"],
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            if spy_prices is not None and len(spy_prices) > 20:
                returns = spy_prices["close"].pct_change().dropna()
                vol_20d = returns.tail(20).std() * np.sqrt(252)
                sma_50 = spy_prices["close"].tail(50).mean()
                current_price = spy_prices["close"].iloc[-1]

                is_low_vol = vol_20d < 0.18
                is_bull = current_price > sma_50

                if is_low_vol and is_bull:
                    regime = "low_vol_bull"
                    regime_label = "Low-Volatility Bull Market"
                elif not is_low_vol and is_bull:
                    regime = "high_vol_bull"
                    regime_label = "High-Volatility Bull Market"
                elif is_low_vol and not is_bull:
                    regime = "low_vol_bear"
                    regime_label = "Low-Volatility Bear Market"
                else:
                    regime = "high_vol_bear"
                    regime_label = "High-Volatility Bear Market"

                return {
                    "regime": regime,
                    "regime_label": regime_label,
                    "volatility_20d": float(vol_20d),
                    "is_low_vol": is_low_vol,
                    "is_bull": is_bull,
                    "spy_vs_sma50": float((current_price / sma_50 - 1) * 100),
                    "data_available": True,
                }
        except Exception as e:
            logger.debug(f"Could not fetch regime data: {e}")

        # Default fallback
        return {
            "regime": "unknown",
            "regime_label": "Unknown (insufficient data)",
            "volatility_20d": None,
            "is_low_vol": None,
            "is_bull": None,
            "spy_vs_sma50": None,
            "data_available": False,
        }

    def _get_pipeline_statistics(self) -> dict[str, Any]:
        """
        Get hypothesis pipeline funnel statistics from lineage.

        Returns counts at each stage and conversion rates.

        Returns:
            Dict with funnel statistics
        """
        try:
            # Query lineage for hypothesis counts by status
            # In production, this would query the actual lineage table
            stats_by_status = {}

            for status in ["draft", "testing", "validated", "rejected", "deployed"]:
                hypotheses = self.api.list_hypotheses(status=status)
                stats_by_status[status] = len(hypotheses) if hypotheses else 0

            total_generated = sum(stats_by_status.values())

            return {
                "total_generated": total_generated,
                "draft": stats_by_status.get("draft", 0),
                "testing": stats_by_status.get("testing", 0),
                "validated": stats_by_status.get("validated", 0),
                "rejected": stats_by_status.get("rejected", 0),
                "deployed": stats_by_status.get("deployed", 0),
                "conversion_to_validated": (
                    stats_by_status.get("validated", 0) / total_generated * 100
                    if total_generated > 0 else 0
                ),
                "conversion_to_deployed": (
                    stats_by_status.get("deployed", 0) / total_generated * 100
                    if total_generated > 0 else 0
                ),
            }
        except Exception as e:
            logger.debug(f"Could not fetch pipeline statistics: {e}")
            return {
                "total_generated": 0,
                "draft": 0,
                "testing": 0,
                "validated": 0,
                "rejected": 0,
                "deployed": 0,
                "conversion_to_validated": 0,
                "conversion_to_deployed": 0,
            }

    def _get_hypothesis_context(self, hypothesis_id: str) -> dict[str, Any]:
        """
        Get hypothesis title and thesis for report context.

        Args:
            hypothesis_id: Hypothesis ID to look up

        Returns:
            Dict with title and thesis fields
        """
        try:
            hyp = self.api.get_hypothesis_with_metadata(hypothesis_id)
            return {
                "title": hyp.get("title", "") if hyp else "",
                "thesis": hyp.get("thesis", "") if hyp else "",
            }
        except Exception:
            return {"title": "", "thesis": ""}
