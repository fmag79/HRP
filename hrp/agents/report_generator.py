"""
Report Generator Agent - Synthesizes research findings into human-readable reports.

Generates daily and weekly research summaries by aggregating data from:
- Hypothesis registry (pipeline status)
- MLflow experiments (model performance)
- Lineage table (agent activity)
- Feature store (signal analysis)

Outputs structured markdown reports for CIO review.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from hrp.agents.sdk_agent import SDKAgent, SDKAgentConfig
from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType


@dataclass
class ReportGeneratorConfig(SDKAgentConfig):
    """Configuration for Report Generator agent."""

    # Report settings
    report_type: str = "daily"  # "daily" or "weekly"
    report_dir: str = ""  # Set from config in __post_init__

    def __post_init__(self):
        if not self.report_dir:
            from hrp.utils.config import get_config
            self.report_dir = str(get_config().data.reports_dir)

    # Content sections to include
    include_sections: list[str] = field(default_factory=lambda: [
        "executive_summary",
        "hypothesis_pipeline",
        "experiments",
        "signals",
        "insights",
        "agent_activity",
    ])

    # Lookback period for weekly reports
    lookback_days: int = 7


class ReportGenerator(SDKAgent):
    """
    SDK Agent that generates human-readable research summaries.

    Aggregates data from:
    - Hypothesis registry (pipeline status)
    - MLflow (experiment results)
    - Lineage table (agent activity)
    - Features (signal analysis)

    Outputs structured markdown reports for CIO review.

    Example:
        generator = ReportGenerator(report_type="daily")
        result = generator.run()
        # Writes to: docs/reports/2026-01-26/2026-01-26-07-00-daily.md
    """

    ACTOR = "agent:report-generator"

    def __init__(
        self,
        report_type: str = "daily",
        config: ReportGeneratorConfig | None = None,
        api: PlatformAPI | None = None,
    ):
        """
        Initialize the Report Generator agent.

        Args:
            report_type: Type of report ("daily" or "weekly")
            config: Agent configuration
            api: PlatformAPI instance (created if not provided)
        """
        base_config = config or ReportGeneratorConfig(report_type=report_type)

        super().__init__(
            job_id="report_generator",
            actor=self.ACTOR,
            config=base_config,
            dependencies=[],
        )

        self.report_type = report_type
        self.api = api or PlatformAPI()

    def execute(self) -> dict[str, Any]:
        """
        Execute the Report Generator.

        Returns:
            Dict with report_path, token_usage, and metadata
        """
        logger.info(f"Generating {self.report_type} report")

        # Gather data from all sources
        hypothesis_data = self._gather_hypothesis_data()
        experiment_data = self._gather_experiment_data()
        signal_data = self._gather_signal_data()
        agent_activity = self._gather_agent_activity()

        # Build context for insights generation
        context = {
            "hypothesis_data": hypothesis_data,
            "experiment_data": experiment_data,
            "signal_data": signal_data,
            "agent_activity": agent_activity,
            "report_type": self.report_type,
            "generated_at": datetime.now(),
        }

        # Generate AI-powered insights
        insights = self._generate_insights(context)

        # Render report
        if self.report_type == "daily":
            markdown = self._render_daily_report(context, insights)
        else:
            markdown = self._render_weekly_report(context, insights)

        # Write report to disk
        report_path = self._write_report(markdown)

        logger.info(f"Report written to {report_path}")

        return {
            "report_path": report_path,
            "report_type": self.report_type,
            "token_usage": {
                "input": self.token_usage.input_tokens,
                "output": self.token_usage.output_tokens,
                "total": self.token_usage.total_tokens,
                "estimated_cost_usd": self.token_usage.estimated_cost_usd,
            },
        }

    # ==========================================================================
    # DATA GATHERING METHODS (Task 2)
    # ==========================================================================

    def _gather_hypothesis_data(self) -> dict[str, Any]:
        """Gather hypothesis pipeline data."""
        try:
            draft = self.api.list_hypotheses(status="draft", limit=10)
            testing = self.api.list_hypotheses(status="testing", limit=10)
            validated = self.api.list_hypotheses(status="validated", limit=10)
            deployed = self.api.list_hypotheses(status="deployed", limit=10)

            return {
                "draft": draft[:5],  # Top 5 for report
                "testing": testing[:5],
                "validated": validated[:5],
                "deployed": deployed[:5],
                "total_counts": {
                    "draft": len(draft),
                    "testing": len(testing),
                    "validated": len(validated),
                    "deployed": len(deployed),
                },
            }
        except Exception as e:
            logger.warning(f"Failed to gather hypothesis data: {e}")
            return {
                "draft": [],
                "testing": [],
                "validated": [],
                "deployed": [],
                "total_counts": {"draft": 0, "testing": 0, "validated": 0, "deployed": 0},
            }

    def _gather_experiment_data(self) -> dict[str, Any]:
        """Gather MLflow experiment data."""
        try:
            # Get recent experiments from lineage
            lookback = 7 if self.report_type == "daily" else 30
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            all_events = self.api.get_lineage(limit=200)
            experiments = [
                (e["hypothesis_id"], e["experiment_id"], e["timestamp"])
                for e in all_events
                if e["event_type"] == "experiment_completed"
                and e.get("timestamp") and str(e["timestamp"]) >= start_date
            ][:20]

            top_experiments = []
            model_performance = {}

            for exp in experiments:
                hypothesis_id, experiment_id, timestamp = exp
                exp_details = self.api.get_experiment(experiment_id)
                if exp_details:
                    top_experiments.append({
                        "experiment_id": experiment_id,
                        "hypothesis_id": hypothesis_id,
                        "timestamp": timestamp,
                        "metrics": exp_details.get("metrics", {}),
                    })

                    # Track model performance
                    model = exp_details.get("params", {}).get("model", "unknown")
                    if model not in model_performance:
                        model_performance[model] = []

                    sharpe = exp_details.get("metrics", {}).get("sharpe_ratio", 0)
                    if sharpe:
                        model_performance[model].append(sharpe)

            # Calculate average performance per model
            avg_performance = {}
            for model, sharpes in model_performance.items():
                if sharpes:
                    avg_performance[model] = sum(sharpes) / len(sharpes)

            return {
                "total_experiments": len(experiments),
                "top_experiments": top_experiments[:5],
                "model_performance": avg_performance,
            }
        except Exception as e:
            logger.warning(f"Failed to gather experiment data: {e}")
            return {
                "total_experiments": 0,
                "top_experiments": [],
                "model_performance": {},
            }

    def _gather_signal_data(self) -> dict[str, Any]:
        """Gather signal discovery data from lineage."""
        try:
            lookback = 1 if self.report_type == "daily" else 7
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            # Get Signal Scientist run events
            all_events = self.api.get_lineage(limit=200)
            signal_events = [
                (e["lineage_id"], e["actor"], e["timestamp"], e["details"])
                for e in all_events
                if e["event_type"] == "agent_run_complete"
                and e.get("actor") == "agent:signal-scientist"
                and e.get("timestamp") and str(e["timestamp"]) >= start_date
            ][:5]

            recent_discoveries = []
            for event in signal_events:
                _, _, timestamp, details_json = event
                try:
                    details = json.loads(details_json) if details_json else {}
                    signals_found = details.get("signals_found", 0)
                    hypotheses_created = details.get("hypotheses_created", 0)
                    recent_discoveries.append({
                        "timestamp": timestamp,
                        "signals_found": signals_found,
                        "hypotheses_created": hypotheses_created,
                    })
                except json.JSONDecodeError:
                    pass

            # Get best signals from hypothesis metadata
            best_signals = []
            hypotheses = self.api.list_hypotheses(status="validated", limit=10)
            for hyp in hypotheses:
                metadata = hyp.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                signal_name = metadata.get("signal_feature")
                ic_value = metadata.get("signal_ic")
                if signal_name and ic_value:
                    best_signals.append({
                        "hypothesis_id": hyp["hypothesis_id"],
                        "signal": signal_name,
                        "ic": ic_value,
                    })

            return {
                "recent_discoveries": recent_discoveries,
                "best_signals": sorted(best_signals, key=lambda x: x.get("ic", 0), reverse=True)[:5],
            }
        except Exception as e:
            logger.warning(f"Failed to gather signal data: {e}")
            return {
                "recent_discoveries": [],
                "best_signals": [],
            }

    def _gather_agent_activity(self) -> dict[str, Any]:
        """Gather recent agent activity from lineage."""
        try:
            lookback = 1 if self.report_type == "daily" else 1  # Check recent activity
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            agents = {
                "signal_scientist": "agent:signal-scientist",
                "alpha_researcher": "agent:alpha-researcher",
                "ml_scientist": "agent:ml-scientist",
                "ml_quality_sentinel": "agent:ml-quality-sentinel",
                "validation_analyst": "agent:validation-analyst",
            }

            all_events = self.api.get_lineage(limit=200)

            activity = {}
            for name, actor in agents.items():
                # Check for recent successful runs
                matching = [
                    e for e in all_events
                    if e["event_type"] == "agent_run_complete"
                    and e.get("actor") == actor
                    and e.get("timestamp") and str(e["timestamp"]) >= start_date
                ]
                result = (matching[0]["timestamp"], matching[0]["details"]) if matching else None

                if result:
                    timestamp, details_json = result
                    status = "success"
                    try:
                        details = json.loads(details_json) if details_json else {}
                    except json.JSONDecodeError:
                        details = {}
                else:
                    status = "pending"
                    timestamp = None
                    details = {}

                activity[name] = {
                    "status": status,
                    "last_run": timestamp,
                    "details": details,
                }

            return activity
        except Exception as e:
            logger.warning(f"Failed to gather agent activity: {e}")
            return {
                "signal_scientist": {"status": "pending"},
                "alpha_researcher": {"status": "pending"},
                "ml_scientist": {"status": "pending"},
                "ml_quality_sentinel": {"status": "pending"},
                "validation_analyst": {"status": "pending"},
            }

    # ==========================================================================
    # CLAUDE-POWERED INSIGHTS (Task 3)
    # ==========================================================================

    def _generate_insights(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate actionable insights using Claude."""
        try:
            prompt = self._build_insights_prompt(context)
            response = self.invoke_claude(prompt=prompt)

            content = response.get("content", "")

            # Try to extract JSON from response
            insights = self._parse_insights_response(content)

            return insights if insights else self._generate_fallback_insights(context)
        except Exception as e:
            logger.warning(f"Claude insights generation failed: {e}")
            return self._generate_fallback_insights(context)

    def _parse_insights_response(self, content: str) -> list[dict[str, Any]] | None:
        """Parse Claude's JSON response into insights list."""
        try:
            # Look for JSON block in response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            elif "[" in content:
                start = content.index("[")
                end = content.rindex("]") + 1
                json_str = content[start:end]
            else:
                return None

            data = json.loads(json_str)

            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, ValueError, IndexError):
            return None

    def _generate_fallback_insights(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate fallback insights without Claude."""
        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]

        insights = []

        # Deployment insights
        validated_count = hyp_data["total_counts"]["validated"]
        if validated_count > 0:
            insights.append({
                "priority": "high",
                "category": "deployment",
                "insight": f"{validated_count} hypotheses awaiting deployment review",
                "action": "Review validated hypotheses for deployment approval",
            })

        # Pipeline insights
        draft_count = hyp_data["total_counts"]["draft"]
        testing_count = hyp_data["total_counts"]["testing"]
        if draft_count > 5:
            insights.append({
                "priority": "medium",
                "category": "research",
                "insight": f"{draft_count} hypotheses in draft queue",
                "action": "Consider running Alpha Researcher to promote promising candidates",
            })

        # Experiment insights
        if exp_data["total_experiments"] > 0:
            insights.append({
                "priority": "low",
                "category": "research",
                "insight": f"{exp_data['total_experiments']} experiments completed since last report",
                "action": "Review ML Quality Sentinel audits for any issues",
            })

        return insights if insights else [{
            "priority": "low",
            "category": "general",
            "insight": "No critical items requiring attention",
            "action": "Research pipeline operating normally",
        }]

    def _build_insights_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for Claude to generate insights."""
        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]
        signal_data = context["signal_data"]

        # Build context summary
        validated_awaiting = hyp_data["total_counts"]["validated"]
        top_experiment = exp_data["top_experiments"][0] if exp_data["top_experiments"] else None
        best_signal = signal_data["best_signals"][0] if signal_data["best_signals"] else None

        prompt = f"""You are the Report Generator agent for a quantitative research platform.

Synthesize the following research data into 3-5 actionable insights for the CIO.

## Current State
- Hypotheses in pipeline: Draft={hyp_data['total_counts']['draft']}, Testing={hyp_data['total_counts']['testing']}, Validated={validated_awaiting}, Deployed={hyp_data['total_counts']['deployed']}
- Experiments completed: {exp_data['total_experiments']}
"""

        if top_experiment:
            prompt += f"- Top experiment: {top_experiment['experiment_id']} (Sharpe: {top_experiment['metrics'].get('sharpe_ratio', 'N/A')})\n"

        if best_signal:
            prompt += f"- Best signal: {best_signal['signal']} (IC: {best_signal['ic']})\n"

        prompt += """
## Your Task
Identify the most important actions for the CIO. Consider:
1. Deployment decisions (what's ready?)
2. Data quality issues (what's stale?)
3. Research opportunities (what's promising?)
4. Risk concerns (what needs attention?)

Return a JSON list of insights:
```json
[
    {
        "priority": "high|medium|low",
        "category": "deployment|data|research|risk",
        "insight": "...",
        "action": "..."
    }
]
```
"""
        return prompt

    # ==========================================================================
    # REPORT RENDERING (Task 4)
    # ==========================================================================

    def _render_daily_report(
        self, context: dict[str, Any], insights: list[dict[str, Any]]
    ) -> str:
        """Render daily report markdown with institutional formatting."""
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_insights, render_agent_activity, render_alert_banner,
            render_status_table, render_section_divider, format_metric,
        )

        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]
        signal_data = context["signal_data"]
        agent_activity = context["agent_activity"]
        generated_at = context["generated_at"]

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Daily Research Report",
            report_type="daily",
            date_str=generated_at.strftime("%Y-%m-%d"),
        ))

        # ── KPI Dashboard ──
        draft_count = hyp_data["total_counts"]["draft"]
        testing_count = hyp_data["total_counts"]["testing"]
        validated_count = hyp_data["total_counts"]["validated"]
        deployed_count = hyp_data["total_counts"]["deployed"]
        exp_count = exp_data["total_experiments"]

        top_sharpe = "N/A"
        if exp_data["top_experiments"]:
            top_exp = exp_data["top_experiments"][0]
            raw_sharpe = top_exp["metrics"].get("sharpe_ratio", "N/A")
            top_sharpe = format_metric(raw_sharpe, "f2")

        parts.append(render_kpi_dashboard([
            {"icon": "📝", "label": "Draft", "value": draft_count, "detail": "hypotheses"},
            {"icon": "🧪", "label": "Testing", "value": testing_count, "detail": "in progress"},
            {"icon": "✅", "label": "Validated", "value": validated_count, "detail": "ready"},
            {"icon": "🚀", "label": "Deployed", "value": deployed_count, "detail": "live"},
        ]))

        # ── Alert banner for validated hypotheses awaiting deployment ──
        alerts = []
        if validated_count > 0:
            alerts.append(f"{validated_count} hypotheses VALIDATED and awaiting deployment review")
        if exp_count == 0:
            alerts.append("No ML experiments completed today — check pipeline status")
        if alerts:
            parts.append(render_alert_banner(alerts, severity="warning" if exp_count > 0 else "critical"))

        # ── Executive Summary ──
        parts.append("## 📋 Executive Summary\n")
        parts.append(f"- 📝 **{draft_count}** new hypotheses in draft")
        parts.append(f"- 🧪 **{testing_count}** hypotheses in testing")
        parts.append(f"- 🔬 **{exp_count}** ML experiments completed")
        parts.append(f"- 📈 **Best Sharpe**: {top_sharpe}")
        parts.append("")

        # ── Hypothesis Pipeline ──
        parts.append(render_section_divider("📊 Hypothesis Pipeline"))

        if hyp_data["draft"]:
            rows = []
            for hyp in hyp_data["draft"][:5]:
                metadata = hyp.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                signal = metadata.get("signal_feature", "—")
                ic = format_metric(metadata.get("signal_ic"), "f3") if metadata.get("signal_ic") else "—"
                rows.append([hyp["hypothesis_id"], hyp.get("title", "N/A")[:40], signal, ic])
            parts.append(render_status_table(
                "📝 New Hypotheses (Draft)",
                ["ID", "Title", "Signal", "IC"],
                rows,
            ))

        if hyp_data["validated"]:
            rows = []
            for hyp in hyp_data["validated"][:5]:
                rows.append([hyp["hypothesis_id"], hyp.get("title", "N/A")[:40], "See MLflow", "Validated"])
            parts.append(render_status_table(
                "✅ Validated (Ready for Deployment)",
                ["ID", "Title", "Sharpe", "Status"],
                rows,
                status_col=3,
            ))

        # ── Experiment Results ──
        if exp_data["top_experiments"]:
            rows = []
            for exp in exp_data["top_experiments"][:3]:
                exp_id = exp["experiment_id"][:12]
                model = exp.get("metrics", {}).get("model", "N/A")
                sharpe = format_metric(exp["metrics"].get("sharpe_ratio"), "f2")
                rows.append([exp_id, model, sharpe, "N/A", "🧪 Testing"])
            parts.append(render_status_table(
                "🔬 Top Experiments",
                ["Experiment", "Model", "Sharpe", "IC", "Status"],
                rows,
            ))
        else:
            parts.append("### 🔬 Top Experiments\n\n> _No experiments completed in this period_\n")

        # ── Signal Analysis ──
        parts.append(render_section_divider("📡 Signal Analysis"))
        if signal_data["best_signals"]:
            parts.append("| Rank | Signal | IC | Hypothesis |")
            parts.append("|------|--------|-----|------------|")
            for i, sig in enumerate(signal_data["best_signals"][:5], 1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}.")
                parts.append(f"| {medal} | `{sig['signal']}` | {format_metric(sig['ic'], 'f3')} | {sig['hypothesis_id']} |")
            parts.append("")
        else:
            parts.append("> _No new signal discoveries_\n")

        # ── Actionable Insights ──
        parts.append(render_insights("Actionable Insights", insights))

        # ── Agent Activity ──
        parts.append(render_agent_activity(agent_activity))

        # ── Footer ──
        parts.append(render_footer(
            agent_name="report-generator",
            timestamp=generated_at,
            cost_usd=self.token_usage.estimated_cost_usd,
        ))

        return "\n".join(parts)

    def _render_weekly_report(
        self, context: dict[str, Any], insights: list[dict[str, Any]]
    ) -> str:
        """Render weekly report markdown with institutional formatting."""
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_pipeline_flow, render_insights, render_health_gauges,
            render_status_table, render_section_divider, render_alert_banner,
            render_progress_bar, format_metric,
        )

        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]
        signal_data = context["signal_data"]
        generated_at = context["generated_at"]

        week_start = (generated_at - timedelta(days=7)).strftime("%B %d, %Y")
        week_end = generated_at.strftime("%B %d, %Y")

        draft_count = hyp_data["total_counts"]["draft"]
        testing_count = hyp_data["total_counts"]["testing"]
        validated_count = hyp_data["total_counts"]["validated"]
        deployed_count = hyp_data["total_counts"]["deployed"]
        exp_count = exp_data["total_experiments"]
        total_hyp = draft_count + testing_count + validated_count + deployed_count

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Weekly Research Report",
            report_type="weekly",
            date_str=generated_at.strftime("%Y-%m-%d"),
            subtitle=f"📅 Week: {week_start} → {week_end}",
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📝", "label": "Hypotheses", "value": total_hyp, "detail": f"+{draft_count} new"},
            {"icon": "🧪", "label": "Experiments", "value": exp_count, "detail": "completed"},
            {"icon": "✅", "label": "Validated", "value": validated_count, "detail": "ready"},
            {"icon": "💰", "label": "API Cost", "value": f"${self.token_usage.estimated_cost_usd:.2f}", "detail": "Claude API"},
        ]))

        # ── Alert banner for validated awaiting deployment ──
        if validated_count > 0:
            parts.append(render_alert_banner(
                [f"{validated_count} hypotheses VALIDATED and awaiting deployment review",
                 "Action: Review in CIO dashboard for paper portfolio allocation"],
                severity="warning",
            ))

        # ── Pipeline Flow ──
        parts.append(render_pipeline_flow([
            {"icon": "📡", "label": "Signal", "count": draft_count},
            {"icon": "🔬", "label": "Research", "count": testing_count},
            {"icon": "🧪", "label": "ML Train", "count": exp_count},
            {"icon": "✅", "label": "Validate", "count": validated_count},
            {"icon": "🚀", "label": "Deploy", "count": deployed_count},
        ]))

        # ── Health Gauges ──
        # Calculate pipeline health metrics
        pipeline_throughput = (validated_count / max(total_hyp, 1)) * 100
        experiment_rate = min((exp_count / max(testing_count, 1)) * 100, 100)

        parts.append(render_health_gauges([
            {"label": "Pipeline Throughput", "value": pipeline_throughput, "max_val": 100,
             "trend": "up" if validated_count > 0 else "stable"},
            {"label": "Experiment Rate", "value": experiment_rate, "max_val": 100,
             "trend": "up" if exp_count > 0 else "down"},
            {"label": "Deployment Ready", "value": validated_count, "max_val": max(total_hyp, 1),
             "trend": "up" if validated_count > 0 else "stable"},
        ]))

        # ── Top Hypotheses ──
        if hyp_data["validated"]:
            rows = []
            for hyp in hyp_data["validated"][:5]:
                rows.append([hyp["hypothesis_id"], hyp.get("title", "N/A")[:50], "Validated"])
            parts.append(render_status_table(
                "✅ Newly Validated (Ready for Your Review)",
                ["ID", "Title", "Status"],
                rows,
                status_col=2,
            ))

        # ── Model Performance ──
        if exp_data["model_performance"]:
            parts.append(render_section_divider("🧪 Model Performance"))
            parts.append("| Model | Avg Sharpe | Rating |")
            parts.append("|-------|-----------|--------|")
            for model, avg_sharpe in sorted(
                exp_data["model_performance"].items(), key=lambda x: x[1], reverse=True
            ):
                bar = render_progress_bar(avg_sharpe, 2.0, width=10, show_pct=False)
                rating = "🟢" if avg_sharpe >= 1.0 else "🟡" if avg_sharpe >= 0.5 else "🔴"
                parts.append(f"| **{model}** | {avg_sharpe:.2f} | {bar} {rating} |")
            parts.append("")

        # ── Signal Discoveries ──
        if signal_data["best_signals"]:
            parts.append(render_section_divider("📡 Signal Discoveries"))
            parts.append("| Rank | Feature | IC | Hypothesis |")
            parts.append("|------|---------|-----|------------|")
            for i, sig in enumerate(signal_data["best_signals"][:5], 1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}.")
                parts.append(f"| {medal} | `{sig['signal']}` | {format_metric(sig['ic'], 'f3')} | {sig['hypothesis_id']} |")
            parts.append("")

        # ── Action Items ──
        parts.append(render_insights("Action Items for You", insights))

        # ── Footer ──
        parts.append(render_footer(
            agent_name="report-generator",
            timestamp=generated_at,
            cost_usd=self.token_usage.estimated_cost_usd,
            extra_lines=[f"📅 Week: {week_start} → {week_end}"],
        ))

        return "\n".join(parts)

    def _get_report_filename(self) -> str:
        """Generate report filename with timestamp."""
        now = datetime.now()
        return f"{now.strftime('%Y-%m-%d-%H-%M')}-{self.report_type}.md"

    # ==========================================================================
    # REPORT WRITING (Task 5)
    # ==========================================================================

    def _write_report(self, markdown: str) -> str:
        """Write report to dated folder."""
        try:
            # Create dated directory
            now = datetime.now()
            date_dir = now.strftime("%Y-%m-%d")
            report_dir = self.config.report_dir

            # Build full path
            dir_path = os.path.join(report_dir, date_dir)
            os.makedirs(dir_path, exist_ok=True)

            # Generate filename
            filename = self._get_report_filename()
            filepath = os.path.join(dir_path, filename)

            # Write markdown
            with open(filepath, "w") as f:
                f.write(markdown)

            logger.info(f"Report written to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to write report: {e}")
            return ""

    # ==========================================================================
    # CLAUDE INTEGRATION
    # ==========================================================================

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Claude."""
        return """You are the Report Generator agent in the HRP quantitative research platform.

Your role is to synthesize research findings from multiple agents and data sources
into clear, actionable summaries for the CIO.

You have expertise in:
- Quantitative research methodologies
- Statistical analysis and model evaluation
- Systematic trading strategies
- Research pipeline management

Your reports should be:
- Clear and concise
- Data-driven and factual
- Focused on actionable insights
- Honest about limitations and uncertainties

Remember: Your audience is the CIO who needs to make deployment decisions.
Highlight what's ready, what needs attention, and what the next steps should be."""

    def _get_available_tools(self) -> list[dict]:
        """Get tools available to Claude."""
        # Report Generator doesn't need tools - uses context provided in prompt
        return []
