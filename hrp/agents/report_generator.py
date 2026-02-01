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
from hrp.data.db import get_db
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
            db = get_db()

            # Get recent experiments from lineage
            lookback = 7 if self.report_type == "daily" else 30
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            experiments = db.fetchall(
                """
                SELECT DISTINCT hypothesis_id, experiment_id, timestamp
                FROM lineage
                WHERE event_type = 'experiment_completed'
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                (start_date,),
            )

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
            db = get_db()

            lookback = 1 if self.report_type == "daily" else 7
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            # Get Signal Scientist run events
            signal_events = db.fetchall(
                """
                SELECT lineage_id, actor, timestamp, details
                FROM lineage
                WHERE event_type = 'agent_run_complete'
                  AND actor = 'agent:signal-scientist'
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 5
                """,
                (start_date,),
            )

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
            db = get_db()

            lookback = 1 if self.report_type == "daily" else 1  # Check recent activity
            start_date = (date.today() - timedelta(days=lookback)).isoformat()

            agents = {
                "signal_scientist": "agent:signal-scientist",
                "alpha_researcher": "agent:alpha-researcher",
                "ml_scientist": "agent:ml-scientist",
                "ml_quality_sentinel": "agent:ml-quality-sentinel",
                "validation_analyst": "agent:validation-analyst",
            }

            activity = {}
            for name, actor in agents.items():
                # Check for recent successful runs
                result = db.fetchone(
                    """
                    SELECT timestamp, details
                    FROM lineage
                    WHERE event_type = 'agent_run_complete'
                      AND actor = ?
                      AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (actor, start_date),
                )

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
        """Render daily report markdown."""
        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]
        signal_data = context["signal_data"]
        agent_activity = context["agent_activity"]
        generated_at = context["generated_at"]

        lines = []
        lines.append(f"# HRP Research Report - Daily {generated_at.strftime('%Y-%m-%d')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        draft_count = hyp_data["total_counts"]["draft"]
        testing_count = hyp_data["total_counts"]["testing"]
        exp_count = exp_data["total_experiments"]

        lines.append(f"- {draft_count} new hypotheses created")
        lines.append(f"- {testing_count} hypotheses in testing")
        lines.append(f"- {exp_count} ML experiments completed")

        if exp_data["top_experiments"]:
            top_exp = exp_data["top_experiments"][0]
            top_sharpe = top_exp["metrics"].get("sharpe_ratio", "N/A")
            lines.append(f"- Best model Sharpe: {top_sharpe:.2f}" if isinstance(top_sharpe, (int, float)) else f"- Best model Sharpe: {top_sharpe}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Hypothesis Pipeline
        lines.append("## Hypothesis Pipeline")
        lines.append("")

        if hyp_data["draft"]:
            lines.append("### New Hypotheses (Draft)")
            lines.append("| ID | Title | Signal | IC |")
            lines.append("|----|-------|--------|-----|")
            for hyp in hyp_data["draft"][:5]:
                metadata = hyp.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                signal = metadata.get("signal_feature", "N/A")
                ic = metadata.get("signal_ic", "N/A")
                lines.append(f"| {hyp['hypothesis_id']} | {hyp.get('title', 'N/A')[:40]} | {signal} | {ic} |")
            lines.append("")

        if hyp_data["validated"]:
            lines.append("### Validated This Week (Ready for Deployment)")
            lines.append("| ID | Title | Sharpe | Stability |")
            lines.append("|----|-------|--------|-----------|")
            for hyp in hyp_data["validated"][:5]:
                lines.append(f"| {hyp['hypothesis_id']} | {hyp.get('title', 'N/A')[:40]} | See MLflow | ✅ |")
            lines.append("")

        # Experiment Results
        lines.append("## Experiment Results (Top 3)")
        lines.append("| Experiment | Model | Sharpe | IC | Status |")
        lines.append("|------------|-------|--------|-----|--------|")
        for exp in exp_data["top_experiments"][:3]:
            exp_id = exp["experiment_id"][:12]
            model = exp.get("metrics", {}).get("model", "N/A")
            sharpe = exp["metrics"].get("sharpe_ratio", "N/A")
            lines.append(f"| {exp_id} | {model} | {sharpe} | N/A | Testing |")
        lines.append("")

        # Signal Analysis
        lines.append("## Signal Analysis")
        if signal_data["best_signals"]:
            best = signal_data["best_signals"][0]
            lines.append(f"- **Best validated signal**: `{best['signal']}` (IC={best['ic']})")
        else:
            lines.append("- No new signal discoveries")
        lines.append("")

        # Actionable Insights
        lines.append("## Actionable Insights")
        for i, insight in enumerate(insights, 1):
            priority_emoji = "🔴" if insight["priority"] == "high" else "🟡" if insight["priority"] == "medium" else "🟢"
            category = insight["category"].upper()
            lines.append(f"{i}. **{priority_emoji} [{category}]** {insight['action']}")
        lines.append("")

        # Agent Activity Summary
        lines.append("## Agent Activity Summary")
        for agent_name, activity in agent_activity.items():
            status_emoji = "✅" if activity["status"] == "success" else "⏳"
            lines.append(f"- {agent_name.replace('_', ' ').title()}: {status_emoji}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M')} ET")
        lines.append(f"Token cost: ${self.token_usage.estimated_cost_usd:.4f}")

        return "\n".join(lines)

    def _render_weekly_report(
        self, context: dict[str, Any], insights: list[dict[str, Any]]
    ) -> str:
        """Render weekly report markdown."""
        hyp_data = context["hypothesis_data"]
        exp_data = context["experiment_data"]
        signal_data = context["signal_data"]
        generated_at = context["generated_at"]

        lines = []
        lines.append(f"# HRP Research Report - Weekly {generated_at.strftime('%Y-%m-%d')}")
        lines.append("")

        # Week at a Glance
        week_start = (generated_at - timedelta(days=7)).strftime("%B %d, %Y")
        lines.append(f"## Week at a Glance ({week_start} - {generated_at.strftime('%B %d, %Y')})")

        draft_count = hyp_data["total_counts"]["draft"]
        testing_count = hyp_data["total_counts"]["testing"]
        validated_count = hyp_data["total_counts"]["validated"]
        exp_count = exp_data["total_experiments"]

        lines.append(f"- **New hypotheses**: {draft_count} created")
        lines.append(f"- **Hypotheses in testing**: {testing_count}")
        lines.append(f"- **Validated**: {validated_count}")
        lines.append(f"- **Experiments**: {exp_count} completed")
        lines.append(f"- **Research spend**: ${self.token_usage.estimated_cost_usd:.2f} (Claude API)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Pipeline Velocity
        lines.append("## Pipeline Velocity")
        lines.append("")
        lines.append("```")
        lines.append("Signal Scientist → Alpha Researcher → ML Scientist → Validation → Deploy")
        lines.append(f"    [{draft_count}]  →     [{testing_count}]           →     [{exp_count}]    →    [{validated_count}]    →   [{hyp_data['total_counts']['deployed']}]")
        lines.append("```")
        lines.append("")

        # Top Hypotheses
        if hyp_data["validated"]:
            lines.append("## Top Hypotheses This Week")
            lines.append("")
            lines.append("### Newly Validated (Ready for Your Review)")
            lines.append("| ID | Title | Status |")
            lines.append("|----|-------|--------|")
            for hyp in hyp_data["validated"][:5]:
                lines.append(f"| {hyp['hypothesis_id']} | {hyp.get('title', 'N/A')[:50]} | ✅ Validated |")
            lines.append("")

        # Model Performance
        if exp_data["model_performance"]:
            lines.append("## Experiment Insights")
            lines.append("")
            lines.append("### Model Performance")
            lines.append("| Model | Avg Sharpe |")
            lines.append("|-------|------------|")
            for model, avg_sharpe in sorted(exp_data["model_performance"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {model} | {avg_sharpe:.2f} |")
            lines.append("")

        # Signal Discoveries
        if signal_data["best_signals"]:
            lines.append("## Signal Discoveries")
            lines.append("")
            lines.append("### Best Validated Signals")
            lines.append("| Feature | IC | Hypothesis |")
            lines.append("|---------|-----|------------|")
            for sig in signal_data["best_signals"][:5]:
                lines.append(f"| `{sig['signal']}` | {sig['ic']:.3f} | {sig['hypothesis_id']} |")
            lines.append("")

        # Action Items
        lines.append("## Action Items for You")
        for i, insight in enumerate(insights, 1):
            priority_emoji = "🔴" if insight["priority"] == "high" else "🟡" if insight["priority"] == "medium" else "🟢"
            lines.append(f"{i}. **{priority_emoji}** {insight['action']}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M')} ET")
        lines.append(f"Week: {week_start} - {generated_at.strftime('%B %d, %Y')}")
        lines.append(f"Token cost: ${self.token_usage.estimated_cost_usd:.4f}")

        return "\n".join(lines)

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
