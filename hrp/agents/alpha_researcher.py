"""
Alpha Researcher Agent - SDK-powered hypothesis refinement and strategy generation.

Reviews draft hypotheses from Signal Scientist and refines them with:
- Economic rationale analysis
- Regime context awareness
- Related hypothesis search
- Strengthened falsification criteria
- NEW: Strategy generation from economic first principles

Outputs updated hypotheses, new strategy specifications, and lineage events
for ML Scientist and Quant Developer to pick up.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from hrp.agents.sdk_agent import SDKAgent, SDKAgentConfig
from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType, log_event


@dataclass
class StrategySpec:
    """Strategy specification from Alpha Researcher.

    Represents a trading strategy in economic (not code) terms.
    This is the output of Alpha Researcher's strategy generation
    and input to Quant Developer's backtesting pipeline.
    """

    name: str  # Strategy name (e.g., "earnings_momentum_post_miss")
    title: str  # Human-readable title
    economic_rationale: str  # WHY the alpha should exist
    universe: str  # Universe description (e.g., "S&P 500 ex-financials, market cap > $5B")
    long_logic: str  # Economic logic for long positions
    short_logic: str | None  # Economic logic for short positions (None if long-only)
    holding_period_days: int  # Target holding period in trading days
    rebalance_cadence: str  # Rebalance frequency (e.g., "weekly", "monthly")
    risk_constraints: dict[str, Any]  # Risk limits (sector exposure, position limits)
    regime_behavior: dict[str, str]  # Expected behavior in bull/bear/sideways regimes
    baseline_requirement: str  # Benchmark requirement (e.g., "Beat SPY by 2% annually")
    failure_modes: list[str]  # How the strategy could fail
    source: str  # Source of strategy idea (claude_ideation, literature_patterns, pattern_mining)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "title": self.title,
            "economic_rationale": self.economic_rationale,
            "universe": self.universe,
            "long_logic": self.long_logic,
            "short_logic": self.short_logic,
            "holding_period_days": self.holding_period_days,
            "rebalance_cadence": self.rebalance_cadence,
            "risk_constraints": self.risk_constraints,
            "regime_behavior": self.regime_behavior,
            "baseline_requirement": self.baseline_requirement,
            "failure_modes": self.failure_modes,
            "source": self.source,
        }


@dataclass
class AlphaResearcherConfig(SDKAgentConfig):
    """Configuration for Alpha Researcher agent."""

    # Which hypotheses to process
    hypothesis_ids: list[str] | None = None  # None = all draft hypotheses

    # Regime analysis settings
    regime_lookback_days: int = 252 * 3  # 3 years of history for regime detection

    # NEW: Strategy generation settings
    enable_strategy_generation: bool = True
    generation_target_count: int = 3  # New strategies per run
    generation_sources: list[str] = field(default_factory=lambda: [
        "claude_ideation",
        "literature_patterns",
        "pattern_mining"
    ])

    # Documentation output
    write_research_note: bool = True
    research_note_dir: str = ""  # Set from config in __post_init__
    write_strategy_docs: bool = True
    strategy_docs_dir: str = ""  # Set from config in __post_init__

    def __post_init__(self):
        if not self.research_note_dir:
            from hrp.utils.config import get_config
            self.research_note_dir = str(get_config().data.research_dir)
        if not self.strategy_docs_dir:
            from hrp.utils.config import get_config
            self.strategy_docs_dir = str(get_config().data.strategies_dir)


@dataclass
class HypothesisAnalysis:
    """Analysis result for a single hypothesis."""

    hypothesis_id: str
    economic_rationale: str
    regime_notes: str
    related_hypotheses: list[str]
    refined_thesis: str
    refined_falsification: str
    recommendation: str
    status_updated: bool = False


@dataclass
class AlphaResearcherReport:
    """Report from an Alpha Researcher run."""

    hypotheses_reviewed: int
    hypotheses_promoted: int
    hypotheses_deferred: int
    analyses: list[HypothesisAnalysis]
    research_note_path: str | None
    token_usage: dict[str, int]
    run_id: str


class AlphaResearcher(SDKAgent):
    """
    SDK Agent that reviews and refines draft hypotheses.

    Extends SDKAgent with Claude-powered reasoning to:
    1. Analyze economic rationale behind signals
    2. Check regime context (bull/bear/sideways performance)
    3. Find related hypotheses in the registry
    4. Refine thesis and falsification criteria
    5. Promote hypotheses from "draft" to "testing" status

    Example:
        researcher = AlphaResearcher(
            hypothesis_ids=["HYP-2025-001"],  # or None for all drafts
        )
        result = researcher.run()
    """

    ACTOR = "agent:alpha-researcher"

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        config: AlphaResearcherConfig | None = None,
        api: PlatformAPI | None = None,
    ):
        """
        Initialize the Alpha Researcher agent.

        Args:
            hypothesis_ids: Specific hypotheses to process (None = all drafts)
            config: Agent configuration
            api: PlatformAPI instance (created if not provided)
        """
        base_config = config or AlphaResearcherConfig(hypothesis_ids=hypothesis_ids)

        super().__init__(
            job_id="alpha_researcher",
            actor=self.ACTOR,
            config=base_config,
            dependencies=["hypotheses"],
        )

        self.hypothesis_ids = hypothesis_ids or (base_config.hypothesis_ids if config else None)
        self.api = api or PlatformAPI()
        self._analyses: list[HypothesisAnalysis] = []

    def execute(self) -> dict[str, Any]:
        """
        Execute the Alpha Researcher analysis.

        Performs:
        1. Reviews draft hypotheses (existing behavior)
        2. Generates new strategies from economic first principles (NEW)

        Returns:
            Dict with analysis results
        """
        # Part 1: Review draft hypotheses (existing behavior)
        if self.hypothesis_ids:
            hypotheses = [
                self.api.get_hypothesis(hid)
                for hid in self.hypothesis_ids
                if self.api.get_hypothesis(hid)
            ]
        else:
            hypotheses = self.api.list_hypotheses(status="draft")

        promoted = 0
        deferred = 0
        reviewed_ids = []

        if hypotheses:
            logger.info(f"Processing {len(hypotheses)} draft hypotheses")

            for hypothesis in hypotheses:
                # Checkpoint before each hypothesis
                self.checkpoint({
                    "processing": hypothesis["hypothesis_id"],
                    "completed": [a.hypothesis_id for a in self._analyses],
                })

                try:
                    analysis = self._analyze_hypothesis(hypothesis)
                    self._analyses.append(analysis)
                    reviewed_ids.append(hypothesis["hypothesis_id"])

                    if analysis.status_updated:
                        promoted += 1
                    else:
                        deferred += 1

                except Exception as e:
                    logger.error(f"Failed to analyze {hypothesis['hypothesis_id']}: {e}")
                    deferred += 1

        # Part 2: Generate new strategies (NEW)
        strategies_generated = []
        strategy_docs_written = []

        if isinstance(self.config, AlphaResearcherConfig) and self.config.enable_strategy_generation:
            logger.info(f"Generating {self.config.generation_target_count} new strategies")
            new_specs = self.generate_strategies(
                target_count=self.config.generation_target_count,
                sources=self.config.generation_sources,
            )

            for spec in new_specs:
                strategies_generated.append(spec.name)

                # Write strategy spec document
                if self.config.write_strategy_docs:
                    doc_path = self._write_strategy_spec_doc(spec)
                    if doc_path:
                        strategy_docs_written.append(doc_path)

                # Create hypothesis in registry from strategy spec
                try:
                    hyp_id = self._create_hypothesis_from_strategy(spec)
                    logger.info(f"Created hypothesis {hyp_id} from strategy {spec.name}")
                except Exception as e:
                    logger.error(f"Failed to create hypothesis for {spec.name}: {e}")

        # Write research note if configured
        research_note_path = None
        if isinstance(self.config, AlphaResearcherConfig) and self.config.write_research_note:
            research_note_path = self._write_research_note(strategies_generated=strategies_generated)

        # Log completion event (NEW)
        log_event(
            event_type=EventType.ALPHA_RESEARCHER_COMPLETE.value,
            actor=self.ACTOR,
            details={
                "hypotheses_reviewed": len(reviewed_ids),
                "hypotheses_promoted": promoted,
                "hypotheses_deferred": deferred,
                "strategies_generated": len(strategies_generated),
                "strategy_docs_written": len(strategy_docs_written),
                "reviewed_ids": reviewed_ids,
                "strategy_names": strategies_generated,
            },
        )

        return {
            "hypotheses_reviewed": len(hypotheses) if hypotheses else 0,
            "hypotheses_promoted": promoted,
            "hypotheses_deferred": deferred,
            "analyses": [
                {
                    "hypothesis_id": a.hypothesis_id,
                    "recommendation": a.recommendation,
                    "status_updated": a.status_updated,
                }
                for a in self._analyses
            ],
            "research_note_path": research_note_path,
            "strategies_generated": strategies_generated,
            "strategy_docs_written": strategy_docs_written,
            "token_usage": {
                "input": self.token_usage.input_tokens,
                "output": self.token_usage.output_tokens,
                "total": self.token_usage.total_tokens,
            },
        }

    def _analyze_hypothesis(self, hypothesis: dict) -> HypothesisAnalysis:
        """
        Analyze a single hypothesis using Claude.

        Args:
            hypothesis: Hypothesis dict from registry

        Returns:
            HypothesisAnalysis with findings
        """
        hypothesis_id = hypothesis["hypothesis_id"]
        logger.info(f"Analyzing hypothesis {hypothesis_id}")

        # Gather context for Claude
        context = self._gather_context(hypothesis)

        # Call Claude for analysis
        response = self.invoke_claude(
            prompt=self._build_analysis_prompt(hypothesis, context),
            tools=self._get_available_tools(),
        )

        # Parse Claude's response
        analysis = self._parse_analysis_response(hypothesis_id, response)

        # Update hypothesis in registry
        if analysis.recommendation.lower() in ("proceed", "promote", "testing"):
            self._update_hypothesis(hypothesis, analysis)
            analysis.status_updated = True

        # Log to lineage
        self._log_analysis_event(hypothesis_id, analysis)

        return analysis

    def _gather_context(self, hypothesis: dict) -> dict[str, Any]:
        """Gather context data for hypothesis analysis."""
        context: dict[str, Any] = {}

        # Get related hypotheses
        all_hypotheses = self.api.list_hypotheses()
        related = self._find_related_hypotheses(hypothesis, all_hypotheses)
        context["related_hypotheses"] = related

        # Try to get regime information
        try:
            from hrp.ml.regime import HMMConfig, RegimeDetector
            from hrp.research.benchmark import get_benchmark_prices

            # Get price data for regime detection
            prices = get_benchmark_prices(
                benchmark="SPY",
                start=date.today().replace(year=date.today().year - 3),
                end=date.today(),
            )

            if prices is not None and not prices.empty:
                config = HMMConfig(n_regimes=3, features=["returns_20d", "volatility_20d"])
                detector = RegimeDetector(config)
                detector.fit(prices)
                stats = detector.get_regime_statistics(prices)
                context["regime_stats"] = {
                    "current_regime": str(stats.current_regime),
                    "regime_durations": stats.regime_durations,
                }
        except Exception as e:
            logger.debug(f"Could not get regime context: {e}")
            context["regime_stats"] = None

        return context

    def _find_related_hypotheses(
        self, hypothesis: dict, all_hypotheses: list[dict]
    ) -> list[dict]:
        """Find hypotheses related to the given one."""
        related = []
        thesis = hypothesis.get("thesis", "").lower()
        title = hypothesis.get("title", "").lower()

        for other in all_hypotheses:
            if other["hypothesis_id"] == hypothesis["hypothesis_id"]:
                continue

            other_thesis = other.get("thesis", "").lower()
            other_title = other.get("title", "").lower()

            # Simple keyword matching for relatedness
            # Could be enhanced with embeddings later
            keywords = ["momentum", "volatility", "volume", "trend", "mean", "reversion"]
            shared_keywords = [
                kw
                for kw in keywords
                if (kw in thesis or kw in title) and (kw in other_thesis or kw in other_title)
            ]

            if shared_keywords:
                related.append({
                    "hypothesis_id": other["hypothesis_id"],
                    "title": other.get("title", ""),
                    "status": other.get("status", ""),
                    "shared_concepts": shared_keywords,
                })

        return related[:5]  # Limit to 5 most related

    def _build_analysis_prompt(self, hypothesis: dict, context: dict) -> str:
        """Build the analysis prompt for Claude."""
        related_section = ""
        if context.get("related_hypotheses"):
            related_items = [
                f"- {r['hypothesis_id']}: {r['title']} (status: {r['status']}, shared: {r['shared_concepts']})"
                for r in context["related_hypotheses"]
            ]
            related_section = f"""
## Related Hypotheses in Registry
{chr(10).join(related_items)}
"""

        regime_section = ""
        if context.get("regime_stats"):
            stats = context["regime_stats"]
            regime_section = f"""
## Current Market Regime
- Current regime: {stats['current_regime']}
- Regime durations: {stats['regime_durations']}
"""

        return f"""You are the Alpha Researcher agent for a quantitative research platform.
Your role is to review draft hypotheses and determine if they have sound economic rationale.

## Hypothesis to Review

**ID:** {hypothesis['hypothesis_id']}
**Title:** {hypothesis.get('title', 'N/A')}
**Thesis:** {hypothesis.get('thesis', 'N/A')}
**Testable Prediction:** {hypothesis.get('testable_prediction', 'N/A')}
**Falsification Criteria:** {hypothesis.get('falsification_criteria', 'N/A')}
**Status:** {hypothesis.get('status', 'draft')}
{related_section}
{regime_section}

## Your Analysis Tasks

1. **Economic Rationale**: Explain WHY this signal might work. What market inefficiency does it exploit? Is there academic or practitioner support for this type of signal?

2. **Regime Context**: How might this signal perform in different market regimes (bull/bear/sideways)? Is it likely to be regime-dependent?

3. **Related Hypotheses**: Consider the related hypotheses listed above. Is this hypothesis novel, or a variant of existing ideas? Any conflicts?

4. **Refined Thesis**: Improve the thesis statement with economic reasoning.

5. **Refined Falsification**: Strengthen the falsification criteria to be more specific and measurable.

6. **Recommendation**: Should this hypothesis proceed to ML testing? Options:
   - "PROCEED" - Sound rationale, ready for testing
   - "DEFER" - Needs more refinement or data before testing
   - "REJECT" - Fundamental issues with the hypothesis

## Response Format

Respond with a JSON object containing:
```json
{{
    "economic_rationale": "...",
    "regime_notes": "...",
    "related_hypothesis_notes": "...",
    "refined_thesis": "...",
    "refined_falsification": "...",
    "recommendation": "PROCEED|DEFER|REJECT",
    "recommendation_reason": "..."
}}
```
"""

    def _parse_analysis_response(
        self, hypothesis_id: str, response: dict
    ) -> HypothesisAnalysis:
        """Parse Claude's response into HypothesisAnalysis."""
        content = response.get("content", "")

        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            elif "{" in content:
                # Find JSON object
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            data = json.loads(json_str)

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"Could not parse JSON from response: {e}")
            # Create minimal analysis from raw response
            data = {
                "economic_rationale": content[:500] if content else "Analysis unavailable",
                "regime_notes": "See full analysis",
                "related_hypothesis_notes": "",
                "refined_thesis": "",
                "refined_falsification": "",
                "recommendation": "DEFER",
                "recommendation_reason": "Could not parse structured response",
            }

        return HypothesisAnalysis(
            hypothesis_id=hypothesis_id,
            economic_rationale=data.get("economic_rationale", ""),
            regime_notes=data.get("regime_notes", ""),
            related_hypotheses=self._extract_related_ids(
                data.get("related_hypothesis_notes", "")
            ),
            refined_thesis=data.get("refined_thesis", ""),
            refined_falsification=data.get("refined_falsification", ""),
            recommendation=data.get("recommendation", "DEFER"),
        )

    def _extract_related_ids(self, notes: str) -> list[str]:
        """Extract hypothesis IDs mentioned in notes."""
        import re

        pattern = r"HYP-\d{4}-\d{3}"
        return list(set(re.findall(pattern, notes)))

    def _update_hypothesis(self, hypothesis: dict, analysis: HypothesisAnalysis) -> None:
        """Update hypothesis in registry with analysis results."""
        hypothesis_id = hypothesis["hypothesis_id"]

        # Prepare metadata updates
        existing_metadata = hypothesis.get("metadata", {}) or {}
        if isinstance(existing_metadata, str):
            try:
                existing_metadata = json.loads(existing_metadata)
            except json.JSONDecodeError:
                existing_metadata = {}

        updated_metadata = {
            **existing_metadata,
            "regime_notes": analysis.regime_notes,
            "related_hypotheses": analysis.related_hypotheses,
            "alpha_researcher_review_date": datetime.now().isoformat(),
            "economic_rationale": analysis.economic_rationale[:500],  # Truncate for storage
        }

        # Update thesis if refined version provided
        refined_thesis = analysis.refined_thesis or hypothesis.get("thesis", "")

        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status="testing",
                actor=self.ACTOR,
                metadata=updated_metadata,
            )
            logger.info(f"Updated hypothesis {hypothesis_id} to 'testing' status")
        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")
            raise

    def _log_analysis_event(
        self, hypothesis_id: str, analysis: HypothesisAnalysis
    ) -> None:
        """Log analysis to lineage."""
        log_event(
            event_type=EventType.ALPHA_RESEARCHER_REVIEW.value,
            actor=self.ACTOR,
            hypothesis_id=hypothesis_id,
            details={
                "recommendation": analysis.recommendation,
                "status_updated": analysis.status_updated,
                "economic_rationale_summary": analysis.economic_rationale[:200],
                "regime_notes_summary": analysis.regime_notes[:200],
                "related_hypotheses": analysis.related_hypotheses,
            },
        )

    def _write_research_note(self, strategies_generated: list[str] | None = None) -> str | None:
        """Write research note to docs/research/."""
        if not self._analyses:
            return None

        config = self.config
        if not isinstance(config, AlphaResearcherConfig):
            return None

        from datetime import datetime as dt

        from hrp.agents.report_formatting import (
            render_footer,
            render_header,
            render_insights,
            render_kpi_dashboard,
            render_section_divider,
            render_status_table,
        )

        today = date.today().isoformat()
        promoted = sum(1 for a in self._analyses if a.status_updated)
        deferred = len(self._analyses) - promoted
        is_generation_run = bool(strategies_generated)

        if is_generation_run:
            slug = "02-alpha-researcher-generation"
            title = "Alpha Researcher — Strategy Generation"
        else:
            slug = "02-alpha-researcher-review"
            title = "Alpha Researcher — Hypothesis Review"

        parts: list[str] = []

        # ── Header ──
        parts.append(render_header(
            title=title,
            report_type="agent-execution",
            date_str=today,
        ))

        # ── KPI Dashboard ──
        kpis = [
            {"icon": "📋", "label": "Reviewed", "value": len(self._analyses), "detail": "hypotheses"},
            {"icon": "✅", "label": "Promoted", "value": promoted, "detail": "to testing"},
            {"icon": "⏸️", "label": "Deferred", "value": deferred, "detail": "needs work"},
            {"icon": "🪙", "label": "Cost", "value": f"${self.token_usage.estimated_cost_usd:.4f}", "detail": "tokens"},
        ]
        if is_generation_run:
            kpis.insert(3, {"icon": "🧬", "label": "Strategies", "value": len(strategies_generated), "detail": "generated"})
        parts.append(render_kpi_dashboard(kpis))

        # ── Summary table ──
        rows = []
        for a in self._analyses:
            status = "Promoted" if a.status_updated else "Deferred"
            rows.append([a.hypothesis_id, a.recommendation[:40], status])
        parts.append(render_status_table(
            "📋 Decision Summary",
            ["Hypothesis", "Recommendation", "Status"],
            rows,
            status_col=2,
        ))

        # ── Per-hypothesis details ──
        parts.append(render_section_divider("📊 Detailed Analysis"))

        for analysis in self._analyses:
            emoji = "✅" if analysis.status_updated else "⏸️"
            status_label = "Promoted to testing" if analysis.status_updated else "Deferred"
            parts.append(f"### {emoji} {analysis.hypothesis_id}: **{status_label}**\n")
            parts.append(f"**Recommendation:** {analysis.recommendation}\n")
            parts.append(f"**Economic Rationale:**\n{analysis.economic_rationale}\n")
            parts.append(f"**Regime Analysis:**\n{analysis.regime_notes}\n")

            related = ", ".join(analysis.related_hypotheses) if analysis.related_hypotheses else "None identified"
            parts.append(f"**Related Hypotheses:** {related}\n")

            if analysis.refined_thesis:
                parts.append(f"**Refined Thesis:**\n{analysis.refined_thesis}\n")
            if analysis.refined_falsification:
                parts.append(f"**Refined Falsification:**\n{analysis.refined_falsification}\n")

            parts.append("---\n")

        # ── Strategies Generated (generation runs only) ──
        if is_generation_run:
            parts.append(render_section_divider("🧬 Strategies Generated"))
            for name in strategies_generated:
                parts.append(f"- {name}")
            parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="alpha-researcher",
            timestamp=dt.now(),
            cost_usd=self.token_usage.estimated_cost_usd,
        ))

        content = "\n".join(parts)

        # Write to file
        from hrp.agents.output_paths import research_note_path

        filepath = str(research_note_path(slug))

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Research note written to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write research note: {e}")
            return None

    # =========================================================================
    # STRATEGY GENERATION (NEW)
    # =========================================================================

    def generate_strategies(
        self,
        target_count: int = 3,
        sources: list[str] | None = None,
    ) -> list[StrategySpec]:
        """
        Generate new strategy concepts from mixed sources.

        Args:
            target_count: Number of strategies to generate
            sources: List of generation sources (claude_ideation, literature_patterns, pattern_mining)

        Returns:
            List of StrategySpec objects
        """
        sources = sources or ["claude_ideation", "literature_patterns", "pattern_mining"]
        all_strategies: list[StrategySpec] = []

        # Generate from each source
        for source in sources:
            try:
                if source == "claude_ideation":
                    strategies = self._generate_from_claude_ideation()
                elif source == "literature_patterns":
                    strategies = self._generate_from_literature_patterns()
                elif source == "pattern_mining":
                    strategies = self._generate_from_pattern_mining()
                else:
                    logger.warning(f"Unknown generation source: {source}")
                    continue

                all_strategies.extend(strategies)
                logger.info(f"Generated {len(strategies)} strategies from {source}")

            except Exception as e:
                logger.error(f"Failed to generate strategies from {source}: {e}")

        # Return top target_count strategies
        return all_strategies[:target_count]

    def _generate_from_claude_ideation(self) -> list[StrategySpec]:
        """
        Use Claude to ideate novel strategies from first principles.

        Generates strategies based on:
        - Market inefficiencies (behavioral biases, structural factors)
        - Academic research (momentum, value, quality factors)
        - Practical considerations (execution, capacity, costs)

        Returns:
            List of StrategySpec objects
        """
        prompt = """You are the Alpha Researcher agent for a quantitative research platform.

Your task is to generate novel trading strategy concepts based on economic first principles.

## Platform Context
- Universe: S&P 500 ex-financials, long-only US equities
- Timeframe: Daily rebalancing
- Features available: 44 technical and fundamental features (momentum, volatility, volume, oscillators, trend, moving averages, ratios, fundamentals)
- Cost model: IBKR-style (5 bps commission, 10 bps slippage)

## Strategy Requirements

Generate 1-3 strategy concepts. Each strategy must include:

1. **Name**: snake_case identifier (e.g., "post_earnings_momentum")
2. **Title**: Human-readable name (e.g., "Post-Earnings Momentum")
3. **Economic Rationale**: WHY the alpha should exist (market inefficiency, behavioral effect, structural factor)
4. **Universe**: Specific universe filter (e.g., "S&P 500 ex-financials, market cap > $5B")
5. **Long Logic**: Economic logic for long positions (what to buy and why)
6. **Short Logic**: None (long-only) or short logic if applicable
7. **Holding Period**: Target holding period in trading days (5-60)
8. **Rebalance Cadence**: "weekly", "bi-weekly", or "monthly"
9. **Risk Constraints**: Dict with max positions, sector limits, position size limits
10. **Regime Behavior**: Expected performance in bull/bear/sideways markets
11. **Baseline Requirement**: Minimum performance benchmark (e.g., "Beat SPY by 2% annually")
12. **Failure Modes**: List of 3-5 conditions that would invalidate the thesis

## Response Format

Return a JSON object with a "strategies" array:

```json
{
    "strategies": [
        {
            "name": "strategy_name",
            "title": "Strategy Title",
            "economic_rationale": "...",
            "universe": "...",
            "long_logic": "...",
            "short_logic": null,
            "holding_period_days": 20,
            "rebalance_cadence": "weekly",
            "risk_constraints": {"max_positions": 20, "max_sector_exposure": 0.25, "max_position_pct": 0.05},
            "regime_behavior": {"bull": "expected outperformance", "bear": "may underperform", "sideways": "neutral"},
            "baseline_requirement": "...",
            "failure_modes": ["...", "...", "..."]
        }
    ]
}
```

Focus on economically sound strategies with clear rationale, NOT data mining or overfitting.
"""

        response = self.invoke_claude(prompt=prompt, tools=[])

        try:
            content = response.get("content", "")

            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            elif "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            data = json.loads(json_str)

            strategies = []
            for s in data.get("strategies", []):
                spec = StrategySpec(
                    name=s["name"],
                    title=s["title"],
                    economic_rationale=s["economic_rationale"],
                    universe=s["universe"],
                    long_logic=s["long_logic"],
                    short_logic=s.get("short_logic"),
                    holding_period_days=s["holding_period_days"],
                    rebalance_cadence=s["rebalance_cadence"],
                    risk_constraints=s["risk_constraints"],
                    regime_behavior=s["regime_behavior"],
                    baseline_requirement=s["baseline_requirement"],
                    failure_modes=s["failure_modes"],
                    source="claude_ideation",
                )
                strategies.append(spec)

            return strategies

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse Claude ideation response: {e}")
            return []

    def _generate_from_literature_patterns(self) -> list[StrategySpec]:
        """
        Adapt known academic factors to platform context.

        Returns:
            List of StrategySpec objects based on academic literature
        """
        # Pre-defined literature-based strategies
        # These are well-known factors from academic research
        strategies = []

        # 1. Post-earnings announcement drift (Bernard & Thomas, 1989)
        strategies.append(StrategySpec(
            name="post_earnings_drift",
            title="Post-Earnings Announcement Drift",
            economic_rationale=(
                "Markets underreact to earnings surprises due to: (1) analyst revision lag, "
                "(2) institutional flow frictions, (3) gradual information diffusion. "
                "Stocks with positive earnings surprises continue outperforming for 1-3 months."
            ),
            universe="S&P 500 ex-financials, liquid stocks with earnings data",
            long_logic="Long top decile of stocks with positive standardized earnings surprise",
            short_logic=None,  # Long-only for simplicity
            holding_period_days=20,
            rebalance_cadence="weekly",
            risk_constraints={
                "max_positions": 20,
                "max_sector_exposure": 0.25,
                "max_position_pct": 0.05,
            },
            regime_behavior={
                "bull": "strong outperformance (risk-on, surprises matter)",
                "bear": "may underperform (risk-off, earnings secondary)",
                "sideways": "moderate outperformance",
            },
            baseline_requirement="Beat equal-weight long-only by 2% annually, Sharpe > 0.8",
            failure_modes=[
                "Prolonged bear market reduces earnings signal relevance",
                "Regulatory changes affecting disclosure timing",
                "Increased hedge fund crowding in earnings trade",
                "Analyst forecast improvements reducing surprise magnitude",
            ],
            source="literature_patterns",
        ))

        # 2. Momentum continuation (Jegadeesh & Titman, 1993)
        strategies.append(StrategySpec(
            name="momentum_continuation",
            title="Momentum Continuation (3-12 month)",
            economic_rationale=(
                "Momentum persists due to: (1) delayed overreaction to news, "
                "(2) gradual information diffusion, (3) herding behavior. "
                "Past winners continue outperforming for 3-12 months before reversing."
            ),
            universe="S&P 500 ex-financials, market cap > $1B",
            long_logic="Long top decile of 12-month returns (skipping most recent month)",
            short_logic=None,  # Long-only
            holding_period_days=20,
            rebalance_cadence="monthly",
            risk_constraints={
                "max_positions": 20,
                "max_sector_exposure": 0.30,
                "max_position_pct": 0.06,
            },
            regime_behavior={
                "bull": "strong outperformance",
                "bear": "crash risk (momentum crashes in bear markets)",
                "sideways": "moderate outperformance",
            },
            baseline_requirement="Beat SPY by 3% annually, Sharpe > 0.9",
            failure_modes=[
                "Momentum crash (rapid reversal after sustained outperformance)",
                "Regime shift to mean-reversion market",
                "Increased factor crowding reducing alpha",
                "Transaction costs eating into narrow margins",
            ],
            source="literature_patterns",
        ))

        # 3. Low volatility anomaly (Baker et al., 2011)
        strategies.append(StrategySpec(
            name="low_volatility_anomaly",
            title="Low Volatility Anomaly",
            economic_rationale=(
                "Low-volatility stocks outperform on risk-adjusted basis due to: "
                "(1) benchmark constraints (institutional can't go all-in on low vol), "
                "(2) preference for lottery-like stocks, (3) misperception of risk-return tradeoff."
            ),
            universe="S&P 500 ex-financials, market cap > $5B",
            long_logic="Long bottom quintile of 60-day volatility (exclude bottom 10% to avoid illiquid stocks)",
            short_logic=None,  # Long-only
            holding_period_days=60,
            rebalance_cadence="monthly",
            risk_constraints={
                "max_positions": 30,
                "max_sector_exposure": 0.25,
                "max_position_pct": 0.04,
            },
            regime_behavior={
                "bull": "may lag high-beta stocks (FOMO benefits high vol)",
                "bear": "strong outperformance (defensive, lower drawdowns)",
                "sideways": "consistent outperformance",
            },
            baseline_requirement="Beat SPY on Sharpe ratio by 0.3, max drawdown < 15%",
            failure_modes=[
                "Prolonged bull market favors high-beta stocks",
                "Low vol stocks become overvalued (crowding)",
                "Rising rate environment hurting defensive sectors",
                "Volatility regime shift to persistently low levels",
            ],
            source="literature_patterns",
        ))

        return strategies

    def _generate_from_pattern_mining(self) -> list[StrategySpec]:
        """
        Extend patterns from existing successful hypotheses.

        Returns:
            List of StrategySpec objects based on existing hypothesis patterns
        """
        strategies = []

        # Get existing hypotheses
        existing = self.api.list_hypotheses(status="validated")

        if not existing:
            logger.info("No validated hypotheses to mine for patterns")
            return []

        # Look for patterns in successful hypotheses
        # This is a simplified implementation - could be enhanced with embeddings

        # Pattern 1: Multi-factor momentum (combine multiple momentum signals)
        strategies.append(StrategySpec(
            name="multi_factor_momentum",
            title="Multi-Factor Momentum",
            economic_rationale=(
                "Combining multiple momentum signals (price, earnings, revisions) "
                "provides more robust alpha by: (1) confirming signal across domains, "
                "(2) reducing false positives, (3) capturing different aspects of momentum."
            ),
            universe="S&P 500 ex-financials, market cap > $2B",
            long_logic="Long top quintile of composite momentum score (price + earnings + estimate revision momentum)",
            short_logic=None,
            holding_period_days=20,
            rebalance_cadence="weekly",
            risk_constraints={
                "max_positions": 25,
                "max_sector_exposure": 0.25,
                "max_position_pct": 0.05,
            },
            regime_behavior={
                "bull": "strong outperformance",
                "bear": "moderate underperformance (momentum struggles)",
                "sideways": "neutral to positive",
            },
            baseline_requirement="Beat single-factor momentum by 1.5% annually, Sharpe > 1.0",
            failure_modes=[
                "Momentum regime shift to mean-reversion",
                "Factor crowding reducing multi-factor edge",
                "Increased correlation between factors reducing diversification benefit",
                "Transaction costs from frequent rebalancing",
            ],
            source="pattern_mining",
        ))

        # Pattern 2: Mean reversion with trend filter
        strategies.append(StrategySpec(
            name="trend_filtered_reversal",
            title="Trend-Filtered Mean Reversion",
            economic_rationale=(
                "Mean reversion works best when: (1) price deviation is extreme (>2 sigma), "
                "(2) longer-term trend is neutral (not strong momentum), "
                "(3) volatility is elevated (overreaction). Trend filter prevents "
                "catching falling knives in persistent downtrends."
            ),
            universe="S&P 500 ex-financials, liquid stocks only",
            long_logic="Long oversold stocks (RSI < 30) in neutral trend regime (price near 200-day SMA)",
            short_logic=None,
            holding_period_days=10,
            rebalance_cadence="weekly",
            risk_constraints={
                "max_positions": 15,
                "max_sector_exposure": 0.20,
                "max_position_pct": 0.07,
            },
            regime_behavior={
                "bull": "moderate outperformance (oversold bounces)",
                "bear": "poor performance (trend filter prevents entries)",
                "sideways": "strong outperformance (ideal range-trading environment)",
            },
            baseline_requirement="Beat SPY by 2% in sideways markets, Sharpe > 0.7",
            failure_modes=[
                "Strong trending market (no mean reversion opportunities)",
                "Gap risk (stocks continue down after entry)",
                "Low volatility regime (no oversold conditions)",
                "High correlation reducing stock-picking edge",
            ],
            source="pattern_mining",
        ))

        return strategies

    def _write_strategy_spec_doc(self, strategy: StrategySpec) -> str | None:
        """
        Write detailed strategy specification to docs/strategies/.

        Args:
            strategy: StrategySpec to write

        Returns:
            Filepath if successful, None otherwise
        """
        if not isinstance(self.config, AlphaResearcherConfig):
            return None

        # Create directory if it doesn't exist
        os.makedirs(self.config.strategy_docs_dir, exist_ok=True)

        # Generate filename
        filename = f"{strategy.name}.md"
        filepath = os.path.join(self.config.strategy_docs_dir, filename)

        # Generate content
        content = f"""# Strategy: {strategy.title}

## Economic Rationale
{strategy.economic_rationale}

## Strategy Specification

**Universe:** {strategy.universe}
**Long Logic:** {strategy.long_logic}
**Short Logic:** {strategy.short_logic or 'Long-only (no short positions)'}
**Holding Period:** {strategy.holding_period_days} trading days
**Rebalance Cadence:** {strategy.rebalance_cadence}

### Risk Constraints
"""

        for key, value in strategy.risk_constraints.items():
            content += f"- **{key}:** {value}\n"

        content += f"""
## Regime Behavior
"""

        for regime, behavior in strategy.regime_behavior.items():
            content += f"- **{regime.capitalize()}:** {behavior}\n"

        content += f"""
## Baseline Requirement
{strategy.baseline_requirement}

## Failure Modes

This strategy may fail if:
"""

        for i, mode in enumerate(strategy.failure_modes, 1):
            content += f"{i}. {mode}\n"

        content += f"""
## Implementation Notes

**Source:** {strategy.source}
**Generated:** {datetime.now().isoformat()}
**Generated by:** Alpha Researcher (agent:alpha-researcher)

---

## Next Steps

1. **Feature Selection:** Identify required features from the 44 available
2. **Signal Definition:** Map economic logic to specific feature combinations
3. **Backtest:** Run historical backtest with realistic costs
4. **Validation:** Walk-forward validation, regime analysis
5. **Deployment:** Paper trading before live deployment

---

*This is a strategy specification from the Alpha Researcher. It describes WHAT to trade and WHY, not HOW to implement it. Implementation details are handled by the Quant Developer.*
"""

        # Write to file
        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Strategy spec written to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write strategy spec: {e}")
            return None

    def _create_hypothesis_from_strategy(self, strategy: StrategySpec) -> str | None:
        """
        Create a hypothesis in the registry from a strategy spec.

        Args:
            strategy: StrategySpec to convert to hypothesis

        Returns:
            hypothesis_id if successful, None otherwise
        """
        try:
            # Create hypothesis from strategy
            hyp_id = self.api.create_hypothesis(
                title=strategy.title,
                thesis=strategy.economic_rationale + "\n\n" + strategy.long_logic,
                prediction=strategy.baseline_requirement,
                falsification=f"Sharpe < 0.7 or fails in {len(strategy.failure_modes)}+ of the identified failure modes",
                actor=self.ACTOR,
            )

            # Add strategy spec to metadata
            self.api.update_hypothesis(
                hypothesis_id=hyp_id,
                status="draft",
                actor=self.ACTOR,
                metadata={
                    "strategy_spec": strategy.to_dict(),
                    "strategy_source": strategy.source,
                    "strategy_doc": str(Path(self.config.strategy_docs_dir) / f"{strategy.name}.md"),
                },
            )

            return hyp_id

        except Exception as e:
            logger.error(f"Failed to create hypothesis from strategy: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Claude."""
        return """You are the Alpha Researcher agent in the HRP quantitative research platform.

Your role is to review draft hypotheses created by the Signal Scientist and determine
if they have sound economic rationale before they proceed to ML testing.

You have expertise in:
- Behavioral finance and market microstructure
- Academic literature on market anomalies (momentum, value, quality factors)
- Regime-aware investing and market cycle analysis
- Statistical arbitrage and systematic trading strategies

Your analysis should be:
- Grounded in economic theory and academic research
- Practical and actionable
- Clear about when to proceed vs when to defer
- Honest about limitations and uncertainties

Remember: You are reviewing signals that have already shown statistical promise.
Your job is to add the economic reasoning layer - WHY might this work?"""

    def _get_available_tools(self) -> list[dict]:
        """Get tools available to Claude."""
        # For now, return empty list - Claude uses context provided in prompt
        # Future: Add tools for querying additional data
        return []
