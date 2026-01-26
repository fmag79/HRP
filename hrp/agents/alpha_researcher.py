"""
Alpha Researcher Agent - SDK-powered hypothesis refinement.

Reviews draft hypotheses from Signal Scientist and refines them with:
- Economic rationale analysis
- Regime context awareness
- Related hypothesis search
- Strengthened falsification criteria

Outputs updated hypotheses and lineage events for ML Scientist to pick up.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.agents.sdk_agent import SDKAgent, SDKAgentConfig
from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType, log_event


@dataclass
class AlphaResearcherConfig(SDKAgentConfig):
    """Configuration for Alpha Researcher agent."""

    # Which hypotheses to process
    hypothesis_ids: list[str] | None = None  # None = all draft hypotheses

    # Regime analysis settings
    regime_lookback_days: int = 252 * 3  # 3 years of history for regime detection

    # Research note output
    write_research_note: bool = True
    research_note_dir: str = "docs/research"


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

        Returns:
            Dict with analysis results
        """
        # Get hypotheses to process
        if self.hypothesis_ids:
            hypotheses = [
                self.api.get_hypothesis(hid)
                for hid in self.hypothesis_ids
                if self.api.get_hypothesis(hid)
            ]
        else:
            hypotheses = self.api.list_hypotheses(status="draft")

        if not hypotheses:
            logger.info("No draft hypotheses to process")
            return {
                "hypotheses_reviewed": 0,
                "hypotheses_promoted": 0,
                "hypotheses_deferred": 0,
                "analyses": [],
            }

        logger.info(f"Processing {len(hypotheses)} hypotheses")

        promoted = 0
        deferred = 0

        for hypothesis in hypotheses:
            # Checkpoint before each hypothesis
            self.checkpoint({
                "processing": hypothesis["hypothesis_id"],
                "completed": [a.hypothesis_id for a in self._analyses],
            })

            try:
                analysis = self._analyze_hypothesis(hypothesis)
                self._analyses.append(analysis)

                if analysis.status_updated:
                    promoted += 1
                else:
                    deferred += 1

            except Exception as e:
                logger.error(f"Failed to analyze {hypothesis['hypothesis_id']}: {e}")
                deferred += 1

        # Write research note if configured
        research_note_path = None
        if isinstance(self.config, AlphaResearcherConfig) and self.config.write_research_note:
            research_note_path = self._write_research_note()

        return {
            "hypotheses_reviewed": len(hypotheses),
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

            # Get price data for regime detection
            prices = self.api.get_prices(
                symbols=["SPY"],  # Use SPY as market proxy
                start=date.today().replace(
                    year=date.today().year - 3
                ),  # 3 years
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

    def _write_research_note(self) -> str | None:
        """Write research note to docs/research/."""
        if not self._analyses:
            return None

        config = self.config
        if not isinstance(config, AlphaResearcherConfig):
            return None

        # Create research note content
        today = date.today().isoformat()
        promoted = sum(1 for a in self._analyses if a.status_updated)
        deferred = len(self._analyses) - promoted

        content = f"""# Alpha Researcher Report - {today}

## Summary
- Hypotheses reviewed: {len(self._analyses)}
- Promoted to testing: {promoted}
- Deferred: {deferred}
- Token usage: {self.token_usage.total_tokens} tokens (${self.token_usage.estimated_cost_usd:.4f})

---

"""

        for analysis in self._analyses:
            status = "✅ Promoted to testing" if analysis.status_updated else "⏸️ Deferred"
            content += f"""## {analysis.hypothesis_id}

**Status:** {status}
**Recommendation:** {analysis.recommendation}

### Economic Rationale
{analysis.economic_rationale}

### Regime Analysis
{analysis.regime_notes}

### Related Hypotheses
{', '.join(analysis.related_hypotheses) if analysis.related_hypotheses else 'None identified'}

### Refined Thesis
{analysis.refined_thesis if analysis.refined_thesis else 'No refinement suggested'}

### Refined Falsification Criteria
{analysis.refined_falsification if analysis.refined_falsification else 'No refinement suggested'}

---

"""

        # Write to file
        import os

        os.makedirs(config.research_note_dir, exist_ok=True)
        filepath = f"{config.research_note_dir}/{today}-alpha-researcher.md"

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Research note written to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write research note: {e}")
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
