# Agent Definition: Alpha Researcher

**Date:** January 26, 2026
**Status:** Implemented
**Type:** SDK Agent (Claude reasoning via Agent SDK)

---

## Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Alpha Researcher |
| **Actor ID** | `agent:alpha-researcher` |
| **Type** | SDK Agent (Claude reasoning) |
| **Role** | Hypothesis refinement, economic rationale, regime analysis |
| **Implementation** | `hrp/agents/alpha_researcher.py` |
| **Trigger** | Lineage event (after Signal Scientist) + MCP on-demand |
| **Upstream** | Signal Scientist (creates draft hypotheses) |
| **Downstream** | Code Materializer (translates strategy specs) |

---

## Purpose

Reviews and refines draft hypotheses using Claude's reasoning capabilities, and generates novel strategy concepts. The Alpha Researcher:

1. **Analyzes economic rationale** - Why might this signal work?
2. **Checks regime context** - Does signal work across market conditions?
3. **Searches related hypotheses** - Novel or variant of existing idea?
4. **Promotes or rejects** hypotheses based on analysis
5. **Generates new strategies** - Creates novel strategy concepts from economic principles
6. **Writes strategy specifications** - Economic specs only, NO code generation
7. **Tags strategy class** - For adaptive IC thresholds (cross-sectional, time-series, ML)
8. **Triggers downstream** Code Materializer via lineage events

---

## Core Capabilities

### 1. Hypothesis Review Pipeline

```python
from hrp.agents import AlphaResearcher

# Review draft hypotheses
researcher = AlphaResearcher()
result = researcher.run()

print(f"Analyzed: {result['hypotheses_analyzed']}")
print(f"Promoted: {result['promoted_to_testing']}")
# Research note: docs/research/YYYY-MM-DD-alpha-researcher.md
```

### 2. Economic Rationale Analysis

For each draft hypothesis:
- Evaluates theoretical foundation
- Checks for academic/practitioner support
- Identifies market inefficiency being exploited
- Assesses economic plausibility

### 3. Regime Context Analysis

```python
# Uses HMM regime detection
regimes = api.detect_regime(prices)

# Analyzes signal performance by regime:
# - Bull markets
# - Bear markets
# - Sideways/ranging markets
```

### 4. Related Hypothesis Search

- Queries hypothesis registry for similar features/signals
- Identifies conflicting hypotheses
- Notes if idea is novel or variant of existing

### 5. Strategy Generation (NEW)

Generates novel strategy concepts from three sources:

```python
# Enable strategy generation
researcher = AlphaResearcher(
    enable_strategy_generation=True,
    generation_target_count=3,  # Number of strategies to generate
    generation_sources=[
        "claude_ideation",      # Use Claude to brainstorm novel concepts
        "literature_patterns",  # Adapt known academic factors
        "pattern_mining",       # Extend existing successful patterns
    ],
)
result = researcher.run()

print(f"Strategies generated: {result['strategies_generated']}")
# Strategy specs written to: docs/strategies/{strategy_name}/spec.md
```

**Generation Sources:**

| Source | Description | Example |
|--------|-------------|---------|
| **Claude Ideation** | Brainstorm novel concepts from economic first principles | "Post-earnings drift with sentiment filter" |
| **Literature Patterns** | Adapt published academic factors to platform constraints | "Momentum factor with volatility weighting" |
| **Pattern Mining** | Extend patterns from existing successful hypotheses | "Combine momentum_20d + low_volatility for dual strategy" |

**Note:** Alpha Researcher generates economic strategy specifications ONLY.
Code generation is handled by Code Materializer (downstream).

### 6. Strategy Classification

Tags each hypothesis with strategy class for adaptive IC thresholds:

| Strategy Class | IC Pass | IC Kill | Examples |
|----------------|---------|---------|----------|
| `cross_sectional_factor` | ≥ 0.015 | < 0.005 | Value, quality, low-vol |
| `time_series_momentum` | ≥ 0.02 | < 0.01 | Trend-following |
| `ml_composite` | ≥ 0.025 | < 0.01 | Multi-feature ML |

Updated in hypothesis metadata: `strategy_class` field.

---

## Configuration

```python
@dataclass
class AlphaResearcherConfig(SDKAgentConfig):
    hypothesis_ids: list[str] | None = None  # Specific IDs, or None for all drafts
    max_hypotheses_per_run: int = 10
    include_regime_analysis: bool = True
    include_related_search: bool = True
    auto_promote: bool = True  # Auto-promote approved hypotheses to 'testing'

    # Strategy generation settings (NEW)
    enable_strategy_generation: bool = True
    generation_target_count: int = 3
    generation_sources: list[str] = field(default_factory=lambda: [
        "claude_ideation",
        "literature_patterns",
        "pattern_mining",
    ])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hypothesis_ids` | `None` (all drafts) | Specific hypotheses to review |
| `max_hypotheses_per_run` | `10` | Limit per run (cost control) |
| `include_regime_analysis` | `True` | Analyze regime-conditional performance |
| `include_related_search` | `True` | Search for related hypotheses |
| `auto_promote` | `True` | Auto-promote approved hypotheses |
| `enable_strategy_generation` | `True` | Enable novel strategy generation |
| `generation_target_count` | `3` | Number of new strategies to generate |
| `generation_sources` | `[claude_ideation, literature_patterns, pattern_mining]` | Strategy sources |

---

## Outputs

### 1. Hypothesis Analysis

```python
@dataclass
class HypothesisAnalysis:
    hypothesis_id: str
    economic_rationale: str
    regime_notes: str | None
    related_hypotheses: list[str]
    recommendation: str  # "promote", "reject", "needs_more_data"
    confidence: float  # 0.0 - 1.0
    reasoning: str
```

### 2. Alpha Researcher Report

```python
@dataclass
class AlphaResearcherReport:
    report_date: date
    hypotheses_analyzed: int
    promoted_to_testing: int
    rejected: int
    needs_more_data: int
    analyses: list[HypothesisAnalysis]
    token_usage: TokenUsage
    duration_seconds: float
```

### 3. Hypothesis Updates

For each analyzed hypothesis:
- Updates `thesis` with economic rationale
- Updates `falsification` criteria
- Adds metadata: `regime_notes`, `related_hypotheses`, `alpha_researcher_review_date`
- Changes status: "draft" → "testing" (if promoted)

### 4. Lineage Events

- `ALPHA_RESEARCHER_REVIEW`: Per-hypothesis review event
- `AGENT_RUN_COMPLETE`: Triggers ML Scientist
- `HYPOTHESIS_STATUS_CHANGE`: Status updates logged

### 5. Research Note

- Location: `docs/research/YYYY-MM-DD-alpha-researcher.md`
- Contents: Economic analysis, regime findings, recommendations

---

## Trigger Model

### Primary: Lineage Event Trigger

```python
# Alpha Researcher listens for Signal Scientist completion
scheduler.register_lineage_trigger(
    event_type="AGENT_RUN_COMPLETE",
    actor_filter="agent:signal-scientist",
    callback=trigger_alpha_researcher,
)
```

### Secondary: MCP On-Demand

```python
# MCP tool: run_alpha_researcher
result = run_alpha_researcher(hypothesis_id="HYP-2026-042")
```

---

## SDK Agent Infrastructure

### Base Class

Extends `SDKAgent` which provides:
- Claude API integration
- Tool registration
- Cost tracking (tokens)
- Checkpoint management

### Available Tools

| Tool | Purpose | Read-only |
|------|---------|-----------|
| `get_hypothesis` | Read hypothesis details | Yes |
| `update_hypothesis` | Update thesis, status, metadata | No |
| `list_hypotheses` | Find related hypotheses | Yes |
| `get_prices` | Analyze price history | Yes |
| `get_features` | Check signal behavior | Yes |
| `detect_regime` | Get historical regime labels | Yes |

**Not available:** `run_backtest` - ML Scientist handles backtesting

### Cost Controls

| Level | Limit | Action |
|-------|-------|--------|
| Per-run | 50K input + 10K output tokens | Agent stops, logs partial |
| Daily | Aggregate across SDK agents | Scheduler pauses SDK agents |
| Per-hypothesis | Tracked per hypothesis | Expensive hypotheses flagged |

### Checkpoint & Resume

- **Granularity:** Per hypothesis
- On failure: resumes from last incomplete hypothesis
- Already-processed hypotheses not re-processed

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Claude API** | Reasoning via Agent SDK |
| **Hypothesis Registry** | Reads drafts, updates with analysis |
| **HMM Regime Detection** | `detect_regime()` for regime context |
| **MLflow** | Logs review sessions |
| **Lineage** | Logs events, triggers downstream |
| **Signal Scientist** | Upstream - creates draft hypotheses |
| **Code Materializer** | Downstream - translates strategy specs to code |

---

## Example Research Note

```markdown
# Alpha Researcher Report - 2026-01-26

## Summary
- Hypotheses reviewed: 3
- Promoted to testing: 2
- Needs more data: 1

## HYP-2026-042: momentum_20d predicts monthly returns

**Economic Rationale:**
Momentum effect is well-documented (Jegadeesh & Titman 1993). Signal likely
captures trend-following behavior and slow information diffusion.

**Regime Analysis:**
- Bull markets: IC = 0.045 (strong)
- Bear markets: IC = 0.012 (weak)
- Sideways: IC = 0.028 (moderate)
Signal is regime-dependent; works best in trending markets.

**Related Hypotheses:**
- HYP-2026-031: momentum_60d (similar, longer lookback)
- HYP-2026-018: returns_252d (annual momentum, correlated)

**Recommendation:** Proceed to ML testing with regime-aware model.
Status updated: draft → testing

---

## HYP-2026-043: rsi_14d mean reversion

**Economic Rationale:**
Mean reversion at extremes is behaviorally motivated (overreaction).
RSI captures short-term overbought/oversold conditions.

**Regime Analysis:**
- Bull markets: IC = 0.031 (moderate)
- Bear markets: IC = 0.008 (weak)
- Sideways: IC = 0.042 (strong)
Works best in ranging markets; may underperform in trends.

**Recommendation:** Proceed to ML testing.
Status updated: draft → testing

---
*Generated by Alpha Researcher (agent:alpha-researcher)*
Token usage: 12,450 input / 3,200 output
```

---

## Document History

- **2026-01-26:** Initial standalone agent definition created
- **2026-01-26:** Extracted from research-agents-design.md for standalone reference
- **2026-01-29:** Enhanced with strategy generation capabilities (3 sources: Claude ideation, literature patterns, pattern mining)
- **2026-01-29:** Updated downstream to Code Materializer (specs only, NO code generation), added strategy classification
