# Report Generator Agent Design

**Date:** January 26, 2026
**Status:** Design Approved
**Feature ID:** F-026
**Agent Type:** SDK Agent (Claude-powered)

---

## Overview

The Report Generator agent synthesizes findings from all other research agents and creates human-readable summaries for the CIO. It aggregates data from the hypotheses registry, MLflow experiments, lineage events, and feature store to produce actionable research reports.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Report Generator                          │
│                   (extends SDKAgent)                         │
└─────────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   ┌─────────┐       ┌──────────┐       ┌─────────────┐
   │  Daily  │       │  Weekly  │       │ On-demand   │
   │ Report  │       │  Report  │       │  Summary    │
   │ (7 AM)  │       │(Sun 8PM) │       │  (manual)   │
   └─────────┘       └──────────┘       └─────────────┘
```

### Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Base class | `SDKAgent` | Needs reasoning for narrative synthesis |
| Schedule | Daily (7 AM) + Weekly (Sunday 8 PM) | Quick daily updates, comprehensive weekly |
| Output location | `docs/reports/YYYY-MM-DD/` | Organized by date, version controlled |
| Filename format | `YYYY-MM-DD-HH-MM-{daily,weekly}.md` | Supports multiple runs per day |
| Data sources | Hypotheses + MLflow + Lineage + Features | Complete research picture |

---

## File Structure

```
docs/reports/
├── 2026-01-26/
│   ├── 2026-01-26-07-00-daily.md
│   ├── 2026-01-26-12-30-daily.md      (if run on-demand)
│   └── 2026-01-26-20-00-weekly.md      (Sunday only)
└── 2026-01-27/
    └── 2026-01-27-07-00-daily.md
```

**Implementation location:** `hrp/agents/report_generator.py`

---

## Daily Report Format

```markdown
# HRP Research Report - Daily 2026-01-26

## Executive Summary
- 2 new hypotheses created
- 1 hypothesis promoted to testing
- 5 ML experiments completed
- Best model: Ridge (IC=0.042, Sharpe=1.2)

---

## Hypothesis Pipeline

### New Hypotheses (Draft)
| ID | Title | Signal | IC |
|----|-------|--------|-----|
| HYP-2026-047 | Momentum predicts returns | momentum_20d | 0.045 |
| HYP-2026-048 | Volatility mean reversion | volatility_60d | 0.032 |

### Recently Promoted (→ Testing)
| ID | Title | Economic Rationale | Next Step |
|----|-------|-------------------|-----------|
| HYP-2026-042 | 20-day momentum | Trend-following, slow diffusion | ML Scientist validation |

### Validated This Week (Ready for Deployment)
| ID | Sharpe | Max DD | Stability |
|----|--------|--------|-----------|
| HYP-2026-038 | 1.45 | -12% | 0.85 ✅ |

---

## Experiment Results (Top 3)
| Experiment | Model | Sharpe | IC | Status |
|------------|-------|--------|-----|--------|
| exp-123 | Ridge | 1.45 | 0.045 | Validated |
| exp-124 | LightGBM | 1.32 | 0.038 | Testing |
| exp-125 | Lasso | 1.18 | 0.031 | Testing |

---

## Signal Analysis
- **Best new signal**: `momentum_20d` (IC=0.045, p<0.01)
- **Combination tested**: `momentum_20d + volatility_60d` (IC=0.052)

---

## Actionable Insights
1. **Review HYP-2026-038** - Passed all validation, ready for deployment decision
2. **Investigate sector rotation** - `volatility_60d` showing regime-dependent performance
3. **Compute fundamentals** - P/E ratio feature stale (last update: 2025-12-15)

---

## Agent Activity Summary
- Signal Scientist: ✅ Ran (2 signals found)
- Alpha Researcher: ✅ Ran (1 hypothesis promoted)
- ML Scientist: ⏳ Pending (1 hypothesis in queue)
- ML Quality Sentinel: ✅ Ran (no critical issues)
- Validation Analyst: ✅ Ran (HYP-2026-038 validated)

---

Generated at: 2026-01-26 07:00 ET
Token cost: $0.015
```

---

## Weekly Report Format

```markdown
# HRP Research Report - Weekly 2026-01-26

## Week at a Glance (Jan 20-26, 2026)
- **New hypotheses**: 8 created, 5 promoted to testing
- **Experiments**: 23 completed, 2 validated
- **Best finding**: HYP-2026-042 (momentum_20d, Sharpe=1.45)
- **Research spend**: $2.45 (Claude API)

---

## Pipeline Velocity

```
Signal Scientist → Alpha Researcher → ML Scientist → Validation → Deploy
    [8]  →           [5]           →     [2]    →    [2]    →   [0]
```

**Conversion funnel:**
- Draft → Testing: 63% (5/8)
- Testing → Validated: 40% (2/5)
- Validated → Deployed: 0% (0/2) ⚠️ *Deployment decision needed*

---

## Top Hypotheses This Week

### Newly Validated (Ready for Your Review)
| ID | Title | Sharpe | Max DD | Stability | Regime Robust |
|----|-------|--------|--------|-----------|---------------|
| HYP-2026-038 | 20-day momentum | 1.45 | -12% | 0.85 ✅ | Bull markets only ⚠️ |
| HYP-2026-042 | Volatility reversal | 1.28 | -18% | 0.92 ✅ | All regimes ✅ |

**Recommendation**: HYP-2026-042 is deployment-ready. HYP-2026-038 needs regime filter.

### Best Performers
| ID | IC | Sharpe | Win Rate | Profit Factor |
|----|-----|--------|----------|---------------|
| HYP-2026-038 | 0.045 | 1.45 | 52% | 1.8 |
| HYP-2026-042 | 0.038 | 1.28 | 54% | 1.6 |

---

## Experiment Insights

### Model Performance
| Model | Avg Sharpe | Best Sharpe | Win Rate |
|-------|------------|-------------|----------|
| Ridge | 1.22 | 1.45 | 68% |
| LightGBM | 1.18 | 1.32 | 62% |
| Lasso | 0.95 | 1.15 | 54% |

### Feature Importance (Top 5)
1. `momentum_20d` - 23% importance, avg IC=0.041
2. `volatility_60d` - 18% importance, avg IC=0.032
3. `rsi_14d` - 12% importance, avg IC=0.028
4. `returns_60d` - 10% importance, avg IC=0.025
5. `volume_ratio` - 8% importance, avg IC=0.022

---

## Signal Discoveries

### New Signals (IC > 0.03)
| Feature | IC | Horizon | Status |
|---------|-----|---------|--------|
| momentum_20d | 0.045 | 20d → HYP-2026-047 | Validated ✅ |
| volatility_60d | 0.032 | 20d → HYP-2026-048 | Testing |

### Signal Decay Analysis
- `momentum_20d`: IC decays from 0.045 (5d) → 0.038 (20d) → 0.022 (60d)
- Suggests: Shorter holding periods for momentum strategies

---

## Risk & Quality Summary
- **ML Quality Sentinel**: 23 experiments audited, 2 warnings (feature count > 40)
- **Validation Analyst**: 2 hypotheses stress-tested, both passed
- **Overfitting risk**: Low (avg Sharpe decay: train 1.6 → test 1.3)

---

## Action Items for You
1. **[DEPLOYMENT DECISION]** HYP-2026-038 and HYP-2026-042 await your approval
2. **[DATA]** Refresh fundamentals (stale since Dec 15)
3. **[RESEARCH]** Investigate regime-aware modeling for momentum signals
4. **[INFRA]** Consider adding sector data for better diversification analysis

---

## Next Week's Focus
- Validate remaining 3 hypotheses in testing
- Test regime-filtered momentum strategy
- Complete sector data integration (F-020)

---

Generated at: 2026-01-26 20:00 ET
Week: Jan 20-26, 2026
Token cost: $0.085 (this report)
```

---

## Implementation

### Class Structure

```python
@dataclass
class ReportGeneratorConfig(SDKAgentConfig):
    """Configuration for Report Generator agent."""

    report_type: Literal["daily", "weekly"] = "daily"
    report_dir: str = "docs/reports"
    include_sections: list[str] = field(default_factory=lambda: [
        "executive_summary", "hypothesis_pipeline", "experiments",
        "signals", "insights", "agent_activity"
    ])
    lookback_days: int = 7  # For weekly reports


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
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `execute()` | Main entry point, orchestrates report generation |
| `_gather_hypothesis_data()` | Get pipeline status from hypotheses table |
| `_gather_experiment_data()` | Fetch MLflow results for top experiments |
| `_gather_signal_data()` | Get Signal Scientist findings from lineage |
| `_gather_agent_activity()` | Query lineage for agent runs |
| `_generate_insights()` | Use Claude to synthesize actionable recommendations |
| `_render_daily_report()` | Build daily markdown from sections |
| `_render_weekly_report()` | Build weekly markdown with analysis |
| `_write_report()` | Write markdown to dated folder |

### Data Flow

```
1. GATHER DATA (parallel queries)
   ├── Hypotheses: draft, testing, validated counts
   ├── MLflow: top experiments, model performance
   ├── Lineage: agent runs, events, signal findings
   └── Features: IC values, signal statistics

2. SYNTHESIZE (Claude-powered)
   └── Generate insights + recommendations

3. RENDER (template-based)
   └── Build markdown from sections

4. OUTPUT
   └── Write to docs/reports/YYYY-MM-DD/HH-MM-{daily,weekly}.md
```

---

## Claude Integration

### Deterministic Sections (Python code)

No Claude API calls - direct data rendering:
- Hypothesis pipeline tables
- Experiment results tables
- Signal statistics
- Agent activity log
- Metadata (generated time, cost)

### Reasoning Sections (Claude-powered)

**Executive Summary**: Synthesize key highlights into 2-3 bullet points

**Actionable Insights**: "What should the CIO do next?"
- Deployment decisions
- Data quality issues
- Research opportunities
- Risk concerns

**Weekly Analysis**: Deeper synthesis for weekly reports
- Pipeline velocity analysis
- Trend identification
- Next week priorities

### Prompt Structure

```python
def _build_insights_prompt(self, context: dict, report_type: str) -> str:
    """Build prompt for Claude to generate actionable insights."""

    if report_type == "daily":
        return f"""You are the Report Generator agent for a quantitative research platform.

Synthesize the following research data into 3-5 actionable insights for the CIO.

## Current State
- Hypotheses in pipeline: {context['hypothesis_counts']}
- Top experiment: {context['top_experiment']}
- Best signal: {context['best_signal']}
- Validated awaiting deployment: {context['deployment_queue']}

## Your Task
Identify the most important actions for the CIO. Consider:
1. Deployment decisions (what's ready?)
2. Data quality issues (what's stale?)
3. Research opportunities (what's promising?)
4. Risk concerns (what needs attention?)

Return a JSON list of insights:
```json
[
    {{
        "priority": "high|medium|low",
        "category": "deployment|data|research|risk",
        "insight": "...",
        "action": "..."
    }}
]
```"""

    else:  # weekly
        return f"""You are the Report Generator agent generating a weekly research summary.

## Week Overview
- New hypotheses: {context['new_hypotheses']}
- Experiments completed: {context['experiments_completed']}
- Validated: {context['validated_count']}
- Pipeline velocity: {context['funnel']}

## Key Metrics
{context['metrics_table']}

## Your Task
Provide a comprehensive weekly analysis including:
1. Pipeline health assessment
2. Top 3 findings this week
3. Trend identification (improving, declining, stagnant)
4. Next week priorities

Return structured JSON for report generation."""
```

---

## Scheduling

### Daily Report
- **Time**: 7:00 AM ET (before market open)
- **Trigger**: Scheduled job via APScheduler
- **CLI flag**: `--with-daily-report`

### Weekly Report
- **Time**: Sunday 8:00 PM ET
- **Trigger**: Scheduled job via APScheduler
- **CLI flag**: `--with-weekly-report`

### On-Demand
- **Trigger**: MCP tool `run_report_generator(report_type="daily|weekly")`
- **Use case**: Ad-hoc summary after important runs

### Scheduler Integration

```python
# In hrp/agents/scheduler.py
def setup_daily_report(self, time: str = "07:00"):
    """Schedule daily research report."""
    self.scheduler.add_job(
        func=lambda: ReportGenerator(report_type="daily").run(),
        trigger="cron",
        hour=int(time.split(":")[0]),
        minute=int(time.split(":")[1]),
        id="daily_report",
        replace_existing=True,
    )

def setup_weekly_report(self, time: str = "20:00"):
    """Schedule weekly research report."""
    self.scheduler.add_job(
        func=lambda: ReportGenerator(report_type="weekly").run(),
        trigger="cron",
        day_of_week="sun",
        hour=int(time.split(":")[0]),
        minute=int(time.split(":")[1]),
        id="weekly_report",
        replace_existing=True,
    )
```

---

## Testing Strategy

### Test Files

| Test File | Coverage |
|-----------|----------|
| `tests/test_agents/test_report_generator.py` | All Report Generator tests |
| `tests/test_agents/test_research_agents.py` | Update for new agent |

### Key Test Cases

1. **Daily report generation**
   - Correct data gathering from all sources
   - Markdown formatting
   - File output location (dated folder)
   - Filename includes timestamp

2. **Weekly report generation**
   - Extended lookback period (7 days)
   - Pipeline funnel calculation
   - Trend analysis rendering

3. **Claude integration**
   - Insights generation (mock Claude response)
   - Prompt construction
   - JSON parsing with fallback

4. **Error handling**
   - Missing MLflow connection
   - Empty hypotheses list
   - Stale data detection

5. **Scheduler integration**
   - Daily job scheduling
   - Weekly job scheduling
   - Job execution

### Verification Commands

```bash
# Run Report Generator tests
pytest tests/test_agents/test_report_generator.py -v

# Run on-demand report
python -c "
from hrp.agents.report_generator import ReportGenerator
result = ReportGenerator(report_type='daily').run()
print(f'Report: {result[\"report_path\"]}')
"

# Verify output
ls -la docs/reports/$(date +%Y-%m-%d)/
```

---

## Success Criteria

- [ ] Daily reports generate correctly at 7 AM ET
- [ ] Weekly reports generate correctly on Sunday 8 PM ET
- [ ] Reports include all 6 sections (executive, pipeline, experiments, signals, insights, activity)
- [ ] Claude-powered insights are actionable and relevant
- [ ] File structure matches `docs/reports/YYYY-MM-DD/HH-MM-{daily,weekly}.md`
- [ ] Scheduler integration works with CLI flags
- [ ] MCP tool `run_report_generator` available for on-demand runs
- [ ] Test coverage >90%
- [ ] Full test suite passes (2,115+ tests)

---

## Dependencies

| Dependency | Required For | Status |
|------------|--------------|--------|
| `SDKAgent` base class | Claude integration, checkpoint system | ✅ Built |
| Platform API | Hypotheses, experiments, data queries | ✅ Built |
| MLflow client | Experiment results | ✅ Built |
| Lineage system | Agent activity tracking | ✅ Built |
| APScheduler | Scheduled execution | ✅ Built |

---

## Document History

- **2026-01-26:** Initial design created

