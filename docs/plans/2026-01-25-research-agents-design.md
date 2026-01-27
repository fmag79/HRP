# Research Agents Design

**Date:** January 25, 2026
**Status:** Implementation In Progress (8/12 steps complete)
**Related:** Tier 2 Intelligence (85% complete) - Research Agents feature

---

## Goal

Build a multi-agent quant research team that runs autonomously and coordinates through a shared workspace. Agents simulate roles found at top quantitative hedge funds (Two Sigma, DE Shaw, Renaissance, Citadel).

---

## Final Design Decisions

| Decision | Choice |
|----------|--------|
| Agent count | 8 agents (Option A - lean) |
| Implementation | Case-by-case: 3 SDK, 3 Custom, 1 Hybrid, 2 Already Built |
| Build order | ~~Alpha Researcher~~ ✅ → ML Quality Sentinel → Validation Analyst |
| Communication | Shared registries only (no direct agent-to-agent) |
| Scheduling | APScheduler (extend existing infrastructure) |
| SDK infrastructure | Shared `SDKAgent` base class |

### Agent Implementation Matrix

| Agent | Type | Status | Reasoning |
|-------|------|--------|-----------|
| Signal Scientist | Custom | ✅ Built | Deterministic: calculate IC, threshold check |
| ML Scientist | Custom | ✅ Built | Deterministic: walk-forward validation pipeline |
| Alpha Researcher | SDK | ✅ Built | Needs reasoning: "is this pattern meaningful?" |
| ML Quality Sentinel | Custom | To build (2nd) | Deterministic: run checklist of validations |
| Validation Analyst | Hybrid | To build (3rd) | Mix: deterministic tests + reasoning for edge cases |
| Risk Manager | Custom | To build | Deterministic: check limits, flag violations |
| Quant Developer | SDK | To build | Needs reasoning: "how to implement this strategy?" |
| Report Generator | SDK | To build | Needs reasoning: "what's the narrative here?" |

### Coordination Model
**Decision:** Autonomous with shared workspace
- Each agent works independently
- Findings shared to common registry (hypotheses, MLflow, lineage)
- Other agents pick up relevant work
- User (CIO) reviews validated findings for final approval

### Execution Model
**Decision:** Both scheduled and on-demand
- Scheduled runs via APScheduler (existing infrastructure)
- Manual triggers via MCP tools for ad-hoc research

---

## Research Findings: Real Hedge Fund Structures

### Organizational Models

1. **Centralized/Collaborative** (DE Shaw, Two Sigma, RenTech)
   - Single cohesive team, shared knowledge base
   - Collaboration across disciplines
   - Best fit for our shared workspace model

2. **Pod Structure** (Citadel, Millennium)
   - Independent trading teams with separate P&Ls
   - Not suitable for personal research platform

### Key Insight: Risk Independence

> "The first sign of a powerless CRO is direct reporting to a single portfolio manager."
> — The Hedge Fund Journal

Risk management must be independent from alpha generation to prevent conflicts of interest.

### Sources
- [D.E. Shaw – the quant king](https://rupakghose.substack.com/p/de-shaw-the-quant-king)
- [The Hedge Fund Journal - Risk Practices](https://thehedgefundjournal.com/risk-practices-in-hedge-funds/)
- [QuantStart - Getting a Job at a Top Tier Quant Fund](https://www.quantstart.com/articles/Getting-a-Job-in-a-Top-Tier-Quant-Hedge-Fund/)
- [Street of Walls - Role of the Quantitative Analyst](https://www.streetofwalls.com/finance-training-courses/quantitative-hedge-fund-training/role-of-quant-analyst/)

---

## Original Proposal: 15 Agents (5 Pods)

```
CIO / Research Director (final authority)
│
├── Quantitative Research Pod
│   ├── Alpha Researcher (Quant)
│   ├── Feature & Signal Scientist
│   └── Regime & Market Structure Analyst
│
├── ML & Data Science Pod
│   ├── ML Scientist (Modeling)
│   ├── Training Reliability Auditor
│   └── Data Integrity & Leakage Sentinel
│
├── Engineering Pod
│   ├── Quant Developer (Infra & Performance)
│   ├── Pipeline & Experiment Orchestrator
│   └── Dashboard & UX Architect
│
├── Risk & Evaluation Pod
│   ├── Risk Manager Agent
│   ├── Execution & Cost Realism Analyst
│   └── Model Validation & Stress Tester
│
└── Deployment & Ops Pod
    ├── Paper Trading Operator
    ├── Live Trading Readiness Agent
    └── Monitoring & Kill-Switch Agent
```

### Strengths
- Good separation of concerns into pods
- Risk & Evaluation Pod is independent (matches best practice)
- ML Pod has specialized audit roles (Training Reliability, Leakage Sentinel)
- Regime Analyst addresses market condition awareness
- Forward-thinking Deployment Pod for Tier 4

### Concerns
1. **15 agents is high** - Coordination overhead, some may be idle
2. **Potential overlap** between agents (see consolidation analysis)
3. **Deployment Pod premature** - HRP at Tier 2, paper trading is Tier 4
4. **Dashboard & UX Architect** - Project work, not ongoing agent

---

## Consolidation Analysis

### Quantitative Research Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Alpha Researcher | Keep | Core hypothesis development |
| Feature & Signal Scientist | Keep | Distinct: signal discovery vs strategy design |
| Regime & Market Structure Analyst | Merge into Alpha Researcher | Can be a capability, not separate agent |

**Result:** 3 → 2 agents

### ML & Data Science Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| ML Scientist | Keep | Core modeling work |
| Training Reliability Auditor | Merge | Both are ML quality control |
| Data Integrity & Leakage Sentinel | Merge | Same category |

**Merged into:** ML Quality Sentinel (training reliability + leakage detection + data integrity)

**Result:** 3 → 2 agents

### Engineering Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Quant Developer | Keep | Core implementation |
| Pipeline & Experiment Orchestrator | Merge into Quant Dev | Infrastructure is Quant Dev territory |
| Dashboard & UX Architect | Remove | Project work, not ongoing agent |

**Result:** 3 → 1 agent

### Risk & Evaluation Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Risk Manager | Keep | Core independent oversight |
| Execution & Cost Realism Analyst | Merge | Both validate real-world viability |
| Model Validation & Stress Tester | Merge | Same validation focus |

**Merged into:** Validation Analyst (stress testing + execution realism + model validation)

**Result:** 3 → 2 agents

### Deployment & Ops Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Paper Trading Operator | Defer | Tier 4 functionality |
| Live Trading Readiness Agent | Defer | Tier 4 functionality |
| Monitoring & Kill-Switch Agent | Defer | Tier 4 functionality |

**Result:** 3 → 0 agents (deferred to Tier 4)

---

## Proposed Options

### Option A: 8 Agents (Lean) - RECOMMENDED

```
You (CIO/Research Director)
│
├── Alpha Researcher (with regime awareness)
├── Signal Scientist
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
│
├── Risk Manager
├── Validation Analyst
│
└── Report Generator
```

**Why 8 agents:**
- Each agent has distinct, full-time job
- No idle agents waiting for work
- Covers full research lifecycle: Discovery → Modeling → Implementation → Risk → Reporting
- Matches current HRP platform capabilities
- Easy to expand later

### Option B: 10 Agents (Moderate)

```
You (CIO/Research Director)
│
├── Alpha Researcher
├── Signal Scientist
├── Regime Analyst (kept separate)
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
│
├── Risk Manager
├── Validation Analyst
│
├── Report Generator
└── Deployment Monitor (placeholder)
```

### Option C: 12 Agents (Full but streamlined)

```
You (CIO/Research Director)
│
├── Alpha Researcher
├── Signal Scientist
├── Regime Analyst
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
├── Pipeline Orchestrator (kept separate)
│
├── Risk Manager
├── Validation Analyst
│
├── Report Generator
└── Deployment Monitor
```

---

## Agent Role Definitions (Option A)

### 1. Alpha Researcher
**Focus:** Hypothesis development, strategy design, regime awareness
**Inputs:** Market data, features, existing hypotheses
**Outputs:** New hypotheses with thesis, prediction, falsification criteria
**Schedule:** Weekly discovery runs + on-demand
**Uses:** `create_hypothesis`, `get_features`, `get_prices`, `run_backtest`

> **Detailed spec:** See [Alpha Researcher Specification](#alpha-researcher-specification) below

### 2. Signal Scientist
**Focus:** Feature engineering, signal discovery, predictive pattern identification
**Inputs:** Price data, existing features, alternative data
**Outputs:** New feature definitions, signal strength reports, IC analysis
**Schedule:** Weekly scans + on-demand
**Uses:** `get_available_features`, `get_features`, feature computation APIs

### 3. ML Scientist
**Focus:** Model development, training, hyperparameter optimization
**Inputs:** Features, targets, hypothesis specifications
**Outputs:** Trained models, walk-forward validation results
**Schedule:** On-demand (triggered by new hypotheses)
**Uses:** `train_ml_model`, `run_walk_forward_validation`, `get_supported_models`

### 4. ML Quality Sentinel
**Focus:** Training reliability, data leakage detection, overfitting prevention
**Inputs:** Training data, model outputs, feature sets
**Outputs:** Audit reports, leakage warnings, reliability scores
**Schedule:** Runs after every ML training job
**Uses:** Overfitting guards, leakage validators, Sharpe decay monitoring

### 5. Quant Developer
**Focus:** Backtest implementation, code quality, performance optimization, pipeline orchestration
**Inputs:** Strategy specifications, model outputs
**Outputs:** Optimized backtests, performance reports, infrastructure improvements
**Schedule:** On-demand + monitors experiment queue
**Uses:** `run_backtest`, MLflow APIs, feature computation

### 6. Risk Manager
**Focus:** Position limits, drawdown monitoring, portfolio risk, independent oversight
**Inputs:** Backtest results, portfolio state, market conditions
**Outputs:** Risk reports, limit violations, strategy vetoes
**Schedule:** Continuous monitoring + reviews before deployment approval
**Uses:** Risk validation APIs, strategy metrics

### 7. Validation Analyst
**Focus:** Stress testing, execution realism, model validation, out-of-sample testing
**Inputs:** Strategies, backtest results, market scenarios
**Outputs:** Stress test reports, execution cost estimates, validation verdicts
**Schedule:** Runs on all strategies before promotion to "validated"
**Uses:** Parameter sensitivity, robustness testing, statistical validation

### 8. Report Generator
**Focus:** Synthesize findings, create human-readable summaries
**Inputs:** All agent outputs, hypothesis status, experiment results
**Outputs:** Weekly research reports, hypothesis summaries, action recommendations
**Schedule:** Weekly + on-demand
**Uses:** `get_lineage`, `list_hypotheses`, `get_experiment`, `analyze_results`

---

## Shared Workspace: How Agents Coordinate

> **Architecture Diagram:** See [Data Pipeline Architecture](../architecture/data-pipeline-diagram.md) for visual diagrams of agent coordination flows.

### Registry Points (existing HRP infrastructure)
1. **Hypothesis Registry** - Agents create/update hypotheses with status
2. **MLflow Experiments** - All training/backtest results logged
3. **Lineage System** - Full audit trail with actor tracking
4. **Feature Store** - Shared feature definitions and values

### Workflow Example
```
1. Signal Scientist discovers promising momentum signal
   → Creates draft hypothesis HYP-2026-042

2. Alpha Researcher picks up HYP-2026-042
   → Refines thesis, adds falsification criteria
   → Updates status to "testing"

3. ML Scientist sees "testing" hypothesis
   → Runs walk-forward validation
   → Logs results to MLflow

4. ML Quality Sentinel audits the training
   → Checks for leakage, overfitting
   → Adds audit report to lineage

5. Validation Analyst runs stress tests
   → Parameter sensitivity, regime analysis
   → Updates hypothesis with validation results

6. Risk Manager reviews validated hypothesis
   → Checks risk limits, portfolio fit
   → Approves or flags concerns

7. Report Generator summarizes for CIO review
   → Weekly report includes HYP-2026-042 findings
   → CIO decides whether to deploy
```

---

## Implementation Considerations

### Agent Infrastructure

**Two base classes:**

1. **`ResearchAgent`** (existing) - For Custom agents
   - Extends `IngestionJob` pattern
   - Deterministic workflows
   - Used by: Signal Scientist, ML Scientist, ML Quality Sentinel, Risk Manager

2. **`SDKAgent`** (to build) - For SDK agents
   - Wraps Claude Agent SDK
   - Reasoning capabilities via Claude API
   - Common setup: API auth, tool registration, error handling, cost tracking
   - Used by: Alpha Researcher, Quant Developer, Report Generator

3. **Hybrid agents** (Validation Analyst)
   - Extend `ResearchAgent` but can invoke Claude API for specific reasoning steps

### MCP Integration
- Agents use existing 22 MCP tools
- No direct agent-to-agent communication (shared registries only)
- Actor tracking already supports agent identification

### Permissions
- Agents cannot deploy strategies (existing rule)
- Risk Manager can veto but not approve deployment
- Only CIO (user) has final approval authority

---

## Next Steps

1. [x] Decide on Option A, B, or C → **Option A (8 agents)**
2. [x] Decide SDK vs Custom per agent → **See matrix above**
3. [x] Decide communication model → **Shared registries only**
4. [x] Decide scheduling → **APScheduler**
5. [x] Write Alpha Researcher specification → **See spec below**
6. [ ] Design `SDKAgent` base class
7. [x] Build lineage event watcher → **See [Data Pipeline Diagram](../architecture/data-pipeline-diagram.md#event-driven-agent-coordination)**
8. [x] Implement Alpha Researcher (SDK) → Uses Claude API for economic rationale analysis
9. [ ] Implement ML Quality Sentinel (Custom)
10. [ ] Implement Validation Analyst (Hybrid)
11. [x] Test coordination through shared workspace → Event-driven triggers working
12. [ ] Expand to remaining agents

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Agent implementation | Case-by-case: SDK for reasoning, Custom for deterministic |
| Scheduling | APScheduler (extend existing) |
| Communication | Shared registries only |
| Priority | Alpha Researcher → ML Quality Sentinel → Validation Analyst |

## SDK Agent Operations

### Error Handling: Checkpoint & Resume
- Save agent state to lineage after each tool call
- On failure, can resume from last checkpoint
- Partial work preserved, not wasted

### Cost Controls: Layered Budget System

| Level | Limit | Action |
|-------|-------|--------|
| Per-run | Max tokens per agent execution (e.g., 50K in + 10K out) | Agent stops, logs partial work |
| Daily | Aggregate across all SDK agents | Scheduler pauses SDK agents |
| Per-hypothesis | Track spend per hypothesis | Expensive hypotheses flagged for review |

### Testing Strategy: Mixed Approach

| Test Type | Method | Cost |
|-----------|--------|------|
| Unit tests | Mock Claude responses (recorded/replayed) | Free |
| Integration tests | Claude Haiku (cheap model) | ~10x cheaper |
| E2E / Staging | Full model (Sonnet/Opus) | Full cost |

---

## Alpha Researcher Specification

### Overview

| Aspect | Decision |
|--------|----------|
| Type | SDK Agent (Claude reasoning) |
| Trigger | Lineage event (after Signal Scientist) + MCP on-demand |
| Scope | Refine thesis + regime context + related hypotheses |
| Tools | Read-only analysis (no backtests) |
| Outputs | Hypothesis update + lineage event + research note |
| Checkpoints | Per hypothesis |

### Trigger Model

**Primary: Lineage Event Trigger**
- Signal Scientist logs `AGENT_RUN_COMPLETE` event to lineage
- Scheduler's event watcher detects new event
- Triggers Alpha Researcher automatically
- Processes all hypotheses in "draft" status

**Secondary: MCP On-Demand**
- New tool: `run_alpha_researcher(hypothesis_id: str | None)`
- If `hypothesis_id` provided: process single hypothesis
- If `None`: process all draft hypotheses

**Infrastructure required:** Lineage event watcher in scheduler (new component)

### Scope: What Alpha Researcher Does

For each draft hypothesis:

1. **Analyze economic rationale**
   - Why might this signal work?
   - What market inefficiency does it exploit?
   - Is there academic/practitioner support?

2. **Check regime context**
   - Call `detect_regime` for historical regime labels
   - Analyze: does signal work in bull/bear/sideways markets?
   - Add regime-conditional notes to hypothesis

3. **Search related hypotheses**
   - Query hypothesis registry for similar features/signals
   - Check for conflicting hypotheses
   - Note if novel or variant of existing idea

4. **Update hypothesis**
   - Refine thesis with economic reasoning
   - Strengthen falsification criteria
   - Add metadata: `regime_notes`, `related_hypotheses`
   - Update status: "draft" → "testing"

5. **Log to lineage**
   - Event type: `ALPHA_RESEARCHER_REVIEW`
   - Include reasoning summary

### Tools Available

| Tool | Purpose | Exists? |
|------|---------|---------|
| `get_hypothesis` | Read draft hypothesis details | ✅ |
| `update_hypothesis` | Update thesis, status, metadata | ✅ |
| `list_hypotheses` | Find related/conflicting hypotheses | ✅ |
| `get_prices` | Analyze price history for context | ✅ |
| `get_features` | Check signal behavior | ✅ |
| `detect_regime` | Get historical regime labels (HMM) | ✅ |

**Not available:** `run_backtest` - ML Scientist handles backtesting

### Outputs

**1. Hypothesis Updates (per hypothesis)**
- Updated `thesis` with economic rationale
- Updated `falsification` criteria
- New `metadata` fields:
  - `regime_notes`: regime-conditional analysis
  - `related_hypotheses`: list of related hypothesis IDs
  - `alpha_researcher_review_date`: timestamp
- Status change: "draft" → "testing"

**2. Lineage Events (per hypothesis)**
- Event type: `ALPHA_RESEARCHER_REVIEW`
- Actor: `agent:alpha-researcher`
- Details: reasoning summary, regime findings, related hypotheses

**3. Research Note (per run)**
- Location: `docs/research/YYYY-MM-DD-alpha-researcher.md`
- Contents:
  - Summary of hypotheses processed
  - Key findings per hypothesis
  - Regime context observations
  - Recommendations for ML Scientist

### Checkpoint & Resume

**Granularity:** Per hypothesis

```
Workflow:
1. Fetch all draft hypotheses
2. For each hypothesis:          ← CHECKPOINT after each
   a. Analyze economic rationale
   b. Check regime context
   c. Search related hypotheses
   d. Update hypothesis record
   e. Log to lineage
3. Write research note
```

**On failure:**
- Agent state saved after each hypothesis
- Resume picks up from last incomplete hypothesis
- Already-processed hypotheses not re-processed

### Example Research Note

```markdown
# Alpha Researcher Report - 2026-01-26

## Summary
- Hypotheses reviewed: 3
- Promoted to testing: 2
- Needs more data: 1

## HYP-2026-042: momentum_20d predicts monthly returns

**Economic Rationale:**
Momentum effect is well-documented (Jegadeesh & Titman 1993).
Signal likely captures trend-following behavior and slow
information diffusion.

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
## HYP-2026-043: ...
```

### New Infrastructure Required

1. **Lineage Event Watcher**
   - Component in scheduler
   - Polls lineage for new `AGENT_RUN_COMPLETE` events
   - Triggers dependent agents

2. **MCP Tool: `run_alpha_researcher`**
   - Parameters: `hypothesis_id: str | None`
   - Returns: processing summary

3. **`SDKAgent` Base Class**
   - Claude API setup
   - Tool registration
   - Cost tracking
   - Checkpoint management

---

## Document History

- **2026-01-25:** Initial brainstorm captured from conversation
- **2026-01-26:** Design decisions finalized (8 agents, hybrid SDK/Custom, build order)
- **2026-01-26:** Added Alpha Researcher detailed specification
- **2026-01-26:** Lineage event watcher implemented; Alpha Researcher implemented; event-driven coordination tested
- **2026-01-26:** Added reference to [Data Pipeline Architecture](../architecture/data-pipeline-diagram.md)
