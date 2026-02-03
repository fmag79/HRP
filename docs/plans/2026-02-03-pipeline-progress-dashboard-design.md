# Pipeline Progress Dashboard Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a visual Kanban board to the Streamlit dashboard showing hypotheses flowing through the 10-stage agent pipeline, with ability to launch agents directly from the UI.

**Architecture:** Streamlit page with tabbed Kanban columns, per-card quick actions, and a collapsible agent control panel. Dashboard calls agents via direct Python imports; MCP server exposes same agents for Claude access.

**Tech Stack:** Streamlit, existing HRP agents, FastMCP

---

## 1. Data Model

### Pipeline Stage Mapping

The Kanban board determines pipeline stage from lineage events:

| Lineage Event | Kanban Column | Tab |
|---------------|---------------|-----|
| `hypothesis_created` | Draft | Discovery |
| `alpha_researcher_complete` | Testing | Discovery |
| `ml_scientist_validation` | ML Audit | Validation |
| `ml_quality_sentinel_audit` | Quant Dev | Validation |
| `quant_developer_complete` | Kill Gate | Validation |
| `kill_gate_enforcer_complete` | Stress Test | Review |
| `validation_analyst_complete` | Risk Review | Review |
| `risk_manager_assessment` | CIO Review | Review |
| `cio_agent_decision` | Human Approval | Review |
| status = `deployed` | Deployed | Review |

### Data Query

```python
def get_hypothesis_pipeline_stages() -> list[dict]:
    """
    Get all active hypotheses with their current pipeline stage.

    Returns list of:
    {
        "hypothesis_id": "HYP-2026-012",
        "title": "Momentum Factor Strategy",
        "status": "testing",
        "pipeline_stage": "ml_audit",
        "stage_entered_at": "2026-02-03T10:30:00",
        "time_in_stage_seconds": 3600,
        "last_event": {...},
        "metrics": {"ic": 0.032, "sharpe": 1.24}
    }
    """
```

---

## 2. UI Components

### 2.1 Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Pipeline Progress                          [🤖 Agent Panel] [⟳ Refresh]│
├─────────────────────────────────────────────────────────────────────────┤
│  [Discovery] [Validation] [Review]  ← Tabs                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Kanban columns based on selected tab                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Tab Groupings

- **Discovery Tab:** Draft, Testing (2 columns)
- **Validation Tab:** ML Audit, Quant Dev, Kill Gate (3 columns)
- **Review Tab:** Stress Test, Risk Review, CIO Review, Human Approval (4 columns)

### 2.3 Hypothesis Card

```
┌──────────────────────────────────────┐
│ HYP-2026-012                         │
│ Momentum Factor Strategy             │
│ ⏳ 2h │ IC: 0.032 │ Sharpe: 1.24    │
│ ──────────────────────────────────── │
│ [▶ Run ML Scientist] [📋 Details]   │  ← On hover/expand
└──────────────────────────────────────┘
```

Card styling by status:
- 🔄 Running: Blue border, pulse animation
- ✅ Pass: Green border
- ⚠️ Warning: Yellow border
- ❌ Failed: Red border
- ⏳ Waiting: Gray border

### 2.4 Agent Panel (Collapsible)

Grid of agent cards:
```
┌─────────────────────┐ ┌─────────────────────┐
│ 🔬 Signal Scientist │ │ 📊 Alpha Researcher │
│ [▶ Run Scan]        │ │ [▶ Run (3 drafts)]  │
│ Last: 2d ago        │ │ Last: 4h ago        │
└─────────────────────┘ └─────────────────────┘
```

Plus "Run Full Pipeline" button for batch execution.

---

## 3. Agent Integration

### 3.1 Dashboard Agent Calls (Direct Python)

```python
# hrp/dashboard/components/agent_panel.py

from hrp.agents.research_agents import (
    SignalScientist, MLScientist, MLQualitySentinel,
    KillGateEnforcer, ValidationAnalyst, RiskManager
)
from hrp.agents.alpha_researcher import AlphaResearcher
from hrp.agents.cio import CIOAgent
from hrp.agents.quant_developer import QuantDeveloper
from hrp.agents.report_generator import ReportGenerator

AGENT_REGISTRY = {
    "signal_scientist": {
        "class": SignalScientist,
        "name": "Signal Scientist",
        "icon": "🔬",
        "description": "Automated signal discovery scan",
        "requires_hypothesis": False,
    },
    "alpha_researcher": {
        "class": AlphaResearcher,
        "name": "Alpha Researcher",
        "icon": "📊",
        "description": "Review draft hypotheses",
        "requires_hypothesis": False,  # Finds drafts automatically
        "eligible_statuses": ["draft"],
    },
    # ... etc for all 10 agents
}

def run_agent_sync(agent_key: str, hypothesis_ids: list[str] | None = None) -> dict:
    """Run an agent synchronously and return results."""
    agent_info = AGENT_REGISTRY[agent_key]
    agent_class = agent_info["class"]

    if hypothesis_ids:
        agent = agent_class(hypothesis_ids=hypothesis_ids, send_alerts=False)
    else:
        agent = agent_class(send_alerts=False)

    return agent.run()
```

### 3.2 MCP Tools (for Claude)

Add to `hrp/mcp/research_server.py`:

```python
@mcp.tool()
def run_signal_scientist() -> dict:
    """Run Signal Scientist to scan for new signals and create draft hypotheses."""

@mcp.tool()
def run_ml_scientist(hypothesis_ids: list[str] | None = None) -> dict:
    """Run ML Scientist to perform walk-forward validation on testing hypotheses."""

@mcp.tool()
def run_kill_gate_enforcer(hypothesis_ids: list[str] | None = None) -> dict:
    """Run Kill Gate Enforcer to check validated hypotheses against kill gates."""

@mcp.tool()
def run_validation_analyst(hypothesis_ids: list[str] | None = None) -> dict:
    """Run Validation Analyst to perform stress testing."""

@mcp.tool()
def run_risk_manager(hypothesis_ids: list[str] | None = None) -> dict:
    """Run Risk Manager to assess portfolio risk and issue vetoes."""

@mcp.tool()
def run_cio_agent(hypothesis_ids: list[str] | None = None) -> dict:
    """Run CIO Agent to score hypotheses and make CONTINUE/KILL decisions."""

@mcp.tool()
def run_quant_developer(hypothesis_ids: list[str] | None = None) -> dict:
    """Run Quant Developer to create production backtests with realistic costs."""
```

---

## 4. File Structure

```
hrp/
├── dashboard/
│   ├── pages/
│   │   └── pipeline_progress.py    # NEW - Main page
│   ├── components/
│   │   ├── pipeline_kanban.py      # NEW - Kanban board
│   │   └── agent_panel.py          # NEW - Agent launcher
│   ├── pipeline_data.py            # NEW - Data queries
│   └── app.py                      # MODIFY - Navigation
│
├── mcp/
│   └── research_server.py          # MODIFY - Add agent tools
│
└── agents/
    └── __init__.py                 # MODIFY - Clean exports
```

---

## 5. Implementation Tasks

### Task 1: Data Layer (`pipeline_data.py`)
- `get_hypothesis_pipeline_stages()` - Query hypotheses with stage info
- `get_stage_for_hypothesis()` - Determine stage from lineage events
- `get_eligible_hypotheses_for_agent()` - Filter by status/stage

### Task 2: Kanban Component (`pipeline_kanban.py`)
- `render_kanban_column()` - Single column with cards
- `render_hypothesis_card()` - Card with status, metrics, actions
- `render_card_actions()` - Quick action buttons

### Task 3: Agent Panel (`agent_panel.py`)
- `AGENT_REGISTRY` - All agents with metadata
- `render_agent_panel()` - Collapsible grid of agent cards
- `render_agent_card()` - Single agent with run button
- `run_agent_sync()` - Execute agent and return results

### Task 4: Main Page (`pipeline_progress.py`)
- Page layout with header and controls
- Tab navigation (Discovery, Validation, Review)
- Integration of Kanban + Agent Panel
- Refresh and auto-refresh logic

### Task 5: Navigation (`app.py`)
- Add "Pipeline Progress" to sidebar
- Add route to render function

### Task 6: MCP Agent Tools (`research_server.py`)
- Add 7 new agent tools
- Consistent error handling and response format

### Task 7: Tests
- Test `get_hypothesis_pipeline_stages()`
- Test `get_eligible_hypotheses_for_agent()`
- Test agent panel rendering

---

## 6. UI Execution Model

**Synchronous with Spinner:**

```python
with st.spinner(f"Running {agent_name}..."):
    result = run_agent_sync(agent_key, hypothesis_ids)

if result.get("status") == "success":
    st.success(f"✅ {agent_name} completed: {result.get('message')}")
else:
    st.error(f"❌ {agent_name} failed: {result.get('error')}")

st.rerun()  # Refresh to show updated pipeline state
```

---

## Document Version

**Version:** 1.0
**Created:** 2026-02-03
**Author:** Claude + User
**Status:** Approved for implementation
