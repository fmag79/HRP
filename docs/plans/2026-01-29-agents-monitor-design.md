# Agents Monitor Design

**Date:** January 29, 2026
**Status:** Design Complete
**Purpose:** Real-time monitoring and historical audit for all HRP research agents

---

## Executive Summary

A Streamlit-based dashboard providing unified visibility into all 11 HRP research agents. Features a two-section single-page layout with real-time status monitoring (top) and historical timeline view (bottom). Uses hybrid auto-refresh that updates when agents are active and pauses when idle.

---

## Section 1: Overview

### Purpose

Provide visibility into agent activity for:
- **Operations**: Monitor agent health, identify stuck/failed runs
- **Research**: Track hypothesis progression through pipeline
- **Debugging**: Review past agent runs, identify error patterns
- **Audit**: Full lineage trail of all agent actions

### Key Features

**Real-Time Monitor (Top Section):**
- Live status of all 11 agents
- Comprehensive metrics: status, current task, elapsed time, progress
- Color-coded indicators (🟦 Running, 🟢 Completed, 🔴 Failed, ⚪ Idle)
- Hybrid auto-refresh: 2s when active, 10s when idle

**Historical Timeline (Bottom Section):**
- Chronological list of all past agent runs
- Minimal by default: agent, status, timestamp, hypothesis ID
- Click-to-expand: duration, items processed, metrics, errors, log links
- Filters: by agent, date range, status

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Agents Monitor                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ REAL-TIME MONITOR (auto-refresh when active)         │  │
│  │  [x] Auto-refresh  [Refresh Now]                      │  │
│  │                                                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ Signal   │  │ Alpha    │  │ Code     │  ...    │  │
│  │  │ Scientist│  │Researcher│  │Material. │          │  │
│  │  │ 🟦 Run   │  │ ⚪ Idle  │  │ 🟢 Comp   │          │  │
│  │  │ 12s      │  │ -        │  │ 45s      │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ HISTORICAL TIMELINE (click to expand)                │  │
│  │  Filter: [Signal ✓] [Alpha ✓] ... | Date: [___]      │  │
│  │                                                        │  │
│  │  ✅ Signal Scientist — COMPLETED • 14:30 • HYP-001    │  │
│  │     └─ Duration: 45s, 5 hypotheses created           │  │
│  │  🔄 Alpha Researcher — RUNNING • 14:29 • HYP-002      │  │
│  │     └─ 2/3 hypotheses reviewed, [Expand]             │  │
│  │  ❌ ML Quality Sentinel — FAILED • 14:25 • HYP-003     │  │
│  │     └─ ERROR: Sharpe decay > 50%, [View Logs]        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Section 2: Data Sources

### Primary Data: Lineage Table

The monitor queries `hrp.duckdb.lineage` for all agent events.

**Key Event Types:**
- `AGENT_RUN_START` - Agent begins execution
- `AGENT_RUN_COMPLETE` - Agent finishes successfully
- `{AGENT}_COMPLETE` - Agent-specific completion events
- `{AGENT}_FAILED` - Agent-specific failure events
- `hypothesis_created`, `hypothesis_updated` - Hypothesis tracking

### Status Inference Algorithm

Agent status is inferred from lineage events, not from process monitoring:

```python
def _infer_agent_status(events: list) -> str:
    """
    Infer agent status from recent lineage events.

    Logic:
    1. No events → IDLE
    2. Has START but no COMPLETE (within 5 min) → RUNNING
    3. Latest event has error/FAILED → FAILED
    4. Latest event is COMPLETE → COMPLETED
    5. No recent events (no activity in 1 hr) → IDLE
    """
```

---

## Section 3: Page Structure

### File: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Layout:**
```python
# Page header with controls
st.title("🤖 Agents Monitor")
[Auto-refresh checkbox] [Refresh Now button]

# Real-time section
st.subheader("Real-Time Monitor")
[4-column grid of agent cards]

# Divider
st.markdown("---")

# Historical section
st.subheader("Historical Timeline")
[Filters: Agent, Status, Date Range, Limit]
[Expandable timeline events]

# Auto-refresh logic at bottom
[Hybrid refresh: 2s active / 10s idle]
```

### Agent Card Display

Each agent shows:
- Icon + Name
- Status badge (color-coded)
- Progress bar (if running)
- Elapsed time (if running)
- Current hypothesis (if running)
- Last event timestamp
- Quick stats (processed, success rate)

### Timeline Event Display

Each event shows (collapsed):
- Status icon + Agent name + Status + Timestamp + Hypothesis ID

Expanded view shows:
- Duration, items processed
- Metrics (Sharpe, IC, etc.)
- Error message (if failed)
- Action links (MLflow, Hypothesis details)

---

## Section 4: Backend Functions

### File: `hrp/dashboard/agents_monitor.py`

**Core Functions:**

```python
@dataclass
class AgentStatus:
    """Status of a single agent."""
    agent_id: str
    name: str
    status: str  # running, completed, failed, idle
    last_event: dict | None
    elapsed_seconds: int | None
    current_hypothesis: str | None
    progress_percent: float | None
    stats: dict | None


def get_all_agent_status(api: PlatformAPI) -> list[AgentStatus]:
    """Get current status of all agents from lineage events."""

def get_timeline(
    api: PlatformAPI,
    agents: list[str],
    statuses: list[str],
    date_range: tuple,
    limit: int,
) -> list[dict]:
    """Get historical timeline of agent events."""
```

### Agent List

All 11 agents monitored:
1. Signal Scientist
2. Alpha Researcher
3. Code Materializer
4. ML Scientist
5. ML Quality Sentinel
6. Quant Developer
7. Pipeline Orchestrator
8. Validation Analyst
9. Risk Manager
10. CIO Agent
11. Report Generator

---

## Section 5: Hybrid Auto-Refresh

### Refresh Logic

```python
# Initial state
st.session_state.refresh_interval = 5  # seconds
st.session_state.last_activity = None

# Each rerun:
active_agents = [a for a in agents if a["status"] == "running"]

if active_agents:
    st.session_state.last_activity = now()
    st.session_state.refresh_interval = 2  # Fast refresh
elif st.session_state.last_activity:
    idle_time = (now() - st.session_state.last_activity).total_seconds()
    if idle_time > 30:
        st.session_state.refresh_interval = 10  # Slow refresh

# Sleep and rerun
time.sleep(st.session_state.refresh_interval)
st.rerun()
```

### Benefits

- Fast updates when agents are active (2s)
- Saves resources when idle (10s)
- No manual refresh needed (but available)
- Respects browser tab (pauses when tab inactive via Streamlit)

---

## Section 6: File Structure

### Files to Create

| File | Purpose |
|------|---------|
| `hrp/dashboard/pages/2_Agents_Monitor.py` | Main Streamlit page |
| `hrp/dashboard/agents_monitor.py` | Backend functions |
| `tests/dashboard/test_agents_monitor.py` | Tests |

### Files to Modify

| File | Changes |
|------|---------|
| `hrp/research/lineage.py` | Add `AGENT_RUN_START` event type |
| `hrp/dashboard/__init__.py` | Export monitor functions |

---

## Section 7: Error Handling

### Edge Cases

| Situation | Handling |
|-----------|----------|
| No lineage events | Show as "Idle" |
| Stale agent (>1 hour) | Show as "Idle" + last activity |
| Concurrent runs | Show most recent |
| Missing hypothesis ID | Display "N/A" |
| Failed agent | Red card + error message |
| Empty timeline | "No events found" |

### Error Boundaries

```python
try:
    agents = get_all_agent_status(api)
except Exception as e:
    st.error(f"Failed to load agent status: {e}")
    st.button("Retry")
    agents = []
```

---

## Section 8: Performance

### Optimizations

| Concern | Solution |
|---------|----------|
| Large lineage table | Index on `(actor, timestamp)` |
| Frequent polling | `@st.cache_data(ttl=5)` |
| Timeline size | Limit to 500 events max |
| Concurrent users | Streamlit session isolation |

### Caching Strategy

```python
@st.cache_data(ttl=5)
def get_all_agent_status_cached() -> list[AgentStatus]:
    """Cached for 5 seconds."""

@st.cache_data(ttl=10)
def get_timeline_cached(...) -> list[dict]:
    """Cached for 10 seconds (timeline changes less)."""
```

---

## Section 9: UI Components

### Agent Card Component

```python
def render_agent_card(agent: dict):
    """Render a single agent status card."""
    status_colors = {
        "running": "🟦",
        "completed": "🟢",
        "failed": "🔴",
        "idle": "⚪",
    }

    with st.container():
        st.markdown(f"### {status_colors[agent['status']]} {agent['name']}")
        st.markdown(f"**Status:** `{agent['status'].upper()}`")

        if agent["status"] == "running":
            if agent.get("progress_percent"):
                st.progress(agent["progress_percent"] / 100)
            st.caption(f"⏱ Elapsed: {agent.get('elapsed_seconds', 0)}s")
            if agent.get("current_hypothesis"):
                st.caption(f"📋 Processing: `{agent['current_hypothesis']}`")

        st.caption(f"🕐 Last: {format_timestamp(agent['last_event']['timestamp'])}")
        st.markdown("---")
```

### Timeline Event Component

```python
def render_timeline_event(event: dict):
    """Render expandable timeline event."""
    status_icons = {
        "completed": "✅",
        "failed": "❌",
        "running": "🔄",
    }

    with st.expander(
        f"{status_icons[event['status']]} **{event['agent_name']}** — "
        f"{event['status'].upper()} • {format_timestamp(event['timestamp'])} "
        f"• `{event.get('hypothesis_id', 'N/A')}`"
    ):
        # Expanded content
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Details**")
            if event.get("duration_seconds"):
                st.caption(f"⏱ {event['duration_seconds']}s")
            if event.get("items_processed"):
                st.caption(f"📊 {event['items_processed']} items")

        with col2:
            st.markdown("**Results**")
            for key, value in event.get("metrics", {}).items():
                st.caption(f"{key}: {value}")
            if event.get("error"):
                st.error(event["error"])

        with col3:
            st.markdown("**Actions**")
            if event.get("mlflow_run_id"):
                st.link_button("View in MLflow", f"http://localhost:5000/{event['mlflow_run_id']}")
```

---

## Section 10: Implementation Priority

### Phase 1: Core (MVP)

| Task | Description |
|------|-------------|
| Add `AGENT_RUN_START` event type | Update `EventType` enum |
| Create `agents_monitor.py` | Backend functions |
| Create `2_Agents_Monitor.py` | Streamlit page |
| Basic agent status display | 4-column grid, no progress |

### Phase 2: Enhanced

| Task | Description |
|------|-------------|
| Add progress tracking | Extract from event details |
| Add timeline view | Historical events with filters |
| Add auto-refresh | Hybrid refresh logic |
| Add error handling | Boundaries for failed queries |

### Phase 3: Polish

| Task | Description |
|------|-------------|
| Add caching | `@st.cache_data` decorators |
| Add performance monitoring | Query timing |
| Add tests | Unit tests for functions |
| Add documentation | README, comments |

---

## Section 11: Success Criteria

- [ ] All 11 agents display with correct status
- [ ] Running agents update every 2 seconds
- [ ] Idle agents update every 10 seconds
- [ ] Timeline shows at least 50 events with filters working
- [ ] Click-to-expand shows full event details
- [ ] Failed agents show error messages
- [ ] Page loads in < 2 seconds
- [ ] Auto-refresh can be toggled on/off

---

## Document History

- **2026-01-29:** Initial design created
