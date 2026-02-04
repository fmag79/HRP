# Fix HRP Agent Pipeline: Decision Pipeline Audit

**Date:** 2026-02-02
**Branch:** `fix/decision-pipeline-audit`
**Status:** ⚠️ OBSOLETE - Code Materializer agent was removed; pipeline fixes completed separately

> **Note (2026-02-03):** This plan references the Code Materializer agent which was removed from the pipeline. The pipeline issues identified here were resolved through separate commits. This document is kept for historical reference only.

## Problem Summary

The decision pipeline documented in `docs/agents/decision-pipeline.md` describes an 11-stage event-driven agent pipeline. Audit of the actual code reveals **4 critical issues** that break the chain, plus an **EventType enum / DB schema mismatch** that causes runtime errors.

### Pipeline Chain Status

```
Signal Scientist ──OK──► Alpha Researcher ──SKIPS──► [Code Materializer] ──► ML Scientist
                                                        ▲ BROKEN                  │
                                                        (no trigger,              │ BROKEN
                                                         bad EventType)           │ (never emits
                                                                                  │  experiment_completed)
                                                                                  ▼
ML Quality Sentinel ◄──DEAD── (event never fires)    Quant Developer ──OK──►
Pipeline Orchestrator ──OK──► Validation Analyst ──GAP──► Risk Manager ──GAP──► CIO Agent
                                                    ▲ NO TRIGGER           ▲ NO TRIGGER
```

## Issues

### Issue 1: EventType enum and DB CHECK constraint are out of sync

**Files:** `hrp/research/lineage.py`, `hrp/data/schema.py`

The Python `EventType` enum has 22 values. The DB `CHECK` constraint only allows 20 different values, and the two sets overlap imperfectly. Agent-emitted events like `ml_quality_sentinel_audit`, `alpha_researcher_complete`, `quant_developer_backtest_complete`, `pipeline_orchestrator_complete`, `risk_review_complete`, `risk_veto`, `kill_gate_triggered` pass Python validation but **fail the DB INSERT** due to the CHECK constraint.

**Fix:** Update the DB CHECK constraint in `schema.py` to include ALL values from the EventType enum, plus add the missing event types (`code_materializer_complete`, `experiment_completed`, `cio_agent_decision`, `validation_analyst_complete`, `risk_manager_assessment`). Also add any new EventType enum values needed.

### Issue 2: Code Materializer uses non-existent EventType

**File:** `hrp/agents/code_materializer.py:67`

References `EventType.CODE_MATERIALIZER_COMPLETE` which doesn't exist in the enum. Will crash at runtime with `AttributeError`.

**Fix:** Add `CODE_MATERIALIZER_COMPLETE = "code_materializer_complete"` to the `EventType` enum.

### Issue 3: ML Scientist doesn't emit `experiment_completed`

**File:** `hrp/agents/ml_scientist.py` (MLScientist class, around line 231)

ML Scientist emits only `EventType.AGENT_RUN_COMPLETE` (generic). The downstream trigger for ML Quality Sentinel listens for `experiment_completed`, which is never emitted. This breaks the chain.

**Fix:** Add `EXPERIMENT_COMPLETED = "experiment_completed"` to EventType enum (it's in the DB CHECK already), and have ML Scientist emit `EventType.EXPERIMENT_COMPLETED` after each hypothesis is processed in `execute()`.

### Issue 4: Missing trigger wiring for last 3 pipeline stages

**Files:** `hrp/agents/run_job.py` (lines 150-256), `hrp/agents/scheduler.py` (lines 1229-1365)

Only 6 of the needed 9 event triggers are registered. Missing:

- `validation_analyst_complete` → Risk Manager
- `risk_manager_assessment` → CIO Agent
- `cio_agent_decision` → Human notification

**Fix:**

1. Add `VALIDATION_ANALYST_COMPLETE = "validation_analyst_complete"` to EventType (and have ValidationAnalyst emit it)
2. Add `RISK_MANAGER_ASSESSMENT = "risk_manager_assessment"` to EventType (and have RiskManager emit it)
3. Add `CIO_AGENT_DECISION = "cio_agent_decision"` to EventType (and have CIO Agent emit it)
4. Register 3 new triggers in both `run_job.py:run_agent_pipeline()` and `scheduler.py:setup_research_agent_triggers()`

### Issue 5: Code Materializer skipped in trigger chain

**Files:** `hrp/agents/run_job.py`, `hrp/agents/scheduler.py`

The trigger chain goes Alpha Researcher → ML Scientist directly, skipping Code Materializer. The decision pipeline doc says it should be: Alpha Researcher → Code Materializer → ML Scientist.

**Fix:** Insert Code Materializer between Alpha Researcher and ML Scientist in the trigger chain:

- `alpha_researcher_complete` → Code Materializer
- `code_materializer_complete` → ML Scientist

## Implementation Steps

### Step 1: Sync EventType enum with pipeline requirements

**File:** `hrp/research/lineage.py`

Add missing EventType values:

```python
CODE_MATERIALIZER_COMPLETE = "code_materializer_complete"
EXPERIMENT_COMPLETED = "experiment_completed"
CIO_AGENT_DECISION = "cio_agent_decision"
VALIDATION_ANALYST_COMPLETE = "validation_analyst_complete"
RISK_MANAGER_ASSESSMENT = "risk_manager_assessment"
```

### Step 2: Update DB CHECK constraint

**File:** `hrp/data/schema.py` (line 438-444)

Replace the CHECK constraint to include ALL EventType enum values. The mismatch is larger than just the 5 new values — there are **17 values** in the enum (existing + new) that are missing from the CHECK constraint, including `hypothesis_flagged`, `agent_run_start`, `alpha_researcher_review`, `validation_analyst_review`, `risk_review_complete`, `risk_veto`, `quant_developer_backtest_complete`, `alpha_researcher_complete`, `pipeline_orchestrator_complete`, `kill_gate_triggered`, `ml_quality_sentinel_audit`, and `data_ingestion`, plus the 5 new ones. Generate the list programmatically or manually enumerate all values. This is DDL — existing tables need migration via `ALTER TABLE` or table recreation.

### Step 3: Fix Code Materializer event emission

**File:** `hrp/agents/code_materializer.py:67`

Already references `EventType.CODE_MATERIALIZER_COMPLETE` — will work once Step 1 adds it to the enum. No code change needed here.

### Step 4: Fix ML Scientist to emit `experiment_completed`

**File:** `hrp/agents/ml_scientist.py` (around line 217-224)

After walk-forward validation completes for each hypothesis, emit `EventType.EXPERIMENT_COMPLETED` with `hypothesis_id` and best experiment's `experiment_id`. Insert after the status counter updates (line 222) but still inside the `try` block:

```python
# Emit experiment_completed for downstream triggers
self._log_agent_event(
    event_type=EventType.EXPERIMENT_COMPLETED,
    hypothesis_id=hypothesis.get("hypothesis_id"),
    experiment_id=best.experiment_id if best else None,
    details={
        "status": status,
        "trials": len(hyp_results),
        "best_ic": best.mean_ic if best else None,
    },
)
```

### Step 5: Fix Validation Analyst to emit `validation_analyst_complete`

**File:** `hrp/agents/validation_analyst.py` (around line 209-217)

After the existing `AGENT_RUN_COMPLETE` event at the end of `execute()`, add:

```python
self._log_agent_event(
    event_type=EventType.VALIDATION_ANALYST_COMPLETE,
    details={
        "hypotheses_validated": len(validations),
        "hypotheses_passed": passed_count,
        "hypotheses_failed": failed_count,
        "duration_seconds": duration,
    },
)
```

### Step 6: Fix Risk Manager to emit `risk_manager_assessment`

**File:** `hrp/agents/risk_manager.py` (around line 191-200)

RiskManager currently has no `AGENT_RUN_COMPLETE` emission (unlike other agents). The `RISK_MANAGER_ASSESSMENT` event will serve as the completion signal. Insert before the return at the end of `execute()`:

```python
self._log_agent_event(
    event_type=EventType.RISK_MANAGER_ASSESSMENT,
    details={
        "hypotheses_assessed": len(assessments),
        "hypotheses_passed": passed_count,
        "hypotheses_vetoed": vetoed_count,
        "duration_seconds": time.time() - start_time,
    },
)
```

### Step 7: Fix CIO Agent to emit `cio_agent_decision`

**File:** `hrp/agents/cio.py` (around line 259-267)

CIO Agent extends `SDKAgent` (line 118), so it inherits `_log_agent_event` from `base.py`. Use that for consistency with all other agents (the existing `self.api.log_event()` at line 417 is for model deployment — a different context). Before the return in `execute()`, add:

```python
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
```

Also add the import at the top of `cio.py`:

```python
from hrp.research.lineage import EventType
```

### Step 8: Wire up the complete trigger chain

**Files:** `hrp/agents/run_job.py`, `hrp/agents/scheduler.py`

Update both files to register 9 triggers (currently 6). Changes needed:

**8a. Rewire trigger 2:** Change `alpha_researcher_complete` target from ML Scientist to Code Materializer.

In `run_job.py` (line 173-186) and `scheduler.py` (line 1250-1270):

```python
# Trigger 2: Alpha Researcher → Code Materializer
def on_alpha_researcher_complete(event: dict) -> None:
    details = event.get("details", {})
    promoted_ids = details.get("reviewed_ids", [])
    for hypothesis_id in promoted_ids:
        logger.info(f"Triggering Code Materializer for hypothesis {hypothesis_id}")
        from hrp.agents.code_materializer import CodeMaterializer
        materializer = CodeMaterializer(hypothesis_ids=[hypothesis_id])
        materializer.run()

watcher.register_trigger(
    event_type="alpha_researcher_complete",
    callback=on_alpha_researcher_complete,
    actor_filter="agent:alpha-researcher",
    name="alpha_researcher_to_code_materializer",
)
```

**8b. Add trigger 3 (new):** Code Materializer → ML Scientist.

```python
# Trigger 3: Code Materializer → ML Scientist
def on_code_materializer_complete(event: dict) -> None:
    hypothesis_id = event.get("hypothesis_id")
    if hypothesis_id:
        logger.info(f"Triggering ML Scientist for hypothesis {hypothesis_id}")
        scientist = MLScientist(hypothesis_ids=[hypothesis_id])
        scientist.run()

watcher.register_trigger(
    event_type="code_materializer_complete",
    callback=on_code_materializer_complete,
    actor_filter="agent:code-materializer",
    name="code_materializer_to_ml_scientist",
)
```

**8c. Add trigger 8 (new):** Validation Analyst → Risk Manager.

```python
# Trigger 8: Validation Analyst → Risk Manager
def on_validation_analyst_complete(event: dict) -> None:
    details = event.get("details", {})
    passed = details.get("hypotheses_passed", 0)
    if passed > 0:
        logger.info(f"Triggering Risk Manager for {passed} passed hypotheses")
        from hrp.agents.risk_manager import RiskManager
        risk_mgr = RiskManager(hypothesis_ids=None, send_alerts=True)
        risk_mgr.run()

watcher.register_trigger(
    event_type="validation_analyst_complete",
    callback=on_validation_analyst_complete,
    actor_filter="agent:validation-analyst",
    name="validation_analyst_to_risk_manager",
)
```

**8d. Add trigger 9 (new):** Risk Manager → CIO Agent.

```python
# Trigger 9: Risk Manager → CIO Agent
def on_risk_manager_assessment(event: dict) -> None:
    details = event.get("details", {})
    passed = details.get("hypotheses_passed", 0)
    if passed > 0:
        logger.info(f"Triggering CIO Agent for {passed} risk-cleared hypotheses")
        from hrp.agents.cio import CIOAgent
        agent = CIOAgent(
            job_id=f"cio-triggered-{date.today().strftime('%Y%m%d')}",
            actor="agent:cio",
        )
        agent.execute()

watcher.register_trigger(
    event_type="risk_manager_assessment",
    callback=on_risk_manager_assessment,
    actor_filter="agent:risk-manager",
    name="risk_manager_to_cio_agent",
)
```

### Step 9: Update scheduler docstring

**File:** `hrp/agents/scheduler.py` (line 1186-1200)

Update the `setup_research_agent_triggers` docstring to reflect the full 9-trigger chain:

```
Full trigger chain:
- Signal Scientist (hypothesis_created) → Alpha Researcher
- Alpha Researcher (alpha_researcher_complete) → Code Materializer
- Code Materializer (code_materializer_complete) → ML Scientist
- ML Scientist (experiment_completed) → ML Quality Sentinel
- ML Quality Sentinel (ml_quality_sentinel_audit, passed) → Quant Developer
- Quant Developer (quant_developer_backtest_complete) → Pipeline Orchestrator
- Pipeline Orchestrator (pipeline_orchestrator_complete) → Validation Analyst
- Validation Analyst (validation_analyst_complete) → Risk Manager
- Risk Manager (risk_manager_assessment) → CIO Agent
```

### Step 10: Verify docs match code

**File:** `docs/agents/decision-pipeline.md`

The doc already matches the target state. No changes needed.

## Files to Modify

| File | Change |
|------|--------|
| `hrp/research/lineage.py` | Add 5 new EventType enum values |
| `hrp/data/schema.py` | Update CHECK constraint to include all event types |
| `hrp/agents/ml_scientist.py` | Emit `experiment_completed` per hypothesis |
| `hrp/agents/validation_analyst.py` | Emit `validation_analyst_complete` at end of `execute()` |
| `hrp/agents/risk_manager.py` | Emit `risk_manager_assessment` at end of `execute()` |
| `hrp/agents/cio.py` | Import `EventType`, emit `cio_agent_decision` via `_log_agent_event` |
| `hrp/agents/run_job.py` | Add 3 new triggers, rewire Alpha Researcher → Code Materializer |
| `hrp/agents/scheduler.py` | Same trigger changes as run_job.py, update docstring |

## Verification

1. `pytest tests/ -v` — no regressions
2. Enum/schema sync: verify every `EventType` value appears in the DB CHECK constraint
3. Import test: `python -c "from hrp.agents.code_materializer import CodeMaterializer"` — no crash
4. Lineage write test: write each new event type to lineage table, verify no CHECK constraint violations
5. Trigger chain test: manually insert a `hypothesis_created` lineage event and run `agent-pipeline` job, verify downstream triggers fire through the full chain

## Risk Assessment

- **Low risk:** Steps 1-3 (enum additions, CHECK constraint update, Code Materializer fix) — additive changes only
- **Medium risk:** Steps 4-7 (new event emissions) — existing agents gain new log calls; could fail if DB migration not applied first
- **Medium risk:** Steps 8 (trigger rewiring) — changes existing Alpha Researcher → ML Scientist link; if Code Materializer has issues, ML Scientist won't be triggered
- **Mitigation:** Run Step 2 (DB migration) before deploying any agent changes
