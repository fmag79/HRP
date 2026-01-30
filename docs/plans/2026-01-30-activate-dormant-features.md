# Activate Dormant Features — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up all implemented-but-unused features so the platform operates end-to-end from signal discovery through model staging.

**Architecture:** 8 changes grouped by effort — bug fixes first, then wiring, dead code removal, and ML deployment integration. Each task is independent except Task 7 depends on Task 5 (removing dead code before adding new code to platform.py).

**Tech Stack:** Python 3.11, Streamlit, APScheduler, MLflow, DuckDB

---

### Task 1: Fix missing email method

**Files:**
- Modify: `hrp/api/platform.py` (lines 1397-1414)
- Test: `tests/test_api/test_platform.py`

**Step 1: Write the failing test**

```python
# In tests/test_api/test_platform.py — add to existing test class
def test_send_quality_alerts_uses_summary_email(self, mock_db):
    """Test that quality alerts use send_summary_email instead of nonexistent send_quality_alert."""
    api = PlatformAPI()

    # Create a mock report
    from unittest.mock import MagicMock, patch
    mock_report = MagicMock()
    mock_report.health_score = 75.0
    mock_report.critical_issues = 2
    mock_report.warning_issues = 3
    mock_report.results = []
    mock_report.generated_at.isoformat.return_value = "2026-01-30T00:00:00"

    with patch("hrp.notifications.email.EmailNotifier.send_summary_email", return_value=True) as mock_send:
        api._send_quality_alerts(mock_report)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert "Quality Alert" in call_args.kwargs.get("subject", call_args[0][0] if call_args[0] else "")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_platform.py::TestPlatformAPI::test_send_quality_alerts_uses_summary_email -v`
Expected: FAIL (AttributeError: send_quality_alert)

**Step 3: Implement the fix**

In `hrp/api/platform.py`, replace lines 1397-1414:

```python
def _send_quality_alerts(self, report) -> None:
    """Send email alerts for critical quality issues."""
    try:
        from hrp.notifications.email import EmailNotifier

        notifier = EmailNotifier()
        notifier.send_summary_email(
            subject=f"HRP Quality Alert: {report.critical_issues} critical issues (score: {report.health_score:.0f})",
            summary_data={
                "health_score": report.health_score,
                "critical_issues": report.critical_issues,
                "warning_issues": report.warning_issues,
                "timestamp": report.generated_at.isoformat(),
            },
        )
        logger.info(
            f"Sent quality alerts: {report.critical_issues} critical issues"
        )
    except Exception as e:
        logger.error(f"Failed to send quality alerts: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api/test_platform.py::TestPlatformAPI::test_send_quality_alerts_uses_summary_email -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/api/platform.py tests/test_api/test_platform.py
git commit -m "fix(api): use send_summary_email for quality alerts instead of nonexistent method"
```

---

### Task 2: Add dashboard pages to navigation

**Files:**
- Modify: `hrp/dashboard/app.py` (lines 890-894 and 999-1010)

**Step 1: Update navigation selectbox**

In `hrp/dashboard/app.py` line 892, change the options list:

```python
options=["Home", "Data Health", "Ingestion Status", "Hypotheses", "Experiments", "Agents Monitor", "Job Health"],
```

**Step 2: Add routing branches**

After line 1008 (`elif page == "Experiments": render_experiments()`), add:

```python
    elif page == "Agents Monitor":
        from hrp.dashboard.pages.agents_monitor import render_agents_monitor
        render_agents_monitor()
    elif page == "Job Health":
        from hrp.dashboard.pages.job_health import render_job_health
        render_job_health()
```

Note: Check the actual function names exported from those page modules. The `2_Agents_Monitor.py` file may use a Streamlit pages convention. Read it first to confirm the render function name.

**Step 3: Verify manually**

Run: `streamlit run hrp/dashboard/app.py`
Check: Both pages appear in sidebar and load without errors.

**Step 4: Commit**

```bash
git add hrp/dashboard/app.py
git commit -m "feat(dashboard): add Agents Monitor and Job Health to sidebar navigation"
```

---

### Task 3: Scheduler two-tier flags

**Files:**
- Modify: `hrp/agents/run_scheduler.py` (full rewrite of CLI args and setup logic)
- Test: `tests/test_agents/test_scheduler.py` (if scheduler CLI tests exist)

**Step 1: Replace CLI arguments**

Remove these individual flags (lines 107-206 in run_scheduler.py):
- `--with-research-triggers`, `--trigger-poll-interval`
- `--with-signal-scan`, `--signal-scan-time`, `--signal-scan-day`, `--ic-threshold`
- `--with-quality-sentinel`, `--sentinel-time`
- `--with-daily-report`, `--daily-report-time`
- `--with-weekly-report`, `--weekly-report-time`
- `--with-quality-monitoring`, `--quality-monitor-time`, `--health-threshold`
- `--with-cio-review`, `--cio-review-time`, `--cio-review-day`

Replace with two flags:

```python
    # Tier-based job groups
    parser.add_argument(
        "--with-data-jobs",
        action="store_true",
        help="Enable extended data jobs: fundamentals timeseries, snapshot fundamentals, sectors, cleanup",
    )
    parser.add_argument(
        "--with-research-pipeline",
        action="store_true",
        help="Enable full research pipeline: signal scan, alpha researcher, pipeline orchestrator, "
             "quality sentinel, CIO review, reports, event-driven agent triggers",
    )
```

**Step 2: Replace setup logic**

Replace the individual flag checks (lines 240-291) with:

```python
    # Extended data jobs
    if args.with_data_jobs:
        logger.info("Setting up extended data jobs...")
        scheduler.setup_weekly_fundamentals_timeseries()
        scheduler.setup_weekly_snapshot_fundamentals()
        scheduler.setup_weekly_sectors()
        scheduler.setup_weekly_cleanup()

    # Full research pipeline
    if args.with_research_pipeline:
        logger.info("Setting up full research pipeline...")
        scheduler.setup_weekly_signal_scan()
        scheduler.setup_model_monitoring(send_alerts=True)
        scheduler.setup_daily_report()
        scheduler.setup_weekly_report()
        scheduler.setup_quality_monitoring(send_alerts=True)
        scheduler.setup_weekly_cio_review()
        scheduler.setup_weekly_alpha_researcher()
        scheduler.setup_daily_pipeline_orchestrator()
        scheduler.setup_research_agent_triggers()
```

**Step 3: Update the start logic**

Change the start condition (lines 293-299):

```python
    # Start scheduler
    if args.with_research_pipeline:
        logger.info("Starting scheduler with research agent triggers...")
        scheduler.start_with_triggers()
    else:
        logger.info("Starting scheduler...")
        scheduler.start()
```

**Step 4: Update module docstring**

Replace the docstring (lines 1-23) with:

```python
"""
Run the HRP data ingestion scheduler.

Always runs:
- Daily price ingestion at 6:00 PM ET
- Daily universe update at 6:05 PM ET
- Daily feature computation at 6:10 PM ET
- Daily backup at 2:00 AM ET
- Weekly fundamentals ingestion (Saturday 10 AM ET)

--with-data-jobs adds:
- Weekly fundamentals timeseries, snapshot fundamentals, sectors, cleanup

--with-research-pipeline adds:
- Signal scan, alpha researcher, pipeline orchestrator
- Model monitoring, quality monitoring, CIO review
- Daily/weekly reports, event-driven agent triggers

Usage:
    python -m hrp.agents.run_scheduler
    python -m hrp.agents.run_scheduler --with-data-jobs
    python -m hrp.agents.run_scheduler --with-data-jobs --with-research-pipeline
"""
```

**Step 5: Update CLAUDE.md services table**

In CLAUDE.md, update the scheduler commands to reflect the new flags.

**Step 6: Run existing tests**

Run: `pytest tests/test_agents/ -v -k scheduler`
Expected: PASS (fix any broken tests referencing old flags)

**Step 7: Commit**

```bash
git add hrp/agents/run_scheduler.py CLAUDE.md
git commit -m "refactor(scheduler): replace granular flags with --with-data-jobs and --with-research-pipeline"
```

---

### Task 4: Event-driven agents (resolved by Task 3)

No separate implementation needed. When `--with-research-pipeline` calls `setup_research_agent_triggers()`, the ValidationAnalyst, RiskManager, and QuantDeveloper agents are automatically activated through the `LineageEventWatcher`. Verify this by reading the trigger setup code.

**Verification:**

Run: `grep -n "ValidationAnalyst\|RiskManager\|QuantDeveloper" hrp/agents/scheduler.py`
Expected: These agents are instantiated inside the event-driven trigger handlers.

---

### Task 5: Remove unused price adjustment API methods

**Files:**
- Modify: `hrp/api/platform.py` (delete lines 379-563)
- Modify: `tests/test_api/test_platform.py` (remove corresponding tests)
- Modify: `tests/test_api/test_corporate_actions_api.py` (remove if it tests these methods)

**Step 1: Identify tests to remove**

Run: `grep -n "adjust_prices_for_splits\|adjust_prices_for_dividends" tests/test_api/test_platform.py tests/test_api/test_corporate_actions_api.py tests/test_research/test_backtest.py`

Remove test functions that directly test these two API methods. Keep any tests that test the internal backtest adjustment logic.

**Step 2: Delete the methods from platform.py**

Remove `adjust_prices_for_splits()` (lines 379-453) and `adjust_prices_for_dividends()` (lines 455-563) from `hrp/api/platform.py`.

**Step 3: Delete corresponding tests**

Remove test functions identified in Step 1.

**Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: PASS with no import errors. If anything imported these methods, fix the imports.

**Step 5: Commit**

```bash
git add hrp/api/platform.py tests/
git commit -m "refactor(api): remove unused adjust_prices_for_splits and adjust_prices_for_dividends"
```

---

### Task 6: Simplify data retention

**Files:**
- Modify: `hrp/data/retention/policy.py` — delete `RetentionTier`, `RetentionPolicy`, `RetentionEngine`, `DEFAULT_POLICIES`
- Modify: `hrp/data/retention/cleanup.py` — delete `DataArchivalJob`, keep `DataCleanupJob`
- Modify: `hrp/data/retention/__init__.py` — update exports
- Modify: `tests/test_data/test_retention.py` — remove tests for deleted classes

**Step 1: Simplify `__init__.py`**

```python
"""
Data cleanup for HRP.

Provides scheduled cleanup of orphaned records and old logs.
"""

from hrp.data.retention.cleanup import DataCleanupJob

__all__ = [
    "DataCleanupJob",
]
```

**Step 2: Remove archival classes from cleanup.py**

Delete the `DataArchivalJob` class. Keep `DataCleanupJob` and `CleanupResult`.

Update `DataCleanupJob` to remove any dependency on `RetentionEngine` or `RetentionTier`. If it uses `RetentionEngine.get_cleanup_candidates()`, replace with direct DuckDB queries for old records.

**Step 3: Delete or gut policy.py**

If `DataCleanupJob` no longer imports from `policy.py`, delete the file entirely. If it still needs a simple threshold, inline the logic.

**Step 4: Update tests**

Remove tests for `RetentionPolicy`, `RetentionEngine`, `DataArchivalJob`, `RetentionTier`. Keep tests for `DataCleanupJob`.

**Step 5: Run tests**

Run: `pytest tests/test_data/test_retention.py -v`
Expected: PASS

Run: `pytest tests/ -v --tb=short`
Expected: No import errors anywhere.

**Step 6: Commit**

```bash
git add hrp/data/retention/ tests/test_data/test_retention.py
git commit -m "refactor(retention): remove archival tier system, keep DataCleanupJob only"
```

---

### Task 7: Wire ML deployment into CIO Agent

**Files:**
- Modify: `hrp/agents/cio.py` (lines 240-256, inside execute() after decisions loop)
- Test: `tests/test_agents/test_cio.py`

**Step 1: Write the failing test**

```python
def test_cio_stages_model_on_continue_decision(self):
    """CIO Agent should register and stage model for CONTINUE decisions."""
    from unittest.mock import MagicMock, patch

    cio = CIOAgent()

    # Mock a CONTINUE decision with experiment data
    mock_score = MagicMock()
    mock_score.decision = "CONTINUE"
    mock_score.total = 0.85

    with patch.object(cio, "_stage_model_for_deployment") as mock_stage:
        cio._maybe_stage_model(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            experiment_data={"experiment_id": "exp-123", "model_type": "ridge"},
        )
        mock_stage.assert_called_once_with(
            hypothesis_id="HYP-2026-001",
            experiment_data={"experiment_id": "exp-123", "model_type": "ridge"},
        )


def test_cio_does_not_stage_on_kill_decision(self):
    """CIO Agent should NOT stage model for KILL decisions."""
    cio = CIOAgent()

    with patch.object(cio, "_stage_model_for_deployment") as mock_stage:
        cio._maybe_stage_model(
            hypothesis_id="HYP-2026-001",
            decision="KILL",
            experiment_data={"experiment_id": "exp-123"},
        )
        mock_stage.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_cio.py -v -k "stage_model"`
Expected: FAIL (AttributeError: _maybe_stage_model)

**Step 3: Implement in cio.py**

Add two new methods to CIOAgent:

```python
def _maybe_stage_model(
    self,
    hypothesis_id: str,
    decision: str,
    experiment_data: dict,
) -> None:
    """Stage model for deployment if decision is CONTINUE."""
    if decision != "CONTINUE":
        return
    self._stage_model_for_deployment(hypothesis_id, experiment_data)

def _stage_model_for_deployment(
    self,
    hypothesis_id: str,
    experiment_data: dict,
) -> None:
    """Register model and deploy to staging."""
    try:
        experiment_id = experiment_data.get("experiment_id")
        model_type = experiment_data.get("model_type", "unknown")
        model_name = f"hyp_{hypothesis_id}_{model_type}"

        self.api.register_model(
            model=None,  # MLflow will load from experiment
            model_name=model_name,
            model_type=model_type,
            features=experiment_data.get("features", []),
            target=experiment_data.get("target", "returns_20d"),
            metrics=experiment_data.get("metrics", {}),
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
        )

        self.api.deploy_model(
            model_name=model_name,
            model_version="1",
            validation_data={},
            environment="staging",
            actor="agent:cio",
        )

        self.api.log_event(
            event_type="model_deployed",
            actor="agent:cio",
            hypothesis_id=hypothesis_id,
            details={
                "model_name": model_name,
                "environment": "staging",
                "experiment_id": experiment_id,
            },
        )

        logger.info(f"Staged model {model_name} for hypothesis {hypothesis_id}")

    except Exception as e:
        logger.error(f"Failed to stage model for {hypothesis_id}: {e}")
```

Then in `execute()`, after line 246 (inside the decisions loop, after `decisions.append(...)`), add:

```python
            # Stage model if CONTINUE
            self._maybe_stage_model(
                hypothesis_id=hypothesis_id,
                decision=score.decision,
                experiment_data=experiment_data,
            )
```

**Step 4: Run tests**

Run: `pytest tests/test_agents/test_cio.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/agents/cio.py tests/test_agents/test_cio.py
git commit -m "feat(cio): auto-stage model to staging on CONTINUE decision"
```

---

### Task 8: Add drift check to predict_model()

**Files:**
- Modify: `hrp/api/platform.py` (predict_model method, lines 1740-1786)
- Test: `tests/test_api/test_platform.py`

**Step 1: Write the failing test**

```python
def test_predict_model_runs_drift_check(self, mock_db):
    """predict_model should run drift check after generating predictions."""
    api = PlatformAPI()

    with patch("hrp.ml.inference.ModelPredictor") as MockPredictor, \
         patch.object(api, "check_model_drift") as mock_drift, \
         patch.object(api, "log_event"):

        mock_predictor = MockPredictor.return_value
        mock_predictor.predict_batch.return_value = pd.DataFrame({
            "symbol": ["AAPL"], "prediction": [0.05],
        })
        mock_predictor.model_version = "1"

        mock_drift.return_value = {"drift_detected": False}

        api.predict_model(
            model_name="test_model",
            symbols=["AAPL"],
            as_of_date=date(2026, 1, 30),
        )

        mock_drift.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_platform.py -v -k "predict_model_runs_drift"`
Expected: FAIL (check_model_drift not called)

**Step 3: Add drift check to predict_model**

In `hrp/api/platform.py`, after the lineage log_event call in predict_model (line 1784), add:

```python
        # Run drift check
        try:
            drift_result = self.check_model_drift(
                model_name=model_name,
                current_data=predictions,
                reference_data=None,
            )
            if drift_result.get("drift_detected"):
                logger.warning(f"Drift detected for model {model_name}")
                self.log_event(
                    event_type="model_drift_detected",
                    actor="system",
                    details={
                        "model_name": model_name,
                        "drift_score": drift_result.get("drift_score"),
                    },
                )
        except Exception as e:
            logger.warning(f"Drift check failed for {model_name}: {e}")
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_platform.py -v -k "predict_model"`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/api/platform.py tests/test_api/test_platform.py
git commit -m "feat(api): run drift check automatically after predict_model"
```

---

### Task 9: Update documentation

**Files:**
- Modify: `CLAUDE.md` — update scheduler commands
- Modify: `docs/plans/Project-Status.md` — update Tier 2 status to reflect wired-up features

**Step 1: Update CLAUDE.md scheduler section**

Replace the scheduler service commands:

```markdown
| Scheduler | `python -m hrp.agents.run_scheduler` | - |
| Scheduler (full) | `python -m hrp.agents.run_scheduler --with-data-jobs --with-research-pipeline` | - |
```

**Step 2: Commit**

```bash
git add CLAUDE.md docs/plans/Project-Status.md
git commit -m "docs: update scheduler flags and project status for activated features"
```
