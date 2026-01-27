# Report Generator Implementation Plan

**Feature:** F-026 Report Generator Agent
**Status:** Ready to Implement
**Estimated Effort:** ~3-4 hours

---

## Overview

Implement the Report Generator agent to synthesize research findings into human-readable daily and weekly reports. This agent completes the Tier 2 Intelligence tier by providing the final synthesis capability.

---

## Implementation Tasks

### Task 1: Create Report Generator Base Class

**File:** `hrp/agents/report_generator.py`

**Steps:**
1. Create `ReportGeneratorConfig` dataclass extending `SDKAgentConfig`
   - Add `report_type`, `report_dir`, `include_sections`, `lookback_days` fields
2. Create `ReportGenerator` class extending `SDKAgent`
   - Set `ACTOR = "agent:report-generator"`
   - Implement `__init__` with config handling
   - Create stub `execute()` method

**Verification:**
```bash
# Test instantiation
python -c "
from hrp.agents.report_generator import ReportGenerator
gen = ReportGenerator(report_type='daily')
print(f'Actor: {gen.ACTOR}')
print(f'Config: {gen.config}')
"
# Expected output: Actor: agent:report-generator
```

---

### Task 2: Implement Data Gathering Methods

**File:** `hrp/agents/report_generator.py`

**Steps:**
1. Implement `_gather_hypothesis_data()`
   - Query hypotheses by status (draft, testing, validated)
   - Count pipeline stage conversions
   - Get recent validated hypotheses
   - Return dict with counts and sample data

2. Implement `_gather_experiment_data()`
   - Query MLflow for recent experiments
   - Get top 5 by Sharpe ratio
   - Aggregate model performance statistics
   - Return dict with experiment summaries

3. Implement `_gather_signal_data()`
   - Query lineage for `SIGNAL_SCIENTIST_RUN` events
   - Extract IC values and discovered signals
   - Return dict with signal statistics

4. Implement `_gather_agent_activity()`
   - Query lineage for recent agent runs
   - Get status of each agent (Signal Scientist, Alpha Researcher, etc.)
   - Return dict with agent activity summary

**Verification:**
```bash
# Test data gathering (requires test DB)
pytest tests/test_agents/test_report_generator.py::test_gather_hypothesis_data -v
pytest tests/test_agents/test_report_generator.py::test_gather_experiment_data -v
pytest tests/test_agents/test_report_generator.py::test_gather_signal_data -v
pytest tests/test_agents/test_report_generator.py::test_gather_agent_activity -v
```

---

### Task 3: Implement Claude-Powered Insights Generation

**File:** `hrp/agents/report_generator.py`

**Steps:**
1. Implement `_build_insights_prompt(context, report_type)`
   - Create daily-specific prompt template
   - Create weekly-specific prompt template
   - Include all gathered context in prompt

2. Implement `_generate_insights(context, report_type)`
   - Call `self.invoke_claude()` with insights prompt
   - Parse JSON response from Claude
   - Handle parsing failures with fallback
   - Return list of insight dicts

3. Implement `_get_system_prompt()`
   - Return system prompt for Report Generator role
   - Emphasize actionable, concise recommendations

**Verification:**
```bash
# Test insights generation (mock Claude)
pytest tests/test_agents/test_report_generator.py::test_generate_insights -v --mock-claude
```

---

### Task 4: Implement Report Rendering

**File:** `hrp/agents/report_generator.py`

**Steps:**
1. Implement `_render_daily_report(context, insights)`
   - Build markdown sections for daily report
   - Include: executive summary, hypothesis pipeline, experiments, signals, insights, agent activity
   - Use context data and generated insights
   - Return complete markdown string

2. Implement `_render_weekly_report(context, insights)`
   - Build markdown sections for weekly report
   - Include extended sections: pipeline velocity, top hypotheses, experiment insights, signal discoveries
   - Add trend analysis and next week priorities
   - Return complete markdown string

3. Implement `_get_report_filename()`
   - Generate filename: `YYYY-MM-DD-HH-MM-{daily,weekly}.md`
   - Use current datetime

**Verification:**
```bash
# Test report rendering
pytest tests/test_agents/test_report_generator.py::test_render_daily_report -v
pytest tests/test_agents/test_report_generator.py::test_render_weekly_report -v
```

---

### Task 5: Implement Report Writing

**File:** `hrp/agents/report_generator.py`

**Steps:**
1. Implement `_write_report(markdown, report_type)`
   - Create dated directory: `docs/reports/YYYY-MM-DD/`
   - Write markdown to file with timestamped filename
   - Log output location
   - Return filepath

2. Update `execute()` method
   - Call all gather methods
   - Generate insights
   - Render report (daily or weekly)
   - Write report to disk
   - Return result dict with filepath and metadata

**Verification:**
```bash
# Test report writing
pytest tests/test_agents/test_report_generator.py::test_write_report -v

# Verify file structure
python -c "
from hrp.agents.report_generator import ReportGenerator
result = ReportGenerator(report_type='daily').run()
print(f'Report written to: {result[\"report_path\"]}')

import os
assert os.path.exists(result['report_path'])
print('✅ File exists')
"
```

---

### Task 6: Add MCP Tool Integration

**File:** `hrp/mcp/research_server.py`

**Steps:**
1. Add `run_report_generator` tool
   - Parameters: `report_type: str` ("daily" or "weekly")
   - Calls `ReportGenerator(report_type).run()`
   - Returns report path and summary

2. Update tool registry

**Verification:**
```bash
# Test MCP tool
pytest tests/test_mcp/test_tools.py::test_run_report_generator -v
```

---

### Task 7: Add Scheduler Integration

**File:** `hrp/agents/scheduler.py`

**Steps:**
1. Implement `setup_daily_report(time="07:00")`
   - Add scheduled job for daily report
   - Uses Cron trigger at specified time

2. Implement `setup_weekly_report(time="20:00")`
   - Add scheduled job for Sunday weekly report
   - Uses Cron trigger with `day_of_week="sun"`

3. Add CLI flags to `run_scheduler.py`
   - `--with-daily-report`: Enable daily reports
   - `--daily-report-time`: Set daily time (default 07:00)
   - `--with-weekly-report`: Enable weekly reports
   - `--weekly-report-time`: Set weekly time (default 20:00)

**Verification:**
```bash
# Test scheduler integration
pytest tests/test_agents/test_scheduler.py::test_daily_report_scheduling -v
pytest tests/test_agents/test_scheduler.py::test_weekly_report_scheduling -v

# Test CLI flags
python -m hrp.agents.run_scheduler --help | grep report
```

---

### Task 8: Update Documentation

**Files:** `CLAUDE.md`, `docs/plans/Project-Status-Rodmap.md`

**Steps:**
1. Add Report Generator to research agents list in CLAUDE.md
2. Add usage examples for on-demand report generation
3. Update Project Status to show F-026 complete
4. Update Tier 2 progress to 100%

---

### Task 9: Write Tests

**File:** `tests/test_agents/test_report_generator.py`

**Test Classes:**
- `TestReportGeneratorConfig` - Config dataclass tests
- `TestReportGeneratorInit` - Initialization tests
- `TestGatherHypothesisData` - Hypothesis data gathering
- `TestGatherExperimentData` - MLflow data gathering
- `TestGatherSignalData` - Signal data gathering
- `TestGatherAgentActivity` - Agent activity tracking
- `TestGenerateInsights` - Claude integration (mocked)
- `TestRenderDailyReport` - Daily report rendering
- `TestRenderWeeklyReport` - Weekly report rendering
- `TestWriteReport` - File output
- `TestExecute` - End-to-end execution
- `TestSchedulerIntegration` - Scheduled execution

**Target:** 25+ tests, >90% coverage

---

### Task 10: Final Verification

**Steps:**
1. Run full test suite
2. Generate sample daily report
3. Generate sample weekly report
4. Verify file structure
5. Verify report content quality
6. Update documentation

**Verification Commands:**
```bash
# Run all tests
pytest tests/ -v

# Generate sample reports
python -c "
from hrp.agents.report_generator import ReportGenerator

# Daily report
daily_result = ReportGenerator(report_type='daily').run()
print(f'Daily report: {daily_result[\"report_path\"]}')

# Weekly report
weekly_result = ReportGenerator(report_type='weekly').run()
print(f'Weekly report: {weekly_result[\"report_path\"]}')
"

# Verify output
ls -la docs/reports/$(date +%Y-%m-%d)/
cat docs/reports/$(date +%Y-%m-%d)/*.md | head -50
```

---

## Success Criteria

- [ ] All 10 tasks completed
- [ ] Daily reports generate at scheduled time
- [ ] Weekly reports generate on Sunday evening
- [ ] Reports include all 6 sections
- [ ] Claude-powered insights are actionable
- [ ] File structure: `docs/reports/YYYY-MM-DD/HH-MM-{daily,weekly}.md`
- [ ] MCP tool available for on-demand runs
- [ ] Test coverage >90%
- [ ] Full test suite passes (2,115+ tests)
- [ ] Documentation updated
- [ ] Tier 2 Intelligence marked 100% complete

---

## Dependencies

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| SDKAgent base class | ✅ Built | `hrp/agents/sdk_agent.py` |
| Platform API | ✅ Built | `hrp/api/platform.py` |
| MLflow integration | ✅ Built | Existing in research layer |
| Lineage system | ✅ Built | `hrp/research/lineage.py` |
| Scheduler infrastructure | ✅ Built | `hrp/agents/scheduler.py` |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Claude API costs | Limit tokens, use Haiku for testing |
| MLflow connection issues | Graceful degradation, log warnings |
| Empty data (no hypotheses/experiments) | Handle gracefully, report "no activity" |
| File system permissions | Verify directory creation, log errors |

---

## Document History

- **2026-01-26:** Implementation plan created
