# Scheduler Configuration Guide

## Architecture

HRP uses **individual launchd jobs** instead of a long-lived daemon. Each job runs at its scheduled time, holds the DuckDB lock briefly, then exits. This eliminates DB lock contention between scheduled jobs and interactive use (MCP server, dashboard).

**Key components:**
- `hrp/agents/run_job.py` — CLI that runs a single job and exits
- `launchd/com.hrp.*.plist` — macOS launchd job definitions (12 files)
- `scripts/manage_launchd.sh` — Install/uninstall helper

## Quick Start

```bash
# Install all jobs
scripts/manage_launchd.sh install

# Check status
scripts/manage_launchd.sh status

# Test a job manually
python -m hrp.agents.run_job --job prices --dry-run
```

## Schedule

### Daily Jobs

| Time (ET) | Job | Description |
|-----------|-----|-------------|
| 02:00 | `backup` | Database + MLflow backup |
| 06:00 | `quality-monitoring` | Data quality checks with alerting |
| 06:15 | `quality-sentinel` | ML Quality Sentinel audit |
| 07:00 | `daily-report` | Daily research report |
| 18:00 | `prices` | Price ingestion (after market close) |
| 18:05 | `universe` | Universe update |
| 18:10 | `features` | Feature computation |

### Periodic Jobs

| Schedule | Job | Description |
|----------|-----|-------------|
| Every 15 min | `agent-pipeline` | Check lineage events, run downstream agents |
| Mon 19:00 | `signal-scan` | Weekly signal discovery scan |
| Fri 17:00 | `cio-review` | Weekly CIO Agent review |
| Sat 10:00 | `fundamentals` | Weekly fundamentals ingestion |
| Sun 20:00 | `weekly-report` | Weekly research report |

## Agent Pipeline Chaining

The `agent-pipeline` job runs every 15 minutes and replaces the old `LineageEventWatcher` daemon. It polls the lineage table for unprocessed events and runs downstream agents:

```
Signal Scientist (hypothesis_created) → Alpha Researcher
Alpha Researcher (complete)           → ML Scientist
ML Scientist (experiment_completed)   → ML Quality Sentinel
ML Quality Sentinel (audit passed)    → Quant Developer
Quant Developer (backtest complete)   → Pipeline Orchestrator
Pipeline Orchestrator (complete)      → Validation Analyst
```

Each invocation processes all pending events, then exits. Worst-case latency for a single chain step is 15 minutes.

## Management

```bash
# Install all jobs (also removes old daemon plist)
scripts/manage_launchd.sh install

# Uninstall all jobs
scripts/manage_launchd.sh uninstall

# Show loaded jobs
scripts/manage_launchd.sh status

# Reload after editing plists
scripts/manage_launchd.sh reload
```

## Running Jobs Manually

```bash
# Run with real execution
python -m hrp.agents.run_job --job prices

# Dry run (log what would happen without executing)
python -m hrp.agents.run_job --job prices --dry-run

# Signal scan with custom IC threshold
python -m hrp.agents.run_job --job signal-scan --ic-threshold 0.05

# Fundamentals with specific source
python -m hrp.agents.run_job --job fundamentals --fundamentals-source yfinance
```

## Customizing Schedules

Edit the plist files in `launchd/` and reload:

```bash
# Edit schedule
$EDITOR launchd/com.hrp.prices.plist

# Reload to pick up changes
scripts/manage_launchd.sh reload
```

### StartCalendarInterval format

```xml
<!-- Daily at 18:00 -->
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>18</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>

<!-- Monday at 19:00 (Weekday: 1=Mon, 0=Sun) -->
<key>StartCalendarInterval</key>
<dict>
    <key>Weekday</key>
    <integer>1</integer>
    <key>Hour</key>
    <integer>19</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

### StartInterval format (for polling jobs)

```xml
<!-- Every 15 minutes (900 seconds) -->
<key>StartInterval</key>
<integer>900</integer>
```

## Logs

Each job writes to `~/hrp-data/logs/<job-name>.log` and `~/hrp-data/logs/<job-name>.error.log`.

```bash
# View recent output for a job
tail -f ~/hrp-data/logs/prices.log

# View errors
tail -f ~/hrp-data/logs/prices.error.log

# View all HRP logs
ls -lt ~/hrp-data/logs/*.log
```

## Troubleshooting

### Jobs not running

```bash
# Check if jobs are loaded
scripts/manage_launchd.sh status

# Check launchd errors for a specific job
launchctl list com.hrp.prices

# Exit status 0 = last run succeeded, non-zero = failed
```

### DB lock errors

If a job reports `DuckDB lock`, another job or process is writing. The individual-job architecture minimizes this since each job holds the lock briefly. If it persists:

1. Check if the old daemon is still running: `pgrep -f run_scheduler`
2. Kill it: `kill $(pgrep -f run_scheduler)`
3. Ensure `com.hrp.scheduler.plist` is removed: `scripts/manage_launchd.sh install` handles this

### Job runs but no output

Check that `PYTHONPATH` and `WorkingDirectory` in the plist match your repo location. If you moved the repo, update the plists and reload.

## Legacy Daemon

The old `run_scheduler.py` daemon is preserved for manual use:

```bash
python -m hrp.agents.run_scheduler --with-quality-monitoring
```

This is not recommended for production since it holds a persistent DB connection. Use the individual launchd jobs instead.
