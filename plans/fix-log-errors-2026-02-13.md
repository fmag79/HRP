# Fix Plan: HRP Log Errors (2026-02-13)

Identified from error logs on 2026-02-13. Five issues, three root causes.

## Issue Summary

| # | Issue | Severity | Root Cause |
|---|-------|----------|------------|
| 1 | DuckDB connection conflict | Critical | Ops server holds persistent read-only connection; batch jobs open read-write — DuckDB rejects mixed modes on same file |
| 2 | Claude API auth failure (daily-report) | High | `ANTHROPIC_API_KEY` not available in launchd job environment |
| 3 | Features: 376/396 symbols skipped | Low | Expected — system has <60 days of price history; 60-day rolling window requirement |
| 4 | Foreign key constraint (ingestion_log) | High | Job IDs (e.g. `ml_quality_sentinel_audit`, `cio-weekly-20260213`) not registered in `data_sources` table |
| 5 | Email notifications disabled | Low | `RESEND_API_KEY` / `NOTIFICATION_EMAIL` not in launchd environment |

---

## Fix 1: DuckDB Connection Conflict (Critical)

**Problem:** The ops-server process (`com.hrp.ops-server`) runs persistently with `KeepAlive: true`. It creates a `PlatformAPI(read_only=True)` in `check_system_ready()` which initializes a `ConnectionPool` with `read_only=True`. When batch jobs (prices, features, etc.) start via launchd and open the same DuckDB file with `read_only=False`, DuckDB rejects the connection because you can't have mixed read-only and read-write connections to the same file simultaneously.

**Affected jobs:** agent-pipeline (every 15 min), quality-monitoring (daily), and any job that opens read-write while ops-server holds a connection.

**Root cause code:**

- `hrp/ops/server.py:50` — `PlatformAPI(read_only=True)` creates a singleton that persists
- `hrp/data/db.py:206-209` — `ConnectionPool` opens with `duckdb.connect(path, read_only=True)`
- Batch jobs import `get_db()` (default `read_only=False`) — `duckdb.connect(path)` (read-write)
- DuckDB enforces: all connections to a file must use the same mode

**Fix:** Change the ops server to use **read-write mode**, matching the batch jobs. The ops server only does SELECT queries, so read-write mode is safe — it simply won't write. This makes all processes use the same connection config.

**Files to modify:**
1. `hrp/ops/server.py:50` — Change `PlatformAPI(read_only=True)` to `PlatformAPI(read_only=False)`

**Alternative (if read-write singleton causes issues):** Make `check_system_ready()` use a short-lived non-singleton connection:
```python
api = PlatformAPI(read_only=False, use_singleton=False)
```
This opens, checks, and closes — no persistent connection blocking batch jobs.

**Verify:** After fix, run `python -m hrp.agents.run_job --job quality-monitoring` while ops-server is running. All 5 quality checks should pass without connection errors.

---

## Fix 2: Environment Variables Not Loaded in launchd Jobs (High)

**Problem:** `run_job.py` never imports `hrp.utils.config`, which is the module that calls `load_dotenv()`. Since launchd does **not** inherit the user's shell environment, env vars from `~/.zshrc` or `~/.bashrc` are unavailable. The `.env` file at `/Users/openclaw/Projects/HRP/.env` exists but is never loaded.

This causes:
- `ANTHROPIC_API_KEY` missing → Claude API calls fail in daily-report (Issue 2)
- `RESEND_API_KEY` missing → email notifications disabled (Issue 5)

**Root cause code:**
- `hrp/agents/run_job.py` — no `load_dotenv()` call, no import of `hrp.utils.config`
- `hrp/utils/config.py:15-18` — has `load_dotenv()` but only runs when imported
- `hrp/agents/sdk_agent.py` — relies on `ANTHROPIC_API_KEY` being in `os.environ`

**Fix:** Add `load_dotenv()` early in `run_job.py`'s `main()` before any job runs.

**File to modify:**
1. `hrp/agents/run_job.py` — Add at the top of the `main` block (before `_setup_logging`):
```python
from dotenv import load_dotenv
load_dotenv()
```

This loads `.env` from `WorkingDirectory` (`/Users/openclaw/Projects/HRP`), which is set in every launchd plist. All env vars (ANTHROPIC_API_KEY, RESEND_API_KEY, NOTIFICATION_EMAIL, etc.) will then be available.

**Verify:**
1. Confirm `.env` has `ANTHROPIC_API_KEY` set: `grep ANTHROPIC_API_KEY /Users/openclaw/Projects/HRP/.env`
2. Run `python -m hrp.agents.run_job --job daily-report` — Claude API calls should succeed
3. Check `~/hrp-data/logs/daily-report.error.log` — no auth errors

---

## Fix 3: Features — Insufficient Price Data (Low, No Action Needed)

**Problem:** 376/396 symbols skipped because they have fewer than 60 days of price history. The 60-day minimum is required for `volatility_60d` (60-day rolling window).

**Root cause code:**
- `hrp/data/ingestion/features.py:105-109` — `if len(prices_df) < 60: continue`

**Assessment:** This is **working as designed**. The system was recently bootstrapped and hasn't accumulated enough history. The `backfill_progress_20260212.json` file at project root confirms this is a new setup. After ~90 calendar days of daily price ingestion, all symbols will have 60+ trading days.

**No code changes needed.** Monitor: the 20/396 success count should increase daily as history accumulates.

**Optional acceleration:** If faster feature coverage is needed, backfill historical prices:
```bash
python -m hrp.agents.run_job --job fundamentals-backfill --days 365
```
(Only if a price backfill job exists — check `JOBS` dict in `run_job.py`.)

---

## Fix 4: Foreign Key Constraint on ingestion_log (High)

**Problem:** When jobs like `quality-sentinel` or `cio-review` start, `_log_start()` inserts into `ingestion_log` with a `source_id` like `ml_quality_sentinel_audit` or `cio-weekly-20260213`. The `ingestion_log` table has `FOREIGN KEY (source_id) REFERENCES data_sources(source_id)`, but these job IDs were never registered in `data_sources`.

**Root cause code:**
- `hrp/data/schema.py:455` — `FOREIGN KEY (source_id) REFERENCES data_sources(source_id)`
- `hrp/agents/jobs.py:353` — `self.api.log_job_start(self.job_id)` inserts dynamic job IDs
- `data_sources` table only has data provider entries (yfinance, simfin, etc.), not job IDs

**Impact:** Jobs complete successfully but their run history isn't tracked in `ingestion_log`. The `_log_success` fallback at line 370 logs a warning and continues.

**Fix:** Remove the FK constraint from `ingestion_log.source_id`. This matches the precedent already set in `hypothesis_experiments` (line 459 comment: "FK constraint removed due to DuckDB limitation"). The `source_id` column in `ingestion_log` serves as a label for what ran, not a strict reference to data sources.

**Files to modify:**
1. `hrp/data/schema.py:455` — Remove the line `FOREIGN KEY (source_id) REFERENCES data_sources(source_id)`

**Migration required:** The existing table needs the constraint dropped. Create a migration:
```sql
-- DuckDB doesn't support ALTER TABLE DROP CONSTRAINT directly.
-- Recreate the table without the FK:
CREATE TABLE ingestion_log_new AS SELECT * FROM ingestion_log;
DROP TABLE ingestion_log;
CREATE TABLE ingestion_log (
    log_id INTEGER PRIMARY KEY,
    source_id VARCHAR,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    records_fetched INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'running',
    error_message VARCHAR,
    CHECK (records_fetched >= 0),
    CHECK (records_inserted >= 0),
    CHECK (status IN ('running', 'completed', 'failed'))
);
INSERT INTO ingestion_log SELECT * FROM ingestion_log_new;
DROP TABLE ingestion_log_new;
```

**Verify:** Run `python -m hrp.agents.run_job --job quality-sentinel` — no FK constraint errors in log, `ingestion_log` entry created.

---

## Fix 5: Email Notifications (Low)

**Resolved by Fix 2.** Once `load_dotenv()` is added to `run_job.py`, `RESEND_API_KEY` and `NOTIFICATION_EMAIL` from `.env` will be loaded automatically.

**Prerequisite:** Ensure these values are set in `/Users/openclaw/Projects/HRP/.env`:
```
RESEND_API_KEY=re_xxxx
NOTIFICATION_EMAIL=your@email.com
```

If these aren't configured yet, this is a setup task, not a code fix.

---

## Implementation Order

1. **Fix 2** (load_dotenv in run_job.py) — one-line change, unblocks Issues 2 and 5
2. **Fix 1** (ops server connection mode) — one-line change, unblocks the most critical issue
3. **Fix 4** (FK constraint removal) — schema change + migration script
4. Reload launchd: `scripts/manage_launchd.sh reload`

## Verification Checklist

- [ ] `python -m hrp.agents.run_job --job quality-monitoring` — all 5 checks pass (no connection error)
- [ ] `python -m hrp.agents.run_job --job daily-report` — Claude API call succeeds (no auth error)
- [ ] `python -m hrp.agents.run_job --job quality-sentinel` — ingestion_log entry created (no FK error)
- [ ] `python -m hrp.agents.run_job --job agent-pipeline` — lineage poll succeeds (no connection error)
- [ ] `grep -c ERROR ~/hrp-data/logs/*.error.log` — error counts reduced vs. today
