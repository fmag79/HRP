# Phase 3: Monitoring Setup Guide

**Date:** January 24, 2026  
**Phase:** 3 of 5  
**Purpose:** Establish monitoring infrastructure for universe scheduling

---

## Overview

This guide provides monitoring queries, health checks, and operational procedures for the universe scheduling feature.

---

## 1. Database Monitoring Queries

### 1.1 Check Universe Update Job History

**Query Recent Universe Updates:**
```python
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT 
            source_id,
            status,
            records_fetched,
            records_inserted,
            started_at,
            completed_at,
            EXTRACT(EPOCH FROM (completed_at - started_at)) as duration_seconds,
            error_message
        FROM ingestion_log 
        WHERE source_id = 'universe_update'
        ORDER BY started_at DESC 
        LIMIT 10
    ''').fetchall()
    
    for row in result:
        print(f"Date: {row[4].date()}")
        print(f"  Status: {row[1]}")
        print(f"  Fetched: {row[2]}, Inserted: {row[3]}")
        print(f"  Duration: {row[6]:.2f}s")
        if row[7]:
            print(f"  Error: {row[7]}")
        print()
```

**Expected Output (Success):**
```
Date: 2026-01-25
  Status: completed
  Fetched: 503, Inserted: 396
  Duration: 0.51s

Date: 2026-01-24
  Status: completed
  Fetched: 503, Inserted: 396
  Duration: 0.52s
```

### 1.2 Check Universe Lineage Events

**Query Recent Universe Changes:**
```python
from hrp.data.db import DatabaseManager
import json

db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT 
            event_type,
            actor,
            timestamp,
            details
        FROM lineage 
        WHERE event_type = 'universe_update'
        ORDER BY timestamp DESC 
        LIMIT 5
    ''').fetchall()
    
    for row in result:
        details = json.loads(row[3]) if isinstance(row[3], str) else row[3]
        print(f"Date: {row[2].date()}")
        print(f"  Actor: {row[1]}")
        print(f"  Total Constituents: {details.get('total_constituents', 0)}")
        print(f"  Included: {details.get('included', 0)}")
        print(f"  Excluded: {details.get('excluded', 0)}")
        print(f"  Added: {details.get('added', 0)}")
        print(f"  Removed: {details.get('removed', 0)}")
        
        if details.get('added', 0) > 0 or details.get('removed', 0) > 0:
            print(f"  ⚠️  S&P 500 CHANGES DETECTED!")
        print()
```

**Expected Output (No Changes):**
```
Date: 2026-01-25
  Actor: agent:scheduler
  Total Constituents: 503
  Included: 396
  Excluded: 107
  Added: 0
  Removed: 0

Date: 2026-01-24
  Actor: user:manual_cli
  Total Constituents: 503
  Included: 396
  Excluded: 107
  Added: 387
  Removed: 1
```

### 1.3 Track S&P 500 Changes Over Time

**Query Universe History:**
```python
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT 
            symbol,
            name,
            sector,
            added_at,
            removed_at
        FROM universe 
        WHERE added_at >= CURRENT_DATE - INTERVAL '30 days'
           OR removed_at >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY 
            COALESCE(added_at, removed_at) DESC
    ''').fetchall()
    
    for row in result:
        if row[4]:  # removed_at is set
            print(f"❌ REMOVED: {row[0]} ({row[1]})")
            print(f"   Sector: {row[2]}")
            print(f"   Removed: {row[4].date()}")
        else:  # newly added
            print(f"✅ ADDED: {row[0]} ({row[1]})")
            print(f"   Sector: {row[2]}")
            print(f"   Added: {row[3].date()}")
        print()
```

### 1.4 Check Current Universe Statistics

**Query Universe Health:**
```python
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.connection() as conn:
    # Total active symbols
    total = conn.execute('''
        SELECT COUNT(*) 
        FROM universe 
        WHERE removed_at IS NULL
    ''').fetchone()[0]
    
    # By sector
    sectors = conn.execute('''
        SELECT sector, COUNT(*) 
        FROM universe 
        WHERE removed_at IS NULL
        GROUP BY sector
        ORDER BY COUNT(*) DESC
    ''').fetchall()
    
    print(f"Total Active Symbols: {total}")
    print(f"\nBy Sector:")
    for sector, count in sectors:
        print(f"  {sector}: {count}")
```

**Expected Output:**
```
Total Active Symbols: 396

By Sector:
  Information Technology: 75
  Health Care: 62
  Financials: 0  (excluded by design)
  Consumer Discretionary: 54
  Industrials: 71
  Communication Services: 25
  Consumer Staples: 32
  Energy: 22
  Utilities: 28
  Real Estate: 0  (excluded by design)
  Materials: 27
```

---

## 2. Log Monitoring

### 2.1 Log File Locations

```bash
# Main output log (minimal)
~/hrp-data/logs/scheduler.log

# Detailed error/info log (primary)
~/hrp-data/logs/scheduler.error.log
```

### 2.2 Success Log Patterns

**Universe Update Success:**
```
INFO - Starting job universe_update (attempt 1/4)
INFO - Updating S&P 500 universe as of YYYY-MM-DD
INFO - Fetching S&P 500 constituents from https://en.wikipedia.org/wiki/...
INFO - Fetched XXX S&P 500 constituents
INFO - Universe updated: XXX included, XXX excluded, XXX added, XXX removed
INFO - Job universe_update completed successfully
```

**Key Success Indicators:**
- Status: `completed successfully`
- Fetched: ~500-510 constituents
- Duration: < 5 seconds
- No error messages

### 2.3 Failure Log Patterns

**HTTP Error (Wikipedia):**
```
ERROR - Failed to fetch S&P 500 constituents: HTTP Error 403: Forbidden
ERROR - Job universe_update failed (attempt 1): RuntimeError
INFO - Email sent successfully: ❌ HRP Job Failed: universe_update
```

**Network Error:**
```
ERROR - Failed to fetch S&P 500 constituents: URLError
ERROR - Job universe_update failed (attempt 1): RuntimeError
INFO - Retrying job universe_update (attempt 2/4) in X seconds...
```

**Database Error:**
```
ERROR - Database error: ...
ERROR - Job universe_update failed (attempt 1): DatabaseError
```

**Key Failure Indicators:**
- Status: `failed`
- Error message present
- Email notification sent
- Retry attempts logged

### 2.4 Log Monitoring Commands

**Watch Live Logs:**
```bash
# Tail logs in real-time
tail -f ~/hrp-data/logs/scheduler.error.log

# Filter for universe updates only
tail -f ~/hrp-data/logs/scheduler.error.log | grep universe

# Check for errors
tail -100 ~/hrp-data/logs/scheduler.error.log | grep ERROR
```

**Search Historical Logs:**
```bash
# Find all universe update runs
grep "universe_update" ~/hrp-data/logs/scheduler.error.log

# Find failures
grep "Job universe_update failed" ~/hrp-data/logs/scheduler.error.log

# Find success completions
grep "universe_update completed successfully" ~/hrp-data/logs/scheduler.error.log

# Count runs by status
grep -c "universe_update completed successfully" ~/hrp-data/logs/scheduler.error.log
grep -c "universe_update failed" ~/hrp-data/logs/scheduler.error.log
```

**Check Today's Universe Update:**
```bash
# Get today's date in log format (adjust for your timezone)
TODAY=$(date +"%Y-%m-%d")

# Find today's universe update
grep "$TODAY.*universe_update" ~/hrp-data/logs/scheduler.error.log
```

---

## 3. Health Check Commands

### 3.1 Quick Health Check Script

**Create:** `~/hrp-data/scripts/check_universe_health.py`

```python
#!/usr/bin/env python
"""Quick health check for universe scheduling."""

from hrp.data.db import DatabaseManager
from datetime import date, timedelta
import sys

def check_health():
    """Check universe update health."""
    db = DatabaseManager()
    
    print("=" * 60)
    print("HRP Universe Update Health Check")
    print("=" * 60)
    print()
    
    with db.connection() as conn:
        # Check last update
        last_run = conn.execute('''
            SELECT status, started_at, records_fetched, records_inserted
            FROM ingestion_log 
            WHERE source_id = 'universe_update'
            ORDER BY started_at DESC 
            LIMIT 1
        ''').fetchone()
        
        if not last_run:
            print("❌ No universe update runs found!")
            return False
        
        status, started_at, fetched, inserted = last_run
        days_ago = (date.today() - started_at.date()).days
        
        print(f"Last Run: {started_at.date()} ({days_ago} days ago)")
        print(f"Status: {status}")
        print(f"Fetched: {fetched}, Inserted: {inserted}")
        print()
        
        # Check if stale
        if days_ago > 2:
            print(f"⚠️  WARNING: Last update was {days_ago} days ago!")
            print()
        
        # Check if failed
        if status != 'completed':
            print(f"❌ ERROR: Last update failed!")
            
            # Get error message
            error = conn.execute('''
                SELECT error_message
                FROM ingestion_log 
                WHERE source_id = 'universe_update'
                ORDER BY started_at DESC 
                LIMIT 1
            ''').fetchone()[0]
            
            print(f"Error: {error}")
            print()
            return False
        
        # Check universe size
        active_count = conn.execute('''
            SELECT COUNT(*) 
            FROM universe 
            WHERE removed_at IS NULL
        ''').fetchone()[0]
        
        print(f"Active Symbols: {active_count}")
        
        # Sanity check (S&P 500 should be ~400 after exclusions)
        if active_count < 300 or active_count > 500:
            print(f"⚠️  WARNING: Unexpected universe size: {active_count}")
            print("   Expected: 300-500 symbols")
            print()
        else:
            print("✅ Universe size looks healthy")
            print()
        
        # Check for recent changes
        recent_changes = conn.execute('''
            SELECT COUNT(*) 
            FROM universe 
            WHERE added_at >= CURRENT_DATE - INTERVAL '7 days'
               OR removed_at >= CURRENT_DATE - INTERVAL '7 days'
        ''').fetchone()[0]
        
        if recent_changes > 0:
            print(f"ℹ️  {recent_changes} symbol(s) changed in last 7 days")
            print()
        
        print("✅ Overall health: GOOD")
        return True

if __name__ == "__main__":
    success = check_health()
    sys.exit(0 if success else 1)
```

**Make executable:**
```bash
chmod +x ~/hrp-data/scripts/check_universe_health.py
```

**Run:**
```bash
cd /path/to/HRP
python ~/hrp-data/scripts/check_universe_health.py
```

### 3.2 Service Status Check

**Check if scheduler is running:**
```bash
#!/bin/bash
# Check HRP scheduler status

echo "=== HRP Scheduler Status ==="
echo

# Check launchd
if launchctl list | grep -q "com.hrp.scheduler"; then
    echo "✅ Service loaded in launchd"
    launchctl list | grep hrp
else
    echo "❌ Service NOT loaded in launchd"
    exit 1
fi

echo

# Check process
if ps aux | grep -v grep | grep -q "run_scheduler.py"; then
    echo "✅ Scheduler process running"
    ps aux | grep -v grep | grep "run_scheduler.py"
else
    echo "❌ Scheduler process NOT running"
    exit 1
fi

echo
echo "✅ Service is healthy"
```

**Save as:** `~/hrp-data/scripts/check_scheduler_status.sh`

### 3.3 CLI Health Checks

**Check scheduled jobs:**
```bash
cd /path/to/HRP
python -m hrp.agents.cli list-jobs
```

**Expected output should include:**
```
ID: universe_update
  Name: Daily Universe Update
  Trigger: cron[hour='18', minute='5']
```

**Check recent job history:**
```bash
python -m hrp.agents.cli job-status --limit 20
```

**Check specific job history:**
```bash
python -m hrp.agents.cli job-status --job-id universe_update --limit 10
```

---

## 4. Alert Triggers

### 4.1 Email Notifications

**Automatic Alerts:**
- ✅ Configured: Job failures send email via Resend API
- ✅ Recipient: Configured in environment
- ✅ Template: Professional failure notification with details

**Alert Contents:**
- Job ID and name
- Failure timestamp
- Error message
- Retry count
- Link to logs

### 4.2 Manual Alert Setup

**Optional: Cron-based health check**

Add to crontab (`crontab -e`):
```bash
# Check scheduler is running (every hour)
0 * * * * launchctl list | grep -q hrp || echo "HRP scheduler down!" | mail -s "HRP Alert" you@example.com

# Run health check daily at 7 PM (after universe update)
0 19 * * * cd /path/to/HRP && python ~/hrp-data/scripts/check_universe_health.py || echo "Universe health check failed!" | mail -s "HRP Alert" you@example.com
```

---

## 5. Troubleshooting Guide

### 5.1 Universe Update Not Running

**Symptoms:**
- No new entries in `ingestion_log`
- Logs show no universe_update activity

**Diagnosis:**
```bash
# Check scheduler is running
launchctl list | grep hrp
ps aux | grep run_scheduler

# Check job is registered
cd /path/to/HRP
python -m hrp.agents.cli list-jobs | grep universe
```

**Resolution:**
```bash
# Restart scheduler
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Verify jobs
tail -20 ~/hrp-data/logs/scheduler.error.log | grep universe
```

### 5.2 Universe Update Failing

**Symptoms:**
- Status: `failed` in `ingestion_log`
- Error messages in logs
- Email notifications received

**Common Causes & Fixes:**

**1. Wikipedia 403 Forbidden**
```
ERROR: Failed to fetch S&P 500 constituents: HTTP Error 403
```
**Fix:** User-Agent issue - should be fixed in code, but verify:
```bash
cd /path/to/HRP
grep "User-Agent" hrp/data/universe.py
# Should show: req.add_header('User-Agent', 'Mozilla/5.0...')
```

**2. Network/DNS Error**
```
ERROR: Failed to fetch S&P 500 constituents: URLError
```
**Fix:** Check internet connection, retry will happen automatically

**3. Database Error**
```
ERROR: Database error: ...
```
**Fix:** Check database accessibility:
```bash
ls -lh ~/hrp-data/hrp.duckdb
cd /path/to/HRP
python -c "from hrp.data.db import DatabaseManager; db = DatabaseManager(); print('DB OK')"
```

### 5.3 Unexpected Universe Size

**Symptoms:**
- Active symbols < 300 or > 500
- Sudden large changes

**Diagnosis:**
```python
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.connection() as conn:
    # Check recent additions/removals
    result = conn.execute('''
        SELECT 
            COUNT(*) FILTER (WHERE added_at >= CURRENT_DATE - INTERVAL '7 days') as recent_adds,
            COUNT(*) FILTER (WHERE removed_at >= CURRENT_DATE - INTERVAL '7 days') as recent_removes,
            COUNT(*) FILTER (WHERE removed_at IS NULL) as current_total
        FROM universe
    ''').fetchone()
    
    print(f"Recent additions: {result[0]}")
    print(f"Recent removals: {result[1]}")
    print(f"Current total: {result[2]}")
```

**Resolution:**
- If massive changes: Verify S&P 500 list hasn't changed dramatically
- Check exclusion rules are working correctly
- Review lineage table for details

### 5.4 Manual Re-run

**If universe update fails, run manually:**
```bash
cd /path/to/HRP
python -m hrp.agents.cli run-now --job universe
```

**Check result:**
```bash
# View output
echo $?  # 0 = success, 1 = failure

# Check database
python -c "
from hrp.data.db import DatabaseManager
db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT status, started_at 
        FROM ingestion_log 
        WHERE source_id = 'universe_update'
        ORDER BY started_at DESC 
        LIMIT 1
    ''').fetchone()
    print(f'Status: {result[0]}, Time: {result[1]}')
"
```

---

## 6. Performance Monitoring

### 6.1 Execution Time Tracking

**Query average execution time:**
```python
from hrp.data.db import DatabaseManager

db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT 
            COUNT(*) as runs,
            AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_seconds,
            MIN(EXTRACT(EPOCH FROM (completed_at - started_at))) as min_seconds,
            MAX(EXTRACT(EPOCH FROM (completed_at - started_at))) as max_seconds
        FROM ingestion_log 
        WHERE source_id = 'universe_update'
          AND status = 'completed'
          AND started_at >= CURRENT_DATE - INTERVAL '30 days'
    ''').fetchone()
    
    print(f"Runs (30 days): {result[0]}")
    print(f"Average time: {result[1]:.2f}s")
    print(f"Min time: {result[2]:.2f}s")
    print(f"Max time: {result[3]:.2f}s")
```

**Expected:** < 5 seconds average

### 6.2 Resource Usage

**Check scheduler memory/CPU:**
```bash
ps aux | grep run_scheduler | grep -v grep | awk '{print "CPU: "$3"% | Memory: "$4"% | RSS: "$6/1024"MB"}'
```

**Expected:**
- CPU: < 1% (idle, waiting for scheduled times)
- Memory: < 200 MB
- RSS: Stable over time

---

## 7. Operational Procedures

### 7.1 Daily Operations

**Morning Check (Optional):**
```bash
# Quick status check
launchctl list | grep hrp

# Check last night's runs
tail -50 ~/hrp-data/logs/scheduler.error.log | grep -E "(universe_update|completed|failed)"
```

**Evening (After 6:05 PM):**
```bash
# Check today's universe update
tail -100 ~/hrp-data/logs/scheduler.error.log | grep "$(date +%Y-%m-%d).*universe_update"

# Or run health check
cd /path/to/HRP
python ~/hrp-data/scripts/check_universe_health.py
```

### 7.2 Weekly Review

**Every Friday:**
1. Check for S&P 500 changes in last 7 days
2. Review any failures from the week
3. Verify log files aren't growing too large
4. Check backup job is running

```bash
# Check log size
du -h ~/hrp-data/logs/scheduler.error.log

# If > 100MB, consider rotation
```

### 7.3 Monthly Maintenance

**First of each month:**
1. Review lineage history for trends
2. Verify universe exclusions still appropriate
3. Check database size
4. Clean old logs if needed

```bash
# Database size
du -h ~/hrp-data/hrp.duckdb

# Backup count
ls -l ~/hrp-data/backups/ | wc -l
```

---

## 8. Dashboard Queries (Optional Future Enhancement)

These queries can be used to build monitoring dashboards:

### 8.1 Universe Update Success Rate
```sql
SELECT 
    DATE(started_at) as date,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successes,
    ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM ingestion_log 
WHERE source_id = 'universe_update'
  AND started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(started_at)
ORDER BY date DESC
```

### 8.2 S&P 500 Churn Rate
```sql
SELECT 
    DATE_TRUNC('month', COALESCE(added_at, removed_at)) as month,
    COUNT(*) FILTER (WHERE added_at IS NOT NULL) as additions,
    COUNT(*) FILTER (WHERE removed_at IS NOT NULL) as removals
FROM universe
WHERE COALESCE(added_at, removed_at) >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY month
ORDER BY month DESC
```

### 8.3 Sector Distribution Over Time
```sql
SELECT 
    DATE(added_at) as date,
    sector,
    COUNT(*) as additions
FROM universe
WHERE added_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY date, sector
ORDER BY date DESC, additions DESC
```

---

## 9. Ops Server Integration

The HRP Ops Server provides health endpoints and Prometheus metrics for comprehensive monitoring. This section covers integrating the ops server with universe scheduling monitoring.

### 9.1 Health Check Endpoints

The ops server exposes three endpoints for monitoring. See [Ops Server Guide](ops-server.md) for full details.

**Liveness Probe:**
```bash
# Check if ops server is running
curl http://localhost:8080/health

# Response: {"status": "ok", "timestamp": "2026-01-25T18:10:00.123456"}
```

**Readiness Probe:**
```bash
# Check if database and API are accessible
curl http://localhost:8080/ready

# Response (healthy):
# {
#   "status": "ready",
#   "checks": {"database": "ok", "api": "ok"}
# }
```

**Prometheus Metrics:**
```bash
# Get metrics for scraping
curl http://localhost:8080/metrics
```

### 9.2 Start the Ops Server

```bash
# Start manually
python -m hrp.ops

# Or as launchd service
launchctl load ~/Library/LaunchAgents/com.hrp.ops-server.plist

# Verify it's running
curl http://localhost:8080/health
```

### 9.3 Prometheus Queries for Universe Health

Use these PromQL queries in Prometheus or Grafana to monitor universe scheduling:

**API Availability:**
```promql
# Request rate to ready endpoint (indicates health check activity)
rate(hrp_http_requests_total{endpoint="/ready"}[5m])

# Error rate on health endpoints
sum(rate(hrp_http_requests_total{endpoint=~"/health|/ready", status=~"5.."}[5m]))
/ sum(rate(hrp_http_requests_total{endpoint=~"/health|/ready"}[5m]))
```

**Latency Monitoring:**
```promql
# 95th percentile latency for ready endpoint
histogram_quantile(0.95, rate(hrp_http_request_duration_seconds_bucket{endpoint="/ready"}[5m]))

# Average response time
rate(hrp_http_request_duration_seconds_sum{endpoint="/ready"}[5m])
/ rate(hrp_http_request_duration_seconds_count{endpoint="/ready"}[5m])
```

**Connection Monitoring:**
```promql
# Active connections to ops server
hrp_active_connections

# Alert if connections spike
hrp_active_connections > 10
```

### 9.4 Alert Thresholds

Configure alert thresholds via environment variables or YAML. See [Alert Thresholds Guide](alert-thresholds.md) for all options.

**Key thresholds for universe monitoring:**

| Threshold | Default | Environment Variable |
|-----------|---------|---------------------|
| Health score warning | 90.0 | `HRP_THRESHOLD_HEALTH_SCORE_WARNING` |
| Health score critical | 70.0 | `HRP_THRESHOLD_HEALTH_SCORE_CRITICAL` |
| Freshness warning | 3 days | `HRP_THRESHOLD_FRESHNESS_WARNING_DAYS` |
| Freshness critical | 5 days | `HRP_THRESHOLD_FRESHNESS_CRITICAL_DAYS` |
| Ingestion success warning | 95% | `HRP_THRESHOLD_INGESTION_SUCCESS_RATE_WARNING` |
| Ingestion success critical | 80% | `HRP_THRESHOLD_INGESTION_SUCCESS_RATE_CRITICAL` |

**Example: Set stricter universe freshness thresholds:**
```bash
export HRP_THRESHOLD_FRESHNESS_WARNING_DAYS=1
export HRP_THRESHOLD_FRESHNESS_CRITICAL_DAYS=2
```

### 9.5 Combined Monitoring Workflow

This workflow combines ops server health checks with database monitoring for comprehensive universe scheduling oversight.

**Step 1: Check ops server health**
```bash
#!/bin/bash
# check_full_health.sh

HOST=${HRP_OPS_HOST:-localhost}
PORT=${HRP_OPS_PORT:-8080}

echo "=== HRP Universe Monitoring Workflow ==="
echo

# Step 1: Check ops server
echo "1. Ops Server Health:"
HEALTH=$(curl -s -w "%{http_code}" -o /tmp/health.json http://${HOST}:${PORT}/health)
if [ "$HEALTH" = "200" ]; then
    echo "   [OK] Ops server running"
else
    echo "   [FAIL] Ops server not responding (HTTP $HEALTH)"
    exit 1
fi

# Step 2: Check readiness (database + API)
echo "2. System Readiness:"
READY=$(curl -s -w "%{http_code}" -o /tmp/ready.json http://${HOST}:${PORT}/ready)
if [ "$READY" = "200" ]; then
    echo "   [OK] Database and API ready"
else
    echo "   [FAIL] System not ready (HTTP $READY)"
    cat /tmp/ready.json | jq .
    exit 1
fi
```

**Step 2: Check universe update status**
```bash
# Continue from above script...

# Step 3: Check last universe update
echo "3. Last Universe Update:"
cd /path/to/HRP
python -c "
from hrp.data.db import DatabaseManager
db = DatabaseManager()
with db.connection() as conn:
    result = conn.execute('''
        SELECT status, started_at, records_inserted
        FROM ingestion_log
        WHERE source_id = 'universe_update'
        ORDER BY started_at DESC
        LIMIT 1
    ''').fetchone()
    if result:
        from datetime import date
        days_ago = (date.today() - result[1].date()).days
        status = 'OK' if result[0] == 'completed' and days_ago <= 2 else 'WARNING'
        print(f'   [{status}] Status: {result[0]}, {days_ago} days ago, {result[2]} records')
    else:
        print('   [WARNING] No universe updates found')
"
```

**Step 3: Check metrics**
```bash
# Continue from above script...

# Step 4: Check Prometheus metrics
echo "4. Request Metrics:"
curl -s http://${HOST}:${PORT}/metrics | grep -E "^hrp_http_requests_total" | head -5

echo
echo "=== Monitoring Complete ==="
```

**Save as:** `~/hrp-data/scripts/check_full_health.sh`

### 9.6 Grafana Dashboard Setup

Create a Grafana dashboard for universe monitoring:

**Panel 1: API Health**
```promql
# Success rate over time
1 - (sum(rate(hrp_http_requests_total{status=~"5.."}[5m])) / sum(rate(hrp_http_requests_total[5m])))
```

**Panel 2: Latency Heatmap**
```promql
# Response time distribution
rate(hrp_http_request_duration_seconds_bucket[5m])
```

**Panel 3: Active Connections**
```promql
# Current connections
hrp_active_connections
```

**Alert Rule Example:**
```yaml
# Grafana alert rule for ops server health
- alert: HRPOpsServerDown
  expr: up{job="hrp-ops"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "HRP Ops Server is down"
    description: "The HRP ops server has been unreachable for more than 1 minute."

- alert: HRPHighErrorRate
  expr: sum(rate(hrp_http_requests_total{status=~"5.."}[5m])) / sum(rate(hrp_http_requests_total[5m])) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "HRP high error rate detected"
    description: "Error rate is above 5% for the last 5 minutes."
```

### 9.7 Related Documentation

- [Ops Server Guide](ops-server.md) - Full ops server documentation
- [Alert Thresholds Guide](alert-thresholds.md) - Configurable threshold settings
- [Scheduler Configuration Guide](../setup/Scheduler-Configuration-Guide.md) - launchd job setup

---

## 10. Quick Reference

### Essential Commands
```bash
# Service status
launchctl list | grep hrp

# View live logs
tail -f ~/hrp-data/logs/scheduler.error.log

# List jobs
python -m hrp.agents.cli list-jobs

# Manual run
python -m hrp.agents.cli run-now --job universe

# Health check
python ~/hrp-data/scripts/check_universe_health.py

# Ops server health (requires ops server running)
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

### Key Files
```
Service: ~/Library/LaunchAgents/com.hrp.scheduler.plist
Ops Server: ~/Library/LaunchAgents/com.hrp.ops-server.plist
Logs: ~/hrp-data/logs/scheduler.error.log
Ops Logs: ~/hrp-data/logs/ops-server.log
Database: ~/hrp-data/hrp.duckdb
Scripts: ~/hrp-data/scripts/
Thresholds: ~/hrp-data/config/thresholds.yaml
```

### Database Tables
```
ingestion_log - Job execution history
lineage - Universe change events
universe - Current/historical S&P 500 symbols
```

---

**Monitoring Setup Complete**  
**Ready for:** Phase 4 (Production Validation)  
**First Run:** January 25, 2026 @ 18:05 ET
