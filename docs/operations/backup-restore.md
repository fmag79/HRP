# Backup & Restore Operations Guide

## Overview

HRP uses automated backups to protect the DuckDB database and MLflow experiment artifacts. This guide covers:

- Backup schedule and locations
- Manual backup procedures
- Restore procedures
- Disaster recovery scenarios
- Troubleshooting

---

## Backup Architecture

### What Gets Backed Up

| Component | Location | Description |
|-----------|----------|-------------|
| DuckDB Database | `~/hrp-data/hrp.duckdb` | All prices, features, hypotheses, lineage |
| MLflow Artifacts | `~/hrp-data/mlflow/` | Experiment logs, model artifacts, metrics |

### Backup Location

Backups are stored in `~/hrp-data/backups/` with timestamped directories:

```
~/hrp-data/backups/
├── backup_20260124_020000/
│   ├── hrp.duckdb
│   ├── mlruns/
│   ├── checksums.txt
│   └── metadata.json
├── backup_20260123_020000/
│   └── ...
└── backup_20260122_020000/
    └── ...
```

### Backup Contents

Each backup directory contains:

- `hrp.duckdb` - Full database copy
- `mlruns/` - MLflow directory (if included)
- `checksums.txt` - SHA-256 checksums for verification
- `metadata.json` - Backup metadata (timestamp, size, version)

---

## Automated Backups

### Schedule

Weekly backups run automatically on **Saturday at 2:00 AM ET** (configurable).

### Enable Automated Backups

#### Option A: Background Service (Production - macOS)

The scheduler automatically includes weekly backups when run as a launchd service (see cookbook section 7.2 for full setup).

```bash
# Service includes backup by default
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Check it's running
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log
```

#### Option B: Foreground/Testing

**Using the scheduler runner (recommended):**

```bash
# Run scheduler with backup enabled (default)
python run_scheduler.py

# Customize backup time and retention
python run_scheduler.py --backup-time 02:00 --backup-keep-days 30
```

**Programmatically:**

```python
from hrp.agents.scheduler import IngestionScheduler
import signal

scheduler = IngestionScheduler()
scheduler.setup_weekly_backup(
    backup_time="02:00",  # 2 AM
    day_of_week="sat",    # Saturday
    keep_days=30          # Keep 30 days of backups
)
scheduler.start()

# Keep running
signal.pause()
```

### Verify Backup Schedule

```bash
python -m hrp.agents.cli list-jobs
```

---

## Manual Backup Procedures

### Create a Backup

```bash
# Using CLI
python -m hrp.data.backup --backup

# Output:
# Backup created successfully:
#   Path: /Users/you/hrp-data/backups/backup_20260124_143052
#   Size: 245.3 MB
#   Checksum: verified
```

### Create Backup (Programmatic)

```python
from hrp.data.backup import create_backup

result = create_backup(
    backup_dir=None,           # Uses default: ~/hrp-data/backups/
    include_mlflow=True,       # Include MLflow artifacts
)

print(f"Backup created at: {result['path']}")
print(f"Size: {result['size_mb']} MB")
print(f"Checksum file: {result['checksum_file']}")
```

### List Available Backups

```bash
python -m hrp.data.backup --list

# Output:
# Available backups:
#   1. backup_20260124_020000 (245.3 MB) - 2026-01-24 02:00:00
#   2. backup_20260123_020000 (244.8 MB) - 2026-01-23 02:00:00
#   3. backup_20260122_020000 (243.1 MB) - 2026-01-22 02:00:00
```

---

## Verify Backup Integrity

### Why Verify?

Backups can become corrupted due to:
- Disk errors
- Incomplete writes
- File system issues

Always verify before restoring.

### Verify a Backup

```bash
python -m hrp.data.backup --verify /path/to/backup_YYYYMMDD_HHMMSS

# Output (success):
# ✓ Backup verified successfully
#   All 2 files match checksums

# Output (failure):
# ✗ Backup verification failed
#   Checksum mismatch: hrp.duckdb
```

### Verify Programmatically

```python
from hrp.data.backup import verify_backup
from pathlib import Path

is_valid = verify_backup(Path("/path/to/backup_20260124_020000"))
if is_valid:
    print("Backup is valid")
else:
    print("Backup is corrupted!")
```

---

## Restore Procedures

### Pre-Restore Checklist

1. **Check ops server health** - Document current system state before changes
2. **Stop all services** - Dashboard, scheduler, any running jobs
3. **Verify the backup** - Ensure backup integrity before restoring
4. **Document current state** - Note what data will be lost (if any)
5. **Confirm restore target** - Ensure you're restoring to the correct location

#### Verify Ops Server Health (Pre-Restore)

Before starting a restore, check the ops server to document the current system state:

```bash
# Check liveness (server running)
curl http://localhost:8080/health

# Check readiness (database connectivity)
curl http://localhost:8080/ready
```

If the readiness check already fails (e.g., due to database corruption), proceed with the restore. Document the error for your records.

See [Ops Server Guide](ops-server.md) for detailed health endpoint documentation.

### Restore from Backup

```bash
# Stop services first
pkill -f "streamlit run"
pkill -f "hrp.agents"

# Verify backup before restoring
python -m hrp.data.backup --verify /path/to/backup_20260124_020000

# Restore
python -m hrp.data.backup --restore /path/to/backup_20260124_020000

# Output:
# Restoring from backup...
# ✓ Restored hrp.duckdb
# ✓ Restored mlruns/
# Restore complete!
```

### Restore to Different Location

```bash
python -m hrp.data.backup --restore /path/to/backup_20260124_020000 \
    --target-dir /path/to/new/location
```

### Restore Programmatically

```python
from hrp.data.backup import restore_backup, verify_backup
from pathlib import Path

backup_path = Path("/path/to/backup_20260124_020000")

# Always verify first
if not verify_backup(backup_path):
    raise RuntimeError("Backup verification failed!")

# Restore
success = restore_backup(
    backup_path=backup_path,
    target_dir=None,  # Uses default HRP data directory
)

if success:
    print("Restore completed successfully")
```

### Post-Restore Validation

After restoring, verify the system works correctly before restarting services.

#### Step 1: Verify Ops Server Health

Check both liveness and readiness endpoints to confirm the restore was successful:

```bash
# Start the ops server (if not running as a service)
python -m hrp.ops &

# Wait for startup
sleep 2

# Check liveness
curl http://localhost:8080/health
# Expected: {"status": "ok", ...}

# Check readiness (verifies database connectivity)
curl http://localhost:8080/ready
# Expected: {"status": "ready", "checks": {"database": "ok", "api": "ok"}, ...}
```

If the readiness check fails, the database may not have been restored correctly. Check the error message and verify the database file exists at `~/hrp-data/hrp.duckdb`.

#### Step 2: Verify API and Data Integrity

```python
from hrp.api.platform import PlatformAPI
from datetime import date

api = PlatformAPI()
health = api.health_check()

print(f"Status: {health['status']}")
print(f"Database: {health['database']}")
print(f"Tables: {health['tables']}")

# Verify data integrity
prices_count = len(api.get_prices(['AAPL'], start=date(2020,1,1), end=date.today()))
print(f"Price records: {prices_count}")
```

#### Step 3: Restart Services

Once validation passes, restart your services:

```bash
# Restart dashboard
streamlit run hrp/dashboard/app.py &

# Restart scheduler (if using)
python -m hrp.agents.run_scheduler &
```

See [Ops Server Guide](ops-server.md) for detailed health endpoint documentation and troubleshooting.

---

## Backup Rotation

### Automatic Rotation

Backups older than `keep_days` are automatically deleted during scheduled backups.

### Manual Rotation

```bash
# Delete backups older than 30 days
python -m hrp.data.backup --rotate --keep-days 30

# Output:
# Rotating backups (keeping last 30 days)...
# Deleted 5 old backups
# Remaining: 30 backups
```

### Rotation Programmatically

```python
from hrp.data.backup import rotate_backups
from pathlib import Path

deleted_count = rotate_backups(
    backup_dir=Path("~/hrp-data/backups").expanduser(),
    keep_days=30
)
print(f"Deleted {deleted_count} old backups")
```

---

## Disaster Recovery Scenarios

### Scenario 1: Corrupted Database

**Symptoms:**
- Database errors on queries
- "Database is corrupted" messages
- Unexpected query results

**Recovery Steps:**

1. Stop all services
2. List available backups: `python -m hrp.data.backup --list`
3. Choose most recent backup before corruption
4. Verify backup: `python -m hrp.data.backup --verify /path/to/backup`
5. Restore: `python -m hrp.data.backup --restore /path/to/backup`
6. Validate: Run `api.health_check()`
7. Restart services

**Data Loss:** Any data added since the backup

### Scenario 2: Accidental Data Deletion

**Symptoms:**
- Missing hypotheses, experiments, or price data
- "Not found" errors for known entities

**Recovery Steps:**

Same as Scenario 1. Choose backup from before deletion occurred.

### Scenario 3: System Migration

**Moving to new machine:**

1. On old machine: `python -m hrp.data.backup --backup`
2. Copy backup to new machine (scp, rsync, etc.)
3. On new machine: Install HRP
4. Restore: `python -m hrp.data.backup --restore /path/to/backup`

### Scenario 4: Backup Verification Failure

**If backup verification fails:**

1. Try the next most recent backup
2. If all recent backups fail, check for disk errors
3. Contact support with error messages

---

## Troubleshooting

### Common Issues

#### "Backup directory not found"

```bash
# Create the backup directory
mkdir -p ~/hrp-data/backups
```

#### "Permission denied"

Ensure you have write permissions:
```bash
chmod 755 ~/hrp-data/backups
```

#### "Database is locked"

Stop all processes accessing the database:
```bash
pkill -f "streamlit run"
pkill -f "hrp.agents"
```

#### "Checksum mismatch"

The backup is corrupted. Try a different backup.

#### "Not enough disk space"

Check available space and delete old backups:
```bash
df -h ~/hrp-data
python -m hrp.data.backup --rotate --keep-days 7
```

### Getting Help

If you encounter issues not covered here:

1. Check the logs: `tail -f ~/hrp-data/logs/backup.log`
2. Open an issue with error messages and steps to reproduce

---

## Best Practices

1. **Verify backups regularly** - Don't assume backups are good
2. **Test restore procedures** - Practice restoring to a test location
3. **Monitor backup size** - Growing rapidly may indicate issues
4. **Keep multiple retention periods** - Daily (7 days), weekly (4 weeks), monthly (12 months)
5. **Store backups off-site** - Copy critical backups to external storage

---

## Quick Reference

| Action | Command |
|--------|---------|
| Create backup | `python -m hrp.data.backup --backup` |
| List backups | `python -m hrp.data.backup --list` |
| Verify backup | `python -m hrp.data.backup --verify /path/to/backup` |
| Restore backup | `python -m hrp.data.backup --restore /path/to/backup` |
| Rotate old backups | `python -m hrp.data.backup --rotate --keep-days 30` |
