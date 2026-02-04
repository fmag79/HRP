# HRP Deployment Guide

Guide for deploying HRP scheduler and ops server as production background services on macOS.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Variables](#environment-variables)
3. [Startup Validation](#startup-validation)
4. [Initial Setup](#initial-setup)
5. [Background Service Setup (launchd)](#background-service-setup-launchd)
6. [Ops Server Deployment](#ops-server-deployment)
7. [Service Management](#service-management)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Alternative Deployment Methods](#alternative-deployment-methods)

---

## Prerequisites

- macOS with Python 3.11+
- HRP installed with all dependencies
- Virtual environment activated
- Database initialized at `~/hrp-data/hrp.duckdb`

```bash
# Verify installation
python -c "from hrp.agents.scheduler import IngestionScheduler; print('✓ HRP installed')"

# Verify database exists
ls -lh ~/hrp-data/hrp.duckdb
```

---

## Environment Variables

### Core Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HRP_DB_PATH` | Database path | `~/hrp-data/hrp.duckdb` | No |
| `HRP_DATA_DIR` | Data directory | `~/hrp-data/` | No |
| `RESEND_API_KEY` | Resend API key for email notifications | - | For alerts |
| `NOTIFICATION_EMAIL` | Email address for notifications | - | For alerts |
| `NOTIFICATION_FROM_EMAIL` | From address | `onboarding@resend.dev` | No |
| `SIMFIN_API_KEY` | SimFin API key for fundamentals | - | For fundamentals |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude agents | - | For production |

### Tier 3: Operations & Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HRP_ENVIRONMENT` | Environment mode: `development`, `staging`, `production` | `development` | No |
| `HRP_AUTH_ENABLED` | Enable dashboard authentication | `true` | No |
| `HRP_AUTH_COOKIE_KEY` | Secret key for auth cookies (32+ chars) | - | For auth |
| `HRP_AUTH_USERS_FILE` | Path to users YAML file | `~/hrp-data/auth/users.yaml` | No |
| `HRP_OPS_HOST` | Ops server bind host | `0.0.0.0` | No |
| `HRP_OPS_PORT` | Ops server bind port | `8080` | No |
| `HRP_THRESHOLD_*` | Override alert thresholds (see [Alert Thresholds](alert-thresholds.md)) | - | No |

**Environment-specific requirements:**
- **Development:** No required secrets
- **Staging:** `ANTHROPIC_API_KEY` recommended
- **Production:** `ANTHROPIC_API_KEY` required (startup validation fails without it)

---

## Startup Validation

HRP provides fail-fast startup validation to ensure required configuration is present before running services.

### Using fail_fast_startup()

Call `fail_fast_startup()` at application entry points (API, dashboard, CLI) to validate configuration:

```python
from hrp.utils.startup import fail_fast_startup

# Strict mode - raises RuntimeError on failure
fail_fast_startup()
```

**What it validates:**
- Required secrets for the current environment (e.g., `ANTHROPIC_API_KEY` in production)
- Database connectivity (via secrets validation)
- Essential configuration

**Example error:**
```
RuntimeError: Startup validation failed:
  - Missing required secret: ANTHROPIC_API_KEY
```

### Non-strict Validation

For scripts that want to handle validation errors manually:

```python
from hrp.utils.startup import validate_startup

errors = validate_startup()
if errors:
    print("Warnings:", errors)
    # Handle gracefully or continue with degraded functionality
```

### Environment-specific Validation

| Environment | `HRP_ENVIRONMENT` | Required Secrets |
|-------------|-------------------|------------------|
| Development | `development` (default) | None |
| Staging | `staging` | `ANTHROPIC_API_KEY` (warning only) |
| Production | `production` | `ANTHROPIC_API_KEY` (fails startup) |

**Important:** Always set `HRP_ENVIRONMENT=production` in production deployments to ensure all required secrets are validated.

---

## Initial Setup

### 1. Create Required Directories

```bash
mkdir -p ~/hrp-data/logs
mkdir -p ~/hrp-data/backups
mkdir -p ~/Library/LaunchAgents
```

### 2. Test Scheduler Manually

Before setting up as a service, test the scheduler works:

```bash
cd /path/to/HRP
python run_scheduler.py
# Press Ctrl+C after a few seconds to stop
```

You should see:
```
INFO - Ingestion scheduler initialized
INFO - Setting up daily data ingestion pipeline...
INFO - Scheduled price ingestion at 18:00 ET
INFO - Scheduled feature computation at 18:10 ET
INFO - Scheduled daily backup at 02:00 ET
INFO - Scheduler is running with 3 jobs
```

---

## Background Service Setup (launchd)

### Step 1: Create Service Configuration

Create the file `~/Library/LaunchAgents/com.hrp.scheduler.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.scheduler</string>
    
    <!-- Path to Python in your virtual environment -->
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/path/to/HRP/.venv/bin/python</string>
        <string>/Users/YOUR_USERNAME/path/to/HRP/run_scheduler.py</string>
    </array>
    
    <!-- Working directory -->
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/path/to/HRP</string>
    
    <!-- Start automatically on login -->
    <key>RunAtLoad</key>
    <true/>
    
    <!-- Restart if crashed -->
    <key>KeepAlive</key>
    <true/>
    
    <!-- Log files -->
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/hrp-data/logs/scheduler.log</string>
    
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/hrp-data/logs/scheduler.error.log</string>
    
    <!-- Environment variables -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HRP_DB_PATH</key>
        <string>/Users/YOUR_USERNAME/hrp-data/hrp.duckdb</string>
        <!-- Optional: Add API keys if needed -->
        <!-- <key>RESEND_API_KEY</key> -->
        <!-- <string>your_api_key_here</string> -->
    </dict>
    
    <!-- Run in background -->
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
```

**Important:** Replace `YOUR_USERNAME` and paths with your actual values.

**Finding your Python path:**
```bash
cd /path/to/HRP
source .venv/bin/activate
which python
# Use this path in ProgramArguments
```

### Step 2: Set Permissions

```bash
chmod 644 ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Step 3: Load the Service

```bash
# Load and start the service
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Verify it's running
launchctl list | grep hrp
# Output: 12345	0	com.hrp.scheduler
```

### Step 4: Verify Operation

```bash
# Check logs (wait a few seconds for startup)
tail -20 ~/hrp-data/logs/scheduler.error.log

# Should show:
# INFO - Scheduler is running with 3 jobs:
# INFO -   - daily_backup: next run at ...
# INFO -   - price_ingestion: next run at ...
# INFO -   - feature_computation: next run at ...
```

---

## Ops Server Deployment

The HRP Ops Server provides health endpoints (`/health`, `/ready`) and Prometheus metrics (`/metrics`) for production monitoring. For detailed configuration, see [Ops Server Guide](ops-server.md).

### Quick Setup

The ops server has its own launchd plist: `launchd/com.hrp.ops-server.plist`

```bash
# Install all HRP launchd jobs (including ops server)
./scripts/manage_launchd.sh install

# Check status of all HRP jobs
./scripts/manage_launchd.sh status

# View ops server logs
tail -f ~/hrp-data/logs/ops-server.error.log
```

### Manual Installation

If you prefer to manage the ops server separately:

```bash
# Copy plist to LaunchAgents
cp launchd/com.hrp.ops-server.plist ~/Library/LaunchAgents/

# Load (start) the service
launchctl load ~/Library/LaunchAgents/com.hrp.ops-server.plist

# Verify it's running
launchctl list | grep hrp.ops
curl http://localhost:8080/health
```

### Service Configuration

The `com.hrp.ops-server.plist` configures:
- **RunAtLoad:** Starts automatically on login
- **KeepAlive:** Restarts if the process exits
- **Host:** `127.0.0.1` (localhost only for security)
- **Port:** `8080`
- **Environment:** `HRP_ENVIRONMENT=production`

**Log Locations:**
- stdout: `~/hrp-data/logs/ops-server.log`
- stderr: `~/hrp-data/logs/ops-server.error.log`

### Verify Ops Server Health

```bash
# Liveness probe
curl http://localhost:8080/health

# Readiness probe (checks database connectivity)
curl http://localhost:8080/ready

# Prometheus metrics
curl http://localhost:8080/metrics | grep hrp_
```

For advanced configuration including Prometheus scraping setup and alert thresholds, see:
- [Ops Server Guide](ops-server.md)
- [Alert Thresholds Guide](alert-thresholds.md)

---

## Service Management

### Check Status

```bash
# Check if service is loaded
launchctl list | grep hrp

# View process details
ps aux | grep run_scheduler

# Check logs
tail -f ~/hrp-data/logs/scheduler.error.log
```

### Stop Service

```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Start Service

```bash
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Restart Service

```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Remove Service Permanently

```bash
# Unload first
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Delete plist file
rm ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Update Service Configuration

After editing the plist file:

```bash
# Unload old config
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Load new config
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

---

## Monitoring

### View Live Logs

```bash
# All logs (INFO, ERROR, DEBUG)
tail -f ~/hrp-data/logs/scheduler.error.log

# Last 50 lines
tail -50 ~/hrp-data/logs/scheduler.error.log

# Search for errors
grep ERROR ~/hrp-data/logs/scheduler.error.log
```

### Check Job History

```bash
# View recent job executions
python -m hrp.agents.cli job-status

# View specific job
python -m hrp.agents.cli job-status --job-id price_ingestion --limit 20

# View all jobs
python -m hrp.agents.cli job-status --limit 100
```

### Monitor Scheduled Jobs

```bash
# List upcoming jobs
python -m hrp.agents.cli list-jobs
```

**Expected output:**
```
Scheduled Jobs:
--------------------------------------------------------------------------------
ID: daily_backup
  Name: Daily Backup
  Next Run: 2026-01-25 02:00:00-05:00
  Trigger: cron[hour='2', minute='0']

ID: price_ingestion
  Name: Daily Price Ingestion
  Next Run: 2026-01-25 18:00:00-05:00
  Trigger: cron[hour='18', minute='0']

ID: feature_computation
  Name: Daily Feature Computation
  Next Run: 2026-01-25 18:10:00-05:00
  Trigger: cron[hour='18', minute='10']
```

### Check System Resources

```bash
# CPU and memory usage
ps aux | grep run_scheduler

# Disk usage
du -sh ~/hrp-data/*
```

---

## Troubleshooting

### Service Not Starting

**Check if plist is loaded:**
```bash
launchctl list | grep hrp
```

**Check for syntax errors in plist:**
```bash
plutil -lint ~/Library/LaunchAgents/com.hrp.scheduler.plist
# Should output: OK
```

**Check permissions:**
```bash
ls -l ~/Library/LaunchAgents/com.hrp.scheduler.plist
# Should be: -rw-r--r--
```

**View system log:**
```bash
log show --predicate 'subsystem == "com.apple.xpc.launchd"' --last 1h | grep hrp
```

### Service Crashes Immediately

**Check logs:**
```bash
cat ~/hrp-data/logs/scheduler.error.log
```

**Common issues:**

1. **Wrong Python path** - Verify virtual environment path
2. **Missing dependencies** - Activate venv and run `pip install -r requirements.txt`
3. **Database not found** - Check `HRP_DB_PATH` in plist
4. **Permission issues** - Ensure log directory exists and is writable

**Manual test:**
```bash
# Run the exact command from plist
/path/to/.venv/bin/python /path/to/HRP/run_scheduler.py
# Check for errors
```

### Jobs Not Running

**Check scheduler is running:**
```bash
launchctl list | grep hrp
tail -f ~/hrp-data/logs/scheduler.error.log
```

**Verify job times:**
```bash
python -m hrp.agents.cli list-jobs
```

**Run job manually to test:**
```bash
python -m hrp.agents.cli run-now --job prices --symbols AAPL
```

### High Resource Usage

**Check database size:**
```bash
du -h ~/hrp-data/hrp.duckdb
```

**Check log file sizes:**
```bash
du -h ~/hrp-data/logs/*
```

**Rotate large log files:**
```bash
# Backup and truncate
mv ~/hrp-data/logs/scheduler.error.log ~/hrp-data/logs/scheduler.error.log.old
touch ~/hrp-data/logs/scheduler.error.log

# Restart service
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

---

## Alternative Deployment Methods

### Using nohup (Simple Background Process)

```bash
cd /path/to/HRP
nohup python run_scheduler.py > ~/hrp-data/logs/scheduler.log 2>&1 &

# Save PID
echo $! > ~/hrp-data/scheduler.pid

# Kill later
kill $(cat ~/hrp-data/scheduler.pid)
```

**Pros:** Simple, works everywhere  
**Cons:** Doesn't survive logout, no auto-restart

### Using screen (Detachable Terminal)

```bash
# Start session
screen -S hrp-scheduler

# Run scheduler
cd /path/to/HRP
python run_scheduler.py

# Detach: Ctrl+A then D

# Reattach later
screen -r hrp-scheduler

# List sessions
screen -ls

# Kill session
screen -X -S hrp-scheduler quit
```

**Pros:** Can reattach, view logs interactively  
**Cons:** Doesn't survive reboot, manual management

### Using tmux (Terminal Multiplexer)

```bash
# Start session
tmux new -s hrp-scheduler

# Run scheduler
cd /path/to/HRP
python run_scheduler.py

# Detach: Ctrl+B then D

# Reattach later
tmux attach -t hrp-scheduler

# List sessions
tmux ls

# Kill session
tmux kill-session -t hrp-scheduler
```

**Pros:** Better than screen, can split panes  
**Cons:** Doesn't survive reboot, manual management

### Using Docker (Future)

For containerized deployment (not yet implemented):

```dockerfile
# Future: Dockerfile for HRP
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "run_scheduler.py"]
```

---

## Best Practices

### 1. Log Rotation

Set up log rotation to prevent disk space issues:

```bash
# Create logrotate config (if using Linux)
# macOS: use newsyslog or manual rotation
```

### 2. Monitoring Alerts

Set up email notifications for job failures (already included in HRP):

```bash
# In .env file
RESEND_API_KEY=your_key_here
NOTIFICATION_EMAIL=you@example.com
```

### 3. Regular Backups

The scheduler automatically creates daily backups at 2 AM. Verify:

```bash
# List recent backups
python -m hrp.data.backup --list

# Verify a backup
python -m hrp.data.backup --verify ~/hrp-data/backups/backup_YYYYMMDD_HHMMSS
```

### 4. Health Checks

Create a cron job to verify the scheduler is running:

```bash
# Add to your crontab (crontab -e)
0 * * * * launchctl list | grep hrp || echo "HRP scheduler not running!" | mail -s "HRP Alert" you@example.com
```

---

## Production Checklist

Before deploying to production:

### Environment Setup
- [ ] Database initialized and populated
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Log directory created with proper permissions (`mkdir -p ~/hrp-data/logs`)
- [ ] Backup directory created (`mkdir -p ~/hrp-data/backups`)
- [ ] Auth directory created (`mkdir -p ~/hrp-data/auth`)

### Environment Variables
- [ ] `HRP_ENVIRONMENT=production` set
- [ ] `ANTHROPIC_API_KEY` configured (required for production)
- [ ] `RESEND_API_KEY` configured (if using email alerts)
- [ ] `NOTIFICATION_EMAIL` configured (if using email alerts)
- [ ] `HRP_AUTH_COOKIE_KEY` configured (32+ chars, if using dashboard auth)

### Startup Validation
- [ ] Run `fail_fast_startup()` passes without errors:
  ```bash
  python -c "from hrp.utils.startup import fail_fast_startup; fail_fast_startup(); print('OK')"
  ```

### Scheduler Service
- [ ] Scheduler tested manually (`python run_scheduler.py`)
- [ ] launchd plist created with correct paths
- [ ] Scheduler service loaded and running
- [ ] Scheduler logs show successful startup
- [ ] First job execution verified

### Ops Server
- [ ] Ops server launchd plist installed (`com.hrp.ops-server.plist`)
- [ ] Ops server running and healthy (`curl http://localhost:8080/health`)
- [ ] Readiness probe passing (`curl http://localhost:8080/ready`)
- [ ] Prometheus metrics exposed (`curl http://localhost:8080/metrics`)

### Monitoring & Alerts
- [ ] Email notifications tested (if configured)
- [ ] Backup job tested
- [ ] Alert thresholds configured (optional, see [Alert Thresholds](alert-thresholds.md))
- [ ] Prometheus scraping configured (optional, see [Ops Server](ops-server.md))

---

## Support

For issues or questions:
1. Check scheduler logs: `~/hrp-data/logs/scheduler.error.log`
2. Check ops server logs: `~/hrp-data/logs/ops-server.error.log`
3. Test scheduler manually: `python run_scheduler.py`
4. Test ops server manually: `python -m hrp.ops`
5. Check job history: `python -m hrp.agents.cli job-status`
6. Verify startup validation: `python -c "from hrp.utils.startup import fail_fast_startup; fail_fast_startup()"`

## Related Documentation

- [Ops Server Guide](ops-server.md) - Health endpoints and Prometheus metrics
- [Alert Thresholds Guide](alert-thresholds.md) - Configurable monitoring thresholds
- [Cookbook](cookbook.md) - Quick reference guide
- [Monitoring Setup](monitoring-universe-scheduling.md) - Database monitoring queries
