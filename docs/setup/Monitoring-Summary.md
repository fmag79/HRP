# Automated Data Quality Monitoring System - Implementation Summary

## Overview

I've successfully created a comprehensive automated monitoring system for data quality issues in the HRP platform. The system provides:

1. **Daily Automated Quality Checks** - Runs at 6 AM ET (configurable)
2. **Health Score Tracking** - Single metric (0-100) with trend analysis
3. **Threshold-Based Alerting** - Email alerts for critical issues
4. **Data Freshness Monitoring** - Alerts when data is stale
5. **Anomaly Detection** - Automatic detection of data issues
6. **Dashboard Integration** - Visual monitoring via Streamlit

## What Was Implemented

### 1. Core Monitoring Module (`hrp/monitoring/quality_monitor.py`)

**New Classes:**
- `DataQualityMonitor` - Main monitoring class
- `MonitoringThresholds` - Configurable alert thresholds
- `MonitoringResult` - Result object with all check data
- `run_quality_monitor_with_alerts()` - Convenience function

**Features:**
- Daily quality check execution
- Health score trend calculation (improving/stable/declining)
- Actionable recommendations generation
- Multi-level alerting (warning/critical)
- Integration with existing QualityReport system

### 2. Scheduler Integration (`hrp/agents/scheduler.py`)

**New Method:**
- `setup_quality_monitoring()` - Configure daily quality checks

**Updated CLI (`hrp/agents/run_scheduler.py`):**
- `--with-quality-monitoring` - Enable quality monitoring
- `--quality-monitor-time` - Set check time (default: 06:00)
- `--health-threshold` - Set warning threshold (default: 90.0)

### 3. Documentation

**Created:**
- `docs/setup/Automated-Monitoring-Setup.md` - Complete setup guide
- `scripts/start_monitoring.sh` - Quick start script

## Current Monitoring Status

### Quality Checks (5 Active)

| Check | Status | Description |
|-------|--------|-------------|
| **Price Anomaly** | ✅ PASS | Detects >50% moves, invalid prices |
| **Completeness** | ⚠️ 387 issues | Missing prices for active symbols |
| **Gap Detection** | ✅ PASS | Missing trading days in price history |
| **Stale Data** | ❌ 1 critical | Symbols not updated recently |
| **Volume Anomaly** | ✅ PASS | Negative/zero volume detection |

### Current Health Score

```
Health Score: 50/100
Status: CRITICAL
Trend: Stable
```

**Issues Detected:**
- Critical: 1 (stale data)
- Warnings: 387 (completeness)
- Total: 388

### Recommendations Generated

1. URGENT: Health score critically low. Review all critical issues immediately.
2. Fix 1 critical stale_data issues
3. High warning count (387). Schedule maintenance to reduce noise.
4. 1 symbols have stale data. Check ingestion pipeline.

## Alert Thresholds

### Default Thresholds

```python
MonitoringThresholds(
    health_score_warning=90.0,      # Alert if score < 90
    health_score_critical=70.0,     # Alert if score < 70
    freshness_warning_days=3,       # Warning if data > 3 days old
    freshness_critical_days=5,      # Critical if data > 5 days old
    anomaly_count_critical=100,     # Alert if > 100 anomalies
)
```

### Alert Types

1. **Health Score Warning** - Score < 90
   - Sends daily summary email
   - Includes recommendations

2. **Health Score Critical** - Score < 70
   - Sends immediate critical alert
   - Lists all critical issues

3. **Critical Issues Alert** - Any critical issues detected
   - Immediate email with issue details
   - Sent regardless of health score

4. **Data Freshness Alert** - Data too old
   - Warning: > 3 days stale
   - Critical: > 5 days stale

5. **Anomaly Spike Alert** - Unusually high anomaly count
   - Triggered when > 100 anomalies
   - Indicates systemic data issues

## Usage Examples

### 1. Quick Start - Test Monitoring

```bash
# Run the quick start script
bash scripts/start_monitoring.sh
```

### 2. Manual Quality Check

```python
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
from datetime import date

result = run_quality_monitor_with_alerts(
    as_of_date=date.today(),
    send_alerts=True,
)

print(f"Health Score: {result.health_score}/100")
print(f"Trend: {result.trend}")
print(f"Alerts Sent: {sum(result.alerts_sent.values())}")
```

### 3. Start Automated Scheduler

```bash
# Basic monitoring (daily quality checks)
python -m hrp.agents.run_scheduler --with-quality-monitoring

# Full monitoring (quality + reports + data ingestion)
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    --with-daily-report \
    --daily-report-time="07:00"
```

### 4. Custom Thresholds

```python
from hrp.monitoring.quality_monitor import (
    DataQualityMonitor,
    MonitoringThresholds,
)

# Strict thresholds for production
strict_thresholds = MonitoringThresholds(
    health_score_warning=95.0,
    health_score_critical=80.0,
    freshness_warning_days=1,
    freshness_critical_days=3,
)

monitor = DataQualityMonitor(
    thresholds=strict_thresholds,
    send_alerts=True,
)
result = monitor.run_daily_check()
```

## Email Notification Setup

### Required Environment Variables

```bash
# Required for email alerts
export RESEND_API_KEY="your_resend_api_key"
export NOTIFICATION_EMAIL="your_email@example.com"

# Optional: Customize from email
export NOTIFICATION_FROM_EMAIL="noreply@hrp.local"
```

### Get Resend API Key

1. Go to https://resend.com
2. Sign up/login
3. Go to API Keys
4. Create new API key
5. Copy and set as environment variable

### Test Email Notifications

```python
from hrp.notifications.email import EmailNotifier

notifier = EmailNotifier()
success = notifier.send_email(
    subject="Test Email from HRP",
    body="This is a test email to verify notifications are working.",
)

print(f"Email sent: {success}")
```

## Dashboard Monitoring

### Access Data Health Page

```bash
# Start dashboard
streamlit run hrp/dashboard/app.py

# Navigate to
http://localhost:8501/Data_Health
```

### Dashboard Features

- **Real-time Health Score** - Large display with color coding
- **Historical Trend Chart** - 90-day health score trend
- **Quality Checks Summary** - Table of all checks with status
- **Flagged Anomalies** - Drill-down into specific issues
- **Ingestion Status** - Recent jobs and success rates
- **Symbol Coverage** - Per-symbol data completeness

## Production Deployment

### Using launchd (macOS)

Create `~/Library/LaunchAgents/com.hrp.scheduler.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hrp.scheduler</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/HRP/.venv/bin/python</string>
        <string>-m</string>
        <string>hrp.agents.run_scheduler</string>
        <string>--with-quality-monitoring</string>
        <string>--quality-monitor-time=06:00</string>
        <string>--health-threshold=90.0</string>
        <string>--with-daily-report</string>
        <string>--daily-report-time=07:00</string>
    </array>
    <key>StandardOutPath</key>
    <string>~/hrp-data/logs/scheduler.out.log</string>
    <key>StandardErrorPath</key>
    <string>~/hrp-data/logs/scheduler.error.log</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

Load the scheduler:

```bash
# Load the scheduler
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Check status
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Stop scheduler
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Monitoring Commands

### Check Scheduler Status

```bash
# See if scheduler is running
launchctl list | grep hrp

# View scheduled jobs
python -c "
from hrp.agents.scheduler import IngestionScheduler
scheduler = IngestionScheduler()
jobs = scheduler.list_jobs()
for job in jobs:
    print(f\"{job['id']}: {job['name']}\")
"
```

### View Quality Reports

```python
from hrp.data.quality.report import QualityReportGenerator
from datetime import date, timedelta

generator = QualityReportGenerator()

# Current report
report = generator.generate_report(date.today())
print(f"Health Score: {report.health_score}/100")
print(f"Critical: {report.critical_issues}")
print(f"Warnings: {report.warning_issues}")

# Historical reports (last 7 days)
for days_ago in range(7, 0, -1):
    report_date = date.today() - timedelta(days=days_ago)
    report = generator.generate_report(report_date)
    print(f"{report_date}: {report.health_score}/100")
```

### Get Health Trend

```python
from hrp.monitoring.quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
summary = monitor.get_health_summary(days=30)

print(f"Current Health: {summary['current_health_score']}")
print(f"Trend Direction: {summary['trend_direction']}")
print(f"Data Points: {summary['trend_data_points']}")
```

## Troubleshooting

### No Email Alerts Received

1. **Check environment variables:**
   ```bash
   echo $RESEND_API_KEY
   echo $NOTIFICATION_EMAIL
   ```

2. **Verify Resend API key:**
   - Login to https://resend.com
   - Generate new API key if needed

3. **Check logs:**
   ```bash
   tail -f ~/hrp-data/logs/scheduler.error.log | grep -i email
   ```

### Health Score Declining

1. **Review recommendations:**
   ```python
   from hrp.monitoring.quality_monitor import DataQualityMonitor

   monitor = DataQualityMonitor()
   result = monitor.run_daily_check()
   for rec in result.recommendations:
       print(f"- {rec}")
   ```

2. **Check top issues:**
   ```python
   from hrp.data.db import get_db

   db = get_db()
   query = """
       SELECT check_name, COUNT(*) as issue_count
       FROM quality_reports
       WHERE severity = 'critical'
       GROUP BY check_name
       ORDER BY issue_count DESC
   """
   df = db.fetchdf(query)
   print(df)
   ```

### Scheduler Not Running

```bash
# Check if running
launchctl list | grep hrp

# View logs
tail -f ~/hrp-data/logs/scheduler.error.log

# Restart
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

## Files Created/Modified

### New Files
1. `/path/to/HRP/hrp/monitoring/quality_monitor.py` - Core monitoring module
2. `/path/to/HRP/docs/setup/Automated-Monitoring-Setup.md` - Complete setup guide
3. `/path/to/HRP/scripts/start_monitoring.sh` - Quick start script

### Modified Files
1. `/path/to/HRP/hrp/monitoring/__init__.py` - Added exports
2. `/path/to/HRP/hrp/agents/scheduler.py` - Added `setup_quality_monitoring()` method
3. `/path/to/HRP/hrp/agents/run_scheduler.py` - Added CLI options

## Next Steps

### Immediate Actions Required

1. **Configure Email Alerts** (Optional but Recommended)
   ```bash
   export RESEND_API_KEY="your_key_here"
   export NOTIFICATION_EMAIL="your_email@example.com"
   ```

2. **Address Critical Issues**
   - Fix 1 stale data issue
   - Reduce 387 completeness warnings

3. **Start Automated Monitoring**
   ```bash
   python -m hrp.agents.run_scheduler --with-quality-monitoring
   ```

### Optional Enhancements

1. **Customize Thresholds**
   - Adjust health score thresholds based on requirements
   - Set stricter freshness limits if needed

2. **Add Additional Checks**
   - Create custom quality checks for specific requirements
   - Add domain-specific validations

3. **Integration with External Systems**
   - Send alerts to Slack, PagerDuty, etc.
   - Create custom monitoring dashboards

## Testing

All quality-related tests pass:

```bash
pytest tests/ -k "quality" -v
# 133 tests passed
```

## Summary

The automated monitoring system is now fully operational and provides:

- ✅ Daily automated quality checks
- ✅ Health score tracking with trend analysis
- ✅ Threshold-based alerting (5 different alert types)
- ✅ Email notification support (Resend)
- ✅ Dashboard integration for visual monitoring
- ✅ Actionable recommendations
- ✅ Scheduler integration for automation
- ✅ Comprehensive documentation

**Current Status:**
- Health Score: 50/100 (Critical)
- Active Alerts: 1 critical, 387 warnings
- Recommendations: 4 actionable items
- Monitoring: Ready for production deployment

**To start monitoring:**
```bash
bash scripts/start_monitoring.sh
python -m hrp.agents.run_scheduler --with-quality-monitoring
```
