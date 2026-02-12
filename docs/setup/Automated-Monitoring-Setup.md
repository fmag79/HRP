# Automated Data Quality Monitoring System

## Overview

The HRP platform includes a comprehensive automated monitoring system that continuously monitors data quality, tracks health scores, and sends alerts when issues are detected. This system ensures data integrity and provides early warning of potential problems.

## Features

### 1. **Daily Quality Checks**
- Runs automatically every day at 6 AM ET (configurable)
- Checks 5 quality dimensions:
  - Price anomalies (>50% moves, invalid data)
  - Completeness (missing prices for active symbols)
  - Gap detection (missing trading days)
  - Stale data (symbols not updated recently)
  - Volume anomalies (negative or zero volume)

### 2. **Health Score Tracking**
- Single metric (0-100) representing overall data health
- Tracked over time in the database
- Trend analysis (improving/stable/declining)
- Historical visualization in dashboard

### 3. **Threshold-Based Alerting**
Configurable thresholds trigger different alert levels:

| Metric | Warning | Critical |
|--------|---------|----------|
| Health Score | < 90 | < 70 |
| Data Freshness | > 3 days stale | > 5 days stale |
| Anomaly Count | - | > 100 anomalies |

### 4. **Email Notifications**
- **Critical Alerts**: Immediate email for critical issues
- **Daily Summary**: Daily email with health score and all issues
- **Trend Warnings**: Alerts when health score is declining

## Quick Start

### 1. Configure Environment Variables

Set up email notifications (optional but recommended):

```bash
# Required for email alerts
export RESEND_API_KEY="your_resend_api_key"
export NOTIFICATION_EMAIL="your_email@example.com"

# Optional: Customize from email
export NOTIFICATION_FROM_EMAIL="noreply@hrp.local"
```

### 2. Test the Monitoring System

Run a manual quality check to verify setup:

```bash
python -c "
from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts
from datetime import date

result = run_quality_monitor_with_alerts(
    as_of_date=date.today(),
    send_alerts=True,
)

print(f'Health Score: {result.health_score}/100')
print(f'Trend: {result.trend}')
print(f'Critical Issues: {result.critical_issues}')
print(f'Warnings: {result.warning_issues}')
print(f'Alerts Sent: {sum(result.alerts_sent.values())}')
print(f'Recommendations:')
for rec in result.recommendations:
    print(f'  - {rec}')
"
```

### 3. Start the Automated Scheduler

Launch the scheduler with quality monitoring enabled:

```bash
# Basic setup (daily quality monitoring + daily data ingestion)
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0
```

## Scheduler Configuration

### Full Monitoring Setup

Enable all monitoring and reporting features:

```bash
python -m hrp.agents.run_scheduler \
    # Data Ingestion Pipeline
    --price-time="18:00" \
    --universe-time="18:05" \
    --feature-time="18:10" \
    \
    # Quality Monitoring
    --with-quality-monitoring \
    --quality-monitor-time="06:00" \
    --health-threshold=90.0 \
    \
    # Daily Backup
    --backup-time="02:00" \
    --backup-keep-days=30 \
    \
    # Research Reports
    --with-daily-report \
    --daily-report-time="07:00" \
    --with-weekly-report \
    --weekly-report-time="20:00" \
    \
    # Weekly Fundamentals
    --fundamentals-time="10:00" \
    --fundamentals-day="sat" \
    --fundamentals-source="simfin"
```

### Minimal Monitoring Setup

Just quality monitoring (no research agents):

```bash
python -m hrp.agents.run_scheduler \
    --with-quality-monitoring \
    --quality-monitor-time="06:00"
```

## Command-Line Options

### Quality Monitoring Options

| Option | Default | Description |
|--------|---------|-------------|
| `--with-quality-monitoring` | off | Enable daily quality monitoring |
| `--quality-monitor-time` | 06:00 | Time to run quality check (HH:MM) |
| `--health-threshold` | 90.0 | Health score threshold for warnings |

### Data Ingestion Options

| Option | Default | Description |
|--------|---------|-------------|
| `--price-time` | 18:00 | Time for price ingestion (6 PM ET) |
| `--universe-time` | 18:05 | Time for universe update |
| `--feature-time` | 18:10 | Time for feature computation |
| `--backup-time` | 02:00 | Time for daily backup (2 AM ET) |
| `--backup-keep-days` | 5 | Days of backups to retain |
| `--no-backup` | off | Disable daily backup |

### Research Agent Options

| Option | Default | Description |
|--------|---------|-------------|
| `--with-research-triggers` | off | Enable event-driven agent pipeline |
| `--with-signal-scan` | off | Enable weekly signal scan |
| `--signal-scan-time` | 19:00 | Time for signal scan (7 PM ET) |
| `--signal-scan-day` | mon | Day for signal scan |
| `--ic-threshold` | 0.03 | Minimum IC to create hypothesis |
| `--with-quality-sentinel` | off | Enable ML Quality Sentinel |
| `--sentinel-time` | 06:00 | Time for ML Quality Sentinel |
| `--trigger-poll-interval` | 60 | Lineage event poll interval (seconds) |

### Report Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--with-daily-report` | off | Enable daily research report |
| `--daily-report-time` | 07:00 | Time for daily report (7 AM ET) |
| `--with-weekly-report` | off | Enable weekly research report |
| `--weekly-report-time` | 20:00 | Time for weekly report (8 PM ET) |

## Understanding Alerts

### Alert Types

#### 1. Health Score Alerts
- **Warning**: Health score below 90
  - Action: Review warning issues, schedule maintenance
- **Critical**: Health score below 70
  - Action: Immediate investigation, address critical issues

#### 2. Critical Issues Alert
Sent whenever critical issues are detected, regardless of health score:
- Price anomalies (invalid data)
- Stale data (symbols not updated)
- Data freshness problems
- Volume anomalies

#### 3. Data Freshness Alert
Triggered when price data is too old:
- **Warning**: Data > 3 days old
- **Critical**: Data > 5 days old
- Action: Check ingestion pipeline, verify data sources

#### 4. Anomaly Spike Alert
Sent when anomaly count is unusually high:
- Trigger: > 100 total anomalies
- Action: Investigate systemic data issues

### Alert Email Contents

**Critical Alert Email:**
- Subject: `[CRITICAL] Data Quality Alert - YYYY-MM-DD`
- Contents:
  - Report date
  - Number of critical issues
  - Table of first 20 critical issues (symbol, check, description)

**Daily Summary Email:**
- Subject: `✅ Daily Data Quality Report - YYYY-MM-DD`
- Contents:
  - Health score (0-100) with color coding
  - Overall status (PASSED/FAILED)
  - Summary statistics (checks run, issues by severity)
  - Detailed check results table
  - Runtime metrics

## Programmatic Usage

### Running Manual Quality Checks

```python
from hrp.monitoring.quality_monitor import DataQualityMonitor, MonitoringThresholds
from datetime import date

# Create monitor with custom thresholds
monitor = DataQualityMonitor(
    thresholds=MonitoringThresholds(
        health_score_warning=85.0,  # Stricter threshold
        health_score_critical=65.0,
        freshness_warning_days=2,
        freshness_critical_days=4,
    ),
    send_alerts=True,
)

# Run daily check
result = monitor.run_daily_check(as_of_date=date.today())

# Access results
print(f"Health Score: {result.health_score}/100")
print(f"Trend: {result.trend}")
print(f"Critical Issues: {result.critical_issues}")
print(f"Warnings: {result.warning_issues}")
print(f"Alerts Sent: {result.alerts_sent}")
print(f"Recommendations:")
for rec in result.recommendations:
    print(f"  - {rec}")
```

### Getting Health Summary

```python
from hrp.monitoring.quality_monitor import DataQualityMonitor

monitor = DataQualityMonitor()
summary = monitor.get_health_summary(days=30)

print(f"Current Health: {summary['current_health_score']}")
print(f"Trend: {summary['trend_direction']}")
print(f"Data Points: {summary['trend_data_points']}")
```

### Custom Alert Thresholds

```python
from hrp.monitoring.quality_monitor import (
    DataQualityMonitor,
    MonitoringThresholds,
    run_quality_monitor_with_alerts,
)

# Define strict thresholds for production
strict_thresholds = MonitoringThresholds(
    health_score_warning=95.0,   # Alert at 95
    health_score_critical=80.0,  # Critical at 80
    freshness_warning_days=1,    # Warning after 1 day
    freshness_critical_days=3,   # Critical after 3 days
    anomaly_count_critical=50,   # Alert at 50 anomalies
)

# Run with custom thresholds
result = run_quality_monitor_with_alerts(
    thresholds=strict_thresholds,
    send_alerts=True,
)
```

## Dashboard Monitoring

### Data Health Page

Access the Data Health dashboard at `http://localhost:8501/Data_Health`:

**Features:**
- Real-time health score display
- Historical trend chart (90 days)
- Quality checks summary table
- Flagged anomalies drill-down
- Ingestion status monitoring
- Symbol coverage analysis

**Metrics Displayed:**
- Total symbols
- Total records
- Date range
- Data freshness
- Ingestion success rate
- Last successful ingestion

### Quality Alert Banner

The dashboard shows an alert banner when issues are detected:

- **Critical**: Red banner with critical issue count
- **Warnings**: Yellow banner with warning count
- **Healthy**: No banner (health score >= 90)

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
   - Update environment variable

3. **Check logs:**
   ```bash
   tail -f ~/hrp-data/logs/scheduler.error.log | grep -i email
   ```

4. **Test email sending:**
   ```python
   from hrp.notifications.email import EmailNotifier

   notifier = EmailNotifier()
   success = notifier.send_email(
       subject="Test Email",
       body="This is a test email from HRP.",
   )
   print(f"Email sent: {success}")
   ```

### Health Score Declining

1. **Check recent quality reports:**
   ```python
   from hrp.data.quality.report import QualityReportGenerator
   from datetime import date, timedelta

   generator = QualityReportGenerator()
   for days_ago in range(7, 0, -1):
       report_date = date.today() - timedelta(days=days_ago)
       report = generator.generate_report(report_date)
       print(f"{report_date}: {report.health_score}/100")
   ```

2. **Review recommendations:**
   ```python
   from hrp.monitoring.quality_monitor import DataQualityMonitor

   monitor = DataQualityMonitor()
   result = monitor.run_daily_check()
   for rec in result.recommendations:
       print(f"- {rec}")
   ```

3. **Identify top issues:**
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

1. **Check scheduler status:**
   ```bash
   launchctl list | grep hrp
   ```

2. **View scheduler logs:**
   ```bash
   tail -f ~/hrp-data/logs/scheduler.error.log
   ```

3. **Restart scheduler:**
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
   launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
   ```

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
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

### Monitoring Checklist

- [ ] Email notifications configured and tested
- [ ] Scheduler running via launchd
- [ ] Quality monitoring enabled (daily 6 AM ET)
- [ ] Health score threshold set appropriately
- [ ] Dashboard accessible at `http://localhost:8501/Data_Health`
- [ ] Logs being written to `~/hrp-data/logs/`
- [ ] Daily backup configured (2 AM ET)
- [ ] Data freshness alerts working
- [ ] Trend tracking enabled (7+ days of data)

## Advanced Configuration

### Custom Quality Checks

Add custom quality checks by extending `QualityCheck`:

```python
from hrp.data.quality.checks import QualityCheck, QualityIssue, IssueSeverity

class CustomQualityCheck(QualityCheck):
    """Custom quality check example."""

    def run(self, as_of_date: date) -> CheckResult:
        # Implement check logic
        issues = []
        # ... detect issues ...

        return CheckResult(
            check_name="custom_check",
            passed=len(issues) == 0,
            critical_count=sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            warning_count=sum(1 for i in issues if i.severity == IssueSeverity.WARNING),
            issues=issues,
            run_time_ms=elapsed_ms,
        )
```

### Integration with External Systems

Send alerts to external monitoring systems:

```python
from hrp.monitoring.quality_monitor import DataQualityMonitor

class CustomDataQualityMonitor(DataQualityMonitor):
    def _check_and_alert(self, report, as_of_date):
        # Call parent to send email alerts
        alerts_sent = super()._check_and_alert(report, as_of_date)

        # Send to external system (e.g., Slack, PagerDuty)
        if report.health_score < 70:
            self.send_to_pagerduty({
                "severity": "critical",
                "health_score": report.health_score,
                "critical_issues": report.critical_issues,
            })

        return alerts_sent

    def send_to_pagerduty(self, alert_data):
        # Implement PagerDuty integration
        pass
```

## API Reference

### DataQualityMonitor

```python
class DataQualityMonitor:
    def __init__(
        self,
        thresholds: MonitoringThresholds | None = None,
        send_alerts: bool = True,
        db_path: str | None = None,
    )

    def run_daily_check(
        self, as_of_date: date | None = None
    ) -> MonitoringResult

    def get_health_summary(self, days: int = 30) -> dict[str, Any]
```

### MonitoringThresholds

```python
@dataclass
class MonitoringThresholds:
    health_score_warning: float = 90.0
    health_score_critical: float = 70.0
    freshness_warning_days: int = 3
    freshness_critical_days: int = 5
    anomaly_count_critical: int = 100
```

### MonitoringResult

```python
@dataclass
class MonitoringResult:
    timestamp: datetime
    health_score: float
    passed: bool
    critical_issues: int
    warning_issues: int
    total_issues: int
    alerts_sent: dict[str, bool]
    recommendations: list[str]
    trend: str  # "improving", "stable", "declining"
```

## Support

For issues or questions:
- Check logs: `~/hrp-data/logs/scheduler.error.log`
- Review dashboard: `http://localhost:8501/Data_Health`
- Run manual check: `python -c "from hrp.monitoring.quality_monitor import run_quality_monitor_with_alerts; run_quality_monitor_with_alerts()"`
