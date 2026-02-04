# Alert Thresholds Guide

**Purpose:** Configurable alert thresholds for HRP monitoring, data quality, and drift detection.

---

## Overview

HRP uses the `OpsThresholds` dataclass to define alert thresholds for:

- **Health Score** - Overall data quality health (0-100 scale)
- **Data Freshness** - Days since last price/feature update
- **Anomaly Detection** - Number of data anomalies detected
- **Model Drift** - Feature drift (PSI, KL divergence) and concept drift (IC decay)
- **Ingestion Health** - Success rate of data ingestion jobs

Thresholds are loaded with this priority: **Environment Variables > YAML Config > Defaults**

---

## OpsThresholds Dataclass

All thresholds are defined in `hrp/ops/thresholds.py`:

```python
from dataclasses import dataclass

@dataclass
class OpsThresholds:
    """Alert thresholds for monitoring."""

    # Health score thresholds (0-100)
    health_score_warning: float = 90.0
    health_score_critical: float = 70.0

    # Data freshness thresholds (days)
    freshness_warning_days: int = 3
    freshness_critical_days: int = 5

    # Anomaly thresholds
    anomaly_count_warning: int = 50
    anomaly_count_critical: int = 100

    # Drift thresholds
    kl_divergence_threshold: float = 0.2
    psi_threshold: float = 0.2
    ic_decay_threshold: float = 0.2

    # Ingestion thresholds (percentage)
    ingestion_success_rate_warning: float = 95.0
    ingestion_success_rate_critical: float = 80.0
```

---

## Threshold Reference

### Health Score Thresholds

Control alerts based on overall data quality health score.

| Threshold | Default | Description |
|-----------|---------|-------------|
| `health_score_warning` | 90.0 | Score below this triggers warning alerts |
| `health_score_critical` | 70.0 | Score below this triggers critical alerts |

**Alerts triggered:**
- **Warning** (score < 90): Logged warning, included in daily summary email
- **Critical** (score < 70): Immediate email alert with "URGENT" prefix

**Used by:** `DataQualityMonitor.run_daily_check()`, Ops Dashboard

### Data Freshness Thresholds

Control alerts based on how stale the price data is.

| Threshold | Default | Description |
|-----------|---------|-------------|
| `freshness_warning_days` | 3 | Days stale before warning |
| `freshness_critical_days` | 5 | Days stale before critical alert |

**Alerts triggered:**
- **Warning** (> 3 days stale): Logged warning, recommendation in quality report
- **Critical** (> 5 days stale): Immediate email alert, "Ingestion may be failing" message

**Used by:** `DataQualityMonitor._generate_recommendations()`, `DataQualityMonitor._send_alerts()`

### Anomaly Thresholds

Control alerts based on the number of data quality issues detected.

| Threshold | Default | Description |
|-----------|---------|-------------|
| `anomaly_count_warning` | 50 | Issue count before warning |
| `anomaly_count_critical` | 100 | Issue count before critical alert |

**Alerts triggered:**
- **Critical** (> 100 anomalies): Email alert with "Anomaly Count Spike" subject

**Used by:** `DataQualityMonitor._send_alerts()`

### Model Drift Thresholds

Control alerts for feature drift and concept drift in ML models.

| Threshold | Default | Description |
|-----------|---------|-------------|
| `kl_divergence_threshold` | 0.2 | KL divergence threshold for distribution drift |
| `psi_threshold` | 0.2 | Population Stability Index threshold for feature drift |
| `ic_decay_threshold` | 0.2 | Information Coefficient decay threshold (20% relative decay) |

**Alerts triggered:**
- **Feature Drift** (PSI > 0.2): Warning logged per feature, drift flagged in result
- **Concept Drift** (IC decay > 0.2): Warning logged, model retraining recommended

**Used by:** `DriftMonitor.check_feature_drift()`, `DriftMonitor.check_concept_drift()`

**Interpretation:**
- **KL Divergence**: Measures how much one distribution differs from another (0 = identical)
- **PSI**: Population Stability Index (< 0.1 = no shift, 0.1-0.2 = moderate, > 0.2 = significant)
- **IC Decay**: Relative decline in Information Coefficient from reference period

### Ingestion Thresholds

Control alerts based on data ingestion job success rates.

| Threshold | Default | Description |
|-----------|---------|-------------|
| `ingestion_success_rate_warning` | 95.0 | Success rate % below this triggers warning |
| `ingestion_success_rate_critical` | 80.0 | Success rate % below this triggers critical alert |

**Used by:** Ingestion job monitoring (future implementation)

---

## Configuration Methods

### Method 1: Environment Variables (Highest Priority)

Set environment variables with the pattern `HRP_THRESHOLD_<FIELD_NAME_UPPER>`:

```bash
# Health score thresholds
export HRP_THRESHOLD_HEALTH_SCORE_WARNING=85
export HRP_THRESHOLD_HEALTH_SCORE_CRITICAL=60

# Data freshness thresholds
export HRP_THRESHOLD_FRESHNESS_WARNING_DAYS=2
export HRP_THRESHOLD_FRESHNESS_CRITICAL_DAYS=4

# Anomaly thresholds
export HRP_THRESHOLD_ANOMALY_COUNT_WARNING=30
export HRP_THRESHOLD_ANOMALY_COUNT_CRITICAL=75

# Drift thresholds
export HRP_THRESHOLD_KL_DIVERGENCE_THRESHOLD=0.15
export HRP_THRESHOLD_PSI_THRESHOLD=0.15
export HRP_THRESHOLD_IC_DECAY_THRESHOLD=0.15

# Ingestion thresholds
export HRP_THRESHOLD_INGESTION_SUCCESS_RATE_WARNING=98
export HRP_THRESHOLD_INGESTION_SUCCESS_RATE_CRITICAL=90
```

Environment variables are useful for:
- Temporary overrides during debugging
- Per-environment configuration (dev vs staging vs production)
- CI/CD pipeline configuration

### Method 2: YAML Config File (Medium Priority)

Create a YAML file at `~/hrp-data/config/thresholds.yaml`:

```yaml
# Health score thresholds (0-100)
health_score_warning: 90.0
health_score_critical: 70.0

# Data freshness thresholds (days)
freshness_warning_days: 3
freshness_critical_days: 5

# Anomaly thresholds (count)
anomaly_count_warning: 50
anomaly_count_critical: 100

# Drift thresholds (0.0-1.0)
kl_divergence_threshold: 0.2
psi_threshold: 0.2
ic_decay_threshold: 0.2

# Ingestion thresholds (percentage)
ingestion_success_rate_warning: 95.0
ingestion_success_rate_critical: 80.0
```

YAML configuration is useful for:
- Persistent configuration across restarts
- Version-controlled threshold settings
- Sharing configuration across team members

### Method 3: Defaults (Lowest Priority)

If no environment variable or YAML config is set, the default values from the dataclass are used.

---

## Configuration Examples

### Stricter Monitoring (Production)

For production environments requiring tighter monitoring:

```yaml
# ~/hrp-data/config/thresholds.yaml (Production)
health_score_warning: 95.0
health_score_critical: 80.0

freshness_warning_days: 1
freshness_critical_days: 2

anomaly_count_warning: 25
anomaly_count_critical: 50

kl_divergence_threshold: 0.1
psi_threshold: 0.1
ic_decay_threshold: 0.1

ingestion_success_rate_warning: 99.0
ingestion_success_rate_critical: 95.0
```

### Relaxed Monitoring (Development)

For development environments with more tolerance:

```yaml
# ~/hrp-data/config/thresholds.yaml (Development)
health_score_warning: 80.0
health_score_critical: 50.0

freshness_warning_days: 7
freshness_critical_days: 14

anomaly_count_warning: 100
anomaly_count_critical: 200

kl_divergence_threshold: 0.3
psi_threshold: 0.3
ic_decay_threshold: 0.3

ingestion_success_rate_warning: 90.0
ingestion_success_rate_critical: 70.0
```

---

## Programmatic Usage

### Loading Thresholds

```python
from hrp.ops.thresholds import load_thresholds, OpsThresholds

# Load with priority: Env > YAML > Defaults
thresholds = load_thresholds()

# Load from custom YAML path
thresholds = load_thresholds(config_path="/path/to/custom/thresholds.yaml")

# Access threshold values
print(f"Health warning: {thresholds.health_score_warning}")
print(f"Health critical: {thresholds.health_score_critical}")
print(f"Freshness warning: {thresholds.freshness_warning_days} days")
```

### Using in Monitoring Code

```python
from hrp.ops.thresholds import load_thresholds

thresholds = load_thresholds()

# Check health score
health_score = 85.0
if health_score < thresholds.health_score_critical:
    send_critical_alert("Health score critically low!")
elif health_score < thresholds.health_score_warning:
    log_warning("Health score below target")

# Check data freshness
days_stale = 4
if days_stale > thresholds.freshness_critical_days:
    send_critical_alert("Data critically stale!")
elif days_stale > thresholds.freshness_warning_days:
    log_warning("Data becoming stale")
```

### Creating Custom Thresholds

```python
from hrp.ops.thresholds import OpsThresholds

# Create with custom values (not recommended for production)
custom = OpsThresholds(
    health_score_warning=95.0,
    health_score_critical=80.0,
    freshness_warning_days=1,
    freshness_critical_days=2,
)
```

---

## Alert Flow Summary

```
Data Quality Check
        |
        v
+------------------+
|  Health Score    |
|  < critical?     |---> YES ---> Immediate Critical Email
+------------------+
        | NO
        v
+------------------+
|  Health Score    |
|  < warning?      |---> YES ---> Warning in Daily Summary
+------------------+
        | NO
        v
+------------------+
|  Data Freshness  |
|  > critical?     |---> YES ---> Immediate Critical Email
+------------------+
        | NO
        v
+------------------+
|  Anomaly Count   |
|  > critical?     |---> YES ---> Anomaly Spike Email
+------------------+
        | NO
        v
    Normal Operation
```

---

## Monitoring Integration

### Ops Dashboard

The Streamlit Ops Dashboard displays current threshold values:

```
System Status > Alert Thresholds

Health Thresholds:
- Warning: 90.0%
- Critical: 70.0%

Freshness Thresholds:
- Warning: 3 days
- Critical: 5 days
```

Access at: `http://localhost:8501` (page 9: Ops)

### Prometheus Metrics

Threshold breaches are not directly exposed as Prometheus metrics, but the underlying health scores are:

```promql
# Health score from /ready endpoint
hrp_data_health_score

# Alert when health score drops below warning
hrp_data_health_score < 90
```

---

## Troubleshooting

### Thresholds Not Loading from YAML

**Check file exists:**
```bash
ls -la ~/hrp-data/config/thresholds.yaml
```

**Check YAML syntax:**
```bash
python -c "import yaml; yaml.safe_load(open('$HOME/hrp-data/config/thresholds.yaml'))"
```

**Check load logs:**
```python
from hrp.ops.thresholds import load_thresholds
import logging
logging.basicConfig(level=logging.DEBUG)
thresholds = load_thresholds()
```

### Environment Variable Not Taking Effect

**Verify environment variable is set:**
```bash
env | grep HRP_THRESHOLD
```

**Verify type conversion:**
```python
import os
print(os.environ.get("HRP_THRESHOLD_HEALTH_SCORE_WARNING"))
# Should print the value as string, e.g., "85"
```

**Check for typos in variable name:**
```bash
# Correct format
HRP_THRESHOLD_HEALTH_SCORE_WARNING=85

# Wrong (lowercase)
HRP_THRESHOLD_health_score_warning=85

# Wrong (missing THRESHOLD)
HRP_HEALTH_SCORE_WARNING=85
```

### Alerts Not Triggering

**Verify thresholds are loaded:**
```python
from hrp.ops.thresholds import load_thresholds
thresholds = load_thresholds()
print(f"Health warning: {thresholds.health_score_warning}")
print(f"Health critical: {thresholds.health_score_critical}")
```

**Verify alert manager is configured:**
```bash
# Check if email notifications are configured
env | grep -E "RESEND_API_KEY|NOTIFICATION_EMAIL"
```

---

## Related Documentation

- [Ops Server Guide](ops-server.md) - Health endpoints and Prometheus metrics
- [Deployment Guide](deployment.md) - Production deployment procedures
- [Monitoring Setup](monitoring-universe-scheduling.md) - Database monitoring queries
- [Cookbook](cookbook.md) - Practical recipes for common operations

---

## Key Files

| File | Purpose |
|------|---------|
| `hrp/ops/thresholds.py` | OpsThresholds dataclass and load_thresholds() function |
| `hrp/monitoring/quality_monitor.py` | DataQualityMonitor using health/freshness/anomaly thresholds |
| `hrp/monitoring/drift_monitor.py` | DriftMonitor using PSI/KL/IC decay thresholds |
| `hrp/dashboard/pages/9_Ops.py` | Ops Dashboard displaying threshold values |
| `~/hrp-data/config/thresholds.yaml` | YAML configuration file (optional) |
