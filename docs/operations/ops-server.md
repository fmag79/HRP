# Ops Server Guide

**Purpose:** Health endpoints, Prometheus metrics, and system monitoring for HRP.

---

## Overview

The HRP Ops Server is a lightweight FastAPI application that provides:

- **Liveness Probe** (`/health`) - Confirms the server is running
- **Readiness Probe** (`/ready`) - Verifies database and API connectivity
- **Prometheus Metrics** (`/metrics`) - Exposes application metrics for scraping

Use this server for Kubernetes deployments, load balancer health checks, and Prometheus-based monitoring infrastructure.

---

## Quick Start

```bash
# Start ops server (default: 0.0.0.0:8080)
python -m hrp.ops

# Custom host/port
python -m hrp.ops --host 127.0.0.1 --port 9090

# Verify it's running
curl http://localhost:8080/health
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HRP_OPS_HOST` | Bind host address | `0.0.0.0` |
| `HRP_OPS_PORT` | Bind port | `8080` |
| `HRP_ENVIRONMENT` | Environment mode (`development`, `staging`, `production`) | `development` |

### Configuration via Environment

```bash
# Set custom host/port
export HRP_OPS_HOST=127.0.0.1
export HRP_OPS_PORT=9090
python -m hrp.ops

# Or inline
HRP_OPS_HOST=127.0.0.1 HRP_OPS_PORT=9090 python -m hrp.ops
```

### Command Line Arguments

```bash
python -m hrp.ops --help

# Options:
#   --host TEXT     Bind host (default: 0.0.0.0)
#   --port INTEGER  Bind port (default: 8080)
```

Command line arguments take precedence over environment variables.

---

## Health Endpoints

### GET /health (Liveness Probe)

Returns 200 if the server is running. Use this for Kubernetes liveness probes or load balancer health checks.

```bash
curl http://localhost:8080/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "timestamp": "2026-02-04T15:30:00.123456"
}
```

### GET /ready (Readiness Probe)

Returns 200 if the system is ready to serve requests (database connected, API functional). Returns 503 if not ready.

```bash
curl http://localhost:8080/ready
```

**Response (200 OK - Ready):**
```json
{
  "status": "ready",
  "timestamp": "2026-02-04T15:30:00.123456",
  "checks": {
    "database": "ok",
    "api": "ok"
  }
}
```

**Response (503 Service Unavailable - Not Ready):**
```json
{
  "status": "not_ready",
  "timestamp": "2026-02-04T15:30:00.123456",
  "checks": {
    "database": "error",
    "error": "Connection refused"
  }
}
```

### GET /metrics (Prometheus Metrics)

Returns Prometheus-formatted metrics for scraping.

```bash
curl http://localhost:8080/metrics
```

**Response (text/plain):**
```
# HELP hrp_http_requests_total Total HTTP requests
# TYPE hrp_http_requests_total counter
hrp_http_requests_total{method="GET",endpoint="/health",status="200"} 42.0
hrp_http_requests_total{method="GET",endpoint="/ready",status="200"} 38.0

# HELP hrp_http_request_duration_seconds HTTP request duration in seconds
# TYPE hrp_http_request_duration_seconds histogram
hrp_http_request_duration_seconds_bucket{le="0.005",method="GET",endpoint="/health"} 42.0
hrp_http_request_duration_seconds_bucket{le="0.01",method="GET",endpoint="/health"} 42.0
...

# HELP hrp_active_connections Number of active connections
# TYPE hrp_active_connections gauge
hrp_active_connections 2.0
```

---

## Prometheus Metrics Reference

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hrp_http_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests received |
| `hrp_http_request_duration_seconds` | Histogram | `method`, `endpoint` | Request duration in seconds |
| `hrp_active_connections` | Gauge | - | Current number of active connections |

### Example Prometheus Queries

```promql
# Request rate per second (last 5 minutes)
rate(hrp_http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(hrp_http_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(hrp_http_requests_total{status=~"5.."}[5m])) / sum(rate(hrp_http_requests_total[5m]))
```

---

## Startup Validation

For production deployments, use startup validation to fail fast on missing configuration:

```python
from hrp.utils.startup import fail_fast_startup

# Call at application entry points
fail_fast_startup()
```

This validates:
- Required secrets for the current environment (e.g., `ANTHROPIC_API_KEY` in production)
- Database connectivity
- Essential configuration

**Example Error:**
```
RuntimeError: Startup validation failed:
  - Missing required secret: ANTHROPIC_API_KEY
```

### Using fail_fast_startup() in Custom Scripts

```python
from hrp.utils.startup import fail_fast_startup, validate_startup

# Strict mode - raises exception on failure
fail_fast_startup()

# Non-strict mode - returns list of errors
errors = validate_startup()
if errors:
    print("Warnings:", errors)
```

---

## Running as a launchd Service

### Install the Service

The ops server can run as a macOS launchd service for automatic startup and restart.

**Service Plist:** `launchd/com.hrp.ops-server.plist`

```bash
# Install all HRP launchd jobs (including ops server)
./scripts/manage_launchd.sh install

# Check status of all HRP jobs
./scripts/manage_launchd.sh status

# Uninstall all HRP launchd jobs
./scripts/manage_launchd.sh uninstall

# Reload all jobs (uninstall + install)
./scripts/manage_launchd.sh reload
```

**Note:** The `manage_launchd.sh` script manages all HRP jobs at once (prices, features, universe, ops-server, etc.). To manage the ops server individually, use the manual launchctl commands below.

### Manual launchd Management

```bash
# Load (start) the service
launchctl load ~/Library/LaunchAgents/com.hrp.ops-server.plist

# Unload (stop) the service
launchctl unload ~/Library/LaunchAgents/com.hrp.ops-server.plist

# Check if running
launchctl list | grep hrp.ops

# View logs
tail -f ~/hrp-data/logs/ops-server.log
tail -f ~/hrp-data/logs/ops-server.error.log
```

### Service Configuration

The launchd plist configures:
- **RunAtLoad:** Starts automatically on login
- **KeepAlive:** Restarts if the process exits
- **Host:** `127.0.0.1` (localhost only for security)
- **Port:** `8080`
- **Environment:** `HRP_ENVIRONMENT=production`

**Log Locations:**
- stdout: `~/hrp-data/logs/ops-server.log`
- stderr: `~/hrp-data/logs/ops-server.error.log`

---

## Prometheus Scraping Setup

### prometheus.yml Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hrp-ops'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    extra_hosts:
      - "host.docker.internal:host-gateway"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

With `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'hrp-ops'
    static_configs:
      - targets: ['host.docker.internal:8080']
```

---

## Example Curl Commands

### Health Check Script

```bash
#!/bin/bash
# health_check.sh - Quick health check for HRP ops server

HOST=${HRP_OPS_HOST:-localhost}
PORT=${HRP_OPS_PORT:-8080}

echo "=== HRP Ops Server Health Check ==="
echo

# Liveness
echo "Liveness (/health):"
curl -s http://${HOST}:${PORT}/health | jq .
echo

# Readiness
echo "Readiness (/ready):"
READY=$(curl -s -w "\n%{http_code}" http://${HOST}:${PORT}/ready)
HTTP_CODE=$(echo "$READY" | tail -1)
BODY=$(echo "$READY" | head -n -1)
echo "$BODY" | jq .
echo "HTTP Status: $HTTP_CODE"
echo

# Metrics sample
echo "Metrics (/metrics) - first 20 lines:"
curl -s http://${HOST}:${PORT}/metrics | head -20
echo "..."
```

### Individual Commands

```bash
# Liveness probe
curl http://localhost:8080/health

# Liveness with HTTP status code
curl -w "\nHTTP Status: %{http_code}\n" http://localhost:8080/health

# Readiness probe
curl http://localhost:8080/ready

# Readiness with error handling
curl -f http://localhost:8080/ready || echo "System not ready!"

# Prometheus metrics
curl http://localhost:8080/metrics

# Metrics filtered for HRP-specific
curl -s http://localhost:8080/metrics | grep "^hrp_"

# JSON formatted health check
curl -s http://localhost:8080/health | python -m json.tool

# Continuous monitoring (every 5 seconds)
watch -n 5 'curl -s http://localhost:8080/ready | jq .'
```

---

## Programmatic Usage

### Create and Run the App

```python
from hrp.ops.server import create_app, run_server

# Create the FastAPI app (for testing or custom hosting)
app = create_app()

# Run with default settings
run_server()

# Run with custom host/port
run_server(host="127.0.0.1", port=9090)
```

### Check System Readiness

```python
from hrp.ops.server import check_system_ready

is_ready, details = check_system_ready()
print(f"Ready: {is_ready}")
print(f"Details: {details}")
# Ready: True
# Details: {'database': 'ok', 'api': 'ok'}
```

### Using the MetricsCollector

```python
from hrp.ops.metrics import MetricsCollector

collector = MetricsCollector()

# Get system metrics (CPU, memory, disk)
system = collector.collect_system_metrics()
print(f"CPU: {system['cpu_percent']}%")
print(f"Memory: {system['memory_percent']}%")

# Get data pipeline health
data = collector.collect_data_metrics()
print(f"Data status: {data['status']}")
print(f"Health score: {data.get('health_score', 'N/A')}")
```

---

## Troubleshooting

### Server Won't Start

**Port already in use:**
```bash
# Check what's using the port
lsof -i :8080

# Kill the process
kill <PID>

# Or use a different port
python -m hrp.ops --port 9090
```

**Database connection error:**
```bash
# Verify database exists
ls -la ~/hrp-data/hrp.duckdb

# Test database connection
python -c "from hrp.api.platform import PlatformAPI; api = PlatformAPI(); print(api.health_check())"
```

### Readiness Probe Returns 503

**Check the response:**
```bash
curl -v http://localhost:8080/ready
```

**Common causes:**
1. Database file missing or corrupted
2. Database locked by another process
3. Missing environment variables

**Resolution:**
```bash
# Check database health
python -c "
from hrp.api.platform import PlatformAPI
api = PlatformAPI()
print(api.health_check())
"
```

### Metrics Not Showing

**Verify metrics endpoint:**
```bash
curl -v http://localhost:8080/metrics
```

**Check for custom metrics:**
```bash
curl -s http://localhost:8080/metrics | grep hrp_
```

---

## Related Documentation

- [Alert Thresholds](alert-thresholds.md) - Configuring monitoring thresholds
- [Deployment Guide](deployment.md) - Production deployment procedures
- [Monitoring Setup](monitoring-universe-scheduling.md) - Database monitoring queries
- [Cookbook](cookbook.md) - Practical recipes for common operations

---

## Key Files

| File | Purpose |
|------|---------|
| `hrp/ops/server.py` | FastAPI app with health endpoints |
| `hrp/ops/__main__.py` | CLI entry point |
| `hrp/ops/thresholds.py` | Configurable alert thresholds |
| `hrp/ops/metrics.py` | MetricsCollector for system/data metrics |
| `hrp/utils/startup.py` | Startup validation utilities |
| `launchd/com.hrp.ops-server.plist` | macOS launchd service configuration |
