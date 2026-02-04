# Tier 3 Phase 2: Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement operational monitoring, security hardening, and reliability infrastructure for HRP.

**Architecture:** FastAPI ops server exposing `/health`, `/ready`, `/metrics` endpoints. Configurable thresholds via YAML + env vars. Connection pooling with retry/backoff for DuckDB. Secret filtering in logs. Integration test fixtures for isolated testing.

**Tech Stack:** FastAPI, uvicorn, prometheus-client, DuckDB WAL mode, pytest fixtures

---

## Phase 0: Repository Cleanup

### Task 0.1: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add artifact patterns to .gitignore**

```bash
cat >> .gitignore << 'EOF'

# Generated artifacts (do not commit)
mlflow.db
coverage.xml
test_results.txt
*.coverage
.coverage
htmlcov/
EOF
```

**Step 2: Verify patterns added**

Run: `grep -c "mlflow.db" .gitignore`
Expected: `1`

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add generated artifacts to .gitignore"
```

---

### Task 0.2: Remove tracked artifacts from git

**Files:**
- Remove from git: `mlflow.db`, `coverage.xml`, `test_results.txt`

**Step 1: Remove from git tracking (keep local files)**

```bash
git rm --cached mlflow.db coverage.xml test_results.txt 2>/dev/null || echo "Files not tracked"
```

**Step 2: Verify files untracked**

Run: `git status --porcelain | grep -E "mlflow.db|coverage.xml" | head -1`
Expected: Empty or starts with `??` (untracked)

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove generated artifacts from git tracking"
```

---

## Phase 1: Fix Packaging + CLI Entrypoint

### Task 1.1: Create unified CLI entrypoint

**Files:**
- Create: `hrp/cli.py`
- Test: `tests/test_cli_entrypoint.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_entrypoint.py
"""Test CLI entrypoint."""

def test_cli_module_imports():
    """CLI module should import without error."""
    from hrp import cli
    assert hasattr(cli, "main")


def test_cli_main_is_callable():
    """CLI main should be callable."""
    from hrp.cli import main
    assert callable(main)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_entrypoint.py -v`
Expected: FAIL with "No module named 'hrp.cli'"

**Step 3: Write minimal implementation**

```python
# hrp/cli.py
"""HRP CLI - unified command-line interface."""

from hrp.agents.cli import main

if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_entrypoint.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/cli.py tests/test_cli_entrypoint.py
git commit -m "feat: add unified CLI entrypoint"
```

---

### Task 1.2: Update pyproject.toml scripts

**Files:**
- Modify: `pyproject.toml`

**Step 1: Check current scripts section**

Run: `grep -A5 "\[project.scripts\]" pyproject.toml || echo "No scripts section"`

**Step 2: Add/update scripts section**

Add or update in `pyproject.toml`:

```toml
[project.scripts]
hrp = "hrp.cli:main"
```

**Step 3: Reinstall package**

Run: `pip install -e .`
Expected: Successfully installed hrp

**Step 4: Verify CLI works**

Run: `hrp --help 2>&1 | head -3`
Expected: Output showing CLI usage

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add hrp CLI script to pyproject.toml"
```

---

## Phase 2: Align Versioning

### Task 2.1: Dynamic version from metadata

**Files:**
- Modify: `hrp/__init__.py`
- Test: `tests/test_version.py`

**Step 1: Write the failing test**

```python
# tests/test_version.py
"""Test version alignment."""

def test_version_is_string():
    """Version should be a string."""
    import hrp
    assert isinstance(hrp.__version__, str)


def test_version_matches_pyproject():
    """Version should match pyproject.toml."""
    import hrp
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    expected = data["project"]["version"]
    assert hrp.__version__ == expected
```

**Step 2: Run test to verify current state**

Run: `pytest tests/test_version.py -v`
Expected: May pass or fail depending on current implementation

**Step 3: Update hrp/__init__.py**

```python
# hrp/__init__.py
"""HRP - Hedgefund Research Platform."""

try:
    from importlib.metadata import version
    __version__ = version("hrp")
except Exception:
    __version__ = "0.0.0"  # Fallback for development
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_version.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/__init__.py tests/test_version.py
git commit -m "feat: align version with pyproject.toml metadata"
```

---

## Phase 3: Harden Config + Secrets

### Task 3.1: Create startup validation module

**Files:**
- Create: `hrp/utils/startup.py`
- Test: `tests/test_utils/test_startup.py`

**Step 1: Write the failing test**

```python
# tests/test_utils/test_startup.py
"""Test startup validation."""

import os
import pytest


class TestValidateStartup:
    """Tests for validate_startup function."""

    def test_returns_empty_list_in_development(self, monkeypatch):
        """No errors in development mode."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "development")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        assert errors == []

    def test_requires_anthropic_key_in_production(self, monkeypatch):
        """Production requires ANTHROPIC_API_KEY."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        assert any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_no_errors_when_secrets_present(self, monkeypatch):
        """No errors when required secrets are set."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from hrp.utils.startup import validate_startup
        errors = validate_startup()
        assert errors == []


class TestFailFastStartup:
    """Tests for fail_fast_startup function."""

    def test_raises_on_missing_secrets(self, monkeypatch):
        """Should raise RuntimeError when secrets missing in production."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "production")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from hrp.utils.startup import fail_fast_startup
        with pytest.raises(RuntimeError, match="Startup validation failed"):
            fail_fast_startup()

    def test_passes_in_development(self, monkeypatch):
        """Should not raise in development."""
        monkeypatch.setenv("HRP_ENVIRONMENT", "development")

        from hrp.utils.startup import fail_fast_startup
        fail_fast_startup()  # Should not raise
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils/test_startup.py -v`
Expected: FAIL with "No module named 'hrp.utils.startup'"

**Step 3: Write minimal implementation**

```python
# hrp/utils/startup.py
"""Startup validation for HRP.

Provides fail-fast validation of required configuration and secrets.
"""

from __future__ import annotations

import os
from loguru import logger


# Required secrets by environment
REQUIRED_PRODUCTION_SECRETS = [
    "ANTHROPIC_API_KEY",
]

OPTIONAL_SECRETS = [
    "POLYGON_API_KEY",
    "SIMFIN_API_KEY",
    "RESEND_API_KEY",
]


def validate_startup() -> list[str]:
    """
    Validate all required config at startup.

    Returns:
        List of error messages. Empty if all valid.
    """
    errors = []
    env = os.getenv("HRP_ENVIRONMENT", "development")

    if env == "production":
        for key in REQUIRED_PRODUCTION_SECRETS:
            if not os.getenv(key):
                errors.append(f"Missing required secret: {key}")

    return errors


def fail_fast_startup() -> None:
    """
    Validate startup and raise if invalid.

    Call this at application entry points (API, dashboard, CLI).

    Raises:
        RuntimeError: If required configuration is missing.
    """
    errors = validate_startup()
    if errors:
        error_msg = "Startup validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.debug("Startup validation passed")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils/test_startup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/utils/startup.py tests/test_utils/test_startup.py
git commit -m "feat: add startup validation with fail-fast for production secrets"
```

---

### Task 3.2: Add secret filtering to logging

**Files:**
- Create: `hrp/utils/log_filter.py`
- Test: `tests/test_utils/test_log_filter.py`

**Step 1: Write the failing test**

```python
# tests/test_utils/test_log_filter.py
"""Test secret filtering in logs."""


def test_filter_secrets_masks_api_keys():
    """API keys should be masked."""
    from hrp.utils.log_filter import filter_secrets

    text = "Using ANTHROPIC_API_KEY=sk-ant-abc123 for requests"
    result = filter_secrets(text)
    assert "sk-ant-abc123" not in result
    assert "***" in result


def test_filter_secrets_masks_passwords():
    """Passwords should be masked."""
    from hrp.utils.log_filter import filter_secrets

    text = "password=super_secret_123"
    result = filter_secrets(text)
    assert "super_secret_123" not in result


def test_filter_secrets_preserves_safe_text():
    """Non-secret text should be preserved."""
    from hrp.utils.log_filter import filter_secrets

    text = "Processing 100 records for AAPL"
    result = filter_secrets(text)
    assert result == text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils/test_log_filter.py -v`
Expected: FAIL with "No module named 'hrp.utils.log_filter'"

**Step 3: Write minimal implementation**

```python
# hrp/utils/log_filter.py
"""Secret filtering for log messages.

Prevents accidental logging of API keys, passwords, and tokens.
"""

from __future__ import annotations

import re

# Patterns to mask in log messages
SECRET_PATTERNS = [
    # API keys with values
    (r'(ANTHROPIC_API_KEY|POLYGON_API_KEY|SIMFIN_API_KEY|RESEND_API_KEY)\s*[=:]\s*\S+', r'\1=***'),
    # Generic API key patterns
    (r'(api[_-]?key|apikey)\s*[=:]\s*\S+', r'\1=***', re.IGNORECASE),
    # Password patterns
    (r'(password|passwd|pwd)\s*[=:]\s*\S+', r'\1=***', re.IGNORECASE),
    # Token patterns
    (r'(token|secret|credential)\s*[=:]\s*\S+', r'\1=***', re.IGNORECASE),
    # Bearer tokens
    (r'Bearer\s+\S+', 'Bearer ***'),
    # sk-ant-* pattern (Anthropic keys)
    (r'sk-ant-[a-zA-Z0-9-]+', '***'),
]


def filter_secrets(text: str) -> str:
    """
    Mask secrets in text.

    Args:
        text: Input text that may contain secrets

    Returns:
        Text with secrets masked as ***
    """
    result = text
    for pattern in SECRET_PATTERNS:
        if len(pattern) == 3:
            regex, replacement, flags = pattern
            result = re.sub(regex, replacement, result, flags=flags)
        else:
            regex, replacement = pattern
            result = re.sub(regex, replacement, result)
    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils/test_log_filter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/utils/log_filter.py tests/test_utils/test_log_filter.py
git commit -m "feat: add secret filtering for log messages"
```

---

## Phase 4: HTTP Health Server

### Task 4.1: Add FastAPI dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies to pyproject.toml**

Add to `[project.dependencies]` or `[project.optional-dependencies.ops]`:

```toml
[project.optional-dependencies]
ops = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "prometheus-client>=0.19.0",
]
```

**Step 2: Install dependencies**

Run: `pip install -e ".[ops]"`
Expected: Successfully installed fastapi, uvicorn, prometheus-client

**Step 3: Verify imports work**

Run: `python -c "import fastapi; import uvicorn; import prometheus_client; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add FastAPI/uvicorn/prometheus-client dependencies for ops"
```

---

### Task 4.2: Create ops module with health endpoint

**Files:**
- Create: `hrp/ops/__init__.py`
- Create: `hrp/ops/server.py`
- Test: `tests/test_ops/test_server.py`

**Step 1: Write the failing test**

```python
# tests/test_ops/test_server.py
"""Test ops server endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for ops server."""
    from hrp.ops.server import create_app
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_timestamp(self, client):
        """Health endpoint should include timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ops/test_server.py -v`
Expected: FAIL with "No module named 'hrp.ops'"

**Step 3: Write minimal implementation**

```python
# hrp/ops/__init__.py
"""HRP Ops module - health endpoints and metrics."""

from hrp.ops.server import create_app, run_server

__all__ = ["create_app", "run_server"]
```

```python
# hrp/ops/server.py
"""FastAPI ops server with health endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create FastAPI application for ops endpoints."""
    app = FastAPI(
        title="HRP Ops",
        description="Health and metrics endpoints for HRP",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness probe - returns 200 if API is responsive."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the ops server with uvicorn."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ops/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ops/__init__.py hrp/ops/server.py tests/test_ops/test_server.py
git commit -m "feat: add ops server with /health endpoint"
```

---

### Task 4.3: Add /ready endpoint

**Files:**
- Modify: `hrp/ops/server.py`
- Test: `tests/test_ops/test_server.py`

**Step 1: Add tests for /ready endpoint**

Add to `tests/test_ops/test_server.py`:

```python
class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_200_when_healthy(self, client, monkeypatch):
        """Ready endpoint should return 200 when system is healthy."""
        # Mock healthy system
        monkeypatch.setattr(
            "hrp.ops.server.check_system_ready",
            lambda: (True, {})
        )
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_returns_503_when_unhealthy(self, client, monkeypatch):
        """Ready endpoint should return 503 when system is unhealthy."""
        monkeypatch.setattr(
            "hrp.ops.server.check_system_ready",
            lambda: (False, {"database": "disconnected"})
        )
        response = client.get("/ready")
        assert response.status_code == 503
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ops/test_server.py::TestReadyEndpoint -v`
Expected: FAIL

**Step 3: Implement /ready endpoint**

Update `hrp/ops/server.py`:

```python
# hrp/ops/server.py
"""FastAPI ops server with health endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, Response


def check_system_ready() -> tuple[bool, dict]:
    """
    Check if system is ready to serve requests.

    Returns:
        Tuple of (is_ready, details_dict)
    """
    details = {}
    is_ready = True

    try:
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI(read_only=True)
        health = api.health_check()
        api.close()

        details["database"] = health.get("database", "unknown")
        details["api"] = health.get("api", "unknown")

        if health.get("database") != "ok":
            is_ready = False
    except Exception as e:
        details["error"] = str(e)
        is_ready = False

    return is_ready, details


def create_app() -> FastAPI:
    """Create FastAPI application for ops endpoints."""
    app = FastAPI(
        title="HRP Ops",
        description="Health and metrics endpoints for HRP",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness probe - returns 200 if API is responsive."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/ready")
    def ready(response: Response):
        """Readiness probe - returns 200 if system is ready, 503 otherwise."""
        is_ready, details = check_system_ready()

        result = {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now().isoformat(),
            "checks": details,
        }

        if not is_ready:
            response.status_code = 503

        return result

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the ops server with uvicorn."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ops/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hrp/ops/server.py tests/test_ops/test_server.py
git commit -m "feat: add /ready endpoint with system health check"
```

---

### Task 4.4: Add CLI entry point for ops server

**Files:**
- Create: `hrp/ops/__main__.py`

**Step 1: Create __main__.py**

```python
# hrp/ops/__main__.py
"""Run HRP ops server: python -m hrp.ops"""

import argparse
import os

from hrp.ops.server import run_server


def main() -> None:
    """Main entry point for ops server."""
    parser = argparse.ArgumentParser(description="HRP Ops Server")
    parser.add_argument(
        "--host",
        default=os.getenv("HRP_OPS_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HRP_OPS_PORT", "8080")),
        help="Bind port (default: 8080)",
    )
    args = parser.parse_args()

    print(f"Starting HRP Ops Server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs**

Run: `python -m hrp.ops --help`
Expected: Shows help with --host and --port options

**Step 3: Commit**

```bash
git add hrp/ops/__main__.py
git commit -m "feat: add CLI entry point for ops server"
```

---

## Phase 5: Configurable Alert Thresholds

### Task 5.1: Create thresholds dataclass

**Files:**
- Create: `hrp/ops/thresholds.py`
- Test: `tests/test_ops/test_thresholds.py`

**Step 1: Write the failing test**

```python
# tests/test_ops/test_thresholds.py
"""Test configurable thresholds."""

import pytest


class TestOpsThresholds:
    """Tests for OpsThresholds dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        from hrp.ops.thresholds import OpsThresholds

        t = OpsThresholds()
        assert t.health_score_warning == 90.0
        assert t.health_score_critical == 70.0
        assert t.freshness_warning_days == 3
        assert t.freshness_critical_days == 5

    def test_custom_values(self):
        """Should accept custom values."""
        from hrp.ops.thresholds import OpsThresholds

        t = OpsThresholds(health_score_warning=85.0)
        assert t.health_score_warning == 85.0


class TestLoadThresholds:
    """Tests for load_thresholds function."""

    def test_loads_defaults(self):
        """Should load defaults when no config exists."""
        from hrp.ops.thresholds import load_thresholds

        t = load_thresholds(config_path="/nonexistent/path.yaml")
        assert t.health_score_warning == 90.0

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("HRP_THRESHOLD_HEALTH_SCORE_WARNING", "85")

        from hrp.ops.thresholds import load_thresholds
        t = load_thresholds(config_path="/nonexistent/path.yaml")
        assert t.health_score_warning == 85.0

    def test_yaml_config(self, tmp_path):
        """YAML config should override defaults."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("health_score_warning: 80.0\n")

        from hrp.ops.thresholds import load_thresholds
        t = load_thresholds(config_path=str(config_file))
        assert t.health_score_warning == 80.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ops/test_thresholds.py -v`
Expected: FAIL with "cannot import name 'OpsThresholds'"

**Step 3: Write minimal implementation**

```python
# hrp/ops/thresholds.py
"""Configurable alert thresholds for HRP ops.

Thresholds can be configured via:
1. Environment variables (HRP_THRESHOLD_*)
2. YAML config file (~~/hrp-data/config/thresholds.yaml)
3. Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from loguru import logger


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


def load_thresholds(config_path: str | None = None) -> OpsThresholds:
    """
    Load thresholds with priority: Environment > YAML > Defaults.

    Args:
        config_path: Path to YAML config. Defaults to ~/hrp-data/config/thresholds.yaml

    Returns:
        OpsThresholds instance
    """
    thresholds = OpsThresholds()

    # Load from YAML if exists
    if config_path is None:
        config_path = os.path.expanduser("~/hrp-data/config/thresholds.yaml")

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}

            for key, value in yaml_config.items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)

            logger.debug(f"Loaded thresholds from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load thresholds from {config_path}: {e}")

    # Apply environment variable overrides
    for field in fields(thresholds):
        env_key = f"HRP_THRESHOLD_{field.name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                field_type = field.type
                if field_type == float:
                    setattr(thresholds, field.name, float(env_value))
                elif field_type == int:
                    setattr(thresholds, field.name, int(env_value))
                else:
                    setattr(thresholds, field.name, env_value)
                logger.debug(f"Threshold override: {field.name}={env_value}")
            except ValueError as e:
                logger.warning(f"Invalid env value for {env_key}: {e}")

    return thresholds
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ops/test_thresholds.py -v`
Expected: PASS

**Step 5: Update __init__.py and commit**

```python
# Update hrp/ops/__init__.py
"""HRP Ops module - health endpoints and metrics."""

from hrp.ops.server import create_app, run_server
from hrp.ops.thresholds import OpsThresholds, load_thresholds

__all__ = ["create_app", "run_server", "OpsThresholds", "load_thresholds"]
```

```bash
git add hrp/ops/thresholds.py hrp/ops/__init__.py tests/test_ops/test_thresholds.py
git commit -m "feat: add configurable alert thresholds with YAML and env support"
```

---

## Remaining Phases (Summary)

### Phase 6: Prometheus Metrics + Ops Dashboard
- Task 6.1: Add `/metrics` endpoint with prometheus-client
- Task 6.2: Create `hrp/ops/metrics.py` with MetricsCollector
- Task 6.3: Create `hrp/dashboard/pages/ops.py`
- Task 6.4: Add Ops page routing in `hrp/dashboard/app.py`
- Task 6.5: Create launchd plist

### Phase 7: DB/Process Concurrency Hardening
- Task 7.1: Create `hrp/data/connection_pool.py` with retry/backoff
- Task 7.2: Create `hrp/utils/locks.py` for job locking
- Task 7.3: Enable DuckDB WAL mode
- Task 7.4: Audit and set read_only=True where appropriate

### Phase 8: Pipeline Integration Tests
- Task 8.1: Create `tests/fixtures/integration_db.py` with seed data
- Task 8.2: Create `tests/integration/test_golden_path.py`
- Task 8.3: Register fixtures in `conftest.py`

---

## Verification

After completing all phases:

```bash
# Phase 0: Artifacts not in git
git status | grep -E "mlflow.db|coverage.xml" && echo "FAIL" || echo "PASS"

# Phase 1: CLI works
hrp --help

# Phase 2: Version aligned
python -c "import hrp; print(hrp.__version__)"

# Phase 3: Startup validation
HRP_ENVIRONMENT=development python -c "from hrp.utils.startup import fail_fast_startup; fail_fast_startup(); print('PASS')"

# Phase 4-6: Ops server
python -m hrp.ops --port 8080 &
sleep 2
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
kill %1

# Phase 8: Integration tests
pytest tests/integration/ -v
```

---

## Rollout Checklist

- [ ] Phase 0: `.gitignore` updated, artifacts removed from git
- [ ] Phase 1: `hrp` CLI command works
- [ ] Phase 2: Version from pyproject.toml
- [ ] Phase 3: Startup validation with fail-fast
- [ ] Phase 3: Secret filtering in logs
- [ ] Phase 4: `/health` and `/ready` endpoints
- [ ] Phase 5: Configurable thresholds
- [ ] Phase 6: `/metrics` endpoint + Ops dashboard
- [ ] Phase 7: Connection pooling with retry
- [ ] Phase 8: Integration tests pass
- [ ] Docs updated: CLAUDE.md, cookbook.md, Project-status.md, changelog.md
