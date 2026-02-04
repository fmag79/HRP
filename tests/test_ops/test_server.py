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


class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_200_when_healthy(self, client, monkeypatch):
        """Ready endpoint should return 200 when system is healthy."""
        # Mock healthy system
        monkeypatch.setattr("hrp.ops.server.check_system_ready", lambda: (True, {}))
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_returns_503_when_unhealthy(self, client, monkeypatch):
        """Ready endpoint should return 503 when system is unhealthy."""
        monkeypatch.setattr(
            "hrp.ops.server.check_system_ready", lambda: (False, {"database": "disconnected"})
        )
        response = client.get("/ready")
        assert response.status_code == 503

    def test_ready_returns_status_ready_when_healthy(self, client, monkeypatch):
        """Ready endpoint should return status ready when healthy."""
        monkeypatch.setattr("hrp.ops.server.check_system_ready", lambda: (True, {}))
        response = client.get("/ready")
        data = response.json()
        assert data["status"] == "ready"

    def test_ready_returns_status_not_ready_when_unhealthy(self, client, monkeypatch):
        """Ready endpoint should return status not_ready when unhealthy."""
        monkeypatch.setattr(
            "hrp.ops.server.check_system_ready", lambda: (False, {"database": "disconnected"})
        )
        response = client.get("/ready")
        data = response.json()
        assert data["status"] == "not_ready"

    def test_ready_includes_timestamp(self, client, monkeypatch):
        """Ready endpoint should include timestamp."""
        monkeypatch.setattr("hrp.ops.server.check_system_ready", lambda: (True, {}))
        response = client.get("/ready")
        data = response.json()
        assert "timestamp" in data

    def test_ready_includes_checks_details(self, client, monkeypatch):
        """Ready endpoint should include checks details."""
        monkeypatch.setattr(
            "hrp.ops.server.check_system_ready",
            lambda: (True, {"database": "ok", "api": "ok"}),
        )
        response = client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert data["checks"]["database"] == "ok"
        assert data["checks"]["api"] == "ok"
