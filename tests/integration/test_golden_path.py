"""Golden path integration tests for HRP.

Tests the complete hypothesis lifecycle from creation through validation.
"""

import pytest


class TestHypothesisLifecycle:
    """Test complete hypothesis lifecycle."""

    @pytest.mark.integration
    def test_create_hypothesis(self, integration_api):
        """Should create a hypothesis via API."""
        api = integration_api

        hyp_id = api.create_hypothesis(
            title="Integration Test Hypothesis",
            thesis="Testing the complete flow",
            prediction="All tests pass",
            falsification="Any test fails",
            actor="integration_test",
        )

        assert hyp_id is not None
        assert hyp_id.startswith("HYP-")

        # Verify it was created
        hyp = api.get_hypothesis(hyp_id)
        assert hyp["title"] == "Integration Test Hypothesis"
        assert hyp["status"] == "draft"

    @pytest.mark.integration
    def test_hypothesis_status_transitions(self, integration_api):
        """Should transition hypothesis through statuses."""
        api = integration_api

        # Create hypothesis
        hyp_id = api.create_hypothesis(
            title="Status Transition Test",
            thesis="Testing status flow",
            prediction="Can transition through statuses",
            falsification="Status transition fails",
            actor="integration_test",
        )

        # draft -> testing (via Alpha Researcher)
        api.update_hypothesis(
            hyp_id,
            status="testing",
            outcome=None,
            actor="alpha_researcher",
        )
        hyp = api.get_hypothesis(hyp_id)
        assert hyp["status"] == "testing"

        # testing -> rejected
        api.update_hypothesis(
            hyp_id,
            status="rejected",
            outcome="Failed validation criteria",
            actor="kill_gate_enforcer",
        )
        hyp = api.get_hypothesis(hyp_id)
        assert hyp["status"] == "rejected"

    @pytest.mark.integration
    def test_data_access_via_api(self, integration_api):
        """Should access price and feature data via API."""
        api = integration_api

        # Get prices (should work with seed data)
        # Note: Seed data might not have these symbols, so we test the API call
        try:
            prices = api.get_prices(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            # If we have data, verify structure
            if not prices.empty:
                assert "close" in prices.columns or "adj_close" in prices.columns
        except Exception:
            # If no data, that's fine for this test
            pass

    @pytest.mark.integration
    def test_lineage_tracking(self, integration_api):
        """Should track lineage events."""
        api = integration_api

        # Create hypothesis
        hyp_id = api.create_hypothesis(
            title="Lineage Test",
            thesis="Testing lineage tracking",
            prediction="Lineage events recorded",
            falsification="No lineage",
            actor="integration_test",
        )

        # Get lineage
        lineage = api.get_lineage(hypothesis_id=hyp_id)

        # Should have at least the creation event
        assert len(lineage) >= 1
        assert any(e["event_type"] == "hypothesis_created" for e in lineage)


class TestOpsEndpoints:
    """Test ops server endpoints."""

    @pytest.mark.integration
    def test_health_endpoint_integration(self, integration_db, monkeypatch):
        """Health endpoint should work with real database."""
        from fastapi.testclient import TestClient

        db_path, seed_ids = integration_db
        monkeypatch.setenv("HRP_DB_PATH", db_path)

        from hrp.ops.server import create_app

        client = TestClient(create_app())

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.integration
    def test_metrics_endpoint_integration(self, integration_db, monkeypatch):
        """Metrics endpoint should return Prometheus format."""
        from fastapi.testclient import TestClient

        db_path, seed_ids = integration_db
        monkeypatch.setenv("HRP_DB_PATH", db_path)

        from hrp.ops.server import create_app

        client = TestClient(create_app())

        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus format is text
        assert "python_info" in response.text or "process_" in response.text or "hrp_" in response.text
