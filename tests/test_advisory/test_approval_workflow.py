"""Tests for the approval workflow."""

from datetime import date, datetime, timedelta

import pytest

from hrp.advisory.approval_workflow import ApprovalResult, ApprovalWorkflow


@pytest.fixture
def approval_db(test_db):
    """Test database with recommendations in various states."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Pending approval recommendations
    for i, (sym, signal) in enumerate([
        ("AAPL", 0.8), ("MSFT", 0.6), ("GOOGL", 0.5),
    ]):
        db.execute(
            "INSERT INTO recommendations "
            "(recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, target_price, stop_price, position_pct, "
            "time_horizon_days, status, model_name, batch_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                f"REC-PEND-{i+1:03d}", sym, "BUY", "HIGH", signal,
                150.0 + i * 10, 170.0 + i * 10, 140.0 + i * 10, 0.08,
                20, "pending_approval", "test_model", "BATCH-PEND",
                datetime(2024, 1, 10),
            ],
        )

    # Active recommendation (already approved)
    db.execute(
        "INSERT INTO recommendations "
        "(recommendation_id, symbol, action, confidence, signal_strength, "
        "entry_price, target_price, stop_price, position_pct, "
        "time_horizon_days, status, model_name, batch_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "REC-ACTIVE-001", "NVDA", "BUY", "MEDIUM", 0.4,
            200.0, 220.0, 185.0, 0.05,
            20, "active", "test_model", "BATCH-ACT",
            datetime(2024, 1, 5),
        ],
    )

    # Closed recommendation (should not be approvable/rejectable)
    db.execute(
        "INSERT INTO recommendations "
        "(recommendation_id, symbol, action, confidence, signal_strength, "
        "entry_price, target_price, stop_price, position_pct, "
        "time_horizon_days, status, model_name, batch_id, created_at, "
        "closed_at, close_price, realized_return) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "REC-CLOSED-001", "AMZN", "BUY", "HIGH", 0.7,
            170.0, 190.0, 160.0, 0.06,
            20, "closed_profit", "test_model", "BATCH-OLD",
            datetime(2024, 1, 1), datetime(2024, 1, 15), 185.0, 0.088,
        ],
    )

    yield test_db


@pytest.fixture
def workflow(approval_db):
    """ApprovalWorkflow instance with test database."""
    from hrp.api.platform import PlatformAPI
    return ApprovalWorkflow(api=PlatformAPI(), dry_run=True)


class TestApprovalWorkflow:
    def test_get_pending(self, workflow):
        pending = workflow.get_pending()
        assert len(pending) == 3
        assert set(pending["symbol"].tolist()) == {"AAPL", "MSFT", "GOOGL"}

    def test_approve_pending(self, workflow):
        result = workflow.approve("REC-PEND-001", actor="user")
        assert isinstance(result, ApprovalResult)
        assert result.action == "approved"
        assert "dry-run" in result.message

        # Verify status changed to active
        rec = workflow.api.get_recommendation_by_id("REC-PEND-001")
        assert rec["status"] == "active"

    def test_approve_already_active(self, workflow):
        # Active recs can also be "approved" (idempotent)
        result = workflow.approve("REC-ACTIVE-001", actor="user")
        assert result.action == "approved"

    def test_approve_closed_rejected(self, workflow):
        result = workflow.approve("REC-CLOSED-001", actor="user")
        assert result.action == "rejected"
        assert "closed_profit" in result.message

    def test_approve_nonexistent(self, workflow):
        result = workflow.approve("REC-NONEXISTENT", actor="user")
        assert result.action == "rejected"
        assert "not found" in result.message.lower()

    def test_agent_cannot_approve(self, workflow):
        result = workflow.approve("REC-PEND-001", actor="agent:cio-agent")
        assert result.action == "rejected"
        assert "Permission denied" in result.message

    def test_reject_pending(self, workflow):
        result = workflow.reject("REC-PEND-002", actor="user", reason="Not convinced")
        assert result.action == "cancelled"
        assert "Not convinced" in result.message

        # Verify status changed to cancelled
        rec = workflow.api.get_recommendation_by_id("REC-PEND-002")
        assert rec["status"] == "cancelled"
        assert rec["closed_at"] is not None

    def test_reject_closed_fails(self, workflow):
        result = workflow.reject("REC-CLOSED-001", actor="user")
        assert result.action == "rejected"
        assert "closed_profit" in result.message

    def test_approve_all(self, workflow):
        results = workflow.approve_all(actor="user")
        assert len(results) == 3
        approved = [r for r in results if r.action == "approved"]
        assert len(approved) == 3

        # All should now be active
        pending = workflow.get_pending()
        assert pending.empty

    def test_submit_for_approval(self, workflow):
        # Move the active rec to pending_approval
        count = workflow.submit_for_approval(["REC-ACTIVE-001"])
        assert count == 1

        rec = workflow.api.get_recommendation_by_id("REC-ACTIVE-001")
        assert rec["status"] == "pending_approval"

    def test_submit_for_approval_ignores_closed(self, workflow):
        count = workflow.submit_for_approval(["REC-CLOSED-001"])
        assert count == 0

    def test_submit_for_approval_ignores_nonexistent(self, workflow):
        count = workflow.submit_for_approval(["REC-NONEXISTENT"])
        assert count == 0


class TestApprovalWorkflowAPI:
    """Test the PlatformAPI approval methods."""

    def test_api_approve(self, approval_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        result = api.approve_recommendation("REC-PEND-001", actor="user", dry_run=True)
        assert result["action"] == "approved"
        assert result["recommendation_id"] == "REC-PEND-001"

    def test_api_reject(self, approval_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        result = api.reject_recommendation("REC-PEND-002", actor="user", reason="test")
        assert result["action"] == "cancelled"

    def test_api_approve_all(self, approval_db):
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI()
        results = api.approve_all_recommendations(actor="user", dry_run=True)
        assert len(results) == 3
        assert all(r["action"] == "approved" for r in results)
