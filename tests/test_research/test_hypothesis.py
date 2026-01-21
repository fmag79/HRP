"""
Comprehensive tests for the HRP Hypothesis Registry.

Tests cover:
- Hypothesis ID generation
- CRUD operations (create, read, update, delete)
- Lifecycle and state transitions
- Experiment linking
- Validation requirements
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables
from hrp.research.hypothesis import (
    VALID_TRANSITIONS,
    create_hypothesis,
    delete_hypothesis,
    get_experiment_links,
    get_experiments,
    get_hypothesis,
    get_next_hypothesis_id,
    link_experiment,
    list_hypotheses,
    update_hypothesis,
    validate_hypothesis_status,
)


@pytest.fixture
def test_db():
    """
    Create a temporary DuckDB database for testing with schema initialized.

    Yields the database path and ensures cleanup after tests.
    """
    # Reset any existing singleton instance
    DatabaseManager.reset()

    # Create temp directory for test database
    temp_dir = tempfile.mkdtemp()
    actual_db_path = os.path.join(temp_dir, "hrp.duckdb")

    # Set environment variable to use test database
    old_env = os.environ.get("HRP_DATA_DIR")
    os.environ["HRP_DATA_DIR"] = temp_dir

    # Initialize the database manager with the test path
    DatabaseManager.reset()
    db = get_db(actual_db_path)

    # Create all tables
    create_tables(actual_db_path)

    yield actual_db_path

    # Cleanup
    DatabaseManager.reset()
    if old_env is not None:
        os.environ["HRP_DATA_DIR"] = old_env
    elif "HRP_DATA_DIR" in os.environ:
        del os.environ["HRP_DATA_DIR"]

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_hypothesis_data():
    """Standard hypothesis data for testing."""
    return {
        "title": "Momentum predicts returns",
        "thesis": "Stocks with high 12-month returns continue outperforming",
        "prediction": "Top decile momentum > SPY by 3% annually",
        "falsification": "Sharpe < SPY or p-value > 0.05",
        "actor": "user",
    }


@pytest.fixture
def created_hypothesis(test_db, sample_hypothesis_data):
    """Create and return a hypothesis for testing."""
    hyp_id = create_hypothesis(**sample_hypothesis_data)
    return hyp_id


class TestHypothesisIDGeneration:
    """Tests for get_next_hypothesis_id()."""

    def test_first_hypothesis_id(self, test_db):
        """First hypothesis in a year should have sequence 001."""
        hyp_id = get_next_hypothesis_id()
        year = datetime.now().year
        assert hyp_id == f"HYP-{year}-001"

    def test_sequential_ids(self, test_db):
        """Subsequent hypotheses should have incrementing sequence numbers."""
        year = datetime.now().year

        # Create multiple hypotheses
        id1 = create_hypothesis(
            title="Test 1",
            thesis="Thesis 1",
            prediction="Prediction 1",
            falsification="Falsification 1",
            actor="user",
        )
        id2 = create_hypothesis(
            title="Test 2",
            thesis="Thesis 2",
            prediction="Prediction 2",
            falsification="Falsification 2",
            actor="user",
        )
        id3 = create_hypothesis(
            title="Test 3",
            thesis="Thesis 3",
            prediction="Prediction 3",
            falsification="Falsification 3",
            actor="user",
        )

        assert id1 == f"HYP-{year}-001"
        assert id2 == f"HYP-{year}-002"
        assert id3 == f"HYP-{year}-003"

    def test_id_format(self, test_db):
        """ID should follow format HYP-YYYY-NNN."""
        hyp_id = get_next_hypothesis_id()
        parts = hyp_id.split("-")

        assert len(parts) == 3
        assert parts[0] == "HYP"
        assert len(parts[1]) == 4  # Year
        assert parts[1].isdigit()
        assert len(parts[2]) == 3  # Zero-padded sequence
        assert parts[2].isdigit()

    def test_sequence_continues_after_deletion(self, test_db):
        """Sequence should continue even after hypotheses are deleted."""
        year = datetime.now().year

        # Create and delete a hypothesis
        id1 = create_hypothesis(
            title="Test 1",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )
        delete_hypothesis(id1)

        # Next ID should be 002, not 001
        id2 = create_hypothesis(
            title="Test 2",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )

        assert id2 == f"HYP-{year}-002"


class TestHypothesisCRUD:
    """Tests for create, read, update, delete operations."""

    def test_create_hypothesis(self, test_db, sample_hypothesis_data):
        """Creating a hypothesis should return a valid ID and store data."""
        hyp_id = create_hypothesis(**sample_hypothesis_data)

        assert hyp_id is not None
        assert hyp_id.startswith("HYP-")

        hyp = get_hypothesis(hyp_id)
        assert hyp is not None
        assert hyp["title"] == sample_hypothesis_data["title"]
        assert hyp["thesis"] == sample_hypothesis_data["thesis"]
        assert hyp["testable_prediction"] == sample_hypothesis_data["prediction"]
        assert hyp["falsification_criteria"] == sample_hypothesis_data["falsification"]
        assert hyp["status"] == "draft"
        assert hyp["created_by"] == sample_hypothesis_data["actor"]

    def test_create_with_agent_actor(self, test_db):
        """Creating a hypothesis with an agent actor."""
        hyp_id = create_hypothesis(
            title="Agent hypothesis",
            thesis="Agent thesis",
            prediction="Agent prediction",
            falsification="Agent falsification",
            actor="agent:discovery",
        )

        hyp = get_hypothesis(hyp_id)
        assert hyp["created_by"] == "agent:discovery"

    def test_get_hypothesis(self, created_hypothesis):
        """Getting a hypothesis should return all fields."""
        hyp = get_hypothesis(created_hypothesis)

        assert hyp is not None
        assert "hypothesis_id" in hyp
        assert "title" in hyp
        assert "thesis" in hyp
        assert "testable_prediction" in hyp
        assert "falsification_criteria" in hyp
        assert "status" in hyp
        assert "created_at" in hyp
        assert "created_by" in hyp
        assert "updated_at" in hyp
        assert "outcome" in hyp
        assert "confidence_score" in hyp

    def test_get_nonexistent_hypothesis(self, test_db):
        """Getting a non-existent hypothesis should return None."""
        hyp = get_hypothesis("HYP-9999-999")
        assert hyp is None

    def test_list_hypotheses(self, test_db):
        """Listing hypotheses should return all non-deleted hypotheses."""
        # Create multiple hypotheses
        create_hypothesis(
            title="Test 1",
            thesis="Thesis 1",
            prediction="Prediction 1",
            falsification="Falsification 1",
            actor="user",
        )
        create_hypothesis(
            title="Test 2",
            thesis="Thesis 2",
            prediction="Prediction 2",
            falsification="Falsification 2",
            actor="agent:discovery",
        )

        hypotheses = list_hypotheses()

        assert len(hypotheses) == 2
        titles = [h["title"] for h in hypotheses]
        assert "Test 1" in titles
        assert "Test 2" in titles

    def test_list_hypotheses_filter_by_status(self, test_db):
        """Filtering by status should only return matching hypotheses."""
        id1 = create_hypothesis(
            title="Draft Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )
        id2 = create_hypothesis(
            title="Testing Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )

        # Move id2 to testing
        update_hypothesis(id2, status="testing")

        draft_list = list_hypotheses(status="draft")
        testing_list = list_hypotheses(status="testing")

        assert len(draft_list) == 1
        assert draft_list[0]["title"] == "Draft Hyp"
        assert len(testing_list) == 1
        assert testing_list[0]["title"] == "Testing Hyp"

    def test_list_hypotheses_filter_by_actor(self, test_db):
        """Filtering by actor should only return matching hypotheses."""
        create_hypothesis(
            title="User Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )
        create_hypothesis(
            title="Agent Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="agent:discovery",
        )

        user_list = list_hypotheses(actor="user")
        agent_list = list_hypotheses(actor="agent:discovery")

        assert len(user_list) == 1
        assert user_list[0]["title"] == "User Hyp"
        assert len(agent_list) == 1
        assert agent_list[0]["title"] == "Agent Hyp"

    def test_list_hypotheses_excludes_deleted(self, test_db):
        """Deleted hypotheses should not appear in list."""
        id1 = create_hypothesis(
            title="Active Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )
        id2 = create_hypothesis(
            title="Deleted Hyp",
            thesis="Thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )

        delete_hypothesis(id2)

        hypotheses = list_hypotheses()

        assert len(hypotheses) == 1
        assert hypotheses[0]["title"] == "Active Hyp"

    def test_update_hypothesis_outcome(self, created_hypothesis):
        """Updating outcome should persist the change."""
        result = update_hypothesis(created_hypothesis, outcome="Hypothesis confirmed")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["outcome"] == "Hypothesis confirmed"
        assert hyp["updated_at"] is not None

    def test_update_hypothesis_confidence_score(self, created_hypothesis):
        """Updating confidence score should persist the change."""
        result = update_hypothesis(created_hypothesis, confidence_score=0.85)

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["confidence_score"] == 0.85

    def test_update_hypothesis_invalid_confidence_score(self, created_hypothesis):
        """Confidence score outside 0-1 range should raise ValueError."""
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            update_hypothesis(created_hypothesis, confidence_score=1.5)

        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            update_hypothesis(created_hypothesis, confidence_score=-0.1)

    def test_update_nonexistent_hypothesis(self, test_db):
        """Updating a non-existent hypothesis should return False."""
        result = update_hypothesis("HYP-9999-999", outcome="Test")
        assert result is False

    def test_update_no_changes(self, created_hypothesis):
        """Update with no parameters should return False."""
        result = update_hypothesis(created_hypothesis)
        assert result is False

    def test_delete_hypothesis(self, created_hypothesis):
        """Deleting a hypothesis should soft delete (set status to 'deleted')."""
        result = delete_hypothesis(created_hypothesis)

        assert result is True

        # Should still be retrievable
        hyp = get_hypothesis(created_hypothesis)
        assert hyp is not None
        assert hyp["status"] == "deleted"

    def test_delete_nonexistent_hypothesis(self, test_db):
        """Deleting a non-existent hypothesis should return False."""
        result = delete_hypothesis("HYP-9999-999")
        assert result is False

    def test_double_delete(self, created_hypothesis):
        """Deleting an already deleted hypothesis should still work."""
        delete_hypothesis(created_hypothesis)
        # Should not raise an error, just return True
        # Note: Current implementation allows this; the test documents behavior
        result = delete_hypothesis(created_hypothesis)
        # The second delete may return True or False depending on implementation
        # Just verify no exception is raised


class TestHypothesisLifecycle:
    """Tests for status transitions and lifecycle management."""

    def test_initial_status_is_draft(self, created_hypothesis):
        """Newly created hypothesis should have 'draft' status."""
        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "draft"

    def test_valid_transition_draft_to_testing(self, created_hypothesis):
        """Draft hypothesis should be able to transition to testing."""
        result = update_hypothesis(created_hypothesis, status="testing")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "testing"

    def test_valid_transition_draft_to_deleted(self, created_hypothesis):
        """Draft hypothesis should be able to transition to deleted."""
        result = update_hypothesis(created_hypothesis, status="deleted")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "deleted"

    def test_valid_transition_testing_to_draft(self, created_hypothesis):
        """Testing hypothesis can go back to draft (reopen)."""
        update_hypothesis(created_hypothesis, status="testing")
        result = update_hypothesis(created_hypothesis, status="draft")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "draft"

    def test_valid_transition_testing_to_rejected(self, created_hypothesis):
        """Testing hypothesis can be rejected."""
        update_hypothesis(created_hypothesis, status="testing")
        result = update_hypothesis(created_hypothesis, status="rejected")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "rejected"

    def test_valid_transition_rejected_to_draft(self, created_hypothesis):
        """Rejected hypothesis can be reopened as draft."""
        update_hypothesis(created_hypothesis, status="testing")
        update_hypothesis(created_hypothesis, status="rejected")
        result = update_hypothesis(created_hypothesis, status="draft")

        assert result is True

        hyp = get_hypothesis(created_hypothesis)
        assert hyp["status"] == "draft"

    def test_valid_transition_validated_to_deployed(self, test_db):
        """Validated hypothesis can be deployed."""
        hyp_id = create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )

        # Move to testing
        update_hypothesis(hyp_id, status="testing")

        # Link an experiment (required for validation)
        link_experiment(hyp_id, "exp-001")

        # Now validate
        update_hypothesis(hyp_id, status="validated")

        # Deploy
        result = update_hypothesis(hyp_id, status="deployed")

        assert result is True

        hyp = get_hypothesis(hyp_id)
        assert hyp["status"] == "deployed"

    def test_valid_transition_deployed_to_validated(self, test_db):
        """Deployed hypothesis can be undeployed back to validated."""
        hyp_id = create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )

        # Full lifecycle to deployed
        update_hypothesis(hyp_id, status="testing")
        link_experiment(hyp_id, "exp-001")
        update_hypothesis(hyp_id, status="validated")
        update_hypothesis(hyp_id, status="deployed")

        # Undeploy
        result = update_hypothesis(hyp_id, status="validated")

        assert result is True

        hyp = get_hypothesis(hyp_id)
        assert hyp["status"] == "validated"

    def test_invalid_transition_draft_to_validated(self, created_hypothesis):
        """Draft hypothesis cannot directly transition to validated."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            update_hypothesis(created_hypothesis, status="validated")

    def test_invalid_transition_draft_to_deployed(self, created_hypothesis):
        """Draft hypothesis cannot directly transition to deployed."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            update_hypothesis(created_hypothesis, status="deployed")

    def test_invalid_transition_draft_to_rejected(self, created_hypothesis):
        """Draft hypothesis cannot directly transition to rejected."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            update_hypothesis(created_hypothesis, status="rejected")

    def test_invalid_transition_from_deleted(self, created_hypothesis):
        """Deleted hypothesis cannot transition to any other status."""
        update_hypothesis(created_hypothesis, status="deleted")

        for status in ["draft", "testing", "validated", "rejected", "deployed"]:
            with pytest.raises(ValueError, match="Invalid status transition"):
                update_hypothesis(created_hypothesis, status=status)

    def test_cannot_validate_without_experiment(self, created_hypothesis):
        """Cannot transition to validated without a linked experiment."""
        update_hypothesis(created_hypothesis, status="testing")

        with pytest.raises(ValueError, match="Cannot validate hypothesis"):
            update_hypothesis(created_hypothesis, status="validated")

    def test_cannot_validate_from_draft(self, created_hypothesis):
        """Cannot validate directly from draft status."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            update_hypothesis(created_hypothesis, status="validated")

    def test_cannot_deploy_non_validated(self, created_hypothesis):
        """Cannot deploy a hypothesis that is not validated."""
        update_hypothesis(created_hypothesis, status="testing")

        with pytest.raises(ValueError, match="Invalid status transition"):
            update_hypothesis(created_hypothesis, status="deployed")

    def test_all_valid_transitions(self, test_db):
        """Verify all transitions defined in VALID_TRANSITIONS work correctly."""
        for from_status, to_statuses in VALID_TRANSITIONS.items():
            for to_status in to_statuses:
                # Create a fresh hypothesis for each transition test
                hyp_id = create_hypothesis(
                    title=f"Test {from_status} to {to_status}",
                    thesis="Test",
                    prediction="Test",
                    falsification="Test",
                    actor="user",
                )

                # Get to the from_status
                if from_status == "draft":
                    pass  # Already in draft
                elif from_status == "testing":
                    update_hypothesis(hyp_id, status="testing")
                elif from_status == "validated":
                    update_hypothesis(hyp_id, status="testing")
                    link_experiment(hyp_id, f"exp-{hyp_id}")
                    update_hypothesis(hyp_id, status="validated")
                elif from_status == "rejected":
                    update_hypothesis(hyp_id, status="testing")
                    update_hypothesis(hyp_id, status="rejected")
                elif from_status == "deployed":
                    update_hypothesis(hyp_id, status="testing")
                    link_experiment(hyp_id, f"exp-{hyp_id}")
                    update_hypothesis(hyp_id, status="validated")
                    update_hypothesis(hyp_id, status="deployed")
                elif from_status == "deleted":
                    update_hypothesis(hyp_id, status="deleted")

                # Handle special cases for validation requirements
                if to_status == "validated" and from_status == "testing":
                    # Need to link experiment first
                    link_experiment(hyp_id, f"exp-validate-{hyp_id}")

                # Now try the transition
                result = update_hypothesis(hyp_id, status=to_status)
                assert result is True, f"Transition {from_status} -> {to_status} failed"

                hyp = get_hypothesis(hyp_id)
                assert hyp["status"] == to_status, f"Status not updated for {from_status} -> {to_status}"


class TestExperimentLinking:
    """Tests for linking experiments to hypotheses."""

    def test_link_experiment(self, created_hypothesis):
        """Linking an experiment should succeed."""
        result = link_experiment(created_hypothesis, "exp-001")

        assert result is True

    def test_link_experiment_with_relationship(self, created_hypothesis):
        """Linking with a specific relationship should be stored."""
        link_experiment(created_hypothesis, "exp-001", relationship="primary")
        link_experiment(created_hypothesis, "exp-002", relationship="supporting")
        link_experiment(created_hypothesis, "exp-003", relationship="exploratory")

        links = get_experiment_links(created_hypothesis)

        relationships = {link["experiment_id"]: link["relationship"] for link in links}

        assert relationships["exp-001"] == "primary"
        assert relationships["exp-002"] == "supporting"
        assert relationships["exp-003"] == "exploratory"

    def test_link_experiment_to_nonexistent_hypothesis(self, test_db):
        """Linking to a non-existent hypothesis should return False."""
        result = link_experiment("HYP-9999-999", "exp-001")
        assert result is False

    def test_link_same_experiment_twice(self, created_hypothesis):
        """Linking same experiment twice should update relationship."""
        link_experiment(created_hypothesis, "exp-001", relationship="exploratory")
        link_experiment(created_hypothesis, "exp-001", relationship="primary")

        links = get_experiment_links(created_hypothesis)

        # Should only have one link
        assert len(links) == 1
        assert links[0]["relationship"] == "primary"

    def test_get_experiments(self, created_hypothesis):
        """Getting experiments should return list of experiment IDs."""
        link_experiment(created_hypothesis, "exp-001")
        link_experiment(created_hypothesis, "exp-002")
        link_experiment(created_hypothesis, "exp-003")

        experiments = get_experiments(created_hypothesis)

        assert len(experiments) == 3
        assert "exp-001" in experiments
        assert "exp-002" in experiments
        assert "exp-003" in experiments

    def test_get_experiments_empty(self, created_hypothesis):
        """Getting experiments when none linked should return empty list."""
        experiments = get_experiments(created_hypothesis)
        assert experiments == []

    def test_get_experiment_links(self, created_hypothesis):
        """Getting experiment links should return full details."""
        link_experiment(created_hypothesis, "exp-001", relationship="primary")

        links = get_experiment_links(created_hypothesis)

        assert len(links) == 1
        assert links[0]["experiment_id"] == "exp-001"
        assert links[0]["relationship"] == "primary"
        assert "created_at" in links[0]

    def test_get_experiment_links_empty(self, created_hypothesis):
        """Getting experiment links when none linked should return empty list."""
        links = get_experiment_links(created_hypothesis)
        assert links == []

    def test_link_multiple_experiments(self, created_hypothesis):
        """Can link multiple experiments to one hypothesis."""
        for i in range(5):
            link_experiment(created_hypothesis, f"exp-{i:03d}")

        experiments = get_experiments(created_hypothesis)
        assert len(experiments) == 5


class TestValidateHypothesisStatus:
    """Tests for validate_hypothesis_status()."""

    def test_validate_ready_hypothesis(self, created_hypothesis):
        """Hypothesis in testing with linked experiment should be ready to validate."""
        update_hypothesis(created_hypothesis, status="testing")
        link_experiment(created_hypothesis, "exp-001")

        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is True
        assert result["reasons"] == []

    def test_validate_without_testing_status(self, created_hypothesis):
        """Hypothesis not in testing status should fail validation check."""
        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is False
        assert any("testing" in reason.lower() for reason in result["reasons"])

    def test_validate_without_experiment(self, created_hypothesis):
        """Hypothesis without linked experiment should fail validation check."""
        update_hypothesis(created_hypothesis, status="testing")

        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is False
        assert any("experiment" in reason.lower() for reason in result["reasons"])

    def test_validate_nonexistent_hypothesis(self, test_db):
        """Validating non-existent hypothesis should return can_validate=False."""
        result = validate_hypothesis_status("HYP-9999-999")

        assert result["can_validate"] is False
        assert any("not found" in reason.lower() for reason in result["reasons"])

    def test_validate_warnings_no_outcome(self, created_hypothesis):
        """Missing outcome should generate a warning."""
        update_hypothesis(created_hypothesis, status="testing")
        link_experiment(created_hypothesis, "exp-001")

        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is True
        assert any("outcome" in warning.lower() for warning in result["warnings"])

    def test_validate_warnings_no_confidence_score(self, created_hypothesis):
        """Missing confidence score should generate a warning."""
        update_hypothesis(created_hypothesis, status="testing")
        link_experiment(created_hypothesis, "exp-001")

        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is True
        assert any("confidence" in warning.lower() for warning in result["warnings"])

    def test_validate_with_outcome_and_confidence(self, created_hypothesis):
        """Hypothesis with outcome and confidence should have fewer warnings."""
        update_hypothesis(created_hypothesis, status="testing")
        update_hypothesis(created_hypothesis, outcome="Positive result")
        update_hypothesis(created_hypothesis, confidence_score=0.9)
        link_experiment(created_hypothesis, "exp-001")

        result = validate_hypothesis_status(created_hypothesis)

        assert result["can_validate"] is True
        assert len(result["warnings"]) == 0


class TestHypothesisEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_create_hypothesis_with_long_text(self, test_db):
        """Should handle long text in thesis and prediction fields."""
        long_text = "A" * 10000

        hyp_id = create_hypothesis(
            title="Long text test",
            thesis=long_text,
            prediction=long_text,
            falsification=long_text,
            actor="user",
        )

        hyp = get_hypothesis(hyp_id)
        assert hyp["thesis"] == long_text
        assert hyp["testable_prediction"] == long_text
        assert hyp["falsification_criteria"] == long_text

    def test_create_hypothesis_with_special_characters(self, test_db):
        """Should handle special characters in text fields."""
        special_text = "Test with 'quotes', \"double quotes\", and special chars: <>&!@#$%^*()"

        hyp_id = create_hypothesis(
            title=special_text,
            thesis=special_text,
            prediction=special_text,
            falsification=special_text,
            actor="user",
        )

        hyp = get_hypothesis(hyp_id)
        assert hyp["title"] == special_text

    def test_create_hypothesis_with_unicode(self, test_db):
        """Should handle unicode characters in text fields."""
        unicode_text = "Test with unicode: \u00e9\u00e8\u00ea \u4e2d\u6587 \U0001f4c8"

        hyp_id = create_hypothesis(
            title=unicode_text,
            thesis=unicode_text,
            prediction=unicode_text,
            falsification=unicode_text,
            actor="user",
        )

        hyp = get_hypothesis(hyp_id)
        assert hyp["title"] == unicode_text

    def test_confidence_score_boundaries(self, created_hypothesis):
        """Confidence score at boundaries should work."""
        # Test minimum (0.0)
        update_hypothesis(created_hypothesis, confidence_score=0.0)
        hyp = get_hypothesis(created_hypothesis)
        assert hyp["confidence_score"] == 0.0

        # Test maximum (1.0)
        update_hypothesis(created_hypothesis, confidence_score=1.0)
        hyp = get_hypothesis(created_hypothesis)
        assert hyp["confidence_score"] == 1.0

    def test_list_hypotheses_combined_filters(self, test_db):
        """Should handle combined status and actor filters."""
        create_hypothesis(
            title="User Draft",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        id2 = create_hypothesis(
            title="User Testing",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        create_hypothesis(
            title="Agent Draft",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="agent:discovery",
        )

        update_hypothesis(id2, status="testing")

        # User + Draft
        result = list_hypotheses(status="draft", actor="user")
        assert len(result) == 1
        assert result[0]["title"] == "User Draft"

        # User + Testing
        result = list_hypotheses(status="testing", actor="user")
        assert len(result) == 1
        assert result[0]["title"] == "User Testing"

        # Agent + Draft
        result = list_hypotheses(status="draft", actor="agent:discovery")
        assert len(result) == 1
        assert result[0]["title"] == "Agent Draft"

    def test_experiment_ordering(self, created_hypothesis):
        """Experiments should be returned in descending order by created_at."""
        import time

        link_experiment(created_hypothesis, "exp-first")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        link_experiment(created_hypothesis, "exp-second")
        time.sleep(0.01)
        link_experiment(created_hypothesis, "exp-third")

        experiments = get_experiments(created_hypothesis)

        # Most recent should be first
        assert experiments[0] == "exp-third"
        assert experiments[2] == "exp-first"


class TestValidTransitionsMapping:
    """Tests to verify VALID_TRANSITIONS constant is complete and correct."""

    def test_all_statuses_have_transitions(self):
        """All statuses should have an entry in VALID_TRANSITIONS."""
        expected_statuses = {"draft", "testing", "validated", "rejected", "deployed", "deleted"}
        assert set(VALID_TRANSITIONS.keys()) == expected_statuses

    def test_deleted_is_terminal(self):
        """Deleted status should have no valid transitions (terminal)."""
        assert VALID_TRANSITIONS["deleted"] == set()

    def test_draft_transitions(self):
        """Draft should only transition to testing or deleted."""
        assert VALID_TRANSITIONS["draft"] == {"testing", "deleted"}

    def test_testing_transitions(self):
        """Testing should transition to validated, rejected, draft, or deleted."""
        assert VALID_TRANSITIONS["testing"] == {"validated", "rejected", "draft", "deleted"}

    def test_validated_transitions(self):
        """Validated should transition to deployed, rejected, or deleted."""
        assert VALID_TRANSITIONS["validated"] == {"deployed", "rejected", "deleted"}

    def test_rejected_transitions(self):
        """Rejected should transition to draft or deleted."""
        assert VALID_TRANSITIONS["rejected"] == {"draft", "deleted"}

    def test_deployed_transitions(self):
        """Deployed should transition to validated or deleted."""
        assert VALID_TRANSITIONS["deployed"] == {"validated", "deleted"}

    def test_no_self_transitions(self):
        """No status should be able to transition to itself."""
        for status, transitions in VALID_TRANSITIONS.items():
            assert status not in transitions, f"{status} should not transition to itself"
