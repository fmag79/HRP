"""
Comprehensive tests for the HRP Lineage/Audit Trail system.

Tests cover all lineage functions including event logging, querying,
chain reconstruction, and status summarization.
"""

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import TABLES
from hrp.research.lineage import (
    EventType,
    LineageEvent,
    create_child_event,
    get_agent_activity,
    get_deployment_trace,
    get_events_between,
    get_hypothesis_chain,
    get_lineage,
    get_recent_events,
    log_event,
    summarize_hypothesis_status,
)


@pytest.fixture
def test_db():
    """
    Create an isolated test database with lineage schema.

    This fixture:
    1. Creates a temporary DuckDB file
    2. Resets the singleton DatabaseManager
    3. Sets the HRP_DATA_DIR environment variable
    4. Creates required tables (lineage, hypothesis_experiments)
    5. Yields the database manager
    6. Cleans up after the test
    """
    # Create temp directory and database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "hrp.duckdb")

    # Reset the singleton to force new connection
    DatabaseManager.reset()

    # Set environment variable for database location
    old_env = os.environ.get("HRP_DATA_DIR")
    os.environ["HRP_DATA_DIR"] = temp_dir

    # Get fresh database manager
    db = get_db()

    # Create required tables
    with db.connection() as conn:
        # Create sequence for lineage auto-increment PK
        conn.execute("CREATE SEQUENCE IF NOT EXISTS lineage_seq START 1")
        # Create lineage table
        conn.execute(TABLES["lineage"])
        # Create hypothesis_experiments table (needed for get_hypothesis_chain)
        conn.execute(TABLES["hypothesis_experiments"])
        # Create hypotheses table (might be referenced)
        conn.execute(TABLES["hypotheses"])

    yield db

    # Cleanup
    DatabaseManager.reset()
    if old_env is not None:
        os.environ["HRP_DATA_DIR"] = old_env
    else:
        os.environ.pop("HRP_DATA_DIR", None)

    # Remove temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEventType:
    """Tests for the EventType enum."""

    def test_all_event_types_exist(self):
        """Verify all expected event types are defined."""
        expected_types = [
            "hypothesis_created",
            "hypothesis_updated",
            "hypothesis_deleted",
            "experiment_run",
            "experiment_linked",
            "validation_passed",
            "validation_failed",
            "deployment_approved",
            "deployment_rejected",
            "agent_run_complete",
            "agent_run_start",  # NEW
            "data_ingestion",
            "system_error",
        ]

        actual_types = [e.value for e in EventType]

        for expected in expected_types:
            assert expected in actual_types, f"Missing event type: {expected}"

    def test_event_type_is_string_enum(self):
        """EventType values should be usable as strings."""
        assert EventType.HYPOTHESIS_CREATED.value == "hypothesis_created"
        assert str(EventType.HYPOTHESIS_CREATED) == "EventType.HYPOTHESIS_CREATED"

    def test_validation_analyst_event_type_exists(self):
        """VALIDATION_ANALYST_REVIEW event type is defined."""
        assert hasattr(EventType, "VALIDATION_ANALYST_REVIEW")
        assert EventType.VALIDATION_ANALYST_REVIEW.value == "validation_analyst_review"

    def test_agent_run_start_event_type_exists(self):
        """AGENT_RUN_START event type should exist in EventType enum."""
        assert hasattr(EventType, "AGENT_RUN_START")
        assert EventType.AGENT_RUN_START == "agent_run_start"


class TestLineageEvent:
    """Tests for the LineageEvent dataclass."""

    def test_to_dict_converts_timestamp_to_iso(self, test_db):
        """to_dict should convert datetime to ISO format string."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        event = LineageEvent(
            lineage_id=1,
            event_type="hypothesis_created",
            timestamp=ts,
            actor="user",
            hypothesis_id="HYP-2025-001",
            experiment_id=None,
            details={"title": "Test"},
            parent_lineage_id=None,
        )

        result = event.to_dict()

        assert result["timestamp"] == "2025-01-15T10:30:00+00:00"
        assert result["lineage_id"] == 1
        assert result["details"] == {"title": "Test"}


class TestLogEvent:
    """Tests for the log_event function."""

    def test_log_event_basic(self, test_db):
        """Basic event logging should return a positive lineage_id."""
        lineage_id = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            details={"title": "Test Hypothesis"},
            hypothesis_id="HYP-2025-001",
        )

        assert lineage_id > 0

    def test_log_event_with_all_parameters(self, test_db):
        """Event logging with all parameters should work."""
        lineage_id = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:validator",
            details={"sharpe": 1.5, "returns": 0.15},
            hypothesis_id="HYP-2025-001",
            experiment_id="EXP-001",
            parent_lineage_id=None,
        )

        assert lineage_id > 0

        # Verify event was stored correctly
        events = get_lineage(hypothesis_id="HYP-2025-001")
        assert len(events) == 1
        assert events[0]["experiment_id"] == "EXP-001"
        assert events[0]["actor"] == "agent:validator"

    def test_log_event_invalid_type_raises(self, test_db):
        """Invalid event types should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            log_event(
                event_type="invalid_type",
                actor="user",
            )

        assert "Invalid event_type" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_log_event_increments_id(self, test_db):
        """Sequential events should have incrementing IDs."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )
        id2 = log_event(
            event_type=EventType.HYPOTHESIS_UPDATED.value,
            actor="user",
        )
        id3 = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
        )

        assert id2 == id1 + 1
        assert id3 == id2 + 1

    def test_log_event_stores_details_as_json(self, test_db):
        """Details should be stored and retrieved correctly."""
        complex_details = {
            "metrics": {"sharpe": 1.5, "max_drawdown": -0.15},
            "config": {"lookback": 20, "symbols": ["AAPL", "MSFT"]},
            "notes": "Initial backtest run",
        }

        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            details=complex_details,
            hypothesis_id="HYP-TEST",
        )

        events = get_lineage(hypothesis_id="HYP-TEST")
        assert len(events) == 1
        assert events[0]["details"]["metrics"]["sharpe"] == 1.5
        assert "AAPL" in events[0]["details"]["config"]["symbols"]

    def test_log_event_with_none_details(self, test_db):
        """Event with None details should store empty dict."""
        log_event(
            event_type=EventType.SYSTEM_ERROR.value,
            actor="system",
            details=None,
        )

        events = get_lineage(actor="system")
        assert events[0]["details"] == {}

    def test_log_event_sets_timestamp(self, test_db):
        """Events should have valid ISO format timestamps."""
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-TS-TEST",
        )

        events = get_lineage(hypothesis_id="HYP-TS-TEST")
        timestamp_str = events[0]["timestamp"]

        # Should be parseable as ISO format
        event_ts = datetime.fromisoformat(timestamp_str)
        assert event_ts is not None, "Timestamp should be parseable"

        # Should be a reasonable date (in the current year)
        assert event_ts.year >= 2024, "Timestamp should have reasonable year"

        # Should have the same date as today (regardless of timezone)
        today = datetime.now().date()
        assert event_ts.date() == today, "Timestamp should be from today"

    def test_log_event_with_parent_lineage_id(self, test_db):
        """Events can reference a parent event."""
        parent_id = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-PARENT",
        )

        child_id = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id="HYP-PARENT",
            parent_lineage_id=parent_id,
        )

        events = get_lineage(hypothesis_id="HYP-PARENT", limit=10)
        child_event = next(e for e in events if e["lineage_id"] == child_id)

        assert child_event["parent_lineage_id"] == parent_id


class TestCreateChildEvent:
    """Tests for the create_child_event function."""

    def test_child_inherits_hypothesis_id(self, test_db):
        """Child events should inherit hypothesis_id from parent."""
        parent_id = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-INHERIT-001",
        )

        child_id = create_child_event(
            parent_lineage_id=parent_id,
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:validator",
        )

        events = get_lineage(hypothesis_id="HYP-INHERIT-001")
        child_event = next(e for e in events if e["lineage_id"] == child_id)

        assert child_event["hypothesis_id"] == "HYP-INHERIT-001"
        assert child_event["parent_lineage_id"] == parent_id

    def test_child_inherits_experiment_id(self, test_db):
        """Child events should inherit experiment_id from parent."""
        parent_id = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id="HYP-002",
            experiment_id="EXP-002",
        )

        child_id = create_child_event(
            parent_lineage_id=parent_id,
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            details={"passed_checks": ["sharpe", "drawdown"]},
        )

        events = get_lineage(experiment_id="EXP-002")
        child_event = next(e for e in events if e["lineage_id"] == child_id)

        assert child_event["experiment_id"] == "EXP-002"
        assert child_event["hypothesis_id"] == "HYP-002"

    def test_child_event_nonexistent_parent_raises(self, test_db):
        """Creating child with non-existent parent should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_child_event(
                parent_lineage_id=99999,
                event_type=EventType.VALIDATION_PASSED.value,
                actor="user",
            )

        assert "not found" in str(exc_info.value)

    def test_child_chain_multiple_levels(self, test_db):
        """Child events can form multi-level chains."""
        # Create chain: created -> experiment -> validation -> approval
        created_id = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-CHAIN",
        )

        experiment_id = create_child_event(
            parent_lineage_id=created_id,
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
        )

        validation_id = create_child_event(
            parent_lineage_id=experiment_id,
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
        )

        approval_id = create_child_event(
            parent_lineage_id=validation_id,
            event_type=EventType.DEPLOYMENT_APPROVED.value,
            actor="user",
        )

        # All events should have the same hypothesis_id
        events = get_lineage(hypothesis_id="HYP-CHAIN")
        assert len(events) == 4

        # Verify chain structure
        events_by_id = {e["lineage_id"]: e for e in events}
        assert events_by_id[experiment_id]["parent_lineage_id"] == created_id
        assert events_by_id[validation_id]["parent_lineage_id"] == experiment_id
        assert events_by_id[approval_id]["parent_lineage_id"] == validation_id


class TestGetLineage:
    """Tests for the get_lineage function."""

    def test_get_lineage_by_hypothesis(self, test_db):
        """Filter events by hypothesis_id."""
        # Create events for different hypotheses
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-A",
        )
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id="HYP-B",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id="HYP-A",
        )

        events_a = get_lineage(hypothesis_id="HYP-A")
        events_b = get_lineage(hypothesis_id="HYP-B")

        assert len(events_a) == 2
        assert len(events_b) == 1
        assert all(e["hypothesis_id"] == "HYP-A" for e in events_a)

    def test_get_lineage_by_experiment(self, test_db):
        """Filter events by experiment_id."""
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            experiment_id="EXP-001",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            experiment_id="EXP-002",
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            experiment_id="EXP-001",
        )

        events = get_lineage(experiment_id="EXP-001")

        assert len(events) == 2
        assert all(e["experiment_id"] == "EXP-001" for e in events)

    def test_get_lineage_by_actor(self, test_db):
        """Filter events by actor."""
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:discovery",
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
        )

        user_events = get_lineage(actor="user")
        discovery_events = get_lineage(actor="agent:discovery")

        assert len(user_events) == 1
        assert len(discovery_events) == 1
        assert user_events[0]["actor"] == "user"

    def test_get_lineage_by_event_type(self, test_db):
        """Filter events by event_type."""
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:discovery",
        )

        experiment_events = get_lineage(event_type=EventType.EXPERIMENT_RUN.value)

        assert len(experiment_events) == 2
        assert all(e["event_type"] == "experiment_run" for e in experiment_events)

    def test_get_lineage_multiple_filters(self, test_db):
        """Multiple filters should be combined with AND."""
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id="HYP-MULTI",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:discovery",
            hypothesis_id="HYP-MULTI",
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="user",
            hypothesis_id="HYP-MULTI",
        )

        events = get_lineage(
            hypothesis_id="HYP-MULTI",
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
        )

        assert len(events) == 1
        assert events[0]["actor"] == "user"
        assert events[0]["event_type"] == "experiment_run"

    def test_get_lineage_limit(self, test_db):
        """Limit parameter should restrict result count."""
        for i in range(10):
            log_event(
                event_type=EventType.DATA_INGESTION.value,
                actor="system",
                details={"batch": i},
                dedupe_window_seconds=0,  # Disable deduplication for test
            )

        events = get_lineage(actor="system", limit=5)

        assert len(events) == 5

    def test_get_lineage_ordered_by_timestamp_desc(self, test_db):
        """Results should be ordered by timestamp descending (newest first)."""
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            details={"order": 1},
            hypothesis_id="HYP-ORDER",
        )
        time.sleep(0.01)  # Small delay to ensure different timestamps
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            details={"order": 2},
            hypothesis_id="HYP-ORDER",
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="user",
            details={"order": 3},
            hypothesis_id="HYP-ORDER",
        )

        events = get_lineage(hypothesis_id="HYP-ORDER")

        # Newest first (order 3)
        assert events[0]["details"]["order"] == 3
        assert events[-1]["details"]["order"] == 1

    def test_get_lineage_no_filters_returns_all(self, test_db):
        """No filters should return all events up to limit."""
        for i in range(5):
            log_event(
                event_type=EventType.DATA_INGESTION.value,
                actor=f"actor_{i}",
            )

        events = get_lineage()

        assert len(events) == 5


class TestGetHypothesisChain:
    """Tests for the get_hypothesis_chain function."""

    def test_get_hypothesis_chain_chronological(self, test_db):
        """Chain should be in chronological order (oldest first)."""
        hyp_id = "HYP-CHRONO"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"step": 1},
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"step": 2},
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
            details={"step": 3},
        )

        chain = get_hypothesis_chain(hyp_id)

        # Oldest first
        assert len(chain) == 3
        assert chain[0]["details"]["step"] == 1
        assert chain[1]["details"]["step"] == 2
        assert chain[2]["details"]["step"] == 3

    def test_get_hypothesis_chain_includes_linked_experiments(self, test_db):
        """Chain should include events from linked experiments."""
        hyp_id = "HYP-LINKED"
        exp_id = "EXP-LINKED"

        # Create hypothesis event
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        # Create experiment event (linked via experiment_id only)
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            experiment_id=exp_id,
        )

        # Link experiment to hypothesis in the junction table
        test_db.execute(
            "INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id) VALUES (?, ?)",
            (hyp_id, exp_id),
        )

        chain = get_hypothesis_chain(hyp_id)

        # Should include both events
        experiment_events = [e for e in chain if e["experiment_id"] == exp_id]
        assert len(experiment_events) >= 1

    def test_get_hypothesis_chain_empty_for_nonexistent(self, test_db):
        """Non-existent hypothesis should return empty chain."""
        chain = get_hypothesis_chain("HYP-NONEXISTENT")

        assert chain == []

    def test_get_hypothesis_chain_deduplicates(self, test_db):
        """Chain should not have duplicate events."""
        hyp_id = "HYP-DEDUP"

        # Create events that might be matched multiple ways
        for i in range(3):
            log_event(
                event_type=EventType.EXPERIMENT_RUN.value,
                actor="user",
                hypothesis_id=hyp_id,
                details={"run": i},
            )

        chain = get_hypothesis_chain(hyp_id)
        lineage_ids = [e["lineage_id"] for e in chain]

        # No duplicates
        assert len(lineage_ids) == len(set(lineage_ids))


class TestGetRecentEvents:
    """Tests for the get_recent_events function."""

    def test_get_recent_events_default_24h(self, test_db):
        """Default should return events from last 24 hours."""
        # Create recent event
        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
        )

        events = get_recent_events()

        # Should include our event (created just now)
        assert len(events) >= 1

    def test_get_recent_events_with_actor_filter(self, test_db):
        """Actor filter should work with time filter."""
        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
        )
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )

        system_events = get_recent_events(hours=24, actor="system")

        assert len(system_events) == 1
        assert system_events[0]["actor"] == "system"

    def test_get_recent_events_custom_hours(self, test_db):
        """Custom hours parameter should work."""
        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
        )

        # Very short window - should still include recent event
        events = get_recent_events(hours=1)

        assert len(events) >= 1

    def test_get_recent_events_ordered_desc(self, test_db):
        """Results should be ordered by timestamp descending."""
        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
            details={"order": 1},
            dedupe_window_seconds=0,  # Disable deduplication for test
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
            details={"order": 2},
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        events = get_recent_events(hours=1, actor="system")

        # Newest first
        assert events[0]["details"]["order"] == 2


class TestGetAgentActivity:
    """Tests for the get_agent_activity function."""

    def test_get_agent_activity_matches_agent_prefix(self, test_db):
        """Should match 'agent:name' format."""
        log_event(
            event_type=EventType.AGENT_RUN_COMPLETE.value,
            actor="agent:discovery",
            details={"found": 5},
        )
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )

        activity = get_agent_activity("discovery", days=7)

        assert len(activity) == 1
        assert activity[0]["actor"] == "agent:discovery"

    def test_get_agent_activity_custom_days(self, test_db):
        """Custom days parameter should work."""
        log_event(
            event_type=EventType.AGENT_RUN_COMPLETE.value,
            actor="agent:validator",
        )

        activity = get_agent_activity("validator", days=1)

        assert len(activity) >= 1

    def test_get_agent_activity_no_matches(self, test_db):
        """Non-existent agent should return empty list."""
        activity = get_agent_activity("nonexistent-agent", days=7)

        assert activity == []

    def test_get_agent_activity_multiple_events(self, test_db):
        """Should return all events for the agent."""
        for i in range(5):
            log_event(
                event_type=EventType.AGENT_RUN_COMPLETE.value,
                actor="agent:scheduler",
                details={"run": i},
                dedupe_window_seconds=0,  # Disable deduplication for test
            )

        activity = get_agent_activity("scheduler", days=7)

        assert len(activity) == 5


class TestGetDeploymentTrace:
    """Tests for the get_deployment_trace function."""

    def test_get_deployment_trace_filters_events(self, test_db):
        """Should only return deployment-relevant event types."""
        hyp_id = "HYP-DEPLOY"

        # Create various events
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.HYPOTHESIS_UPDATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.DEPLOYMENT_APPROVED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        trace = get_deployment_trace(hyp_id)

        # Should exclude HYPOTHESIS_UPDATED
        event_types = [e["event_type"] for e in trace]
        assert "hypothesis_updated" not in event_types
        assert "hypothesis_created" in event_types
        assert "experiment_run" in event_types
        assert "validation_passed" in event_types
        assert "deployment_approved" in event_types

    def test_get_deployment_trace_chronological(self, test_db):
        """Trace should be in chronological order."""
        hyp_id = "HYP-DEPLOY-CHRONO"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"step": 1},
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"step": 2},
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"step": 3},
        )

        trace = get_deployment_trace(hyp_id)

        # Check chronological order
        assert trace[0]["details"]["step"] == 1
        assert trace[-1]["details"]["step"] == 3

    def test_get_deployment_trace_includes_rejection(self, test_db):
        """Should include deployment_rejected events."""
        hyp_id = "HYP-REJECTED"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.VALIDATION_FAILED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.DEPLOYMENT_REJECTED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        trace = get_deployment_trace(hyp_id)

        event_types = [e["event_type"] for e in trace]
        assert "validation_failed" in event_types
        assert "deployment_rejected" in event_types

    def test_get_deployment_trace_empty_for_nonexistent(self, test_db):
        """Non-existent hypothesis should return empty trace."""
        trace = get_deployment_trace("HYP-NONEXISTENT")

        assert trace == []


class TestGetEventsBetween:
    """Tests for the get_events_between function."""

    def test_get_events_between_inclusive(self, test_db):
        """Should include events at both boundaries."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            details={"event": "start"},
        )
        time.sleep(0.01)
        id2 = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            details={"event": "middle"},
        )
        time.sleep(0.01)
        id3 = log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="user",
            details={"event": "end"},
        )

        events = get_events_between(id1, id3)

        assert len(events) == 3
        lineage_ids = [e["lineage_id"] for e in events]
        assert id1 in lineage_ids
        assert id2 in lineage_ids
        assert id3 in lineage_ids

    def test_get_events_between_chronological_order(self, test_db):
        """Results should be in chronological order (oldest first)."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            details={"order": 1},
        )
        time.sleep(0.01)
        id2 = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            details={"order": 2},
        )
        time.sleep(0.01)
        id3 = log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="user",
            details={"order": 3},
        )

        events = get_events_between(id1, id3)

        assert events[0]["details"]["order"] == 1
        assert events[2]["details"]["order"] == 3

    def test_get_events_between_handles_reversed_ids(self, test_db):
        """Should work even if start_id > end_id."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )
        time.sleep(0.01)
        id2 = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
        )

        # Pass in reverse order
        events = get_events_between(id2, id1)

        # Should still work and include both
        assert len(events) == 2

    def test_get_events_between_nonexistent_start(self, test_db):
        """Non-existent start event should return empty list."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )

        events = get_events_between(99999, id1)

        assert events == []

    def test_get_events_between_nonexistent_end(self, test_db):
        """Non-existent end event should return empty list."""
        id1 = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
        )

        events = get_events_between(id1, 99999)

        assert events == []


class TestSummarizeHypothesisStatus:
    """Tests for the summarize_hypothesis_status function."""

    def test_summarize_basic_hypothesis(self, test_db):
        """Basic summary should include all expected fields."""
        hyp_id = "HYP-SUMMARY-001"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["hypothesis_id"] == hyp_id
        assert summary["created_by"] == "user"
        assert summary["created_at"] is not None
        assert summary["event_count"] == 1
        assert summary["experiment_count"] == 0
        assert summary["validation_status"] == "pending"
        assert summary["deployment_status"] == "pending"

    def test_summarize_nonexistent_hypothesis(self, test_db):
        """Non-existent hypothesis should return error."""
        summary = summarize_hypothesis_status("HYP-NONEXISTENT")

        assert summary["hypothesis_id"] == "HYP-NONEXISTENT"
        assert "error" in summary
        assert "No lineage events found" in summary["error"]

    def test_summarize_validation_passed(self, test_db):
        """Should detect validation passed status."""
        hyp_id = "HYP-VALID-PASS"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["validation_status"] == "passed"

    def test_summarize_validation_failed(self, test_db):
        """Should detect validation failed status."""
        hyp_id = "HYP-VALID-FAIL"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.VALIDATION_FAILED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["validation_status"] == "failed"

    def test_summarize_validation_most_recent_wins(self, test_db):
        """Most recent validation event should determine status."""
        hyp_id = "HYP-VALID-RECENT"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.VALIDATION_FAILED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        # Most recent is PASSED
        assert summary["validation_status"] == "passed"

    def test_summarize_deployment_approved(self, test_db):
        """Should detect deployment approved status."""
        hyp_id = "HYP-DEPLOY-APPROVED"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.DEPLOYMENT_APPROVED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["deployment_status"] == "approved"

    def test_summarize_deployment_rejected(self, test_db):
        """Should detect deployment rejected status."""
        hyp_id = "HYP-DEPLOY-REJECTED"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.DEPLOYMENT_REJECTED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["deployment_status"] == "rejected"

    def test_summarize_experiment_count(self, test_db):
        """Should count unique experiments."""
        hyp_id = "HYP-EXP-COUNT"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            experiment_id="EXP-001",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            experiment_id="EXP-002",
        )
        # Same experiment, different event
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
            experiment_id="EXP-001",
        )

        summary = summarize_hypothesis_status(hyp_id)

        # Should count 2 unique experiments
        assert summary["experiment_count"] == 2

    def test_summarize_last_event(self, test_db):
        """Should include last event in summary."""
        hyp_id = "HYP-LAST-EVENT"

        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )
        time.sleep(0.01)
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"this_is_last": True},
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["last_event"] is not None
        assert summary["last_event"]["details"]["this_is_last"] is True

    def test_summarize_full_lifecycle(self, test_db):
        """Should correctly summarize a full hypothesis lifecycle."""
        hyp_id = "HYP-LIFECYCLE"

        # Full lifecycle: create -> experiments -> validation -> deployment
        log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"title": "Momentum Strategy"},
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            experiment_id="EXP-LC-001",
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="agent:discovery",
            hypothesis_id=hyp_id,
            experiment_id="EXP-LC-002",
        )
        log_event(
            event_type=EventType.VALIDATION_FAILED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.HYPOTHESIS_UPDATED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"adjusted": "lookback period"},
        )
        log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            experiment_id="EXP-LC-003",
        )
        log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
        )
        log_event(
            event_type=EventType.DEPLOYMENT_APPROVED.value,
            actor="user",
            hypothesis_id=hyp_id,
        )

        summary = summarize_hypothesis_status(hyp_id)

        assert summary["hypothesis_id"] == hyp_id
        assert summary["created_by"] == "user"
        assert summary["event_count"] == 8
        assert summary["experiment_count"] == 3
        assert summary["validation_status"] == "passed"
        assert summary["deployment_status"] == "approved"


class TestParseDetails:
    """Tests for the _parse_details helper function."""

    def test_parse_details_handles_none(self, test_db):
        """None should return empty dict."""
        log_event(
            event_type=EventType.SYSTEM_ERROR.value,
            actor="system",
            details=None,
        )

        events = get_lineage(actor="system")
        assert events[0]["details"] == {}

    def test_parse_details_handles_dict(self, test_db):
        """Dict should pass through unchanged."""
        original = {"key": "value", "nested": {"a": 1}}

        log_event(
            event_type=EventType.DATA_INGESTION.value,
            actor="system",
            details=original,
        )

        events = get_lineage(actor="system")
        assert events[0]["details"]["key"] == "value"
        assert events[0]["details"]["nested"]["a"] == 1


class TestIntegration:
    """Integration tests for the lineage system."""

    def test_complete_audit_trail(self, test_db):
        """Test a complete audit trail from hypothesis to deployment."""
        hyp_id = "HYP-AUDIT-001"

        # Step 1: User creates hypothesis
        created_id = log_event(
            event_type=EventType.HYPOTHESIS_CREATED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={
                "title": "Momentum predicts returns",
                "thesis": "High momentum stocks outperform",
            },
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Step 2: Run initial experiment
        exp1_id = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"experiment_id": "EXP-001", "sharpe": 0.8},
            parent_lineage_id=created_id,
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Step 3: Agent runs validation
        val1_id = log_event(
            event_type=EventType.VALIDATION_FAILED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
            details={"reason": "Sharpe below threshold"},
            parent_lineage_id=exp1_id,
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Step 4: User adjusts and re-runs
        exp2_id = log_event(
            event_type=EventType.EXPERIMENT_RUN.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"experiment_id": "EXP-002", "sharpe": 1.5},
            parent_lineage_id=val1_id,
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Step 5: Validation passes
        val2_id = log_event(
            event_type=EventType.VALIDATION_PASSED.value,
            actor="agent:validator",
            hypothesis_id=hyp_id,
            details={"all_checks": "passed"},
            parent_lineage_id=exp2_id,
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Step 6: User approves deployment
        deploy_id = log_event(
            event_type=EventType.DEPLOYMENT_APPROVED.value,
            actor="user",
            hypothesis_id=hyp_id,
            details={"notes": "Approved for paper trading"},
            parent_lineage_id=val2_id,
            dedupe_window_seconds=0,  # Disable deduplication for test
        )

        # Verify complete chain
        chain = get_hypothesis_chain(hyp_id)
        assert len(chain) == 6

        # Verify deployment trace
        trace = get_deployment_trace(hyp_id)
        # Deployment trace includes: hypothesis_created, 2x experiment_run, validation_failed,
        # validation_passed, deployment_approved (6 events total)
        assert len(trace) == 6

        # Verify summary
        summary = summarize_hypothesis_status(hyp_id)
        assert summary["validation_status"] == "passed"
        assert summary["deployment_status"] == "approved"
        assert summary["event_count"] == 6

    def test_multiple_hypotheses_isolated(self, test_db):
        """Events from different hypotheses should be properly isolated."""
        hyp_a = "HYP-A-ISOLATED"
        hyp_b = "HYP-B-ISOLATED"

        # Create events for hypothesis A
        for i in range(3):
            log_event(
                event_type=EventType.EXPERIMENT_RUN.value,
                actor="user",
                hypothesis_id=hyp_a,
                details={"hyp": "A", "run": i},
                dedupe_window_seconds=0,  # Disable deduplication for test
            )

        # Create events for hypothesis B
        for i in range(5):
            log_event(
                event_type=EventType.EXPERIMENT_RUN.value,
                actor="user",
                hypothesis_id=hyp_b,
                details={"hyp": "B", "run": i},
                dedupe_window_seconds=0,  # Disable deduplication for test
            )

        # Query each hypothesis
        events_a = get_lineage(hypothesis_id=hyp_a)
        events_b = get_lineage(hypothesis_id=hyp_b)

        assert len(events_a) == 3
        assert len(events_b) == 5

        # Verify no cross-contamination
        assert all(e["details"]["hyp"] == "A" for e in events_a)
        assert all(e["details"]["hyp"] == "B" for e in events_b)

    def test_agent_tracking(self, test_db):
        """Test agent activity tracking across multiple runs."""
        agents = ["discovery", "validator", "scheduler"]

        # Create activity for each agent
        for agent in agents:
            for i in range(3):
                log_event(
                    event_type=EventType.AGENT_RUN_COMPLETE.value,
                    actor=f"agent:{agent}",
                    details={"run": i, "success": True},
                    dedupe_window_seconds=0,  # Disable deduplication for test
                )

        # Query each agent's activity
        for agent in agents:
            activity = get_agent_activity(agent, days=1)
            assert len(activity) == 3
            assert all(f"agent:{agent}" in a["actor"] for a in activity)
