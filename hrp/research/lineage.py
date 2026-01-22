"""
Lineage/Audit Trail system for HRP.

Tracks all significant actions for auditability and provides query capabilities
to understand the history of hypotheses, experiments, and system events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from loguru import logger

from hrp.data.db import get_db


class EventType(str, Enum):
    """Supported lineage event types."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    HYPOTHESIS_DELETED = "hypothesis_deleted"
    EXPERIMENT_RUN = "experiment_run"
    EXPERIMENT_LINKED = "experiment_linked"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    DEPLOYMENT_APPROVED = "deployment_approved"
    DEPLOYMENT_REJECTED = "deployment_rejected"
    AGENT_RUN_COMPLETE = "agent_run_complete"
    DATA_INGESTION = "data_ingestion"
    SYSTEM_ERROR = "system_error"


@dataclass
class LineageEvent:
    """
    Represents a single event in the lineage chain.

    Attributes:
        lineage_id: Unique identifier for this event
        event_type: Type of event (see EventType enum)
        timestamp: UTC timestamp when event occurred
        actor: Who/what caused the event (e.g., 'user', 'agent:discovery')
        hypothesis_id: Associated hypothesis ID if applicable
        experiment_id: Associated experiment ID if applicable
        details: Additional event-specific data as dictionary
        parent_lineage_id: ID of parent event for chaining
    """

    lineage_id: int
    event_type: str
    timestamp: datetime
    actor: str
    hypothesis_id: str | None
    experiment_id: str | None
    details: dict
    parent_lineage_id: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


def _parse_details(details_raw: Any) -> dict:
    """Parse details field from database (may be JSON string or dict)."""
    if details_raw is None:
        return {}
    if isinstance(details_raw, dict):
        return details_raw
    if isinstance(details_raw, str):
        try:
            return json.loads(details_raw)
        except json.JSONDecodeError:
            return {"raw": details_raw}
    return {"raw": str(details_raw)}


def _row_to_event(row: tuple) -> LineageEvent:
    """Convert a database row to a LineageEvent."""
    return LineageEvent(
        lineage_id=row[0],
        event_type=row[1],
        timestamp=row[2] if isinstance(row[2], datetime) else datetime.fromisoformat(str(row[2])),
        actor=row[3],
        hypothesis_id=row[4],
        experiment_id=row[5],
        details=_parse_details(row[6]),
        parent_lineage_id=row[7],
    )


def _get_next_lineage_id() -> int:
    """Get the next available lineage_id."""
    db = get_db()
    result = db.fetchone("SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage")
    return result[0] if result else 1


def log_event(
    event_type: str,
    actor: str,
    details: dict | None = None,
    hypothesis_id: str | None = None,
    experiment_id: str | None = None,
    parent_lineage_id: int | None = None,
) -> int:
    """
    Log a lineage event to the database.

    Args:
        event_type: Type of event (use EventType enum values)
        actor: Who/what caused the event (e.g., 'user', 'agent:discovery')
        details: Additional event-specific data as dictionary
        hypothesis_id: Associated hypothesis ID if applicable
        experiment_id: Associated experiment ID if applicable
        parent_lineage_id: ID of parent event for creating chains

    Returns:
        lineage_id: The ID of the newly created event

    Raises:
        ValueError: If event_type is not a valid EventType
    """
    # Validate event type
    valid_types = {e.value for e in EventType}
    if event_type not in valid_types:
        raise ValueError(
            f"Invalid event_type '{event_type}'. "
            f"Valid types: {sorted(valid_types)}"
        )

    db = get_db()
    lineage_id = _get_next_lineage_id()
    timestamp = datetime.now(timezone.utc)
    details_json = json.dumps(details or {})

    query = """
        INSERT INTO lineage (
            lineage_id, event_type, timestamp, actor,
            hypothesis_id, experiment_id, details, parent_lineage_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    db.execute(
        query,
        (
            lineage_id,
            event_type,
            timestamp,
            actor,
            hypothesis_id,
            experiment_id,
            details_json,
            parent_lineage_id,
        ),
    )

    logger.info(
        f"Logged lineage event: {event_type} by {actor} "
        f"(lineage_id={lineage_id}, hypothesis={hypothesis_id}, experiment={experiment_id})"
    )

    return lineage_id


def create_child_event(
    parent_lineage_id: int,
    event_type: str,
    actor: str,
    details: dict | None = None,
) -> int:
    """
    Create a child event linked to a parent event.

    This is a convenience function for creating event chains. It automatically
    copies the hypothesis_id and experiment_id from the parent event.

    Args:
        parent_lineage_id: ID of the parent event
        event_type: Type of the new event
        actor: Who/what caused the event
        details: Additional event-specific data

    Returns:
        lineage_id: The ID of the newly created child event

    Raises:
        ValueError: If parent_lineage_id does not exist
    """
    db = get_db()
    parent = db.fetchone(
        "SELECT hypothesis_id, experiment_id FROM lineage WHERE lineage_id = ?",
        (parent_lineage_id,),
    )

    if parent is None:
        raise ValueError(f"Parent lineage event {parent_lineage_id} not found")

    hypothesis_id, experiment_id = parent

    return log_event(
        event_type=event_type,
        actor=actor,
        details=details,
        hypothesis_id=hypothesis_id,
        experiment_id=experiment_id,
        parent_lineage_id=parent_lineage_id,
    )


def get_lineage(
    hypothesis_id: str | None = None,
    experiment_id: str | None = None,
    event_type: str | None = None,
    actor: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Query lineage events with optional filters.

    Args:
        hypothesis_id: Filter by hypothesis ID
        experiment_id: Filter by experiment ID
        event_type: Filter by event type
        actor: Filter by actor (exact match)
        limit: Maximum number of events to return (default 100)

    Returns:
        List of event dictionaries ordered by timestamp descending
    """
    db = get_db()

    conditions = []
    params = []

    if hypothesis_id is not None:
        conditions.append("hypothesis_id = ?")
        params.append(hypothesis_id)

    if experiment_id is not None:
        conditions.append("experiment_id = ?")
        params.append(experiment_id)

    if event_type is not None:
        conditions.append("event_type = ?")
        params.append(event_type)

    if actor is not None:
        conditions.append("actor = ?")
        params.append(actor)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT lineage_id, event_type, timestamp, actor,
               hypothesis_id, experiment_id, details, parent_lineage_id
        FROM lineage
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    rows = db.fetchall(query, tuple(params))
    return [_row_to_event(row).to_dict() for row in rows]


def get_hypothesis_chain(hypothesis_id: str) -> list[dict]:
    """
    Get the full event chain for a hypothesis from creation to current state.

    Returns all events related to the hypothesis, including linked experiments,
    ordered chronologically (oldest first) to show the progression.

    Args:
        hypothesis_id: The hypothesis ID to trace

    Returns:
        List of event dictionaries in chronological order
    """
    db = get_db()

    # Get all events directly related to the hypothesis
    # or related through experiments linked to the hypothesis
    query = """
        WITH RECURSIVE
        -- Get all experiments linked to this hypothesis
        linked_experiments AS (
            SELECT experiment_id
            FROM hypothesis_experiments
            WHERE hypothesis_id = ?
        ),
        -- Get all direct hypothesis events and experiment events
        base_events AS (
            SELECT lineage_id, event_type, timestamp, actor,
                   hypothesis_id, experiment_id, details, parent_lineage_id
            FROM lineage
            WHERE hypothesis_id = ?
               OR experiment_id IN (SELECT experiment_id FROM linked_experiments)
        ),
        -- Also follow parent chains
        event_chain AS (
            SELECT * FROM base_events
            UNION
            SELECT l.lineage_id, l.event_type, l.timestamp, l.actor,
                   l.hypothesis_id, l.experiment_id, l.details, l.parent_lineage_id
            FROM lineage l
            INNER JOIN event_chain ec ON l.lineage_id = ec.parent_lineage_id
        )
        SELECT DISTINCT lineage_id, event_type, timestamp, actor,
                        hypothesis_id, experiment_id, details, parent_lineage_id
        FROM event_chain
        ORDER BY timestamp ASC
    """

    rows = db.fetchall(query, (hypothesis_id, hypothesis_id))
    return [_row_to_event(row).to_dict() for row in rows]


def get_recent_events(
    hours: int = 24,
    actor: str | None = None,
) -> list[dict]:
    """
    Get events from the last N hours.

    Args:
        hours: Number of hours to look back (default 24)
        actor: Optional actor filter

    Returns:
        List of event dictionaries ordered by timestamp descending
    """
    db = get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    if actor is not None:
        query = """
            SELECT lineage_id, event_type, timestamp, actor,
                   hypothesis_id, experiment_id, details, parent_lineage_id
            FROM lineage
            WHERE timestamp >= ? AND actor = ?
            ORDER BY timestamp DESC
        """
        rows = db.fetchall(query, (cutoff, actor))
    else:
        query = """
            SELECT lineage_id, event_type, timestamp, actor,
                   hypothesis_id, experiment_id, details, parent_lineage_id
            FROM lineage
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """
        rows = db.fetchall(query, (cutoff,))

    return [_row_to_event(row).to_dict() for row in rows]


def get_agent_activity(
    agent_name: str,
    days: int = 7,
) -> list[dict]:
    """
    Get all activity for a specific agent over the last N days.

    Args:
        agent_name: Name of the agent (e.g., 'discovery', 'validator')
                   Will match actors starting with 'agent:' prefix
        days: Number of days to look back (default 7)

    Returns:
        List of event dictionaries ordered by timestamp descending
    """
    db = get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Match both 'agent:name' format and just 'name' if it starts with agent:
    actor_pattern = f"agent:{agent_name}"

    query = """
        SELECT lineage_id, event_type, timestamp, actor,
               hypothesis_id, experiment_id, details, parent_lineage_id
        FROM lineage
        WHERE timestamp >= ?
          AND (actor = ? OR actor LIKE ?)
        ORDER BY timestamp DESC
    """

    rows = db.fetchall(query, (cutoff, actor_pattern, f"{actor_pattern}%"))
    return [_row_to_event(row).to_dict() for row in rows]


def get_deployment_trace(hypothesis_id: str) -> list[dict]:
    """
    Get the full trace from hypothesis creation to deployment decision.

    This is useful for answering "Why was this strategy deployed?" by showing
    all validation and approval events.

    Args:
        hypothesis_id: The hypothesis ID to trace

    Returns:
        List of event dictionaries filtered to deployment-relevant events
    """
    chain = get_hypothesis_chain(hypothesis_id)

    # Filter to deployment-relevant event types
    deployment_types = {
        EventType.HYPOTHESIS_CREATED.value,
        EventType.EXPERIMENT_RUN.value,
        EventType.VALIDATION_PASSED.value,
        EventType.VALIDATION_FAILED.value,
        EventType.DEPLOYMENT_APPROVED.value,
        EventType.DEPLOYMENT_REJECTED.value,
    }

    return [event for event in chain if event["event_type"] in deployment_types]


def get_events_between(
    start_event_id: int,
    end_event_id: int,
) -> list[dict]:
    """
    Get all events that occurred between two specific events (by timestamp).

    Useful for understanding what happened between validation and deployment.

    Args:
        start_event_id: Lineage ID of the starting event
        end_event_id: Lineage ID of the ending event

    Returns:
        List of event dictionaries in chronological order
    """
    db = get_db()

    # Get timestamps for the boundary events
    start = db.fetchone(
        "SELECT timestamp FROM lineage WHERE lineage_id = ?",
        (start_event_id,),
    )
    end = db.fetchone(
        "SELECT timestamp FROM lineage WHERE lineage_id = ?",
        (end_event_id,),
    )

    if start is None or end is None:
        logger.warning(f"Could not find events {start_event_id} and/or {end_event_id}")
        return []

    start_time = start[0]
    end_time = end[0]

    # Ensure correct order
    if start_time > end_time:
        start_time, end_time = end_time, start_time

    query = """
        SELECT lineage_id, event_type, timestamp, actor,
               hypothesis_id, experiment_id, details, parent_lineage_id
        FROM lineage
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
    """

    rows = db.fetchall(query, (start_time, end_time))
    return [_row_to_event(row).to_dict() for row in rows]


def summarize_hypothesis_status(hypothesis_id: str) -> dict[str, Any]:
    """
    Get a summary of the current status of a hypothesis based on lineage.

    Returns:
        Dictionary with:
        - hypothesis_id: The hypothesis ID
        - created_at: When the hypothesis was created
        - created_by: Who created it
        - last_event: Most recent event
        - experiment_count: Number of experiments run
        - validation_status: 'passed', 'failed', or 'pending'
        - deployment_status: 'approved', 'rejected', or 'pending'
        - event_count: Total number of events
    """
    chain = get_hypothesis_chain(hypothesis_id)

    if not chain:
        return {
            "hypothesis_id": hypothesis_id,
            "error": "No lineage events found",
        }

    # Find creation event
    creation_event = next(
        (e for e in chain if e["event_type"] == EventType.HYPOTHESIS_CREATED.value),
        None,
    )

    # Count experiments
    experiment_ids = {e["experiment_id"] for e in chain if e["experiment_id"]}

    # Determine validation status (most recent validation event wins)
    validation_status = "pending"
    for event in reversed(chain):
        if event["event_type"] == EventType.VALIDATION_PASSED.value:
            validation_status = "passed"
            break
        elif event["event_type"] == EventType.VALIDATION_FAILED.value:
            validation_status = "failed"
            break

    # Determine deployment status
    deployment_status = "pending"
    for event in reversed(chain):
        if event["event_type"] == EventType.DEPLOYMENT_APPROVED.value:
            deployment_status = "approved"
            break
        elif event["event_type"] == EventType.DEPLOYMENT_REJECTED.value:
            deployment_status = "rejected"
            break

    return {
        "hypothesis_id": hypothesis_id,
        "created_at": creation_event["timestamp"] if creation_event else None,
        "created_by": creation_event["actor"] if creation_event else None,
        "last_event": chain[-1] if chain else None,
        "experiment_count": len(experiment_ids),
        "validation_status": validation_status,
        "deployment_status": deployment_status,
        "event_count": len(chain),
    }
