"""
Hypothesis Registry for HRP.

Manages research hypotheses through their full lifecycle:
draft -> testing -> validated/rejected -> deployed

All database access goes through hrp.data.db.get_db().
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal

from loguru import logger

from hrp.data.db import get_db


# Valid status values
HypothesisStatus = Literal["draft", "testing", "validated", "rejected", "deployed", "deleted"]

# Valid transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"testing", "deleted"},
    "testing": {"validated", "rejected", "draft", "deleted"},
    "validated": {"deployed", "rejected", "deleted"},
    "rejected": {"draft", "deleted"},  # Can reopen as draft
    "deployed": {"validated", "deleted"},  # Can undeploy back to validated
    "deleted": set(),  # Terminal state
}


@dataclass
class HypothesisRecord:
    """Dataclass representing a hypothesis record."""

    hypothesis_id: str
    title: str
    thesis: str
    testable_prediction: str
    falsification_criteria: str
    status: str
    created_at: datetime
    created_by: str
    updated_at: datetime | None
    outcome: str | None
    confidence_score: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime objects to ISO strings
        if d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        if d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def get_next_hypothesis_id() -> str:
    """
    Generate the next hypothesis ID in format 'HYP-{year}-{sequence}'.

    Example: HYP-2025-001, HYP-2025-002, etc.
    Sequence resets each year.
    """
    db = get_db()
    current_year = datetime.now().year
    prefix = f"HYP-{current_year}-"

    # Find the highest sequence number for this year
    query = """
        SELECT hypothesis_id
        FROM hypotheses
        WHERE hypothesis_id LIKE ?
        ORDER BY hypothesis_id DESC
        LIMIT 1
    """
    result = db.fetchone(query, (f"{prefix}%",))

    if result:
        # Extract sequence number and increment
        last_id = result[0]
        sequence = int(last_id.split("-")[-1]) + 1
    else:
        sequence = 1

    return f"{prefix}{sequence:03d}"


def _row_to_record(row: tuple) -> HypothesisRecord:
    """Convert a database row to a HypothesisRecord."""
    return HypothesisRecord(
        hypothesis_id=row[0],
        title=row[1],
        thesis=row[2],
        testable_prediction=row[3],
        falsification_criteria=row[4],
        status=row[5],
        created_at=row[6],
        created_by=row[7],
        updated_at=row[8],
        outcome=row[9],
        confidence_score=float(row[10]) if row[10] is not None else None,
    )


def _row_to_dict(row: tuple) -> dict:
    """Convert a database row to a dictionary."""
    return _row_to_record(row).to_dict()


def create_hypothesis(
    title: str,
    thesis: str,
    prediction: str,
    falsification: str,
    actor: str = "user",
) -> str:
    """
    Create a new hypothesis.

    Args:
        title: Short descriptive title
        thesis: The core hypothesis statement
        prediction: Testable prediction that would support the thesis
        falsification: Criteria that would disprove the hypothesis
        actor: Who created it ('user' or 'agent:<name>')

    Returns:
        The generated hypothesis ID (e.g., 'HYP-2025-001')
    """
    db = get_db()
    hypothesis_id = get_next_hypothesis_id()

    query = """
        INSERT INTO hypotheses (
            hypothesis_id, title, thesis, testable_prediction,
            falsification_criteria, status, created_at, created_by
        ) VALUES (?, ?, ?, ?, ?, 'draft', CURRENT_TIMESTAMP, ?)
    """

    db.execute(query, (hypothesis_id, title, thesis, prediction, falsification, actor))

    logger.info(f"Created hypothesis {hypothesis_id}: {title}")

    # Log to lineage
    _log_lineage(
        event_type="hypothesis_created",
        hypothesis_id=hypothesis_id,
        actor=actor,
        details={"title": title},
    )

    return hypothesis_id


def get_hypothesis(hypothesis_id: str) -> dict | None:
    """
    Get a hypothesis by ID.

    Args:
        hypothesis_id: The hypothesis ID (e.g., 'HYP-2025-001')

    Returns:
        Dictionary with hypothesis data, or None if not found
    """
    db = get_db()

    query = """
        SELECT hypothesis_id, title, thesis, testable_prediction,
               falsification_criteria, status, created_at, created_by,
               updated_at, outcome, confidence_score
        FROM hypotheses
        WHERE hypothesis_id = ?
    """

    result = db.fetchone(query, (hypothesis_id,))

    if result:
        return _row_to_dict(result)
    return None


def list_hypotheses(
    status: str | None = None,
    actor: str | None = None,
) -> list[dict]:
    """
    List hypotheses with optional filters.

    Args:
        status: Filter by status (e.g., 'draft', 'testing')
        actor: Filter by creator (e.g., 'user', 'agent:discovery')

    Returns:
        List of hypothesis dictionaries
    """
    db = get_db()

    query = """
        SELECT hypothesis_id, title, thesis, testable_prediction,
               falsification_criteria, status, created_at, created_by,
               updated_at, outcome, confidence_score
        FROM hypotheses
        WHERE status != 'deleted'
    """
    params: list = []

    if status:
        query += " AND status = ?"
        params.append(status)

    if actor:
        query += " AND created_by = ?"
        params.append(actor)

    query += " ORDER BY created_at DESC"

    results = db.fetchall(query, tuple(params) if params else None)

    return [_row_to_dict(row) for row in results]


def update_hypothesis(
    hypothesis_id: str,
    status: str | None = None,
    outcome: str | None = None,
    confidence_score: float | None = None,
) -> bool:
    """
    Update a hypothesis.

    Args:
        hypothesis_id: The hypothesis ID
        status: New status (must be valid transition)
        outcome: Outcome text (typically set when validated/rejected)
        confidence_score: Confidence score 0.0-1.0 (set when validated)

    Returns:
        True if updated, False if not found or invalid transition

    Raises:
        ValueError: If status transition is invalid or validation requirements not met
    """
    db = get_db()

    # Get current hypothesis
    current = get_hypothesis(hypothesis_id)
    if not current:
        logger.warning(f"Hypothesis {hypothesis_id} not found")
        return False

    current_status = current["status"]

    # Validate status transition
    if status:
        if status not in VALID_TRANSITIONS.get(current_status, set()):
            raise ValueError(
                f"Invalid status transition: {current_status} -> {status}. "
                f"Valid transitions: {VALID_TRANSITIONS.get(current_status, set())}"
            )

        # Check validation requirements for 'validated' status (only when coming from 'testing')
        # When undeploying (deployed→validated), skip this check
        if status == "validated" and current_status == "testing":
            validation_result = validate_hypothesis_status(hypothesis_id)
            if not validation_result["can_validate"]:
                raise ValueError(
                    f"Cannot validate hypothesis: {validation_result['reasons']}"
                )

        # Check requirements for 'deployed' status
        if status == "deployed" and current_status != "validated":
            raise ValueError("Can only deploy a validated hypothesis")

    # Build update query
    updates = []
    params: list = []

    if status:
        updates.append("status = ?")
        params.append(status)

    if outcome is not None:
        updates.append("outcome = ?")
        params.append(outcome)

    if confidence_score is not None:
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        updates.append("confidence_score = ?")
        params.append(confidence_score)

    if not updates:
        return False

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(hypothesis_id)

    query = f"UPDATE hypotheses SET {', '.join(updates)} WHERE hypothesis_id = ?"

    db.execute(query, tuple(params))

    logger.info(f"Updated hypothesis {hypothesis_id}: status={status}, outcome={outcome}")

    # Log to lineage
    _log_lineage(
        event_type="hypothesis_updated",
        hypothesis_id=hypothesis_id,
        actor="system",
        details={"status": status, "outcome": outcome, "confidence_score": confidence_score},
    )

    return True


def delete_hypothesis(hypothesis_id: str) -> bool:
    """
    Soft delete a hypothesis (sets status to 'deleted').

    Args:
        hypothesis_id: The hypothesis ID

    Returns:
        True if deleted, False if not found
    """
    current = get_hypothesis(hypothesis_id)
    if not current:
        logger.warning(f"Hypothesis {hypothesis_id} not found")
        return False

    # Soft delete by setting status
    db = get_db()
    query = """
        UPDATE hypotheses
        SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
        WHERE hypothesis_id = ?
    """
    db.execute(query, (hypothesis_id,))

    logger.info(f"Deleted hypothesis {hypothesis_id}")

    _log_lineage(
        event_type="hypothesis_deleted",
        hypothesis_id=hypothesis_id,
        actor="system",
        details={},
    )

    return True


def link_experiment(
    hypothesis_id: str,
    experiment_id: str,
    relationship: str = "primary",
) -> bool:
    """
    Link an experiment to a hypothesis.

    Args:
        hypothesis_id: The hypothesis ID
        experiment_id: MLflow experiment/run ID
        relationship: Type of relationship ('primary', 'supporting', 'exploratory')

    Returns:
        True if linked successfully
    """
    db = get_db()

    # Verify hypothesis exists
    hypothesis = get_hypothesis(hypothesis_id)
    if not hypothesis:
        logger.warning(f"Hypothesis {hypothesis_id} not found")
        return False

    # Insert link (will fail silently if already exists due to PK constraint)
    query = """
        INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id, relationship)
        VALUES (?, ?, ?)
        ON CONFLICT (hypothesis_id, experiment_id) DO UPDATE SET relationship = ?
    """
    db.execute(query, (hypothesis_id, experiment_id, relationship, relationship))

    logger.info(f"Linked experiment {experiment_id} to hypothesis {hypothesis_id}")

    _log_lineage(
        event_type="experiment_linked",
        hypothesis_id=hypothesis_id,
        experiment_id=experiment_id,
        actor="system",
        details={"relationship": relationship},
    )

    return True


def get_experiments(hypothesis_id: str) -> list[str]:
    """
    Get all experiment IDs linked to a hypothesis.

    Args:
        hypothesis_id: The hypothesis ID

    Returns:
        List of experiment IDs
    """
    db = get_db()

    query = """
        SELECT experiment_id
        FROM hypothesis_experiments
        WHERE hypothesis_id = ?
        ORDER BY created_at DESC
    """

    results = db.fetchall(query, (hypothesis_id,))

    return [row[0] for row in results]


def get_experiment_links(hypothesis_id: str) -> list[dict]:
    """
    Get all experiment links with relationship info.

    Args:
        hypothesis_id: The hypothesis ID

    Returns:
        List of dicts with experiment_id, relationship, created_at
    """
    db = get_db()

    query = """
        SELECT experiment_id, relationship, created_at
        FROM hypothesis_experiments
        WHERE hypothesis_id = ?
        ORDER BY created_at DESC
    """

    results = db.fetchall(query, (hypothesis_id,))

    return [
        {
            "experiment_id": row[0],
            "relationship": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
        }
        for row in results
    ]


def validate_hypothesis_status(hypothesis_id: str) -> dict:
    """
    Check if a hypothesis is ready for validation.

    Validation requirements:
    - Must be in 'testing' status
    - Must have at least one linked experiment
    - Should have outcome and confidence_score set (warning if not)

    Args:
        hypothesis_id: The hypothesis ID

    Returns:
        Dictionary with:
            - can_validate: bool
            - reasons: list of strings explaining why validation is blocked
            - warnings: list of strings with non-blocking issues
    """
    hypothesis = get_hypothesis(hypothesis_id)

    if not hypothesis:
        return {
            "can_validate": False,
            "reasons": ["Hypothesis not found"],
            "warnings": [],
        }

    reasons: list[str] = []
    warnings: list[str] = []

    # Check status
    if hypothesis["status"] != "testing":
        reasons.append(f"Status must be 'testing', currently '{hypothesis['status']}'")

    # Check for linked experiments
    experiments = get_experiments(hypothesis_id)
    if not experiments:
        reasons.append("Must have at least one linked experiment")

    # Check for outcome (warning only)
    if not hypothesis.get("outcome"):
        warnings.append("No outcome text recorded")

    # Check for confidence score (warning only)
    if hypothesis.get("confidence_score") is None:
        warnings.append("No confidence score set")

    return {
        "can_validate": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
    }


def _log_lineage(
    event_type: str,
    hypothesis_id: str | None = None,
    experiment_id: str | None = None,
    actor: str = "system",
    details: dict | None = None,
) -> None:
    """Log an event to the lineage table."""
    db = get_db()

    # Convert details dict to JSON string
    import json

    details_json = json.dumps(details) if details else None

    query = """
        INSERT INTO lineage (event_type, actor, hypothesis_id, experiment_id, details)
        VALUES (?, ?, ?, ?, ?)
    """

    try:
        db.execute(query, (event_type, actor, hypothesis_id, experiment_id, details_json))
    except Exception as e:
        # Don't fail the main operation if lineage logging fails
        logger.warning(f"Failed to log lineage event: {e}")
