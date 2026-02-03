"""Pipeline data layer for the Pipeline Progress dashboard.

Provides functions to query hypotheses with their current pipeline stage
based on lineage events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger


# Database path
DB_PATH = Path.home() / "hrp-data" / "hrp.duckdb"

# Pipeline stage definitions ordered by progression
PIPELINE_STAGES = [
    {
        "key": "draft",
        "name": "Draft",
        "description": "New hypothesis awaiting review",
        "tab": "discovery",
        "lineage_event": None,  # Initial state
        "eligible_status": ["draft"],
    },
    {
        "key": "testing",
        "name": "Testing",
        "description": "ML Scientist validation in progress",
        "tab": "discovery",
        "lineage_event": "alpha_researcher_complete",
        "eligible_status": ["testing"],
    },
    {
        "key": "ml_audit",
        "name": "ML Audit",
        "description": "Quality Sentinel audit",
        "tab": "validation",
        "lineage_event": "ml_scientist_validation",
        "eligible_status": ["validated"],
    },
    {
        "key": "quant_dev",
        "name": "Quant Dev",
        "description": "Production backtesting",
        "tab": "validation",
        "lineage_event": "ml_quality_sentinel_audit",
        "eligible_status": ["validated"],
    },
    {
        "key": "kill_gate",
        "name": "Kill Gate",
        "description": "Kill gate enforcement",
        "tab": "validation",
        "lineage_event": "quant_developer_complete",
        "eligible_status": ["validated"],
    },
    {
        "key": "stress_test",
        "name": "Stress Test",
        "description": "Validation analyst stress testing",
        "tab": "review",
        "lineage_event": "kill_gate_enforcer_complete",
        "eligible_status": ["validated"],
    },
    {
        "key": "risk_review",
        "name": "Risk Review",
        "description": "Risk manager assessment",
        "tab": "review",
        "lineage_event": "validation_analyst_complete",
        "eligible_status": ["validated"],
    },
    {
        "key": "cio_review",
        "name": "CIO Review",
        "description": "CIO agent scoring",
        "tab": "review",
        "lineage_event": "risk_manager_assessment",
        "eligible_status": ["validated"],
    },
    {
        "key": "human_approval",
        "name": "Human Approval",
        "description": "Awaiting human CIO decision",
        "tab": "review",
        "lineage_event": "cio_agent_decision",
        "eligible_status": ["validated"],
    },
    {
        "key": "deployed",
        "name": "Deployed",
        "description": "Strategy deployed to paper trading",
        "tab": "review",
        "lineage_event": None,  # Uses status field
        "eligible_status": ["deployed"],
    },
]

# Map lineage events to stage keys for lookup
LINEAGE_TO_STAGE = {
    stage["lineage_event"]: stage["key"]
    for stage in PIPELINE_STAGES
    if stage["lineage_event"]
}

# Stage keys in order for progression comparison
STAGE_ORDER = [stage["key"] for stage in PIPELINE_STAGES]


def _get_db_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Get a database connection."""
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def get_pipeline_stage_for_hypothesis(
    hypothesis_id: str,
    status: str,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """
    Determine the pipeline stage for a hypothesis based on lineage events.

    Args:
        hypothesis_id: The hypothesis ID
        status: Current hypothesis status
        con: Optional existing database connection

    Returns:
        Dict with stage info: key, name, entered_at, last_event
    """
    should_close = con is None
    if con is None:
        con = _get_db_connection()

    try:
        # Handle terminal statuses
        if status == "deployed":
            return {
                "key": "deployed",
                "name": "Deployed",
                "entered_at": None,
                "last_event": None,
            }

        if status == "rejected":
            return {
                "key": "rejected",
                "name": "Rejected",
                "entered_at": None,
                "last_event": None,
            }

        if status == "deleted":
            return {
                "key": "deleted",
                "name": "Deleted",
                "entered_at": None,
                "last_event": None,
            }

        # Get relevant lineage events for this hypothesis
        # Look for the most recent stage-transition event
        relevant_events = [
            "alpha_researcher_complete",
            "ml_scientist_validation",
            "ml_quality_sentinel_audit",
            "quant_developer_complete",
            "kill_gate_enforcer_complete",
            "validation_analyst_complete",
            "risk_manager_assessment",
            "cio_agent_decision",
        ]

        placeholders = ", ".join(["?" for _ in relevant_events])
        query = f"""
            SELECT event_type, timestamp, details
            FROM lineage
            WHERE hypothesis_id = ?
              AND event_type IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT 1
        """

        result = con.execute(query, [hypothesis_id] + relevant_events).fetchone()

        if result is None:
            # No progression events - still in draft or early testing
            if status == "draft":
                return {
                    "key": "draft",
                    "name": "Draft",
                    "entered_at": None,
                    "last_event": None,
                }
            else:
                # In testing but no ML Scientist event yet
                return {
                    "key": "testing",
                    "name": "Testing",
                    "entered_at": None,
                    "last_event": None,
                }

        event_type, timestamp, details = result

        # Find the NEXT stage after the completed event
        completed_stage = LINEAGE_TO_STAGE.get(event_type)
        if completed_stage:
            stage_idx = STAGE_ORDER.index(completed_stage)
            # Move to next stage if not at end
            if stage_idx + 1 < len(STAGE_ORDER):
                next_stage_key = STAGE_ORDER[stage_idx + 1]
                next_stage = next(s for s in PIPELINE_STAGES if s["key"] == next_stage_key)
                return {
                    "key": next_stage_key,
                    "name": next_stage["name"],
                    "entered_at": timestamp,
                    "last_event": {
                        "event_type": event_type,
                        "timestamp": timestamp,
                        "details": details,
                    },
                }

        # Fallback to current stage based on status
        if status == "draft":
            return {"key": "draft", "name": "Draft", "entered_at": None, "last_event": None}
        elif status == "testing":
            return {"key": "testing", "name": "Testing", "entered_at": None, "last_event": None}
        else:
            return {"key": "validated", "name": "Validated", "entered_at": None, "last_event": None}

    finally:
        if should_close:
            con.close()


def get_hypothesis_pipeline_stages(
    exclude_terminal: bool = True,
) -> list[dict[str, Any]]:
    """
    Get all hypotheses with their current pipeline stage.

    Args:
        exclude_terminal: If True, exclude rejected/deleted hypotheses

    Returns:
        List of hypothesis dicts with pipeline stage info
    """
    con = _get_db_connection()

    try:
        # Build status filter
        if exclude_terminal:
            status_filter = "AND status NOT IN ('rejected', 'deleted')"
        else:
            status_filter = ""

        # Get all hypotheses with metadata
        query = f"""
            SELECT
                hypothesis_id,
                title,
                thesis,
                status,
                created_at,
                updated_at,
                metadata
            FROM hypotheses
            WHERE 1=1 {status_filter}
            ORDER BY created_at DESC
        """

        rows = con.execute(query).fetchall()

        results = []
        now = datetime.now(timezone.utc)

        for row in rows:
            hypothesis_id, title, thesis, status, created_at, updated_at, metadata = row

            # Get pipeline stage
            stage_info = get_pipeline_stage_for_hypothesis(hypothesis_id, status, con)

            # Calculate time in stage
            if stage_info.get("entered_at"):
                entered = stage_info["entered_at"]
                if isinstance(entered, str):
                    entered = datetime.fromisoformat(entered.replace("Z", "+00:00"))
                if entered.tzinfo is None:
                    entered = entered.replace(tzinfo=timezone.utc)
                time_in_stage = (now - entered).total_seconds()
            else:
                # Use created_at as fallback
                if created_at:
                    if isinstance(created_at, str):
                        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        created = created_at
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    time_in_stage = (now - created).total_seconds()
                else:
                    time_in_stage = 0

            # Extract metrics from metadata if available
            metrics = {}
            if metadata and isinstance(metadata, dict):
                # Look for validation results
                if "validation_analyst_review" in metadata:
                    val_review = metadata["validation_analyst_review"]
                    metrics["sharpe"] = val_review.get("sharpe")
                    metrics["max_drawdown"] = val_review.get("max_drawdown")
                if "ml_scientist_validation" in metadata:
                    ml_val = metadata["ml_scientist_validation"]
                    metrics["ic"] = ml_val.get("mean_ic")
                    metrics["stability"] = ml_val.get("stability_score")

            results.append({
                "hypothesis_id": hypothesis_id,
                "title": title or "Untitled",
                "thesis": thesis,
                "status": status,
                "pipeline_stage": stage_info["key"],
                "pipeline_stage_name": stage_info["name"],
                "stage_entered_at": stage_info.get("entered_at"),
                "time_in_stage_seconds": int(time_in_stage),
                "last_event": stage_info.get("last_event"),
                "metrics": metrics,
                "created_at": created_at,
                "updated_at": updated_at,
            })

        return results

    finally:
        con.close()


def get_hypotheses_by_stage(stage_key: str) -> list[dict[str, Any]]:
    """Get all hypotheses at a specific pipeline stage."""
    all_hypotheses = get_hypothesis_pipeline_stages()
    return [h for h in all_hypotheses if h["pipeline_stage"] == stage_key]


def get_stage_counts() -> dict[str, int]:
    """Get count of hypotheses at each pipeline stage."""
    all_hypotheses = get_hypothesis_pipeline_stages()
    counts = {stage["key"]: 0 for stage in PIPELINE_STAGES}
    counts["rejected"] = 0

    for h in all_hypotheses:
        stage = h["pipeline_stage"]
        if stage in counts:
            counts[stage] += 1

    return counts


def get_eligible_hypotheses_for_agent(agent_key: str) -> list[dict[str, Any]]:
    """
    Get hypotheses eligible for a specific agent to process.

    Args:
        agent_key: The agent identifier

    Returns:
        List of eligible hypotheses
    """
    all_hypotheses = get_hypothesis_pipeline_stages()

    # Agent eligibility rules
    eligibility_rules = {
        "signal_scientist": lambda h: False,  # Doesn't process existing hypotheses
        "alpha_researcher": lambda h: h["status"] == "draft",
        "ml_scientist": lambda h: h["status"] == "testing",
        "ml_quality_sentinel": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "ml_audit",
        "quant_developer": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "quant_dev",
        "kill_gate_enforcer": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "kill_gate",
        "validation_analyst": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "stress_test",
        "risk_manager": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "risk_review",
        "cio_agent": lambda h: h["status"] == "validated" and h["pipeline_stage"] == "cio_review",
        "report_generator": lambda h: False,  # Doesn't process hypotheses
    }

    rule = eligibility_rules.get(agent_key)
    if rule is None:
        logger.warning(f"Unknown agent key: {agent_key}")
        return []

    return [h for h in all_hypotheses if rule(h)]


def format_time_in_stage(seconds: int) -> str:
    """Format time in stage as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    else:
        days = seconds // 86400
        return f"{days}d"
