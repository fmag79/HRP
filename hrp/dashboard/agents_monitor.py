"""Agents Monitor backend functions."""

from __future__ import annotations

import duckdb
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType


# Database path
DB_PATH = Path.home() / "hrp-data" / "hrp.duckdb"


def _get_lineage_read_only(actor: str | None = None, limit: int = 100) -> list[dict]:
    """Get lineage events using read-only database connection."""
    conditions = []
    params = []

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

    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            rows = con.execute(query, params).fetchall()
            return [
                {
                    "lineage_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "actor": row[3],
                    "hypothesis_id": row[4],
                    "experiment_id": row[5],
                    "details": row[6],
                    "parent_lineage_id": row[7],
                }
                for row in rows
            ]
        finally:
            con.close()
    except Exception as e:
        # If read-only fails (DB doesn't exist or other error), return empty
        return []


# Agent registry with actor IDs and display names
AGENT_REGISTRY: dict[str, dict[str, str]] = {
    "signal-scientist": {
        "actor": "agent:signal-scientist",
        "name": "Signal Scientist",
    },
    "alpha-researcher": {
        "actor": "agent:alpha-researcher",
        "name": "Alpha Researcher",
    },
    "code-materializer": {
        "actor": "agent:code-materializer",
        "name": "Code Materializer",
    },
    "ml-scientist": {
        "actor": "agent:ml-scientist",
        "name": "ML Scientist",
    },
    "ml-quality-sentinel": {
        "actor": "agent:ml-quality-sentinel",
        "name": "ML Quality Sentinel",
    },
    "quant-developer": {
        "actor": "agent:quant-developer",
        "name": "Quant Developer",
    },
    "pipeline-orchestrator": {
        "actor": "agent:pipeline-orchestrator",
        "name": "Pipeline Orchestrator",
    },
    "validation-analyst": {
        "actor": "agent:validation-analyst",
        "name": "Validation Analyst",
    },
    "risk-manager": {
        "actor": "agent:risk-manager",
        "name": "Risk Manager",
    },
    "cio": {
        "actor": "agent:cio",
        "name": "CIO Agent",
    },
    "report-generator": {
        "actor": "agent:report-generator",
        "name": "Report Generator",
    },
}


@dataclass
class AgentStatus:
    """Status of a single agent."""
    agent_id: str
    name: str
    status: str  # running, completed, failed, idle
    last_event: dict | None
    elapsed_seconds: int | None
    current_hypothesis: str | None
    progress_percent: float | None
    stats: dict | None


def _infer_agent_status(events: list[dict]) -> str:
    """
    Infer agent status from recent lineage events.

    Logic:
    1. No events → IDLE
    2. Has START but no COMPLETE (within 5 min) → RUNNING
    3. Latest event has error/FAILED → FAILED
    4. Latest event is COMPLETE → COMPLETED
    5. No recent events (no activity in 24 hours) → IDLE (but show last state)
    """
    if not events:
        return "idle"

    now = datetime.now(timezone.utc)
    latest = events[0]  # Events are ordered by timestamp DESC
    latest_time = datetime.fromisoformat(latest["timestamp"])

    # Check if event is stale (no activity in 24 hours)
    # But still show the last status, just mark as old
    is_stale = (now - latest_time).total_seconds() > 86400  # 24 hours

    # Check for agent start without complete
    has_start = any(e["event_type"] == EventType.AGENT_RUN_START.value for e in events)
    has_complete = any(
        e["event_type"] == EventType.AGENT_RUN_COMPLETE.value for e in events
    )

    if has_start and not has_complete:
        # Started but not completed - check if within 5 minutes
        start_time = datetime.fromisoformat(events[-1]["timestamp"])  # Oldest event
        if (now - start_time).total_seconds() < 300:
            return "running"
        # If it's been longer, might be stale running
        if not is_stale:
            return "running"

    # Check for failed status in latest event
    if "failed" in latest["event_type"].lower() or latest.get("details", {}).get("error"):
        return "failed"

    # Check for completed
    if latest["event_type"] == EventType.AGENT_RUN_COMPLETE.value:
        return "completed"

    # Default to showing the last meaningful status
    # Check the most recent event type
    latest_type = latest["event_type"]
    if "complete" in latest_type.lower():
        return "completed"
    elif "start" in latest_type.lower():
        return "running" if not is_stale else "idle"
    elif "failed" in latest_type.lower():
        return "failed"

    # Default to idle if no clear status
    return "idle"


def get_all_agent_status(api: PlatformAPI | None = None) -> list[AgentStatus]:
    """Get current status of all agents from lineage events."""
    statuses = []

    for agent_id, agent_info in AGENT_REGISTRY.items():
        # Get recent events for this agent (using read-only DB access)
        events = _get_lineage_read_only(actor=agent_info["actor"], limit=50)

        # Infer status
        status = _infer_agent_status(events)

        # Extract last event info
        last_event = events[0] if events else None

        # Calculate elapsed time if running
        elapsed_seconds = None
        if status == "running" and events:
            start_event = next(
                (e for e in events if e["event_type"] == EventType.AGENT_RUN_START.value),
                None,
            )
            if start_event:
                start_time = datetime.fromisoformat(start_event["timestamp"])
                elapsed_seconds = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds()
                )

        # Extract hypothesis ID if available
        current_hypothesis = last_event.get("hypothesis_id") if last_event else None

        statuses.append(
            AgentStatus(
                agent_id=agent_id,
                name=agent_info["name"],
                status=status,
                last_event=last_event,
                elapsed_seconds=elapsed_seconds,
                current_hypothesis=current_hypothesis,
                progress_percent=None,  # Will be filled by progress tracking
                stats=None,  # Will be filled by stats extraction
            )
        )

    return statuses


def get_timeline(
    api: PlatformAPI | None = None,
    agents: list[str] | None = None,
    statuses: list[str] | None = None,
    date_range: tuple | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get historical timeline of agent events."""
    # Build actor filter from agent IDs
    actors = None
    if agents:
        actors = [AGENT_REGISTRY[a]["actor"] for a in agents if a in AGENT_REGISTRY]

    # Get lineage events for specified actors (or all if None)
    events = []
    for agent_id, agent_info in AGENT_REGISTRY.items():
        if actors and agent_info["actor"] not in actors:
            continue

        # Use read-only database access
        agent_events = _get_lineage_read_only(actor=agent_info["actor"], limit=limit)

        # Enrich events with agent display name
        for event in agent_events:
            event["agent_name"] = agent_info["name"]
            event["agent_id"] = agent_id

        events.extend(agent_events)

    # Sort by timestamp descending
    events.sort(key=lambda e: e["timestamp"], reverse=True)

    # Apply date range filter if specified
    if date_range:
        start_date, end_date = date_range
        events = [
            e for e in events
            if start_date <= datetime.fromisoformat(e["timestamp"]).date() <= end_date
        ]

    # Apply limit
    return events[:limit]
