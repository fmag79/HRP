"""Agents Monitor backend functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrp.api.platform import PlatformAPI


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


def get_all_agent_status(api: PlatformAPI) -> list[AgentStatus]:
    """Get current status of all agents from lineage events."""
    # Placeholder implementation
    return []


def get_timeline(
    api: PlatformAPI,
    agents: list[str] | None = None,
    statuses: list[str] | None = None,
    date_range: tuple | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get historical timeline of agent events."""
    # Placeholder implementation
    return []
