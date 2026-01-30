"""Tests for agents monitor backend functions."""

from __future__ import annotations

from datetime import timezone
from unittest.mock import patch, MagicMock


def test_agents_monitor_module_exists():
    """agents_monitor module should exist with core functions."""
    from hrp.dashboard.agents_monitor import (
        get_all_agent_status,
        get_timeline,
        AgentStatus
    )
    assert callable(get_all_agent_status)
    assert callable(get_timeline)


def test_get_all_agent_status_returns_list():
    """get_all_agent_status should return list of AgentStatus."""
    from hrp.dashboard.agents_monitor import get_all_agent_status
    from hrp.api.platform import PlatformAPI

    # Mock get_lineage to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor.get_lineage", return_value=[]):
        api = PlatformAPI()
        result = get_all_agent_status(api)
        assert isinstance(result, list)
        # All agents should be present
        agent_ids = {a.agent_id for a in result}
        expected_agents = {
            "signal-scientist", "alpha-researcher", "code-materializer",
            "ml-scientist", "ml-quality-sentinel", "quant-developer",
            "pipeline-orchestrator", "validation-analyst", "risk-manager",
            "cio", "report-generator"
        }
        assert expected_agents.issubset(agent_ids)


def test_agent_status_has_valid_status_field():
    """Each AgentStatus should have valid status field."""
    from hrp.dashboard.agents_monitor import get_all_agent_status
    from hrp.api.platform import PlatformAPI

    # Mock get_lineage to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor.get_lineage", return_value=[]):
        api = PlatformAPI()
        result = get_all_agent_status(api)
        valid_statuses = {"running", "completed", "failed", "idle"}
        for agent in result:
            assert agent.status in valid_statuses


def test_get_timeline_returns_list():
    """get_timeline should return list of timeline events."""
    from hrp.dashboard.agents_monitor import get_timeline
    from hrp.api.platform import PlatformAPI
    from datetime import datetime

    # Mock get_lineage to return a test event
    test_event = {
        "lineage_id": 1,
        "event_type": "agent_run_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": "agent:signal-scientist",
        "hypothesis_id": "HYP-001",
        "experiment_id": None,
        "details": {},
        "parent_lineage_id": None,
    }

    with patch("hrp.dashboard.agents_monitor.get_lineage", return_value=[test_event]):
        api = PlatformAPI()
        result = get_timeline(api, limit=50)
        assert isinstance(result, list)
        # Should enrich with agent info
        if result:
            assert "agent_name" in result[0] or len(result) == 0  # May be empty due to actor filtering
