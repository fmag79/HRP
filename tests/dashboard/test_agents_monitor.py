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

    # Mock _get_lineage_read_only to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor._get_lineage_read_only", return_value=[]):
        result = get_all_agent_status()
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

    # Mock _get_lineage_read_only to return empty events (all agents will be idle)
    with patch("hrp.dashboard.agents_monitor._get_lineage_read_only", return_value=[]):
        result = get_all_agent_status()
        valid_statuses = {"running", "completed", "failed", "idle"}
        for agent in result:
            assert agent.status in valid_statuses


def test_get_timeline_returns_list():
    """get_timeline should return list of timeline events."""
    from hrp.dashboard.agents_monitor import get_timeline
    from datetime import datetime

    # Mock _get_lineage_read_only to return a test event
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

    with patch("hrp.dashboard.agents_monitor._get_lineage_read_only", return_value=[test_event]):
        result = get_timeline(limit=50)
        assert isinstance(result, list)
        # Should enrich with agent info
        if result:
            assert "agent_name" in result[0] or len(result) == 0  # May be empty due to actor filtering


def test_status_inference_running():
    """Status inference should detect running agents."""
    from hrp.dashboard.agents_monitor import _infer_agent_status
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    events = [
        {
            "lineage_id": 1,
            "event_type": "agent_run_start",
            "timestamp": (now - timedelta(seconds=30)).isoformat(),
            "actor": "agent:test",
            "hypothesis_id": None,
            "experiment_id": None,
            "details": {},
            "parent_lineage_id": None,
        },
    ]
    status = _infer_agent_status(events)
    assert status == "running"


def test_status_inference_completed():
    """Status inference should detect completed agents."""
    from hrp.dashboard.agents_monitor import _infer_agent_status
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    # Use very recent events (within seconds) to ensure they're not stale
    # NOTE: Events must be sorted DESC by timestamp (newest first)
    events = [
        {
            "lineage_id": 2,
            "event_type": "agent_run_complete",
            "timestamp": (now - timedelta(seconds=2)).isoformat(),
            "actor": "agent:test",
            "hypothesis_id": None,
            "experiment_id": None,
            "details": {},
            "parent_lineage_id": None,
        },
        {
            "lineage_id": 1,
            "event_type": "agent_run_start",
            "timestamp": (now - timedelta(seconds=5)).isoformat(),
            "actor": "agent:test",
            "hypothesis_id": None,
            "experiment_id": None,
            "details": {},
            "parent_lineage_id": None,
        },
    ]
    status = _infer_agent_status(events)
    # Latest event is agent_run_complete, should be completed
    assert status == "completed"


def test_status_inference_idle():
    """Status inference should detect idle agents (no events)."""
    from hrp.dashboard.agents_monitor import _infer_agent_status

    status = _infer_agent_status([])
    assert status == "idle"


def test_status_inference_failed():
    """Status inference should detect failed agents."""
    from hrp.dashboard.agents_monitor import _infer_agent_status
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    events = [
        {
            "lineage_id": 1,
            "event_type": "agent_run_complete",
            "timestamp": (now - timedelta(seconds=60)).isoformat(),
            "actor": "agent:test",
            "hypothesis_id": None,
            "experiment_id": None,
            "details": {"error": "Something went wrong"},
            "parent_lineage_id": None,
        },
    ]
    status = _infer_agent_status(events)
    assert status == "failed"


def test_agent_registry_complete():
    """AGENT_REGISTRY should have all 11 agents."""
    from hrp.dashboard.agents_monitor import AGENT_REGISTRY

    expected_agents = {
        "signal-scientist", "alpha-researcher", "code-materializer",
        "ml-scientist", "ml-quality-sentinel", "quant-developer",
        "pipeline-orchestrator", "validation-analyst", "risk-manager",
        "cio", "report-generator"
    }
    assert set(AGENT_REGISTRY.keys()) == expected_agents

    for agent_id, info in AGENT_REGISTRY.items():
        assert "actor" in info
        assert "name" in info
        assert info["actor"].startswith("agent:")
