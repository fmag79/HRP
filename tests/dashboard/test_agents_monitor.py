"""Tests for agents monitor backend functions."""

from __future__ import annotations


def test_agents_monitor_module_exists():
    """agents_monitor module should exist with core functions."""
    from hrp.dashboard.agents_monitor import (
        get_all_agent_status,
        get_timeline,
        AgentStatus
    )
    assert callable(get_all_agent_status)
    assert callable(get_timeline)
