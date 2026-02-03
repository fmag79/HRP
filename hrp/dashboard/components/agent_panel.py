"""Agent launcher panel for the Pipeline Progress dashboard.

Provides a registry of all 10 agents with UI controls to run them
on eligible hypotheses.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
from loguru import logger

from hrp.dashboard.pipeline_data import get_eligible_hypotheses_for_agent

# Lazy imports for agents to avoid circular dependencies
def _get_signal_scientist():
    from hrp.agents.signal_scientist import SignalScientist
    return SignalScientist

def _get_alpha_researcher():
    from hrp.agents.alpha_researcher import AlphaResearcher
    return AlphaResearcher

def _get_ml_scientist():
    from hrp.agents.ml_scientist import MLScientist
    return MLScientist

def _get_ml_quality_sentinel():
    from hrp.agents.ml_quality_sentinel import MLQualitySentinel
    return MLQualitySentinel

def _get_quant_developer():
    from hrp.agents.quant_developer import QuantDeveloper
    return QuantDeveloper

def _get_kill_gate_enforcer():
    from hrp.agents.kill_gate_enforcer import KillGateEnforcer
    return KillGateEnforcer

def _get_validation_analyst():
    from hrp.agents.validation_analyst import ValidationAnalyst
    return ValidationAnalyst

def _get_risk_manager():
    from hrp.agents.risk_manager import RiskManager
    return RiskManager

def _get_cio_agent():
    from hrp.agents.cio import CIOAgent
    return CIOAgent

def _get_report_generator():
    from hrp.agents.report_generator import ReportGenerator
    return ReportGenerator


# Agent registry with metadata for each agent
AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    "signal_scientist": {
        "class": _get_signal_scientist,
        "name": "Signal Scientist",
        "icon": "🔬",
        "description": "Automated signal discovery using IC analysis",
        "requires_hypothesis": False,
    },
    "alpha_researcher": {
        "class": _get_alpha_researcher,
        "name": "Alpha Researcher",
        "icon": "🔍",
        "description": "Reviews draft hypotheses, promotes to testing",
        "requires_hypothesis": True,
    },
    "ml_scientist": {
        "class": _get_ml_scientist,
        "name": "ML Scientist",
        "icon": "🧠",
        "description": "Walk-forward validation of testing hypotheses",
        "requires_hypothesis": True,
    },
    "ml_quality_sentinel": {
        "class": _get_ml_quality_sentinel,
        "name": "ML Quality Sentinel",
        "icon": "🛡️",
        "description": "Audits experiments for overfitting",
        "requires_hypothesis": True,
    },
    "quant_developer": {
        "class": _get_quant_developer,
        "name": "Quant Developer",
        "icon": "💻",
        "description": "Production backtesting with costs",
        "requires_hypothesis": True,
    },
    "kill_gate_enforcer": {
        "class": _get_kill_gate_enforcer,
        "name": "Kill Gate Enforcer",
        "icon": "🚫",
        "description": "Enforces kill gates on hypotheses",
        "requires_hypothesis": True,
    },
    "validation_analyst": {
        "class": _get_validation_analyst,
        "name": "Validation Analyst",
        "icon": "📊",
        "description": "Pre-deployment stress testing",
        "requires_hypothesis": True,
    },
    "risk_manager": {
        "class": _get_risk_manager,
        "name": "Risk Manager",
        "icon": "⚠️",
        "description": "Portfolio risk assessment with veto power",
        "requires_hypothesis": True,
    },
    "cio_agent": {
        "class": _get_cio_agent,
        "name": "CIO Agent",
        "icon": "👔",
        "description": "Scores hypotheses across 4 dimensions",
        "requires_hypothesis": True,
    },
    "report_generator": {
        "class": _get_report_generator,
        "name": "Report Generator",
        "icon": "📝",
        "description": "Daily/weekly research summaries",
        "requires_hypothesis": False,
    },
}


def run_agent_sync(
    agent_key: str,
    hypothesis_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute an agent synchronously with send_alerts=False.

    Args:
        agent_key: Key from AGENT_REGISTRY
        hypothesis_ids: Optional list of hypothesis IDs to process

    Returns:
        Dict with 'success' bool and 'result' or 'error' key
    """
    try:
        agent_info = AGENT_REGISTRY.get(agent_key)
        if agent_info is None:
            return {"success": False, "error": f"Unknown agent: {agent_key}"}

        # Get the agent class via lazy loader
        agent_class_loader: Callable = agent_info["class"]
        agent_class = agent_class_loader()

        # Instantiate and run based on agent type
        if agent_key == "signal_scientist":
            # SignalScientist doesn't process hypotheses
            agent = agent_class(
                symbols=None,  # Will use universe
                features=None,  # Will use all features
                create_hypotheses=True,
            )
            result = agent.run()

        elif agent_key == "report_generator":
            # ReportGenerator doesn't process hypotheses
            agent = agent_class(report_type="daily")
            result = agent.run()

        elif agent_key == "alpha_researcher":
            agent = agent_class(hypothesis_ids=hypothesis_ids)
            result = agent.run()

        elif agent_key == "ml_scientist":
            agent = agent_class(hypothesis_ids=hypothesis_ids)
            result = agent.run()

        elif agent_key == "ml_quality_sentinel":
            agent = agent_class(
                hypothesis_ids=hypothesis_ids,
                send_alerts=False,
            )
            result = agent.run()

        elif agent_key == "quant_developer":
            agent = agent_class(hypothesis_ids=hypothesis_ids)
            result = agent.run()

        elif agent_key == "kill_gate_enforcer":
            agent = agent_class(hypothesis_ids=hypothesis_ids)
            result = agent.run()

        elif agent_key == "validation_analyst":
            agent = agent_class(
                hypothesis_ids=hypothesis_ids,
                send_alerts=False,
            )
            result = agent.run()

        elif agent_key == "risk_manager":
            agent = agent_class(
                hypothesis_ids=hypothesis_ids,
                send_alerts=False,
            )
            result = agent.run()

        elif agent_key == "cio_agent":
            agent = agent_class()
            result = agent.run()

        else:
            return {"success": False, "error": f"Agent not implemented: {agent_key}"}

        logger.info(f"Agent {agent_key} completed successfully")
        return {"success": True, "result": result}

    except Exception as e:
        logger.exception(f"Agent {agent_key} failed: {e}")
        return {"success": False, "error": str(e)}


def render_agent_card(agent_key: str, agent_info: dict[str, Any]) -> None:
    """
    Render a single agent card with run button and status.

    Args:
        agent_key: Registry key for the agent
        agent_info: Agent metadata from registry
    """
    icon = agent_info["icon"]
    name = agent_info["name"]
    description = agent_info["description"]
    requires_hypothesis = agent_info["requires_hypothesis"]

    # Get eligible hypotheses count
    if requires_hypothesis:
        eligible = get_eligible_hypotheses_for_agent(agent_key)
        eligible_count = len(eligible)
        eligible_ids = [h["hypothesis_id"] for h in eligible]
    else:
        eligible_count = None
        eligible_ids = None

    # Card container
    with st.container():
        st.markdown(f"**{icon} {name}**")
        st.caption(description)

        # Build button label
        if requires_hypothesis:
            button_label = f"Run ({eligible_count})"
            disabled = eligible_count == 0
        else:
            button_label = "Run"
            disabled = False

        # Run button with unique key
        button_key = f"run_agent_{agent_key}"
        if st.button(button_label, key=button_key, disabled=disabled, use_container_width=True):
            with st.spinner(f"Running {name}..."):
                result = run_agent_sync(agent_key, hypothesis_ids=eligible_ids)

            if result["success"]:
                st.success(f"{name} completed!")
            else:
                st.error(f"Error: {result['error']}")

            # Trigger rerun to refresh data
            st.rerun()


def render_agent_panel() -> None:
    """
    Render the collapsible agent panel with 2x5 grid of agent cards.
    """
    with st.expander("🤖 Agent Panel", expanded=False):
        st.markdown("Launch agents to process hypotheses through the pipeline.")

        # Create 2 rows of 5 agents each
        agent_keys = list(AGENT_REGISTRY.keys())

        # First row: first 5 agents
        row1_cols = st.columns(5)
        for i, col in enumerate(row1_cols):
            if i < len(agent_keys):
                agent_key = agent_keys[i]
                with col:
                    render_agent_card(agent_key, AGENT_REGISTRY[agent_key])

        # Second row: remaining 5 agents
        row2_cols = st.columns(5)
        for i, col in enumerate(row2_cols):
            idx = i + 5
            if idx < len(agent_keys):
                agent_key = agent_keys[idx]
                with col:
                    render_agent_card(agent_key, AGENT_REGISTRY[agent_key])
