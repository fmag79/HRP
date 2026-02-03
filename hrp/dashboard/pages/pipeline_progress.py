"""Pipeline Progress dashboard page.

Combines the Kanban board and agent panel with tabbed layout to track
hypotheses through the research pipeline.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from hrp.dashboard.components.agent_panel import render_agent_panel
from hrp.dashboard.components.pipeline_kanban import render_kanban_column
from hrp.dashboard.pipeline_data import (
    PIPELINE_STAGES,
    get_hypothesis_pipeline_stages,
    get_stage_counts,
)


# Stage mappings for tabs
DISCOVERY_STAGES = ["draft", "testing"]
VALIDATION_STAGES = ["ml_audit", "quant_dev", "kill_gate"]
REVIEW_STAGES = ["stress_test", "risk_review", "cio_review", "human_approval", "deployed"]


def _render_summary_stats(stage_counts: dict[str, int]) -> None:
    """Render summary statistics at the top of the page."""
    # Calculate totals for each tab
    discovery_total = sum(stage_counts.get(s, 0) for s in DISCOVERY_STAGES)
    validation_total = sum(stage_counts.get(s, 0) for s in VALIDATION_STAGES)
    review_total = sum(stage_counts.get(s, 0) for s in REVIEW_STAGES)
    grand_total = discovery_total + validation_total + review_total

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Total Active
            </div>
            <div style="color: #f1f5f9; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {grand_total}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #60a5fa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Discovery
            </div>
            <div style="color: #60a5fa; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {discovery_total}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #f59e0b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Validation
            </div>
            <div style="color: #f59e0b; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {validation_total}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #10b981; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Review
            </div>
            <div style="color: #10b981; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {review_total}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_kanban_for_stages(
    hypotheses: list[dict[str, Any]],
    stage_keys: list[str],
) -> None:
    """Render a Kanban board for the given stages."""
    # Filter hypotheses by stage
    hypotheses_by_stage: dict[str, list[dict[str, Any]]] = {key: [] for key in stage_keys}

    for h in hypotheses:
        stage = h.get("pipeline_stage")
        if stage in hypotheses_by_stage:
            hypotheses_by_stage[stage].append(h)

    # Create columns for each stage
    columns = st.columns(len(stage_keys))

    for col, stage_key in zip(columns, stage_keys):
        with col:
            stage_hypotheses = hypotheses_by_stage.get(stage_key, [])
            render_kanban_column(stage_key, stage_hypotheses)


def render() -> None:
    """Main entry point for the Pipeline Progress page."""
    # Page header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Pipeline Progress
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Track hypotheses through the research pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # Load data
    with st.spinner("Loading pipeline data..."):
        hypotheses = get_hypothesis_pipeline_stages(exclude_terminal=True)
        stage_counts = get_stage_counts()

    # Summary stats
    _render_summary_stats(stage_counts)

    st.divider()

    # Agent panel
    render_agent_panel()

    st.divider()

    # Tab layout for pipeline stages
    tab_discovery, tab_validation, tab_review = st.tabs([
        "Discovery",
        "Validation",
        "Review",
    ])

    with tab_discovery:
        st.markdown("""
        <p style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 1rem;">
            New hypotheses being explored and initial testing phase.
        </p>
        """, unsafe_allow_html=True)
        _render_kanban_for_stages(hypotheses, DISCOVERY_STAGES)

    with tab_validation:
        st.markdown("""
        <p style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 1rem;">
            ML quality audits, production backtesting, and kill gate enforcement.
        </p>
        """, unsafe_allow_html=True)
        _render_kanban_for_stages(hypotheses, VALIDATION_STAGES)

    with tab_review:
        st.markdown("""
        <p style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 1rem;">
            Stress testing, risk assessment, CIO scoring, and final human approval.
        </p>
        """, unsafe_allow_html=True)
        _render_kanban_for_stages(hypotheses, REVIEW_STAGES)


# Allow running as standalone page for development
if __name__ == "__main__":
    st.set_page_config(
        page_title="HRP - Pipeline Progress",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    render()
