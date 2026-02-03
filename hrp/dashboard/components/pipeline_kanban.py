"""
Kanban board component for the Pipeline Progress dashboard.

Renders hypothesis cards organized by pipeline stage with status indicators,
metrics, and quick actions.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from hrp.dashboard.pipeline_data import PIPELINE_STAGES, format_time_in_stage


# Color scheme for card statuses
CARD_COLORS = {
    "running": {
        "bg": "rgba(59, 130, 246, 0.15)",
        "border": "rgba(59, 130, 246, 0.4)",
        "indicator": "#3b82f6",
    },
    "pass": {
        "bg": "rgba(16, 185, 129, 0.15)",
        "border": "rgba(16, 185, 129, 0.4)",
        "indicator": "#10b981",
    },
    "warning": {
        "bg": "rgba(245, 158, 11, 0.15)",
        "border": "rgba(245, 158, 11, 0.4)",
        "indicator": "#f59e0b",
    },
    "fail": {
        "bg": "rgba(239, 68, 68, 0.15)",
        "border": "rgba(239, 68, 68, 0.4)",
        "indicator": "#ef4444",
    },
    "waiting": {
        "bg": "rgba(107, 114, 128, 0.15)",
        "border": "rgba(107, 114, 128, 0.4)",
        "indicator": "#6b7280",
    },
}

# Status icons
STATUS_ICONS = {
    "running": "\U0001F504",  # rotating arrows
    "pass": "\u2705",  # check mark
    "warning": "\u26A0\uFE0F",  # warning
    "fail": "\u274C",  # cross mark
    "waiting": "\u23F3",  # hourglass
}


def _get_card_status(hypothesis: dict[str, Any]) -> str:
    """
    Determine the card status based on hypothesis state and metrics.

    Args:
        hypothesis: Hypothesis data dict

    Returns:
        Status key: running, pass, warning, fail, or waiting
    """
    status = hypothesis.get("status", "")
    pipeline_stage = hypothesis.get("pipeline_stage", "")
    metrics = hypothesis.get("metrics", {})
    last_event = hypothesis.get("last_event")

    # Deployed hypotheses are always pass
    if status == "deployed":
        return "pass"

    # Rejected hypotheses are always fail
    if status == "rejected":
        return "fail"

    # Draft hypotheses are waiting
    if status == "draft":
        return "waiting"

    # Check for warning conditions based on metrics
    if metrics:
        sharpe = metrics.get("sharpe")
        stability = metrics.get("stability")

        # Low Sharpe is a warning
        if sharpe is not None and sharpe < 0.5:
            return "warning"

        # Poor stability is a warning (lower is better, > 1.5 is concerning)
        if stability is not None and stability > 1.5:
            return "warning"

    # If there's a recent event, consider it running
    if last_event:
        return "running"

    # Check if in an active processing stage
    active_stages = ["testing", "ml_audit", "quant_dev", "kill_gate", "stress_test", "risk_review", "cio_review"]
    if pipeline_stage in active_stages:
        return "running"

    # Default to waiting if in human_approval
    if pipeline_stage == "human_approval":
        return "waiting"

    return "pass"


def _truncate_title(title: str, max_length: int = 30) -> str:
    """Truncate title to max length with ellipsis."""
    if len(title) <= max_length:
        return title
    return title[: max_length - 3] + "..."


def _format_metrics_html(metrics: dict[str, Any]) -> str:
    """Format metrics as HTML badges."""
    if not metrics:
        return ""

    badges = []
    if metrics.get("ic") is not None:
        ic_val = metrics["ic"]
        color = "#10b981" if ic_val > 0.03 else "#f59e0b" if ic_val > 0 else "#ef4444"
        badges.append(f'<span style="background: {color}20; color: {color}; padding: 2px 6px; '
                      f'border-radius: 4px; font-size: 0.7rem; margin-right: 4px;">IC: {ic_val:.3f}</span>')

    if metrics.get("sharpe") is not None:
        sharpe_val = metrics["sharpe"]
        color = "#10b981" if sharpe_val > 1.0 else "#f59e0b" if sharpe_val > 0.5 else "#ef4444"
        badges.append(f'<span style="background: {color}20; color: {color}; padding: 2px 6px; '
                      f'border-radius: 4px; font-size: 0.7rem; margin-right: 4px;">SR: {sharpe_val:.2f}</span>')

    if metrics.get("stability") is not None:
        stab_val = metrics["stability"]
        color = "#10b981" if stab_val <= 1.0 else "#f59e0b" if stab_val <= 1.5 else "#ef4444"
        badges.append(f'<span style="background: {color}20; color: {color}; padding: 2px 6px; '
                      f'border-radius: 4px; font-size: 0.7rem;">Stab: {stab_val:.2f}</span>')

    return "".join(badges)


def render_hypothesis_card(hypothesis: dict[str, Any]) -> None:
    """
    Render a single hypothesis card.

    Args:
        hypothesis: Hypothesis data dict containing:
            - hypothesis_id: e.g., HYP-2026-012
            - title: Hypothesis title
            - time_in_stage_seconds: Seconds in current stage
            - status: Current status
            - pipeline_stage: Current pipeline stage key
            - metrics: Optional dict with ic, sharpe, stability
    """
    card_status = _get_card_status(hypothesis)
    colors = CARD_COLORS[card_status]
    status_icon = STATUS_ICONS[card_status]

    hypothesis_id = hypothesis.get("hypothesis_id", "Unknown")
    title = hypothesis.get("title", "Untitled")
    time_seconds = hypothesis.get("time_in_stage_seconds", 0)
    metrics = hypothesis.get("metrics", {})

    truncated_title = _truncate_title(title)
    time_str = format_time_in_stage(time_seconds)
    metrics_html = _format_metrics_html(metrics)

    # Card HTML
    card_html = f"""
    <div style="
        background: {colors['bg']};
        border: 1px solid {colors['border']};
        border-left: 3px solid {colors['indicator']};
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <span style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                color: #94a3b8;
                font-weight: 500;
            ">{hypothesis_id}</span>
            <span style="font-size: 0.9rem;" title="{card_status}">{status_icon}</span>
        </div>
        <div style="
            color: #f1f5f9;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        " title="{title}">{truncated_title}</div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="
                color: #6b7280;
                font-size: 0.7rem;
            ">\u23F3 {time_str}</span>
            <div>{metrics_html}</div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_card_actions(hypothesis: dict[str, Any]) -> None:
    """
    Render quick action buttons based on the hypothesis stage.

    Args:
        hypothesis: Hypothesis data dict
    """
    pipeline_stage = hypothesis.get("pipeline_stage", "")
    hypothesis_id = hypothesis.get("hypothesis_id", "")
    status = hypothesis.get("status", "")

    # Define available actions per stage
    if pipeline_stage == "draft":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Review", key=f"review_{hypothesis_id}", use_container_width=True):
                st.session_state[f"action_{hypothesis_id}"] = "review"
                st.info(f"Review {hypothesis_id} - run Alpha Researcher")
        with col2:
            if st.button("Delete", key=f"delete_{hypothesis_id}", use_container_width=True, type="secondary"):
                st.session_state[f"action_{hypothesis_id}"] = "delete"
                st.warning(f"Delete {hypothesis_id}?")

    elif pipeline_stage == "testing":
        if st.button("Run Validation", key=f"validate_{hypothesis_id}", use_container_width=True):
            st.session_state[f"action_{hypothesis_id}"] = "validate"
            st.info(f"Run ML Scientist validation on {hypothesis_id}")

    elif pipeline_stage in ["ml_audit", "quant_dev", "kill_gate", "stress_test", "risk_review", "cio_review"]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Details", key=f"details_{hypothesis_id}", use_container_width=True):
                st.session_state[f"action_{hypothesis_id}"] = "details"
        with col2:
            if st.button("Skip", key=f"skip_{hypothesis_id}", use_container_width=True, type="secondary"):
                st.session_state[f"action_{hypothesis_id}"] = "skip"
                st.warning(f"Skip stage for {hypothesis_id}?")

    elif pipeline_stage == "human_approval":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve", key=f"approve_{hypothesis_id}", use_container_width=True, type="primary"):
                st.session_state[f"action_{hypothesis_id}"] = "approve"
                st.success(f"Approve {hypothesis_id} for deployment")
        with col2:
            if st.button("Reject", key=f"reject_{hypothesis_id}", use_container_width=True, type="secondary"):
                st.session_state[f"action_{hypothesis_id}"] = "reject"
                st.error(f"Reject {hypothesis_id}")

    elif pipeline_stage == "deployed" or status == "deployed":
        if st.button("Undeploy", key=f"undeploy_{hypothesis_id}", use_container_width=True, type="secondary"):
            st.session_state[f"action_{hypothesis_id}"] = "undeploy"
            st.warning(f"Undeploy {hypothesis_id}?")


def render_kanban_column(stage_key: str, hypotheses: list[dict[str, Any]]) -> None:
    """
    Render a single Kanban column with hypothesis cards.

    Args:
        stage_key: Pipeline stage key (e.g., 'draft', 'testing')
        hypotheses: List of hypotheses in this stage
    """
    # Find stage metadata
    stage_meta = next((s for s in PIPELINE_STAGES if s["key"] == stage_key), None)

    if stage_meta:
        stage_name = stage_meta["name"]
        stage_desc = stage_meta["description"]
    else:
        stage_name = stage_key.replace("_", " ").title()
        stage_desc = ""

    count = len(hypotheses)

    # Column header with count badge
    header_html = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(100, 116, 139, 0.3);
    ">
        <div>
            <span style="
                color: #f1f5f9;
                font-weight: 600;
                font-size: 0.95rem;
            ">{stage_name}</span>
        </div>
        <span style="
            background: rgba(100, 116, 139, 0.3);
            color: #94a3b8;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
            font-weight: 500;
        ">{count}</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Stage description
    if stage_desc:
        st.caption(stage_desc)

    # Render cards or empty state
    if not hypotheses:
        st.markdown("""
        <div style="
            color: #6b7280;
            font-size: 0.8rem;
            text-align: center;
            padding: 1.5rem 0.5rem;
            border: 1px dashed rgba(100, 116, 139, 0.3);
            border-radius: 6px;
        ">No hypotheses</div>
        """, unsafe_allow_html=True)
    else:
        for hypothesis in hypotheses:
            render_hypothesis_card(hypothesis)

            # Expandable details with actions
            with st.expander("Details", expanded=False):
                # Thesis preview
                thesis = hypothesis.get("thesis", "")
                if thesis:
                    st.markdown(f"**Thesis:** {thesis[:200]}{'...' if len(thesis) > 200 else ''}")

                # Last event info
                last_event = hypothesis.get("last_event")
                if last_event:
                    event_type = last_event.get("event_type", "Unknown")
                    st.markdown(f"**Last Event:** `{event_type}`")

                # Full metrics
                metrics = hypothesis.get("metrics", {})
                if metrics:
                    st.markdown("**Metrics:**")
                    metrics_cols = st.columns(3)
                    col_idx = 0
                    for key, val in metrics.items():
                        if val is not None:
                            with metrics_cols[col_idx % 3]:
                                st.metric(key.upper(), f"{val:.3f}" if isinstance(val, float) else val)
                            col_idx += 1

                st.divider()

                # Quick actions
                render_card_actions(hypothesis)


def render_kanban_board(
    hypotheses_by_stage: dict[str, list[dict[str, Any]]],
    visible_stages: list[str] | None = None,
) -> None:
    """
    Render the full Kanban board with multiple columns.

    Args:
        hypotheses_by_stage: Dict mapping stage keys to list of hypotheses
        visible_stages: Optional list of stage keys to display (default: all)
    """
    if visible_stages is None:
        visible_stages = [s["key"] for s in PIPELINE_STAGES]

    # Filter to visible stages
    stages_to_render = [s for s in PIPELINE_STAGES if s["key"] in visible_stages]

    if not stages_to_render:
        st.warning("No stages selected to display.")
        return

    # Create columns for each stage
    columns = st.columns(len(stages_to_render))

    for col, stage in zip(columns, stages_to_render):
        with col:
            stage_hypotheses = hypotheses_by_stage.get(stage["key"], [])
            render_kanban_column(stage["key"], stage_hypotheses)
