"""
Hypotheses Page for HRP Dashboard.

Provides functionality to:
- Browse and filter hypotheses
- View hypothesis details with linked experiments and lineage
- Create new hypotheses
- Update hypothesis status
"""

from datetime import datetime
from typing import Any

import streamlit as st

from hrp.research.hypothesis import (
    VALID_TRANSITIONS,
    create_hypothesis,
    get_experiment_links,
    get_hypothesis,
    list_hypotheses,
    update_hypothesis,
)
from hrp.research.lineage import get_hypothesis_chain


# Status display configuration
STATUS_COLORS: dict[str, str] = {
    "draft": "gray",
    "testing": "blue",
    "validated": "green",
    "rejected": "red",
    "deployed": "orange",
}

STATUS_ICONS: dict[str, str] = {
    "draft": "pencil",
    "testing": "hourglass",
    "validated": "check",
    "rejected": "x",
    "deployed": "rocket",
}

ALL_STATUSES: list[str] = ["draft", "testing", "validated", "rejected", "deployed"]


def format_datetime(dt: datetime | str | None) -> str:
    """Format a datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M")


def render_status_badge(status: str) -> None:
    """Render a colored status badge."""
    color = STATUS_COLORS.get(status, "gray")
    st.markdown(
        f'<span style="background-color: {color}; color: white; '
        f'padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">'
        f"{status.upper()}</span>",
        unsafe_allow_html=True,
    )


def render_hypothesis_card(h: dict[str, Any]) -> None:
    """Render a compact hypothesis card in the list view."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(f"**{h['hypothesis_id']}**: {h.get('title', 'Untitled')}")

    with col2:
        render_status_badge(h.get("status", "draft"))

    with col3:
        created_at = h.get("created_at")
        st.caption(format_datetime(created_at))


def render_hypothesis_details(hypothesis_id: str) -> None:
    """Render full hypothesis details including experiments and lineage."""
    h = get_hypothesis(hypothesis_id)

    if h is None:
        st.error(f"Hypothesis {hypothesis_id} not found.")
        return

    # Header with ID and status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(h.get("title", "Untitled"))
        st.caption(f"ID: {hypothesis_id}")
    with col2:
        render_status_badge(h.get("status", "draft"))

    st.divider()

    # Main content in tabs
    detail_tabs = st.tabs(["Details", "Experiments", "Lineage", "Update Status"])

    # --- Details Tab ---
    with detail_tabs[0]:
        st.markdown("### Thesis")
        st.info(h.get("thesis", "No thesis provided."))

        st.markdown("### Testable Prediction")
        st.success(h.get("testable_prediction", "No prediction provided."))

        st.markdown("### Falsification Criteria")
        st.warning(h.get("falsification_criteria", "No falsification criteria provided."))

        # Metadata
        st.markdown("### Metadata")
        meta_col1, meta_col2, meta_col3 = st.columns(3)

        with meta_col1:
            st.metric("Created By", h.get("created_by", "Unknown"))

        with meta_col2:
            st.metric("Created At", format_datetime(h.get("created_at")))

        with meta_col3:
            st.metric("Last Updated", format_datetime(h.get("updated_at")))

        # Outcome and confidence (if set)
        if h.get("outcome"):
            st.markdown("### Outcome")
            st.write(h.get("outcome"))

        if h.get("confidence_score") is not None:
            st.metric("Confidence Score", f"{h.get('confidence_score'):.2%}")

    # --- Experiments Tab ---
    with detail_tabs[1]:
        st.markdown("### Linked Experiments")

        experiment_links = get_experiment_links(hypothesis_id)

        if not experiment_links:
            st.info("No experiments linked to this hypothesis yet.")
            st.caption(
                "Run a backtest with this hypothesis to link experiments, "
                "or use the Platform API to link existing experiments."
            )
        else:
            for exp in experiment_links:
                with st.container():
                    exp_col1, exp_col2, exp_col3 = st.columns([3, 1, 1])

                    with exp_col1:
                        st.code(exp["experiment_id"], language=None)

                    with exp_col2:
                        relationship = exp.get("relationship", "primary")
                        st.caption(f"Relationship: {relationship}")

                    with exp_col3:
                        st.caption(format_datetime(exp.get("created_at")))

                    st.divider()

    # --- Lineage Tab ---
    with detail_tabs[2]:
        st.markdown("### Audit Trail")

        lineage_events = get_hypothesis_chain(hypothesis_id)

        if not lineage_events:
            st.info("No lineage events recorded for this hypothesis.")
        else:
            for event in lineage_events:
                with st.container():
                    event_col1, event_col2 = st.columns([3, 1])

                    with event_col1:
                        event_type = event.get("event_type", "unknown")
                        st.markdown(f"**{event_type}**")

                        # Show details if available
                        details = event.get("details", {})
                        if details:
                            detail_items = []
                            for key, value in details.items():
                                if value is not None:
                                    detail_items.append(f"{key}: {value}")
                            if detail_items:
                                st.caption(" | ".join(detail_items))

                    with event_col2:
                        st.caption(f"By: {event.get('actor', 'unknown')}")
                        st.caption(format_datetime(event.get("timestamp")))

                    st.divider()

    # --- Update Status Tab ---
    with detail_tabs[3]:
        st.markdown("### Update Hypothesis Status")

        current_status = h.get("status", "draft")
        valid_next_statuses = VALID_TRANSITIONS.get(current_status, set())

        if not valid_next_statuses:
            st.warning(
                f"Hypothesis is in '{current_status}' state and cannot be transitioned further."
            )
        else:
            st.info(f"Current status: **{current_status}**")
            st.caption(f"Valid transitions: {', '.join(sorted(valid_next_statuses))}")

            with st.form(f"update_status_{hypothesis_id}"):
                new_status = st.selectbox(
                    "New Status",
                    options=sorted(valid_next_statuses),
                    key=f"new_status_{hypothesis_id}",
                )

                outcome_text = st.text_area(
                    "Outcome (optional)",
                    value=h.get("outcome") or "",
                    help="Describe the outcome of testing/validation.",
                    key=f"outcome_{hypothesis_id}",
                )

                confidence = st.slider(
                    "Confidence Score (optional)",
                    min_value=0.0,
                    max_value=1.0,
                    value=h.get("confidence_score") or 0.5,
                    step=0.01,
                    help="How confident are you in the validation result?",
                    key=f"confidence_{hypothesis_id}",
                )

                submitted = st.form_submit_button("Update Status")

                if submitted:
                    try:
                        # Note: deployed status requires user approval through Platform API
                        if new_status == "deployed":
                            st.error(
                                "Deployment must be approved through the Platform API "
                                "using api.approve_deployment(). "
                                "This ensures proper audit trail and permission checks."
                            )
                        else:
                            update_hypothesis(
                                hypothesis_id=hypothesis_id,
                                status=new_status,
                                outcome=outcome_text if outcome_text else None,
                                confidence_score=confidence if new_status == "validated" else None,
                            )
                            st.success(f"Status updated to '{new_status}'.")
                            st.rerun()
                    except ValueError as e:
                        st.error(f"Failed to update status: {e}")


def render_create_form() -> None:
    """Render the form for creating a new hypothesis."""
    st.markdown("### Create New Hypothesis")

    st.info(
        "A good hypothesis should be specific, measurable, and falsifiable. "
        "Define clear criteria for what would prove or disprove your thesis."
    )

    with st.form("new_hypothesis", clear_on_submit=True):
        title = st.text_input(
            "Title",
            placeholder="e.g., Momentum predicts returns",
            help="A short, descriptive title for your hypothesis.",
        )

        thesis = st.text_area(
            "Thesis",
            placeholder="e.g., Stocks with high 12-month returns continue outperforming the market over the following month.",
            help="The core hypothesis statement. What do you believe to be true?",
            height=100,
        )

        prediction = st.text_area(
            "Testable Prediction",
            placeholder="e.g., Top decile momentum stocks outperform SPY by at least 3% annually.",
            help="A specific, measurable prediction that would support your thesis.",
            height=100,
        )

        falsification = st.text_area(
            "Falsification Criteria",
            placeholder="e.g., Sharpe ratio < SPY or p-value > 0.05 in backtests.",
            help="What evidence would disprove your hypothesis? Be specific.",
            height=100,
        )

        actor = st.text_input(
            "Created By",
            value="user",
            help="Who is creating this hypothesis? Use 'user' or 'agent:<name>'.",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Create Hypothesis", type="primary")

        if submitted:
            # Validation
            errors = []
            if not title.strip():
                errors.append("Title is required.")
            if not thesis.strip():
                errors.append("Thesis is required.")
            if not prediction.strip():
                errors.append("Testable prediction is required.")
            if not falsification.strip():
                errors.append("Falsification criteria is required.")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    hypothesis_id = create_hypothesis(
                        title=title.strip(),
                        thesis=thesis.strip(),
                        prediction=prediction.strip(),
                        falsification=falsification.strip(),
                        actor=actor.strip(),
                    )
                    st.success(f"Created hypothesis: **{hypothesis_id}**")
                    st.balloons()

                    # Store the new hypothesis ID for immediate viewing
                    st.session_state.selected_hypothesis = hypothesis_id
                    st.session_state.active_tab = "Browse"

                except Exception as e:
                    st.error(f"Failed to create hypothesis: {e}")


def render_browse_tab() -> None:
    """Render the browse/list view for hypotheses."""
    # Filters
    filter_col1, filter_col2 = st.columns([1, 3])

    with filter_col1:
        status_options = ["All"] + ALL_STATUSES
        status_filter = st.selectbox(
            "Filter by Status",
            options=status_options,
            index=0,
            key="status_filter",
        )

    with filter_col2:
        search_term = st.text_input(
            "Search",
            placeholder="Search by ID or title...",
            key="search_term",
        )

    # Fetch hypotheses
    status_param = status_filter if status_filter != "All" else None
    hypotheses = list_hypotheses(status=status_param)

    # Apply search filter
    if search_term:
        search_lower = search_term.lower()
        hypotheses = [
            h for h in hypotheses
            if search_lower in h.get("hypothesis_id", "").lower()
            or search_lower in h.get("title", "").lower()
        ]

    # Summary stats
    st.divider()
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    all_hypotheses = list_hypotheses()
    status_counts = {}
    for h in all_hypotheses:
        status = h.get("status", "draft")
        status_counts[status] = status_counts.get(status, 0) + 1

    with stat_col1:
        st.metric("Total", len(all_hypotheses))
    with stat_col2:
        st.metric("Draft", status_counts.get("draft", 0))
    with stat_col3:
        st.metric("Testing", status_counts.get("testing", 0))
    with stat_col4:
        st.metric("Validated", status_counts.get("validated", 0))
    with stat_col5:
        st.metric("Deployed", status_counts.get("deployed", 0))

    st.divider()

    # Results count
    st.caption(f"Showing {len(hypotheses)} hypothesis(es)")

    if not hypotheses:
        st.info("No hypotheses found. Create one in the 'Create New' tab.")
        return

    # List hypotheses
    for h in hypotheses:
        hypothesis_id = h.get("hypothesis_id", "Unknown")

        with st.expander(
            f"{hypothesis_id}: {h.get('title', 'Untitled')} [{h.get('status', 'draft')}]",
            expanded=(st.session_state.get("selected_hypothesis") == hypothesis_id),
        ):
            render_hypothesis_details(hypothesis_id)


def render() -> None:
    """Main render function for the Hypotheses page."""
    st.title("Hypotheses")
    st.caption(
        "Manage research hypotheses through their lifecycle: "
        "draft -> testing -> validated/rejected -> deployed"
    )

    # Initialize session state
    if "selected_hypothesis" not in st.session_state:
        st.session_state.selected_hypothesis = None

    # Main tabs
    tab1, tab2 = st.tabs(["Browse", "Create New"])

    with tab1:
        render_browse_tab()

    with tab2:
        render_create_form()


# Allow running as standalone page for development
if __name__ == "__main__":
    st.set_page_config(
        page_title="HRP - Hypotheses",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    render()
