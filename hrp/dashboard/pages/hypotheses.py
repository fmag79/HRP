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


# Status display configuration with custom colors
STATUS_COLORS: dict[str, str] = {
    "draft": "#6b7280",      # gray
    "testing": "#3b82f6",    # blue
    "validated": "#10b981",  # green
    "rejected": "#ef4444",   # red
    "deployed": "#a78bfa",   # purple
}

STATUS_BACKGROUNDS: dict[str, str] = {
    "draft": "rgba(107, 114, 128, 0.15)",
    "testing": "rgba(59, 130, 246, 0.15)",
    "validated": "rgba(16, 185, 129, 0.15)",
    "rejected": "rgba(239, 68, 68, 0.15)",
    "deployed": "rgba(167, 139, 250, 0.15)",
}

STATUS_ICONS: dict[str, str] = {
    "draft": "✏️",
    "testing": "⏳",
    "validated": "✅",
    "rejected": "❌",
    "deployed": "🚀",
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
    """Render a colored status badge with custom styling."""
    color = STATUS_COLORS.get(status, "#6b7280")
    bg = STATUS_BACKGROUNDS.get(status, "rgba(107, 114, 128, 0.15)")
    icon = STATUS_ICONS.get(status, "•")

    st.markdown(
        f'<span style="background: {bg}; color: {color}; border: 1px solid {color}; '
        f'padding: 0.25rem 0.625rem; border-radius: 6px; font-size: 0.75rem; '
        f'font-weight: 600; font-family: "JetBrains Mono", monospace; letter-spacing: 0.05em;">'
        f"{icon} {status.upper()}</span>",
        unsafe_allow_html=True,
    )


def render_hypothesis_card(h: dict[str, Any]) -> None:
    """Render a compact hypothesis card in the list view."""
    status = h.get("status", "draft")
    color = STATUS_COLORS.get(status, "#6b7280")
    bg = STATUS_BACKGROUNDS.get(status, "rgba(107, 114, 128, 0.15)")

    st.markdown(f"""
    <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1rem;
                border-left: 3px solid {color}; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="color: {color}; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.875rem;">
                        {h['hypothesis_id']}
                    </span>
                </div>
                <div style="color: #f1f5f9; font-weight: 500; margin-bottom: 0.25rem;">
                    {h.get('title', 'Untitled')}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    {format_datetime(h.get('created_at'))}
                </div>
            </div>
            <div>
                <span style="background: {bg}; color: {color}; border: 1px solid {color};
                           padding: 0.25rem 0.625rem; border-radius: 6px; font-size: 0.7rem;
                           font-weight: 600; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em;">
                    {status}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_hypothesis_details(hypothesis_id: str) -> None:
    """Render full hypothesis details including experiments and lineage."""
    h = get_hypothesis(hypothesis_id)

    if h is None:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 1rem;">
            <div style="color: #f87171;">Hypothesis {hypothesis_id} not found.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Header with ID and status - custom styled
    status = h.get("status", "draft")
    color = STATUS_COLORS.get(status, "#6b7280")
    bg = STATUS_BACKGROUNDS.get(status, "rgba(107, 114, 128, 0.15)")

    st.markdown(f"""
    <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;">
            <div style="flex: 1;">
                <div style="color: #6b7280; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;">
                    {hypothesis_id}
                </div>
                <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.25rem;">
                    {h.get("title", "Untitled")}
                </div>
            </div>
            <div>
                <span style="background: {bg}; color: {color}; border: 1px solid {color};
                           padding: 0.375rem 0.75rem; border-radius: 6px; font-size: 0.75rem;
                           font-weight: 600; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em;">
                    {STATUS_ICONS.get(status, '•')} {status}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main content in tabs
    detail_tabs = st.tabs(["📋 Details", "🔬 Experiments", "📜 Lineage", "✏️ Update"])

    # --- Details Tab ---
    with detail_tabs[0]:
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                Thesis
            </div>
            <div style="color: #f1f5f9; line-height: 1.6;">
                {h.get("thesis", "No thesis provided.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="color: #10b981; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                Testable Prediction
            </div>
            <div style="color: #f1f5f9; line-height: 1.6;">
                {h.get("testable_prediction", "No prediction provided.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="color: #f59e0b; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                Falsification Criteria
            </div>
            <div style="color: #f1f5f9; line-height: 1.6;">
                {h.get("falsification_criteria", "No falsification criteria provided.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metadata
        st.markdown("""
        <p style="color: #9ca3af; font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem;">
            Metadata
        </p>
        """, unsafe_allow_html=True)

        meta_col1, meta_col2, meta_col3 = st.columns(3)

        with meta_col1:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #374151; border-radius: 6px; padding: 0.75rem;">
                <div style="color: #6b7280; font-size: 0.75rem; margin-bottom: 0.25rem;">Created By</div>
                <div style="color: #f1f5f9; font-weight: 500;">{h.get("created_by", "Unknown")}</div>
            </div>
            """, unsafe_allow_html=True)

        with meta_col2:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #374151; border-radius: 6px; padding: 0.75rem;">
                <div style="color: #6b7280; font-size: 0.75rem; margin-bottom: 0.25rem;">Created At</div>
                <div style="color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;">{format_datetime(h.get("created_at"))}</div>
            </div>
            """, unsafe_allow_html=True)

        with meta_col3:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #374151; border-radius: 6px; padding: 0.75rem;">
                <div style="color: #6b7280; font-size: 0.75rem; margin-bottom: 0.25rem;">Last Updated</div>
                <div style="color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;">{format_datetime(h.get("updated_at"))}</div>
            </div>
            """, unsafe_allow_html=True)

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
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 2rem;">
        <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
            <span style="font-size: 1.25rem;">💡</span>
            <div>
                <div style="color: #60a5fa; font-weight: 600; margin-bottom: 0.25rem;">
                    Hypothesis Guidelines
                </div>
                <div style="color: #9ca3af; font-size: 0.875rem;">
                    A good hypothesis should be specific, measurable, and falsifiable. Define clear criteria for what would prove or disprove your thesis.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("new_hypothesis", clear_on_submit=True):
        st.markdown("""
        <p style="color: #9ca3af; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
            Hypothesis Details
        </p>
        """, unsafe_allow_html=True)

        title = st.text_input(
            "Title",
            placeholder="e.g., Momentum predicts returns",
            help="A short, descriptive title for your hypothesis.",
            label_visibility="visible",
        )

        thesis = st.text_area(
            "Thesis",
            placeholder="e.g., Stocks with high 12-month returns continue outperforming the market over the following month.",
            help="The core hypothesis statement. What do you believe to be true?",
            height=100,
            label_visibility="visible",
        )

        prediction = st.text_area(
            "Testable Prediction",
            placeholder="e.g., Top decile momentum stocks outperform SPY by at least 3% annually.",
            help="A specific, measurable prediction that would support your thesis.",
            height=100,
            label_visibility="visible",
        )

        falsification = st.text_area(
            "Falsification Criteria",
            placeholder="e.g., Sharpe ratio < SPY or p-value > 0.05 in backtests.",
            help="What evidence would disprove your hypothesis? Be specific.",
            height=100,
            label_visibility="visible",
        )

        actor = st.text_input(
            "Created By",
            value="user",
            help="Who is creating this hypothesis? Use 'user' or 'agent:<name>'.",
            label_visibility="visible",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("➕ Create Hypothesis", type="primary", use_container_width=True)

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
                st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 1rem; margin-top: 1rem;">
                    <div style="color: #f87171; font-weight: 600; margin-bottom: 0.5rem;">Please fix the following errors:</div>
                </div>
                """, unsafe_allow_html=True)
                for error in errors:
                    st.markdown(f"""
                    <div style="color: #fca5a5; font-size: 0.875rem; margin-left: 0.5rem;">
                        • {error}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                try:
                    hypothesis_id = create_hypothesis(
                        title=title.strip(),
                        thesis=thesis.strip(),
                        prediction=prediction.strip(),
                        falsification=falsification.strip(),
                        actor=actor.strip(),
                    )
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1.25rem; margin-top: 1rem;">
                        <div style="color: #10b981; font-weight: 600; font-size: 1.125rem; margin-bottom: 0.5rem;">
                            ✅ Hypothesis Created Successfully
                        </div>
                        <div style="color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;">
                            {hypothesis_id}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

                    # Store the new hypothesis ID for immediate viewing
                    st.session_state.selected_hypothesis = hypothesis_id
                    st.session_state.active_tab = "Browse"

                except Exception as e:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 1rem; margin-top: 1rem;">
                        <div style="color: #f87171;">Failed to create hypothesis: {e}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_browse_tab() -> None:
    """Render the browse/list view for hypotheses."""
    # Filters with custom styling
    st.markdown("""
    <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; gap: 1rem; align-items: center;">
            <div style="flex: 1;">
                <label style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.5rem;">
                    Filter by Status
                </label>
            </div>
            <div style="flex: 3;">
                <label style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.5rem;">
                    Search
                </label>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([1, 3])

    with filter_col1:
        status_options = ["All"] + ALL_STATUSES
        status_filter = st.selectbox(
            "Filter by Status",
            options=status_options,
            index=0,
            label_visibility="collapsed",
            key="status_filter",
        )

    with filter_col2:
        search_term = st.text_input(
            "Search",
            placeholder="🔍 Search by ID or title...",
            label_visibility="collapsed",
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

    # Summary stats with custom styling
    st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

    all_hypotheses = list_hypotheses()
    status_counts = {}
    for h in all_hypotheses:
        status = h.get("status", "draft")
        status_counts[status] = status_counts.get(status, 0) + 1

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    with stat_col1:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Total
            </div>
            <div style="color: #f1f5f9; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {len(all_hypotheses)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col2:
        st.markdown(f"""
        <div style="background: rgba(107, 114, 128, 0.1); border: 1px solid rgba(107, 114, 128, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Draft
            </div>
            <div style="color: #9ca3af; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {status_counts.get("draft", 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col3:
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #60a5fa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Testing
            </div>
            <div style="color: #60a5fa; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {status_counts.get("testing", 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col4:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #10b981; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Validated
            </div>
            <div style="color: #10b981; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {status_counts.get("validated", 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col5:
        st.markdown(f"""
        <div style="background: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.2); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="color: #a78bfa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Deployed
            </div>
            <div style="color: #a78bfa; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                {status_counts.get("deployed", 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

    # Results count
    st.markdown(f"""
    <p style="color: #6b7280; font-size: 0.875rem; margin-bottom: 1.5rem;">
        Showing <span style="color: #f1f5f9; font-weight: 600;">{len(hypotheses)}</span> hypothesis(es)
    </p>
    """, unsafe_allow_html=True)

    if not hypotheses:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            <div>No hypotheses found. Create one in the "Create New" tab.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # List hypotheses with custom styled cards
    for h in hypotheses:
        hypothesis_id = h.get("hypothesis_id", "Unknown")

        with st.expander(
            f"{hypothesis_id}: {h.get('title', 'Untitled')} [{h.get('status', 'draft')}]",
            expanded=(st.session_state.get("selected_hypothesis") == hypothesis_id),
        ):
            render_hypothesis_details(hypothesis_id)


def render() -> None:
    """Main render function for the Hypotheses page."""
    # Load custom CSS
    try:
        with open("hrp/dashboard/static/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # Page header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Hypotheses
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Manage research hypotheses through their lifecycle: draft → testing → validated/rejected → deployed
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "selected_hypothesis" not in st.session_state:
        st.session_state.selected_hypothesis = None

    # Main tabs with custom styling
    tab1, tab2 = st.tabs(["🔍 Browse", "➕ Create New"])

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
