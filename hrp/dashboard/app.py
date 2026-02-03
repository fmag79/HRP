"""
HRP Dashboard - Main Application

Multipage Streamlit dashboard for the Hedgefund Research Platform.
Provides access to system status, data health, hypotheses, and experiments.
"""

import sys
from pathlib import Path

# Version from pyproject.toml
DASHBOARD_VERSION = "1.6.0"

# Add project root to Python path so Streamlit can find the hrp module
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime
from typing import Any

import streamlit as st
from loguru import logger

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="HRP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_api() -> Any:
    """
    Get or create the Platform API instance.

    Uses Streamlit session state to maintain a single API instance.
    Handles scheduler database lock conflicts gracefully.
    """
    if "api" not in st.session_state:
        try:
            from hrp.api.platform import PlatformAPI

            st.session_state.api = PlatformAPI()  # Use default read-write mode
            st.session_state.db_error = None
            logger.info("PlatformAPI initialized for dashboard")
        except Exception as e:
            logger.error(f"Failed to initialize PlatformAPI: {e}")
            st.session_state.api = None
            st.session_state.db_error = e
    return st.session_state.api


def render_db_error() -> None:
    """
    Render database error with scheduler conflict resolution UI.

    Called when PlatformAPI initialization fails due to database lock.
    """
    if "db_error" not in st.session_state:
        return

    error = st.session_state.db_error
    if error is None:
        return

    from hrp.dashboard.components import render_scheduler_conflict
    from hrp.utils.scheduler import is_duckdb_lock_error

    if is_duckdb_lock_error(error):
        render_scheduler_conflict(error)
    else:
        st.error("Database Connection Error")
        st.exception(error)


# =============================================================================
# Page Rendering Functions
# =============================================================================


def render_home() -> None:
    """Render the Home page with system status and recent activity."""
    # Page header with gradient text
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Dashboard
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Hedgefund Research Platform — Systematic Trading Strategy Development
        </p>
    </div>
    """, unsafe_allow_html=True)

    api = get_api()

    # Show database error if API initialization failed
    if api is None:
        render_db_error()
        return

    # System Status Section with custom cards
    st.markdown("""
    <p style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 1rem;">
        System Status
    </p>
    """, unsafe_allow_html=True)

    try:
        health = api.health_check()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            api_status = health.get("api", "unknown")
            status_color = "#10b981" if api_status == "ok" else "#ef4444"
            status_text = "ONLINE" if api_status == "ok" else api_status.upper()
            st.markdown(f"""
            <div class="status-card {'status-ok' if api_status == 'ok' else 'status-error'}" style="padding: 1.25rem;">
                <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    API
                </div>
                <div style="color: {status_color}; font-size: 1.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            db_status = health.get("database", "unknown")
            status_color = "#10b981" if db_status == "ok" else "#ef4444"
            status_text = "CONNECTED" if db_status == "ok" else db_status.upper()
            st.markdown(f"""
            <div class="status-card {'status-ok' if db_status == 'ok' else 'status-error'}" style="padding: 1.25rem;">
                <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Database
                </div>
                <div style="color: {status_color}; font-size: 1.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            table_count = len([t for t, v in health.get("tables", {}).items() if v.get("status") == "ok"])
            total_tables = len(health.get("tables", {}))
            status_color = "#10b981" if table_count == total_tables and total_tables > 0 else "#f59e0b"
            st.markdown(f"""
            <div class="status-card {'status-ok' if table_count == total_tables and total_tables > 0 else ''}" style="padding: 1.25rem;">
                <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Tables
                </div>
                <div style="color: {status_color}; font-size: 1.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                    {table_count}/{total_tables}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            scheduler_info = health.get("scheduler", {})
            if scheduler_info.get("is_running"):
                status_text = "RUNNING"
                status_color = "#10b981"
                subtext = f"PID {scheduler_info.get('pid', 'N/A')}"
            elif scheduler_info.get("is_installed"):
                status_text = "STOPPED"
                status_color = "#6b7280"
                subtext = "Ready to start"
            else:
                status_text = "N/A"
                status_color = "#6b7280"
                subtext = "Not installed"

            st.markdown(f"""
            <div class="status-card {'status-ok' if scheduler_info.get('is_running') else ''}" style="padding: 1.25rem;">
                <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Scheduler
                </div>
                <div style="color: {status_color}; font-size: 1.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                    {status_text}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem;">
                    {subtext}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Table Details with styled expander
        with st.expander("📊 Table Details", expanded=False):
            tables = health.get("tables", {})
            if tables:
                for name, info in tables.items():
                    status_color = "#10b981" if info.get("status") == "ok" else "#ef4444"
                    count = info.get("count", 0)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;
                                padding: 0.75rem; background: #1e293b; border: 1px solid #374151; border-radius: 6px; margin-bottom: 0.5rem;">
                        <div>
                            <span style="color: #f1f5f9; font-weight: 500;">{name}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="color: {status_color}; font-size: 0.75rem; padding: 0.125rem 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 4px;">
                                {info.get('status', 'unknown').upper()}
                            </span>
                            <span style="color: #9ca3af; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;">
                                {count:,} rows
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No table information available")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        st.error(f"Failed to retrieve system status: {e}")

    # Recent Activity Section
    st.markdown("""<div style="height: 1px; background: #374151; margin: 2.5rem 0;"></div>""", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 1rem;">
        Recent Activity
    </p>
    """, unsafe_allow_html=True)

    try:
        events = api.get_lineage(limit=10)

        if events:
            for idx, event in enumerate(events):
                timestamp = event.get("timestamp", "")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                event_type = event.get("event_type", "unknown")
                actor = event.get("actor", "unknown")
                details = event.get("details", {})

                # Format event display
                icon = _get_event_icon(event_type)
                summary = _format_event_summary(event_type, details)

                # Color coding by event type
                event_colors = {
                    "hypothesis_created": "#3b82f6",
                    "hypothesis_updated": "#60a5fa",
                    "experiment_run": "#8b5cf6",
                    "deployment_approved": "#10b981",
                }
                event_color = event_colors.get(event_type, "#6b7280")

                st.markdown(f"""
                <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;
                            border-left: 3px solid {event_color};">
                    <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                        <span style="font-size: 1.25rem;">{icon}</span>
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                                <span style="color: {event_color}; font-weight: 600; font-size: 0.875rem;">
                                    {event_type.replace('_', ' ').title()}
                                </span>
                                <span style="color: #6b7280;">•</span>
                                <span style="color: #9ca3af; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                                    {actor}
                                </span>
                            </div>
                            <div style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 0.5rem;">
                                {summary}
                            </div>
                            <div style="color: #6b7280; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                                {timestamp}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6b7280;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
                <div>No recent activity recorded</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Failed to retrieve recent activity: {e}")
        st.warning("Could not load recent activity")

    # Quick Stats
    st.markdown("""<div style="height: 1px; background: #374151; margin: 2.5rem 0;"></div>""", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 1rem;">
        Research Overview
    </p>
    """, unsafe_allow_html=True)

    try:
        col1, col2, col3, col4 = st.columns(4)

        hypotheses = api.list_hypotheses(limit=1000)

        with col1:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #374151; border-radius: 8px; padding: 1.25rem; text-align: center;">
                <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Total
                </div>
                <div style="color: #f1f5f9; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {len(hypotheses)}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem;">
                    Hypotheses
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            active = len([h for h in hypotheses if h.get("status") in ("draft", "testing")])
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
                        border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 1.25rem; text-align: center;">
                <div style="color: #60a5fa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Active
                </div>
                <div style="color: #60a5fa; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {active}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem;">
                    In Progress
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            validated = len([h for h in hypotheses if h.get("status") == "validated"])
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                        border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1.25rem; text-align: center;">
                <div style="color: #10b981; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Validated
                </div>
                <div style="color: #10b981; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {validated}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem;">
                    Ready for Deployment
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            deployed = len([h for h in hypotheses if h.get("status") == "deployed"])
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
                        border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 8px; padding: 1.25rem; text-align: center;">
                <div style="color: #a78bfa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    Deployed
                </div>
                <div style="color: #a78bfa; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {deployed}
                </div>
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem;">
                    In Production
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Failed to retrieve quick stats: {e}")
        st.warning("Could not load statistics")


def render_data_health() -> None:
    """Render the Data Health page with ingestion status."""
    from hrp.dashboard.pages import data_health

    data_health.render()


def render_ingestion_status() -> None:
    """Render the Data Ingestion Status page."""
    from hrp.dashboard.pages import ingestion_status

    ingestion_status.render()


def render_hypotheses() -> None:
    """Render the Hypotheses page for browsing, creating, and viewing hypotheses."""
    st.title("Hypotheses")
    st.markdown("Manage research hypotheses for systematic strategy development.")

    api = get_api()

    if api is None:
        st.error("Platform API not available. Check database connection.")
        return

    # Tabs for different hypothesis operations
    tab_browse, tab_create, tab_view = st.tabs(["Browse", "Create New", "View Details"])

    with tab_browse:
        _render_hypotheses_browse(api)

    with tab_create:
        _render_hypotheses_create(api)

    with tab_view:
        _render_hypotheses_view(api)


def _render_hypotheses_browse(api: Any) -> None:
    """Render the hypothesis browsing interface."""
    st.subheader("Browse Hypotheses")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "draft", "testing", "validated", "rejected", "deployed"],
        )

    with col2:
        limit = st.number_input("Max Results", min_value=10, max_value=500, value=50)

    # Fetch hypotheses
    try:
        status = None if status_filter == "All" else status_filter
        hypotheses = api.list_hypotheses(status=status, limit=limit)

        if hypotheses:
            st.markdown(f"**Found {len(hypotheses)} hypotheses**")

            for hyp in hypotheses:
                status_color = _get_status_color(hyp.get("status", ""))
                with st.expander(f"{hyp.get('hypothesis_id')} - {hyp.get('title')} [{hyp.get('status')}]"):
                    st.markdown(f"**Status:** :{status_color}[{hyp.get('status')}]")
                    st.markdown(f"**Thesis:** {hyp.get('thesis')}")
                    st.markdown(f"**Prediction:** {hyp.get('prediction')}")
                    st.markdown(f"**Falsification:** {hyp.get('falsification')}")
                    st.markdown(f"**Created by:** `{hyp.get('created_by')}`")
                    st.markdown(f"**Created at:** {hyp.get('created_at')}")

                    if hyp.get("outcome"):
                        st.markdown(f"**Outcome:** {hyp.get('outcome')}")

                    if hyp.get("confidence_score"):
                        st.markdown(f"**Confidence Score:** {hyp.get('confidence_score')}")
        else:
            st.info("No hypotheses found matching the criteria.")

    except Exception as e:
        logger.error(f"Failed to list hypotheses: {e}")
        st.error(f"Error loading hypotheses: {e}")


def _render_hypotheses_create(api: Any) -> None:
    """Render the hypothesis creation form."""
    st.subheader("Create New Hypothesis")

    with st.form("create_hypothesis"):
        title = st.text_input(
            "Title",
            placeholder="e.g., Momentum predicts returns",
            help="Short descriptive title for the hypothesis",
        )

        thesis = st.text_area(
            "Thesis",
            placeholder="e.g., Stocks with high 12-month returns continue outperforming",
            help="The hypothesis being tested",
        )

        prediction = st.text_area(
            "Testable Prediction",
            placeholder="e.g., Top decile momentum > SPY by 3% annually",
            help="Specific, measurable prediction to test",
        )

        falsification = st.text_area(
            "Falsification Criteria",
            placeholder="e.g., Sharpe < SPY or p-value > 0.05",
            help="Criteria that would falsify the hypothesis",
        )

        actor = st.text_input(
            "Actor",
            value="user",
            help="Who is creating this hypothesis (user or agent:<name>)",
        )

        submitted = st.form_submit_button("Create Hypothesis", type="primary")

        if submitted:
            if not all([title, thesis, prediction, falsification]):
                st.error("All fields are required.")
            else:
                try:
                    hypothesis_id = api.create_hypothesis(
                        title=title,
                        thesis=thesis,
                        prediction=prediction,
                        falsification=falsification,
                        actor=actor,
                    )
                    st.success(f"Created hypothesis: **{hypothesis_id}**")
                    logger.info(f"Dashboard created hypothesis: {hypothesis_id}")
                except Exception as e:
                    logger.error(f"Failed to create hypothesis: {e}")
                    st.error(f"Failed to create hypothesis: {e}")


def _render_hypotheses_view(api: Any) -> None:
    """Render the hypothesis detail view."""
    st.subheader("View Hypothesis Details")

    hypothesis_id = st.text_input(
        "Hypothesis ID",
        placeholder="e.g., HYP-2025-001",
        help="Enter the hypothesis ID to view details",
    )

    if hypothesis_id:
        try:
            hyp = api.get_hypothesis(hypothesis_id)

            if hyp:
                st.markdown(f"## {hyp.get('title')}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Status:** {hyp.get('status')}")
                    st.markdown(f"**Created by:** `{hyp.get('created_by')}`")
                    st.markdown(f"**Created at:** {hyp.get('created_at')}")

                with col2:
                    if hyp.get("updated_at"):
                        st.markdown(f"**Updated at:** {hyp.get('updated_at')}")
                    if hyp.get("confidence_score"):
                        st.markdown(f"**Confidence Score:** {hyp.get('confidence_score')}")

                st.divider()

                st.markdown("### Thesis")
                st.markdown(hyp.get("thesis"))

                st.markdown("### Testable Prediction")
                st.markdown(hyp.get("prediction"))

                st.markdown("### Falsification Criteria")
                st.markdown(hyp.get("falsification"))

                if hyp.get("outcome"):
                    st.markdown("### Outcome")
                    st.markdown(hyp.get("outcome"))

                # Show linked experiments
                st.divider()
                st.markdown("### Linked Experiments")

                try:
                    experiment_ids = api.get_experiments_for_hypothesis(hypothesis_id)
                    if experiment_ids:
                        for exp_id in experiment_ids:
                            st.markdown(f"- `{exp_id}`")
                    else:
                        st.info("No experiments linked to this hypothesis yet.")
                except Exception as e:
                    logger.warning(f"Could not load experiments for hypothesis: {e}")
                    st.warning("Could not load linked experiments")

                # Show lineage
                st.divider()
                st.markdown("### Activity History")

                try:
                    events = api.get_lineage(hypothesis_id=hypothesis_id, limit=20)
                    if events:
                        for event in events:
                            timestamp = event.get("timestamp", "")
                            if isinstance(timestamp, datetime):
                                timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            st.caption(f"{timestamp} - {event.get('event_type')} by {event.get('actor')}")
                    else:
                        st.info("No activity recorded for this hypothesis.")
                except Exception as e:
                    logger.warning(f"Could not load lineage for hypothesis: {e}")
                    st.warning("Could not load activity history")

            else:
                st.warning(f"Hypothesis {hypothesis_id} not found.")

        except Exception as e:
            logger.error(f"Failed to retrieve hypothesis: {e}")
            st.error(f"Error: {e}")


def render_experiments() -> None:
    """Render the Experiments page with MLflow integration."""
    st.title("Experiments")
    st.markdown("View and compare backtesting experiments tracked in MLflow.")

    api = get_api()

    if api is None:
        st.error("Platform API not available. Check database connection.")
        return

    # Tabs for different experiment operations
    tab_recent, tab_compare, tab_details = st.tabs(["Recent Experiments", "Compare", "View Details"])

    with tab_recent:
        _render_experiments_recent(api)

    with tab_compare:
        _render_experiments_compare(api)

    with tab_details:
        _render_experiments_details(api)


def _render_experiments_recent(api: Any) -> None:
    """Render recent experiments list."""
    st.subheader("Recent Experiments")

    try:
        # Get recent experiment events from lineage
        events = api.get_lineage(limit=50)
        experiment_events = [e for e in events if e.get("event_type") == "experiment_run"]

        if experiment_events:
            st.markdown(f"**Found {len(experiment_events)} recent experiments**")

            for event in experiment_events:
                details = event.get("details", {})
                experiment_id = details.get("experiment_id", event.get("experiment_id", "Unknown"))
                timestamp = event.get("timestamp", "")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                with st.expander(f"{experiment_id} - {timestamp}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Config Name:** {details.get('config_name', 'N/A')}")
                        st.markdown(f"**Symbols:** {details.get('symbols_count', 'N/A')}")
                        st.markdown(f"**Period:** {details.get('start_date', 'N/A')} to {details.get('end_date', 'N/A')}")

                    with col2:
                        sharpe = details.get("sharpe_ratio")
                        total_return = details.get("total_return")
                        if sharpe is not None:
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        if total_return is not None:
                            st.metric("Total Return", f"{total_return:.2%}")

                    if event.get("hypothesis_id"):
                        st.markdown(f"**Linked Hypothesis:** `{event.get('hypothesis_id')}`")
        else:
            st.info("No experiments found. Run a backtest to create your first experiment.")

    except Exception as e:
        logger.error(f"Failed to retrieve experiments: {e}")
        st.error(f"Error loading experiments: {e}")


def _render_experiments_compare(api: Any) -> None:
    """Render experiment comparison interface."""
    st.subheader("Compare Experiments")

    experiment_ids_input = st.text_area(
        "Experiment IDs (one per line)",
        placeholder="Enter MLflow run IDs to compare\ne.g.:\nabc123\ndef456",
        help="Enter the MLflow run IDs of experiments to compare",
    )

    if experiment_ids_input:
        experiment_ids = [eid.strip() for eid in experiment_ids_input.split("\n") if eid.strip()]

        if len(experiment_ids) >= 2:
            try:
                comparison_df = api.compare_experiments(experiment_ids)

                if not comparison_df.empty:
                    st.markdown("### Comparison Results")
                    st.dataframe(comparison_df, use_container_width=True)

                    # Highlight best values
                    st.markdown("### Key Metrics")
                    metrics_to_highlight = ["sharpe_ratio", "total_return", "max_drawdown"]

                    for metric in metrics_to_highlight:
                        if metric in comparison_df.columns:
                            if metric == "max_drawdown":
                                best_idx = comparison_df[metric].idxmax()  # Less negative is better
                            else:
                                best_idx = comparison_df[metric].idxmax()

                            best_value = comparison_df.loc[best_idx, metric]
                            st.markdown(f"**Best {metric}:** `{best_idx}` with {best_value:.4f}")
                else:
                    st.warning("No data found for the provided experiment IDs.")

            except Exception as e:
                logger.error(f"Failed to compare experiments: {e}")
                st.error(f"Error comparing experiments: {e}")
        else:
            st.info("Enter at least 2 experiment IDs to compare.")


def _render_experiments_details(api: Any) -> None:
    """Render experiment details view."""
    st.subheader("View Experiment Details")

    experiment_id = st.text_input(
        "Experiment ID",
        placeholder="Enter MLflow run ID",
        help="Enter the MLflow run ID to view details",
    )

    if experiment_id:
        try:
            exp = api.get_experiment(experiment_id)

            if exp:
                st.markdown(f"## Experiment: `{experiment_id}`")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Status:** {exp.get('status')}")
                    start_time = exp.get("start_time")
                    if start_time:
                        st.markdown(f"**Start Time:** {datetime.fromtimestamp(start_time / 1000)}")

                with col2:
                    end_time = exp.get("end_time")
                    if end_time:
                        st.markdown(f"**End Time:** {datetime.fromtimestamp(end_time / 1000)}")

                st.divider()

                # Parameters
                st.markdown("### Parameters")
                params = exp.get("params", {})
                if params:
                    param_data = [{"Parameter": k, "Value": v} for k, v in params.items()]
                    st.table(param_data)
                else:
                    st.info("No parameters recorded")

                # Metrics
                st.markdown("### Metrics")
                metrics = exp.get("metrics", {})
                if metrics:
                    # Display key metrics as metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        if "sharpe_ratio" in metrics:
                            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

                    with col2:
                        if "total_return" in metrics:
                            st.metric("Total Return", f"{metrics['total_return']:.2%}")

                    with col3:
                        if "max_drawdown" in metrics:
                            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

                    with col4:
                        if "cagr" in metrics:
                            st.metric("CAGR", f"{metrics['cagr']:.2%}")

                    # All metrics table
                    with st.expander("All Metrics"):
                        metric_data = [{"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else v} for k, v in metrics.items()]
                        st.table(metric_data)
                else:
                    st.info("No metrics recorded")

                # Tags
                st.markdown("### Tags")
                tags = exp.get("tags", {})
                if tags:
                    # Filter out mlflow system tags
                    user_tags = {k: v for k, v in tags.items() if not k.startswith("mlflow.")}
                    if user_tags:
                        tag_data = [{"Tag": k, "Value": v} for k, v in user_tags.items()]
                        st.table(tag_data)
                    else:
                        st.info("No custom tags")
                else:
                    st.info("No tags recorded")

            else:
                st.warning(f"Experiment {experiment_id} not found.")

        except Exception as e:
            logger.error(f"Failed to retrieve experiment: {e}")
            st.error(f"Error: {e}")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_event_icon(event_type: str) -> str:
    """Get an icon for an event type."""
    icons = {
        "hypothesis_created": "📝",
        "hypothesis_updated": "✏️",
        "experiment_run": "🔬",
        "deployment_approved": "🚀",
    }
    return icons.get(event_type, "📌")


def _format_event_summary(event_type: str, details: dict) -> str:
    """Format a summary of an event."""
    if event_type == "hypothesis_created":
        return details.get("title", "New hypothesis")
    elif event_type == "hypothesis_updated":
        old_status = details.get("old_status", "")
        new_status = details.get("new_status", "")
        return f"Status: {old_status} -> {new_status}"
    elif event_type == "experiment_run":
        sharpe = details.get("sharpe_ratio")
        if sharpe is not None:
            return f"Sharpe: {sharpe:.2f}"
        return "Backtest completed"
    elif event_type == "deployment_approved":
        return details.get("title", "Strategy deployed")
    else:
        return str(details) if details else "Event recorded"


def _get_status_color(status: str) -> str:
    """Get a color name for a hypothesis status."""
    colors = {
        "draft": "gray",
        "testing": "blue",
        "validated": "green",
        "rejected": "red",
        "deployed": "violet",
    }
    return colors.get(status, "gray")


def render_pipeline_progress() -> None:
    """Render the Pipeline Progress page."""
    from hrp.dashboard.pages import pipeline_progress

    pipeline_progress.render()


def render_agents_monitor() -> None:
    """Render the Agents Monitor page."""
    from hrp.dashboard.pages import agents_monitor_page

    agents_monitor_page.render()


def render_job_health() -> None:
    """Render the Job Health page."""
    from hrp.dashboard.pages import job_health

    job_health.render()


# =============================================================================
# Sidebar and Navigation
# =============================================================================


def render_sidebar() -> str:
    """Render the sidebar and return the selected page."""
    with st.sidebar:
        # Load custom CSS
        with open("hrp/dashboard/static/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        # HRP Branding
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em;">
                HRP
            </h1>
            <p style="margin: 0; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">
                Hedgefund Research Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

        # Navigation
        st.markdown("""
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
            Navigation
        </p>
        """, unsafe_allow_html=True)

        page = st.selectbox(
            "Select Page",
            options=["Home", "Data Health", "Ingestion Status", "Hypotheses", "Experiments", "Pipeline Progress", "Agents Monitor", "Job Health"],
            label_visibility="collapsed",
        )

        st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

        # Quick Links
        st.markdown("""
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
            Quick Links
        </p>
        """, unsafe_allow_html=True)

        # MLflow UI button with custom styling
        st.markdown(
            """
            <a href="http://127.0.0.1:5000" target="_blank" rel="noopener noreferrer"
               onclick="window.open('http://127.0.0.1:5000', '_blank'); return false;"
               style="
                   display: flex;
                   align-items: center;
                   justify-content: center;
                   background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                   color: white;
                   text-decoration: none;
                   padding: 0.625rem 1rem;
                   border-radius: 6px;
                   font-size: 0.875rem;
                   font-weight: 500;
                   width: 100%;
                   margin-bottom: 0.5rem;
                   transition: all 0.15s ease;
                   box-shadow: 0 0 15px rgba(59, 130, 246, 0.25);
               "
               onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 0 25px rgba(59, 130, 246, 0.4)';"
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 0 15px rgba(59, 130, 246, 0.25)';">
                <span style="margin-right: 0.5rem;">📊</span>
                MLflow UI
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style="padding: 0.625rem 1rem; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 6px; margin-bottom: 0.5rem;">
            <a href="docs/" target="_blank" style="color: #60a5fa; text-decoration: none; font-size: 0.875rem;">
                📚 Documentation
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

        # Scheduler Status
        st.markdown("""
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
            Scheduler
        </p>
        """, unsafe_allow_html=True)

        from hrp.dashboard.components import render_scheduler_status

        render_scheduler_status()

        st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

        # System Info
        st.markdown("""
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
            System
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #9ca3af; line-height: 1.6;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #6b7280;">Version</span>
                <span style="font-family: 'JetBrains Mono', monospace;">{DASHBOARD_VERSION}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
                <span style="color: #6b7280;">Updated</span>
                <span style="font-family: 'JetBrains Mono', monospace;">{datetime.now().strftime('%H:%M')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return page


# =============================================================================
# Main Application
# =============================================================================


def main() -> None:
    """Main application entry point."""
    logger.info("HRP Dashboard started")

    # Render sidebar and get selected page
    page = render_sidebar()

    # Check if database is locked before rendering pages
    api = get_api()
    if api is None:
        render_db_error()
        return

    # Route to the selected page
    if page == "Home":
        render_home()
    elif page == "Data Health":
        render_data_health()
    elif page == "Ingestion Status":
        render_ingestion_status()
    elif page == "Hypotheses":
        render_hypotheses()
    elif page == "Experiments":
        render_experiments()
    elif page == "Pipeline Progress":
        render_pipeline_progress()
    elif page == "Agents Monitor":
        render_agents_monitor()
    elif page == "Job Health":
        render_job_health()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
