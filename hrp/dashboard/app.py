"""
HRP Dashboard - Main Application

Multipage Streamlit dashboard for the Hedgefund Research Platform.
Provides access to system status, data health, hypotheses, and experiments.
"""

import sys
from pathlib import Path

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
    """
    if "api" not in st.session_state:
        try:
            from hrp.api.platform import PlatformAPI

            st.session_state.api = PlatformAPI()
            logger.info("PlatformAPI initialized for dashboard")
        except Exception as e:
            logger.error(f"Failed to initialize PlatformAPI: {e}")
            st.session_state.api = None
    return st.session_state.api


# =============================================================================
# Page Rendering Functions
# =============================================================================


def render_home() -> None:
    """Render the Home page with system status and recent activity."""
    st.title("HRP Dashboard")
    st.markdown("**Hedgefund Research Platform** - Systematic Trading Strategy Development")

    api = get_api()

    # System Status Section
    st.header("System Status")

    if api is None:
        st.error("Platform API not available. Check database connection.")
        return

    try:
        health = api.health_check()

        col1, col2, col3 = st.columns(3)

        with col1:
            api_status = health.get("api", "unknown")
            if api_status == "ok":
                st.success("API: Online")
            else:
                st.error(f"API: {api_status}")

        with col2:
            db_status = health.get("database", "unknown")
            if db_status == "ok":
                st.success("Database: Connected")
            else:
                st.error(f"Database: {db_status}")

        with col3:
            table_count = len([t for t, v in health.get("tables", {}).items() if v.get("status") == "ok"])
            total_tables = len(health.get("tables", {}))
            if table_count == total_tables and total_tables > 0:
                st.success(f"Tables: {table_count}/{total_tables}")
            else:
                st.warning(f"Tables: {table_count}/{total_tables}")

        # Table Details
        with st.expander("Table Details"):
            tables = health.get("tables", {})
            if tables:
                table_data = [
                    {"Table": name, "Status": info.get("status", "unknown"), "Row Count": info.get("count", 0)}
                    for name, info in tables.items()
                ]
                st.table(table_data)
            else:
                st.info("No table information available")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        st.error(f"Failed to retrieve system status: {e}")

    # Recent Activity Section
    st.header("Recent Activity")

    try:
        events = api.get_lineage(limit=10)

        if events:
            for event in events:
                timestamp = event.get("timestamp", "")
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                event_type = event.get("event_type", "unknown")
                actor = event.get("actor", "unknown")
                details = event.get("details", {})

                # Format event display
                icon = _get_event_icon(event_type)
                summary = _format_event_summary(event_type, details)

                st.markdown(f"{icon} **{event_type}** by `{actor}` - {summary}")
                st.caption(f"{timestamp}")
                st.divider()
        else:
            st.info("No recent activity recorded")

    except Exception as e:
        logger.error(f"Failed to retrieve recent activity: {e}")
        st.warning("Could not load recent activity")

    # Quick Stats
    st.header("Quick Stats")

    try:
        col1, col2, col3, col4 = st.columns(4)

        hypotheses = api.list_hypotheses(limit=1000)
        with col1:
            st.metric("Total Hypotheses", len(hypotheses))

        with col2:
            active = len([h for h in hypotheses if h.get("status") in ("draft", "testing")])
            st.metric("Active Hypotheses", active)

        with col3:
            validated = len([h for h in hypotheses if h.get("status") == "validated"])
            st.metric("Validated", validated)

        with col4:
            deployed = len([h for h in hypotheses if h.get("status") == "deployed"])
            st.metric("Deployed", deployed)

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


# =============================================================================
# Sidebar and Navigation
# =============================================================================


def render_sidebar() -> str:
    """Render the sidebar and return the selected page."""
    with st.sidebar:
        # HRP Branding
        st.markdown("# HRP")
        st.markdown("**Hedgefund Research Platform**")
        st.divider()

        # Navigation
        st.markdown("### Navigation")
        page = st.selectbox(
            "Select Page",
            options=["Home", "Data Health", "Ingestion Status", "Hypotheses", "Experiments"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick Links
        st.markdown("### Quick Links")
        # Use styled anchor tag with JavaScript fallback to open in new tab
        # Using 127.0.0.1 instead of localhost because MLflow blocks localhost
        st.markdown(
            """
            <a href="http://127.0.0.1:5000" target="_blank" rel="noopener noreferrer" 
               onclick="window.open('http://127.0.0.1:5000', '_blank'); return false;"
               style="
                   display: inline-block;
                   background-color: #FF4B4B;
                   color: white;
                   text-decoration: none;
                   padding: 0.5rem 1rem;
                   border-radius: 0.25rem;
                   font-size: 0.875rem;
                   width: 100%;
                   text-align: center;
                   margin-bottom: 0.5rem;
                   cursor: pointer;
               ">MLflow UI</a>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("- [Documentation](docs/)")

        st.divider()

        # System Info
        st.markdown("### System Info")
        st.caption(f"Dashboard Version: 0.1.0")
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return page


# =============================================================================
# Main Application
# =============================================================================


def main() -> None:
    """Main application entry point."""
    logger.info("HRP Dashboard started")

    # Render sidebar and get selected page
    page = render_sidebar()

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
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
