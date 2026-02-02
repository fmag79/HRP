"""
Home page for HRP Streamlit dashboard.

Displays system status overview, recent activity, and quick stats.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any

import streamlit as st
from loguru import logger


def _format_timestamp(ts: datetime | str | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "N/A"

    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return ts

    # Make timezone-aware if naive
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta = now - ts

    if delta < timedelta(minutes=1):
        return "just now"
    elif delta < timedelta(hours=1):
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes}m ago"
    elif delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"{hours}h ago"
    elif delta < timedelta(days=7):
        days = delta.days
        return f"{days}d ago"
    else:
        return ts.strftime("%Y-%m-%d %H:%M")


def _get_event_icon(event_type: str) -> str:
    """Get an icon for an event type."""
    icons = {
        "hypothesis_created": "[+]",
        "hypothesis_updated": "[~]",
        "hypothesis_deleted": "[-]",
        "experiment_run": "[E]",
        "experiment_linked": "[L]",
        "validation_passed": "[V]",
        "validation_failed": "[X]",
        "deployment_approved": "[D]",
        "deployment_rejected": "[R]",
        "agent_run_complete": "[A]",
        "data_ingestion": "[I]",
        "system_error": "[!]",
    }
    return icons.get(event_type, "[*]")


def _get_database_stats(api: Any) -> dict[str, Any]:
    """Get database statistics using Platform API."""
    stats: dict[str, Any] = {
        "connected": False,
        "total_symbols": 0,
        "total_price_records": 0,
        "date_range_start": None,
        "date_range_end": None,
        "error": None,
    }

    try:
        health = api.health_check()

        if health.get("database") == "ok":
            stats["connected"] = True

            # Get table counts from health check
            tables = health.get("tables", {})

            # Universe count
            universe_info = tables.get("universe", {})
            if universe_info.get("status") == "ok":
                stats["total_symbols"] = universe_info.get("count", 0)

            # Price records count
            prices_info = tables.get("prices", {})
            if prices_info.get("status") == "ok":
                stats["total_price_records"] = prices_info.get("count", 0)

            # Get date range from prices table
            try:
                from hrp.api.platform import PlatformAPI
                api = PlatformAPI()
                result = api.fetchone_readonly("SELECT MIN(date), MAX(date) FROM prices")
                if result and result[0] is not None:
                    stats["date_range_start"] = result[0]
                    stats["date_range_end"] = result[1]
            except Exception as e:
                logger.warning(f"Could not get date range: {e}")
        else:
            stats["error"] = health.get("database", "Unknown error")

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Error getting database stats: {e}")

    return stats


def _get_hypothesis_stats(api: Any) -> dict[str, int]:
    """Get hypothesis counts by status."""
    stats: dict[str, int] = {
        "draft": 0,
        "testing": 0,
        "validated": 0,
        "rejected": 0,
        "deployed": 0,
        "total": 0,
    }

    try:
        all_hypotheses = api.list_hypotheses()
        stats["total"] = len(all_hypotheses)

        for h in all_hypotheses:
            status = h.get("status", "unknown")
            if status in stats:
                stats[status] += 1

    except Exception as e:
        logger.warning(f"Could not get hypothesis stats: {e}")

    return stats


def _get_recent_experiments_count(api: Any, days: int = 7) -> int:
    """Get count of experiments run in the last N days."""
    try:
        events = api.get_lineage(limit=1000)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        count = 0

        for event in events:
            if event.get("event_type") == "experiment_run":
                ts = event.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                if ts and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if ts and ts >= cutoff:
                    count += 1

        return count

    except Exception as e:
        logger.warning(f"Could not get experiment count: {e}")
        return 0


def render() -> None:
    """Render the home page."""
    st.title("HRP Dashboard")
    st.subheader("Hedgefund Research Platform")

    # Initialize API
    try:
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
    except Exception as e:
        st.error(f"Failed to initialize Platform API: {e}")
        st.info("Make sure the database is initialized. Run: python -m hrp.data.schema --init")
        return

    # =========================================================================
    # System Status Overview
    # =========================================================================
    st.header("System Overview")

    with st.spinner("Loading system status..."):
        db_stats = _get_database_stats(api)

    # Database connection status
    if db_stats["connected"]:
        st.success("Database: Connected")
    else:
        st.error(f"Database: Disconnected - {db_stats.get('error', 'Unknown error')}")
        st.info("Initialize the database with: python -m hrp.data.schema --init")
        return

    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Symbols in Universe",
            value=f"{db_stats['total_symbols']:,}",
        )

    with col2:
        st.metric(
            label="Price Records",
            value=f"{db_stats['total_price_records']:,}",
        )

    with col3:
        if db_stats["date_range_start"]:
            start_str = str(db_stats["date_range_start"])
            st.metric(
                label="Data Start",
                value=start_str[:10] if len(start_str) >= 10 else start_str,
            )
        else:
            st.metric(label="Data Start", value="No data")

    with col4:
        if db_stats["date_range_end"]:
            end_str = str(db_stats["date_range_end"])
            st.metric(
                label="Data End",
                value=end_str[:10] if len(end_str) >= 10 else end_str,
            )
        else:
            st.metric(label="Data End", value="No data")

    st.divider()

    # =========================================================================
    # Quick Stats Cards
    # =========================================================================
    st.header("Research Stats")

    with st.spinner("Loading research stats..."):
        hyp_stats = _get_hypothesis_stats(api)
        recent_experiments = _get_recent_experiments_count(api, days=7)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Hypotheses",
            value=hyp_stats["total"],
        )

    with col2:
        st.metric(
            label="Draft",
            value=hyp_stats["draft"],
        )

    with col3:
        st.metric(
            label="Testing",
            value=hyp_stats["testing"],
        )

    with col4:
        st.metric(
            label="Validated",
            value=hyp_stats["validated"],
        )

    with col5:
        st.metric(
            label="Deployed",
            value=hyp_stats["deployed"],
        )

    # Additional stats row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Rejected Hypotheses",
            value=hyp_stats["rejected"],
        )

    with col2:
        st.metric(
            label="Experiments (7 days)",
            value=recent_experiments,
        )

    with col3:
        # Calculate validation rate
        tested = hyp_stats["validated"] + hyp_stats["rejected"]
        if tested > 0:
            validation_rate = (hyp_stats["validated"] / tested) * 100
            st.metric(
                label="Validation Rate",
                value=f"{validation_rate:.1f}%",
            )
        else:
            st.metric(
                label="Validation Rate",
                value="N/A",
            )

    st.divider()

    # =========================================================================
    # Recent Activity Section
    # =========================================================================
    st.header("Recent Activity")

    try:
        from hrp.research.lineage import get_recent_events

        with st.spinner("Loading recent events..."):
            # Get events from last 24 hours, then fall back to most recent
            recent_events = get_recent_events(hours=24)

            # If no events in last 24 hours, get most recent from lineage
            if not recent_events:
                recent_events = api.get_lineage(limit=10)

        if recent_events:
            # Display last 10 events
            events_to_show = recent_events[:10]

            for event in events_to_show:
                event_type = event.get("event_type", "unknown")
                timestamp = event.get("timestamp")
                actor = event.get("actor", "unknown")
                hypothesis_id = event.get("hypothesis_id")
                experiment_id = event.get("experiment_id")
                details = event.get("details", {})

                # Build event description
                icon = _get_event_icon(event_type)
                time_str = _format_timestamp(timestamp)

                # Format the event type for display
                event_display = event_type.replace("_", " ").title()

                # Build context string
                context_parts = []
                if hypothesis_id:
                    context_parts.append(f"Hypothesis: {hypothesis_id}")
                if experiment_id:
                    context_parts.append(f"Experiment: {experiment_id[:12]}...")
                if details and isinstance(details, dict):
                    if "title" in details:
                        context_parts.append(f'"{details["title"]}"')
                    if "sharpe_ratio" in details:
                        context_parts.append(f"Sharpe: {details['sharpe_ratio']:.2f}")

                context_str = " | ".join(context_parts) if context_parts else ""

                # Display the event
                with st.container():
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        st.text(f"{icon} {time_str}")
                    with cols[1]:
                        st.markdown(f"**{event_display}** by `{actor}`")
                        if context_str:
                            st.caption(context_str)
                    with cols[2]:
                        st.text("")  # Spacer
        else:
            st.info("No recent activity recorded. Start by creating a hypothesis or running a backtest.")

    except ImportError as e:
        st.warning(f"Could not load lineage module: {e}")
        st.info("Recent activity tracking requires the lineage module.")
    except Exception as e:
        st.error(f"Error loading recent activity: {e}")
        logger.error(f"Error in recent activity section: {e}")

    # =========================================================================
    # Quick Actions Footer
    # =========================================================================
    st.divider()
    st.caption("HRP - Hedgefund Research Platform | Long-only US Equities | Daily Timeframe")


# Entry point for Streamlit page
if __name__ == "__main__":
    render()
