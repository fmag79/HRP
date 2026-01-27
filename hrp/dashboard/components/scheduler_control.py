"""
Scheduler control component for the HRP dashboard.

Provides UI to detect and resolve database lock conflicts with the scheduler.
"""

import streamlit as st
from loguru import logger

from hrp.utils.scheduler import (
    get_lock_holder_pid,
    get_scheduler_status,
    is_duckdb_lock_error,
    start_scheduler,
    stop_scheduler,
)


def render_scheduler_conflict(error: Exception) -> bool:
    """
    Render a scheduler conflict resolution UI.

    Displays a warning when the scheduler is holding the database lock and
    provides buttons to stop/start the scheduler.

    Args:
        error: The exception that triggered this UI

    Returns:
        True if the conflict was resolved (scheduler stopped), False otherwise
    """
    if not is_duckdb_lock_error(error):
        return False

    st.error("Database Lock Conflict Detected")
    st.markdown("""
    The dashboard cannot access the database because the **HRP Scheduler** is currently running.

    DuckDB uses file-level locking that prevents concurrent access from multiple processes.
    You can stop the scheduler to release the lock.
    """)

    # Get scheduler status
    scheduler_status = get_scheduler_status()

    # Display scheduler information
    col1, col2 = st.columns(2)

    with col1:
        if scheduler_status.is_running:
            st.success(f":runner: Scheduler **Running** (PID: {scheduler_status.pid})")
            if scheduler_status.command:
                st.caption(f"Command: `{scheduler_status.command}`")
        else:
            st.info(":heavy_check_mark: Scheduler **Stopped**")

    with col2:
        if scheduler_status.is_installed:
            st.caption(":floppy_disk: Launch agent installed")
        else:
            st.warning(":warning: Launch agent not found")

    # Display lock holder info if available
    lock_pid = get_lock_holder_pid(error)
    if lock_pid and lock_pid != scheduler_status.pid:
        st.warning(f":closed_lock_with_key: Lock held by PID {lock_pid} (may not be scheduler)")

    # Action buttons
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if scheduler_status.is_running:
            if st.button(":octagonal_sign: Stop Scheduler", type="primary"):
                with st.spinner("Stopping scheduler..."):
                    result = stop_scheduler()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

    with col2:
        if not scheduler_status.is_running and scheduler_status.is_installed:
            if st.button(":arrow_forward: Start Scheduler"):
                with st.spinner("Starting scheduler..."):
                    result = start_scheduler()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

    with col3:
        if st.button(":arrows_counterclockwise: Refresh Status"):
            st.rerun()

    # Information section
    st.divider()
    with st.expander(":information_source: Why does this happen?"):
        st.markdown("""
        **DuckDB File-Level Locking**

        DuckDB uses file-level locking to ensure data integrity. This means:
        - Only one process can write to the database at a time
        - Even read-only connections require exclusive file access
        - The scheduler maintains a persistent connection for scheduled jobs

        **Solutions:**

        1. **Stop the scheduler** - Use the button above to stop the scheduler temporarily.
           The dashboard will work normally. Remember to restart the scheduler when done.

        2. **Use read-only mode** - The dashboard can open the database in read-only mode,
           but this still requires the scheduler to release the lock first.

        3. **Schedule your work** - The scheduler runs jobs at specific times (e.g., 6 PM ET).
           Plan your dashboard use outside of these windows.

        **Scheduler Jobs:**
        - **Price Ingestion**: 6:00 PM ET (daily)
        - **Universe Update**: 6:05 PM ET (daily)
        - **Feature Computation**: 6:10 PM ET (daily)
        - **Signal Scan**: 7:00 PM ET (Monday)
        - **Quality Sentinel**: 6:00 AM ET (daily)
        """)

    return False  # Conflict not resolved yet


def render_scheduler_status() -> None:
    """
    Render a compact scheduler status indicator in the sidebar.

    Shows scheduler status with appropriate icons and control buttons.
    """
    scheduler_status = get_scheduler_status()

    if scheduler_status.is_running:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    border-radius: 6px;
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                <span style="color: #10b981; margin-right: 0.5rem;">●</span>
                <span style="color: #f1f5f9; font-weight: 500;">Running</span>
            </div>
            <div style="font-size: 0.75rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
                PID: {scheduler_status.pid}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⏹ Stop", key="stop_scheduler_sidebar", use_container_width=True, type="secondary"):
            result = stop_scheduler()
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])
    elif scheduler_status.is_installed:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(107, 114, 128, 0.1) 0%, rgba(107, 114, 128, 0.05) 100%);
                    border: 1px solid rgba(107, 114, 128, 0.2);
                    border-radius: 6px;
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center;">
                <span style="color: #6b7280; margin-right: 0.5rem;">○</span>
                <span style="color: #9ca3af; font-weight: 500;">Stopped</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Start", key="start_scheduler_sidebar", use_container_width=True):
            result = start_scheduler()
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])
    else:
        st.caption(":heavy_minus_sign: Scheduler not installed")
