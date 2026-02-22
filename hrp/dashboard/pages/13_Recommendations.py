"""
Recommendations dashboard page.

Consumer-facing view showing:
- This week's recommendations with approve/reject controls
- Open positions with P&L
- Recent outcomes (closed recommendations)
- Cumulative track record vs SPY
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from hrp.api.platform import PlatformAPI


def _get_rw_api() -> PlatformAPI:
    """Get a read-write API instance for approval actions."""
    if "api_rw" not in st.session_state:
        st.session_state.api_rw = PlatformAPI()
    return st.session_state.api_rw


def render():
    """Render the recommendations page."""
    st.title("Recommendations")

    if "api" not in st.session_state:
        st.session_state.api = PlatformAPI(read_only=True)
    api = st.session_state.api

    # Check if recommendations table exists
    try:
        active = api.get_recommendations(status="active")
        pending = api.get_recommendations(status="pending_approval")
    except Exception:
        st.info(
            "Advisory service not yet initialized. "
            "Run the recommendation agent to generate recommendations."
        )
        return

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)

    all_recs = api.get_recommendation_history(limit=1000)
    if not all_recs.empty:
        closed = all_recs[all_recs["status"].isin(
            ["closed_profit", "closed_loss", "closed_stopped", "expired"]
        )]
        if not closed.empty and "realized_return" in closed.columns:
            returns = closed["realized_return"].astype(float)
            profitable = returns[returns > 0]
            win_rate = len(profitable) / len(closed) if len(closed) > 0 else 0
            avg_return = float(returns.mean())

            col1.metric("Win Rate", f"{win_rate:.0%}")
            col2.metric("Avg Return", f"{avg_return:+.1%}")
            col3.metric("Total Closed", len(closed))
            col4.metric("Open Positions", len(active))
        else:
            col1.metric("Win Rate", "---")
            col2.metric("Avg Return", "---")
            col3.metric("Total Closed", 0)
            col4.metric("Open Positions", len(active))
    else:
        col1.metric("Win Rate", "---")
        col2.metric("Avg Return", "---")
        col3.metric("Total Closed", 0)
        col4.metric("Open Positions", 0)

    st.divider()

    # --- Pending Approval ---
    if not pending.empty:
        st.subheader(f"Pending Approval ({len(pending)})")

        # Batch actions
        batch_col1, batch_col2, _ = st.columns([1, 1, 4])
        with batch_col1:
            if st.button("Approve All", type="primary", key="approve_all"):
                rw_api = _get_rw_api()
                results = rw_api.approve_all_recommendations(actor="user", dry_run=True)
                approved = [r for r in results if r["action"] == "approved"]
                st.success(f"Approved {len(approved)} recommendations (dry-run)")
                st.rerun()
        with batch_col2:
            if st.button("Reject All", key="reject_all"):
                rw_api = _get_rw_api()
                for _, row in pending.iterrows():
                    rw_api.reject_recommendation(
                        row["recommendation_id"], actor="user", reason="batch_reject"
                    )
                st.warning(f"Rejected {len(pending)} recommendations")
                st.rerun()

        for _, rec in pending.iterrows():
            _render_recommendation_card(rec, show_actions=True)

        st.divider()

    # --- This Week's Recommendations ---
    st.subheader("This Week's Recommendations")

    week_ago = date.today() - timedelta(days=7)
    new_recs = api.query_readonly(
        "SELECT recommendation_id, symbol, action, confidence, "
        "signal_strength, entry_price, target_price, stop_price, "
        "position_pct, thesis_plain, risk_plain, status "
        "FROM recommendations "
        "WHERE created_at >= ? AND status = 'active' ORDER BY signal_strength DESC",
        [week_ago],
    )

    if new_recs.empty:
        st.info("No new active recommendations this week.")
    else:
        for _, rec in new_recs.iterrows():
            _render_recommendation_card(rec, show_actions=False)

    st.divider()

    # --- Open Positions ---
    st.subheader("Open Positions")

    if active.empty:
        st.info("No open positions.")
    else:
        display_cols = ["symbol", "action", "confidence", "entry_price",
                        "target_price", "stop_price", "signal_strength"]
        available_cols = [c for c in display_cols if c in active.columns]
        st.dataframe(active[available_cols], use_container_width=True, hide_index=True)

    st.divider()

    # --- Recent Outcomes ---
    st.subheader("Recent Outcomes")

    recent_closed = api.query_readonly(
        "SELECT symbol, action, entry_price, close_price, "
        "realized_return, status, closed_at "
        "FROM recommendations "
        "WHERE status IN ('closed_profit', 'closed_loss', 'closed_stopped', 'expired') "
        "ORDER BY closed_at DESC LIMIT 20"
    )

    if recent_closed.empty:
        st.info("No closed recommendations yet.")
    else:
        for _, rec in recent_closed.iterrows():
            ret = float(rec.get("realized_return", 0))
            icon = "+" if ret > 0 else ""
            status_label = {
                "closed_profit": "profit",
                "closed_loss": "loss",
                "closed_stopped": "stopped",
                "expired": "expired",
            }.get(rec.get("status", ""), "")

            color = "green" if ret > 0 else "red"
            st.markdown(
                f":{color}[**{icon}{ret:.1%}**] "
                f"{rec.get('symbol', '')} --- {status_label} "
                f"(entry ${rec.get('entry_price', 0):.2f} -> "
                f"close ${rec.get('close_price', 0):.2f})"
            )

    st.divider()

    # --- Track Record Chart ---
    st.subheader("Cumulative Track Record")

    if not all_recs.empty:
        closed_sorted = all_recs[
            all_recs["status"].isin(
                ["closed_profit", "closed_loss", "closed_stopped", "expired"]
            )
        ].copy()

        if not closed_sorted.empty and "realized_return" in closed_sorted.columns:
            closed_sorted["closed_at"] = pd.to_datetime(closed_sorted["closed_at"])
            closed_sorted = closed_sorted.sort_values("closed_at")
            closed_sorted["cumulative_return"] = (
                1 + closed_sorted["realized_return"].astype(float)
            ).cumprod() - 1

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=closed_sorted["closed_at"],
                    y=closed_sorted["cumulative_return"] * 100,
                    mode="lines+markers",
                    name="Recommendations",
                    line=dict(color="#58a6ff", width=2),
                )
            )
            fig.update_layout(
                yaxis_title="Cumulative Return (%)",
                xaxis_title="Date",
                template="plotly_dark",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for track record chart.")
    else:
        st.info("No recommendation history available.")


def _render_recommendation_card(rec: pd.Series, show_actions: bool = False) -> None:
    """Render a single recommendation card with optional approve/reject buttons."""
    confidence = rec.get("confidence", "MEDIUM")
    confidence_color = {
        "HIGH": "green", "MEDIUM": "orange", "LOW": "red"
    }.get(confidence, "gray")

    rec_id = rec.get("recommendation_id", "")

    with st.container(border=True):
        header_col, badge_col = st.columns([4, 1])
        with header_col:
            st.markdown(
                f"**{rec.get('action', 'BUY')} {rec.get('symbol', '')}**"
            )
        with badge_col:
            st.markdown(f":{confidence_color}[{confidence}]")

        st.write(rec.get("thesis_plain", ""))

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric("Entry", f"${rec.get('entry_price', 0):.2f}")
        detail_col2.metric("Target", f"${rec.get('target_price', 0):.2f}")
        detail_col3.metric("Stop", f"${rec.get('stop_price', 0):.2f}")

        with st.expander("Risk Details"):
            st.write(rec.get("risk_plain", ""))
            st.write(f"Position size: {float(rec.get('position_pct', 0)):.1%} of portfolio")

        if show_actions and rec_id:
            action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
            with action_col1:
                if st.button("Approve", type="primary", key=f"approve_{rec_id}"):
                    rw_api = _get_rw_api()
                    result = rw_api.approve_recommendation(rec_id, actor="user", dry_run=True)
                    if result["action"] == "approved":
                        st.success(f"Approved (dry-run)")
                    else:
                        st.error(result["message"])
                    st.rerun()
            with action_col2:
                if st.button("Reject", key=f"reject_{rec_id}"):
                    rw_api = _get_rw_api()
                    result = rw_api.reject_recommendation(rec_id, actor="user", reason="user_reject")
                    st.warning("Rejected")
                    st.rerun()


# Streamlit page entry point
render()
