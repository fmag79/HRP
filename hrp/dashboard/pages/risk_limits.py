"""
Risk Limits Configuration Page.

Provides UI for configuring and previewing risk management limits.
"""

import streamlit as st
from loguru import logger

from hrp.api.risk_config import RiskConfigAPI, RiskLimits


def _update_preview() -> None:
    """Trigger preview update when any input changes."""
    st.session_state["preview_triggered"] = True


def render_risk_limits(api) -> None:
    """
    Render the Risk Limits configuration page.

    Args:
        api: PlatformAPI instance
    """
    # Page header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Risk Limits Configuration
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Configure portfolio risk management thresholds and preview impact on hypotheses
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize RiskConfigAPI
    risk_api = RiskConfigAPI(api)

    # Get current limits
    current_limits = risk_api.get_limits()

    # Sidebar controls
    with st.sidebar:
        st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
            Actions
        </p>
        """, unsafe_allow_html=True)

        # Reset button
        if st.button("Reset to Defaults", icon="↺", use_container_width=True):
            with st.spinner("Resetting to defaults..."):
                risk_api.reset_to_defaults()
                st.session_state["risk_limits"] = risk_api.get_limits()
                st.success("Reset to default limits")
                st.rerun()

        # Info about limits
        st.markdown("""
        <div style="padding: 1rem; background: #1e293b; border: 1px solid #374151; border-radius: 8px; margin-top: 1rem;">
            <div style="color: #f1f5f9; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem;">
                ℹ️ About Risk Limits
            </div>
            <div style="color: #9ca3af; font-size: 0.8rem; line-height: 1.5;">
                Risk limits control which hypotheses can proceed to deployment. Limits are enforced by the Risk Manager agent.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state for editable limits
    if "risk_limits" not in st.session_state:
        st.session_state["risk_limits"] = current_limits

    # ══════════════════════════════════════════════════════════════════════
    # Portfolio Risk Limits Section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #374151;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem;">🛡️</span>
            <div style="flex: 1;">
                <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin: 0;">
                    Portfolio Risk Limits
                </div>
                <div style="color: #9ca3af; font-size: 0.875rem; margin-top: 0.25rem;">
                    Configure maximum risk thresholds for portfolio protection
                </div>
            </div>
            <div style="color: #10b981; font-size: 0.75rem; padding: 0.25rem 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 12px; border: 1px solid #10b981;">
                🔴 Live Preview
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Two-column layout for risk limits
    col1, col2 = st.columns(2)

    with col1:
        # Max Drawdown
        max_dd = st.number_input(
            "Max Drawdown (%)",
            min_value=5.0,
            max_value=100.0,
            value=st.session_state["risk_limits"].max_drawdown * 100,
            step=1.0,
            help="Maximum allowed drawdown from peak equity",
            on_change=_update_preview,
        )

        # Drawdown Duration
        dd_duration = st.number_input(
            "Drawdown Duration (days)",
            min_value=30,
            max_value=365,
            value=st.session_state["risk_limits"].max_drawdown_duration_days,
            step=7,
            help="Maximum days allowed to recover from a drawdown",
            on_change=_update_preview,
        )

        # Position Correlation
        pos_corr = st.number_input(
            "Position Correlation",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["risk_limits"].max_position_correlation,
            step=0.05,
            format="%.2f",
            help="Maximum correlation with existing paper portfolio positions",
            on_change=_update_preview,
        )

    with col2:
        # Sector Exposure
        sector_exp = st.number_input(
            "Sector Exposure (%)",
            min_value=5.0,
            max_value=100.0,
            value=st.session_state["risk_limits"].max_sector_exposure * 100,
            step=1.0,
            help="Maximum allocation to any single sector",
            on_change=_update_preview,
        )

        # Max Single Position
        max_single = st.number_input(
            "Max Single Position (%)",
            min_value=1.0,
            max_value=100.0,
            value=st.session_state["risk_limits"].max_single_position * 100,
            step=1.0,
            help="Maximum allocation to a single position",
            on_change=_update_preview,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Portfolio Composition Section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #374151;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem;">📊</span>
            <div style="flex: 1;">
                <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin: 0;">
                    Portfolio Composition
                </div>
                <div style="color: #9ca3af; font-size: 0.875rem; margin-top: 0.25rem;">
                    Configure diversification targets and minimum position requirements
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Min Diversification
        min_div = st.number_input(
            "Min Positions",
            min_value=1,
            max_value=100,
            value=st.session_state["risk_limits"].min_diversification,
            step=1,
            help="Minimum number of positions required",
            on_change=_update_preview,
        )

    with col2:
        # Target Positions
        target_pos = st.number_input(
            "Target Positions",
            min_value=1,
            max_value=100,
            value=st.session_state["risk_limits"].target_positions,
            step=1,
            help="Target number of positions for portfolio construction",
            on_change=_update_preview,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Preview and Save Section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""<div style="height: 1px; background: #374151; margin: 2rem 0;"></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #374151;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem;">🔮</span>
            <div style="flex: 1;">
                <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin: 0;">
                    Impact Preview
                </div>
                <div style="color: #9ca3af; font-size: 0.875rem; margin-top: 0.25rem;">
                    See how these changes would affect current validated hypotheses
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create RiskLimits object from form inputs
    proposed_limits = RiskLimits(
        max_drawdown=max_dd / 100,
        max_drawdown_duration_days=int(dd_duration),
        max_position_correlation=pos_corr,
        max_sector_exposure=sector_exp / 100,
        max_single_position=max_single / 100,
        min_diversification=int(min_div),
        target_positions=int(target_pos),
    )

    # Real-time preview - automatically updates when inputs change
    preview_container = st.container()

    with preview_container:
        try:
            # Calculate impact in real-time
            impact = risk_api.preview_impact(proposed_limits)

            # Display preview results
            st.markdown("""<div style="height: 1px; background: #374151; margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)

            # Summary cards with dynamic styling
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: #1e293b;
                            border: 1px solid #374151; border-radius: 8px;">
                    <div style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 0.5rem;">
                        Total Hypotheses
                    </div>
                    <div style="color: #f1f5f9; font-size: 2.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {impact.total_hypotheses}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Dynamic styling based on pass rate
                pass_rate = impact.hypotheses_passed / impact.total_hypotheses if impact.total_hypotheses > 0 else 0
                border_color = "#10b981" if pass_rate >= 0.7 else "#f59e0b" if pass_rate >= 0.5 else "#ef4444"
                bg_color = "rgba(16, 185, 129, 0.1)" if pass_rate >= 0.7 else "rgba(245, 158, 11, 0.1)" if pass_rate >= 0.5 else "rgba(239, 68, 68, 0.1)"
                text_color = "#10b981" if pass_rate >= 0.7 else "#f59e0b" if pass_rate >= 0.5 else "#ef4444"

                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: {bg_color};
                            border: 1px solid {border_color}; border-radius: 8px;">
                    <div style="color: {text_color}; font-size: 0.875rem; margin-bottom: 0.5rem;">
                        Would Pass ✅
                    </div>
                    <div style="color: {text_color}; font-size: 2.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {impact.hypotheses_passed}
                    </div>
                    <div style="color: {text_color}; font-size: 0.75rem; margin-top: 0.25rem;">
                        {pass_rate:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: rgba(239, 68, 68, 0.1);
                            border: 1px solid #ef4444; border-radius: 8px;">
                    <div style="color: #ef4444; font-size: 0.875rem; margin-bottom: 0.5rem;">
                        Would Veto ❌
                    </div>
                    <div style="color: #ef4444; font-size: 2.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {impact.hypotheses_vetoed}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Veto details
            if impact.veto_details:
                st.markdown("### Vetoed Hypotheses Details")

                st.markdown("""
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <thead>
                        <tr style="background: #1e293b; border-bottom: 2px solid #374151;">
                            <th style="text-align: left; padding: 0.75rem; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;">Hypothesis ID</th>
                            <th style="text-align: left; padding: 0.75rem; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;">Title</th>
                            <th style="text-align: left; padding: 0.75rem; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;">Veto Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                """, unsafe_allow_html=True)

                for veto in impact.veto_details:
                    st.markdown(f"""
                    <tr style="border-bottom: 1px solid #374151;">
                        <td style="padding: 0.75rem; color: #60a5fa; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;">{veto['hypothesis_id']}</td>
                        <td style="padding: 0.75rem; color: #f1f5f9; font-size: 0.875rem;">{veto['title']}</td>
                        <td style="padding: 0.75rem; color: #ef4444; font-size: 0.875rem;">{veto['veto_reason']}</td>
                    </tr>
                    """, unsafe_allow_html=True)

                st.markdown("""
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.info("No hypotheses would be vetoed with these limits ✅")

            # Store proposed limits in session state for saving
            st.session_state["proposed_limits"] = proposed_limits

        except Exception as e:
            logger.error(f"Preview failed: {e}")
            st.error(f"Failed to preview impact: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Save Section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""<div style="height: 1px; background: #374151; margin: 2rem 0;"></div>""", unsafe_allow_html=True)

    # Check if limits have changed
    limits_changed = (
        proposed_limits.max_drawdown != current_limits.max_drawdown
        or proposed_limits.max_drawdown_duration_days != current_limits.max_drawdown_duration_days
        or proposed_limits.max_position_correlation != current_limits.max_position_correlation
        or proposed_limits.max_sector_exposure != current_limits.max_sector_exposure
        or proposed_limits.max_single_position != current_limits.max_single_position
        or proposed_limits.min_diversification != current_limits.min_diversification
        or proposed_limits.target_positions != current_limits.target_positions
    )

    # Save button
    if limits_changed:
        st.warning("⚠️ You have unsaved changes")
        if st.button("Save Changes", icon="💾", type="primary", use_container_width=True):
            with st.spinner("Saving risk limits..."):
                if risk_api.set_limits(proposed_limits):
                    st.success("✅ Risk limits saved successfully!")
                    st.session_state["risk_limits"] = proposed_limits
                    st.rerun()
                else:
                    st.error("Failed to save risk limits")
    else:
        st.info("No changes to save")
