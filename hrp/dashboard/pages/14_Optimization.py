"""
Optimization Dashboard Page.

Provides UI for configuring and running hyperparameter optimization
using Optuna, with cross-validation and study management.
"""

import streamlit as st
from datetime import date
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.api.optimization_api import OptimizationAPI, OptimizationConfig
from hrp.dashboard.components.optimization_controls import (
    render_strategy_selector,
    render_model_selector,
    render_sampler_selector,
    render_trials_slider,
    render_folds_slider,
    render_scoring_selector,
    render_date_range,
    render_feature_selector,
    render_optimization_preview,
    render_results_tab,
    render_fold_analysis_tab,
    render_study_history_tab,
)


def _update_preview() -> None:
    """Trigger preview update when any input changes."""
    st.session_state["preview_triggered"] = True


def render_optimization_page(api: PlatformAPI) -> None:
    """
    Render the Optimization dashboard page.

    Args:
        api: PlatformAPI instance
    """
    # Page header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Strategy Optimization
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Hyperparameter optimization with Optuna and cross-validation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize OptimizationAPI
    opt_api = OptimizationAPI(api)

    # Initialize session state
    if "optimization_result" not in st.session_state:
        st.session_state["optimization_result"] = None
    if "optimization_config" not in st.session_state:
        st.session_state["optimization_config"] = None
    if "last_study_id" not in st.session_state:
        st.session_state["last_study_id"] = None

    # ══════════════════════════════════════════════════════════════════════
    # Sidebar Configuration
    # ══════════════════════════════════════════════════════════════════════

    with st.sidebar:
        st.markdown("### Configuration")

        # Strategy and model selection
        strategy_name = render_strategy_selector(api)
        model_type = render_model_selector()

        st.divider()

        # Optimization settings
        st.markdown("**Optimization Settings**")
        sampler = render_sampler_selector()
        n_trials = render_trials_slider(default=50, min_val=10, max_val=200)
        n_folds = render_folds_slider(default=5, min_val=3, max_val=10)
        scoring = render_scoring_selector()

        st.divider()

        # Date range
        start_date, end_date = render_date_range()

        st.divider()

        # Feature selection
        feature_names = render_feature_selector(api, default_features=None)

        st.divider()

        # Actions
        st.markdown("**Actions**")

        if st.button("🔄 Reset Configuration", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("opt_"):
                    del st.session_state[key]
            st.rerun()

        # Info box
        st.markdown("""
        <div style="padding: 1rem; background: #1e293b; border: 1px solid #374151; border-radius: 8px; margin-top: 1rem;">
            <div style="color: #f1f5f9; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem;">
                ℹ️ About Optimization
            </div>
            <div style="color: #9ca3af; font-size: 0.8rem; line-height: 1.5;">
                Optuna uses TPE (Tree-structured Parzen Estimator) for efficient hyperparameter search.
                Cross-validation ensures robust parameter selection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Main Content Area
    # ══════════════════════════════════════════════════════════════════════

    # Validate configuration
    if not strategy_name:
        st.warning("⚠️ No strategies available. Please configure strategies first.")
        return

    if not feature_names:
        st.warning("⚠️ No features selected. Please select at least one feature.")
        return

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return

    # Build configuration
    try:
        param_space = opt_api.get_default_param_space(model_type)
    except ValueError as e:
        st.error(f"❌ {e}")
        return

    config = OptimizationConfig(
        hypothesis_id=f"opt_{strategy_name}_{model_type}",
        strategy_name=strategy_name,
        model_type=model_type,
        param_space=param_space,
        sampler=sampler,
        n_trials=n_trials,
        n_folds=n_folds,
        scoring=scoring,
        features=feature_names,
        start_date=start_date,
        end_date=end_date,
        pruning_enabled=True,
        study_name=None,  # Auto-generate
    )

    st.session_state["optimization_config"] = config

    # ══════════════════════════════════════════════════════════════════════
    # Configuration Preview
    # ══════════════════════════════════════════════════════════════════════

    st.markdown("### Configuration Preview")

    try:
        preview = opt_api.preview_configuration(config)
        render_optimization_preview(preview)
    except Exception as e:
        logger.error(f"Failed to preview configuration: {e}")
        st.error(f"❌ Failed to generate preview: {e}")
        return

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Run Optimization Button
    # ══════════════════════════════════════════════════════════════════════

    if st.button("▶️ Run Optimization", type="primary", use_container_width=True):
        # Fetch symbols for optimization
        try:
            # Get symbols from universe
            symbols_result = api.fetchall_readonly(
                """
                SELECT DISTINCT symbol
                FROM prices
                WHERE date >= ? AND date <= ?
                ORDER BY symbol
                LIMIT 100
                """,
                (str(start_date), str(end_date)),
            )
            symbols = [r[0] for r in symbols_result]

            if not symbols:
                st.error("❌ No symbols found for selected date range.")
                return

            st.info(f"🎯 Running optimization on {len(symbols)} symbols...")

        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            st.error(f"❌ Failed to fetch symbols: {e}")
            return

        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(current: int, total: int):
            """Update progress bar during optimization."""
            progress = current / total
            progress_bar.progress(progress)
            progress_text.text(f"Trial {current}/{total} ({int(progress * 100)}%)")

        # Run optimization
        try:
            with st.spinner("Running optimization..."):
                result = opt_api.run_optimization(
                    config=config,
                    symbols=symbols,
                    progress_callback=progress_callback,
                )

                # Store result
                st.session_state["optimization_result"] = result
                st.session_state["last_study_id"] = result.study_name

                progress_text.empty()
                progress_bar.empty()

                st.success(f"✅ Optimization complete! Best score: {result.best_score:.4f}")

                # Auto-switch to Results tab
                st.session_state["opt_tab"] = "Results"

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            st.error(f"❌ Optimization failed: {e}")
            progress_text.empty()
            progress_bar.empty()
            return

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Results Tabs
    # ══════════════════════════════════════════════════════════════════════

    # Tab selector
    tabs = st.tabs(["📊 Results", "📈 Fold Analysis", "📚 Study History"])

    # Tab 1: Results
    with tabs[0]:
        if st.session_state["optimization_result"] is not None:
            render_results_tab(st.session_state["optimization_result"])
        else:
            st.info("👆 Configure and run an optimization to see results here.")

    # Tab 2: Fold Analysis
    with tabs[1]:
        if st.session_state["optimization_result"] is not None:
            render_fold_analysis_tab(st.session_state["optimization_result"])
        else:
            st.info("👆 Configure and run an optimization to see fold analysis here.")

    # Tab 3: Study History
    with tabs[2]:
        try:
            studies = opt_api.list_studies()
            render_study_history_tab(studies)
        except Exception as e:
            logger.error(f"Failed to list studies: {e}")
            st.error(f"❌ Failed to load study history: {e}")


def main():
    """Standalone entry point for testing."""
    st.set_page_config(
        page_title="Optimization",
        page_icon="⚙️",
        layout="wide",
    )

    api = PlatformAPI(read_only=True)
    render_optimization_page(api)


if __name__ == "__main__":
    main()
