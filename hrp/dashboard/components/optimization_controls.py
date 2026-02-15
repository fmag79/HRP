"""
Optimization Control UI Components.

Provides Streamlit components for configuring optimization runs, including
strategy selection, model configuration, sampler selection, and result visualization.

Usage:
    from hrp.dashboard.components.optimization_controls import (
        render_strategy_selector,
        render_model_selector,
        render_optimization_preview,
        render_results_tab,
    )

    # In Streamlit app
    strategy = render_strategy_selector(api)
    model = render_model_selector()
    preview = api.optimization.preview_configuration(config)
    render_optimization_preview(preview)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.api.optimization_api import OptimizationPreview
from hrp.ml.optimization import OptimizationResult
from hrp.ml.models import SUPPORTED_MODELS


def render_strategy_selector(api: PlatformAPI) -> str:
    """
    Render strategy dropdown and return selected strategy.

    Args:
        api: PlatformAPI instance for fetching strategies

    Returns:
        Selected strategy name
    """
    from hrp.api.optimization_api import OptimizationAPI

    opt_api = OptimizationAPI(api)

    try:
        strategies = opt_api.get_available_strategies()
    except Exception as e:
        logger.error(f"Failed to fetch strategies: {e}")
        strategies = []

    if not strategies:
        st.warning("No strategies with ML models available.")
        return ""

    selected = st.selectbox(
        "Strategy",
        options=strategies,
        index=0,
        help="Select a strategy to optimize. Only strategies with ML models are shown.",
        key="opt_strategy",
    )

    return selected


def render_model_selector() -> str:
    """
    Render model type dropdown and return selected model.

    Returns:
        Selected model type (e.g., "ridge", "lasso")
    """
    model_options = list(SUPPORTED_MODELS.keys())
    model_labels = [f"{m} ({SUPPORTED_MODELS[m]['family']})" for m in model_options]

    selected_idx = st.selectbox(
        "Model Type",
        options=range(len(model_options)),
        format_func=lambda i: model_labels[i],
        index=0,
        help="Select the ML model to optimize hyperparameters for.",
        key="opt_model",
    )

    return model_options[selected_idx]


def render_sampler_selector() -> str:
    """
    Render Optuna sampler selector and return selected sampler.

    Returns:
        Selected sampler name (e.g., "tpe", "random")
    """
    samplers = {
        "tpe": "TPE (Bayesian) - Recommended for most cases",
        "random": "Random Search - Good baseline",
        "grid": "Grid Search - Exhaustive but slow",
        "cmaes": "CMA-ES - Good for continuous parameters",
    }

    selected = st.selectbox(
        "Sampler",
        options=list(samplers.keys()),
        format_func=lambda k: samplers[k],
        index=0,
        help="TPE (Tree-structured Parzen Estimator) is recommended for most hyperparameter optimization tasks.",
        key="opt_sampler",
    )

    return selected


def render_trials_slider(default: int = 50, min_val: int = 10, max_val: int = 200) -> int:
    """
    Render trials slider with cost estimate.

    Args:
        default: Default number of trials
        min_val: Minimum number of trials
        max_val: Maximum number of trials

    Returns:
        Selected number of trials
    """
    n_trials = st.slider(
        "Number of Trials",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=10,
        help="More trials explore the parameter space better but take longer. 50-100 is usually sufficient.",
        key="opt_trials",
    )

    # Cost estimate
    if n_trials <= 30:
        cost_label = "🟢 Low cost"
    elif n_trials <= 100:
        cost_label = "🟡 Medium cost"
    else:
        cost_label = "🔴 High cost"

    st.caption(f"{cost_label} (~{n_trials // 10}-{n_trials // 5} minutes)")

    return n_trials


def render_folds_slider(default: int = 5, min_val: int = 3, max_val: int = 10) -> int:
    """
    Render CV folds slider.

    Args:
        default: Default number of folds
        min_val: Minimum number of folds
        max_val: Maximum number of folds

    Returns:
        Selected number of folds
    """
    n_folds = st.slider(
        "Cross-Validation Folds",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=1,
        help="Number of cross-validation folds. More folds = more robust but slower. 5 is standard.",
        key="opt_folds",
    )

    return n_folds


def render_scoring_selector() -> str:
    """
    Render scoring metric selector.

    Returns:
        Selected scoring metric (e.g., "ic", "r2")
    """
    metrics = {
        "ic": "Information Coefficient (IC) - Correlation-based",
        "rank_ic": "Rank IC - Spearman correlation",
        "r2": "R² Score - Explained variance",
        "mse": "Mean Squared Error (MSE) - Lower is better",
        "mae": "Mean Absolute Error (MAE) - Lower is better",
        "sharpe": "Sharpe Ratio - Risk-adjusted returns",
    }

    selected = st.selectbox(
        "Scoring Metric",
        options=list(metrics.keys()),
        format_func=lambda k: metrics[k],
        index=0,
        help="IC (Information Coefficient) is recommended for alpha signals. Sharpe for strategy returns.",
        key="opt_scoring",
    )

    return selected


def render_date_range() -> tuple[date, date]:
    """
    Render start/end date pickers.

    Returns:
        Tuple of (start_date, end_date)
    """
    st.markdown("**Date Range**")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2020, 1, 1),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
            help="Start date for optimization data",
            key="opt_start_date",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
            help="End date for optimization data",
            key="opt_end_date",
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")

    return start_date, end_date


def render_feature_selector(
    api: PlatformAPI,
    default_features: list[str] | None = None
) -> list[str]:
    """
    Render multi-select for features with grouping by category.

    Args:
        api: PlatformAPI instance for fetching features
        default_features: Default selected features

    Returns:
        List of selected feature names
    """
    try:
        # Fetch available features from database
        result = api.fetchall_readonly(
            """
            SELECT DISTINCT feature_name
            FROM features
            ORDER BY feature_name
            """
        )
        available_features = [r[0] for r in result]
    except Exception as e:
        logger.error(f"Failed to fetch features: {e}")
        available_features = []

    if not available_features:
        st.warning("No features available. Please compute features first.")
        return []

    # Group features by category (heuristic based on naming)
    categories = {
        "momentum": [],
        "volatility": [],
        "quality": [],
        "value": [],
        "sentiment": [],
        "risk": [],
        "other": [],
    }

    for feature in available_features:
        feature_lower = feature.lower()
        categorized = False
        for category in categories:
            if category in feature_lower:
                categories[category].append(feature)
                categorized = True
                break
        if not categorized:
            categories["other"].append(feature)

    # Default selection
    if default_features is None:
        default_features = [
            f for f in available_features
            if any(keyword in f.lower() for keyword in ["momentum", "volatility", "rsi"])
        ][:5]  # Default to 5 features

    st.markdown("**Feature Selection**")
    st.caption(f"Select features to use in optimization. {len(available_features)} available.")

    # Category-based selection
    selected_features = []

    for category, features in categories.items():
        if not features:
            continue

        with st.expander(f"📊 {category.title()} ({len(features)})", expanded=(category in ["momentum", "volatility"])):
            for feature in features:
                is_default = feature in default_features
                if st.checkbox(
                    feature,
                    value=is_default,
                    key=f"opt_feature_{feature}",
                ):
                    selected_features.append(feature)

    if not selected_features:
        st.warning("⚠️ No features selected. Please select at least one feature.")

    return selected_features


def render_optimization_preview(preview: OptimizationPreview) -> None:
    """
    Render preview card with estimated time and warnings.

    Args:
        preview: OptimizationPreview with cost estimates
    """
    st.markdown("### Configuration Preview")

    # Estimated time
    time_minutes = preview.estimated_time_seconds / 60
    if time_minutes < 1:
        time_str = f"~{int(preview.estimated_time_seconds)}s"
    elif time_minutes < 60:
        time_str = f"~{int(time_minutes)}m"
    else:
        time_str = f"~{int(time_minutes / 60)}h {int(time_minutes % 60)}m"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Estimated Time",
            value=time_str,
            help="Estimated execution time based on configuration",
        )

    with col2:
        st.metric(
            label="Cost Estimate",
            value=preview.estimated_cost_estimate,
            help="Rough cost classification",
        )

    with col3:
        st.metric(
            label="Recommended Sampler",
            value=preview.recommended_sampler.upper(),
            help="Suggested sampler based on parameter space",
        )

    # Parameter space summary
    if preview.parameter_space_summary:
        with st.expander("📋 Parameter Space Summary"):
            for param, range_str in preview.parameter_space_summary.items():
                st.text(f"{param}: {range_str}")

    # Warnings
    if preview.warnings:
        for warning in preview.warnings:
            st.warning(f"⚠️ {warning}")


def render_results_tab(result: OptimizationResult) -> None:
    """
    Render Results tab with best params, progress chart, and parameter importance.

    Args:
        result: OptimizationResult from optimization run
    """
    st.markdown("### Optimization Results")

    # Best parameters
    st.markdown("#### Best Parameters")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(
            label="Best Score",
            value=f"{result.best_score:.4f}",
            help="Best cross-validated score across all trials",
        )

    with col2:
        st.metric(
            label="Best Trial",
            value=f"#{result.best_trial}",
            help="Trial number that achieved best score",
        )

    # Best hyperparameters
    st.markdown("**Best Hyperparameters:**")
    for param, value in result.best_params.items():
        st.text(f"• {param}: {value}")

    st.divider()

    # Optimization progress chart
    st.markdown("#### Optimization Progress")

    if result.trial_history:
        import pandas as pd

        # Create dataframe from trial history
        df = pd.DataFrame(result.trial_history)

        # Progress chart
        fig = go.Figure()

        # All trials
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['score'],
            mode='markers',
            name='Trials',
            marker=dict(size=8, color='lightblue'),
            hovertemplate='Trial %{x}<br>Score: %{y:.4f}<extra></extra>',
        ))

        # Best score line (cumulative best)
        cumulative_best = df['score'].cummax()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=cumulative_best,
            mode='lines',
            name='Best Score',
            line=dict(color='darkgreen', width=2),
            hovertemplate='Trial %{x}<br>Best: %{y:.4f}<extra></extra>',
        ))

        fig.update_layout(
            title="Trial Progress",
            xaxis_title="Trial Number",
            yaxis_title="Score",
            hovermode='closest',
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Parameter importance
    st.markdown("#### Parameter Importance")

    if result.parameter_importance:
        import pandas as pd

        importance_df = pd.DataFrame([
            {"parameter": param, "importance": importance}
            for param, importance in result.parameter_importance.items()
        ]).sort_values("importance", ascending=False)

        fig = px.bar(
            importance_df,
            x='importance',
            y='parameter',
            orientation='h',
            title="Parameter Importance (from Optuna)",
            labels={'importance': 'Importance', 'parameter': 'Parameter'},
            color='importance',
            color_continuous_scale='Viridis',
        )

        fig.update_layout(height=300)

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Parameter importance shows which hyperparameters had the most impact on optimization score. "
            "Higher values indicate more influential parameters."
        )
    else:
        st.info("Parameter importance not available. Run optimization with ≥10 trials to compute importance.")


def render_fold_analysis_tab(result: OptimizationResult) -> None:
    """
    Render Fold Analysis tab with stability metrics.

    Args:
        result: OptimizationResult with fold-wise results
    """
    st.markdown("### Cross-Validation Fold Analysis")

    if not result.fold_results:
        st.warning("No fold results available.")
        return

    import pandas as pd
    import numpy as np

    # Create fold metrics dataframe
    fold_data = []
    for fold_idx, fold_result in enumerate(result.fold_results):
        fold_data.append({
            "fold": f"Fold {fold_idx + 1}",
            "train_score": fold_result.get("train_score", 0),
            "test_score": fold_result.get("test_score", 0),
        })

    df = pd.DataFrame(fold_data)

    # Stability metrics
    test_scores = df['test_score'].values
    train_scores = df['train_score'].values
    stability_score = np.std(test_scores)
    mean_train_test_gap = np.mean(train_scores - test_scores)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Stability Score",
            value=f"{stability_score:.4f}",
            help="Standard deviation of test scores. Lower = more stable.",
            delta=f"{'Low' if stability_score < 0.05 else 'High'} variance",
            delta_color="normal" if stability_score < 0.05 else "inverse",
        )

    with col2:
        st.metric(
            label="Mean Train/Test Gap",
            value=f"{mean_train_test_gap:.4f}",
            help="Average difference between train and test scores. Large gap indicates overfitting.",
            delta=f"{'Low' if mean_train_test_gap < 0.05 else 'High'} overfitting risk",
            delta_color="normal" if mean_train_test_gap < 0.05 else "inverse",
        )

    with col3:
        overfitting_risk = "Low" if mean_train_test_gap < 0.05 else ("Medium" if mean_train_test_gap < 0.1 else "High")
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.metric(
            label="Overfitting Risk",
            value=f"{risk_color[overfitting_risk]} {overfitting_risk}",
            help="Based on train/test gap and stability",
        )

    st.divider()

    # Fold-wise bar chart
    st.markdown("#### Fold-wise Performance")

    df_melted = df.melt(id_vars=['fold'], var_name='split', value_name='score')

    fig = px.bar(
        df_melted,
        x='fold',
        y='score',
        color='split',
        barmode='group',
        title="Train vs Test Score per Fold",
        labels={'score': 'Score', 'fold': 'Fold'},
        color_discrete_map={'train_score': 'lightblue', 'test_score': 'darkgreen'},
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Consistent train/test scores across folds indicate a robust model. "
        "Large train/test gaps suggest overfitting."
    )


def render_study_history_tab(studies: list[dict]) -> None:
    """
    Render Study History tab with comparison options.

    Args:
        studies: List of study metadata dicts from OptimizationAPI.list_studies()
    """
    st.markdown("### Optimization Study History")

    if not studies:
        st.info("No previous optimization runs found. Run your first optimization to see history here.")
        return

    import pandas as pd

    # Convert to dataframe
    df = pd.DataFrame(studies)

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "study_name": "Study ID",
            "datetime_start": st.column_config.DatetimeColumn("Start Time", format="YYYY-MM-DD HH:mm"),
            "n_trials": "Trials",
            "best_value": st.column_config.NumberColumn("Best Score", format="%.4f"),
            "user_attrs": "Attributes",
        },
        hide_index=True,
    )

    st.divider()

    # Study comparison
    st.markdown("#### Compare Studies")

    if len(studies) >= 2:
        selected_studies = st.multiselect(
            "Select studies to compare",
            options=[s['study_name'] for s in studies],
            default=[studies[0]['study_name'], studies[1]['study_name']] if len(studies) >= 2 else [],
            help="Select 2-4 studies to compare side by side",
        )

        if len(selected_studies) >= 2:
            comparison_data = []
            for study_name in selected_studies:
                study = next(s for s in studies if s['study_name'] == study_name)
                comparison_data.append({
                    "Study": study_name[:20],  # Truncate for display
                    "Best Score": study.get('best_value', 0),
                    "Trials": study.get('n_trials', 0),
                    "Date": study.get('datetime_start', '').split('T')[0] if study.get('datetime_start') else 'N/A',
                })

            comparison_df = pd.DataFrame(comparison_data)

            fig = px.bar(
                comparison_df,
                x='Study',
                y='Best Score',
                title="Study Comparison - Best Scores",
                labels={'Best Score': 'Score'},
                color='Best Score',
                color_continuous_scale='Viridis',
            )

            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Export functionality
    st.markdown("#### Export")

    if st.button("📥 Export Study History to CSV", key="export_studies"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="optimization_study_history.csv",
            mime="text/csv",
        )
