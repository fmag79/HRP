"""
Strategy Configuration UI Components.

Provides Streamlit components for configuring different trading strategies.

Usage:
    from hrp.dashboard.components.strategy_config import (
        render_multifactor_config,
        render_ml_predicted_config,
    )

    # In Streamlit app
    if strategy_type == "multifactor":
        config = render_multifactor_config()
    elif strategy_type == "ml_predicted":
        config = render_ml_predicted_config()
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from loguru import logger

from hrp.ml.models import SUPPORTED_MODELS


def get_available_features() -> list[str]:
    """
    Get list of available features from the database.

    Returns:
        List of feature names
    """
    try:
        from hrp.data.db import get_db

        db = get_db()
        result = db.fetchall(
            "SELECT DISTINCT feature_name FROM features ORDER BY feature_name"
        )
        return [r[0] for r in result]
    except Exception as e:
        logger.warning(f"Could not fetch features: {e}")
        # Return default features
        return ["momentum_20d", "volatility_60d"]


def render_multifactor_config() -> dict[str, Any]:
    """
    Render multi-factor strategy configuration UI.

    Provides:
    - Factor selection checkboxes
    - Weight sliders for each selected factor
    - Top N selection

    Returns:
        Dictionary with configuration:
        - feature_weights: dict[str, float]
        - top_n: int
    """
    st.markdown("#### Multi-Factor Configuration")

    # Get available features
    available_features = get_available_features()

    if not available_features:
        st.warning("No features available. Please compute features first.")
        return {"feature_weights": {}, "top_n": 10}

    # Feature selection and weights
    st.markdown("**Select Factors and Weights**")
    st.caption(
        "Positive weights favor higher values. "
        "Negative weights favor lower values (e.g., volatility)."
    )

    feature_weights = {}

    # Create a grid for feature selection
    cols_per_row = 2
    feature_chunks = [
        available_features[i:i + cols_per_row]
        for i in range(0, len(available_features), cols_per_row)
    ]

    for chunk in feature_chunks:
        cols = st.columns(cols_per_row)
        for col, feature in zip(cols, chunk):
            with col:
                # Checkbox to include feature
                include = st.checkbox(
                    feature.replace("_", " ").title(),
                    value=feature in ["momentum_20d"],  # Default momentum selected
                    key=f"mf_include_{feature}",
                )

                if include:
                    # Weight slider (-1 to 1)
                    default_weight = -0.5 if "volatility" in feature else 1.0
                    weight = st.slider(
                        f"Weight",
                        min_value=-1.0,
                        max_value=1.0,
                        value=default_weight,
                        step=0.1,
                        key=f"mf_weight_{feature}",
                    )
                    feature_weights[feature] = weight

    # Top N selection
    st.markdown("**Position Settings**")
    top_n = st.number_input(
        "Number of Holdings (Top N)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Number of top-ranked stocks to hold",
        key="mf_top_n",
    )

    # Show summary
    if feature_weights:
        st.markdown("**Configuration Summary**")
        summary_text = ", ".join(
            [f"{f}: {w:+.1f}" for f, w in feature_weights.items()]
        )
        st.text(f"Factors: {summary_text}")
        st.text(f"Holdings: Top {top_n}")
    else:
        st.warning("Please select at least one factor.")

    return {
        "feature_weights": feature_weights,
        "top_n": int(top_n),
    }


def render_ml_predicted_config() -> dict[str, Any]:
    """
    Render ML-predicted strategy configuration UI.

    Provides:
    - Model type dropdown
    - Feature multi-select
    - Signal method selection
    - Additional parameters

    Returns:
        Dictionary with configuration:
        - model_type: str
        - features: list[str]
        - signal_method: str
        - top_pct: float (for rank method)
        - threshold: float (for threshold method)
        - train_lookback: int
        - retrain_frequency: int
    """
    st.markdown("#### ML-Predicted Strategy Configuration")

    # Model selection
    available_models = list(SUPPORTED_MODELS.keys())
    model_type = st.selectbox(
        "Model Type",
        options=available_models,
        index=available_models.index("ridge") if "ridge" in available_models else 0,
        help="Machine learning model to use for predictions",
        key="mlp_model_type",
    )

    # Feature selection
    available_features = get_available_features()
    default_features = [f for f in ["momentum_20d", "volatility_60d"] if f in available_features]

    features = st.multiselect(
        "Features",
        options=available_features,
        default=default_features or available_features[:2],
        help="Features to use for model training",
        key="mlp_features",
    )

    if not features:
        st.warning("Please select at least one feature.")

    # Signal method
    st.markdown("**Signal Generation**")
    col1, col2 = st.columns(2)

    with col1:
        signal_method = st.selectbox(
            "Signal Method",
            options=["rank", "threshold", "zscore"],
            index=0,
            help=(
                "rank: Select top X% by prediction\n"
                "threshold: Select if prediction >= threshold\n"
                "zscore: Continuous signals (z-score normalized)"
            ),
            key="mlp_signal_method",
        )

    with col2:
        if signal_method == "rank":
            top_pct = st.slider(
                "Top Percentile",
                min_value=0.05,
                max_value=0.50,
                value=0.10,
                step=0.05,
                help="Fraction of stocks to select (e.g., 0.10 = top 10%)",
                key="mlp_top_pct",
            )
            threshold = 0.0
        elif signal_method == "threshold":
            threshold = st.number_input(
                "Prediction Threshold",
                min_value=-0.10,
                max_value=0.20,
                value=0.02,
                step=0.01,
                format="%.2f",
                help="Minimum predicted return to go long",
                key="mlp_threshold",
            )
            top_pct = 0.10
        else:  # zscore
            top_pct = 0.10
            threshold = 0.0
            st.info("Z-score signals are continuous (-inf to +inf)")

    # Training parameters
    st.markdown("**Training Parameters**")
    col1, col2 = st.columns(2)

    with col1:
        train_lookback = st.number_input(
            "Training Lookback (days)",
            min_value=60,
            max_value=756,
            value=252,
            step=21,
            help="Days of historical data for model training",
            key="mlp_train_lookback",
        )

    with col2:
        retrain_frequency = st.number_input(
            "Retrain Frequency (days)",
            min_value=1,
            max_value=63,
            value=21,
            step=1,
            help="Days between model retraining (21 = monthly)",
            key="mlp_retrain_frequency",
        )

    # Show summary
    st.markdown("**Configuration Summary**")
    st.text(f"Model: {model_type}")
    st.text(f"Features: {', '.join(features) if features else 'None'}")
    st.text(f"Signal: {signal_method}" + (f" (top {top_pct*100:.0f}%)" if signal_method == "rank" else ""))
    st.text(f"Training: {train_lookback}d lookback, retrain every {retrain_frequency}d")

    return {
        "model_type": model_type,
        "features": features,
        "signal_method": signal_method,
        "top_pct": top_pct,
        "threshold": threshold,
        "train_lookback": int(train_lookback),
        "retrain_frequency": int(retrain_frequency),
    }
