"""Attribution features for performance analysis.

Exposes performance attribution metrics as computed features in the feature registry.
This allows attribution results to be used as inputs to ML models or tracked over time.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.attribution.factor_attribution import BrinsonAttribution, FactorAttribution
from hrp.data.attribution.feature_importance import FeatureImportanceTracker
from hrp.data.attribution.decision_attribution import DecisionAttributor
from hrp.data.attribution.attribution_config import AttributionConfig
from hrp.data.features.registry import FeatureRegistry, FeatureDefinition


def register_attribution_features(registry: FeatureRegistry) -> None:
    """Register attribution features with the feature registry.

    Args:
        registry: Feature registry instance to register with
    """
    # Factor attribution features
    registry.register(
        FeatureDefinition(
            name="attribution_allocation_effect",
            category="attribution",
            description="Brinson allocation effect - return from sector weighting decisions",
            dependencies=[],
            computation_fn=_compute_allocation_effect,
        )
    )

    registry.register(
        FeatureDefinition(
            name="attribution_selection_effect",
            category="attribution",
            description="Brinson selection effect - return from asset selection within sectors",
            dependencies=[],
            computation_fn=_compute_selection_effect,
        )
    )

    registry.register(
        FeatureDefinition(
            name="attribution_interaction_effect",
            category="attribution",
            description="Brinson interaction effect - interaction between allocation and selection",
            dependencies=[],
            computation_fn=_compute_interaction_effect,
        )
    )

    registry.register(
        FeatureDefinition(
            name="attribution_active_return",
            category="attribution",
            description="Total active return vs benchmark",
            dependencies=[],
            computation_fn=_compute_active_return,
        )
    )

    # Feature importance metrics
    registry.register(
        FeatureDefinition(
            name="feature_importance_momentum",
            category="attribution",
            description="Permutation importance of momentum features in portfolio decisions",
            dependencies=[],
            computation_fn=_compute_momentum_importance,
        )
    )

    registry.register(
        FeatureDefinition(
            name="feature_importance_volatility",
            category="attribution",
            description="Permutation importance of volatility features",
            dependencies=[],
            computation_fn=_compute_volatility_importance,
        )
    )

    # Decision attribution metrics
    registry.register(
        FeatureDefinition(
            name="decision_timing_contribution",
            category="attribution",
            description="Average P&L contribution from entry/exit timing",
            dependencies=[],
            computation_fn=_compute_timing_contribution,
        )
    )

    registry.register(
        FeatureDefinition(
            name="decision_sizing_contribution",
            category="attribution",
            description="Average P&L contribution from position sizing",
            dependencies=[],
            computation_fn=_compute_sizing_contribution,
        )
    )

    logger.info("Registered 8 attribution features")


# Computation functions

def _compute_allocation_effect(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute Brinson allocation effect for symbol.

    This is a simplified version - in production this would query actual
    portfolio holdings and compute real attribution.
    """
    try:
        # Placeholder implementation
        # Real implementation would:
        # 1. Get portfolio weights for symbol over lookback period
        # 2. Get benchmark weights for symbol's sector
        # 3. Calculate allocation effect using Brinson formula
        # For now, return None to indicate feature not yet computed
        return None

    except Exception as e:
        logger.error(f"Error computing allocation effect for {symbol}: {e}")
        return None


def _compute_selection_effect(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute Brinson selection effect for symbol."""
    try:
        # Placeholder - see _compute_allocation_effect for real implementation approach
        return None

    except Exception as e:
        logger.error(f"Error computing selection effect for {symbol}: {e}")
        return None


def _compute_interaction_effect(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute Brinson interaction effect for symbol."""
    try:
        # Placeholder
        return None

    except Exception as e:
        logger.error(f"Error computing interaction effect for {symbol}: {e}")
        return None


def _compute_active_return(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute active return (portfolio - benchmark) for symbol."""
    try:
        # Placeholder
        return None

    except Exception as e:
        logger.error(f"Error computing active return for {symbol}: {e}")
        return None


def _compute_momentum_importance(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute feature importance score for momentum features."""
    try:
        # Placeholder
        # Real implementation would use FeatureImportanceTracker
        return None

    except Exception as e:
        logger.error(f"Error computing momentum importance for {symbol}: {e}")
        return None


def _compute_volatility_importance(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute feature importance score for volatility features."""
    try:
        # Placeholder
        return None

    except Exception as e:
        logger.error(f"Error computing volatility importance for {symbol}: {e}")
        return None


def _compute_timing_contribution(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute average timing contribution from recent trades."""
    try:
        # Placeholder
        # Real implementation would use DecisionAttributor
        return None

    except Exception as e:
        logger.error(f"Error computing timing contribution for {symbol}: {e}")
        return None


def _compute_sizing_contribution(
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
    **kwargs: Any,
) -> float | None:
    """Compute average sizing contribution from recent trades."""
    try:
        # Placeholder
        return None

    except Exception as e:
        logger.error(f"Error computing sizing contribution for {symbol}: {e}")
        return None
