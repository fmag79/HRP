"""Performance attribution module.

Decomposes portfolio returns into explainable components:
- Factor attribution (Brinson-Fachler, regression-based)
- Feature importance tracking (permutation, SHAP)
- Decision attribution (trade-level P&L decomposition)
"""

from .attribution_config import AttributionConfig
from .factor_attribution import (
    AttributionResult,
    BrinsonAttribution,
    FactorAttribution,
)
from .feature_importance import (
    FeatureImportanceTracker,
    ImportanceResult,
    RollingImportance,
)

__all__ = [
    "AttributionConfig",
    "AttributionResult",
    "BrinsonAttribution",
    "FactorAttribution",
    "FeatureImportanceTracker",
    "ImportanceResult",
    "RollingImportance",
]
