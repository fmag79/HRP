"""
Feature store and registry for HRP.

Manages feature definitions, versioning, and computation.
"""

from hrp.data.features.computation import FeatureComputer, register_default_features
from hrp.data.features.registry import FeatureRegistry

__all__ = ["FeatureRegistry", "FeatureComputer", "register_default_features"]

# Auto-register default features on module import
try:
    register_default_features()
except Exception:
    # If registration fails (e.g., database not initialized), continue
    # Features can be registered manually later
    pass
