"""
Dashboard UI components.
"""

from hrp.dashboard.components.strategy_config import (
    render_multifactor_config,
    render_ml_predicted_config,
    get_available_features,
)

__all__ = [
    "render_multifactor_config",
    "render_ml_predicted_config",
    "get_available_features",
]
