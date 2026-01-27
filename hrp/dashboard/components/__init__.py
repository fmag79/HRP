"""
Dashboard UI components.
"""

from hrp.dashboard.components.strategy_config import (
    render_multifactor_config,
    render_ml_predicted_config,
    get_available_features,
)
from hrp.dashboard.components.walkforward_viz import (
    render_walkforward_splits,
    render_fold_metrics_heatmap,
    render_fold_comparison_chart,
    render_stability_summary,
)
from hrp.dashboard.components.sharpe_decay_viz import (
    render_sharpe_decay_heatmap,
    render_generalization_summary,
    render_parameter_sensitivity_chart,
    render_top_bottom_params,
)
from hrp.dashboard.components.scheduler_control import (
    render_scheduler_conflict,
    render_scheduler_status,
)

__all__ = [
    # Strategy config
    "render_multifactor_config",
    "render_ml_predicted_config",
    "get_available_features",
    # Walk-forward visualization
    "render_walkforward_splits",
    "render_fold_metrics_heatmap",
    "render_fold_comparison_chart",
    "render_stability_summary",
    # Sharpe decay visualization
    "render_sharpe_decay_heatmap",
    "render_generalization_summary",
    "render_parameter_sensitivity_chart",
    "render_top_bottom_params",
    # Scheduler control
    "render_scheduler_conflict",
    "render_scheduler_status",
]
