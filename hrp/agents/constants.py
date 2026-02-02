"""
Shared constants for research agents.

Adaptive IC thresholds by strategy class, used across multiple agents.
"""

# Adaptive IC thresholds by strategy class
IC_THRESHOLDS = {
    "cross_sectional_factor": {
        "pass": 0.015,
        "kill": 0.005,
        "description": "Value, quality, low-vol factors"
    },
    "time_series_momentum": {
        "pass": 0.02,
        "kill": 0.01,
        "description": "Trend-following strategies"
    },
    "ml_composite": {
        "pass": 0.025,
        "kill": 0.01,
        "description": "Multi-feature ML models"
    },
    "default": {
        "pass": 0.03,
        "kill": 0.01,
        "description": "Legacy uniform threshold"
    }
}


def get_ic_thresholds(strategy_class: str) -> dict:
    """
    Get IC thresholds for a strategy class.

    Args:
        strategy_class: One of: cross_sectional_factor, time_series_momentum,
                       ml_composite, default

    Returns:
        dict with 'pass' and 'kill' thresholds

    Examples:
        >>> thresholds = get_ic_thresholds("cross_sectional_factor")
        >>> assert thresholds["pass"] == 0.015
    """
    return IC_THRESHOLDS.get(strategy_class, IC_THRESHOLDS["default"])
