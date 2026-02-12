"""
Risk feature definitions for Value-at-Risk and CVaR.

Features:
    - var_95_1d: 1-day Value-at-Risk at 95% confidence (parametric method)
    - cvar_95_1d: 1-day Conditional VaR at 95% confidence
    - var_99_1d: 1-day Value-at-Risk at 99% confidence
    - mc_var_95_1d: 1-day Value-at-Risk at 95% (Monte Carlo method)
    - var_95_10d: 10-day Value-at-Risk at 95% confidence

All VaR values are expressed as positive numbers representing potential loss magnitudes.
"""

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.risk.risk_config import VaRConfig, VaRMethod, Distribution
from hrp.data.risk.var_calculator import VaRCalculator


def _compute_rolling_var(
    prices: pd.DataFrame,
    confidence_level: float,
    time_horizon: int,
    method: VaRMethod,
    distribution: Distribution = Distribution.NORMAL,
    window: int = 252,
    return_cvar: bool = False,
) -> pd.DataFrame:
    """
    Compute rolling VaR or CVaR for all symbols in the price DataFrame.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column
        confidence_level: Confidence level (e.g., 0.95, 0.99)
        time_horizon: Time horizon in days
        method: VaR calculation method
        distribution: Distribution assumption for parametric/MC methods
        window: Rolling window size for returns (default: 252 days = 1 year)
        return_cvar: If True, return CVaR instead of VaR

    Returns:
        DataFrame with rolling VaR or CVaR values, indexed by (date, symbol)
    """
    # Extract close prices and unstack to wide format
    close = prices["close"].unstack(level="symbol")

    # Calculate returns
    returns = close.pct_change()

    # Initialize calculator
    config = VaRConfig(
        confidence_level=confidence_level,
        time_horizon=time_horizon,
        method=method,
        distribution=distribution,
    )
    calculator = VaRCalculator(config)

    # Compute rolling VaR for each symbol
    var_results = {}

    for symbol in close.columns:
        symbol_returns = returns[symbol].dropna()

        if len(symbol_returns) < max(window, 60):
            # Not enough data - return NaN series
            var_results[symbol] = pd.Series(index=close.index, dtype=float)
            var_results[symbol][:] = np.nan
            continue

        # Compute rolling VaR
        rolling_values = []
        dates = []

        for i in range(len(symbol_returns)):
            # Need at least 'window' observations or 60 (whichever is larger)
            if i < max(window, 60):
                rolling_values.append(np.nan)
                dates.append(symbol_returns.index[i])
                continue

            # Get window of returns
            window_returns = symbol_returns.iloc[max(0, i - window + 1) : i + 1].values

            try:
                # Calculate VaR/CVaR
                result = calculator.calculate(window_returns, config=config)
                value = result.cvar if return_cvar else result.var
                rolling_values.append(value)
                dates.append(symbol_returns.index[i])

            except Exception as e:
                logger.warning(f"VaR calculation failed for {symbol} at index {i}: {e}")
                rolling_values.append(np.nan)
                dates.append(symbol_returns.index[i])

        var_results[symbol] = pd.Series(rolling_values, index=dates)

    # Combine into DataFrame
    var_df = pd.DataFrame(var_results)

    # Stack back to multi-index format
    result = var_df.stack(level=0, future_stack=True)
    result.index.names = ["date", "symbol"]

    return result.to_frame()


def compute_var_95_1d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-day Value-at-Risk at 95% confidence using parametric method.

    VaR represents the potential loss at the 95th percentile - there is a 5%
    chance that losses will exceed this amount over a 1-day period.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with var_95_1d values (positive loss magnitudes)
    """
    result = _compute_rolling_var(
        prices,
        confidence_level=0.95,
        time_horizon=1,
        method=VaRMethod.PARAMETRIC,
        distribution=Distribution.NORMAL,
    )

    result.columns = ["var_95_1d"]
    return result


def compute_cvar_95_1d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-day Conditional VaR (Expected Shortfall) at 95% confidence.

    CVaR represents the expected loss given that losses exceed the VaR threshold.
    It provides a measure of tail risk beyond VaR.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with cvar_95_1d values (positive loss magnitudes)
    """
    result = _compute_rolling_var(
        prices,
        confidence_level=0.95,
        time_horizon=1,
        method=VaRMethod.PARAMETRIC,
        distribution=Distribution.NORMAL,
        return_cvar=True,
    )

    result.columns = ["cvar_95_1d"]
    return result


def compute_var_99_1d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-day Value-at-Risk at 99% confidence using parametric method.

    VaR99 is more conservative than VaR95, capturing extreme tail events with
    only a 1% probability of exceedance.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with var_99_1d values (positive loss magnitudes)
    """
    result = _compute_rolling_var(
        prices,
        confidence_level=0.99,
        time_horizon=1,
        method=VaRMethod.PARAMETRIC,
        distribution=Distribution.NORMAL,
    )

    result.columns = ["var_99_1d"]
    return result


def compute_mc_var_95_1d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-day Value-at-Risk at 95% confidence using Monte Carlo simulation.

    MC VaR fits a t-distribution to historical returns and simulates 10,000 paths
    to estimate the loss distribution. This captures fat tails better than the
    parametric method.

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with mc_var_95_1d values (positive loss magnitudes)
    """
    result = _compute_rolling_var(
        prices,
        confidence_level=0.95,
        time_horizon=1,
        method=VaRMethod.MONTE_CARLO,
        distribution=Distribution.T,
    )

    result.columns = ["mc_var_95_1d"]
    return result


def compute_var_95_10d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 10-day Value-at-Risk at 95% confidence using parametric method.

    Multi-day VaR captures the risk over a longer holding period. Under normal
    assumptions, 10-day VaR ≈ 1-day VaR × sqrt(10).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with var_95_10d values (positive loss magnitudes)
    """
    result = _compute_rolling_var(
        prices,
        confidence_level=0.95,
        time_horizon=10,
        method=VaRMethod.PARAMETRIC,
        distribution=Distribution.NORMAL,
    )

    result.columns = ["var_95_10d"]
    return result


# Feature registration for the feature store
RISK_FEATURES = [
    "var_95_1d",
    "cvar_95_1d",
    "var_99_1d",
    "mc_var_95_1d",
    "var_95_10d",
]


def register_risk_features() -> None:
    """
    Register risk features in the feature registry.

    This integrates VaR/CVaR features with the HRP feature system.
    """
    from hrp.data.features.registry import FeatureRegistry

    registry = FeatureRegistry()

    features = [
        {
            "feature_name": "var_95_1d",
            "version": "v1",
            "computation_fn": compute_var_95_1d,
            "description": "1-day Value-at-Risk at 95% confidence (parametric method). Potential loss magnitude with 5% probability of exceedance.",
        },
        {
            "feature_name": "cvar_95_1d",
            "version": "v1",
            "computation_fn": compute_cvar_95_1d,
            "description": "1-day Conditional VaR at 95% confidence (Expected Shortfall). Expected loss given that losses exceed VaR threshold.",
        },
        {
            "feature_name": "var_99_1d",
            "version": "v1",
            "computation_fn": compute_var_99_1d,
            "description": "1-day Value-at-Risk at 99% confidence (parametric method). Extreme tail risk with 1% probability of exceedance.",
        },
        {
            "feature_name": "mc_var_95_1d",
            "version": "v1",
            "computation_fn": compute_mc_var_95_1d,
            "description": "1-day VaR at 95% (Monte Carlo simulation with t-distribution). Captures fat tails better than parametric method.",
        },
        {
            "feature_name": "var_95_10d",
            "version": "v1",
            "computation_fn": compute_var_95_10d,
            "description": "10-day Value-at-Risk at 95% confidence (parametric method). Multi-day holding period risk.",
        },
    ]

    for feature in features:
        try:
            # Check if already registered
            existing = registry.get(feature["feature_name"], feature["version"])
            if existing:
                logger.debug(
                    f"Risk feature {feature['feature_name']} ({feature['version']}) already registered"
                )
                continue

            # Register the feature
            registry.register_feature(
                feature_name=feature["feature_name"],
                computation_fn=feature["computation_fn"],
                version=feature["version"],
                description=feature["description"],
                is_active=True,
            )
            logger.info(f"Registered risk feature: {feature['feature_name']} ({feature['version']})")

        except Exception as e:
            # Feature might already exist
            logger.debug(f"Could not register {feature['feature_name']}: {e}")
