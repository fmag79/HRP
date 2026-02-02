"""
HMM-based structural regime detection for scenario analysis.

Combines volatility and trend HMMs to classify market periods into 4 structural regimes:
- low_vol_bull: Low volatility, positive returns
- low_vol_bear: Low volatility, negative returns
- high_vol_bull: High volatility, positive returns
- high_vol_bear: High volatility, negative returns (crisis)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


# Type alias for structural regimes
StructuralRegime = Literal[
    "low_vol_bull",
    "low_vol_bear",
    "high_vol_bull",
    "high_vol_bear",
]


class VolatilityHMM:
    """Volatility regime classification using HMM."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self._model = None
        self._fitted = False

    def fit(self, volatility: np.ndarray) -> None:
        """Fit HMM to volatility series."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for regime detection. "
                "Install with: pip install hmmlearn>=0.3.0"
            ) from e

        # Reshape for HMM
        X = volatility.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
        )
        self._model.fit(X)
        self._fitted = True

    def predict(self, volatility: np.ndarray) -> np.ndarray:
        """Predict volatility regimes."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = volatility.reshape(-1, 1)
        regimes = self._model.predict(X)
        return regimes


class TrendHMM:
    """Trend regime classification using HMM."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self._model = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit HMM to returns series."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for regime detection. "
                "Install with: pip install hmmlearn>=0.3.0"
            ) from e

        X = returns.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
        )
        self._model.fit(X)
        self._fitted = True

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict trend regimes."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = returns.reshape(-1, 1)
        regimes = self._model.predict(X)
        return regimes


def combine_regime_labels(
    vol_regimes: np.ndarray,
    trend_regimes: np.ndarray,
) -> list[StructuralRegime]:
    """
    Combine volatility and trend regimes into structural regimes.

    Args:
        vol_regimes: Volatility regime labels (0=low, 1=high)
        trend_regimes: Trend regime labels (0=bull, 1=bear)

    Returns:
        List of structural regime labels

    Examples:
        >>> vol = [0, 0, 1, 1]
        >>> trend = [0, 1, 0, 1]
        >>> structural = combine_regime_labels(vol, trend)
        >>> assert structural[0] == "low_vol_bull"
        >>> assert structural[3] == "high_vol_bear"
    """
    if len(vol_regimes) != len(trend_regimes):
        raise ValueError(
            f"Volatility and trend regimes must have same length: "
            f"{len(vol_regimes)} != {len(trend_regimes)}"
        )

    vol_map = {0: "low_vol", 1: "high_vol"}
    trend_map = {0: "bull", 1: "bear"}

    structural = []
    for v, t in zip(vol_regimes, trend_regimes):
        regime = f"{vol_map[v]}_{trend_map[t]}"
        structural.append(regime)

    return structural


class StructuralRegimeClassifier:
    """
    Classify market periods into 4 structural regimes using HMM.

    Regimes:
    1. low_vol_bull: Calm uptrend
    2. low_vol_bear: Calm downtrend
    3. high_vol_bull: Volatile uptrend (recovery)
    4. high_vol_bear: Volatile downtrend (crisis)
    """

    def __init__(self):
        self.vol_hmm = VolatilityHMM(n_regimes=2)
        self.trend_hmm = TrendHMM(n_regimes=2)
        self._fitted = False

    def fit(self, prices: pd.DataFrame) -> None:
        """
        Fit HMMs to market price data.

        Args:
            prices: DataFrame with 'close' column
        """
        # Compute features
        returns = prices["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()

        # Align by index intersection (not positional slicing) to prevent
        # returns and volatility having misaligned dates
        common_idx = returns.index.intersection(volatility.index)
        returns = returns.loc[common_idx]
        volatility = volatility.loc[common_idx]

        # Fit HMMs
        self.vol_hmm.fit(volatility.values)
        self.trend_hmm.fit(returns.values)

        self._fitted = True

    def predict(self, prices: pd.DataFrame) -> list[StructuralRegime]:
        """
        Predict structural regimes.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            List of structural regime labels
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        # Compute features
        returns = prices["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()

        # Align by index intersection (not positional slicing)
        common_idx = returns.index.intersection(volatility.index)
        returns = returns.loc[common_idx]
        volatility = volatility.loc[common_idx]

        # Predict regimes
        vol_regimes = self.vol_hmm.predict(volatility.values)
        trend_regimes = self.trend_hmm.predict(returns.values)

        # Combine
        structural = combine_regime_labels(vol_regimes, trend_regimes)

        return structural

    def get_scenario_periods(
        self,
        prices: pd.DataFrame,
        min_days: int = 60,
    ) -> dict[StructuralRegime, list[tuple]]:
        """
        Get continuous periods for each structural regime.

        Args:
            prices: DataFrame with 'close' column and DatetimeIndex
            min_days: Minimum days for a period to be included

        Returns:
            Dict mapping regime to list of (start_date, end_date) tuples
        """
        import itertools

        structural = self.predict(prices)

        # predict() returns results aligned to the common index of returns
        # and volatility, so we use the same alignment logic to get dates
        returns = prices["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()
        common_idx = returns.index.intersection(volatility.index)
        dates = list(common_idx)

        # Group consecutive dates by regime
        periods: dict[StructuralRegime, list[tuple]] = {
            "low_vol_bull": [],
            "low_vol_bear": [],
            "high_vol_bull": [],
            "high_vol_bear": [],
        }

        for regime, group in itertools.groupby(
            zip(dates, structural), key=lambda x: x[1]
        ):
            group_list = list(group)
            start_date = group_list[0][0]
            end_date = group_list[-1][0]
            days = (end_date - start_date).days

            if days >= min_days:
                periods[regime].append((start_date, end_date))

        return periods
