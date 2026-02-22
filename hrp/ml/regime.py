"""
Hidden Markov Model regime detection.

Provides HMM-based market regime identification for adapting
strategy behavior to different market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class HMMConfig:
    """
    Configuration for HMM regime detection.

    Attributes:
        n_regimes: Number of hidden states/regimes (default 3)
        features: Features to use for regime detection
        covariance_type: Type of covariance matrix ("full", "diag", "spherical", "tied")
        n_iter: Maximum number of EM iterations (default 100)
        random_state: Random seed for reproducibility (default 42)
        tol: Convergence threshold (default 1e-4)
    """

    n_regimes: int = 3
    features: list[str] = field(default_factory=lambda: ["returns_20d", "volatility_20d"])
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    tol: float = 1e-4

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {self.n_regimes}")

        valid_cov_types = ("full", "diag", "spherical", "tied")
        if self.covariance_type not in valid_cov_types:
            raise ValueError(
                f"Invalid covariance_type: '{self.covariance_type}'. "
                f"Valid types: {', '.join(valid_cov_types)}"
            )

        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")

        logger.debug(
            f"HMMConfig created: {self.n_regimes} regimes, "
            f"features={self.features}, cov_type={self.covariance_type}"
        )


@dataclass
class RegimeResult:
    """
    Result of regime detection.

    Attributes:
        regimes: Series of regime labels indexed by date
        transition_matrix: State transition probability matrix
        regime_means: Mean values of features in each regime
        regime_covariances: Covariance matrices for each regime
        log_likelihood: Log-likelihood of the fitted model
        regime_durations: Average duration (in periods) of each regime
        regime_labels: Mapping of regime index to MarketRegime enum
    """

    regimes: pd.Series
    transition_matrix: np.ndarray
    regime_means: dict[int, dict[str, float]]
    regime_covariances: dict[int, np.ndarray]
    log_likelihood: float
    regime_durations: dict[int, float]
    regime_labels: dict[int, MarketRegime] = field(default_factory=dict)


class RegimeDetector:
    """
    Hidden Markov Model regime detector.

    Uses Gaussian HMM to identify market regimes based on
    return and volatility features.

    Usage:
        detector = RegimeDetector(HMMConfig(n_regimes=3))
        detector.fit(prices_df)
        regimes = detector.predict(prices_df)
        transition_matrix = detector.get_transition_matrix()
    """

    def __init__(self, config: HMMConfig):
        """
        Initialize regime detector.

        Args:
            config: HMM configuration
        """
        self.config = config
        self._model: Any = None  # GaussianHMM when fitted
        self._fitted = False
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self._fitted

    def _prepare_features(
        self,
        prices: pd.DataFrame,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Prepare features for HMM from price data.

        Args:
            prices: DataFrame with OHLCV data, indexed by date
            fit: If True, compute and store normalization parameters

        Returns:
            2D numpy array of features (n_samples, n_features)
        """
        features_data = []

        for feature_name in self.config.features:
            if feature_name == "returns_20d":
                # 20-day returns
                if "close" in prices.columns:
                    returns = prices["close"].pct_change(20)
                else:
                    returns = prices.iloc[:, 0].pct_change(20)
                features_data.append(returns.values)

            elif feature_name == "volatility_20d":
                # 20-day rolling volatility
                if "close" in prices.columns:
                    returns = prices["close"].pct_change()
                else:
                    returns = prices.iloc[:, 0].pct_change()
                vol = returns.rolling(20).std() * np.sqrt(252)
                features_data.append(vol.values)

            elif feature_name in prices.columns:
                features_data.append(prices[feature_name].values)

            else:
                raise ValueError(f"Unknown feature: {feature_name}")

        X = np.column_stack(features_data)

        # Handle NaN by forward-filling then removing remaining NaN
        df_temp = pd.DataFrame(X)
        df_temp = df_temp.ffill().bfill()
        X = df_temp.values

        # Normalize features
        if fit:
            self._feature_means = np.nanmean(X, axis=0)
            self._feature_stds = np.nanstd(X, axis=0)
            self._feature_stds[self._feature_stds == 0] = 1.0  # Avoid division by zero

        if self._feature_means is not None and self._feature_stds is not None:
            X = (X - self._feature_means) / self._feature_stds

        return X

    def fit(self, prices: pd.DataFrame) -> "RegimeDetector":
        """
        Fit HMM to historical price data.

        Args:
            prices: DataFrame with OHLCV data

        Returns:
            self for method chaining

        Raises:
            ImportError: If hmmlearn is not installed
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for regime detection. "
                "Install with: pip install hmmlearn>=0.3.0"
            ) from e

        logger.info(f"Fitting HMM regime detector with {self.config.n_regimes} regimes")

        # Prepare features
        X = self._prepare_features(prices, fit=True)

        # Remove any remaining NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < self.config.n_regimes * 10:
            raise ValueError(
                f"Insufficient data for {self.config.n_regimes} regimes. "
                f"Need at least {self.config.n_regimes * 10} samples, "
                f"got {len(X_clean)}"
            )

        # Create and fit HMM
        self._model = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            tol=self.config.tol,
        )

        self._model.fit(X_clean)
        self._fitted = True

        logger.info(
            f"HMM fitted: log_likelihood={self._model.score(X_clean):.2f}, "
            f"converged={self._model.monitor_.converged}"
        )

        return self

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each time period.

        Args:
            prices: DataFrame with OHLCV data

        Returns:
            Series of regime labels (integers 0 to n_regimes-1)

        Raises:
            ValueError: If model not fitted
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        X = self._prepare_features(prices, fit=False)

        # Handle NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]

        # Predict
        predictions = self._model.predict(X_clean)

        # Create full-length result with NaN for invalid rows
        full_predictions = np.full(len(X), np.nan)
        full_predictions[valid_mask] = predictions

        # Create Series with date index
        if hasattr(prices, "index"):
            index = prices.index
        else:
            index = pd.RangeIndex(len(prices))

        return pd.Series(full_predictions, index=index, name="regime")

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get state transition probability matrix.

        Returns:
            DataFrame with transition probabilities (from_state -> to_state)

        Raises:
            ValueError: If model not fitted
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call fit() first.")

        trans_mat = self._model.transmat_

        return pd.DataFrame(
            trans_mat,
            index=[f"from_{i}" for i in range(self.config.n_regimes)],
            columns=[f"to_{i}" for i in range(self.config.n_regimes)],
        )

    def get_regime_statistics(self, prices: pd.DataFrame) -> RegimeResult:
        """
        Get comprehensive regime statistics.

        Args:
            prices: DataFrame with OHLCV data (same format as fit/predict)

        Returns:
            RegimeResult with full statistics

        Raises:
            ValueError: If model not fitted
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call fit() first.")

        regimes = self.predict(prices)

        # Transition matrix
        trans_mat = self._model.transmat_

        # Regime means (from HMM parameters, denormalized)
        regime_means = {}
        for i in range(self.config.n_regimes):
            means_normalized = self._model.means_[i]
            means_denorm = (
                means_normalized * self._feature_stds + self._feature_means
            )
            regime_means[i] = {
                feat: float(means_denorm[j])
                for j, feat in enumerate(self.config.features)
            }

        # Regime covariances
        regime_covariances = {}
        for i in range(self.config.n_regimes):
            if self.config.covariance_type == "full":
                regime_covariances[i] = self._model.covars_[i]
            elif self.config.covariance_type == "diag":
                regime_covariances[i] = np.diag(self._model.covars_[i])
            elif self.config.covariance_type == "spherical":
                regime_covariances[i] = np.eye(len(self.config.features)) * self._model.covars_[i]
            else:  # tied
                regime_covariances[i] = self._model.covars_

        # Log-likelihood
        X = self._prepare_features(prices, fit=False)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        log_likelihood = float(self._model.score(X_clean))

        # Regime durations (expected sojourn times)
        regime_durations = {}
        for i in range(self.config.n_regimes):
            # Expected duration = 1 / (1 - self-transition probability)
            p_stay = trans_mat[i, i]
            if p_stay < 1:
                regime_durations[i] = 1.0 / (1.0 - p_stay)
            else:
                regime_durations[i] = float("inf")

        # Label regimes based on mean returns
        regime_labels = self._label_regimes(regime_means)

        return RegimeResult(
            regimes=regimes,
            transition_matrix=trans_mat,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            log_likelihood=log_likelihood,
            regime_durations=regime_durations,
            regime_labels=regime_labels,
        )

    def predict_proba(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime probabilities for each time step.

        Returns DataFrame with columns: P(bull), P(bear), P(sideways), etc.
        Rows are dates. Probabilities sum to 1.0 per row.

        Uses HMM forward algorithm (not just Viterbi decoding).

        Args:
            prices: DataFrame with OHLCV data

        Returns:
            DataFrame with regime probabilities indexed by date

        Raises:
            ValueError: If model not fitted
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        logger.debug("Computing regime probabilities using HMM forward algorithm")

        X = self._prepare_features(prices, fit=False)

        # Handle NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]

        # Get state probabilities using forward algorithm
        proba = self._model.predict_proba(X_clean)

        # Get regime labels for column names
        regime_labels_dict = {}
        # Need to compute regime labels first
        regime_means_dict = {}
        for i in range(self.config.n_regimes):
            means_normalized = self._model.means_[i]
            means_denorm = (
                means_normalized * self._feature_stds + self._feature_means
            )
            regime_means_dict[i] = {
                feat: float(means_denorm[j])
                for j, feat in enumerate(self.config.features)
            }
        regime_labels_dict = self._label_regimes(regime_means_dict)

        # Create DataFrame with labeled columns
        columns = [regime_labels_dict[i].value for i in range(self.config.n_regimes)]
        proba_df = pd.DataFrame(proba, columns=columns)

        # Create full-length result with NaN for invalid rows
        full_proba = pd.DataFrame(np.nan, index=prices.index, columns=columns)
        full_proba.iloc[valid_mask] = proba_df.values

        logger.debug(
            f"Regime probabilities computed: shape={full_proba.shape}, "
            f"columns={columns}"
        )

        return full_proba

    def _label_regimes(
        self,
        regime_means: dict[int, dict[str, float]],
    ) -> dict[int, MarketRegime]:
        """
        Automatically label regimes based on characteristics.

        Uses mean returns and volatility to classify regimes.
        """
        labels = {}

        # Sort regimes by return (if available)
        return_feature = "returns_20d"
        vol_feature = "volatility_20d"

        if return_feature in self.config.features:
            returns_by_regime = [
                (i, regime_means[i].get(return_feature, 0))
                for i in range(self.config.n_regimes)
            ]
            returns_by_regime.sort(key=lambda x: x[1])

            if self.config.n_regimes == 2:
                labels[returns_by_regime[0][0]] = MarketRegime.BEAR
                labels[returns_by_regime[1][0]] = MarketRegime.BULL
            elif self.config.n_regimes == 3:
                labels[returns_by_regime[0][0]] = MarketRegime.BEAR
                labels[returns_by_regime[1][0]] = MarketRegime.SIDEWAYS
                labels[returns_by_regime[2][0]] = MarketRegime.BULL
            elif self.config.n_regimes >= 4:
                # Lowest return + highest vol = CRISIS
                if vol_feature in self.config.features:
                    vol_by_regime = {
                        i: regime_means[i].get(vol_feature, 0)
                        for i in range(self.config.n_regimes)
                    }
                    lowest_return_idx = returns_by_regime[0][0]
                    if vol_by_regime[lowest_return_idx] >= np.median(list(vol_by_regime.values())):
                        labels[lowest_return_idx] = MarketRegime.CRISIS
                    else:
                        labels[lowest_return_idx] = MarketRegime.BEAR

                labels[returns_by_regime[-1][0]] = MarketRegime.BULL

                # Label remaining as SIDEWAYS or BEAR
                for i, _ in returns_by_regime[1:-1]:
                    if i not in labels:
                        labels[i] = MarketRegime.SIDEWAYS
        else:
            # Default labels
            for i in range(self.config.n_regimes):
                labels[i] = MarketRegime.SIDEWAYS

        return labels
