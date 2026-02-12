"""Feature importance tracking for ML-driven strategies.

Tracks which features drive portfolio decisions over time using:
- Permutation importance (model-agnostic)
- SHAP values (tree-based models, optional)
- Rolling importance (time-varying feature importance)
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


@dataclass
class ImportanceResult:
    """Result of a feature importance calculation.

    Attributes:
        feature_name: Name of the feature
        importance_score: Importance score (0-1, normalized)
        direction: Whether feature has positive or negative impact
        period: Time period for this importance measurement (optional)
        method: Method used to compute importance ("permutation" or "shap")
    """

    feature_name: str
    importance_score: float
    direction: Literal["positive", "negative", "neutral"]
    period: str | None = None
    method: Literal["permutation", "shap"] = "permutation"

    def __post_init__(self):
        """Validate fields."""
        if self.importance_score < 0:
            raise ValueError(f"importance_score must be >= 0, got {self.importance_score}")
        if self.direction not in ["positive", "negative", "neutral"]:
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.method not in ["permutation", "shap"]:
            raise ValueError(f"Invalid method: {self.method}")


class FeatureImportanceTracker:
    """Track feature importance using permutation importance or SHAP.

    Provides model-agnostic feature importance tracking via permutation importance,
    with optional SHAP integration for tree-based models.
    """

    def __init__(
        self,
        n_repeats: int = 10,
        random_state: int | None = 42,
        shap_enabled: bool = False,
    ):
        """Initialize feature importance tracker.

        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducibility
            shap_enabled: Whether to use SHAP (requires 'shap' package)
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.shap_enabled = shap_enabled
        self.importance_scores_: dict[str, float] | None = None
        self.feature_directions_: dict[str, str] | None = None

    def compute_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
    ) -> list[ImportanceResult]:
        """Compute permutation importance for all features.

        Args:
            model: Fitted sklearn-compatible model
            X: Feature matrix
            y: Target variable
            feature_names: Optional list of feature names (uses X.columns if None)

        Returns:
            List of ImportanceResult objects, sorted by importance (descending)

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Get feature names
        if feature_names is None:
            feature_names = list(X.columns)

        # Compute permutation importance
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
        )

        # Normalize importance scores to [0, 1]
        importances = result.importances_mean
        if importances.max() > 0:
            importances = importances / importances.max()
        else:
            importances = np.zeros_like(importances)

        # Store importance scores
        self.importance_scores_ = dict(zip(feature_names, importances))

        # Compute feature directions (positive/negative impact)
        self.feature_directions_ = self._compute_feature_directions(
            model, X, y, feature_names
        )

        # Create results
        results = []
        for i, feature_name in enumerate(feature_names):
            results.append(
                ImportanceResult(
                    feature_name=feature_name,
                    importance_score=importances[i],
                    direction=self.feature_directions_[feature_name],  # type: ignore
                    method="permutation",
                )
            )

        # Sort by importance (descending)
        results.sort(key=lambda r: r.importance_score, reverse=True)

        return results

    def compute_shap_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> list[ImportanceResult]:
        """Compute SHAP importance for tree-based models.

        Args:
            model: Fitted tree-based model (LightGBM, XGBoost, etc.)
            X: Feature matrix
            feature_names: Optional list of feature names (uses X.columns if None)

        Returns:
            List of ImportanceResult objects, sorted by importance (descending)

        Raises:
            ImportError: If 'shap' package not installed
            ValueError: If inputs are invalid
        """
        if not self.shap_enabled:
            raise ValueError("SHAP is not enabled. Set shap_enabled=True in constructor.")

        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package not installed. Install with: pip install shap"
            )

        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Get feature names
        if feature_names is None:
            feature_names = list(X.columns)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle multi-output case (take first output)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Compute mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Normalize to [0, 1]
        if mean_abs_shap.max() > 0:
            mean_abs_shap = mean_abs_shap / mean_abs_shap.max()
        else:
            mean_abs_shap = np.zeros_like(mean_abs_shap)

        # Store importance scores
        self.importance_scores_ = dict(zip(feature_names, mean_abs_shap))

        # Compute feature directions from mean SHAP values
        mean_shap = shap_values.mean(axis=0)
        self.feature_directions_ = {}
        for i, feature_name in enumerate(feature_names):
            if mean_shap[i] > 0.01:
                self.feature_directions_[feature_name] = "positive"
            elif mean_shap[i] < -0.01:
                self.feature_directions_[feature_name] = "negative"
            else:
                self.feature_directions_[feature_name] = "neutral"

        # Create results
        results = []
        for i, feature_name in enumerate(feature_names):
            results.append(
                ImportanceResult(
                    feature_name=feature_name,
                    importance_score=mean_abs_shap[i],
                    direction=self.feature_directions_[feature_name],
                    method="shap",
                )
            )

        # Sort by importance (descending)
        results.sort(key=lambda r: r.importance_score, reverse=True)

        return results

    def _compute_feature_directions(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
    ) -> dict[str, str]:
        """Compute direction (positive/negative) for each feature.

        Direction is determined by correlation between feature and model predictions.

        Args:
            model: Fitted model
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature_name -> direction
        """
        directions = {}

        # Get model predictions
        y_pred = model.predict(X)

        # Compute correlation for each feature
        for feature_name in feature_names:
            corr = np.corrcoef(X[feature_name], y_pred)[0, 1]
            if np.isnan(corr):
                directions[feature_name] = "neutral"
            elif corr > 0.1:
                directions[feature_name] = "positive"
            elif corr < -0.1:
                directions[feature_name] = "negative"
            else:
                directions[feature_name] = "neutral"

        return directions


class RollingImportance:
    """Track feature importance over rolling time windows.

    Computes feature importance on sliding windows to detect how importance
    changes over time (useful for regime changes, concept drift).
    """

    def __init__(
        self,
        window_days: int = 60,
        step_days: int = 10,
        tracker: FeatureImportanceTracker | None = None,
    ):
        """Initialize rolling importance tracker.

        Args:
            window_days: Size of rolling window (in trading days)
            step_days: Step size for sliding window
            tracker: FeatureImportanceTracker instance (creates default if None)
        """
        self.window_days = window_days
        self.step_days = step_days
        self.tracker = tracker or FeatureImportanceTracker()
        self.rolling_results_: list[tuple[str, list[ImportanceResult]]] = []

    def compute_rolling_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute feature importance over rolling windows.

        Args:
            model: Fitted sklearn-compatible model
            X: Feature matrix (must align with dates)
            y: Target variable
            dates: DatetimeIndex for X/y
            feature_names: Optional list of feature names

        Returns:
            DataFrame with columns = features, index = window end dates,
            values = importance scores

        Raises:
            ValueError: If inputs are invalid or misaligned
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if not isinstance(dates, pd.DatetimeIndex):
            raise ValueError("dates must be a DatetimeIndex")
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, and dates must have same length")

        # Get feature names
        if feature_names is None:
            feature_names = list(X.columns)

        # Convert to DataFrame with date index
        data = X.copy()
        data.index = dates
        y_indexed = y.copy()
        y_indexed.index = dates

        # Compute rolling importance
        rolling_scores = []
        window_dates = []

        # Slide window
        start_date = dates[0]
        end_date = dates[-1]
        current_end = start_date + pd.Timedelta(days=self.window_days)

        while current_end <= end_date:
            # Get window data
            window_start = current_end - pd.Timedelta(days=self.window_days)
            window_mask = (dates >= window_start) & (dates < current_end)

            if window_mask.sum() >= 10:  # Minimum samples for meaningful importance
                X_window = data[window_mask]
                y_window = y_indexed[window_mask]

                # Refit model on window (or use existing model - depends on use case)
                # For now, we assume model is already fitted and use permutation importance
                try:
                    results = self.tracker.compute_permutation_importance(
                        model, X_window, y_window, feature_names
                    )

                    # Extract scores
                    scores = {r.feature_name: r.importance_score for r in results}
                    rolling_scores.append(scores)
                    window_dates.append(current_end)

                    # Store results with period label
                    period_label = f"{window_start.date()} to {current_end.date()}"
                    self.rolling_results_.append((period_label, results))

                except Exception as e:
                    # Skip windows that fail (e.g., insufficient data)
                    pass

            # Advance window
            current_end += pd.Timedelta(days=self.step_days)

        # Convert to DataFrame
        if len(rolling_scores) == 0:
            raise ValueError("No valid windows found for rolling importance")

        df = pd.DataFrame(rolling_scores, index=window_dates)
        df = df.fillna(0.0)

        return df

    def get_top_features_by_period(
        self, n_features: int = 5
    ) -> dict[str, list[str]]:
        """Get top N features for each time period.

        Args:
            n_features: Number of top features to return per period

        Returns:
            Dictionary mapping period -> list of top feature names
        """
        if not self.rolling_results_:
            raise ValueError("No rolling results available. Run compute_rolling_importance first.")

        top_features = {}
        for period, results in self.rolling_results_:
            # Sort by importance and take top N
            sorted_results = sorted(results, key=lambda r: r.importance_score, reverse=True)
            top_names = [r.feature_name for r in sorted_results[:n_features]]
            top_features[period] = top_names

        return top_features
