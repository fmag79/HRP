"""
Model Drift Detection and Monitoring for HRP.

Provides statistical drift detection for deployed ML models including:
- Prediction drift (KL divergence)
- Feature drift (Population Stability Index - PSI)
- Concept drift (IC decay detection)

Alerts when model performance degrades due to changing market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from hrp.data.db import get_db
from hrp.research.lineage import log_event, EventType


@dataclass
class DriftThresholds:
    """
    Thresholds for triggering drift alerts.

    Attributes:
        kl_threshold: KL divergence threshold for prediction drift (default: 0.2)
        psi_threshold: PSI threshold for feature drift (default: 0.2)
        ic_decay_threshold: IC decay ratio threshold (default: 0.2)
        min_samples: Minimum samples required for drift check (default: 100)
    """

    kl_threshold: float = 0.2
    psi_threshold: float = 0.2
    ic_decay_threshold: float = 0.2
    min_samples: int = 100


@dataclass
class DriftResult:
    """
    Result of a drift detection check.

    Attributes:
        drift_type: Type of drift ('prediction', 'feature', 'concept')
        feature_name: Feature name (None for overall prediction/concept drift)
        metric_value: Calculated drift metric value
        is_drift_detected: Whether drift threshold was exceeded
        threshold_value: Threshold value for comparison
        timestamp: When the check was performed
        details: Additional details about the drift
    """

    drift_type: str
    feature_name: Optional[str]
    metric_value: float
    is_drift_detected: bool
    threshold_value: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "drift_type": self.drift_type,
            "feature_name": self.feature_name,
            "metric_value": self.metric_value,
            "is_drift_detected": self.is_drift_detected,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class DriftMonitor:
    """
    Model drift detection and monitoring system.

    Detects three types of drift:
    1. Prediction Drift: KL divergence between reference and current predictions
    2. Feature Drift: Population Stability Index (PSI) for each feature
    3. Concept Drift: Information Coefficient (IC) decay over time

    Example:
        ```python
        from hrp.monitoring.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        drift_results = monitor.run_drift_check(
            model_name="momentum_strategy",
            current_data=current_features,
            reference_data=historical_features,
        )
        ```
    """

    def __init__(
        self,
        thresholds: DriftThresholds | None = None,
        db_path: str | None = None,
        db=None,
    ):
        """
        Initialize the drift monitor.

        Args:
            thresholds: Custom drift thresholds (uses defaults if None)
            db_path: Optional database path
            db: Optional database connection (uses default if not provided)
        """
        self.thresholds = thresholds or DriftThresholds()
        self.db_path = db_path
        self._db = db or get_db(db_path)

        logger.info(
            f"DriftMonitor initialized (thresholds: "
            f"kl={self.thresholds.kl_threshold}, "
            f"psi={self.thresholds.psi_threshold}, "
            f"ic_decay={self.thresholds.ic_decay_threshold})"
        )

    def check_prediction_drift(
        self,
        model_name: str,
        predictions_ref: np.ndarray,
        predictions_new: np.ndarray,
        model_version: str | None = None,
    ) -> DriftResult:
        """
        Check for prediction distribution drift using KL divergence.

        KL divergence measures the difference between two probability distributions.
        Higher values indicate more drift.

        Args:
            model_name: Name of the model being monitored
            predictions_ref: Reference predictions (e.g., from training)
            predictions_new: New predictions to check for drift
            model_version: Optional model version

        Returns:
            DriftResult with KL divergence metric

        Raises:
            ValueError: If insufficient samples
        """
        if len(predictions_ref) < self.thresholds.min_samples:
            raise ValueError(
                f"Insufficient reference samples: {len(predictions_ref)} "
                f"< {self.thresholds.min_samples}"
            )
        if len(predictions_new) < self.thresholds.min_samples:
            raise ValueError(
                f"Insufficient new samples: {len(predictions_new)} "
                f"< {self.thresholds.min_samples}"
            )

        # Create histograms with same bins
        hist_ref, bins = np.histogram(predictions_ref, bins=50, density=True)
        hist_new, _ = np.histogram(predictions_new, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_ref = hist_ref + epsilon
        hist_new = hist_new + +epsilon

        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_new = hist_new / hist_new.sum()

        # KL divergence
        kl_divergence = np.sum(hist_ref * np.log(hist_ref / hist_new))

        is_drift = kl_divergence > self.thresholds.kl_threshold

        result = DriftResult(
            drift_type="prediction",
            feature_name=None,
            metric_value=float(kl_divergence),
            is_drift_detected=is_drift,
            threshold_value=self.thresholds.kl_threshold,
            timestamp=datetime.now(),
            details={
                "model_name": model_name,
                "model_version": model_version,
                "ref_samples": len(predictions_ref),
                "new_samples": len(predictions_new),
                "ref_mean": float(np.mean(predictions_ref)),
                "new_mean": float(np.mean(predictions_new)),
                "ref_std": float(np.std(predictions_ref)),
                "new_std": float(np.std(predictions_new)),
            },
        )

        # Log to database
        self._log_drift_check(model_name, model_version, result)

        if is_drift:
            logger.warning(
                f"Prediction drift detected for {model_name}: "
                f"KL={kl_divergence:.4f} > {self.thresholds.kl_threshold}"
            )

        return result

    def check_feature_drift(
        self,
        model_name: str,
        features_ref: pd.DataFrame,
        features_new: pd.DataFrame,
        model_version: str | None = None,
    ) -> dict[str, DriftResult]:
        """
        Check for feature drift using Population Stability Index (PSI).

        PSI measures the difference between two distributions.
        PSI > 0.2 indicates significant drift.

        Args:
            model_name: Name of the model being monitored
            features_ref: Reference feature DataFrame
            features_new: New feature DataFrame to check
            model_version: Optional model version

        Returns:
            Dict mapping feature names to DriftResults

        Raises:
            ValueError: If DataFrames have different columns
        """
        if features_ref.columns.tolist() != features_new.columns.tolist():
            raise ValueError("Feature columns must match between reference and new data")

        results = {}

        for column in features_ref.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(features_ref[column]):
                continue

            ref_values = features_ref[column].dropna().values
            new_values = features_new[column].dropna().values

            if len(ref_values) < self.thresholds.min_samples:
                logger.warning(f"Insufficient samples for {column}: {len(ref_values)}")
                continue

            if len(new_values) < self.thresholds.min_samples:
                logger.warning(f"Insufficient samples for {column}: {len(new_values)}")
                continue

            # Create histograms with same bins
            hist_ref, bins = np.histogram(ref_values, bins=10, density=True)
            hist_new, _ = np.histogram(new_values, bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist_ref = hist_ref + epsilon
            hist_new = hist_new + epsilon

            # Normalize
            hist_ref = hist_ref / hist_ref.sum()
            hist_new = hist_new / hist_new.sum()

            # PSI calculation
            psi = np.sum((hist_ref - hist_new) * np.log(hist_ref / hist_new))

            is_drift = psi > self.thresholds.psi_threshold

            result = DriftResult(
                drift_type="feature",
                feature_name=column,
                metric_value=float(psi),
                is_drift_detected=is_drift,
                threshold_value=self.thresholds.psi_threshold,
                timestamp=datetime.now(),
                details={
                    "model_name": model_name,
                    "model_version": model_version,
                    "ref_samples": len(ref_values),
                    "new_samples": len(new_values),
                    "ref_mean": float(np.mean(ref_values)),
                    "new_mean": float(np.mean(new_values)),
                },
            )

            results[column] = result
            self._log_drift_check(model_name, model_version, result)

            if is_drift:
                logger.warning(
                    f"Feature drift detected for {column}: "
                    f"PSI={psi:.4f} > {self.thresholds.psi_threshold}"
                )

        return results

    def check_concept_drift(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        reference_ic: float,
        model_version: str | None = None,
    ) -> DriftResult:
        """
        Check for concept drift using IC (Information Coefficient) decay.

        Concept drift occurs when the relationship between features and target changes.
        This is detected by a significant drop in IC compared to reference.

        Args:
            model_name: Name of the model being monitored
            predictions: Model predictions
            actuals: Actual target values
            reference_ic: Reference IC from training/validation
            model_version: Optional model version

        Returns:
            DriftResult with IC decay metric

        Raises:
            ValueError: If predictions and actuals have different lengths
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                "must have same length"
            )

        # Calculate current IC using Spearman correlation
        current_ic, _ = stats.spearmanr(predictions, actuals)

        # Handle NaN from insufficient variance
        if np.isnan(current_ic):
            current_ic = 0.0

        # Calculate decay ratio
        # Avoid division by zero
        if abs(reference_ic) < 1e-10:
            ic_decay = abs(current_ic)
        else:
            ic_decay = abs((reference_ic - current_ic) / reference_ic)

        is_drift = ic_decay > self.thresholds.ic_decay_threshold

        result = DriftResult(
            drift_type="concept",
            feature_name=None,
            metric_value=float(ic_decay),
            is_drift_detected=is_drift,
            threshold_value=self.thresholds.ic_decay_threshold,
            timestamp=datetime.now(),
            details={
                "model_name": model_name,
                "model_version": model_version,
                "reference_ic": float(reference_ic),
                "current_ic": float(current_ic),
                "samples": len(predictions),
            },
        )

        # Log to database
        self._log_drift_check(model_name, model_version, result)

        if is_drift:
            logger.warning(
                f"Concept drift detected for {model_name}: "
                f"IC decay={ic_decay:.4f} > {self.thresholds.ic_decay_threshold} "
                f"(ref_ic={reference_ic:.4f}, curr_ic={current_ic:.4f})"
            )

        return result

    def run_drift_check(
        self,
        model_name: str,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        model_version: str | None = None,
        predictions_col: str = "prediction",
        target_col: str | None = None,
        reference_ic: float | None = None,
    ) -> dict[str, DriftResult]:
        """
        Run all drift checks for a model.

        Performs prediction drift, feature drift, and concept drift checks.

        Args:
            model_name: Name of the model being monitored
            current_data: Current feature/prediction DataFrame
            reference_data: Reference feature/prediction DataFrame
            model_version: Optional model version
            predictions_col: Column name for predictions
            target_col: Column name for target (required for concept drift)
            reference_ic: Reference IC for concept drift check

        Returns:
            Dict mapping drift check names to DriftResults
        """
        results = {}

        # Prediction drift
        if predictions_col in current_data.columns and predictions_col in reference_data.columns:
            pred_result = self.check_prediction_drift(
                model_name=model_name,
                predictions_ref=reference_data[predictions_col].values,
                predictions_new=current_data[predictions_col].values,
                model_version=model_version,
            )
            results["prediction_drift"] = pred_result

        # Feature drift (exclude prediction and target columns)
        feature_cols = [
            col
            for col in current_data.columns
            if col != predictions_col and col != target_col
        ]
        if feature_cols:
            feature_results = self.check_feature_drift(
                model_name=model_name,
                features_ref=reference_data[feature_cols],
                features_new=current_data[feature_cols],
                model_version=model_version,
            )
            for feat_name, feat_result in feature_results.items():
                results[f"feature_drift_{feat_name}"] = feat_result

        # Concept drift (requires target and reference_ic)
        if (
            target_col
            and target_col in current_data.columns
            and predictions_col in current_data.columns
            and reference_ic is not None
        ):
            concept_result = self.check_concept_drift(
                model_name=model_name,
                predictions=current_data[predictions_col].values,
                actuals=current_data[target_col].values,
                reference_ic=reference_ic,
                model_version=model_version,
            )
            results["concept_drift"] = concept_result

        # Log overall drift check to lineage
        drift_detected = any(r.is_drift_detected for r in results.values())
        log_event(
            event_type=EventType.VALIDATION_PASSED
            if not drift_detected
            else EventType.VALIDATION_FAILED,
            actor="system",
            details={
                "model_name": model_name,
                "model_version": model_version,
                "drift_check_type": "full",
                "drift_detected": drift_detected,
                "num_checks": len(results),
                "num_drifts": sum(1 for r in results.values() if r.is_drift_detected),
            },
        )

        return results

    def _log_drift_check(
        self,
        model_name: str,
        model_version: str | None,
        result: DriftResult,
    ) -> None:
        """Log drift check result to database."""
        try:
            query = """
                INSERT INTO model_drift_checks (
                    model_name,
                    model_version,
                    check_timestamp,
                    drift_type,
                    feature_name,
                    metric_value,
                    is_drift_detected,
                    threshold_value,
                    details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            import json

            self._db.execute(
                query,
                (
                    model_name,
                    model_version,
                    result.timestamp,
                    result.drift_type,
                    result.feature_name,
                    result.metric_value,
                    bool(result.is_drift_detected),  # Convert numpy.bool to Python bool
                    result.threshold_value,
                    json.dumps(result.details),
                ),
            )

        except Exception as e:
            # Table might not exist yet, log warning but don't fail
            logger.warning(f"Failed to log drift check to database: {e}")

    def get_drift_history(
        self,
        model_name: str,
        drift_type: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get drift check history for a model.

        Args:
            model_name: Name of the model
            drift_type: Optional drift type filter
            limit: Maximum number of records to return

        Returns:
            DataFrame with drift check history
        """
        query = """
            SELECT
                check_timestamp,
                drift_type,
                feature_name,
                metric_value,
                is_drift_detected,
                threshold_value,
                details
            FROM model_drift_checks
            WHERE model_name = ?
        """

        params = [model_name]

        if drift_type:
            query += " AND drift_type = ?"
            params.append(drift_type)

        query += " ORDER BY check_timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            return self._db.fetchdf(query, params)
        except Exception:
            # Table might not exist yet
            return pd.DataFrame()


def get_drift_monitor(
    thresholds: DriftThresholds | None = None,
    db_path: str | None = None,
) -> DriftMonitor:
    """
    Get a DriftMonitor instance.

    Args:
        thresholds: Custom drift thresholds
        db_path: Optional database path

    Returns:
        DriftMonitor instance
    """
    return DriftMonitor(thresholds=thresholds, db_path=db_path)
