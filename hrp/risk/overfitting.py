"""
Overfitting prevention mechanisms.

Implements test set discipline and overfitting detection.
"""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.exceptions import OverfittingError


@dataclass
class DecayCheckResult:
    """Result of a Sharpe decay check."""
    passed: bool
    decay_ratio: float
    train_sharpe: float
    test_sharpe: float
    message: str


class SharpeDecayMonitor:
    """
    Monitor for train/test Sharpe ratio decay.

    Detects overfitting by comparing in-sample vs out-of-sample Sharpe ratios.
    A large decay indicates the strategy may be overfit to training data.

    Usage:
        monitor = SharpeDecayMonitor(max_decay_ratio=0.5)
        result = monitor.check(train_sharpe=1.5, test_sharpe=1.0)
        if not result.passed:
            print(f"Warning: {result.message}")
    """

    def __init__(self, max_decay_ratio: float = 0.5):
        """
        Initialize Sharpe decay monitor.

        Args:
            max_decay_ratio: Maximum allowed decay ratio (0.5 = 50% decay allowed)
        """
        if not 0 < max_decay_ratio < 1:
            raise ValueError("max_decay_ratio must be between 0 and 1")
        self.max_decay_ratio = max_decay_ratio

    def check(self, train_sharpe: float, test_sharpe: float) -> DecayCheckResult:
        """
        Check Sharpe decay between train and test.

        Args:
            train_sharpe: In-sample Sharpe ratio
            test_sharpe: Out-of-sample Sharpe ratio

        Returns:
            DecayCheckResult with pass/fail status and details
        """
        # Handle negative test Sharpe
        if test_sharpe < 0:
            return DecayCheckResult(
                passed=False,
                decay_ratio=1.0,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                message=f"Negative test Sharpe ({test_sharpe:.2f}) indicates severe overfitting",
            )

        # Handle zero or negative train Sharpe
        if train_sharpe <= 0:
            return DecayCheckResult(
                passed=True,
                decay_ratio=0.0,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                message="Train Sharpe <= 0, decay ratio not applicable",
            )

        # Calculate decay ratio
        decay_ratio = (train_sharpe - test_sharpe) / train_sharpe
        decay_ratio = max(0, decay_ratio)  # Can't have negative decay

        passed = decay_ratio <= self.max_decay_ratio

        if passed:
            message = f"Sharpe decay {decay_ratio:.1%} within threshold ({self.max_decay_ratio:.1%})"
        else:
            message = (
                f"Sharpe decay {decay_ratio:.1%} exceeds threshold ({self.max_decay_ratio:.1%}). "
                f"Train: {train_sharpe:.2f}, Test: {test_sharpe:.2f}"
            )

        return DecayCheckResult(
            passed=passed,
            decay_ratio=decay_ratio,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            message=message,
        )


@dataclass
class FeatureCountResult:
    """Result of feature count validation."""
    passed: bool
    warning: bool
    feature_count: int
    sample_count: int
    features_per_sample: float
    message: str


class FeatureCountValidator:
    """
    Validator for number of features used in ML models.

    Prevents overfitting by limiting feature count and checking
    features-per-sample ratio.

    Rules:
    - Warn if features > warn_threshold (default 30)
    - Fail if features > max_threshold (default 50)
    - Warn if features/samples > 0.1 (need at least 10 samples per feature)

    Usage:
        validator = FeatureCountValidator(warn_threshold=30)
        result = validator.check(feature_count=25, sample_count=1000)
    """

    def __init__(
        self,
        warn_threshold: int = 30,
        max_threshold: int = 50,
        min_samples_per_feature: int = 10,
    ):
        """
        Initialize feature count validator.

        Args:
            warn_threshold: Feature count that triggers warning
            max_threshold: Feature count that triggers failure
            min_samples_per_feature: Minimum samples per feature ratio
        """
        if warn_threshold >= max_threshold:
            raise ValueError("warn_threshold must be less than max_threshold")

        self.warn_threshold = warn_threshold
        self.max_threshold = max_threshold
        self.min_samples_per_feature = min_samples_per_feature

    def check(self, feature_count: int, sample_count: int) -> FeatureCountResult:
        """
        Validate feature count.

        Args:
            feature_count: Number of features
            sample_count: Number of training samples

        Returns:
            FeatureCountResult with pass/fail/warning status
        """
        features_per_sample = feature_count / sample_count if sample_count > 0 else float('inf')
        samples_per_feature = sample_count / feature_count if feature_count > 0 else float('inf')

        messages = []
        warning = False
        passed = True

        # Check absolute feature count
        if feature_count > self.max_threshold:
            passed = False
            messages.append(
                f"Feature count ({feature_count}) exceeds maximum ({self.max_threshold})"
            )
        elif feature_count > self.warn_threshold:
            warning = True
            messages.append(
                f"Warning: Feature count ({feature_count}) exceeds threshold ({self.warn_threshold})"
            )

        # Check samples-per-feature ratio
        if samples_per_feature < self.min_samples_per_feature:
            warning = True
            messages.append(
                f"Warning: Only {samples_per_feature:.1f} samples per feature "
                f"(recommended: {self.min_samples_per_feature}+)"
            )

        if not messages:
            messages.append(
                f"Feature count ({feature_count}) OK with {samples_per_feature:.0f} samples/feature"
            )

        return FeatureCountResult(
            passed=passed,
            warning=warning,
            feature_count=feature_count,
            sample_count=sample_count,
            features_per_sample=features_per_sample,
            message="; ".join(messages),
        )


def _load_evaluation_count(hypothesis_id: str, db=None) -> int:
    """Load existing evaluation count from database."""
    db = db or get_db()
    
    with db.connection() as conn:
        result = conn.execute(
            """
            SELECT COUNT(*) 
            FROM test_set_evaluations 
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
    
    return result[0] if result else 0


def _log_evaluation(
    hypothesis_id: str,
    override: bool,
    override_reason: str | None,
    metadata: dict[str, Any] | None,
    db=None,
) -> None:
    """Log test set evaluation to database."""
    db = db or get_db()

    # Convert metadata dict to JSON string if present
    metadata_json = json.dumps(metadata) if metadata else None

    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO test_set_evaluations
            (evaluation_id, hypothesis_id, evaluated_at, override, override_reason, metadata)
            VALUES (
                (SELECT COALESCE(MAX(evaluation_id), 0) + 1 FROM test_set_evaluations),
                ?, ?, ?, ?, ?
            )
            """,
            (
                hypothesis_id,
                datetime.utcnow(),
                override,
                override_reason,
                metadata_json,
            ),
        )


class TestSetGuard:
    """
    Guard against excessive test set evaluation.
    
    Enforces limit on number of times test set can be evaluated per hypothesis
    to prevent data snooping and overfitting.
    
    Usage:
        guard = TestSetGuard(hypothesis_id='HYP-2025-001')
        
        with guard.evaluate():
            metrics = model.evaluate(test_data)
    
    Raises:
        OverfittingError: If evaluation limit exceeded without explicit override
    """

    def __init__(self, hypothesis_id: str, max_evaluations: int = 3, db=None):
        """
        Initialize test set guard.

        Args:
            hypothesis_id: Hypothesis ID
            max_evaluations: Maximum allowed evaluations (default 3)
            db: Optional database connection (uses default if not provided)
        """
        self.hypothesis_id = hypothesis_id
        self.max_evaluations = max_evaluations
        self._db = db
        self._count = _load_evaluation_count(hypothesis_id, db=db)
        
        logger.debug(
            f"TestSetGuard for {hypothesis_id}: "
            f"{self._count}/{max_evaluations} evaluations used"
        )

    @property
    def evaluation_count(self) -> int:
        """Current evaluation count."""
        return self._count

    @property
    def remaining_evaluations(self) -> int:
        """Remaining evaluations allowed."""
        return max(0, self.max_evaluations - self._count)

    @contextmanager
    def evaluate(
        self,
        override: bool = False,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for test set evaluation.
        
        Args:
            override: Explicitly override limit (requires reason)
            reason: Reason for override (required if override=True)
            metadata: Optional metadata to log with evaluation
            
        Raises:
            OverfittingError: If limit exceeded without override
            ValueError: If override=True but reason not provided
        """
        if override and not reason:
            raise ValueError("Override requires a reason")

        if not override and self._count >= self.max_evaluations:
            raise OverfittingError(
                f"Test set evaluation limit exceeded for {self.hypothesis_id}. "
                f"Already evaluated {self._count} times (limit: {self.max_evaluations}). "
                f"Use override=True with justification if needed."
            )

        if override:
            logger.warning(
                f"Test set evaluation override for {self.hypothesis_id}: {reason}"
            )

        # Log the evaluation
        _log_evaluation(self.hypothesis_id, override, reason, metadata, db=self._db)
        self._count += 1

        try:
            yield
        except Exception:
            # Evaluation failed, but still counts toward limit
            logger.error(f"Test set evaluation failed for {self.hypothesis_id}")
            raise


def _load_trial_count(hypothesis_id: str, db=None) -> int:
    """Load existing trial count from database."""
    db = db or get_db()

    with db.connection() as conn:
        result = conn.execute(
            """
            SELECT COUNT(*)
            FROM hyperparameter_trials
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()

    return result[0] if result else 0


class HyperparameterTrialCounter:
    """
    Track and limit hyperparameter search trials per hypothesis.

    Prevents overfitting by limiting the number of HP combinations tried.
    All trials are logged to the database for auditability.

    Usage:
        counter = HyperparameterTrialCounter(hypothesis_id='HYP-2025-001', max_trials=50)

        if counter.can_try():
            counter.log_trial(
                model_type='ridge',
                hyperparameters={'alpha': 1.0},
                metric_name='val_r2',
                metric_value=0.85,
            )

        best = counter.get_best_trial()
    """

    def __init__(self, hypothesis_id: str, max_trials: int = 50, db=None):
        """
        Initialize hyperparameter trial counter.

        Args:
            hypothesis_id: Hypothesis ID
            max_trials: Maximum allowed trials (default 50)
            db: Optional database connection (uses default if not provided)
        """
        self.hypothesis_id = hypothesis_id
        self.max_trials = max_trials
        self._db = db
        self._count = _load_trial_count(hypothesis_id, db=db)

        logger.debug(
            f"HyperparameterTrialCounter for {hypothesis_id}: "
            f"{self._count}/{max_trials} trials used"
        )

    @property
    def trial_count(self) -> int:
        """Current trial count."""
        return self._count

    @property
    def remaining_trials(self) -> int:
        """Remaining trials allowed."""
        return max(0, self.max_trials - self._count)

    def can_try(self) -> bool:
        """Check if more trials are allowed."""
        return self._count < self.max_trials

    def log_trial(
        self,
        model_type: str,
        hyperparameters: dict,
        metric_name: str,
        metric_value: float,
        fold_index: int | None = None,
        notes: str | None = None,
    ) -> None:
        """
        Log a hyperparameter trial.

        Args:
            model_type: Type of model (e.g., 'ridge', 'lightgbm')
            hyperparameters: Dictionary of hyperparameters tried
            metric_name: Name of evaluation metric (e.g., 'val_r2')
            metric_value: Value of evaluation metric
            fold_index: Optional fold index for walk-forward validation
            notes: Optional notes about the trial

        Raises:
            OverfittingError: If trial limit exceeded
        """
        if not self.can_try():
            raise OverfittingError(
                f"Hyperparameter trial limit exceeded for {self.hypothesis_id}. "
                f"Already tried {self._count} combinations (limit: {self.max_trials})."
            )

        db = self._db or get_db()
        hp_json = json.dumps(hyperparameters)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO hyperparameter_trials
                (trial_id, hypothesis_id, model_type, hyperparameters,
                 metric_name, metric_value, fold_index, notes)
                VALUES (
                    (SELECT COALESCE(MAX(trial_id), 0) + 1 FROM hyperparameter_trials),
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    self.hypothesis_id,
                    model_type,
                    hp_json,
                    metric_name,
                    metric_value,
                    fold_index,
                    notes,
                ),
            )

        self._count += 1

        logger.debug(
            f"Logged HP trial {self._count}/{self.max_trials} for {self.hypothesis_id}: "
            f"{model_type} with {hyperparameters} -> {metric_name}={metric_value:.4f}"
        )

    def get_best_trial(self, metric_name: str | None = None) -> Optional[dict]:
        """
        Get the best trial by metric value.

        Args:
            metric_name: Optional filter by metric name

        Returns:
            Dictionary with trial details or None if no trials
        """
        db = self._db or get_db()

        query = """
            SELECT model_type, hyperparameters, metric_name, metric_value, fold_index
            FROM hyperparameter_trials
            WHERE hypothesis_id = ?
        """
        params: list[Any] = [self.hypothesis_id]

        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        query += " ORDER BY metric_value DESC LIMIT 1"

        with db.connection() as conn:
            result = conn.execute(query, params).fetchone()

        if not result:
            return None

        return {
            "model_type": result[0],
            "hyperparameters": json.loads(result[1]),
            "metric_name": result[2],
            "metric_value": float(result[3]),
            "fold_index": result[4],
        }


@dataclass
class LeakageCheckResult:
    """Result of target leakage check."""
    passed: bool
    warning: bool
    suspicious_features: list[str]
    correlations: dict[str, float]
    name_warnings: list[str]
    message: str


class TargetLeakageValidator:
    """
    Validator for detecting target leakage in features.

    Detects potential data leakage by:
    1. Checking for suspiciously high correlations with target
    2. Flagging features with future-looking names

    Usage:
        validator = TargetLeakageValidator(correlation_threshold=0.95)
        result = validator.check(features_df, target_series)
        if not result.passed:
            print(f"Leakage detected: {result.suspicious_features}")
    """

    # Patterns that suggest future information
    FUTURE_PATTERNS = [
        r'future',
        r'next',
        r'forward',
        r'lead',
        r'target',
        r't\+\d',  # t+1, t+5, etc.
    ]

    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize target leakage validator.

        Args:
            correlation_threshold: Correlation above this triggers failure (default 0.95)
        """
        if not 0 < correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")

        self.correlation_threshold = correlation_threshold
        self._future_regex = re.compile(
            '|'.join(self.FUTURE_PATTERNS),
            re.IGNORECASE
        )

    def check(self, features: pd.DataFrame, target: pd.Series) -> LeakageCheckResult:
        """
        Check features for target leakage.

        Args:
            features: DataFrame of feature values
            target: Series of target values

        Returns:
            LeakageCheckResult with pass/fail and suspicious features
        """
        suspicious_features: list[str] = []
        correlations: dict[str, float] = {}
        name_warnings: list[str] = []

        # Check correlations
        for col in features.columns:
            try:
                corr = features[col].corr(target)
                abs_corr = abs(corr) if pd.notna(corr) else 0.0
                correlations[col] = abs_corr

                if abs_corr >= self.correlation_threshold:
                    suspicious_features.append(col)
                    logger.warning(
                        f"High correlation detected: {col} has {corr:.3f} correlation with target"
                    )
            except Exception as e:
                logger.debug(f"Could not compute correlation for {col}: {e}")
                correlations[col] = 0.0

        # Check feature names for future-looking patterns
        for col in features.columns:
            if self._future_regex.search(col):
                name_warnings.append(col)
                logger.warning(
                    f"Suspicious feature name: '{col}' may indicate future information"
                )

        # Determine pass/fail
        passed = len(suspicious_features) == 0
        warning = len(name_warnings) > 0

        # Build message
        messages = []
        if suspicious_features:
            messages.append(
                f"High correlation with target: {', '.join(suspicious_features)}"
            )
        if name_warnings:
            messages.append(
                f"Suspicious feature names (may contain future info): {', '.join(name_warnings)}"
            )
        if not messages:
            messages.append("No target leakage detected")

        return LeakageCheckResult(
            passed=passed,
            warning=warning,
            suspicious_features=suspicious_features,
            correlations=correlations,
            name_warnings=name_warnings,
            message="; ".join(messages),
        )
