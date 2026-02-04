"""
ML Scientist agent for automated model training and validation.

Trains and validates ML models for hypotheses in testing status
using walk-forward validation with stability scoring.
"""

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.agents.signal_scientist import SignalScientist
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType


@dataclass
class ModelExperimentResult:
    """Result of a single model experiment."""

    hypothesis_id: str
    model_type: str
    features: list[str]
    model_params: dict[str, Any]
    mean_ic: float
    ic_std: float
    stability_score: float
    is_stable: bool
    n_folds: int
    fold_results: list[dict]
    mlflow_run_id: str
    training_time_seconds: float


@dataclass
class MLScientistReport:
    """Complete ML Scientist run report."""

    run_date: date
    hypotheses_processed: int
    hypotheses_validated: int
    hypotheses_rejected: int
    total_trials: int
    total_training_time_seconds: float
    best_models: list[ModelExperimentResult]
    mlflow_experiment_id: str


class MLScientist(ResearchAgent):
    """
    Trains and validates ML models for hypotheses in testing status.

    The ML Scientist takes hypotheses created by the Signal Scientist
    (or manually) and systematically trains ML models using walk-forward
    validation. It identifies the best model/feature combinations and
    updates hypothesis status based on statistical rigor.

    Features:
    - Multi-model type testing (ridge, lasso, lightgbm)
    - Walk-forward validation with stability scoring
    - Feature combination search
    - Hyperparameter optimization with trial budget
    - Automatic hypothesis status updates
    - MLflow experiment logging
    - Email notifications with results
    """

    DEFAULT_JOB_ID = "ml_scientist_training"
    ACTOR = "agent:ml-scientist"

    # Default model types to test
    DEFAULT_MODEL_TYPES = ["ridge", "lasso", "lightgbm"]

    # Validation thresholds
    IC_THRESHOLD_VALIDATED = 0.03
    IC_THRESHOLD_PROMISING = 0.02
    STABILITY_THRESHOLD_VALIDATED = 1.0
    STABILITY_THRESHOLD_PROMISING = 1.5
    IC_SUSPICIOUS_THRESHOLD = 0.15  # Flag as potential data leakage

    # Trial limits
    MAX_TRIALS_PER_HYPOTHESIS = 50
    MAX_FEATURE_COMBINATIONS = 10
    MAX_FEATURES_PER_MODEL = 3

    # Hyperparameter grids
    HYPERPARAMETER_GRIDS = {
        "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
        "elasticnet": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        },
        "lightgbm": {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
        "xgboost": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
    }

    # Features with high correlation to returns_20d target (potential leakage)
    # These should NOT be used as predictors when target is returns_20d
    LEAKY_FEATURES_BY_TARGET = {
        "returns_20d": {
            "momentum_20d",      # corr=1.00 (identical calculation)
            "price_to_sma_50d",  # corr=0.87
            "price_to_sma_20d",  # corr=0.84
            "rsi_14d",           # corr=0.71
            "roc_10d",           # corr=0.68
            "cci_20d",           # corr=0.65
            "ema_crossover",     # corr=0.61
            "mfi_14d",           # corr=0.59
            "momentum_60d",      # corr=0.56
            "returns_60d",       # corr=0.56
            "price_to_sma_200d", # corr=0.54
        },
        "returns_60d": {
            "momentum_60d",      # identical
            "returns_60d",
        },
        "returns_252d": {
            "momentum_252d",     # identical
            "returns_252d",
        },
    }

    # Safe complementary features (all have <0.15 correlation with returns_20d)
    COMPLEMENTARY_FEATURES = {
        "volatility_60d": ["atr_14d", "volume_ratio", "adx_14d", "bb_width_20d"],
        "volatility_20d": ["atr_14d", "volume_ratio", "adx_14d"],
        "atr_14d": ["volatility_60d", "volume_ratio", "adx_14d", "bb_width_20d"],
        "volume_ratio": ["volatility_60d", "atr_14d", "adx_14d"],
        "bb_width_20d": ["volatility_60d", "atr_14d", "volume_ratio"],
        "adx_14d": ["volatility_60d", "atr_14d", "volume_ratio"],
        # Moderate risk features - only pair with safe features
        "macd_histogram": ["volatility_60d", "atr_14d", "volume_ratio"],
        "returns_1d": ["volatility_60d", "atr_14d", "volume_ratio"],
    }

    # All features (reuse from SignalScientist)
    ALL_FEATURES = SignalScientist.ALL_FEATURES

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        model_types: list[str] | None = None,
        target: str = "returns_20d",
        n_folds: int = 5,
        window_type: str = "expanding",
        start_date: date | None = None,
        end_date: date | None = None,
        symbols: list[str] | None = None,
        max_trials_per_hypothesis: int | None = None,
        skip_hyperparameter_search: bool = False,
        parallel_folds: bool = True,
        purge_days: int | None = None,
    ):
        """
        Initialize the ML Scientist agent.

        Args:
            hypothesis_ids: Specific hypotheses to process (None = all in 'testing')
            model_types: Models to test (default: ridge, lasso, lightgbm)
            target: Target variable name (default: returns_20d)
            n_folds: Number of walk-forward folds (default: 5)
            window_type: 'expanding' or 'rolling' (default: expanding)
            start_date: Start of training date range
            end_date: End of training date range
            symbols: Symbols to use (None = all universe)
            max_trials_per_hypothesis: Max trials per hypothesis (default: 50)
            skip_hyperparameter_search: Use default params only
            parallel_folds: Run folds in parallel (default: True)
            purge_days: Gap between train/test to prevent leakage (None = auto from target)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=["signal_scientist_scan"],
        )
        self.hypothesis_ids = hypothesis_ids
        self.model_types = model_types or self.DEFAULT_MODEL_TYPES
        self.target = target
        self.n_folds = n_folds
        self.window_type = window_type
        self.start_date = start_date or date(2015, 1, 1)
        self.end_date = end_date or date.today()
        self.symbols = symbols
        self.max_trials = max_trials_per_hypothesis or self.MAX_TRIALS_PER_HYPOTHESIS
        self.skip_hyperparameter_search = skip_hyperparameter_search
        self.parallel_folds = parallel_folds
        self.purge_days = purge_days

    def execute(self) -> dict[str, Any]:
        """
        Run ML experimentation on hypotheses in testing status.

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_process()
        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "message": "No hypotheses in testing status",
            }

        # 2. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        if not symbols:
            return {
                "status": "no_symbols",
                "message": "No symbols in universe",
            }

        # 3. Process each hypothesis
        all_results: list[ModelExperimentResult] = []
        validated_count = 0
        rejected_count = 0
        total_trials = 0

        for hypothesis in hypotheses:
            try:
                hyp_results = self._process_hypothesis(hypothesis, symbols)
                all_results.extend(hyp_results)

                # Update hypothesis status based on best result
                if hyp_results:
                    best = max(hyp_results, key=lambda r: self._calculate_model_score(r))
                    status = self._determine_status(best)
                    self._update_hypothesis(hypothesis, best, status)

                    if status == "validated":
                        validated_count += 1
                    elif status == "rejected":
                        rejected_count += 1

                    # Emit experiment_completed for downstream triggers
                    self._log_agent_event(
                        event_type=EventType.EXPERIMENT_COMPLETED,
                        hypothesis_id=hypothesis.get("hypothesis_id"),
                        experiment_id=best.mlflow_run_id if best else None,
                        details={
                            "status": status,
                            "trials": len(hyp_results),
                            "best_ic": best.mean_ic if best else None,
                        },
                    )

                    total_trials += len(hyp_results)
            except Exception as e:
                logger.error(f"Failed to process hypothesis {hypothesis.get('id')}: {e}")

        # 4. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_trials": total_trials,
                "duration_seconds": duration,
            },
        )

        # 5. Send email notification
        self._send_ml_email_notification(
            hypotheses, all_results, validated_count, rejected_count, duration
        )

        return {
            "run_date": date.today().isoformat(),
            "hypotheses_processed": len(hypotheses),
            "hypotheses_validated": validated_count,
            "hypotheses_rejected": rejected_count,
            "total_trials": total_trials,
            "duration_seconds": duration,
        }

    def _get_hypotheses_to_process(self) -> list[dict]:
        """Get hypotheses in testing status."""
        if self.hypothesis_ids:
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis(hid)
                if hyp:
                    hypotheses.append(hyp)
            return hypotheses
        return self.api.list_hypotheses(status="testing")

    def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the current universe."""
        try:
            return self.api.get_universe(date.today())
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}")
            # Fallback to symbols with features
            result = self.api.fetchall_readonly(
                """
                SELECT DISTINCT symbol
                FROM features
                WHERE date >= ?
                ORDER BY symbol
                """,
                (date.today() - timedelta(days=30),),
            )
            return [row[0] for row in result]

    def _process_hypothesis(
        self,
        hypothesis: dict,
        symbols: list[str],
    ) -> list[ModelExperimentResult]:
        """Process a single hypothesis through ML pipeline."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        results = []
        hypothesis_id = hypothesis.get("hypothesis_id", "unknown")

        # Initialize trial counter
        counter = HyperparameterTrialCounter(
            hypothesis_id=hypothesis_id,
            max_trials=self.max_trials,
        )

        # Extract base features from hypothesis
        base_features = self._extract_features_from_hypothesis(hypothesis)

        # Generate feature combinations
        feature_combos = self._generate_feature_combinations(base_features)

        # Test each model type
        for model_type in self.model_types:
            if counter.remaining_trials <= 0:
                logger.info(f"Trial budget exhausted for {hypothesis_id}")
                break

            # Test each feature combination
            for features in feature_combos:
                if counter.remaining_trials <= 0:
                    break

                # Get hyperparameter grid
                if self.skip_hyperparameter_search:
                    param_grid = [{}]  # Default params only
                else:
                    param_grid = self._get_param_grid(model_type)

                # Test each hyperparameter combination
                for params in param_grid:
                    if counter.remaining_trials <= 0:
                        break

                    try:
                        result = self._run_experiment(
                            hypothesis_id=hypothesis_id,
                            model_type=model_type,
                            features=features,
                            model_params=params,
                            symbols=symbols,
                        )

                        if result:
                            # Filter suspicious IC values (potential data leakage)
                            if result.mean_ic > self.IC_SUSPICIOUS_THRESHOLD:
                                logger.warning(
                                    f"Suspicious IC={result.mean_ic:.4f} for {hypothesis_id} "
                                    f"with features {result.features} — possible data leakage"
                                )
                                continue

                            results.append(result)
                            try:
                                counter.log_trial(
                                    model_type=model_type,
                                    hyperparameters={"features": features, **params},
                                    metric_name="mean_ic",
                                    metric_value=result.mean_ic,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to log trial: {e}")

                            # Early stopping if we find excellent result
                            if result.mean_ic > 0.05 and result.mean_ic <= self.IC_SUSPICIOUS_THRESHOLD and result.is_stable:
                                logger.info(
                                    f"Excellent result found for {hypothesis_id}, stopping early"
                                )
                                return results
                    except Exception as e:
                        logger.warning(
                            f"Experiment failed: {model_type}/{features}: {e}"
                        )

        return results

    def _run_experiment(
        self,
        hypothesis_id: str,
        model_type: str,
        features: list[str],
        model_params: dict,
        symbols: list[str],
    ) -> ModelExperimentResult | None:
        """Run a single walk-forward validation experiment."""
        from hrp.ml import WalkForwardConfig, walk_forward_validate

        try:
            start_time = time.time()

            # Derive purge days from target horizon to prevent temporal leakage
            if self.purge_days is not None:
                purge_days = self.purge_days
            else:
                digits = "".join(filter(str.isdigit, self.target))
                purge_days = int(digits) if digits else 20

            config = WalkForwardConfig(
                model_type=model_type,
                target=self.target,
                features=features,
                start_date=self.start_date,
                end_date=self.end_date,
                n_folds=self.n_folds,
                window_type=self.window_type,
                n_jobs=-1 if self.parallel_folds else 1,
                hyperparameters=model_params,
                purge_days=purge_days,
                embargo_days=5,
                tags={"hypothesis_id": hypothesis_id},
            )

            result = walk_forward_validate(
                config=config,
                symbols=symbols,
                log_to_mlflow=True,
            )

            training_time = time.time() - start_time

            # Extract fold IC values
            fold_results = []
            for fold in result.fold_results:
                fold_results.append(fold.metrics)

            mlflow_run_id = result.mlflow_run_id or ""

            # Link experiment to hypothesis so validation guards work
            if mlflow_run_id and hypothesis_id:
                self.api.link_experiment(hypothesis_id, mlflow_run_id)

            return ModelExperimentResult(
                hypothesis_id=hypothesis_id,
                model_type=model_type,
                features=features,
                model_params=model_params,
                mean_ic=result.mean_ic,
                ic_std=result.aggregate_metrics.get("std_ic", 0.0),
                stability_score=result.stability_score,
                is_stable=result.is_stable,
                n_folds=len(result.fold_results),
                fold_results=fold_results,
                mlflow_run_id=mlflow_run_id,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Experiment failed: {model_type}/{features}: {e}")
            return None

    def _extract_features_from_hypothesis(self, hypothesis: dict) -> list[str]:
        """Extract feature names from hypothesis thesis/metadata."""
        # Check metadata first
        metadata = hypothesis.get("metadata", {})
        if isinstance(metadata, dict) and "features" in metadata:
            return metadata["features"]

        # Parse from thesis text
        thesis = hypothesis.get("thesis", "")
        features = []
        for feature in self.ALL_FEATURES:
            if feature in thesis.lower():
                features.append(feature)

        return features if features else ["volatility_60d"]  # Safe default fallback

    def _filter_leaky_features(self, features: list[str]) -> list[str]:
        """
        Remove features that have high correlation with the target.

        This prevents data leakage where features like momentum_20d
        are nearly identical to the returns_20d target.
        """
        leaky = self.LEAKY_FEATURES_BY_TARGET.get(self.target, set())
        safe_features = [f for f in features if f not in leaky]

        if len(safe_features) < len(features):
            removed = set(features) - set(safe_features)
            logger.warning(
                f"Filtered {len(removed)} leaky features for target={self.target}: {removed}"
            )

        return safe_features if safe_features else ["volatility_60d"]  # Safe fallback

    def _generate_feature_combinations(self, base_features: list[str]) -> list[list[str]]:
        """Generate feature combinations to test, filtering leaky features."""
        # First filter any leaky base features
        safe_base = self._filter_leaky_features(base_features)
        combinations = [safe_base]  # Start with safe base

        # Add complementary features (already defined as safe in COMPLEMENTARY_FEATURES)
        for base in safe_base:
            complements = self.COMPLEMENTARY_FEATURES.get(base, [])
            for comp in complements[:2]:  # Limit to top 2 complements
                # Skip if complement is already in base or is leaky
                if comp in safe_base:
                    continue
                if comp in self.LEAKY_FEATURES_BY_TARGET.get(self.target, set()):
                    continue
                combo = safe_base + [comp]
                if len(combo) <= self.MAX_FEATURES_PER_MODEL:
                    combinations.append(combo)

        # Deduplicate and limit
        seen = set()
        unique = []
        for combo in combinations:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                unique.append(combo)

        return unique[: self.MAX_FEATURE_COMBINATIONS]

    def _get_param_grid(self, model_type: str) -> list[dict]:
        """Get hyperparameter combinations for model type."""
        from sklearn.model_selection import ParameterGrid

        grid = self.HYPERPARAMETER_GRIDS.get(model_type, {})
        if not grid:
            return [{}]

        return list(ParameterGrid(grid))[:10]  # Limit combinations

    def _calculate_model_score(self, result: ModelExperimentResult) -> float:
        """Calculate composite score for model ranking."""
        ic_score = result.mean_ic
        stability_penalty = 1 / max(result.stability_score, 0.1)

        # Bonus if all folds have positive IC
        all_positive = all(f.get("ic", 0) > 0 for f in result.fold_results)
        consistency_bonus = 1.2 if all_positive else 1.0

        return ic_score * stability_penalty * consistency_bonus

    def _determine_status(self, result: ModelExperimentResult) -> str:
        """Determine hypothesis status based on best model result."""
        if (
            result.mean_ic >= self.IC_THRESHOLD_VALIDATED
            and result.stability_score <= self.STABILITY_THRESHOLD_VALIDATED
            and result.is_stable
        ):
            return "validated"
        elif (
            result.mean_ic >= self.IC_THRESHOLD_PROMISING
            and result.stability_score <= self.STABILITY_THRESHOLD_PROMISING
        ):
            return "testing"  # Keep in testing for further work
        else:
            return "rejected"

    def _update_hypothesis(
        self,
        hypothesis: dict,
        best_result: ModelExperimentResult,
        status: str,
    ) -> None:
        """Update hypothesis with ML results."""
        hypothesis_id = hypothesis.get("hypothesis_id", "unknown")

        # Build outcome string with ML results
        outcome = (
            f"ML Scientist: {best_result.model_type} model with "
            f"features {best_result.features}, IC={best_result.mean_ic:.4f}, "
            f"stability={best_result.stability_score:.2f}"
        )

        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=status,
                outcome=outcome,
                actor=self.ACTOR,
                metadata={
                    "ml_scientist_results": {
                        "model_type": best_result.model_type,
                        "features": best_result.features,
                        "hyperparameters": best_result.model_params,
                        "target": self.target,
                        "mean_ic": best_result.mean_ic,
                        "ic_std": best_result.ic_std,
                        "stability_score": best_result.stability_score,
                        "n_folds": best_result.n_folds,
                        "mlflow_run_id": best_result.mlflow_run_id,
                    }
                },
            )
            logger.info(f"Updated hypothesis {hypothesis_id} to status={status}")
        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _send_ml_email_notification(
        self,
        hypotheses: list[dict],
        results: list[ModelExperimentResult],
        validated_count: int,
        rejected_count: int,
        duration: float,
    ) -> None:
        """Send email notification with ML results."""
        try:
            notifier = EmailNotifier()

            # Group best results by hypothesis
            best_by_hypothesis = {}
            for result in results:
                hid = result.hypothesis_id
                if hid not in best_by_hypothesis:
                    best_by_hypothesis[hid] = result
                elif self._calculate_model_score(result) > self._calculate_model_score(
                    best_by_hypothesis[hid]
                ):
                    best_by_hypothesis[hid] = result

            summary_data = {
                "run_date": date.today().isoformat(),
                "duration_seconds": f"{duration:.1f}",
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_experiments": len(results),
            }

            # Add top results
            sorted_results = sorted(
                best_by_hypothesis.values(),
                key=lambda r: self._calculate_model_score(r),
                reverse=True,
            )
            for i, result in enumerate(sorted_results[:5]):
                summary_data[f"top_{i+1}_model"] = (
                    f"{result.hypothesis_id}: {result.model_type} "
                    f"IC={result.mean_ic:.4f}, stability={result.stability_score:.2f}"
                )

            subject = (
                f"[HRP] ML Scientist Complete - "
                f"{validated_count} validated, {rejected_count} rejected"
            )

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
