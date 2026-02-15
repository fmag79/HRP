"""
Optimization API for HRP.

Provides UI access to Optuna optimization infrastructure with
configuration preview, execution, and study management.
"""

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
import pandas as pd
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.ml.optimization import (
    OptimizationConfig,
    OptimizationResult,
    cross_validated_optimize,
    _get_sampler,
)
from hrp.ml.models import SUPPORTED_MODELS
from hrp.utils.config import get_config


# Default parameter spaces for each model type
DEFAULT_PARAM_SPACES = {
    "ridge": {
        "alpha": FloatDistribution(0.01, 100.0, log=True)
    },
    "lasso": {
        "alpha": FloatDistribution(0.001, 10.0, log=True)
    },
    "elastic_net": {
        "alpha": FloatDistribution(0.001, 10.0, log=True),
        "l1_ratio": FloatDistribution(0.0, 1.0),
    },
    "random_forest": {
        "n_estimators": IntDistribution(50, 500, step=50),
        "max_depth": IntDistribution(3, 15),
        "min_samples_leaf": IntDistribution(5, 50),
    },
    "lightgbm": {
        "n_estimators": IntDistribution(50, 500, step=50),
        "learning_rate": FloatDistribution(0.01, 0.3, log=True),
        "max_depth": IntDistribution(3, 12),
        "num_leaves": IntDistribution(15, 127),
    },
    "xgboost": {
        "n_estimators": IntDistribution(50, 500, step=50),
        "learning_rate": FloatDistribution(0.01, 0.3, log=True),
        "max_depth": IntDistribution(3, 12),
        "subsample": FloatDistribution(0.6, 1.0),
    },
    "mlp": {
        "hidden_layer_sizes": IntDistribution(50, 200, step=50),
        "alpha": FloatDistribution(0.0001, 0.1, log=True),
    },
}


@dataclass
class OptimizationPreview:
    """Preview of optimization configuration without running."""

    estimated_time_seconds: float
    estimated_cost_estimate: str  # "Low (~1m)", "Medium (~5m)", "High (~15m+)"
    parameter_space_summary: dict[str, str]  # Human-readable param ranges
    recommended_sampler: str  # Based on parameter space
    warnings: list[str]  # e.g., "High trial count may take >10m"


class OptimizationAPI:
    """
    API for optimization configuration and execution.

    Provides methods to:
    - Get default parameter spaces for model types
    - Get available strategies with ML models
    - Estimate execution time for configurations
    - Preview optimization impact without running
    - Run optimization with progress updates
    - List and manage Optuna studies
    """

    # Base time per fold (data fetch + model training)
    BASE_TIME_PER_FOLD = 10  # seconds

    # Model complexity multipliers
    MODEL_COMPLEXITY = {
        "ridge": 1.0,
        "lasso": 1.0,
        "elastic_net": 1.0,
        "random_forest": 2.0,
        "lightgbm": 1.5,
        "xgboost": 1.5,
        "mlp": 1.2,
    }

    # Sampler overhead multipliers
    SAMPLER_OVERHEAD = {
        "tpe": 1.0,
        "cmaes": 1.0,
        "random": 1.2,
        "grid": 2.0,
    }

    # Cost thresholds for user feedback
    COST_THRESHOLDS = {
        "low": 120,  # 2 minutes
        "medium": 300,  # 5 minutes
        "high": 900,  # 15 minutes
    }

    def __init__(self, api: PlatformAPI):
        """
        Initialize OptimizationAPI.

        Args:
            api: PlatformAPI instance for database access
        """
        self.api = api

    def get_default_param_space(self, model_type: str) -> dict[str, BaseDistribution]:
        """
        Get default Optuna parameter space for model type.

        Args:
            model_type: Model type (e.g., 'ridge', 'random_forest')

        Returns:
            Dict of parameter name to Optuna distribution

        Raises:
            ValueError: If model_type not supported
        """
        if model_type not in DEFAULT_PARAM_SPACES:
            available = ", ".join(sorted(DEFAULT_PARAM_SPACES.keys()))
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Available: {available}"
            )

        param_space = DEFAULT_PARAM_SPACES[model_type]
        logger.debug(f"Default param space for {model_type}: {list(param_space.keys())}")
        return param_space.copy()

    def get_available_strategies(self) -> list[str]:
        """
        Get list of strategies with ML models.

        Returns:
            List of strategy identifiers that support ML models

        Note:
            Currently returns all deployed strategies. In the future,
            this could be filtered to only ML-based strategies.
        """
        try:
            strategies = self.api.get_deployed_strategies()
            strategy_ids = [s["hypothesis_id"] for s in strategies]
            logger.debug(f"Found {len(strategy_ids)} deployed strategies")
            return strategy_ids
        except Exception as e:
            logger.warning(f"Failed to get deployed strategies: {e}")
            return []

    def estimate_execution_time(self, config: OptimizationConfig) -> float:
        """
        Estimate execution time in seconds.

        Heuristic:
        - Base time per fold: ~10s (data fetch + model training)
        - Multiply by n_trials × n_folds
        - Adjust for model complexity and sampler overhead

        Args:
            config: Optimization configuration

        Returns:
            Estimated execution time in seconds
        """
        model_complexity = self.MODEL_COMPLEXITY.get(config.model_type, 1.5)
        sampler_overhead = self.SAMPLER_OVERHEAD.get(config.sampler, 1.0)

        estimated_time = (
            config.n_trials
            * config.n_folds
            * self.BASE_TIME_PER_FOLD
            * model_complexity
            * sampler_overhead
        )

        logger.debug(
            f"Estimated time for {config.n_trials} trials, "
            f"{config.n_folds} folds: {estimated_time:.1f}s"
        )
        return estimated_time

    def preview_configuration(self, config: OptimizationConfig) -> OptimizationPreview:
        """
        Preview optimization impact without running.

        Calculates estimated time, cost category, recommended sampler,
        and generates warnings for potentially problematic configurations.

        Args:
            config: Optimization configuration

        Returns:
            OptimizationPreview with time estimate, cost category,
            parameter summary, recommended sampler, and warnings
        """
        # Estimate execution time
        estimated_time = self.estimate_execution_time(config)

        # Determine cost category
        if estimated_time < self.COST_THRESHOLDS["low"]:
            cost_estimate = f"Low (~{int(estimated_time / 60)}m)"
        elif estimated_time < self.COST_THRESHOLDS["medium"]:
            cost_estimate = f"Medium (~{int(estimated_time / 60)}m)"
        else:
            cost_estimate = f"High (~{int(estimated_time / 60)}m+)"

        # Generate parameter space summary
        param_summary = {}
        for name, dist in config.param_space.items():
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    param_summary[name] = f"{dist.low:.4f} - {dist.high:.4f} (log)"
                elif dist.step:
                    param_summary[name] = f"{dist.low:.4f} - {dist.high:.4f} (step {dist.step})"
                else:
                    param_summary[name] = f"{dist.low:.4f} - {dist.high:.4f}"
            elif isinstance(dist, IntDistribution):
                step_str = f", step {dist.step}" if dist.step else ""
                log_str = " (log)" if dist.log else ""
                param_summary[name] = f"{dist.low} - {dist.high}{step_str}{log_str}"
            elif isinstance(dist, CategoricalDistribution):
                param_summary[name] = f"{{{', '.join(map(str, dist.choices[:3]))}{'...' if len(dist.choices) > 3 else ''}}}"
            else:
                param_summary[name] = str(dist)

        # Determine recommended sampler
        # For categorical or discrete spaces, random/TPE works well
        # For continuous spaces, TPE or CMAES is preferred
        # Grid is only recommended when parameters have small, discrete spaces
        has_categorical = any(isinstance(d, CategoricalDistribution) for d in config.param_space.values())
        has_small_discrete = any(
            isinstance(d, IntDistribution) and (d.high - d.low) < 10
            for d in config.param_space.values()
        )

        if config.sampler == "grid":
            recommended = "grid"
        elif has_categorical or has_small_discrete:
            recommended = "tpe"  # TPE handles categorical well
        elif len(config.param_space) <= 3:
            recommended = "tpe"  # TPE efficient for low-dimensional
        else:
            recommended = "cmaes"  # CMAES good for high-dimensional continuous

        # Generate warnings
        warnings = []

        if estimated_time > self.COST_THRESHOLDS["high"]:
            warnings.append(
                f"High estimated time ({int(estimated_time / 60)}+ minutes). "
                f"Consider reducing trials ({config.n_trials}) or folds ({config.n_folds})."
            )

        if config.sampler == "grid" and config.n_trials > 50:
            warnings.append(
                f"Grid sampler with {config.n_trials} trials may be inefficient. "
                f"Consider TPE or CMAES for better convergence."
            )

        if len(config.features) > 30:
            warnings.append(
                f"Large feature set ({len(config.features)} features). "
                f"Consider enabling feature selection to reduce dimensionality."
            )

        if config.n_folds < 3:
            warnings.append(
                f"Low fold count ({config.n_folds}). "
                f"Consider at least 5 folds for more robust CV estimates."
            )

        if not warnings:
            warnings.append("Configuration looks good!")

        preview = OptimizationPreview(
            estimated_time_seconds=estimated_time,
            estimated_cost_estimate=cost_estimate,
            parameter_space_summary=param_summary,
            recommended_sampler=recommended,
            warnings=warnings,
        )

        logger.debug(f"Generated preview for {config.model_type}: {cost_estimate}")
        return preview

    def run_optimization(
        self,
        config: OptimizationConfig,
        symbols: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """
        Run optimization with progress updates.

        Executes cross-validated optimization using Optuna with
        optional progress callback for UI updates.

        Args:
            config: Optimization configuration
            symbols: List of symbols to use
            progress_callback: Optional callback(trial_idx, n_trials) for progress updates

        Returns:
            OptimizationResult with best parameters and all trial results

        Raises:
            ValueError: If no data found for symbols
        """
        logger.info(
            f"Running optimization: {config.model_type}, "
            f"{config.n_trials} trials, {config.n_folds} folds"
        )

        # Wrap progress callback to handle Optuna's internal callbacks
        # We'll need to modify cross_validated_optimize to support this
        # For now, run the optimization directly
        result = cross_validated_optimize(config, symbols, log_to_mlflow=False)

        # Report progress if callback provided (report completion)
        if progress_callback:
            progress_callback(result.n_trials_completed, config.n_trials)

        return result

    def list_studies(self, hypothesis_id: str | None = None) -> list[dict]:
        """
        List Optuna studies from storage.

        Reads Optuna studies from SQLite storage and returns
        metadata about each study.

        Args:
            hypothesis_id: Optional filter for specific hypothesis

        Returns:
            List of study metadata dicts with keys:
                - study_name: Study identifier
                - direction: 'maximize' or 'minimize'
                - n_trials: Number of trials completed
                - best_value: Best objective value
                - datetime_start: Start timestamp
        """
        studies = []

        # Get Optuna directory
        optuna_dir = get_config().data.optuna_dir
        optuna_dir.mkdir(parents=True, exist_ok=True)

        # Find all study databases
        if hypothesis_id:
            db_files = [optuna_dir / f"{hypothesis_id}.db"]
        else:
            db_files = list(optuna_dir.glob("*.db"))

        for db_file in db_files:
            if not db_file.is_file():
                continue

            try:
                # Connect to study storage
                storage = f"sqlite:///{db_file}"

                # Get study name from filename
                study_name = db_file.stem

                # Load study
                study = optuna.load_study(study_name=study_name, storage=storage)

                # Extract metadata
                study_meta = {
                    "study_name": study_name,
                    "direction": study.direction.name,
                    "n_trials": len(study.trials),
                    "best_value": study.best_value if study.trials else None,
                    "datetime_start": study.start_time,
                }

                studies.append(study_meta)

            except Exception as e:
                logger.warning(f"Failed to load study from {db_file}: {e}")

        # Sort by datetime descending
        studies.sort(key=lambda x: x.get("datetime_start") or datetime.min, reverse=True)

        logger.debug(f"Listed {len(studies)} studies from storage")
        return studies

    def get_study_details(self, study_name: str) -> dict:
        """
        Get detailed study information including parameter importance.

        Loads a specific Optuna study and computes parameter importance
        using Optuna's built-in importance evaluators.

        Args:
            study_name: Study identifier

        Returns:
            Dict with study details:
                - study_name: Study identifier
                - direction: Optimization direction
                - n_trials: Number of trials
                - best_params: Best hyperparameters
                - best_value: Best objective value
                - best_trial_number: Trial number of best trial
                - param_importance: Dict of param -> importance score
                - trials_summary: List of trial summaries

        Raises:
            ValueError: If study not found
        """
        # Get Optuna directory
        optuna_dir = get_config().data.optuna_dir
        db_file = optuna_dir / f"{study_name}.db"

        if not db_file.is_file():
            raise ValueError(f"Study not found: {study_name}")

        try:
            # Connect to study storage
            storage = f"sqlite:///{db_file}"

            # Load study
            study = optuna.load_study(study_name=study_name, storage=storage)

            # Get best trial
            best_trial = study.best_trial

            # Compute parameter importance
            param_importance = {}
            try:
                # Use default evaluator (Fanova)
                param_importance = optuna.importance.get_param_importances(study)
            except Exception as e:
                logger.warning(f"Failed to compute param importance: {e}")
                # Fall back to simple correlation-based importance
                if study.trials:
                    # Extract trial data
                    trials_df = study.trials_dataframe()
                    if not trials_df.empty and len(trials_df) > 1:
                        # Get param columns
                        param_cols = [col for col in trials_df.columns if col.startswith("params_")]
                        for col in param_cols:
                            param_name = col.replace("params_", "")
                            try:
                                # Compute correlation with objective
                                corr = trials_df[[col, "value"]].corr().iloc[0, 1]
                                if not np.isnan(corr):
                                    param_importance[param_name] = abs(corr)
                            except Exception:
                                pass

            # Generate trial summaries
            trials_summary = []
            for trial in study.trials:
                trial_summary = {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                }
                trials_summary.append(trial_summary)

            details = {
                "study_name": study_name,
                "direction": study.direction.name,
                "n_trials": len(study.trials),
                "best_params": best_trial.params,
                "best_value": best_trial.value,
                "best_trial_number": best_trial.number,
                "param_importance": param_importance,
                "trials_summary": trials_summary,
            }

            logger.debug(f"Retrieved details for study {study_name}")
            return details

        except Exception as e:
            logger.error(f"Failed to get study details for {study_name}: {e}")
            raise ValueError(f"Failed to load study {study_name}: {e}")

    def delete_study(self, study_name: str) -> bool:
        """
        Delete a study from storage.

        Removes the SQLite database file for the specified study.

        Args:
            study_name: Study identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get Optuna directory
            optuna_dir = get_config().data.optuna_dir
            db_file = optuna_dir / f"{study_name}.db"

            if not db_file.is_file():
                logger.warning(f"Study not found, cannot delete: {study_name}")
                return False

            # Delete the database file
            db_file.unlink()

            logger.info(f"Deleted study: {study_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete study {study_name}: {e}")
            return False
