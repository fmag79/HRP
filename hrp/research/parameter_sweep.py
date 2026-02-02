"""
Parallel parameter sweep with constraints.

Provides efficient parameter exploration with constraint validation
and Sharpe decay analysis for identifying robust parameter regions.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class SweepConstraint:
    """
    Constraint on parameter combinations.

    Attributes:
        constraint_type: Type of constraint
            - "sum_equals": Parameters must sum to value
            - "max_total": Parameters must sum to at most value
            - "min_total": Parameters must sum to at least value
            - "ratio_bound": Ratio between params within bounds
            - "difference_min": Minimum difference between two params
            - "difference_max": Maximum difference between two params
            - "exclusion": Mutually exclusive parameters
            - "range_bound": Single param must be within [value, upper_bound]
            - "product_max": Product of params must be <= value
            - "product_min": Product of params must be >= value
            - "same_sign": All params must have same sign (positive/negative)
            - "step_multiple": Param must be a multiple of value
            - "monotonic_increasing": params[0] < params[1] < params[2] ...
            - "at_least_n_nonzero": At least N params must be non-zero
        params: List of parameter names involved
        value: Constraint value
        upper_bound: Upper bound for ratio_bound/range_bound (optional)
    """

    constraint_type: str
    params: list[str]
    value: float
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        """Validate constraint configuration."""
        valid_types = (
            "sum_equals",
            "max_total",
            "min_total",
            "ratio_bound",
            "difference_min",
            "difference_max",
            "exclusion",
            "range_bound",
            "product_max",
            "product_min",
            "same_sign",
            "step_multiple",
            "monotonic_increasing",
            "at_least_n_nonzero",
        )
        if self.constraint_type not in valid_types:
            raise ValueError(
                f"Invalid constraint_type: '{self.constraint_type}'. "
                f"Valid types: {', '.join(valid_types)}"
            )
        # Single-param constraints
        single_param_types = ("range_bound", "step_multiple")
        # Multi-param constraints requiring at least 2
        multi_param_types = (
            "sum_equals", "ratio_bound", "difference_min", "difference_max",
            "exclusion", "product_max", "product_min", "same_sign",
            "monotonic_increasing", "at_least_n_nonzero",
        )
        if self.constraint_type in multi_param_types and len(self.params) < 2:
            raise ValueError(
                f"Constraint type '{self.constraint_type}' requires at least 2 params"
            )
        if self.constraint_type in single_param_types and len(self.params) < 1:
            raise ValueError(
                f"Constraint type '{self.constraint_type}' requires at least 1 param"
            )


@dataclass
class SweepConfig:
    """
    Configuration for parallel parameter sweep.

    Attributes:
        strategy_type: Type of strategy ("multifactor", "ml_predicted", "momentum")
        param_ranges: Dict mapping param name to list of values
        constraints: List of SweepConstraint objects
        symbols: List of symbols to backtest on
        start_date: Start date for backtest
        end_date: End date for backtest
        n_folds: Number of cross-validation folds (default 5)
        n_jobs: Number of parallel jobs (-1 = use all cores)
        scoring: Scoring metric ("sharpe_ratio", "ic", "total_return")
        min_samples: Minimum samples per fold
        aggregation: How to aggregate across folds ("median" or "mean")
    """

    strategy_type: str
    param_ranges: dict[str, list[Any]]
    constraints: list[SweepConstraint]
    symbols: list[str]
    start_date: date
    end_date: date
    n_folds: int = 5
    n_jobs: int = -1
    scoring: str = "sharpe_ratio"
    min_samples: int = 100
    aggregation: str = "median"

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_strategies = ("multifactor", "ml_predicted", "momentum", "mean_reversion")
        if self.strategy_type not in valid_strategies:
            raise ValueError(
                f"Invalid strategy_type: '{self.strategy_type}'. "
                f"Valid types: {', '.join(valid_strategies)}"
            )
        if not self.param_ranges:
            raise ValueError("param_ranges cannot be empty")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.aggregation not in ("median", "mean"):
            raise ValueError(
                f"aggregation must be 'median' or 'mean', got '{self.aggregation}'"
            )

        logger.debug(
            f"SweepConfig created: {self.strategy_type}, "
            f"{len(self.param_ranges)} params, {len(self.constraints)} constraints"
        )


@dataclass
class SweepResult:
    """
    Result of parameter sweep with train/test analysis.

    Attributes:
        results_df: DataFrame with all param combos and per-fold metrics
        best_params: Best parameter combination
        best_metrics: Metrics for best params
        train_sharpe_matrix: Params as index, folds as columns (train Sharpe)
        test_sharpe_matrix: Params as index, folds as columns (test Sharpe)
        sharpe_diff_matrix: test - train Sharpe per fold
        sharpe_diff_median: Aggregated Sharpe diff across folds
        constraint_violations: Number of param combos that violated constraints
        execution_time_seconds: Total execution time
        generalization_score: Percentage of params where test >= train
    """

    results_df: pd.DataFrame
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    train_sharpe_matrix: pd.DataFrame
    test_sharpe_matrix: pd.DataFrame
    sharpe_diff_matrix: pd.DataFrame
    sharpe_diff_median: pd.Series
    constraint_violations: int
    execution_time_seconds: float
    generalization_score: float


def validate_constraints(
    params: dict[str, Any],
    constraints: list[SweepConstraint],
) -> bool:
    """
    Check if parameter combination satisfies all constraints.

    Args:
        params: Dict of parameter values
        constraints: List of constraints to check

    Returns:
        True if all constraints satisfied, False otherwise
    """
    for constraint in constraints:
        param_values = [params.get(p, 0) for p in constraint.params]

        if constraint.constraint_type == "sum_equals":
            if not np.isclose(sum(param_values), constraint.value):
                return False

        elif constraint.constraint_type == "max_total":
            if sum(param_values) > constraint.value:
                return False

        elif constraint.constraint_type == "min_total":
            # Parameters must sum to at least value
            if sum(param_values) < constraint.value:
                return False

        elif constraint.constraint_type == "difference_min":
            # For difference_min, params[0] - params[1] >= value
            if len(param_values) >= 2:
                if param_values[0] - param_values[1] < constraint.value:
                    return False

        elif constraint.constraint_type == "difference_max":
            # For difference_max, params[0] - params[1] <= value
            if len(param_values) >= 2:
                if param_values[0] - param_values[1] > constraint.value:
                    return False

        elif constraint.constraint_type == "ratio_bound":
            # params[0] / params[1] must be within [value, upper_bound]
            if len(param_values) >= 2 and param_values[1] != 0:
                ratio = param_values[0] / param_values[1]
                if ratio < constraint.value:
                    return False
                if constraint.upper_bound is not None and ratio > constraint.upper_bound:
                    return False

        elif constraint.constraint_type == "exclusion":
            # At most one param can be non-zero
            non_zero_count = sum(1 for v in param_values if v != 0)
            if non_zero_count > 1:
                return False

        elif constraint.constraint_type == "range_bound":
            # Single param must be within [value, upper_bound]
            if len(param_values) >= 1:
                val = param_values[0]
                if val < constraint.value:
                    return False
                if constraint.upper_bound is not None and val > constraint.upper_bound:
                    return False

        elif constraint.constraint_type == "product_max":
            # Product of params must be <= value
            product = 1
            for v in param_values:
                product *= v
            if product > constraint.value:
                return False

        elif constraint.constraint_type == "product_min":
            # Product of params must be >= value
            product = 1
            for v in param_values:
                product *= v
            if product < constraint.value:
                return False

        elif constraint.constraint_type == "same_sign":
            # All params must have the same sign (all positive or all negative)
            if len(param_values) >= 2:
                signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in param_values]
                non_zero_signs = [s for s in signs if s != 0]
                if non_zero_signs and not all(s == non_zero_signs[0] for s in non_zero_signs):
                    return False

        elif constraint.constraint_type == "step_multiple":
            # Param must be a multiple of value (e.g., only multiples of 5)
            if len(param_values) >= 1 and constraint.value != 0:
                for val in param_values:
                    if not np.isclose(val % constraint.value, 0) and not np.isclose(val % constraint.value, constraint.value):
                        return False

        elif constraint.constraint_type == "monotonic_increasing":
            # params[0] < params[1] < params[2] ...
            for i in range(len(param_values) - 1):
                if param_values[i] >= param_values[i + 1]:
                    return False

        elif constraint.constraint_type == "at_least_n_nonzero":
            # At least N params must be non-zero (value = N)
            non_zero_count = sum(1 for v in param_values if v != 0)
            if non_zero_count < int(constraint.value):
                return False

    return True


def _generate_valid_combinations(
    param_ranges: dict[str, list[Any]],
    constraints: list[SweepConstraint],
) -> tuple[list[dict[str, Any]], int]:
    """
    Generate all valid parameter combinations.

    Returns:
        Tuple of (valid_combinations, violation_count)
    """
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    valid_combinations = []
    violations = 0

    for combo in all_combinations:
        if validate_constraints(combo, constraints):
            valid_combinations.append(combo)
        else:
            violations += 1

    logger.debug(
        f"Generated {len(valid_combinations)} valid combinations, "
        f"{violations} violated constraints"
    )

    return valid_combinations, violations


def compute_sharpe_diff_analysis(
    results_df: pd.DataFrame,
    param_columns: list[str],
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Compute Sharpe ratio diff analysis across parameter combinations.

    Args:
        results_df: DataFrame with columns for params and fold metrics
        param_columns: List of parameter column names
        aggregation: "median" or "mean" for aggregation across folds

    Returns:
        Tuple of:
        - sharpe_diff_matrix: Full matrix of test-train diffs per fold
        - sharpe_diff_agg: Aggregated (median/mean) diff per param combo
        - generalization_score: % of combos where agg_diff >= 0
    """
    # Find fold columns
    train_cols = [c for c in results_df.columns if c.startswith("train_sharpe_fold_")]
    test_cols = [c for c in results_df.columns if c.startswith("test_sharpe_fold_")]

    if not train_cols or not test_cols:
        # Return empty results if no fold data
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        return empty_df, empty_series, 0.0

    # Compute diff per fold
    diff_data = {}
    for train_col, test_col in zip(sorted(train_cols), sorted(test_cols)):
        fold_idx = train_col.replace("train_sharpe_fold_", "")
        diff_data[f"diff_fold_{fold_idx}"] = (
            results_df[test_col] - results_df[train_col]
        )

    sharpe_diff_matrix = pd.DataFrame(diff_data, index=results_df.index)

    # Aggregate across folds
    if aggregation == "median":
        sharpe_diff_agg = sharpe_diff_matrix.median(axis=1)
    else:
        sharpe_diff_agg = sharpe_diff_matrix.mean(axis=1)

    # Compute generalization score
    n_generalizing = (sharpe_diff_agg >= 0).sum()
    generalization_score = n_generalizing / len(sharpe_diff_agg) if len(sharpe_diff_agg) > 0 else 0.0

    return sharpe_diff_matrix, sharpe_diff_agg, generalization_score


def _evaluate_single_combination(
    params: dict[str, Any],
    config: SweepConfig,
    fold_idx: int,
) -> dict[str, float]:
    """
    Evaluate a single parameter combination on one fold.

    NOT YET IMPLEMENTED — returns mock data. Do not use for research decisions.

    TODO: Wire to real backtest engine:
    1. Generate signals using the strategy with given params
    2. Run backtest on the fold's train/test split
    3. Return train and test metrics
    """
    raise NotImplementedError(
        "Parameter sweep evaluation is not yet implemented. "
        "_evaluate_single_combination() currently has no connection to the "
        "backtest engine. Do not use parallel_parameter_sweep() for research "
        "decisions until this is wired up."
    )


def parallel_parameter_sweep(
    config: SweepConfig,
    hypothesis_id: str | None = None,
) -> SweepResult:
    """
    Run parallel parameter sweep with constraint validation.

    Args:
        config: Sweep configuration
        hypothesis_id: Optional hypothesis ID for tracking

    Returns:
        SweepResult with comprehensive analysis
    """
    logger.info(
        f"Starting parameter sweep: {config.strategy_type}, "
        f"{len(config.param_ranges)} params, {config.n_folds} folds"
    )

    start_time = time.perf_counter()

    # Generate valid parameter combinations
    valid_combinations, violations = _generate_valid_combinations(
        config.param_ranges, config.constraints
    )

    if not valid_combinations:
        raise ValueError("No valid parameter combinations after applying constraints")

    logger.info(f"Evaluating {len(valid_combinations)} valid parameter combinations")

    # Evaluate all combinations across all folds
    results_data = []
    param_columns = list(config.param_ranges.keys())

    for combo_idx, params in enumerate(valid_combinations):
        row: dict[str, Any] = {"combo_idx": combo_idx}
        row.update(params)

        fold_train_sharpes = []
        fold_test_sharpes = []

        for fold_idx in range(config.n_folds):
            metrics = _evaluate_single_combination(params, config, fold_idx)
            row[f"train_sharpe_fold_{fold_idx}"] = metrics["train_sharpe"]
            row[f"test_sharpe_fold_{fold_idx}"] = metrics["test_sharpe"]
            fold_train_sharpes.append(metrics["train_sharpe"])
            fold_test_sharpes.append(metrics["test_sharpe"])

        # Aggregate metrics
        if config.aggregation == "median":
            row["train_sharpe_agg"] = np.median(fold_train_sharpes)
            row["test_sharpe_agg"] = np.median(fold_test_sharpes)
        else:
            row["train_sharpe_agg"] = np.mean(fold_train_sharpes)
            row["test_sharpe_agg"] = np.mean(fold_test_sharpes)

        results_data.append(row)

    results_df = pd.DataFrame(results_data)

    # Compute Sharpe diff analysis
    sharpe_diff_matrix, sharpe_diff_agg, generalization_score = compute_sharpe_diff_analysis(
        results_df, param_columns, config.aggregation
    )

    # Add aggregated diff to results
    results_df["sharpe_diff_agg"] = sharpe_diff_agg.values

    # Find best params (highest test Sharpe with positive generalization)
    # Prefer params that generalize well
    best_idx = results_df["test_sharpe_agg"].idxmax()
    best_row = results_df.loc[best_idx]
    best_params = {col: best_row[col] for col in param_columns}
    best_metrics = {
        "train_sharpe": best_row["train_sharpe_agg"],
        "test_sharpe": best_row["test_sharpe_agg"],
        "sharpe_diff": best_row.get("sharpe_diff_agg", 0),
    }

    # Build train/test sharpe matrices
    train_cols = [c for c in results_df.columns if c.startswith("train_sharpe_fold_")]
    test_cols = [c for c in results_df.columns if c.startswith("test_sharpe_fold_")]

    train_sharpe_matrix = results_df[train_cols].copy()
    test_sharpe_matrix = results_df[test_cols].copy()

    execution_time = time.perf_counter() - start_time

    result = SweepResult(
        results_df=results_df,
        best_params=best_params,
        best_metrics=best_metrics,
        train_sharpe_matrix=train_sharpe_matrix,
        test_sharpe_matrix=test_sharpe_matrix,
        sharpe_diff_matrix=sharpe_diff_matrix,
        sharpe_diff_median=sharpe_diff_agg,
        constraint_violations=violations,
        execution_time_seconds=execution_time,
        generalization_score=generalization_score,
    )

    logger.info(
        f"Sweep complete: {len(valid_combinations)} combos evaluated, "
        f"best test Sharpe={best_metrics['test_sharpe']:.3f}, "
        f"generalization={generalization_score:.1%}"
    )

    return result
