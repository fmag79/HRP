"""Tests for parallel parameter sweep."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hrp.research.parameter_sweep import (
    SweepConstraint,
    SweepConfig,
    SweepResult,
    validate_constraints,
    compute_sharpe_diff_analysis,
    parallel_parameter_sweep,
    _generate_valid_combinations,
)


class TestSweepConstraint:
    """Tests for SweepConstraint dataclass."""

    def test_constraint_creation(self):
        """Test creating constraint."""
        constraint = SweepConstraint(
            constraint_type="sum_equals",
            params=["weight_a", "weight_b"],
            value=1.0,
        )
        assert constraint.constraint_type == "sum_equals"
        assert constraint.params == ["weight_a", "weight_b"]
        assert constraint.value == 1.0

    def test_constraint_invalid_type(self):
        """Test constraint rejects invalid type."""
        with pytest.raises(ValueError, match="Invalid constraint_type"):
            SweepConstraint(
                constraint_type="invalid_type",
                params=["a", "b"],
                value=1.0,
            )

    def test_constraint_requires_params(self):
        """Test constraint requires at least 2 params for most types."""
        with pytest.raises(ValueError, match="requires at least 2 params"):
            SweepConstraint(
                constraint_type="sum_equals",
                params=["a"],
                value=1.0,
            )


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = SweepConfig(
            strategy_type="multifactor",
            param_ranges={"alpha": [0.1, 0.5, 1.0]},
            constraints=[],
            symbols=["AAPL", "MSFT"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert config.strategy_type == "multifactor"
        assert config.n_folds == 5
        assert config.n_jobs == -1
        assert config.aggregation == "median"

    def test_config_invalid_strategy_type(self):
        """Test config rejects invalid strategy type."""
        with pytest.raises(ValueError, match="Invalid strategy_type"):
            SweepConfig(
                strategy_type="invalid",
                param_ranges={"alpha": [1.0]},
                constraints=[],
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_empty_param_ranges(self):
        """Test config rejects empty param_ranges."""
        with pytest.raises(ValueError, match="param_ranges cannot be empty"):
            SweepConfig(
                strategy_type="multifactor",
                param_ranges={},
                constraints=[],
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
            )

    def test_config_invalid_n_folds(self):
        """Test config rejects n_folds < 2."""
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            SweepConfig(
                strategy_type="multifactor",
                param_ranges={"alpha": [1.0]},
                constraints=[],
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                n_folds=1,
            )

    def test_config_invalid_aggregation(self):
        """Test config rejects invalid aggregation."""
        with pytest.raises(ValueError, match="aggregation must be"):
            SweepConfig(
                strategy_type="multifactor",
                param_ranges={"alpha": [1.0]},
                constraints=[],
                symbols=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                aggregation="invalid",
            )


class TestValidateConstraints:
    """Tests for validate_constraints function."""

    def test_sum_equals_constraint(self):
        """Test sum_equals constraint."""
        constraint = SweepConstraint("sum_equals", ["a", "b"], 1.0)

        # Valid
        assert validate_constraints({"a": 0.6, "b": 0.4}, [constraint]) is True
        assert validate_constraints({"a": 0.5, "b": 0.5}, [constraint]) is True

        # Invalid
        assert validate_constraints({"a": 0.6, "b": 0.6}, [constraint]) is False
        assert validate_constraints({"a": 0.3, "b": 0.3}, [constraint]) is False

    def test_max_total_constraint(self):
        """Test max_total constraint."""
        constraint = SweepConstraint("max_total", ["a", "b", "c"], 1.0)

        # Valid
        assert validate_constraints({"a": 0.3, "b": 0.3, "c": 0.3}, [constraint]) is True
        assert validate_constraints({"a": 0.5, "b": 0.5, "c": 0.0}, [constraint]) is True

        # Invalid
        assert validate_constraints({"a": 0.5, "b": 0.5, "c": 0.5}, [constraint]) is False

    def test_difference_min_constraint(self):
        """Test difference_min constraint (slow - fast >= value)."""
        constraint = SweepConstraint("difference_min", ["slow_period", "fast_period"], 5)

        # Valid: slow - fast >= 5
        assert validate_constraints({"slow_period": 20, "fast_period": 10}, [constraint]) is True
        assert validate_constraints({"slow_period": 15, "fast_period": 10}, [constraint]) is True

        # Invalid: slow - fast < 5
        assert validate_constraints({"slow_period": 12, "fast_period": 10}, [constraint]) is False
        assert validate_constraints({"slow_period": 10, "fast_period": 10}, [constraint]) is False

    def test_ratio_bound_constraint(self):
        """Test ratio_bound constraint."""
        constraint = SweepConstraint("ratio_bound", ["a", "b"], 0.5, upper_bound=2.0)

        # Valid: 0.5 <= a/b <= 2.0
        assert validate_constraints({"a": 1.0, "b": 1.0}, [constraint]) is True
        assert validate_constraints({"a": 1.0, "b": 2.0}, [constraint]) is True

        # Invalid: ratio outside bounds
        assert validate_constraints({"a": 0.3, "b": 1.0}, [constraint]) is False
        assert validate_constraints({"a": 3.0, "b": 1.0}, [constraint]) is False

    def test_exclusion_constraint(self):
        """Test exclusion constraint (at most one non-zero)."""
        constraint = SweepConstraint("exclusion", ["a", "b", "c"], 0)

        # Valid: at most one non-zero
        assert validate_constraints({"a": 1.0, "b": 0, "c": 0}, [constraint]) is True
        assert validate_constraints({"a": 0, "b": 2.0, "c": 0}, [constraint]) is True
        assert validate_constraints({"a": 0, "b": 0, "c": 0}, [constraint]) is True

        # Invalid: more than one non-zero
        assert validate_constraints({"a": 1.0, "b": 1.0, "c": 0}, [constraint]) is False

    def test_multiple_constraints(self):
        """Test multiple constraints together."""
        constraints = [
            SweepConstraint("sum_equals", ["a", "b"], 1.0),
            SweepConstraint("difference_min", ["b", "a"], 0.1),
        ]

        # Valid: a + b = 1.0 AND b - a >= 0.1
        assert validate_constraints({"a": 0.4, "b": 0.6}, constraints) is True

        # Invalid: violates difference_min
        assert validate_constraints({"a": 0.5, "b": 0.5}, constraints) is False

    def test_min_total_constraint(self):
        """Test min_total constraint."""
        constraint = SweepConstraint("min_total", ["a", "b", "c"], 0.5)

        # Valid: sum >= 0.5
        assert validate_constraints({"a": 0.3, "b": 0.3, "c": 0.0}, [constraint]) is True
        assert validate_constraints({"a": 0.5, "b": 0.5, "c": 0.5}, [constraint]) is True

        # Invalid: sum < 0.5
        assert validate_constraints({"a": 0.1, "b": 0.1, "c": 0.1}, [constraint]) is False

    def test_difference_max_constraint(self):
        """Test difference_max constraint (a - b <= value)."""
        constraint = SweepConstraint("difference_max", ["a", "b"], 10)

        # Valid: a - b <= 10
        assert validate_constraints({"a": 20, "b": 15}, [constraint]) is True
        assert validate_constraints({"a": 20, "b": 10}, [constraint]) is True

        # Invalid: a - b > 10
        assert validate_constraints({"a": 30, "b": 10}, [constraint]) is False

    def test_range_bound_constraint(self):
        """Test range_bound constraint (single param in range)."""
        constraint = SweepConstraint("range_bound", ["alpha"], 0.1, upper_bound=10.0)

        # Valid: 0.1 <= alpha <= 10.0
        assert validate_constraints({"alpha": 1.0}, [constraint]) is True
        assert validate_constraints({"alpha": 0.1}, [constraint]) is True
        assert validate_constraints({"alpha": 10.0}, [constraint]) is True

        # Invalid: outside range
        assert validate_constraints({"alpha": 0.05}, [constraint]) is False
        assert validate_constraints({"alpha": 15.0}, [constraint]) is False

    def test_product_max_constraint(self):
        """Test product_max constraint."""
        constraint = SweepConstraint("product_max", ["a", "b"], 100)

        # Valid: a * b <= 100
        assert validate_constraints({"a": 5, "b": 10}, [constraint]) is True
        assert validate_constraints({"a": 10, "b": 10}, [constraint]) is True

        # Invalid: a * b > 100
        assert validate_constraints({"a": 15, "b": 10}, [constraint]) is False

    def test_product_min_constraint(self):
        """Test product_min constraint."""
        constraint = SweepConstraint("product_min", ["a", "b"], 50)

        # Valid: a * b >= 50
        assert validate_constraints({"a": 10, "b": 10}, [constraint]) is True
        assert validate_constraints({"a": 5, "b": 10}, [constraint]) is True

        # Invalid: a * b < 50
        assert validate_constraints({"a": 3, "b": 10}, [constraint]) is False

    def test_same_sign_constraint(self):
        """Test same_sign constraint."""
        constraint = SweepConstraint("same_sign", ["a", "b", "c"], 0)

        # Valid: all same sign
        assert validate_constraints({"a": 1.0, "b": 2.0, "c": 0.5}, [constraint]) is True
        assert validate_constraints({"a": -1.0, "b": -2.0, "c": -0.5}, [constraint]) is True
        assert validate_constraints({"a": 1.0, "b": 0, "c": 2.0}, [constraint]) is True  # Zero ignored

        # Invalid: mixed signs
        assert validate_constraints({"a": 1.0, "b": -1.0, "c": 0.5}, [constraint]) is False

    def test_step_multiple_constraint(self):
        """Test step_multiple constraint."""
        constraint = SweepConstraint("step_multiple", ["period"], 5)

        # Valid: multiples of 5
        assert validate_constraints({"period": 10}, [constraint]) is True
        assert validate_constraints({"period": 25}, [constraint]) is True
        assert validate_constraints({"period": 0}, [constraint]) is True

        # Invalid: not a multiple of 5
        assert validate_constraints({"period": 12}, [constraint]) is False
        assert validate_constraints({"period": 7}, [constraint]) is False

    def test_monotonic_increasing_constraint(self):
        """Test monotonic_increasing constraint."""
        constraint = SweepConstraint("monotonic_increasing", ["short", "medium", "long"], 0)

        # Valid: short < medium < long
        assert validate_constraints({"short": 5, "medium": 10, "long": 20}, [constraint]) is True
        assert validate_constraints({"short": 1, "medium": 2, "long": 3}, [constraint]) is True

        # Invalid: not strictly increasing
        assert validate_constraints({"short": 10, "medium": 10, "long": 20}, [constraint]) is False
        assert validate_constraints({"short": 15, "medium": 10, "long": 20}, [constraint]) is False

    def test_at_least_n_nonzero_constraint(self):
        """Test at_least_n_nonzero constraint."""
        constraint = SweepConstraint("at_least_n_nonzero", ["a", "b", "c", "d"], 2)

        # Valid: at least 2 non-zero
        assert validate_constraints({"a": 1, "b": 1, "c": 0, "d": 0}, [constraint]) is True
        assert validate_constraints({"a": 1, "b": 1, "c": 1, "d": 0}, [constraint]) is True

        # Invalid: fewer than 2 non-zero
        assert validate_constraints({"a": 1, "b": 0, "c": 0, "d": 0}, [constraint]) is False
        assert validate_constraints({"a": 0, "b": 0, "c": 0, "d": 0}, [constraint]) is False

    def test_constraint_validation(self):
        """Test constraint validation works as expected."""
        constraint = SweepConstraint("sum_equals", ["a", "b"], 1.0)
        valid_params = {"a": 0.5, "b": 0.5}
        invalid_params = {"a": 0.3, "b": 0.3}

        assert validate_constraints(valid_params, [constraint]) is True
        assert validate_constraints(invalid_params, [constraint]) is False


class TestGenerateValidCombinations:
    """Tests for _generate_valid_combinations function."""

    def test_no_constraints(self):
        """Test generating combinations without constraints."""
        param_ranges = {"a": [1, 2], "b": [3, 4]}
        combinations, violations = _generate_valid_combinations(param_ranges, [])

        assert len(combinations) == 4
        assert violations == 0

    def test_with_constraint(self):
        """Test generating combinations with constraint."""
        param_ranges = {"a": [0.3, 0.5, 0.7], "b": [0.3, 0.5, 0.7]}
        constraint = SweepConstraint("sum_equals", ["a", "b"], 1.0)

        combinations, violations = _generate_valid_combinations(param_ranges, [constraint])

        # Only (0.3, 0.7), (0.5, 0.5), (0.7, 0.3) should be valid
        assert len(combinations) == 3
        assert violations == 6


class TestComputeSharpeDiffAnalysis:
    """Tests for compute_sharpe_diff_analysis function."""

    def test_sharpe_diff_analysis(self):
        """Test Sharpe diff computation."""
        results_df = pd.DataFrame({
            "param_a": [1, 2, 3],
            "train_sharpe_fold_0": [1.0, 1.2, 1.1],
            "train_sharpe_fold_1": [0.9, 1.1, 1.0],
            "test_sharpe_fold_0": [0.8, 1.0, 0.9],
            "test_sharpe_fold_1": [0.7, 0.9, 0.8],
        })

        diff_matrix, diff_agg, gen_score = compute_sharpe_diff_analysis(
            results_df, ["param_a"], "median"
        )

        # Check diff matrix shape
        assert len(diff_matrix) == 3
        assert "diff_fold_0" in diff_matrix.columns

        # Check diff values (test - train)
        assert diff_matrix["diff_fold_0"].iloc[0] == pytest.approx(-0.2)

        # Check aggregated diff
        assert len(diff_agg) == 3

    def test_generalization_score_calculation(self):
        """Test generalization score is percentage of positive diffs."""
        results_df = pd.DataFrame({
            "param_a": [1, 2, 3, 4],
            "train_sharpe_fold_0": [1.0, 1.0, 1.0, 1.0],
            "test_sharpe_fold_0": [1.1, 0.9, 1.0, 0.8],  # 2 positive, 1 zero, 1 negative
        })

        _, _, gen_score = compute_sharpe_diff_analysis(
            results_df, ["param_a"], "median"
        )

        # 3/4 have diff >= 0 (1.1-1.0=0.1, 1.0-1.0=0, both >= 0)
        assert gen_score == pytest.approx(0.5)  # 2 out of 4


class TestParallelParameterSweep:
    """Tests for parallel_parameter_sweep function.

    Note: _evaluate_single_combination is not yet implemented (raises
    NotImplementedError). These tests verify the error is raised until
    the module is wired to the real backtest engine.
    """

    @pytest.fixture
    def sample_config(self):
        """Create sample SweepConfig."""
        return SweepConfig(
            strategy_type="multifactor",
            param_ranges={"alpha": [0.1, 0.5, 1.0]},
            constraints=[],
            symbols=["AAPL", "MSFT"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=3,
        )

    def test_parallel_execution_raises_not_implemented(self, sample_config):
        """parallel_parameter_sweep raises NotImplementedError (not yet wired to backtest)."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            parallel_parameter_sweep(sample_config)

    def test_with_constraints_raises_not_implemented(self):
        """Constrained sweep also raises NotImplementedError."""
        config = SweepConfig(
            strategy_type="momentum",
            param_ranges={
                "fast_period": [5, 10, 15],
                "slow_period": [10, 20, 30],
            },
            constraints=[
                SweepConstraint("difference_min", ["slow_period", "fast_period"], 10),
            ],
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            n_folds=2,
        )

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            parallel_parameter_sweep(config)


class TestModuleExports:
    """Test that parameter_sweep module is properly exported."""

    def test_import_classes(self):
        """Test importing classes from hrp.research.parameter_sweep."""
        from hrp.research.parameter_sweep import (
            SweepConstraint,
            SweepConfig,
            SweepResult,
            validate_constraints,
            parallel_parameter_sweep,
        )

        assert SweepConstraint is not None
        assert SweepConfig is not None
        assert SweepResult is not None
        assert validate_constraints is not None
        assert parallel_parameter_sweep is not None
