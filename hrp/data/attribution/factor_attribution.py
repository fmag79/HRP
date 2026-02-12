"""Factor-level return attribution.

Implements:
- Brinson-Fachler attribution (allocation, selection, interaction effects)
- Regression-based factor decomposition (Fama-French style)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AttributionResult:
    """Result of a single attribution calculation.

    Attributes:
        factor: Name of the factor/sector (e.g., "Technology", "Market Beta")
        effect_type: Type of effect ("allocation", "selection", "interaction", "factor")
        contribution_pct: Contribution to active return as percentage
        contribution_dollar: Contribution to active return in dollar terms (if portfolio value provided)
        active_return: Total active return (portfolio return - benchmark return)
    """

    factor: str
    effect_type: Literal["allocation", "selection", "interaction", "factor", "residual"]
    contribution_pct: float
    contribution_dollar: float | None = None
    active_return: float | None = None

    def __post_init__(self):
        """Validate fields."""
        if self.effect_type not in ["allocation", "selection", "interaction", "factor", "residual"]:
            raise ValueError(f"Invalid effect_type: {self.effect_type}")


class BrinsonAttribution:
    """Brinson-Fachler attribution analysis.

    Decomposes active return into allocation, selection, and interaction effects
    at the sector/factor level.

    The Brinson-Fachler model attributes active return as:
        Active Return = Σ(Allocation_i + Selection_i + Interaction_i)

    Where for each sector i:
        Allocation_i  = (w_p_i - w_b_i) × (r_b_i - R_b)
        Selection_i   = w_b_i × (r_p_i - r_b_i)
        Interaction_i = (w_p_i - w_b_i) × (r_p_i - r_b_i)

    And:
        w_p_i = portfolio weight in sector i
        w_b_i = benchmark weight in sector i
        r_p_i = portfolio return in sector i
        r_b_i = benchmark return in sector i
        R_b   = total benchmark return
    """

    def __init__(self, validate_summation: bool = True, tolerance: float = 1e-6):
        """Initialize Brinson attribution.

        Args:
            validate_summation: If True, validate that effects sum to active return
            tolerance: Numerical tolerance for validation
        """
        self.validate_summation = validate_summation
        self.tolerance = tolerance

    def attribute(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_value: float | None = None,
    ) -> list[AttributionResult]:
        """Compute Brinson-Fachler attribution.

        Args:
            portfolio_weights: Weights of each sector in portfolio (index = sectors)
            portfolio_returns: Returns of each sector in portfolio
            benchmark_weights: Weights of each sector in benchmark
            benchmark_returns: Returns of each sector in benchmark
            portfolio_value: Optional portfolio value for dollar attribution

        Returns:
            List of AttributionResult objects, one per sector and effect type

        Raises:
            ValueError: If inputs are invalid or summation validation fails
        """
        # Input validation
        self._validate_inputs(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )

        # Align indices (use union of all sectors)
        sectors = portfolio_weights.index.union(benchmark_weights.index)
        portfolio_weights = portfolio_weights.reindex(sectors, fill_value=0.0)
        benchmark_weights = benchmark_weights.reindex(sectors, fill_value=0.0)
        portfolio_returns = portfolio_returns.reindex(sectors, fill_value=0.0)
        benchmark_returns = benchmark_returns.reindex(sectors, fill_value=0.0)

        # Compute total returns
        portfolio_return = (portfolio_weights * portfolio_returns).sum()
        benchmark_return = (benchmark_weights * benchmark_returns).sum()
        active_return = portfolio_return - benchmark_return

        # Compute effects for each sector
        results: list[AttributionResult] = []

        for sector in sectors:
            w_p = portfolio_weights[sector]
            w_b = benchmark_weights[sector]
            r_p = portfolio_returns[sector]
            r_b = benchmark_returns[sector]

            # Brinson-Fachler formulas
            allocation = (w_p - w_b) * (r_b - benchmark_return)
            selection = w_b * (r_p - r_b)
            interaction = (w_p - w_b) * (r_p - r_b)

            # Convert to dollar terms if portfolio value provided
            allocation_dollar = allocation * portfolio_value if portfolio_value else None
            selection_dollar = selection * portfolio_value if portfolio_value else None
            interaction_dollar = (
                interaction * portfolio_value if portfolio_value else None
            )

            # Create results
            results.extend(
                [
                    AttributionResult(
                        factor=sector,
                        effect_type="allocation",
                        contribution_pct=allocation * 100,
                        contribution_dollar=allocation_dollar,
                        active_return=active_return * 100,
                    ),
                    AttributionResult(
                        factor=sector,
                        effect_type="selection",
                        contribution_pct=selection * 100,
                        contribution_dollar=selection_dollar,
                        active_return=active_return * 100,
                    ),
                    AttributionResult(
                        factor=sector,
                        effect_type="interaction",
                        contribution_pct=interaction * 100,
                        contribution_dollar=interaction_dollar,
                        active_return=active_return * 100,
                    ),
                ]
            )

        # Validate summation
        if self.validate_summation:
            total_attribution = sum(r.contribution_pct for r in results) / 100
            if abs(total_attribution - active_return) > self.tolerance:
                raise ValueError(
                    f"Attribution summation failed: {total_attribution:.6f} != {active_return:.6f} "
                    f"(diff: {abs(total_attribution - active_return):.6e})"
                )

        return results

    def _validate_inputs(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
    ):
        """Validate input series.

        Args:
            portfolio_weights: Portfolio weights
            portfolio_returns: Portfolio returns
            benchmark_weights: Benchmark weights
            benchmark_returns: Benchmark returns

        Raises:
            ValueError: If inputs are invalid
        """
        # Check types
        for name, series in [
            ("portfolio_weights", portfolio_weights),
            ("portfolio_returns", portfolio_returns),
            ("benchmark_weights", benchmark_weights),
            ("benchmark_returns", benchmark_returns),
        ]:
            if not isinstance(series, pd.Series):
                raise ValueError(f"{name} must be a pandas Series")

        # Check weights sum to ~1.0
        pw_sum = portfolio_weights.sum()
        bw_sum = benchmark_weights.sum()
        if abs(pw_sum - 1.0) > self.tolerance:
            raise ValueError(
                f"Portfolio weights sum to {pw_sum:.6f}, expected 1.0 (±{self.tolerance})"
            )
        if abs(bw_sum - 1.0) > self.tolerance:
            raise ValueError(
                f"Benchmark weights sum to {bw_sum:.6f}, expected 1.0 (±{self.tolerance})"
            )

        # Check for NaN/inf
        for name, series in [
            ("portfolio_weights", portfolio_weights),
            ("portfolio_returns", portfolio_returns),
            ("benchmark_weights", benchmark_weights),
            ("benchmark_returns", benchmark_returns),
        ]:
            if series.isna().any():
                raise ValueError(f"{name} contains NaN values")
            if np.isinf(series).any():
                raise ValueError(f"{name} contains infinite values")

    def aggregate_by_effect(
        self, results: list[AttributionResult]
    ) -> dict[str, float]:
        """Aggregate attribution results by effect type.

        Args:
            results: List of AttributionResult objects

        Returns:
            Dictionary mapping effect_type -> total contribution (in pct)
        """
        aggregated = {}
        for effect_type in ["allocation", "selection", "interaction"]:
            total = sum(
                r.contribution_pct for r in results if r.effect_type == effect_type
            )
            aggregated[effect_type] = total
        return aggregated


class FactorAttribution:
    """Regression-based factor attribution.

    Uses linear regression to decompose returns into factor exposures.
    Supports multiple factor models:
    - Market model (single factor)
    - Fama-French 3-factor (market, SMB, HML)
    - Fama-French 5-factor (market, SMB, HML, RMW, CMA)
    """

    def __init__(
        self,
        factor_model: Literal["market", "fama_french_3", "fama_french_5"] = "market",
    ):
        """Initialize factor attribution.

        Args:
            factor_model: Which factor model to use
        """
        self.factor_model = factor_model
        self.coefficients_: dict[str, float] | None = None
        self.r_squared_: float | None = None
        self.residual_: float | None = None

    def attribute(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        portfolio_value: float | None = None,
    ) -> list[AttributionResult]:
        """Compute regression-based factor attribution.

        Args:
            portfolio_returns: Time series of portfolio returns
            factor_returns: DataFrame of factor returns (columns = factors)
            portfolio_value: Optional portfolio value for dollar attribution

        Returns:
            List of AttributionResult objects, one per factor + residual

        Raises:
            ValueError: If inputs are invalid or regression fails
        """
        # Input validation
        self._validate_inputs(portfolio_returns, factor_returns)

        # Align time series
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        y = aligned_data.iloc[:, 0].values
        X = aligned_data.iloc[:, 1:].values

        # Run OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X[:, 0], y
        ) if X.shape[1] == 1 else self._multi_factor_regression(X, y)

        # For multi-factor, use numpy lstsq
        if X.shape[1] > 1:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            intercept = 0.0  # Assuming zero intercept for factor models
            r_squared = 1 - (residuals[0] / ((y - y.mean()) ** 2).sum())
        else:
            coeffs = np.array([slope])
            r_squared = r_value**2
            residuals = y - (X[:, 0] * slope + intercept)
            residuals = np.array([np.sum(residuals**2)])

        # Store results
        self.coefficients_ = dict(zip(factor_returns.columns, coeffs))
        self.r_squared_ = r_squared
        self.residual_ = residuals[0] if len(residuals) > 0 else 0.0

        # Compute average portfolio return
        avg_portfolio_return = portfolio_returns.mean()

        # Decompose return into factor contributions
        results: list[AttributionResult] = []
        for factor_name, coeff in self.coefficients_.items():
            avg_factor_return = factor_returns[factor_name].mean()
            contribution = coeff * avg_factor_return
            contribution_dollar = (
                contribution * portfolio_value if portfolio_value else None
            )

            results.append(
                AttributionResult(
                    factor=factor_name,
                    effect_type="factor",
                    contribution_pct=contribution * 100,
                    contribution_dollar=contribution_dollar,
                    active_return=avg_portfolio_return * 100,
                )
            )

        # Add alpha (intercept) as residual
        alpha = avg_portfolio_return - sum(
            coeff * factor_returns[fname].mean()
            for fname, coeff in self.coefficients_.items()
        )
        alpha_dollar = alpha * portfolio_value if portfolio_value else None

        results.append(
            AttributionResult(
                factor="Alpha (residual)",
                effect_type="residual",
                contribution_pct=alpha * 100,
                contribution_dollar=alpha_dollar,
                active_return=avg_portfolio_return * 100,
            )
        )

        return results

    def _validate_inputs(
        self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame
    ):
        """Validate inputs.

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(portfolio_returns, pd.Series):
            raise ValueError("portfolio_returns must be a pandas Series")
        if not isinstance(factor_returns, pd.DataFrame):
            raise ValueError("factor_returns must be a pandas DataFrame")

        if portfolio_returns.isna().any():
            raise ValueError("portfolio_returns contains NaN values")
        if factor_returns.isna().any().any():
            raise ValueError("factor_returns contains NaN values")

        if len(portfolio_returns) < 2:
            raise ValueError("Need at least 2 observations for regression")

    def _multi_factor_regression(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """Run multi-factor regression (helper for stats.linregress compatibility).

        Args:
            X: Factor returns (n_samples, n_factors)
            y: Portfolio returns (n_samples,)

        Returns:
            Dummy tuple matching stats.linregress signature (slope=0, intercept=0, r=0, p=0, std_err=0)
        """
        # This is a placeholder - actual multi-factor regression is handled by lstsq above
        return (0.0, 0.0, 0.0, 1.0, 0.0)
