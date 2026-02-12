"""
Value-at-Risk (VaR) and Conditional VaR (CVaR) calculator.

Implements three methods for calculating VaR:
1. Parametric (normal or t-distribution)
2. Historical Simulation (empirical distribution)
3. Monte Carlo (simulated distribution with t-distribution)
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from loguru import logger
from scipy import stats

from hrp.data.risk.risk_config import Distribution, VaRConfig, VaRMethod


@dataclass
class VaRResult:
    """
    Result of a VaR calculation.

    Attributes:
        var: Value-at-Risk (positive value representing potential loss)
        cvar: Conditional Value-at-Risk (expected loss beyond VaR)
        confidence_level: Confidence level used (e.g., 0.95)
        time_horizon: Time horizon in days
        method: Calculation method used
        timestamp: When the calculation was performed
        portfolio_value: Current portfolio value (optional)
        var_dollar: VaR in dollar terms (optional)
        cvar_dollar: CVaR in dollar terms (optional)
    """

    var: float
    cvar: float
    confidence_level: float
    time_horizon: int
    method: str
    timestamp: datetime
    portfolio_value: float | None = None
    var_dollar: float | None = None
    cvar_dollar: float | None = None

    def __post_init__(self):
        """Calculate dollar values if portfolio_value is provided."""
        if self.portfolio_value is not None:
            self.var_dollar = self.var * self.portfolio_value
            self.cvar_dollar = self.cvar * self.portfolio_value


class VaRCalculator:
    """
    Calculator for Value-at-Risk and Conditional VaR.

    Supports three calculation methods:
    - Parametric: Assumes returns follow a specified distribution (normal or t)
    - Historical: Uses empirical distribution of historical returns
    - Monte Carlo: Simulates returns based on fitted distribution
    """

    def __init__(self, config: VaRConfig | None = None):
        """
        Initialize VaR calculator with configuration.

        Args:
            config: VaRConfig object with calculation parameters
        """
        self.config = config or VaRConfig()
        logger.debug(f"VaR calculator initialized with config: {self.config}")

    def calculate(
        self,
        returns: np.ndarray,
        portfolio_value: float | None = None,
        config: VaRConfig | None = None,
    ) -> VaRResult:
        """
        Calculate VaR and CVaR using the configured method.

        Args:
            returns: Array of historical returns (e.g., daily returns)
            portfolio_value: Optional current portfolio value for dollar VaR
            config: Optional config to override default

        Returns:
            VaRResult with VaR, CVaR, and metadata

        Raises:
            ValueError: If returns array is empty or invalid
        """
        cfg = config or self.config

        # Validate inputs
        if returns is None or len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        if len(returns) < 30:
            raise ValueError("Need at least 30 valid return observations")

        logger.debug(
            f"Calculating VaR: method={cfg.method.value}, "
            f"confidence={cfg.confidence_level}, horizon={cfg.time_horizon}"
        )

        # Route to appropriate method
        if cfg.method == VaRMethod.PARAMETRIC:
            var, cvar = self._parametric_var(returns, cfg)
        elif cfg.method == VaRMethod.HISTORICAL:
            var, cvar = self._historical_var(returns, cfg)
        elif cfg.method == VaRMethod.MONTE_CARLO:
            var, cvar = self._monte_carlo_var(returns, cfg)
        else:
            raise ValueError(f"Unknown VaR method: {cfg.method}")

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=cfg.confidence_level,
            time_horizon=cfg.time_horizon,
            method=cfg.method.value,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
        )

    def _parametric_var(self, returns: np.ndarray, config: VaRConfig) -> tuple[float, float]:
        """
        Calculate VaR using parametric method (normal or t-distribution).

        VaR = -μ + σ * z * sqrt(T)
        where z is the quantile of the distribution at confidence_level

        Args:
            returns: Historical returns
            config: VaR configuration

        Returns:
            Tuple of (VaR, CVaR) as positive values representing losses
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)  # Sample std dev

        # Get quantile based on distribution
        if config.distribution == Distribution.NORMAL:
            # For normal distribution: z-score at confidence level
            z = stats.norm.ppf(1 - config.confidence_level)
        else:  # t-distribution
            z = stats.t.ppf(1 - config.confidence_level, df=config.df)

        # Scale to time horizon using square root rule
        time_scale = np.sqrt(config.time_horizon)

        # VaR formula: negative of the quantile (expressed as positive loss)
        # Lower tail: quantile = mu + z*sigma (z is negative for lower tail)
        var = -(mu * config.time_horizon + z * sigma * time_scale)

        # CVaR: Expected loss beyond VaR
        # For normal: E[L | L > VaR] = -mu*T + sigma*sqrt(T) * phi(z) / alpha
        # where phi is the PDF and alpha is the tail probability
        if config.distribution == Distribution.NORMAL:
            pdf_at_z = stats.norm.pdf(z)
            tail_prob = 1 - config.confidence_level
            # CVaR formula: negative of expected return in the tail
            cvar = -(mu * config.time_horizon) + sigma * time_scale * pdf_at_z / tail_prob
        else:  # t-distribution
            pdf_at_z = stats.t.pdf(z, df=config.df)
            # CVaR for t-distribution
            tail_prob = 1 - config.confidence_level
            scaling_factor = (config.df + z**2) / (config.df - 1)
            cvar = -(mu * config.time_horizon) + sigma * time_scale * pdf_at_z * scaling_factor / tail_prob

        return float(var), float(cvar)

    def _historical_var(self, returns: np.ndarray, config: VaRConfig) -> tuple[float, float]:
        """
        Calculate VaR using historical simulation (empirical distribution).

        For multi-day VaR, uses overlapping windows rather than sqrt(T) scaling.

        Args:
            returns: Historical returns
            config: VaR configuration

        Returns:
            Tuple of (VaR, CVaR) as positive values representing losses
        """
        # For 1-day horizon, use returns directly
        if config.time_horizon == 1:
            scaled_returns = returns
        else:
            # For multi-day horizon, calculate overlapping N-day returns
            # This captures path dependency and autocorrelation
            n = len(returns)
            horizon = config.time_horizon
            if n < horizon + 30:
                raise ValueError(
                    f"Need at least {horizon + 30} returns for {horizon}-day historical VaR"
                )

            # Calculate overlapping N-day cumulative returns
            scaled_returns = np.array(
                [
                    np.sum(returns[i : i + horizon])
                    for i in range(n - horizon + 1)
                ]
            )

        # VaR is the quantile at (1 - confidence_level)
        # We want losses, so negate the returns for the left tail
        losses = -scaled_returns
        var = np.quantile(losses, config.confidence_level)

        # CVaR is the mean of losses beyond VaR
        tail_losses = losses[losses >= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

        return float(var), float(cvar)

    def _monte_carlo_var(self, returns: np.ndarray, config: VaRConfig) -> tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.

        Fits a distribution to historical returns, then simulates many paths
        to build a distribution of potential losses.

        Args:
            returns: Historical returns
            config: VaR configuration

        Returns:
            Tuple of (VaR, CVaR) as positive values representing losses
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Fit distribution parameters if using t-distribution
        if config.distribution == Distribution.T:
            # Fit t-distribution to data to estimate df
            params = stats.t.fit(returns)
            df_fitted = params[0]
            loc = params[1]
            scale = params[2]
            logger.debug(f"Fitted t-distribution: df={df_fitted:.2f}, loc={loc:.4f}, scale={scale:.4f}")
        else:
            df_fitted = None
            loc = mu
            scale = sigma

        # Simulate returns for the time horizon
        simulated_losses = []
        for _ in range(config.n_simulations):
            # Simulate path of returns for time_horizon days
            if config.distribution == Distribution.NORMAL:
                path_returns = np.random.normal(loc, scale, config.time_horizon)
            else:  # t-distribution
                path_returns = stats.t.rvs(df_fitted, loc=loc, scale=scale, size=config.time_horizon)

            # Cumulative loss over the horizon
            cumulative_return = np.sum(path_returns)
            simulated_losses.append(-cumulative_return)

        simulated_losses = np.array(simulated_losses)

        # VaR and CVaR from simulated distribution
        var = np.quantile(simulated_losses, config.confidence_level)
        tail_losses = simulated_losses[simulated_losses >= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

        return float(var), float(cvar)

    def calculate_all_methods(
        self,
        returns: np.ndarray,
        portfolio_value: float | None = None,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
    ) -> dict[str, VaRResult]:
        """
        Calculate VaR using all three methods for comparison.

        Args:
            returns: Historical returns
            portfolio_value: Optional current portfolio value
            confidence_level: Confidence level (default 0.95)
            time_horizon: Time horizon in days (default 1)

        Returns:
            Dictionary mapping method name to VaRResult
        """
        results = {}

        for method in VaRMethod:
            config = VaRConfig(
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method=method,
                distribution=Distribution.T if method == VaRMethod.MONTE_CARLO else Distribution.NORMAL,
            )
            try:
                result = self.calculate(returns, portfolio_value, config)
                results[method.value] = result
            except Exception as e:
                logger.warning(f"Failed to calculate {method.value} VaR: {e}")

        return results
