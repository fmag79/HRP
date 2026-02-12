"""
Configuration for risk metric calculations.

Defines default parameters and configuration dataclasses for VaR and CVaR calculations.
"""

from dataclasses import dataclass
from enum import Enum


class VaRMethod(Enum):
    """Enumeration of supported VaR calculation methods."""

    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


class Distribution(Enum):
    """Probability distributions for parametric and Monte Carlo VaR."""

    NORMAL = "normal"
    T = "t"  # Student's t-distribution


@dataclass
class VaRConfig:
    """
    Configuration for VaR and CVaR calculations.

    Attributes:
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
        time_horizon: Time horizon in days (e.g., 1 for 1-day VaR, 10 for 10-day VaR)
        method: Calculation method (parametric, historical, or Monte Carlo)
        distribution: Distribution assumption for parametric/MC methods (normal or t)
        n_simulations: Number of Monte Carlo simulations (only used for MC method)
        window_size: Historical window size in days (only used for historical method)
        df: Degrees of freedom for t-distribution (only used when distribution=T)
    """

    confidence_level: float = 0.95
    time_horizon: int = 1
    method: VaRMethod = VaRMethod.PARAMETRIC
    distribution: Distribution = Distribution.NORMAL
    n_simulations: int = 10000
    window_size: int = 252  # 1 year of trading days
    df: float = 5.0  # Degrees of freedom for t-distribution

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.time_horizon < 1:
            raise ValueError("time_horizon must be at least 1 day")
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be at least 100")
        if self.window_size < 30:
            raise ValueError("window_size must be at least 30 days")
        if self.df <= 0:
            raise ValueError("df (degrees of freedom) must be positive")


# Common VaR configurations
VAR_95_1D = VaRConfig(confidence_level=0.95, time_horizon=1)
VAR_99_1D = VaRConfig(confidence_level=0.99, time_horizon=1)
VAR_95_10D = VaRConfig(confidence_level=0.95, time_horizon=10)
VAR_99_10D = VaRConfig(confidence_level=0.99, time_horizon=10)

# Monte Carlo with t-distribution (captures fat tails)
MC_VAR_95_1D = VaRConfig(
    confidence_level=0.95,
    time_horizon=1,
    method=VaRMethod.MONTE_CARLO,
    distribution=Distribution.T,
    n_simulations=10000,
)
