"""Attribution configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AttributionMethod(str, Enum):
    """Attribution method enum."""

    BRINSON = "brinson"
    REGRESSION = "regression"


class ImportanceMethod(str, Enum):
    """Feature importance method enum."""

    PERMUTATION = "permutation"
    SHAP = "shap"


@dataclass
class AttributionConfig:
    """Configuration for performance attribution.

    Controls attribution method, benchmark selection, and analysis period.
    """

    # Factor attribution method
    method: Literal["brinson", "regression"] = "brinson"

    # Benchmark for active return calculation
    benchmark: str = "SPY"  # Default to S&P 500

    # Period for attribution analysis (trading days)
    lookback_days: int = 252  # 1 year default

    # Factor model for regression-based attribution
    factor_model: Literal["market", "fama_french_3", "fama_french_5"] = "market"

    # Feature importance settings
    permutation_n_repeats: int = 10
    shap_enabled: bool = False  # Requires 'shap' package
    rolling_window_days: int = 60  # For rolling feature importance

    # Decision attribution settings
    include_timing: bool = True
    include_sizing: bool = True
    include_rebalancing: bool = True

    # Asset classification (for sector/style attribution)
    sector_classification: str | None = None  # Path to CSV with asset->sector mapping

    # Caching
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

    # Validation
    validate_summation: bool = True  # Ensure attribution effects sum correctly
    tolerance: float = 1e-6  # Numerical tolerance for validation


# Default configs for common use cases
DEFAULT_CONFIG = AttributionConfig()

BRINSON_CONFIG = AttributionConfig(
    method="brinson",
    benchmark="SPY",
    factor_model="market",
)

REGRESSION_CONFIG = AttributionConfig(
    method="regression",
    benchmark="SPY",
    factor_model="fama_french_3",
)
