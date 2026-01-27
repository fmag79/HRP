"""
Risk management and validation framework.

Provides overfitting guards, statistical validation, and robustness checks.
"""

from hrp.risk.costs import CostBreakdown, MarketImpactModel
from hrp.risk.limits import (
    RiskLimits,
    LimitViolation,
    PreTradeValidator,
    RiskLimitViolationError,
    ValidationReport as PreTradeValidationReport,
)
from hrp.risk.overfitting import (
    TestSetGuard,
    OverfittingError,
    SharpeDecayMonitor,
    DecayCheckResult,
    FeatureCountValidator,
    FeatureCountResult,
    HyperparameterTrialCounter,
    TargetLeakageValidator,
    LeakageCheckResult,
)
from hrp.risk.validation import (
    ValidationCriteria,
    ValidationResult,
    validate_strategy,
    significance_test,
    calculate_bootstrap_ci,
    bonferroni_correction,
    benjamini_hochberg,
)
from hrp.risk.robustness import (
    RobustnessResult,
    check_parameter_sensitivity,
    check_time_stability,
    check_regime_stability,
)
from hrp.risk.report import ValidationReport, generate_validation_report

# Re-export the hypothesis validation report
HypothesisValidationReport = ValidationReport

__all__ = [
    # Transaction cost model
    "CostBreakdown",
    "MarketImpactModel",
    # Risk limits
    "RiskLimits",
    "LimitViolation",
    "PreTradeValidator",
    "RiskLimitViolationError",
    "PreTradeValidationReport",
    # Overfitting prevention
    "TestSetGuard",
    "OverfittingError",
    "SharpeDecayMonitor",
    "DecayCheckResult",
    "FeatureCountValidator",
    "FeatureCountResult",
    "HyperparameterTrialCounter",
    "TargetLeakageValidator",
    "LeakageCheckResult",
    # Statistical validation
    "ValidationCriteria",
    "ValidationResult",
    "validate_strategy",
    "significance_test",
    "calculate_bootstrap_ci",
    "bonferroni_correction",
    "benjamini_hochberg",
    # Robustness checks
    "RobustnessResult",
    "check_parameter_sensitivity",
    "check_time_stability",
    "check_regime_stability",
    # Validation reports
    "ValidationReport",
    "generate_validation_report",
]
