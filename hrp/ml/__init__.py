"""
ML Framework for HRP.

Provides model training, signal generation, validation, and deployment.
"""

from hrp.ml.models import MLConfig, SUPPORTED_MODELS, get_model, HAS_LIGHTGBM, HAS_XGBOOST
from hrp.ml.signals import predictions_to_signals
from hrp.ml.training import TrainingResult, train_model, load_training_data, select_features
from hrp.ml.validation import (
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    walk_forward_validate,
    generate_folds,
    compute_fold_metrics,
    aggregate_fold_metrics,
)
from hrp.ml.optimization import (
    OptimizationConfig,
    OptimizationResult,
    cross_validated_optimize,
    SCORING_METRICS,
)

# Re-export Optuna distributions for convenience
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
from hrp.ml.regime import (
    MarketRegime,
    HMMConfig,
    RegimeResult,
    RegimeDetector,
)
from hrp.ml.registry import ModelRegistry, RegisteredModel, get_model_registry
from hrp.ml.deployment import (
    DeploymentPipeline,
    DeploymentConfig,
    DeploymentResult,
    get_deployment_pipeline,
)
from hrp.ml.inference import ModelPredictor, PredictionResult, get_model_predictor

__all__ = [
    # Models
    "MLConfig",
    "SUPPORTED_MODELS",
    "get_model",
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    # Signals
    "predictions_to_signals",
    # Training
    "TrainingResult",
    "train_model",
    "load_training_data",
    "select_features",
    # Validation
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
    "walk_forward_validate",
    "generate_folds",
    "compute_fold_metrics",
    "aggregate_fold_metrics",
    # Optimization
    "OptimizationConfig",
    "OptimizationResult",
    "cross_validated_optimize",
    "SCORING_METRICS",
    # Optuna distributions (re-exported for convenience)
    "FloatDistribution",
    "IntDistribution",
    "CategoricalDistribution",
    # Regime Detection
    "MarketRegime",
    "HMMConfig",
    "RegimeResult",
    "RegimeDetector",
    # Model Registry
    "ModelRegistry",
    "RegisteredModel",
    "get_model_registry",
    # Deployment
    "DeploymentPipeline",
    "DeploymentConfig",
    "DeploymentResult",
    "get_deployment_pipeline",
    # Inference
    "ModelPredictor",
    "PredictionResult",
    "get_model_predictor",
]
