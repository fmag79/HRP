"""
ML Framework for HRP.

Provides model training, signal generation, and validation.
"""

from hrp.ml.models import MLConfig, SUPPORTED_MODELS, get_model, HAS_LIGHTGBM, HAS_XGBOOST
from hrp.ml.signals import predictions_to_signals
from hrp.ml.training import TrainingResult, train_model, load_training_data, select_features

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
]
