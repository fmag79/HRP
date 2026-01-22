"""
ML Model Registry.

Provides a centralized registry of supported ML models with
configuration management and instantiation utilities.

Usage:
    from hrp.ml.models import MLConfig, get_model, SUPPORTED_MODELS

    # Create a model configuration
    config = MLConfig(
        model_type="ridge",
        target="returns_20d",
        features=["momentum_20d", "volatility_20d"],
        train_start=date(2015, 1, 1),
        train_end=date(2018, 12, 31),
        validation_start=date(2019, 1, 1),
        validation_end=date(2019, 12, 31),
        test_start=date(2020, 1, 1),
        test_end=date(2023, 12, 31),
    )

    # Instantiate a model
    model = get_model("ridge", {"alpha": 0.5})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Try to import optional dependencies
# Note: We catch Exception because some libraries may raise OSError
# when native dependencies are missing (e.g., libomp for LightGBM)
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    lgb = None

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    xgb = None


# Supported model types and their corresponding classes
SUPPORTED_MODELS: dict[str, type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "mlp": MLPRegressor,
}

# Add optional models if available
if HAS_LIGHTGBM:
    SUPPORTED_MODELS["lightgbm"] = lgb.LGBMRegressor

if HAS_XGBOOST:
    SUPPORTED_MODELS["xgboost"] = xgb.XGBRegressor


@dataclass
class MLConfig:
    """
    Configuration for ML model training.

    Defines the model type, target variable, features, date ranges for
    train/validation/test splits, and hyperparameters.

    Attributes:
        model_type: Type of model to train (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., "returns_20d")
        features: List of feature names to use
        train_start: Start date for training period
        train_end: End date for training period
        validation_start: Start date for validation period
        validation_end: End date for validation period
        test_start: Start date for test period
        test_end: End date for test period
        hyperparameters: Model hyperparameters
        feature_selection: Whether to apply feature selection
        max_features: Maximum number of features to select
    """

    model_type: str
    target: str
    features: list[str]
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    max_features: int = 20

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_type not in SUPPORTED_MODELS:
            available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            raise ValueError(
                f"Unsupported model type: '{self.model_type}'. "
                f"Available types: {available}"
            )

        logger.debug(
            f"MLConfig created: model_type={self.model_type}, "
            f"target={self.target}, features={len(self.features)}"
        )


def get_model(model_type: str, hyperparameters: dict[str, Any] | None = None) -> Any:
    """
    Instantiate a model of the specified type.

    Args:
        model_type: Type of model to instantiate (must be in SUPPORTED_MODELS)
        hyperparameters: Optional dict of hyperparameters to pass to the model

    Returns:
        Instantiated model object

    Raises:
        ValueError: If model_type is not in SUPPORTED_MODELS
    """
    if model_type not in SUPPORTED_MODELS:
        available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Available types: {available}"
        )

    model_class = SUPPORTED_MODELS[model_type]
    params = hyperparameters or {}

    logger.debug(f"Instantiating model: {model_type} with params: {params}")

    return model_class(**params)
