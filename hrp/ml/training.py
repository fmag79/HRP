"""
ML Training Pipeline.

Provides data loading, feature selection, and model training
functionality for the HRP ML framework.

Usage:
    from hrp.ml.training import train_model, load_training_data
    from hrp.ml.models import MLConfig

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

    result = train_model(config, symbols=["AAPL", "MSFT"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hrp.data.db import get_db
from hrp.ml.models import MLConfig, get_model


@dataclass
class TrainingResult:
    """
    Result of model training.

    Attributes:
        model: Trained model object
        config: MLConfig used for training
        metrics: Dictionary of evaluation metrics (train_mse, val_mse, test_mse, etc.)
        feature_importance: Feature importance scores (if available from model)
        selected_features: List of features used after feature selection
    """

    model: Any
    config: MLConfig
    metrics: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)


def _fetch_features(
    symbols: list[str],
    features: list[str],
    start_date: date,
    end_date: date,
    target: str,
    db=None,
) -> pd.DataFrame:
    """
    Fetch features and target from the database.

    Args:
        symbols: List of stock symbols
        features: List of feature names to fetch
        start_date: Start date for data
        end_date: End date for data
        target: Target variable name

    Returns:
        DataFrame with MultiIndex (date, symbol) and columns for features and target
    """
    db = db or get_db()

    # Build list of all feature columns needed (features + target)
    all_features = list(set(features + [target]))
    features_str = ",".join(f"'{f}'" for f in all_features)
    symbols_str = ",".join(f"'{s}'" for s in symbols)

    query = f"""
        SELECT symbol, date, feature_name, value
        FROM features
        WHERE symbol IN ({symbols_str})
          AND date >= ?
          AND date <= ?
          AND feature_name IN ({features_str})
        ORDER BY date, symbol, feature_name
    """

    df = db.fetchdf(query, (start_date, end_date))

    if df.empty:
        logger.warning(
            f"No features found for {symbols} from {start_date} to {end_date}"
        )
        # Return empty DataFrame with correct structure
        index = pd.MultiIndex.from_product(
            [pd.DatetimeIndex([]), symbols],
            names=["date", "symbol"]
        )
        return pd.DataFrame(index=index, columns=all_features)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Pivot to get features as columns
    result = df.pivot_table(
        index=["date", "symbol"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    )

    # Ensure all requested features are present
    for feature_name in all_features:
        if feature_name not in result.columns:
            result[feature_name] = np.nan

    logger.debug(
        f"Fetched {len(result)} rows with {len(all_features)} features "
        f"for {len(symbols)} symbols"
    )

    return result


def load_training_data(
    config: MLConfig,
    symbols: list[str],
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load and split data into train/validation/test sets.

    Args:
        config: MLConfig with date ranges and feature specifications
        symbols: List of stock symbols to include

    Returns:
        Dictionary with keys:
            - X_train, y_train: Training features and target
            - X_val, y_val: Validation features and target
            - X_test, y_test: Test features and target
    """
    logger.info(
        f"Loading training data for {len(symbols)} symbols, "
        f"{len(config.features)} features, target={config.target}"
    )

    # Fetch features from database
    df = _fetch_features(
        symbols=symbols,
        features=config.features,
        start_date=config.train_start,
        end_date=config.test_end,
        target=config.target,
    )

    if df.empty:
        raise ValueError(
            f"No data found for symbols {symbols} "
            f"from {config.train_start} to {config.test_end}"
        )

    # Drop rows with NaN values
    df_clean = df.dropna()

    if df_clean.empty:
        raise ValueError("All data contains NaN values after cleaning")

    # Split by date ranges
    def split_by_date(data: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Filter DataFrame by date range (MultiIndex level 0 is date)."""
        mask = (data.index.get_level_values("date").date >= start) & \
               (data.index.get_level_values("date").date <= end)
        return data.loc[mask]

    train_df = split_by_date(df_clean, config.train_start, config.train_end)
    val_df = split_by_date(df_clean, config.validation_start, config.validation_end)
    test_df = split_by_date(df_clean, config.test_start, config.test_end)

    # Separate features and target
    feature_cols = config.features
    target_col = config.target

    result = {
        "X_train": train_df[feature_cols],
        "y_train": train_df[target_col],
        "X_val": val_df[feature_cols],
        "y_val": val_df[target_col],
        "X_test": test_df[feature_cols],
        "y_test": test_df[target_col],
    }

    logger.info(
        f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows"
    )

    return result


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int,
) -> list[str]:
    """
    Select top features using mutual information regression.

    Args:
        X: Feature DataFrame
        y: Target Series
        max_features: Maximum number of features to select

    Returns:
        List of selected feature names, ordered by importance
    """
    if len(X.columns) <= max_features:
        logger.debug(
            f"Feature selection: {len(X.columns)} features <= max_features ({max_features}), "
            f"keeping all"
        )
        return list(X.columns)

    # Handle NaN values
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    if len(X_clean) == 0:
        logger.warning("No valid data for feature selection, returning all features")
        return list(X.columns)

    logger.debug(
        f"Selecting top {max_features} features from {len(X.columns)} using mutual information"
    )

    # Compute mutual information scores
    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)

    # Create feature importance ranking
    feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Select top features
    selected = feature_scores.head(max_features).index.tolist()

    logger.info(
        f"Feature selection: selected {len(selected)} features from {len(X.columns)}"
    )
    logger.debug(f"Top features: {selected[:5]}")

    return selected


def train_model(
    config: MLConfig,
    symbols: list[str],
    hypothesis_id: str | None = None,
    log_to_mlflow: bool = False,
) -> TrainingResult:
    """
    Train a model according to the configuration.

    Args:
        config: MLConfig specifying model type, features, and date ranges
        symbols: List of stock symbols to train on
        hypothesis_id: Optional hypothesis ID (enables test set guard to prevent overfitting)
        log_to_mlflow: Whether to log experiment to MLflow (default False)

    Returns:
        TrainingResult with trained model, config, metrics, and feature importance
        
    Raises:
        OverfittingError: If hypothesis_id provided and test set evaluation limit exceeded
    """
    logger.info(
        f"Training {config.model_type} model on {len(symbols)} symbols "
        f"with {len(config.features)} features"
    )
    
    if hypothesis_id:
        logger.info(f"Test set guard enabled for hypothesis: {hypothesis_id}")

    # Load data
    data = load_training_data(config, symbols)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Feature selection (if enabled)
    selected_features = list(config.features)
    if config.feature_selection and len(config.features) > config.max_features:
        selected_features = select_features(X_train, y_train, config.max_features)
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

    # Validate feature count (overfitting guard)
    from hrp.risk.overfitting import FeatureCountValidator, TargetLeakageValidator, OverfittingError

    feature_validator = FeatureCountValidator()
    feature_result = feature_validator.check(
        feature_count=len(selected_features),
        sample_count=len(X_train),
    )
    if not feature_result.passed:
        raise OverfittingError(feature_result.message)
    if feature_result.warning:
        logger.warning(f"Feature count warning: {feature_result.message}")

    # Check for target leakage (overfitting guard)
    leakage_validator = TargetLeakageValidator()
    leakage_result = leakage_validator.check(X_train, y_train)
    if not leakage_result.passed:
        raise OverfittingError(f"Target leakage detected: {leakage_result.message}")
    if leakage_result.warning:
        logger.warning(f"Potential leakage warning: {leakage_result.message}")

    # Instantiate and train model
    model = get_model(config.model_type, config.hyperparameters)

    logger.debug(f"Fitting model on {len(X_train)} training samples")
    model.fit(X_train, y_train)

    # Generate predictions for train and validation (always allowed)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate train and validation metrics
    metrics = {
        "train_mse": float(mean_squared_error(y_train, y_train_pred)),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "val_mse": float(mean_squared_error(y_val, y_val_pred)),
        "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
        "val_r2": float(r2_score(y_val, y_val_pred)),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
        "n_features": len(selected_features),
    }

    # Test set evaluation (guarded if hypothesis_id provided)
    if hypothesis_id:
        from hrp.risk.overfitting import TestSetGuard
        
        guard = TestSetGuard(hypothesis_id)
        
        with guard.evaluate(metadata={
            "model_type": config.model_type,
            "n_features": len(selected_features),
            "n_symbols": len(symbols),
        }):
            y_test_pred = model.predict(X_test)
            metrics["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
            metrics["test_mae"] = float(mean_absolute_error(y_test, y_test_pred))
            metrics["test_r2"] = float(r2_score(y_test, y_test_pred))
        
        logger.info(
            f"Test set evaluation {guard.evaluation_count}/{guard.max_evaluations} "
            f"for {hypothesis_id}"
        )
    else:
        # Unguarded evaluation (for ad-hoc experiments without hypothesis)
        y_test_pred = model.predict(X_test)
        metrics["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
        metrics["test_mae"] = float(mean_absolute_error(y_test, y_test_pred))
        metrics["test_r2"] = float(r2_score(y_test, y_test_pred))
        
        logger.warning(
            "Test set evaluated without guard (no hypothesis_id provided). "
            "Consider using hypothesis_id to prevent overfitting."
        )

    logger.info(
        f"Model trained: train_mse={metrics['train_mse']:.6f}, "
        f"val_mse={metrics['val_mse']:.6f}, test_mse={metrics['test_mse']:.6f}"
    )

    # Extract feature importance (if available)
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        # Tree-based models (RandomForest, LightGBM, XGBoost)
        feature_importance = dict(zip(selected_features, model.feature_importances_))
    elif hasattr(model, "coef_"):
        # Linear models (Ridge, Lasso, ElasticNet)
        coef = model.coef_
        if isinstance(coef, np.ndarray) and coef.ndim == 1:
            feature_importance = dict(zip(selected_features, np.abs(coef)))
        else:
            # Handle multi-output case
            feature_importance = dict(zip(selected_features, np.abs(coef).mean(axis=0)))

    # Log to MLflow if requested
    if log_to_mlflow:
        _log_to_mlflow(config, model, metrics, feature_importance, selected_features)

    result = TrainingResult(
        model=model,
        config=config,
        metrics=metrics,
        feature_importance=feature_importance,
        selected_features=selected_features,
    )

    return result


def _log_to_mlflow(
    config: MLConfig,
    model: Any,
    metrics: dict[str, float],
    feature_importance: dict[str, float],
    selected_features: list[str],
) -> None:
    """
    Log training run to MLflow.

    Args:
        config: MLConfig used for training
        model: Trained model
        metrics: Evaluation metrics
        feature_importance: Feature importance scores
        selected_features: Features used for training
    """
    try:
        import mlflow

        # Set experiment name based on model type
        experiment_name = f"hrp_{config.model_type}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", config.model_type)
            mlflow.log_param("target", config.target)
            mlflow.log_param("n_features", len(selected_features))
            mlflow.log_param("train_start", str(config.train_start))
            mlflow.log_param("train_end", str(config.train_end))
            mlflow.log_param("validation_start", str(config.validation_start))
            mlflow.log_param("validation_end", str(config.validation_end))
            mlflow.log_param("test_start", str(config.test_start))
            mlflow.log_param("test_end", str(config.test_end))
            mlflow.log_param("feature_selection", config.feature_selection)
            mlflow.log_param("max_features", config.max_features)

            # Log hyperparameters
            for key, value in config.hyperparameters.items():
                mlflow.log_param(f"hp_{key}", value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"Logged training run to MLflow experiment: {experiment_name}")

    except ImportError:
        logger.warning("MLflow not installed, skipping logging")
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
