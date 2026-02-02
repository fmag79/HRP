"""
Model Inference API for HRP.

Provides model loading, batch prediction, and feature transformation
for production model serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional
import uuid

import numpy as np
import pandas as pd
from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.data.db import get_db
from hrp.ml.registry import ModelRegistry


@dataclass
class PredictionResult:
    """
    Result of a model prediction.

    Attributes:
        predictions: Array of predictions
        model_name: Name of the model that made predictions
        model_version: Version of the model
        prediction_timestamp: When predictions were made
        feature_count: Number of features used
        sample_count: Number of predictions
        metadata: Additional prediction metadata
    """

    predictions: np.ndarray
    model_name: str
    model_version: str
    prediction_timestamp: datetime
    feature_count: int
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """
        Convert predictions to DataFrame.

        Args:
            symbols: Optional symbol names for rows

        Returns:
            DataFrame with predictions
        """
        df = pd.DataFrame(
            {
                "prediction": self.predictions,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "timestamp": self.prediction_timestamp,
            }
        )

        if symbols is not None:
            df["symbol"] = symbols

        return df

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "predictions": self.predictions.tolist(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "feature_count": self.feature_count,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }


class ModelPredictor:
    """
    Model inference for production predictions.

    Handles model loading from MLflow Model Registry,
    feature validation, and batch prediction.

    Example:
        ```python
        from hrp.ml.inference import ModelPredictor

        predictor = ModelPredictor("momentum_strategy")
        predictions = predictor.predict(features_df)
        ```
    """

    def __init__(
        self,
        model_name: str,
        model_version: str | None = None,
        registry: ModelRegistry | None = None,
        db_path: str | None = None,
        db=None,
    ):
        """
        Initialize the model predictor.

        Args:
            model_name: Name of the registered model
            model_version: Specific version (None for production)
            registry: Model registry instance (creates new if None)
            db_path: Optional database path
            db: Optional database connection (uses default if not provided)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.registry = registry or ModelRegistry()
        self._db = db or get_db(db_path)
        self._model = None
        self._model_metadata = {}

        logger.info(f"ModelPredictor initialized for {model_name}")

    def _load_model(self) -> Any:
        """
        Load model from MLflow Model Registry.

        Returns:
            Loaded model object

        Raises:
            ValueError: If model not found or cannot be loaded
        """
        if self._model is not None:
            return self._model

        import mlflow

        # Determine model version
        if self.model_version is None:
            # Get production model
            registered = self.registry.get_production_model(self.model_name)
            if registered is None:
                raise ValueError(
                    f"No production model found for {self.model_name}. "
                    "Specify model_version or deploy to production first."
                )
            self.model_version = registered.model_version
            model_uri = f"models:/{self.model_name}/production"
        else:
            model_uri = f"models:/{self.model_name}/{self.model_version}"

        logger.info(f"Loading model from {model_uri}")
        self._model = mlflow.pyfunc.load_model(model_uri)

        # Store metadata
        self._model_metadata = {
            "model_uri": model_uri,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

        return self._model

    def _validate_features(
        self,
        features: pd.DataFrame,
    ) -> None:
        """
        Validate features match model expectations.

        Args:
            features: Feature DataFrame to validate

        Raises:
            ValueError: If features are invalid
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty")

        if features.isnull().all().all():
            raise ValueError("All feature values are null")

        # Check for sufficient samples
        if len(features) < 1:
            raise ValueError("Insufficient samples for prediction")

    def predict(
        self,
        features: pd.DataFrame,
    ) -> PredictionResult:
        """
        Generate predictions for feature data.

        Args:
            features: DataFrame of features for prediction

        Returns:
            PredictionResult with predictions and metadata

        Raises:
            ValueError: If features are invalid or model cannot be loaded
        """
        logger.info(f"Generating predictions for {self.model_name}...")

        # Load model
        model = self._load_model()

        # Validate features
        self._validate_features(features)

        # Generate predictions
        predictions = model.predict(features)

        # Handle pandas Series vs numpy array
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        prediction_timestamp = datetime.now()

        result = PredictionResult(
            predictions=predictions,
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_timestamp=prediction_timestamp,
            feature_count=len(features.columns),
            sample_count=len(features),
            metadata=self._model_metadata,
        )

        # Log predictions to performance history
        self._log_predictions(result)

        logger.info(
            f"Generated {len(predictions)} predictions for {self.model_name} "
            f"v{self.model_version}"
        )

        return result

    def predict_batch(
        self,
        symbols: list[str],
        as_of_date: date,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple symbols.

        Fetches features from the database and generates predictions.

        Args:
            symbols: List of symbols to predict
            as_of_date: Date to fetch features for
            feature_names: Optional list of feature names (uses model features if None)

        Returns:
            DataFrame with columns: symbol, date, prediction, model_name, model_version

        Raises:
            ValueError: If symbols are invalid or features cannot be fetched
        """
        logger.info(f"Batch prediction for {len(symbols)} symbols as of {as_of_date}...")

        # Load model to get expected features
        model = self._load_model()

        # Fetch features from database
        api = PlatformAPI(db_path=self._db.db_path if hasattr(self._db, 'db_path') else None)

        # Use model's expected features if not specified
        if feature_names is None:
            # Try to get features from model metadata
            feature_names = self._model_metadata.get("features", [])

        if not feature_names:
            raise ValueError(
                "Feature names not specified and not available in model metadata. "
                "Provide feature_names or ensure model was registered with feature list."
            )

        # Fetch features
        features_df = api.get_features(
            symbols=symbols,
            features=feature_names,
            as_of_date=as_of_date,
        )

        if features_df.empty:
            logger.warning(f"No features found for {len(symbols)} symbols as of {as_of_date}")
            return pd.DataFrame(
                columns=["symbol", "date", "prediction", "model_name", "model_version"]
            )

        # Generate predictions
        prediction_result = self.predict(features_df)

        # Create result DataFrame
        result = pd.DataFrame(
            {
                "symbol": symbols[: len(prediction_result.predictions)],
                "date": as_of_date,
                "prediction": prediction_result.predictions,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }
        )

        logger.info(f"Batch prediction complete: {len(result)} predictions generated")
        return result

    def _log_predictions(self, result: PredictionResult) -> None:
        """Log predictions to performance history database."""
        try:
            # Log aggregate statistics
            query = """
                INSERT INTO model_performance_history (
                    model_name,
                    model_version,
                    timestamp,
                    metric_name,
                    metric_value,
                    sample_size
                ) VALUES (?, ?, ?, ?, ?, ?)
            """

            # Log prediction statistics
            predictions = result.predictions
            self._db.execute(
                query,
                (
                    result.model_name,
                    result.model_version,
                    result.prediction_timestamp,
                    "prediction_mean",
                    float(np.mean(predictions)),
                    result.sample_count,
                ),
            )

            self._db.execute(
                query,
                (
                    result.model_name,
                    result.model_version,
                    result.prediction_timestamp,
                    "prediction_std",
                    float(np.std(predictions)),
                    result.sample_count,
                ),
            )

            self._db.execute(
                query,
                (
                    result.model_name,
                    result.model_version,
                    result.prediction_timestamp,
                    "prediction_count",
                    float(result.sample_count),
                    result.sample_count,
                ),
            )

        except Exception as e:
            # Table might not exist yet
            logger.warning(f"Failed to log predictions to database: {e}")

    def get_prediction_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get prediction history for this model.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum records to return

        Returns:
            DataFrame with prediction history
        """
        query = """
            SELECT
                timestamp,
                metric_name,
                metric_value,
                sample_size
            FROM model_performance_history
            WHERE model_name = ?
              AND model_version = ?
        """

        params = [self.model_name, self.model_version]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            return self._db.fetchdf(query, params)
        except Exception:
            # Table might not exist yet
            return pd.DataFrame()


def get_model_predictor(
    model_name: str,
    model_version: str | None = None,
    registry: ModelRegistry | None = None,
    db_path: str | None = None,
) -> ModelPredictor:
    """
    Get a ModelPredictor instance.

    Args:
        model_name: Name of the registered model
        model_version: Specific version (None for production)
        registry: Model registry instance
        db_path: Optional database path

    Returns:
        ModelPredictor instance
    """
    return ModelPredictor(
        model_name=model_name,
        model_version=model_version,
        registry=registry,
        db_path=db_path,
    )
