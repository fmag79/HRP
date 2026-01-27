# ML Framework MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable end-to-end ML workflow - train models on features, generate trading signals from predictions.

**Architecture:** Three modules following existing patterns: `models.py` (model registry with sklearn-compatible interface), `training.py` (training pipeline with MLflow logging), `signals.py` (prediction to signal conversion). All ML operations will eventually be exposed through PlatformAPI but MVP focuses on core functionality.

**Tech Stack:** scikit-learn, LightGBM, pandas, MLflow (existing integration via `hrp/research/mlflow_utils.py`)

---

## Task 1: Model Registry (`hrp/ml/models.py`)

**Files:**
- Create: `hrp/ml/models.py`
- Test: `tests/test_ml/test_models.py`

**Step 1: Write the failing test for MLConfig dataclass**

```python
# tests/test_ml/test_models.py
"""Tests for ML model registry."""

from datetime import date

import pytest

from hrp.ml.models import MLConfig, SUPPORTED_MODELS


class TestMLConfig:
    """Tests for MLConfig dataclass."""

    def test_mlconfig_creation(self):
        """Test creating MLConfig with required fields."""
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
        assert config.model_type == "ridge"
        assert config.target == "returns_20d"
        assert len(config.features) == 2

    def test_mlconfig_defaults(self):
        """Test MLConfig default values."""
        config = MLConfig(
            model_type="lightgbm",
            target="returns_5d",
            features=["momentum_20d"],
            train_start=date(2015, 1, 1),
            train_end=date(2018, 12, 31),
            validation_start=date(2019, 1, 1),
            validation_end=date(2019, 12, 31),
            test_start=date(2020, 1, 1),
            test_end=date(2023, 12, 31),
        )
        assert config.feature_selection is True
        assert config.max_features == 20
        assert config.hyperparameters == {}

    def test_mlconfig_invalid_model_type(self):
        """Test MLConfig rejects invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            MLConfig(
                model_type="invalid_model",
                target="returns_20d",
                features=["momentum_20d"],
                train_start=date(2015, 1, 1),
                train_end=date(2018, 12, 31),
                validation_start=date(2019, 1, 1),
                validation_end=date(2019, 12, 31),
                test_start=date(2020, 1, 1),
                test_end=date(2023, 12, 31),
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_models.py::TestMLConfig -v`
Expected: FAIL with "cannot import name 'MLConfig'"

**Step 3: Write minimal implementation for MLConfig**

```python
# hrp/ml/models.py
"""
ML model registry for HRP.

Provides supported models and configuration for ML training pipeline.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None


# Supported model types and their classes
SUPPORTED_MODELS: dict[str, type] = {
    # Linear models
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    # Tree-based
    "random_forest": RandomForestRegressor,
    # Neural network
    "mlp": MLPRegressor,
}

# Add optional models if available
if HAS_LIGHTGBM:
    SUPPORTED_MODELS["lightgbm"] = LGBMRegressor
if HAS_XGBOOST:
    SUPPORTED_MODELS["xgboost"] = XGBRegressor


@dataclass
class MLConfig:
    """
    Configuration for ML model training.

    Attributes:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        target: Target variable name (e.g., 'returns_20d')
        features: List of feature names from feature store
        train_start: Training period start date
        train_end: Training period end date
        validation_start: Validation period start date
        validation_end: Validation period end date
        test_start: Test period start date
        test_end: Test period end date
        hyperparameters: Model-specific hyperparameters
        feature_selection: Whether to apply automatic feature selection
        max_features: Maximum features to select (prevents overfitting)
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

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in SUPPORTED_MODELS:
            available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            raise ValueError(
                f"Unsupported model type: '{self.model_type}'. "
                f"Available: {available}"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_models.py::TestMLConfig -v`
Expected: PASS

**Step 5: Write tests for get_model function**

```python
# Add to tests/test_ml/test_models.py

class TestGetModel:
    """Tests for get_model function."""

    def test_get_ridge_model(self):
        """Test getting a Ridge model."""
        from hrp.ml.models import get_model

        model = get_model("ridge")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_ridge_with_hyperparams(self):
        """Test getting a Ridge model with hyperparameters."""
        from hrp.ml.models import get_model

        model = get_model("ridge", {"alpha": 0.5})
        assert model.alpha == 0.5

    def test_get_lightgbm_model(self):
        """Test getting a LightGBM model if available."""
        from hrp.ml.models import get_model, HAS_LIGHTGBM

        if not HAS_LIGHTGBM:
            pytest.skip("LightGBM not installed")

        model = get_model("lightgbm")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_invalid_model(self):
        """Test that invalid model type raises error."""
        from hrp.ml.models import get_model

        with pytest.raises(ValueError, match="Unsupported model type"):
            get_model("invalid_model")
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_ml/test_models.py::TestGetModel -v`
Expected: FAIL with "cannot import name 'get_model'"

**Step 7: Implement get_model function**

```python
# Add to hrp/ml/models.py after MLConfig class

def get_model(model_type: str, hyperparameters: dict[str, Any] | None = None):
    """
    Get an instantiated model by type.

    Args:
        model_type: Type of model (must be in SUPPORTED_MODELS)
        hyperparameters: Optional dict of hyperparameters

    Returns:
        Instantiated sklearn-compatible model

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in SUPPORTED_MODELS:
        available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(
            f"Unsupported model type: '{model_type}'. Available: {available}"
        )

    model_class = SUPPORTED_MODELS[model_type]
    params = hyperparameters or {}

    # Apply sensible defaults for certain models
    if model_type == "lightgbm" and "verbose" not in params:
        params["verbose"] = -1  # Suppress output
    if model_type == "xgboost" and "verbosity" not in params:
        params["verbosity"] = 0  # Suppress output
    if model_type == "random_forest" and "n_jobs" not in params:
        params["n_jobs"] = -1  # Use all cores

    logger.debug(f"Creating {model_type} model with params: {params}")
    return model_class(**params)
```

**Step 8: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_models.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add hrp/ml/models.py tests/test_ml/test_models.py
git commit -m "feat(ml): add model registry with MLConfig and get_model"
```

---

## Task 2: Signal Generation (`hrp/ml/signals.py`)

**Files:**
- Create: `hrp/ml/signals.py`
- Test: `tests/test_ml/test_signals.py`

**Step 1: Write failing tests for signal generation**

```python
# tests/test_ml/test_signals.py
"""Tests for ML signal generation."""

import numpy as np
import pandas as pd
import pytest

from hrp.ml.signals import predictions_to_signals


class TestPredictionsToSignals:
    """Tests for predictions_to_signals function."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions DataFrame."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "AAPL": [0.05, 0.02, -0.01, 0.08, 0.03],
                "MSFT": [0.03, 0.06, 0.01, 0.02, 0.07],
                "GOOGL": [-0.02, 0.04, 0.09, 0.01, -0.03],
                "AMZN": [0.01, 0.01, 0.02, 0.05, 0.04],
            },
            index=dates,
        )

    def test_rank_method_top_decile(self, sample_predictions):
        """Test rank method selects top performers."""
        signals = predictions_to_signals(sample_predictions, method="rank", top_pct=0.25)

        # With 4 symbols and top 25%, should select 1 per day
        for date_idx in signals.index:
            assert signals.loc[date_idx].sum() == 1.0

    def test_rank_method_shape_preserved(self, sample_predictions):
        """Test output shape matches input."""
        signals = predictions_to_signals(sample_predictions, method="rank")
        assert signals.shape == sample_predictions.shape
        assert list(signals.columns) == list(sample_predictions.columns)

    def test_threshold_method(self, sample_predictions):
        """Test threshold method."""
        signals = predictions_to_signals(
            sample_predictions, method="threshold", threshold=0.05
        )

        # Check specific values
        assert signals.loc["2023-01-01", "AAPL"] == 1.0  # 0.05 >= 0.05
        assert signals.loc["2023-01-01", "MSFT"] == 0.0  # 0.03 < 0.05

    def test_zscore_method(self, sample_predictions):
        """Test zscore method returns normalized values."""
        signals = predictions_to_signals(sample_predictions, method="zscore")

        # Z-scores should have mean ~0 and std ~1 per row
        for date_idx in signals.index:
            row = signals.loc[date_idx]
            assert abs(row.mean()) < 0.01  # Close to 0
            assert abs(row.std() - 1.0) < 0.01  # Close to 1

    def test_invalid_method(self, sample_predictions):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            predictions_to_signals(sample_predictions, method="invalid")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_signals.py -v`
Expected: FAIL with "cannot import name 'predictions_to_signals'"

**Step 3: Implement signals module**

```python
# hrp/ml/signals.py
"""
Signal generation from ML predictions.

Converts raw model predictions to actionable trading signals.
"""

import numpy as np
import pandas as pd
from loguru import logger


def predictions_to_signals(
    predictions: pd.DataFrame,
    method: str = "rank",
    top_pct: float = 0.1,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Convert raw predictions to actionable trading signals.

    Args:
        predictions: DataFrame with predictions (index=dates, columns=symbols)
        method: Signal generation method
            - 'rank': Cross-sectional rank, go long top percentile
            - 'threshold': Go long if prediction > threshold
            - 'zscore': Normalize predictions cross-sectionally
        top_pct: For rank method, top percentage to select (default 10%)
        threshold: For threshold method, minimum prediction value

    Returns:
        DataFrame of signals (same shape as predictions)
        - For 'rank' and 'threshold': 1.0 = long, 0.0 = no position
        - For 'zscore': continuous signal values

    Raises:
        ValueError: If method is unknown
    """
    if method == "rank":
        return _rank_signals(predictions, top_pct)
    elif method == "threshold":
        return _threshold_signals(predictions, threshold)
    elif method == "zscore":
        return _zscore_signals(predictions)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. Valid methods: rank, threshold, zscore"
        )


def _rank_signals(predictions: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """
    Generate signals based on cross-sectional ranking.

    Selects top percentile of predictions each period.
    """
    # Rank within each row (cross-sectional)
    ranks = predictions.rank(axis=1, pct=True)

    # Select top percentile
    cutoff = 1.0 - top_pct
    signals = (ranks > cutoff).astype(float)

    logger.debug(
        f"Rank signals: top {top_pct*100:.0f}%, "
        f"avg positions per day: {signals.sum(axis=1).mean():.1f}"
    )

    return signals


def _threshold_signals(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Generate signals based on absolute threshold.

    Goes long when prediction exceeds threshold.
    """
    signals = (predictions >= threshold).astype(float)

    logger.debug(
        f"Threshold signals: >= {threshold}, "
        f"avg positions per day: {signals.sum(axis=1).mean():.1f}"
    )

    return signals


def _zscore_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals based on cross-sectional z-score.

    Returns continuous signal proportional to relative strength.
    """
    # Z-score within each row (cross-sectional)
    row_mean = predictions.mean(axis=1)
    row_std = predictions.std(axis=1)

    # Avoid division by zero
    row_std = row_std.replace(0, 1)

    signals = predictions.sub(row_mean, axis=0).div(row_std, axis=0)

    logger.debug(
        f"Z-score signals: range [{signals.min().min():.2f}, {signals.max().max():.2f}]"
    )

    return signals
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_signals.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add hrp/ml/signals.py tests/test_ml/test_signals.py
git commit -m "feat(ml): add signal generation from predictions"
```

---

## Task 3: Training Pipeline (`hrp/ml/training.py`)

**Files:**
- Create: `hrp/ml/training.py`
- Test: `tests/test_ml/test_training.py`

**Step 1: Write failing tests for feature loading**

```python
# tests/test_ml/test_training.py
"""Tests for ML training pipeline."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.ml.models import MLConfig


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample MLConfig."""
        return MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
        )

    @pytest.fixture
    def mock_features_df(self):
        """Create mock features DataFrame."""
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product(
            [dates, symbols], names=["date", "symbol"]
        )

        np.random.seed(42)
        n = len(index)
        return pd.DataFrame(
            {
                "momentum_20d": np.random.randn(n) * 0.1,
                "volatility_20d": np.abs(np.random.randn(n)) * 0.2,
                "returns_20d": np.random.randn(n) * 0.05,
            },
            index=index,
        )

    def test_load_training_data_splits(self, sample_config, mock_features_df):
        """Test that data is split correctly into train/val/test."""
        from hrp.ml.training import load_training_data

        with patch("hrp.ml.training._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            data = load_training_data(sample_config, symbols=["AAPL", "MSFT"])

        assert "X_train" in data
        assert "y_train" in data
        assert "X_val" in data
        assert "y_val" in data
        assert "X_test" in data
        assert "y_test" in data

        # Check date ranges
        assert data["X_train"].index.get_level_values("date").min() >= pd.Timestamp(
            sample_config.train_start
        )
        assert data["X_train"].index.get_level_values("date").max() <= pd.Timestamp(
            sample_config.train_end
        )

    def test_load_training_data_features(self, sample_config, mock_features_df):
        """Test that only requested features are included."""
        from hrp.ml.training import load_training_data

        with patch("hrp.ml.training._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            data = load_training_data(sample_config, symbols=["AAPL", "MSFT"])

        # X should have only the features, not the target
        assert list(data["X_train"].columns) == ["momentum_20d", "volatility_20d"]
        # y should be a Series with the target
        assert data["y_train"].name == "returns_20d"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_training.py::TestLoadTrainingData -v`
Expected: FAIL with "cannot import name 'load_training_data'"

**Step 3: Implement load_training_data function**

```python
# hrp/ml/training.py
"""
ML training pipeline for HRP.

Handles data loading, model training, and MLflow logging.
"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.ml.models import MLConfig, get_model


def _fetch_features(
    symbols: list[str],
    features: list[str],
    start_date: date,
    end_date: date,
    target: str,
) -> pd.DataFrame:
    """
    Fetch features and target from database.

    Returns DataFrame with MultiIndex (date, symbol) and feature columns.
    """
    db = get_db()

    # Include target in features to fetch
    all_features = list(set(features + [target]))
    features_str = ",".join(f"'{f}'" for f in all_features)
    symbols_str = ",".join(f"'{s}'" for s in symbols)

    query = f"""
        SELECT symbol, date, feature_name, value
        FROM features
        WHERE symbol IN ({symbols_str})
          AND feature_name IN ({features_str})
          AND date >= ?
          AND date <= ?
        ORDER BY date, symbol, feature_name
    """

    df = db.fetchdf(query, (start_date, end_date))

    if df.empty:
        raise ValueError(
            f"No features found for {symbols} from {start_date} to {end_date}"
        )

    # Pivot to wide format
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(
        index=["date", "symbol"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    )

    return pivot


def load_training_data(
    config: MLConfig,
    symbols: list[str],
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load and split training data according to config.

    Args:
        config: ML configuration with date ranges
        symbols: List of symbols to include

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    logger.info(
        f"Loading training data for {len(symbols)} symbols, "
        f"features: {config.features}, target: {config.target}"
    )

    # Fetch all data at once (train start to test end)
    all_data = _fetch_features(
        symbols=symbols,
        features=config.features,
        start_date=config.train_start,
        end_date=config.test_end,
        target=config.target,
    )

    # Drop rows with missing values
    all_data = all_data.dropna()

    # Extract features and target
    X = all_data[config.features]
    y = all_data[config.target]
    y.name = config.target

    # Split by date
    dates = X.index.get_level_values("date")

    train_mask = (dates >= pd.Timestamp(config.train_start)) & (
        dates <= pd.Timestamp(config.train_end)
    )
    val_mask = (dates >= pd.Timestamp(config.validation_start)) & (
        dates <= pd.Timestamp(config.validation_end)
    )
    test_mask = (dates >= pd.Timestamp(config.test_start)) & (
        dates <= pd.Timestamp(config.test_end)
    )

    result = {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
    }

    logger.info(
        f"Data split: train={len(result['X_train'])}, "
        f"val={len(result['X_val'])}, test={len(result['X_test'])}"
    )

    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_training.py::TestLoadTrainingData -v`
Expected: All PASS

**Step 5: Write failing tests for train_model function**

```python
# Add to tests/test_ml/test_training.py

class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample MLConfig."""
        return MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=False,  # Disable for simple test
        )

    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        np.random.seed(42)
        n_train, n_val, n_test = 200, 50, 50
        n_features = 2

        dates_train = pd.date_range("2020-01-01", periods=n_train // 2, freq="B")
        dates_val = pd.date_range("2021-01-01", periods=n_val // 2, freq="B")
        dates_test = pd.date_range("2021-07-01", periods=n_test // 2, freq="B")

        def make_data(dates, n):
            symbols = ["AAPL", "MSFT"]
            index = pd.MultiIndex.from_product(
                [dates, symbols], names=["date", "symbol"]
            )
            X = pd.DataFrame(
                np.random.randn(len(index), n_features),
                index=index,
                columns=["momentum_20d", "volatility_20d"],
            )
            y = pd.Series(np.random.randn(len(index)) * 0.05, index=index, name="returns_20d")
            return X, y

        X_train, y_train = make_data(dates_train, n_features)
        X_val, y_val = make_data(dates_val, n_features)
        X_test, y_test = make_data(dates_test, n_features)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def test_train_model_returns_result(self, sample_config, mock_data):
        """Test train_model returns TrainingResult."""
        from hrp.ml.training import train_model, TrainingResult

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data

            result = train_model(sample_config, symbols=["AAPL", "MSFT"])

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert "train_mse" in result.metrics
        assert "val_mse" in result.metrics

    def test_train_model_predictions_shape(self, sample_config, mock_data):
        """Test model can make predictions."""
        from hrp.ml.training import train_model

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data

            result = train_model(sample_config, symbols=["AAPL", "MSFT"])

        # Make predictions on validation set
        preds = result.model.predict(mock_data["X_val"])
        assert len(preds) == len(mock_data["y_val"])
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_ml/test_training.py::TestTrainModel -v`
Expected: FAIL with "cannot import name 'train_model'"

**Step 7: Implement train_model function and TrainingResult**

```python
# Add to hrp/ml/training.py after load_training_data

from dataclasses import dataclass, field
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression


@dataclass
class TrainingResult:
    """Result from model training."""

    model: Any
    config: MLConfig
    metrics: dict[str, float]
    feature_importance: dict[str, float] | None = None
    selected_features: list[str] | None = None

    @property
    def train_mse(self) -> float:
        return self.metrics.get("train_mse", float("nan"))

    @property
    def val_mse(self) -> float:
        return self.metrics.get("val_mse", float("nan"))


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 20,
) -> list[str]:
    """
    Select top features using mutual information.

    Args:
        X: Feature DataFrame
        y: Target Series
        max_features: Maximum number of features to select

    Returns:
        List of selected feature names
    """
    if len(X.columns) <= max_features:
        return list(X.columns)

    logger.info(f"Selecting top {max_features} from {len(X.columns)} features")

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # Get top features
    top_indices = np.argsort(mi_scores)[-max_features:]
    selected = [X.columns[i] for i in top_indices]

    logger.info(f"Selected features: {selected}")
    return selected


def train_model(
    config: MLConfig,
    symbols: list[str],
    log_to_mlflow: bool = False,
) -> TrainingResult:
    """
    Train a model according to configuration.

    Args:
        config: ML configuration
        symbols: List of symbols to train on
        log_to_mlflow: Whether to log to MLflow

    Returns:
        TrainingResult with trained model and metrics
    """
    logger.info(f"Training {config.model_type} model on {len(symbols)} symbols")

    # Load data
    data = load_training_data(config, symbols)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Feature selection (on training data only)
    selected_features = None
    if config.feature_selection and len(config.features) > config.max_features:
        selected_features = select_features(X_train, y_train, config.max_features)
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
    else:
        selected_features = list(config.features)

    # Create and train model
    model = get_model(config.model_type, config.hyperparameters)
    model.fit(X_train, y_train)

    # Calculate metrics
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    metrics = {
        "train_mse": mean_squared_error(y_train, train_preds),
        "train_mae": mean_absolute_error(y_train, train_preds),
        "train_r2": r2_score(y_train, train_preds),
        "val_mse": mean_squared_error(y_val, val_preds),
        "val_mae": mean_absolute_error(y_val, val_preds),
        "val_r2": r2_score(y_val, val_preds),
        "n_train_samples": len(y_train),
        "n_val_samples": len(y_val),
        "n_features": len(selected_features),
    }

    # Get feature importance if available
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(selected_features, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coef = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
        feature_importance = dict(zip(selected_features, np.abs(coef)))

    logger.info(
        f"Training complete: train_mse={metrics['train_mse']:.6f}, "
        f"val_mse={metrics['val_mse']:.6f}"
    )

    return TrainingResult(
        model=model,
        config=config,
        metrics=metrics,
        feature_importance=feature_importance,
        selected_features=selected_features,
    )
```

**Step 8: Run tests to verify they pass**

Run: `pytest tests/test_ml/test_training.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add hrp/ml/training.py tests/test_ml/test_training.py
git commit -m "feat(ml): add training pipeline with data loading and model training"
```

---

## Task 4: Update ML module exports

**Files:**
- Modify: `hrp/ml/__init__.py`

**Step 1: Update __init__.py to export public API**

```python
# hrp/ml/__init__.py
"""
ML Framework for HRP.

Provides model training, signal generation, and validation.
"""

from hrp.ml.models import MLConfig, SUPPORTED_MODELS, get_model
from hrp.ml.signals import predictions_to_signals
from hrp.ml.training import TrainingResult, train_model, load_training_data, select_features

__all__ = [
    # Models
    "MLConfig",
    "SUPPORTED_MODELS",
    "get_model",
    # Signals
    "predictions_to_signals",
    # Training
    "TrainingResult",
    "train_model",
    "load_training_data",
    "select_features",
]
```

**Step 2: Run all ML tests**

Run: `pytest tests/test_ml/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add hrp/ml/__init__.py
git commit -m "feat(ml): export public API from ml module"
```

---

## Task 5: Integration test

**Files:**
- Test: `tests/test_ml/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_ml/test_integration.py
"""Integration tests for ML framework."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hrp.ml import MLConfig, train_model, predictions_to_signals


class TestMLIntegration:
    """End-to-end ML workflow tests."""

    @pytest.fixture
    def mock_features_df(self):
        """Create realistic mock features DataFrame."""
        # 2 years of daily data for 5 symbols
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        index = pd.MultiIndex.from_product(
            [dates, symbols], names=["date", "symbol"]
        )

        np.random.seed(42)
        n = len(index)

        # Create features with some signal
        momentum = np.random.randn(n) * 0.1
        volatility = np.abs(np.random.randn(n)) * 0.2
        # Target has weak correlation with momentum
        target = 0.1 * momentum + np.random.randn(n) * 0.05

        return pd.DataFrame(
            {
                "momentum_20d": momentum,
                "volatility_20d": volatility,
                "returns_20d": target,
            },
            index=index,
        )

    def test_train_and_generate_signals(self, mock_features_df):
        """Test full workflow: train model, make predictions, generate signals."""
        config = MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=False,
        )

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        with patch("hrp.ml.training._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df

            # Train model
            result = train_model(config, symbols=symbols)

        assert result.model is not None
        assert result.val_mse > 0

        # Make predictions on test data
        test_mask = (
            mock_features_df.index.get_level_values("date")
            >= pd.Timestamp(config.test_start)
        ) & (
            mock_features_df.index.get_level_values("date")
            <= pd.Timestamp(config.test_end)
        )
        X_test = mock_features_df.loc[test_mask, config.features]
        predictions = result.model.predict(X_test)

        # Reshape predictions to DataFrame for signal generation
        test_dates = X_test.index.get_level_values("date").unique()
        pred_df = pd.DataFrame(
            predictions.reshape(len(test_dates), len(symbols)),
            index=test_dates,
            columns=symbols,
        )

        # Generate signals
        signals = predictions_to_signals(pred_df, method="rank", top_pct=0.2)

        # With 5 symbols and top 20%, should select 1 per day
        assert signals.shape == pred_df.shape
        assert (signals.sum(axis=1) == 1).all()
```

**Step 2: Run integration test**

Run: `pytest tests/test_ml/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ml/test_integration.py
git commit -m "test(ml): add integration test for ML workflow"
```

---

## Task 6: Update spec to mark Phase 5 MVP complete

**Files:**
- Modify: `docs/plans/2025-01-19-hrp-spec.md`

**Step 1: Update Phase 5 section**

Mark the MVP deliverables as complete:
- [x] ML model registry (`hrp/ml/models.py`)
- [x] Training pipeline (`hrp/ml/training.py`)
- [x] Walk-forward validation (`hrp/ml/validation.py`) - ✅ COMPLETE (expanding/rolling windows, stability score)
- [x] Feature selection (in `hrp/ml/training.py`)
- [x] Signal generation from predictions (`hrp/ml/signals.py`)
- [x] Overfitting guards (test set discipline) - ✅ COMPLETE (TestSetGuard, validation gates, robustness checks)

**Step 2: Commit**

```bash
git add docs/plans/2025-01-19-hrp-spec.md
git commit -m "docs: mark Phase 5 MVP deliverables complete"
```

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | `hrp/ml/models.py` | Model registry with MLConfig and get_model |
| 2 | `hrp/ml/signals.py` | Prediction to signal conversion |
| 3 | `hrp/ml/training.py` | Training pipeline with data loading |
| 4 | `hrp/ml/__init__.py` | Public API exports |
| 5 | `tests/test_ml/test_integration.py` | End-to-end workflow test |
| 6 | Spec update | Documentation |

Total: ~400 lines of implementation, ~300 lines of tests.
