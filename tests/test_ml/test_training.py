"""Tests for ML training pipeline."""
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from hrp.ml.models import MLConfig


class TestLoadTrainingData:
    @pytest.fixture
    def sample_config(self):
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
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        np.random.seed(42)
        n = len(index)
        return pd.DataFrame({
            "momentum_20d": np.random.randn(n) * 0.1,
            "volatility_20d": np.abs(np.random.randn(n)) * 0.2,
            "returns_20d": np.random.randn(n) * 0.05,
        }, index=index)

    def test_load_training_data_splits(self, sample_config, mock_features_df):
        from hrp.ml.training import load_training_data
        with patch("hrp.ml.training._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df
            data = load_training_data(sample_config, symbols=["AAPL", "MSFT"])
        assert "X_train" in data and "y_train" in data
        assert "X_val" in data and "y_val" in data
        assert "X_test" in data and "y_test" in data

    def test_load_training_data_features(self, sample_config, mock_features_df):
        from hrp.ml.training import load_training_data
        with patch("hrp.ml.training._fetch_features") as mock_fetch:
            mock_fetch.return_value = mock_features_df
            data = load_training_data(sample_config, symbols=["AAPL", "MSFT"])
        assert list(data["X_train"].columns) == ["momentum_20d", "volatility_20d"]
        assert data["y_train"].name == "returns_20d"


class TestTrainModel:
    @pytest.fixture
    def sample_config(self):
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
            feature_selection=False,
        )

    @pytest.fixture
    def mock_data(self):
        np.random.seed(42)
        def make_data(start, periods):
            dates = pd.date_range(start, periods=periods, freq="B")
            symbols = ["AAPL", "MSFT"]
            index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
            X = pd.DataFrame(np.random.randn(len(index), 2), index=index, columns=["momentum_20d", "volatility_20d"])
            y = pd.Series(np.random.randn(len(index)) * 0.05, index=index, name="returns_20d")
            return X, y
        X_train, y_train = make_data("2020-01-01", 100)
        X_val, y_val = make_data("2021-01-01", 25)
        X_test, y_test = make_data("2021-07-01", 25)
        return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}

    def test_train_model_returns_result(self, sample_config, mock_data):
        from hrp.ml.training import train_model, TrainingResult
        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data
            result = train_model(sample_config, symbols=["AAPL", "MSFT"])
        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert "train_mse" in result.metrics
        assert "val_mse" in result.metrics

    def test_train_model_predictions_shape(self, sample_config, mock_data):
        from hrp.ml.training import train_model
        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data
            result = train_model(sample_config, symbols=["AAPL", "MSFT"])
        preds = result.model.predict(mock_data["X_val"])
        assert len(preds) == len(mock_data["y_val"])
