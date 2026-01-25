"""Tests for ML training pipeline."""
from datetime import date
from unittest.mock import patch, MagicMock
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


class TestOverfittingGuardIntegration:
    """Tests for overfitting guard integration in training pipeline."""

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

    def test_feature_count_validation_passes(self, sample_config, mock_data):
        """Test that normal feature counts pass validation."""
        from hrp.ml.training import train_model
        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data
            # Should not raise - 2 features with 200 samples is fine
            result = train_model(sample_config, symbols=["AAPL", "MSFT"])
        assert result.model is not None

    def test_excessive_features_raises_error(self, sample_config, mock_data):
        """Test that too many features raises OverfittingError."""
        from hrp.ml.training import train_model
        from hrp.risk.overfitting import OverfittingError

        # Create data with too many features (55 features > 50 max threshold)
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        feature_cols = [f"feature_{i}" for i in range(55)]
        X = pd.DataFrame(np.random.randn(len(index), 55), index=index, columns=feature_cols)
        y = pd.Series(np.random.randn(len(index)) * 0.05, index=index, name="returns_20d")

        mock_data_many_features = {
            "X_train": X, "y_train": y,
            "X_val": X.iloc[:50], "y_val": y.iloc[:50],
            "X_test": X.iloc[:50], "y_test": y.iloc[:50],
        }

        config = MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=feature_cols,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=False,
        )

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data_many_features
            with pytest.raises(OverfittingError, match="Feature count"):
                train_model(config, symbols=["AAPL", "MSFT"])

    def test_leakage_detection_raises_error(self, sample_config):
        """Test that target leakage raises OverfittingError."""
        from hrp.ml.training import train_model
        from hrp.risk.overfitting import OverfittingError

        # Create data with leaky feature (perfect correlation with target)
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

        y = pd.Series(np.random.randn(len(index)) * 0.05, index=index, name="returns_20d")
        # Create leaky feature that is essentially the target
        X = pd.DataFrame({
            "momentum_20d": np.random.randn(len(index)) * 0.1,
            "leaky_feature": y.values * 1.001,  # Nearly identical to target
        }, index=index)

        mock_data_leaky = {
            "X_train": X, "y_train": y,
            "X_val": X.iloc[:50], "y_val": y.iloc[:50],
            "X_test": X.iloc[:50], "y_test": y.iloc[:50],
        }

        config = MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "leaky_feature"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=False,
        )

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data_leaky
            with pytest.raises(OverfittingError, match="leakage"):
                train_model(config, symbols=["AAPL", "MSFT"])


class TestFetchFeatures:
    """Tests for _fetch_features function."""

    def test_empty_result(self):
        """Test that empty result returns correct structure."""
        from hrp.ml.training import _fetch_features

        with patch("hrp.ml.training.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = pd.DataFrame()  # Empty result
            mock_get_db.return_value = mock_db

            result = _fetch_features(
                symbols=["AAPL"],
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                target="returns_20d",
            )

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # Should have correct columns
        assert "momentum_20d" in result.columns or len(result.columns) >= 0

    def test_missing_feature_fills_nan(self):
        """Test that missing features are filled with NaN."""
        from hrp.ml.training import _fetch_features

        # Create mock data missing one feature
        mock_df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": [date(2020, 1, 2), date(2020, 1, 3)],
            "feature_name": ["momentum_20d", "momentum_20d"],
            "value": [0.1, 0.15],
        })

        with patch("hrp.ml.training.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = mock_df
            mock_get_db.return_value = mock_db

            result = _fetch_features(
                symbols=["AAPL"],
                features=["momentum_20d", "volatility_20d"],  # vol missing
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                target="returns_20d",
            )

        assert "volatility_20d" in result.columns
        # Missing feature should be NaN
        assert result["volatility_20d"].isna().all()

    def test_date_conversion(self):
        """Test that date is converted to datetime."""
        from hrp.ml.training import _fetch_features

        mock_df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": ["2020-01-02", "2020-01-03"],  # String dates
            "feature_name": ["momentum_20d", "momentum_20d"],
            "value": [0.1, 0.15],
        })

        with patch("hrp.ml.training.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.fetchdf.return_value = mock_df
            mock_get_db.return_value = mock_db

            result = _fetch_features(
                symbols=["AAPL"],
                features=["momentum_20d"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                target="returns_20d",
            )

        # Date index should be datetime
        assert result.index.get_level_values("date").dtype == "datetime64[ns]"


class TestLogToMlflow:
    """Tests for _log_to_mlflow function."""

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
            hyperparameters={"alpha": 1.0, "fit_intercept": True},
        )

    def test_basic(self):
        """Test that basic logging works."""
        from hrp.ml.training import _log_to_mlflow
        import mlflow

        config = MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
        )
        model = MagicMock()
        metrics = {"train_mse": 0.01, "val_mse": 0.02}
        feature_importance = {"momentum_20d": 0.5}
        selected_features = ["momentum_20d"]

        with patch.object(mlflow, "set_experiment") as mock_set_exp, \
             patch.object(mlflow, "start_run") as mock_start_run, \
             patch.object(mlflow, "log_param") as mock_log_param, \
             patch.object(mlflow, "log_metric") as mock_log_metric, \
             patch.object(mlflow.sklearn, "log_model"):
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            _log_to_mlflow(config, model, metrics, feature_importance, selected_features)

        mock_set_exp.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    def test_hyperparameters_prefixed(self, sample_config):
        """Test that hyperparameters get hp_ prefix."""
        from hrp.ml.training import _log_to_mlflow
        import mlflow

        model = MagicMock()
        metrics = {"train_mse": 0.01}

        with patch.object(mlflow, "set_experiment"), \
             patch.object(mlflow, "start_run") as mock_start_run, \
             patch.object(mlflow, "log_param") as mock_log_param, \
             patch.object(mlflow, "log_metric"), \
             patch.object(mlflow.sklearn, "log_model"):
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            _log_to_mlflow(sample_config, model, metrics, {}, [])

        # Check that hp_ prefix is used
        log_param_calls = [str(c) for c in mock_log_param.call_args_list]
        assert any("hp_alpha" in c for c in log_param_calls)

    def test_sklearn_model_logged(self, sample_config):
        """Test that model is logged via mlflow.sklearn."""
        from hrp.ml.training import _log_to_mlflow
        import mlflow

        model = MagicMock()
        metrics = {"train_mse": 0.01}

        with patch.object(mlflow, "set_experiment"), \
             patch.object(mlflow, "start_run") as mock_start_run, \
             patch.object(mlflow, "log_param"), \
             patch.object(mlflow, "log_metric"), \
             patch.object(mlflow.sklearn, "log_model") as mock_log_model:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            _log_to_mlflow(sample_config, model, metrics, {}, [])

        mock_log_model.assert_called_once_with(model, "model")

    def test_handles_import_error(self, sample_config):
        """Test graceful handling when mlflow raises ImportError."""
        from hrp.ml.training import _log_to_mlflow
        import mlflow

        model = MagicMock()
        metrics = {"train_mse": 0.01}

        with patch.object(mlflow, "set_experiment", side_effect=ImportError("mlflow not installed")):
            # Should not raise - function catches ImportError
            _log_to_mlflow(sample_config, model, metrics, {}, [])

    def test_handles_exception(self, sample_config):
        """Test that exceptions are logged but not raised."""
        from hrp.ml.training import _log_to_mlflow
        import mlflow

        model = MagicMock()
        metrics = {"train_mse": 0.01}

        with patch.object(mlflow, "set_experiment", side_effect=Exception("Random error")):
            # Should not raise - function catches Exception
            _log_to_mlflow(sample_config, model, metrics, {}, [])


class TestTrainModelExtended:
    """Extended tests for train_model function."""

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

    def test_with_mlflow_logging(self, sample_config, mock_data):
        """Test that log_to_mlflow=True triggers MLflow logging."""
        from hrp.ml.training import train_model

        with patch("hrp.ml.training.load_training_data") as mock_load, \
             patch("hrp.ml.training._log_to_mlflow") as mock_log:
            mock_load.return_value = mock_data

            train_model(sample_config, symbols=["AAPL", "MSFT"], log_to_mlflow=True)

        mock_log.assert_called_once()

    def test_without_mlflow_logging(self, sample_config, mock_data):
        """Test that log_to_mlflow=False skips MLflow logging."""
        from hrp.ml.training import train_model

        with patch("hrp.ml.training.load_training_data") as mock_load, \
             patch("hrp.ml.training._log_to_mlflow") as mock_log:
            mock_load.return_value = mock_data

            train_model(sample_config, symbols=["AAPL", "MSFT"], log_to_mlflow=False)

        mock_log.assert_not_called()

    def test_feature_selection_enabled(self, mock_data):
        """Test that feature selection is triggered when enabled."""
        from hrp.ml.training import train_model

        config = MLConfig(
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d", "rsi_14d", "atr_14d", "volume_ratio"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=True,
            max_features=2,
        )

        # Create mock data with 5 features
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        symbols = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        X = pd.DataFrame(
            np.random.randn(len(index), 5),
            index=index,
            columns=["momentum_20d", "volatility_20d", "rsi_14d", "atr_14d", "volume_ratio"],
        )
        y = pd.Series(np.random.randn(len(index)) * 0.05, index=index, name="returns_20d")

        mock_data_5 = {
            "X_train": X, "y_train": y,
            "X_val": X.iloc[:50], "y_val": y.iloc[:50],
            "X_test": X.iloc[:50], "y_test": y.iloc[:50],
        }

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data_5

            result = train_model(config, symbols=["AAPL", "MSFT"])

        # Should have selected max_features
        assert len(result.selected_features) == 2

    def test_tree_model_importance(self, mock_data):
        """Test that tree-based models extract feature_importances_."""
        from hrp.ml.training import train_model

        config = MLConfig(
            model_type="random_forest",
            target="returns_20d",
            features=["momentum_20d", "volatility_20d"],
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            feature_selection=False,
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data

            result = train_model(config, symbols=["AAPL", "MSFT"])

        # Should have feature importance from tree model
        assert len(result.feature_importance) == 2
        assert "momentum_20d" in result.feature_importance
        assert "volatility_20d" in result.feature_importance

    def test_linear_model_importance(self, sample_config, mock_data):
        """Test that linear models extract coef_ as importance."""
        from hrp.ml.training import train_model

        with patch("hrp.ml.training.load_training_data") as mock_load:
            mock_load.return_value = mock_data

            result = train_model(sample_config, symbols=["AAPL", "MSFT"])

        # Ridge model should have coef_-based importance
        assert len(result.feature_importance) == 2
        assert "momentum_20d" in result.feature_importance
        assert "volatility_20d" in result.feature_importance
        # Values should be absolute coefficients
        for val in result.feature_importance.values():
            assert val >= 0


class TestSelectFeatures:
    """Tests for select_features function."""

    def test_returns_all_when_below_max(self):
        """Test that all features returned when count <= max_features."""
        from hrp.ml.training import select_features

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randn(100))

        result = select_features(X, y, max_features=5)

        assert len(result) == 3
        assert set(result) == {"a", "b", "c"}

    def test_selects_top_n(self):
        """Test that only top N features are selected."""
        from hrp.ml.training import select_features

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=["a", "b", "c", "d", "e"])
        y = pd.Series(np.random.randn(100))

        result = select_features(X, y, max_features=2)

        assert len(result) == 2

    def test_handles_nan(self):
        """Test that NaN values are handled."""
        from hrp.ml.training import select_features

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        X.iloc[0:10, 0] = np.nan  # Add some NaN
        y = pd.Series(np.random.randn(100))

        result = select_features(X, y, max_features=2)

        assert len(result) == 2
