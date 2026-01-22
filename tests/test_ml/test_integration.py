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
        assert result.metrics["val_mse"] > 0

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

    def test_different_models(self, mock_features_df):
        """Test that different model types work in the pipeline."""
        symbols = ["AAPL", "MSFT"]

        for model_type in ["ridge", "lasso", "random_forest"]:
            config = MLConfig(
                model_type=model_type,
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

            with patch("hrp.ml.training._fetch_features") as mock_fetch:
                mock_fetch.return_value = mock_features_df[
                    mock_features_df.index.get_level_values("symbol").isin(symbols)
                ]
                result = train_model(config, symbols=symbols)

            assert result.model is not None
            assert "train_mse" in result.metrics
            assert result.selected_features == ["momentum_20d", "volatility_20d"]

    def test_signal_methods(self, mock_features_df):
        """Test all signal generation methods work with predictions."""
        # Create sample predictions
        dates = pd.date_range("2021-07-01", periods=10, freq="B")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        np.random.seed(42)
        pred_df = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)) * 0.05,
            index=dates,
            columns=symbols,
        )

        # Test rank method
        rank_signals = predictions_to_signals(pred_df, method="rank", top_pct=0.34)
        assert rank_signals.shape == pred_df.shape
        assert set(rank_signals.values.flatten()) <= {0.0, 1.0}

        # Test threshold method
        threshold_signals = predictions_to_signals(pred_df, method="threshold", threshold=0.0)
        assert threshold_signals.shape == pred_df.shape

        # Test zscore method
        zscore_signals = predictions_to_signals(pred_df, method="zscore")
        assert zscore_signals.shape == pred_df.shape
        # Z-scores should be continuous
        assert not set(zscore_signals.values.flatten()) <= {0.0, 1.0}
