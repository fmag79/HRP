"""Tests for ML signal generation."""
import numpy as np
import pandas as pd
import pytest
from hrp.ml.signals import predictions_to_signals


class TestPredictionsToSignals:
    @pytest.fixture
    def sample_predictions(self):
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
        signals = predictions_to_signals(sample_predictions, method="rank", top_pct=0.25)
        # With 4 symbols and top 25%, should select 1 per day
        for date_idx in signals.index:
            assert signals.loc[date_idx].sum() == 1.0

    def test_rank_method_shape_preserved(self, sample_predictions):
        signals = predictions_to_signals(sample_predictions, method="rank")
        assert signals.shape == sample_predictions.shape
        assert list(signals.columns) == list(sample_predictions.columns)

    def test_threshold_method(self, sample_predictions):
        signals = predictions_to_signals(sample_predictions, method="threshold", threshold=0.05)
        assert signals.loc["2023-01-01", "AAPL"] == 1.0  # 0.05 >= 0.05
        assert signals.loc["2023-01-01", "MSFT"] == 0.0  # 0.03 < 0.05

    def test_zscore_method(self, sample_predictions):
        signals = predictions_to_signals(sample_predictions, method="zscore")
        # Z-scores should have mean ~0 and std ~1 per row
        for date_idx in signals.index:
            row = signals.loc[date_idx]
            assert abs(row.mean()) < 0.01
            assert abs(row.std() - 1.0) < 0.01

    def test_invalid_method(self, sample_predictions):
        with pytest.raises(ValueError, match="Unknown method"):
            predictions_to_signals(sample_predictions, method="invalid")
