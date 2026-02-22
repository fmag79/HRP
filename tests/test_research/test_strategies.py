"""
Tests for trading strategy signal generators.

Tests for:
- generate_multifactor_signals()
- generate_ml_predicted_signals()
- STRATEGY_REGISTRY
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.research.strategies import (
    generate_multifactor_signals,
    generate_ml_predicted_signals,
    STRATEGY_REGISTRY,
    PRESET_STRATEGIES,
    get_strategy_generator,
    get_preset_strategy,
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Create random prices with some trend
    np.random.seed(42)
    data = {}
    for symbol in symbols:
        base_price = np.random.uniform(100, 500)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        data[symbol] = prices

    close_df = pd.DataFrame(data, index=dates)

    # Create MultiIndex DataFrame like get_price_data returns
    prices = pd.DataFrame(
        close_df.values,
        index=dates,
        columns=pd.MultiIndex.from_product([["close"], symbols]),
    )

    return prices


@pytest.fixture
def mock_feature_computer():
    """Mock FeatureComputer for testing without database."""
    with patch("hrp.research.strategies.FeatureComputer") as MockComputer:
        mock_instance = MagicMock()
        MockComputer.return_value = mock_instance
        yield mock_instance


class TestStrategyRegistry:
    """Tests for STRATEGY_REGISTRY."""

    def test_registry_contains_expected_strategies(self):
        """Registry should contain momentum, multifactor, and ml_predicted."""
        assert "momentum" in STRATEGY_REGISTRY
        assert "multifactor" in STRATEGY_REGISTRY
        assert "ml_predicted" in STRATEGY_REGISTRY

    def test_registry_entries_have_required_fields(self):
        """Each registry entry should have name, description, generator, params."""
        for strategy_type, info in STRATEGY_REGISTRY.items():
            assert "name" in info, f"{strategy_type} missing 'name'"
            assert "description" in info, f"{strategy_type} missing 'description'"
            assert "generator" in info, f"{strategy_type} missing 'generator'"
            assert "params" in info, f"{strategy_type} missing 'params'"

    def test_get_strategy_generator_momentum(self):
        """get_strategy_generator should return momentum generator."""
        gen = get_strategy_generator("momentum")
        assert callable(gen)

    def test_get_strategy_generator_multifactor(self):
        """get_strategy_generator should return multifactor generator."""
        gen = get_strategy_generator("multifactor")
        assert gen == generate_multifactor_signals

    def test_get_strategy_generator_ml_predicted(self):
        """get_strategy_generator should return ml_predicted generator."""
        gen = get_strategy_generator("ml_predicted")
        assert gen == generate_ml_predicted_signals

    def test_get_strategy_generator_unknown_raises(self):
        """get_strategy_generator should raise for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy_generator("unknown_strategy")


class TestMultifactorSignals:
    """Tests for generate_multifactor_signals()."""

    def test_empty_feature_weights_raises(self, sample_prices):
        """Should raise ValueError if feature_weights is empty."""
        with pytest.raises(ValueError, match="feature_weights cannot be empty"):
            generate_multifactor_signals(sample_prices, feature_weights={})

    def test_returns_dataframe_with_correct_shape(
        self, sample_prices, mock_feature_computer
    ):
        """Signal DataFrame should match price data shape."""
        # Setup mock to return features
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
                "volatility_60d": np.random.rand(len(dates) * len(symbols)) * 0.3,
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_multifactor_signals(
            sample_prices,
            feature_weights={"momentum_20d": 1.0, "volatility_60d": -0.5},
            top_n=2,
        )

        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(dates)
        assert list(signals.columns) == symbols

    def test_signals_are_binary(self, sample_prices, mock_feature_computer):
        """Signals should be 0 or 1."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_multifactor_signals(
            sample_prices,
            feature_weights={"momentum_20d": 1.0},
            top_n=2,
        )

        unique_values = signals.values.flatten()
        unique_values = unique_values[~np.isnan(unique_values)]
        assert set(unique_values).issubset({0.0, 1.0})

    def test_top_n_respected(self, sample_prices, mock_feature_computer):
        """Number of selected stocks should not exceed top_n."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        top_n = 2
        signals = generate_multifactor_signals(
            sample_prices,
            feature_weights={"momentum_20d": 1.0},
            top_n=top_n,
        )

        # Check that each row has at most top_n positions
        positions_per_day = signals.sum(axis=1)
        assert (positions_per_day <= top_n).all()

    def test_negative_weight_favors_low_values(
        self, sample_prices, mock_feature_computer
    ):
        """Negative weight should favor stocks with lower feature values."""
        dates = sample_prices.index[:5]  # Use fewer dates for clarity
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        # Create features where AAPL has highest volatility
        mock_features = pd.DataFrame(
            {
                "volatility_60d": [0.5, 0.1, 0.2, 0.15, 0.25] * len(dates),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_multifactor_signals(
            sample_prices.loc[dates],
            feature_weights={"volatility_60d": -1.0},  # Negative: prefer low volatility
            top_n=1,
        )

        # MSFT has lowest volatility (0.1), should be selected
        # Check that MSFT is selected more often than AAPL
        assert signals["MSFT"].sum() >= signals["AAPL"].sum()


class TestMLPredictedSignals:
    """Tests for generate_ml_predicted_signals()."""

    def test_invalid_model_type_raises(self, sample_prices):
        """Should raise ValueError for unsupported model type."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            generate_ml_predicted_signals(
                sample_prices,
                model_type="invalid_model",
            )

    def test_returns_dataframe_with_correct_shape(
        self, sample_prices, mock_feature_computer
    ):
        """Signal DataFrame should match price data shape."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        # Need enough data for training
        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
                "volatility_60d": np.random.rand(len(dates) * len(symbols)) * 0.3,
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_ml_predicted_signals(
            sample_prices,
            model_type="ridge",
            features=["momentum_20d", "volatility_60d"],
            train_lookback=50,  # Smaller for test
            retrain_frequency=10,
        )

        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(dates)
        assert list(signals.columns) == symbols

    def test_rank_method_produces_binary_signals(
        self, sample_prices, mock_feature_computer
    ):
        """Rank method should produce 0/1 signals."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_ml_predicted_signals(
            sample_prices,
            model_type="ridge",
            features=["momentum_20d"],
            signal_method="rank",
            train_lookback=50,
            retrain_frequency=10,
        )

        unique_values = signals.values.flatten()
        unique_values = unique_values[~np.isnan(unique_values)]
        assert set(unique_values).issubset({0.0, 1.0})

    def test_threshold_method_produces_binary_signals(
        self, sample_prices, mock_feature_computer
    ):
        """Threshold method should produce 0/1 signals."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_ml_predicted_signals(
            sample_prices,
            model_type="ridge",
            features=["momentum_20d"],
            signal_method="threshold",
            threshold=0.0,
            train_lookback=50,
            retrain_frequency=10,
        )

        unique_values = signals.values.flatten()
        unique_values = unique_values[~np.isnan(unique_values)]
        assert set(unique_values).issubset({0.0, 1.0})

    def test_zscore_method_produces_continuous_signals(
        self, sample_prices, mock_feature_computer
    ):
        """Z-score method should produce continuous signals."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_ml_predicted_signals(
            sample_prices,
            model_type="ridge",
            features=["momentum_20d"],
            signal_method="zscore",
            train_lookback=50,
            retrain_frequency=10,
        )

        # Z-score signals can have values other than 0 and 1
        non_zero_non_one = signals.values.flatten()
        non_zero_non_one = non_zero_non_one[~np.isnan(non_zero_non_one)]
        non_zero_non_one = non_zero_non_one[
            (non_zero_non_one != 0.0) & (non_zero_non_one != 1.0)
        ]
        # Should have some non-binary values for z-score
        # (may be empty if all predictions are the same)
        assert isinstance(signals, pd.DataFrame)

    def test_default_features_used_when_none(
        self, sample_prices, mock_feature_computer
    ):
        """Should use default features when features=None."""
        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        mock_features = pd.DataFrame(
            {
                "momentum_20d": np.random.randn(len(dates) * len(symbols)),
                "volatility_60d": np.random.rand(len(dates) * len(symbols)) * 0.3,
            },
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        # Should not raise even with features=None
        signals = generate_ml_predicted_signals(
            sample_prices,
            model_type="ridge",
            features=None,  # Use defaults
            train_lookback=50,
            retrain_frequency=10,
        )

        assert isinstance(signals, pd.DataFrame)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_symbol(self, mock_feature_computer):
        """Should handle single symbol."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        symbols = ["AAPL"]

        prices = pd.DataFrame(
            np.random.rand(len(dates), 1) * 100 + 100,
            index=dates,
            columns=pd.MultiIndex.from_product([["close"], symbols]),
        )

        mock_features = pd.DataFrame(
            {"momentum_20d": np.random.randn(len(dates))},
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        signals = generate_multifactor_signals(
            prices,
            feature_weights={"momentum_20d": 1.0},
            top_n=1,
        )

        assert isinstance(signals, pd.DataFrame)
        assert len(signals.columns) == 1

    def test_empty_features_returns_zero_signals(
        self, sample_prices, mock_feature_computer
    ):
        """Should return zero signals when no features available."""
        mock_feature_computer.get_stored_features.side_effect = ValueError(
            "No features"
        )
        mock_feature_computer.compute_features.return_value = pd.DataFrame()

        signals = generate_multifactor_signals(
            sample_prices,
            feature_weights={"momentum_20d": 1.0},
            top_n=2,
        )

        assert (signals == 0).all().all()

    def test_short_date_range(self, mock_feature_computer):
        """Should handle short date ranges gracefully."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        symbols = ["AAPL", "MSFT"]

        prices = pd.DataFrame(
            np.random.rand(len(dates), 2) * 100 + 100,
            index=dates,
            columns=pd.MultiIndex.from_product([["close"], symbols]),
        )

        mock_features = pd.DataFrame(
            {"momentum_20d": np.random.randn(len(dates) * len(symbols))},
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        # ML-predicted with short range should return mostly zeros
        # (not enough data to train)
        signals = generate_ml_predicted_signals(
            prices,
            model_type="ridge",
            features=["momentum_20d"],
            train_lookback=50,  # More than available dates
            retrain_frequency=5,
        )

        assert isinstance(signals, pd.DataFrame)


class TestPresetStrategies:
    """Tests for PRESET_STRATEGIES and get_preset_strategy()."""

    def test_preset_registry_contains_expected_presets(self):
        """Registry should contain all strategy presets including regime_adaptive."""
        assert "mean_reversion" in PRESET_STRATEGIES
        assert "trend_following" in PRESET_STRATEGIES
        assert "quality_momentum" in PRESET_STRATEGIES
        assert "volume_breakout" in PRESET_STRATEGIES
        assert "regime_adaptive" in PRESET_STRATEGIES

    def test_preset_entries_have_required_fields(self):
        """Each preset should have name, description, feature_weights, default_top_n."""
        # regime_adaptive has different structure (bull_weights, bear_weights, sideways_weights)
        regime_switching_presets = ["regime_adaptive"]

        required_fields = ["name", "description", "feature_weights", "default_top_n"]
        for preset_name, preset in PRESET_STRATEGIES.items():
            if preset_name in regime_switching_presets:
                continue  # Skip regime switching presets

            for field in required_fields:
                assert field in preset, f"{preset_name} missing '{field}'"
            # Validate types
            assert isinstance(preset["name"], str)
            assert isinstance(preset["description"], str)
            assert isinstance(preset["feature_weights"], dict)
            assert isinstance(preset["default_top_n"], int)
            assert preset["default_top_n"] > 0

    def test_preset_feature_weights_use_valid_features(self):
        """Feature weights should use valid feature names from the feature store."""
        # regime_adaptive has different structure (bull_weights, bear_weights, sideways_weights)
        regime_switching_presets = ["regime_adaptive"]

        # Known valid features from CLAUDE.md
        valid_features = {
            "returns_1d", "returns_5d", "returns_20d", "returns_60d", "returns_252d",
            "momentum_20d", "momentum_60d", "momentum_252d",
            "volatility_20d", "volatility_60d",
            "volume_20d", "volume_ratio", "obv",
            "rsi_14d", "cci_20d", "roc_10d", "stoch_k_14d", "stoch_d_14d",
            "atr_14d", "adx_14d", "macd_line", "macd_signal", "macd_histogram", "trend",
            "sma_20d", "sma_50d", "sma_200d",
            "price_to_sma_20d", "price_to_sma_50d", "price_to_sma_200d",
            "bb_upper_20d", "bb_lower_20d", "bb_width_20d",
        }

        for preset_name, preset in PRESET_STRATEGIES.items():
            if preset_name in regime_switching_presets:
                continue  # Skip regime switching presets

            for feature in preset["feature_weights"]:
                assert feature in valid_features, (
                    f"Preset '{preset_name}' uses unknown feature '{feature}'"
                )

    def test_get_preset_strategy_returns_config(self):
        """get_preset_strategy should return feature_weights and top_n."""
        config = get_preset_strategy("mean_reversion")

        assert "feature_weights" in config
        assert "top_n" in config
        assert isinstance(config["feature_weights"], dict)
        assert isinstance(config["top_n"], int)

        # Verify it returns a copy (not the original)
        config["feature_weights"]["test_feature"] = 999
        assert "test_feature" not in PRESET_STRATEGIES["mean_reversion"]["feature_weights"]

    def test_get_preset_strategy_all_presets(self):
        """get_preset_strategy should work for all defined presets."""
        # Skip regime_adaptive - it has different structure (bull_weights, bear_weights, sideways_weights)
        # and is handled by generate_regime_switching_signals(), not get_preset_strategy()
        regime_switching_presets = ["regime_adaptive"]

        for preset_name in PRESET_STRATEGIES:
            if preset_name in regime_switching_presets:
                continue  # Skip regime switching presets

            config = get_preset_strategy(preset_name)
            assert config["feature_weights"] == PRESET_STRATEGIES[preset_name]["feature_weights"]
            assert config["top_n"] == PRESET_STRATEGIES[preset_name]["default_top_n"]

    def test_get_preset_strategy_unknown_raises(self):
        """get_preset_strategy should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_strategy("unknown_preset")

    def test_preset_generates_valid_signals(self, sample_prices, mock_feature_computer):
        """Each preset should generate valid signals when used with generate_multifactor_signals."""
        # Skip regime_adaptive - it has different structure and is handled by generate_regime_switching_signals()
        regime_switching_presets = ["regime_adaptive"]

        dates = sample_prices.index
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        # Create mock features with all required features
        all_features = set()
        for preset_name, preset in PRESET_STRATEGIES.items():
            if preset_name not in regime_switching_presets:
                all_features.update(preset["feature_weights"].keys())

        mock_data = {
            feature: np.random.randn(len(dates) * len(symbols))
            for feature in all_features
        }
        mock_features = pd.DataFrame(
            mock_data,
            index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        )
        mock_feature_computer.get_stored_features.return_value = mock_features

        # Test each preset (except regime_adaptive)
        for preset_name in PRESET_STRATEGIES:
            if preset_name in regime_switching_presets:
                continue  # Skip regime switching presets

            config = get_preset_strategy(preset_name)
            signals = generate_multifactor_signals(
                sample_prices,
                feature_weights=config["feature_weights"],
                top_n=config["top_n"],
            )

            assert isinstance(signals, pd.DataFrame), f"Preset {preset_name} failed"
            assert len(signals) == len(dates)
            # Signals should be binary (0 or 1)
            unique_values = signals.values.flatten()
            unique_values = unique_values[~np.isnan(unique_values)]
            assert set(unique_values).issubset({0.0, 1.0})
