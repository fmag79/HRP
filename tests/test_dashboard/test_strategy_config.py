"""
Tests for dashboard strategy configuration component.

Tests hrp.dashboard.components.strategy_config module which provides
UI for configuring trading strategies.
"""

from unittest.mock import patch, MagicMock
import pytest

from hrp.dashboard.components.strategy_config import (
    get_available_features,
    render_multifactor_config,
    render_ml_predicted_config,
)


class TestGetAvailableFeatures:
    """Test get_available_features function."""

    def test_returns_features_from_database(self):
        """Returns list of features when database query succeeds."""
        mock_result = [
            ("momentum_20d",),
            ("volatility_60d",),
            ("rsi_14d",),
        ]

        with patch('hrp.api.platform.PlatformAPI') as mock_api_cls:
            mock_api = MagicMock()
            mock_api.fetchall_readonly.return_value = mock_result
            mock_api_cls.return_value = mock_api

            features = get_available_features()

        assert features == ["momentum_20d", "volatility_60d", "rsi_14d"]

    def test_returns_sorted_features(self):
        """Returns features in alphabetical order (SQL has ORDER BY)."""
        # Database query has ORDER BY, so results come back sorted
        mock_result = [
            ("momentum_20d",),
            ("rsi_14d",),
            ("volatility_60d",),
        ]

        with patch('hrp.api.platform.PlatformAPI') as mock_api_cls:
            mock_api = MagicMock()
            mock_api.fetchall_readonly.return_value = mock_result
            mock_api_cls.return_value = mock_api

            features = get_available_features()

        assert features == ["momentum_20d", "rsi_14d", "volatility_60d"]

    def test_returns_default_on_database_error(self):
        """Returns default features when database query fails."""
        with patch('hrp.api.platform.PlatformAPI') as mock_api_cls:
            mock_api_cls.side_effect = Exception("Database error")

            features = get_available_features()

        assert features == ["momentum_20d", "volatility_60d"]

    def test_returns_empty_list_when_no_features(self):
        """Returns empty list when no features in database."""
        with patch('hrp.api.platform.PlatformAPI') as mock_api_cls:
            mock_api = MagicMock()
            mock_api.fetchall_readonly.return_value = []
            mock_api_cls.return_value = mock_api

            features = get_available_features()

        assert features == []


class TestRenderMultifactorConfig:
    """Test render_multifactor_config function."""

    def test_renders_config_title(self):
        """Renders configuration section title."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = False
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                render_multifactor_config()

        # Check that markdown was called with title
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Multi-Factor Configuration" in call for call in markdown_calls)

    def test_returns_empty_config_when_no_features(self):
        """Returns empty configuration when no features available."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=[]):
                mock_st.selectbox.return_value = "Custom"

                config = render_multifactor_config()

        assert config == {"feature_weights": {}, "top_n": 10}
        mock_st.warning.assert_called()

    def test_includes_preset_strategies(self):
        """Includes preset strategies in dropdown options."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = False
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                render_multifactor_config()

        # Check that selectbox was called with options that include presets
        mock_st.selectbox.assert_called_once()
        call_kwargs = mock_st.selectbox.call_args[1]
        assert "options" in call_kwargs
        options = call_kwargs["options"]
        assert "Custom" in options

    def test_displays_preset_description_when_selected(self):
        """Displays preset description when a preset strategy is selected."""
        from hrp.research.strategies import PRESET_STRATEGIES

        # Get the first preset name
        first_preset_key = list(PRESET_STRATEGIES.keys())[0]
        first_preset_name = PRESET_STRATEGIES[first_preset_key]["name"]

        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                # Select a preset (not "Custom")
                mock_st.selectbox.return_value = first_preset_name
                mock_st.checkbox.return_value = False
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                render_multifactor_config()

        # Should show info about the preset
        mock_st.info.assert_called()

    def test_renders_feature_checkboxes(self):
        """Renders checkboxes for each available feature."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d", "volatility_60d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Should have called checkbox for each feature (in 2x1 grid = 1 column call)
        assert mock_st.checkbox.call_count >= 2

    def test_renders_weight_sliders_for_selected_features(self):
        """Renders weight sliders for features that are selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Should have called slider for selected feature
        assert mock_st.slider.call_count >= 1

    def test_uses_preset_weights_when_preset_selected(self):
        """Uses preset feature weights when preset strategy is selected."""
        from hrp.research.strategies import PRESET_STRATEGIES

        # Get the first preset name
        first_preset_key = list(PRESET_STRATEGIES.keys())[0]
        first_preset_name = PRESET_STRATEGIES[first_preset_key]["name"]

        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                # Select a preset
                mock_st.selectbox.return_value = first_preset_name
                # Feature is selected by preset
                mock_st.checkbox.return_value = True
                # Slider should use preset weight
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Check that slider was called (would be called with preset weight)
        assert mock_st.slider.call_count >= 1

    def test_uses_negative_weight_for_volatility_features(self):
        """Uses negative default weight for volatility features."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["volatility_60d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = -0.5
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Slider should have been called with default for volatility
        assert mock_st.slider.call_count >= 1

    def test_renders_top_n_input(self):
        """Renders number input for top N selection."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Should have called number_input for top_n
        assert mock_st.number_input.call_count >= 1

    def test_shows_configuration_summary(self):
        """Displays summary of selected configuration."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        # Should show summary with features and holdings
        assert any("Configuration Summary" in str(call) for call in mock_st.markdown.call_args_list)

    def test_warns_when_no_features_selected(self):
        """Warns user when no features are selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                # No features selected
                mock_st.checkbox.return_value = False
                mock_st.columns.return_value = [MagicMock(), MagicMock()]
                mock_st.number_input.return_value = 10

                config = render_multifactor_config()

        mock_st.warning.assert_called()

    def test_returns_config_with_feature_weights_and_top_n(self):
        """Returns configuration dictionary with correct structure."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "Custom"
                mock_st.checkbox.return_value = True
                mock_st.slider.return_value = 1.0
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_multifactor_config()

        assert "feature_weights" in config
        assert "top_n" in config
        assert isinstance(config["top_n"], int)


class TestRenderMLPredictedConfig:
    """Test render_ml_predicted_config function."""

    def test_renders_config_title(self):
        """Renders configuration section title."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "ridge"
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Check that markdown was called with title
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("ML-Predicted Strategy Configuration" in call for call in markdown_calls)

    def test_shows_model_type_dropdown(self):
        """Shows dropdown for model type selection."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "ridge"
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should have called selectbox for model type
        assert mock_st.selectbox.call_count >= 1

    def test_shows_feature_multiselect(self):
        """Shows multiselect for feature selection."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "ridge"
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should have called multiselect for features
        mock_st.multiselect.assert_called()

    def test_warns_when_no_features_selected(self):
        """Warns when no features are selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.return_value = "ridge"
                mock_st.multiselect.return_value = []  # No features selected
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        mock_st.warning.assert_called()

    def test_shows_signal_method_selection(self):
        """Shows options for signal generation method."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should have selectbox for signal method
        assert mock_st.selectbox.call_count >= 2

    def test_shows_top_pct_slider_for_rank_method(self):
        """Shows top percentile slider when rank method is selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should show slider for top_pct
        assert mock_st.slider.call_count >= 1

    def test_shows_threshold_input_for_threshold_method(self):
        """Shows threshold input when threshold method is selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "threshold"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.number_input.side_effect = [0.02, 252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should show number_input for threshold (total 3 calls: threshold, train_lookback, retrain_frequency)
        assert mock_st.number_input.call_count >= 3

    def test_shows_info_for_zscore_method(self):
        """Shows info message when zscore method is selected."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "zscore"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should show info about zscore
        mock_st.info.assert_called()

    def test_shows_training_parameters(self):
        """Shows training parameters inputs."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should have number_input for train_lookback and retrain_frequency
        assert mock_st.number_input.call_count >= 2

    def test_shows_configuration_summary(self):
        """Displays summary of ML configuration."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        # Should show summary text with config details
        assert mock_st.text.call_count >= 4  # model, features, signal, training

    def test_returns_config_with_all_required_fields(self):
        """Returns configuration with all required fields."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        assert "model_type" in config
        assert "features" in config
        assert "signal_method" in config
        assert "top_pct" in config
        assert "threshold" in config
        assert "train_lookback" in config
        assert "retrain_frequency" in config
        assert config["model_type"] == "ridge"
        assert config["features"] == ["momentum_20d"]
        assert config["signal_method"] == "rank"

    def test_returns_int_for_numeric_parameters(self):
        """Returns integer values for numeric parameters."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                mock_st.selectbox.side_effect = ["ridge", "rank"]
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.side_effect = [252, 21]
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                config = render_ml_predicted_config()

        assert isinstance(config["train_lookback"], int)
        assert isinstance(config["retrain_frequency"], int)
        assert config["train_lookback"] == 252
        assert config["retrain_frequency"] == 21


class TestStrategyConfigIntegration:
    """Integration tests for strategy config component."""

    def test_functions_are_callable(self):
        """Exported functions are callable."""
        assert callable(get_available_features)
        assert callable(render_multifactor_config)
        assert callable(render_ml_predicted_config)

    def test_get_available_features_signature(self):
        """get_available_features has correct signature."""
        import inspect

        sig = inspect.signature(get_available_features)
        params = list(sig.parameters.keys())

        assert len(params) == 0  # No parameters

    def test_render_multifactor_config_signature(self):
        """render_multifactor_config has correct signature."""
        import inspect

        sig = inspect.signature(render_multifactor_config)
        params = list(sig.parameters.keys())

        assert len(params) == 0  # No parameters

    def test_render_ml_predicted_config_signature(self):
        """render_ml_predicted_config has correct signature."""
        import inspect

        sig = inspect.signature(render_ml_predicted_config)
        params = list(sig.parameters.keys())

        assert len(params) == 0  # No parameters

    def test_render_functions_return_dict(self):
        """Render functions always return dictionaries."""
        with patch('hrp.dashboard.components.strategy_config.st') as mock_st:
            with patch('hrp.dashboard.components.strategy_config.get_available_features', return_value=["momentum_20d"]):
                # Setup mocks with enough side_effect values for both function calls
                # multifactor_config: 1 selectbox call
                # ml_predicted_config: 2 selectbox calls (model_type, signal_method)
                mock_st.selectbox.side_effect = ["Custom", "ridge", "rank"]
                mock_st.checkbox.return_value = False
                mock_st.multiselect.return_value = ["momentum_20d"]
                mock_st.slider.return_value = 0.10
                mock_st.number_input.return_value = 10
                mock_st.columns.return_value = [MagicMock(), MagicMock()]

                multifactor_config = render_multifactor_config()
                ml_config = render_ml_predicted_config()

        assert isinstance(multifactor_config, dict)
        assert isinstance(ml_config, dict)
