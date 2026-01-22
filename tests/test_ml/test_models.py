"""Tests for ML model registry."""
from datetime import date
import pytest
from hrp.ml.models import MLConfig, SUPPORTED_MODELS, get_model, HAS_LIGHTGBM


class TestMLConfig:
    def test_mlconfig_creation(self):
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
        config = MLConfig(
            model_type="ridge",
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


class TestGetModel:
    def test_get_ridge_model(self):
        model = get_model("ridge")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_ridge_with_hyperparams(self):
        model = get_model("ridge", {"alpha": 0.5})
        assert model.alpha == 0.5

    def test_get_lightgbm_model(self):
        if not HAS_LIGHTGBM:
            pytest.skip("LightGBM not installed")
        model = get_model("lightgbm")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_invalid_model(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_model("invalid_model")
