"""Tests for hrp/research/mlflow_utils.py."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import json

import pandas as pd
import pytest

from hrp.research.config import BacktestConfig, BacktestResult, CostModel


class TestSetupMlflow:
    """Tests for setup_mlflow function."""

    def test_default_uri(self):
        """Test that default URI creates local sqlite tracking."""
        with patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow, \
             patch("hrp.research.mlflow_utils.MLFLOW_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()

            from hrp.research.mlflow_utils import setup_mlflow
            setup_mlflow()

        mock_mlflow.set_tracking_uri.assert_called_once()
        call_arg = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert "sqlite:///" in call_arg
        assert "mlflow.db" in call_arg

    def test_custom_uri(self):
        """Test that custom URI is used when provided."""
        with patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            from hrp.research.mlflow_utils import setup_mlflow
            setup_mlflow(tracking_uri="http://mlflow.example.com")

        mock_mlflow.set_tracking_uri.assert_called_with("http://mlflow.example.com")


class TestGetOrCreateExperiment:
    """Tests for get_or_create_experiment function."""

    def test_creates_new(self):
        """Test that new experiment is created if missing."""
        with patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"

            from hrp.research.mlflow_utils import get_or_create_experiment
            result = get_or_create_experiment("test_experiment")

        assert result == "exp-123"
        mock_mlflow.create_experiment.assert_called_once()

    def test_returns_existing(self):
        """Test that existing experiment ID is returned."""
        with patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = "existing-456"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            from hrp.research.mlflow_utils import get_or_create_experiment
            result = get_or_create_experiment("test_experiment")

        assert result == "existing-456"
        mock_mlflow.create_experiment.assert_not_called()


class TestLogBacktest:
    """Tests for log_backtest function."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample BacktestResult for testing."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            sizing_method="equal_weight",
            max_positions=10,
            max_position_pct=0.1,
            costs=CostModel(spread_bps=2.0, slippage_bps=5.0),
        )
        return BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5, "total_return": 0.25, "max_drawdown": -0.15},
            equity_curve=pd.Series([100, 105, 110, 108, 115]),
            trades=pd.DataFrame({"date": ["2020-01-15"], "symbol": ["AAPL"], "side": ["buy"]}),
            benchmark_metrics={"sharpe_ratio": 1.0, "total_return": 0.15},
        )

    def test_basic(self):
        """Test that basic logging works."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment") as mock_exp, \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_exp.return_value = "exp-1"
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            run_id = log_backtest(result)

        assert run_id == "run-123"
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()

    def test_with_hypothesis_id(self):
        """Test that hypothesis tag is set."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(result, hypothesis_id="HYP-2025-001")

        mock_mlflow.set_tag.assert_any_call("hypothesis_id", "HYP-2025-001")

    def test_with_tags(self):
        """Test that custom tags are logged."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(result, tags={"strategy": "momentum", "version": "1.0"})

        mock_mlflow.set_tag.assert_any_call("strategy", "momentum")
        mock_mlflow.set_tag.assert_any_call("version", "1.0")

    def test_with_feature_versions(self):
        """Test that feature versions are logged as params."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(
                result,
                feature_versions={"momentum_20d": "v1.2", "volatility_60d": "v2.0"},
            )

        # Check that feature version params were logged
        log_param_calls = [str(c) for c in mock_mlflow.log_param.call_args_list]
        assert any("feature_version_momentum_20d" in c for c in log_param_calls)

    def test_strategy_config_dict(self):
        """Test that dict strategy config is serialized as JSON."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(
                result,
                strategy_config={"weights": {"momentum": 1.0, "vol": -0.5}},
            )

        # Find the call with strategy_weights
        log_param_calls = mock_mlflow.log_param.call_args_list
        json_call = [c for c in log_param_calls if "strategy_weights" in str(c)]
        assert len(json_call) > 0

    def test_strategy_config_list(self):
        """Test that list strategy config is serialized as comma-separated."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(
                result,
                strategy_config={"features": ["momentum_20d", "vol_60d"]},
            )

        # Check that list was joined
        log_param_calls = [str(c) for c in mock_mlflow.log_param.call_args_list]
        assert any("momentum_20d,vol_60d" in c for c in log_param_calls)

    def test_benchmark_metrics(self):
        """Test that benchmark metrics get benchmark_ prefix."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
            benchmark_metrics={"sharpe_ratio": 1.0, "total_return": 0.15},
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(result)

        # Check benchmark metrics have prefix
        log_metric_calls = [str(c) for c in mock_mlflow.log_metric.call_args_list]
        assert any("benchmark_sharpe_ratio" in c for c in log_metric_calls)
        assert any("benchmark_total_return" in c for c in log_metric_calls)

    def test_nan_metric_skipped(self):
        """Test that NaN metrics are not logged."""
        import math
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5, "bad_metric": float("nan")},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            log_backtest(result)

        # NaN metric should not be logged
        log_metric_calls = [str(c) for c in mock_mlflow.log_metric.call_args_list]
        assert not any("bad_metric" in c for c in log_metric_calls)

    def test_returns_run_id(self):
        """Test that valid run_id is returned."""
        config = BacktestConfig(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5},
            equity_curve=None,
            trades=None,
        )

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.get_or_create_experiment"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "unique-run-id-abc123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            from hrp.research.mlflow_utils import log_backtest
            run_id = log_backtest(result)

        assert run_id == "unique-run-id-abc123"


class TestGetBestRuns:
    """Tests for get_best_runs function."""

    def test_basic(self):
        """Test that DataFrame of runs is returned."""
        mock_runs = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "metrics.sharpe_ratio": [1.5, 1.2],
        })

        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = "exp-1"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment
            mock_mlflow.search_runs.return_value = mock_runs

            from hrp.research.mlflow_utils import get_best_runs
            result = get_best_runs("test_experiment")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_custom_metric(self):
        """Test that runs are sorted by specified metric."""
        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = "exp-1"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment
            mock_mlflow.search_runs.return_value = pd.DataFrame()

            from hrp.research.mlflow_utils import get_best_runs
            get_best_runs("test_experiment", metric="total_return")

        call_args = mock_mlflow.search_runs.call_args
        assert "metrics.total_return DESC" in call_args[1]["order_by"]

    def test_experiment_not_found(self):
        """Test that empty DataFrame is returned when experiment doesn't exist."""
        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None

            from hrp.research.mlflow_utils import get_best_runs
            result = get_best_runs("nonexistent_experiment")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_top_n_limit(self):
        """Test that top_n parameter is respected."""
        with patch("hrp.research.mlflow_utils.setup_mlflow"), \
             patch("hrp.research.mlflow_utils.mlflow") as mock_mlflow:
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = "exp-1"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment
            mock_mlflow.search_runs.return_value = pd.DataFrame()

            from hrp.research.mlflow_utils import get_best_runs
            get_best_runs("test_experiment", top_n=5)

        call_args = mock_mlflow.search_runs.call_args
        assert call_args[1]["max_results"] == 5
