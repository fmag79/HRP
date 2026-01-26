"""Tests for MCP server tools."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.mcp import research_server


# Helper to call the underlying function from a FunctionTool
def call_tool(tool, *args, **kwargs):
    """Call the underlying function of an MCP tool."""
    return tool.fn(*args, **kwargs)


@pytest.fixture
def mock_api():
    """Create a mock PlatformAPI."""
    with patch.object(research_server, "_api", None):
        with patch.object(research_server, "get_api") as mock_get_api:
            api = MagicMock()
            mock_get_api.return_value = api
            yield api


# =============================================================================
# Hypothesis Management Tools Tests
# =============================================================================


class TestListHypotheses:
    """Tests for list_hypotheses tool."""

    def test_list_hypotheses_success(self, mock_api):
        """List hypotheses successfully."""
        mock_api.list_hypotheses.return_value = [
            {
                "hypothesis_id": "HYP-2026-001",
                "title": "Momentum predicts returns",
                "status": "testing",
                "created_at": "2026-01-15T10:00:00",
                "updated_at": None,
            }
        ]

        result = call_tool(research_server.list_hypotheses)

        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["hypothesis_id"] == "HYP-2026-001"
        mock_api.list_hypotheses.assert_called_once_with(status=None, limit=100)

    def test_list_hypotheses_with_filter(self, mock_api):
        """List hypotheses with status filter."""
        mock_api.list_hypotheses.return_value = []

        result = call_tool(research_server.list_hypotheses, status="draft", limit=10)

        mock_api.list_hypotheses.assert_called_once_with(status="draft", limit=10)


class TestGetHypothesis:
    """Tests for get_hypothesis tool."""

    def test_get_hypothesis_found(self, mock_api):
        """Get existing hypothesis."""
        mock_api.get_hypothesis.return_value = {
            "hypothesis_id": "HYP-2026-001",
            "title": "Test",
            "thesis": "Test thesis",
            "prediction": "Test prediction",
            "falsification": "Test criteria",
            "status": "draft",
            "created_at": "2026-01-15T10:00:00",
            "updated_at": None,
        }

        result = call_tool(research_server.get_hypothesis, "HYP-2026-001")

        assert result["success"] is True
        assert result["data"]["hypothesis_id"] == "HYP-2026-001"

    def test_get_hypothesis_not_found(self, mock_api):
        """Handle non-existent hypothesis."""
        mock_api.get_hypothesis.return_value = None

        result = call_tool(research_server.get_hypothesis, "HYP-2026-999")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


class TestCreateHypothesis:
    """Tests for create_hypothesis tool."""

    def test_create_hypothesis_success(self, mock_api):
        """Create hypothesis successfully."""
        mock_api.create_hypothesis.return_value = "HYP-2026-002"

        result = call_tool(
            research_server.create_hypothesis,
            title="New hypothesis",
            thesis="Something is true",
            prediction="We expect X",
            falsification="Reject if Y",
        )

        assert result["success"] is True
        assert result["data"]["hypothesis_id"] == "HYP-2026-002"
        mock_api.create_hypothesis.assert_called_once_with(
            title="New hypothesis",
            thesis="Something is true",
            prediction="We expect X",
            falsification="Reject if Y",
            actor="agent:claude-interactive",
        )


class TestUpdateHypothesis:
    """Tests for update_hypothesis tool."""

    def test_update_hypothesis_success(self, mock_api):
        """Update hypothesis status."""
        result = call_tool(
            research_server.update_hypothesis,
            hypothesis_id="HYP-2026-001",
            status="validated",
            outcome="Results confirmed prediction",
        )

        assert result["success"] is True
        mock_api.update_hypothesis.assert_called_once_with(
            hypothesis_id="HYP-2026-001",
            status="validated",
            outcome="Results confirmed prediction",
            actor="agent:claude-interactive",
        )


# =============================================================================
# Data Access Tools Tests
# =============================================================================


class TestGetUniverse:
    """Tests for get_universe tool."""

    def test_get_universe_success(self, mock_api):
        """Get trading universe."""
        mock_api.get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]

        result = call_tool(research_server.get_universe)

        assert result["success"] is True
        assert len(result["data"]["symbols"]) == 3
        assert result["data"]["count"] == 3

    def test_get_universe_with_date(self, mock_api):
        """Get universe for specific date."""
        mock_api.get_universe.return_value = ["AAPL"]

        result = call_tool(research_server.get_universe, as_of_date="2023-01-15")

        mock_api.get_universe.assert_called_once_with(date(2023, 1, 15))


class TestGetFeatures:
    """Tests for get_features tool."""

    def test_get_features_success(self, mock_api):
        """Get feature values."""
        mock_api.get_features.return_value = pd.DataFrame({
            "symbol": ["AAPL"],
            "momentum_20d": [0.05],
            "volatility_60d": [0.20],
        })

        result = call_tool(
            research_server.get_features,
            symbols=["AAPL"],
            features=["momentum_20d", "volatility_60d"],
            as_of_date="2023-01-15",
        )

        assert result["success"] is True

    def test_get_features_missing_date(self, mock_api):
        """Handle missing date parameter."""
        result = call_tool(
            research_server.get_features,
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=None,
        )

        assert result["success"] is False
        assert "required" in result["error"].lower()


class TestGetAvailableFeatures:
    """Tests for get_available_features tool."""

    def test_get_available_features(self, mock_api):
        """List available features."""
        with patch("hrp.data.features.registry.FeatureRegistry") as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.list_all_features.return_value = [
                {
                    "feature_name": "momentum_20d",
                    "version": "v1",
                    "description": "20-day momentum",
                }
            ]
            MockRegistry.return_value = mock_registry

            result = call_tool(research_server.get_available_features)

            assert result["success"] is True
            assert len(result["data"]) == 1


class TestIsTradingDay:
    """Tests for is_trading_day tool."""

    def test_is_trading_day_true(self, mock_api):
        """Check trading day - yes."""
        mock_api.is_trading_day.return_value = True

        result = call_tool(research_server.is_trading_day, "2023-01-16")

        assert result["success"] is True
        assert result["data"]["is_trading_day"] is True

    def test_is_trading_day_false(self, mock_api):
        """Check trading day - no (weekend)."""
        mock_api.is_trading_day.return_value = False

        result = call_tool(research_server.is_trading_day, "2023-01-15")

        assert result["success"] is True
        assert result["data"]["is_trading_day"] is False


# =============================================================================
# Backtesting Tools Tests
# =============================================================================


class TestRunBacktest:
    """Tests for run_backtest tool."""

    def test_run_backtest_success(self, mock_api):
        """Run backtest successfully."""
        mock_api.run_backtest.return_value = "exp-123"
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.25,
                "max_drawdown": -0.10,
                "cagr": 0.15,
                "volatility": 0.10,
            },
        }

        result = call_tool(
            research_server.run_backtest,
            hypothesis_id="HYP-2026-001",
            symbols=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

        assert result["success"] is True
        assert result["data"]["experiment_id"] == "exp-123"
        assert result["data"]["metrics"]["sharpe_ratio"] == 1.5


class TestGetExperiment:
    """Tests for get_experiment tool."""

    def test_get_experiment_found(self, mock_api):
        """Get existing experiment."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "status": "FINISHED",
            "metrics": {"sharpe_ratio": 1.2},
            "params": {},
            "tags": {},
        }

        result = call_tool(research_server.get_experiment, "exp-123")

        assert result["success"] is True
        assert result["data"]["experiment_id"] == "exp-123"

    def test_get_experiment_not_found(self, mock_api):
        """Handle non-existent experiment."""
        mock_api.get_experiment.return_value = None

        result = call_tool(research_server.get_experiment, "nonexistent")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


class TestCompareExperiments:
    """Tests for compare_experiments tool."""

    def test_compare_experiments_success(self, mock_api):
        """Compare experiments."""
        mock_api.compare_experiments.return_value = pd.DataFrame({
            "sharpe_ratio": [1.2, 1.5],
            "total_return": [0.20, 0.25],
        }, index=["exp-1", "exp-2"])

        result = call_tool(
            research_server.compare_experiments,
            experiment_ids=["exp-1", "exp-2"],
        )

        assert result["success"] is True


# =============================================================================
# Quality & Health Tools Tests
# =============================================================================


class TestRunQualityChecks:
    """Tests for run_quality_checks tool."""

    def test_run_quality_checks_success(self, mock_api):
        """Run quality checks."""
        with patch("hrp.data.quality.report.QualityReportGenerator") as MockGenerator:
            mock_report = MagicMock()
            mock_report.report_date = date(2026, 1, 15)
            mock_report.health_score = 95.0
            mock_report.passed = True
            mock_report.checks_run = 5
            mock_report.checks_passed = 5
            mock_report.critical_issues = 0
            mock_report.warning_issues = 1
            mock_report.get_summary_text.return_value = "All checks passed"
            MockGenerator.return_value.generate_report.return_value = mock_report

            result = call_tool(research_server.run_quality_checks)

            assert result["success"] is True
            assert result["data"]["health_score"] == 95.0
            assert result["data"]["passed"] is True


class TestGetHealthStatus:
    """Tests for get_health_status tool."""

    def test_get_health_status_success(self, mock_api):
        """Get health status."""
        mock_api.health_check.return_value = {
            "api": "ok",
            "database": "ok",
            "tables": {"prices": {"status": "ok", "count": 1000}},
        }

        result = call_tool(research_server.get_health_status)

        assert result["success"] is True
        assert result["data"]["api"] == "ok"
        assert result["data"]["database"] == "ok"


# =============================================================================
# Lineage Tools Tests
# =============================================================================


class TestGetLineage:
    """Tests for get_lineage tool."""

    def test_get_lineage_success(self, mock_api):
        """Get lineage events."""
        mock_api.get_lineage.return_value = [
            {
                "lineage_id": 1,
                "event_type": "hypothesis_created",
                "timestamp": "2026-01-15T10:00:00",
                "actor": "user",
                "hypothesis_id": "HYP-2026-001",
                "experiment_id": None,
                "details": {},
            }
        ]

        result = call_tool(research_server.get_lineage, hypothesis_id="HYP-2026-001")

        assert result["success"] is True
        assert len(result["data"]) == 1


class TestGetDeployedStrategies:
    """Tests for get_deployed_strategies tool."""

    def test_get_deployed_strategies_success(self, mock_api):
        """Get deployed strategies."""
        mock_api.get_deployed_strategies.return_value = [
            {
                "hypothesis_id": "HYP-2025-010",
                "title": "Deployed strategy",
                "status": "deployed",
                "created_at": "2025-06-01T10:00:00",
            }
        ]

        result = call_tool(research_server.get_deployed_strategies)

        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["status"] == "deployed"


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityConstraints:
    """Tests for security constraints."""

    def test_approve_deployment_not_exposed(self):
        """Verify approve_deployment is NOT exposed as a tool."""
        # Get all tool names from the MCP server
        tool_names = list(research_server.mcp._tool_manager._tools.keys())

        assert "approve_deployment" not in tool_names

    def test_actor_is_agent_identifier(self):
        """Verify ACTOR constant identifies as agent."""
        assert research_server.ACTOR.startswith("agent:")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for tool workflows."""

    def test_hypothesis_to_experiment_workflow(self, mock_api):
        """Test workflow: create hypothesis -> run backtest -> get results."""
        # 1. Create hypothesis
        mock_api.create_hypothesis.return_value = "HYP-2026-001"
        create_result = call_tool(
            research_server.create_hypothesis,
            title="Test hypothesis",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
        )
        assert create_result["success"] is True
        hypothesis_id = create_result["data"]["hypothesis_id"]

        # 2. Run backtest
        mock_api.run_backtest.return_value = "exp-123"
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {"sharpe_ratio": 1.2},
        }
        backtest_result = call_tool(
            research_server.run_backtest,
            hypothesis_id=hypothesis_id,
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )
        assert backtest_result["success"] is True
        experiment_id = backtest_result["data"]["experiment_id"]

        # 3. Get experiment results
        mock_api.get_experiment.return_value = {
            "experiment_id": experiment_id,
            "metrics": {"sharpe_ratio": 1.2, "total_return": 0.3, "max_drawdown": -0.1},
            "params": {},
            "tags": {},
        }
        exp_result = call_tool(research_server.get_experiment, experiment_id)
        assert exp_result["success"] is True
        assert exp_result["data"]["metrics"]["sharpe_ratio"] == 1.2

        # 4. Get linked experiments
        mock_api.get_experiments_for_hypothesis.return_value = [experiment_id]
        linked = call_tool(
            research_server.get_experiments_for_hypothesis,
            hypothesis_id,
        )
        assert linked["success"] is True
        assert experiment_id in linked["data"]["experiment_ids"]


# =============================================================================
# Additional Tool Tests for Coverage
# =============================================================================


class TestGetExperimentsForHypothesis:
    """Tests for get_experiments_for_hypothesis tool."""

    def test_get_experiments_for_hypothesis_success(self, mock_api):
        """Get experiments for a hypothesis."""
        mock_api.get_experiments_for_hypothesis.return_value = ["exp-1", "exp-2", "exp-3"]

        result = call_tool(
            research_server.get_experiments_for_hypothesis,
            "HYP-2026-001",
        )

        assert result["success"] is True
        assert result["data"]["hypothesis_id"] == "HYP-2026-001"
        assert len(result["data"]["experiment_ids"]) == 3
        mock_api.get_experiments_for_hypothesis.assert_called_once_with("HYP-2026-001")

    def test_get_experiments_for_hypothesis_empty(self, mock_api):
        """Get experiments when none exist."""
        mock_api.get_experiments_for_hypothesis.return_value = []

        result = call_tool(
            research_server.get_experiments_for_hypothesis,
            "HYP-2026-999",
        )

        assert result["success"] is True
        assert len(result["data"]["experiment_ids"]) == 0
        assert "0 experiments" in result["message"]


class TestGetPrices:
    """Tests for get_prices tool."""

    def test_get_prices_missing_start_date(self, mock_api):
        """Reject request with missing start_date."""
        result = call_tool(
            research_server.get_prices,
            symbols=["AAPL"],
            start_date=None,
            end_date="2023-12-31",
        )

        assert result["success"] is False
        assert "required" in result["message"].lower()

    def test_get_prices_missing_end_date(self, mock_api):
        """Reject request with missing end_date."""
        result = call_tool(
            research_server.get_prices,
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date=None,
        )

        assert result["success"] is False
        assert "required" in result["message"].lower()

    def test_get_prices_small_dataset(self, mock_api):
        """Return full data for small datasets."""
        mock_df = pd.DataFrame({
            "symbol": ["AAPL"] * 50,
            "date": pd.date_range("2023-01-01", periods=50),
            "close": [150.0] * 50,
        })
        mock_api.get_prices.return_value = mock_df

        result = call_tool(
            research_server.get_prices,
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        assert result["success"] is True
        assert "50 price records" in result["message"]
        # Small dataset returns actual data
        assert isinstance(result["data"], list) or "total_rows" not in result["data"]

    def test_get_prices_large_dataset(self, mock_api):
        """Return summary for large datasets."""
        mock_df = pd.DataFrame({
            "symbol": ["AAPL"] * 200,
            "date": pd.date_range("2023-01-01", periods=200),
            "close": [150.0] * 200,
        })
        mock_api.get_prices.return_value = mock_df

        result = call_tool(
            research_server.get_prices,
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert result["success"] is True
        assert "200 price records" in result["message"]
        # Large dataset returns summary
        assert "summary" in result["message"].lower()
        assert result["data"]["total_rows"] == 200


class TestAnalyzeResults:
    """Tests for analyze_results tool."""

    def test_analyze_results_not_found(self, mock_api):
        """Handle non-existent experiment."""
        mock_api.get_experiment.return_value = None

        result = call_tool(research_server.analyze_results, "exp-999")

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_analyze_results_excellent_sharpe(self, mock_api):
        """Analyze experiment with excellent Sharpe."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 2.0,
                "total_return": 0.40,
                "max_drawdown": -0.10,
                "cagr": 0.20,
            },
            "params": {"method": "momentum"},
        }

        result = call_tool(research_server.analyze_results, "exp-123")

        assert result["success"] is True
        assert "excellent" in result["message"].lower()
        assert result["data"]["summary"]["sharpe_ratio"] == 2.0
        assert result["data"]["summary"]["sharpe_assessment"] == "Excellent risk-adjusted returns"

    def test_analyze_results_good_sharpe(self, mock_api):
        """Analyze experiment with good Sharpe."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 1.2,
                "total_return": 0.25,
                "max_drawdown": -0.15,
                "cagr": 0.12,
            },
            "params": {},
        }

        result = call_tool(research_server.analyze_results, "exp-123")

        assert result["success"] is True
        assert "good" in result["message"].lower()
        assert result["data"]["summary"]["sharpe_assessment"] == "Good risk-adjusted returns"

    def test_analyze_results_moderate_sharpe(self, mock_api):
        """Analyze experiment with moderate Sharpe."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 0.7,
                "total_return": 0.10,
                "max_drawdown": -0.20,
                "cagr": 0.05,
            },
            "params": {},
        }

        result = call_tool(research_server.analyze_results, "exp-123")

        assert result["success"] is True
        assert "moderate" in result["message"].lower()
        assert result["data"]["summary"]["sharpe_assessment"] == "Moderate risk-adjusted returns"

    def test_analyze_results_poor_sharpe(self, mock_api):
        """Analyze experiment with poor Sharpe."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {
                "sharpe_ratio": 0.2,
                "total_return": 0.02,
                "max_drawdown": -0.30,
                "cagr": 0.01,
            },
            "params": {},
        }

        result = call_tool(research_server.analyze_results, "exp-123")

        assert result["success"] is True
        assert "poor" in result["message"].lower()
        assert result["data"]["summary"]["sharpe_assessment"] == "Poor risk-adjusted returns"

    def test_analyze_results_insufficient_data(self, mock_api):
        """Handle experiment with missing metrics."""
        mock_api.get_experiment.return_value = {
            "experiment_id": "exp-123",
            "metrics": {},
            "params": {},
        }

        result = call_tool(research_server.analyze_results, "exp-123")

        assert result["success"] is True
        assert "insufficient" in result["data"]["interpretation"].lower()


class TestRunWalkForwardValidation:
    """Tests for run_walk_forward_validation tool."""

    def test_walk_forward_missing_dates(self, mock_api):
        """Reject request with missing dates."""
        result = call_tool(
            research_server.run_walk_forward_validation,
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            symbols=["AAPL"],
            start_date=None,
            end_date="2023-12-31",
        )

        assert result["success"] is False
        assert "required" in result["message"].lower()

    def test_walk_forward_success(self, mock_api):
        """Run walk-forward validation successfully."""
        # Mock the WalkForwardResult
        from datetime import date as dt_date

        mock_fold = MagicMock()
        mock_fold.fold_index = 0
        mock_fold.train_start = dt_date(2020, 1, 1)
        mock_fold.train_end = dt_date(2021, 12, 31)
        mock_fold.test_start = dt_date(2022, 1, 1)
        mock_fold.test_end = dt_date(2022, 6, 30)
        mock_fold.metrics = {"ic": 0.05, "mse": 0.001}

        mock_result = MagicMock()
        mock_result.stability_score = 0.8
        mock_result.is_stable = True
        mock_result.mean_ic = 0.05
        mock_result.aggregate_metrics = {"mean_ic": 0.05}
        mock_result.fold_results = [mock_fold]

        with patch("hrp.ml.validation.walk_forward_validate", return_value=mock_result):
            result = call_tool(
                research_server.run_walk_forward_validation,
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                symbols=["AAPL"],
                start_date="2020-01-01",
                end_date="2023-12-31",
            )

        assert result["success"] is True
        assert result["data"]["stability_score"] == 0.8
        assert result["data"]["is_stable"] is True
        assert "stable" in result["message"].lower()


class TestGetSupportedModels:
    """Tests for get_supported_models tool."""

    def test_get_supported_models(self, mock_api):
        """List supported models."""
        result = call_tool(research_server.get_supported_models)

        assert result["success"] is True
        assert len(result["data"]) > 0
        # Check that model types are returned
        model_types = [m["model_type"] for m in result["data"]]
        assert "ridge" in model_types or "lasso" in model_types


class TestTrainMlModel:
    """Tests for train_ml_model tool."""

    def test_train_model_missing_dates(self, mock_api):
        """Reject request with missing dates."""
        result = call_tool(
            research_server.train_ml_model,
            model_type="ridge",
            target="returns_20d",
            features=["momentum_20d"],
            symbols=["AAPL"],
            train_start="2020-01-01",
            train_end="2021-12-31",
            validation_start="2022-01-01",
            validation_end=None,  # Missing
            test_start="2023-01-01",
            test_end="2023-12-31",
        )

        assert result["success"] is False
        assert "required" in result["message"].lower()

    def test_train_model_success(self, mock_api):
        """Train model successfully."""
        mock_result = MagicMock()
        mock_result.metrics = {"test_mse": 0.001, "test_r2": 0.85}
        mock_result.selected_features = ["momentum_20d"]
        mock_result.feature_importance = {"momentum_20d": 1.0}

        with patch("hrp.ml.training.train_model", return_value=mock_result):
            result = call_tool(
                research_server.train_ml_model,
                model_type="ridge",
                target="returns_20d",
                features=["momentum_20d"],
                symbols=["AAPL"],
                train_start="2020-01-01",
                train_end="2021-12-31",
                validation_start="2022-01-01",
                validation_end="2022-06-30",
                test_start="2022-07-01",
                test_end="2022-12-31",
            )

        assert result["success"] is True
        assert result["data"]["model_type"] == "ridge"
        assert result["data"]["metrics"]["test_mse"] == 0.001


class TestGetDataCoverage:
    """Tests for get_data_coverage tool."""

    def test_data_coverage_missing_dates(self, mock_api):
        """Reject request with missing dates."""
        result = call_tool(
            research_server.get_data_coverage,
            symbols=["AAPL"],
            start_date=None,
            end_date="2023-12-31",
        )

        assert result["success"] is False
        assert "required" in result["message"].lower()

    def test_data_coverage_success(self, mock_api):
        """Get data coverage successfully."""
        mock_api.get_trading_days.return_value = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        mock_df = pd.DataFrame({
            "symbol": ["AAPL"] * 250,
            "date": pd.date_range("2023-01-01", periods=250),
            "close": [150.0] * 250,
        })
        mock_api.get_prices.return_value = mock_df

        result = call_tool(
            research_server.get_data_coverage,
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert result["success"] is True
        assert result["data"]["total_symbols"] == 1
        coverage_by_symbol = result["data"]["coverage_by_symbol"]
        assert len(coverage_by_symbol) == 1
        assert coverage_by_symbol[0]["symbol"] == "AAPL"
        assert coverage_by_symbol[0]["actual_days"] == 250

    def test_data_coverage_with_error(self, mock_api):
        """Handle coverage check error for a symbol."""
        mock_api.get_trading_days.return_value = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        mock_api.get_prices.side_effect = Exception("Symbol not found")

        result = call_tool(
            research_server.get_data_coverage,
            symbols=["INVALID"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert result["success"] is True
        coverage_by_symbol = result["data"]["coverage_by_symbol"]
        assert len(coverage_by_symbol) == 1
        assert coverage_by_symbol[0]["symbol"] == "INVALID"
        assert coverage_by_symbol[0]["actual_days"] == 0
        assert "error" in coverage_by_symbol[0]
