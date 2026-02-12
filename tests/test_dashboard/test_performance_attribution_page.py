"""Tests for Performance Attribution Dashboard page."""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the module directly to avoid circular import issues with __init__.py
import importlib.util
import sys

_module_path = project_root / "hrp" / "dashboard" / "pages" / "12_Performance_Attribution.py"
_spec = importlib.util.spec_from_file_location("performance_attribution", _module_path)
performance_attribution = importlib.util.module_from_spec(_spec)
sys.modules["performance_attribution"] = performance_attribution
_spec.loader.exec_module(performance_attribution)
from hrp.data.attribution.attribution_config import AttributionConfig, AttributionMethod, ImportanceMethod
from hrp.data.attribution.factor_attribution import AttributionResult
from hrp.data.attribution.decision_attribution import TradeDecision


class TestPerformanceAttributionPageStructure:
    """Test that the dashboard page has the expected structure."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_page_has_title(self, mock_api, mock_st):
        """Test that page renders with title."""
        # Arrange
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Act
        performance_attribution.render()

        # Assert
        mock_st.title.assert_called_once()
        assert "Performance Attribution" in str(mock_st.title.call_args)

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_page_has_caption(self, mock_api, mock_st):
        """Test that page has descriptive caption."""
        # Arrange
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Act
        performance_attribution.render()

        # Assert
        mock_st.caption.assert_called()

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_page_has_config_section(self, mock_api, mock_st):
        """Test that configuration section is rendered."""
        # Arrange
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Act
        performance_attribution.render()

        # Assert
        # Check that selectbox is called for config controls
        assert mock_st.selectbox.call_count >= 2  # Method + Importance

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_page_has_all_sections(self, mock_api, mock_st):
        """Test that all major sections are rendered."""
        # Arrange
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        mock_api_instance.query_readonly.return_value = pd.DataFrame()

        # Act
        performance_attribution.render()

        # Assert
        # Check that all subheader sections are called
        subheader_calls = [call[0][0] for call in mock_st.subheader.call_args_list]
        expected_sections = [
            "Attribution Configuration",
            "Performance Summary",
            "Return Attribution Waterfall",
            "Factor Contribution Details",
            "Feature Importance Over Time",
            "Decision Attribution Timeline",
            "Multi-Period Comparison",
        ]

        for section in expected_sections:
            assert any(section in call for call in subheader_calls), f"Missing section: {section}"


class TestConfigSection:
    """Test configuration section rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    def test_render_config_section(self, mock_st):
        """Test config section returns valid AttributionConfig."""
        # Arrange
        mock_st.date_input.side_effect = [
            date.today() - timedelta(days=90),
            date.today(),
        ]
        mock_st.selectbox.side_effect = [
            "brinson",  # method
            "permutation",  # importance_method
        ]

        # Act
        config = performance_attribution._render_config_section()

        # Assert
        assert isinstance(config, AttributionConfig)
        assert config.method == AttributionMethod.BRINSON
        assert config.importance_method == ImportanceMethod.PERMUTATION

    @patch("hrp.dashboard.pages.performance_attribution.st")
    def test_config_section_has_date_inputs(self, mock_st):
        """Test that config section has start and end date inputs."""
        # Arrange
        mock_st.date_input.side_effect = [date.today() - timedelta(days=90), date.today()]
        mock_st.selectbox.side_effect = ["brinson", "permutation"]

        # Act
        performance_attribution._render_config_section()

        # Assert
        assert mock_st.date_input.call_count == 2


class TestSummaryBar:
    """Test summary metrics bar rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._get_portfolio_returns")
    @patch("hrp.dashboard.pages.performance_attribution._get_benchmark_returns")
    def test_summary_bar_with_data(self, mock_benchmark, mock_portfolio, mock_st):
        """Test summary bar renders with valid data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            method=AttributionMethod.BRINSON,
        )

        # Mock returns
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        mock_portfolio.return_value = pd.Series(np.random.normal(0.001, 0.01, len(dates)), index=dates)
        mock_benchmark.return_value = pd.Series(np.random.normal(0.0008, 0.008, len(dates)), index=dates)

        # Act
        performance_attribution._render_summary_bar(mock_api, config)

        # Assert
        # Should call st.metric for portfolio return, benchmark return, active return, period
        assert mock_st.metric.call_count == 4

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._get_portfolio_returns")
    @patch("hrp.dashboard.pages.performance_attribution._get_benchmark_returns")
    def test_summary_bar_with_no_data(self, mock_benchmark, mock_portfolio, mock_st):
        """Test summary bar handles missing data gracefully."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )

        mock_portfolio.return_value = None
        mock_benchmark.return_value = None

        # Act
        performance_attribution._render_summary_bar(mock_api, config)

        # Assert
        mock_st.warning.assert_called_once()


class TestWaterfallChart:
    """Test waterfall chart rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    @patch("hrp.dashboard.pages.performance_attribution._get_portfolio_returns")
    @patch("hrp.dashboard.pages.performance_attribution._get_benchmark_returns")
    def test_waterfall_chart_with_data(self, mock_benchmark, mock_portfolio, mock_attribution, mock_st):
        """Test waterfall chart renders with valid attribution data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )

        # Mock returns
        dates = pd.date_range(config.start_date, config.end_date, freq="D")
        mock_portfolio.return_value = pd.Series(np.ones(len(dates)) * 0.001, index=dates)
        mock_benchmark.return_value = pd.Series(np.ones(len(dates)) * 0.0008, index=dates)

        # Mock attribution results
        mock_attribution.return_value = [
            AttributionResult(factor="Tech", effect_type="allocation", contribution_pct=0.002),
            AttributionResult(factor="Tech", effect_type="selection", contribution_pct=0.001),
        ]

        # Act
        performance_attribution._render_waterfall_chart(mock_api, config)

        # Assert
        mock_st.plotly_chart.assert_called_once()

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    def test_waterfall_chart_with_no_data(self, mock_attribution, mock_st):
        """Test waterfall chart handles empty attribution."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_attribution.return_value = []

        # Act
        performance_attribution._render_waterfall_chart(mock_api, config)

        # Assert
        mock_st.info.assert_called()


class TestFactorContributionTable:
    """Test factor contribution table rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    def test_factor_table_with_data(self, mock_attribution, mock_st):
        """Test factor table renders with attribution results."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())

        mock_attribution.return_value = [
            AttributionResult(
                factor="Technology",
                effect_type="allocation",
                contribution_pct=0.003,
                contribution_dollar=1500.0,
            ),
            AttributionResult(
                factor="Healthcare",
                effect_type="selection",
                contribution_pct=-0.001,
                contribution_dollar=-500.0,
            ),
        ]

        # Act
        performance_attribution._render_factor_contribution_table(mock_api, config)

        # Assert
        mock_st.dataframe.assert_called_once()
        mock_st.download_button.assert_called_once()


class TestFeatureImportanceHeatmap:
    """Test feature importance heatmap rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_feature_importance")
    def test_heatmap_with_data(self, mock_importance, mock_st):
        """Test heatmap renders with feature importance data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today() - timedelta(days=30), end_date=date.today())

        # Mock feature importance DataFrame
        dates = pd.date_range(config.start_date, config.end_date, freq="W")
        features = ["momentum", "volatility", "rsi"]
        data = {feat: np.random.rand(len(dates)) for feat in features}
        mock_importance.return_value = pd.DataFrame(data, index=dates)

        # Act
        performance_attribution._render_feature_importance_heatmap(mock_api, config)

        # Assert
        mock_st.plotly_chart.assert_called_once()
        # Should also display top features
        assert mock_st.caption.call_count >= 1

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_feature_importance")
    def test_heatmap_with_no_data(self, mock_importance, mock_st):
        """Test heatmap handles empty data gracefully."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_importance.return_value = None

        # Act
        performance_attribution._render_feature_importance_heatmap(mock_api, config)

        # Assert
        mock_st.info.assert_called()


class TestDecisionAttributionTimeline:
    """Test decision attribution timeline rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_decision_attribution")
    def test_timeline_with_data(self, mock_decisions, mock_st):
        """Test timeline renders with trade decision data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today() - timedelta(days=30), end_date=date.today())

        mock_decisions.return_value = [
            TradeDecision(
                trade_id="T1",
                asset="AAPL",
                entry_date=date.today() - timedelta(days=10),
                exit_date=date.today() - timedelta(days=5),
                pnl=1000.0,
                timing_contribution=400.0,
                sizing_contribution=300.0,
                residual_contribution=300.0,
            ),
        ]

        # Act
        performance_attribution._render_decision_attribution_timeline(mock_api, config)

        # Assert
        mock_st.plotly_chart.assert_called_once()
        # Should display summary metrics
        assert mock_st.metric.call_count == 3

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_decision_attribution")
    def test_timeline_with_no_data(self, mock_decisions, mock_st):
        """Test timeline handles empty decisions."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_decisions.return_value = []

        # Act
        performance_attribution._render_decision_attribution_timeline(mock_api, config)

        # Assert
        mock_st.info.assert_called()


class TestPeriodComparison:
    """Test multi-period comparison rendering."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    def test_period_comparison_with_data(self, mock_attribution, mock_st):
        """Test period comparison renders with attribution data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today() - timedelta(days=365), end_date=date.today())

        # Return some results for each period
        mock_attribution.return_value = [
            AttributionResult(factor="Tech", effect_type="allocation", contribution_pct=0.01),
            AttributionResult(factor="Tech", effect_type="selection", contribution_pct=0.005),
            AttributionResult(factor="Tech", effect_type="interaction", contribution_pct=0.002),
        ]

        # Act
        performance_attribution._render_period_comparison(mock_api, config)

        # Assert
        mock_st.plotly_chart.assert_called_once()

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    def test_period_comparison_with_no_data(self, mock_attribution, mock_st):
        """Test period comparison handles empty data."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_attribution.return_value = []

        # Act
        performance_attribution._render_period_comparison(mock_api, config)

        # Assert
        mock_st.info.assert_called()


class TestHelperFunctions:
    """Test helper/utility functions."""

    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_get_portfolio_returns_with_data(self, mock_api_class):
        """Test portfolio returns retrieval with database data."""
        # Arrange
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        config = AttributionConfig(
            start_date=date.today() - timedelta(days=10),
            end_date=date.today(),
        )

        # Mock database response
        mock_api.query_readonly.return_value = pd.DataFrame({
            "date": pd.date_range(config.start_date, config.end_date, freq="D"),
            "portfolio_return": np.random.normal(0.001, 0.01, 11),
        })

        # Act
        result = performance_attribution._get_portfolio_returns(mock_api, config)

        # Assert
        assert result is not None
        assert len(result) == 11
        assert result.name == "portfolio_return"

    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_get_portfolio_returns_fallback(self, mock_api_class):
        """Test portfolio returns falls back to synthetic data."""
        # Arrange
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        config = AttributionConfig(
            start_date=date.today() - timedelta(days=10),
            end_date=date.today(),
        )

        # Mock empty database response
        mock_api.query_readonly.return_value = pd.DataFrame()

        # Act
        result = performance_attribution._get_portfolio_returns(mock_api, config)

        # Assert
        assert result is not None
        assert len(result) == 11  # 10 days + 1 for inclusive range

    @patch("hrp.dashboard.pages.performance_attribution.PlatformAPI")
    def test_get_benchmark_returns_with_data(self, mock_api_class):
        """Test benchmark returns retrieval with database data."""
        # Arrange
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        config = AttributionConfig(
            start_date=date.today() - timedelta(days=10),
            end_date=date.today(),
        )

        mock_api.query_readonly.return_value = pd.DataFrame({
            "date": pd.date_range(config.start_date, config.end_date, freq="D"),
            "return": np.random.normal(0.0005, 0.008, 11),
        })

        # Act
        result = performance_attribution._get_benchmark_returns(mock_api, config)

        # Assert
        assert result is not None
        assert len(result) == 11

    def test_calculate_attribution_brinson_method(self):
        """Test attribution calculation with Brinson method."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            method=AttributionMethod.BRINSON,
        )

        # Act
        results = performance_attribution._calculate_attribution(mock_api, config)

        # Assert
        assert isinstance(results, list)
        # Should have results for multiple sectors (Technology, Healthcare, etc.)
        assert len(results) > 0

    def test_calculate_attribution_regression_method(self):
        """Test attribution calculation with regression method."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            method=AttributionMethod.REGRESSION,
        )

        # Act
        results = performance_attribution._calculate_attribution(mock_api, config)

        # Assert
        assert isinstance(results, list)

    def test_calculate_feature_importance(self):
        """Test feature importance calculation returns valid DataFrame."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )

        # Act
        result = performance_attribution._calculate_feature_importance(mock_api, config)

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should have features as columns
        assert len(result.columns) > 0
        # Each row should sum to approximately 1 (normalized)
        row_sums = result.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)

    def test_calculate_decision_attribution(self):
        """Test decision attribution calculation returns valid decisions."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )

        # Act
        results = performance_attribution._calculate_decision_attribution(mock_api, config)

        # Assert
        assert isinstance(results, list)
        # Should have some trades
        assert len(results) > 0
        # Each trade should have required fields
        for decision in results:
            assert isinstance(decision, TradeDecision)
            assert decision.trade_id is not None
            assert decision.pnl is not None
            # P&L decomposition should sum correctly
            total_contrib = (
                decision.timing_contribution
                + decision.sizing_contribution
                + decision.residual_contribution
            )
            assert np.isclose(total_contrib, decision.pnl, atol=0.01)


class TestErrorHandling:
    """Test error handling in dashboard functions."""

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._get_portfolio_returns")
    def test_summary_bar_handles_exception(self, mock_returns, mock_st):
        """Test summary bar handles exceptions gracefully."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_returns.side_effect = Exception("Database error")

        # Act
        performance_attribution._render_summary_bar(mock_api, config)

        # Assert
        mock_st.error.assert_called()

    @patch("hrp.dashboard.pages.performance_attribution.st")
    @patch("hrp.dashboard.pages.performance_attribution._calculate_attribution")
    def test_waterfall_chart_handles_exception(self, mock_attribution, mock_st):
        """Test waterfall chart handles exceptions gracefully."""
        # Arrange
        mock_api = MagicMock()
        config = AttributionConfig(start_date=date.today(), end_date=date.today())
        mock_attribution.side_effect = Exception("Calculation error")

        # Act
        performance_attribution._render_waterfall_chart(mock_api, config)

        # Assert
        mock_st.error.assert_called()
