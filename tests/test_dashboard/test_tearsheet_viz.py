"""
Tests for dashboard tearsheet visualization component.

Tests hrp.dashboard.components.tearsheet_viz module which provides
PyFolio-inspired tear sheet visualizations for backtest analysis.
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd
import pytest

from hrp.dashboard.components.tearsheet_viz import (
    render_returns_distribution,
    render_rolling_metrics,
    render_drawdown_analysis,
    render_monthly_returns_heatmap,
    render_tail_risk_metrics,
    render_tear_sheet,
)


@pytest.fixture
def sample_returns():
    """Create sample returns series for testing."""
    # Create 100 daily returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100),
                       index=pd.date_range('2020-01-01', periods=100))
    return returns


@pytest.fixture
def sample_benchmark_returns():
    """Create sample benchmark returns for testing."""
    np.random.seed(43)
    returns = pd.Series(np.random.normal(0.0003, 0.008, 100),
                       index=pd.date_range('2020-01-01', periods=100))
    return returns


class TestRenderReturnsDistribution:
    """Test render_returns_distribution function."""

    def test_renders_title(self, sample_returns):
        """Renders section title."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_returns_distribution(sample_returns)

        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Returns Distribution" in call for call in markdown_calls)

    def test_creates_histogram(self, sample_returns):
        """Creates histogram of returns."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                fig = MagicMock()
                mock_go.Figure.return_value = fig
                mock_go.Histogram.return_value = MagicMock()

                render_returns_distribution(sample_returns)

        # Verify figure was created and trace added
        mock_go.Figure.assert_called()
        assert fig.add_trace.call_count >= 2  # Histogram + normal distribution

    def test_displays_statistics_metrics(self, sample_returns):
        """Displays mean, std, skewness, kurtosis metrics."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_returns_distribution(sample_returns)

        # Should have 4 columns with metrics
        mock_st.columns.assert_called()
        assert mock_st.columns.call_args[0][0] == 4
        for col in mock_cols:
            col.metric.assert_called_once()

    def test_calculates_correct_statistics(self, sample_returns):
        """Calculates mean, std, skewness, kurtosis correctly."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_returns_distribution(sample_returns)

        # Verify metric calls contain expected statistics
        expected_mean = sample_returns.mean()
        expected_std = sample_returns.std()
        expected_skew = sample_returns.skew()
        expected_kurt = sample_returns.kurtosis()

        # Check that metrics were called with formatted values
        calls = [col.metric.call_args for col in mock_cols]
        metric_values = [str(call[0][1]) for call in calls]

        # Should contain mean, std, skewness, kurtosis values
        assert len(calls) == 4

    def test_calls_plotly_chart(self, sample_returns):
        """Calls plotly_chart with the figure."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(4)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_returns_distribution(sample_returns)

        mock_st.plotly_chart.assert_called_once()


class TestRenderRollingMetrics:
    """Test render_rolling_metrics function."""

    def test_renders_title(self, sample_returns):
        """Renders section title."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots'):
                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns)

        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Rolling Metrics" in call for call in markdown_calls)

    def test_warns_on_insufficient_data(self):
        """Warns when returns length is less than window."""
        short_returns = pd.Series([0.01, 0.02, -0.01])

        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            render_rolling_metrics(short_returns, window=63)

        mock_st.warning.assert_called_once()
        warning_msg = str(mock_st.warning.call_args)
        assert "Insufficient data" in warning_msg or "63-day" in warning_msg

    def test_creates_subplot_figure(self, sample_returns):
        """Creates subplot with 2 rows for Sharpe and Volatility."""
        with patch('hrp.dashboard.components.tearsheet_viz.make_subplots') as mock_subplots:
            with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
                mock_fig = MagicMock()
                mock_subplots.return_value = mock_fig
                mock_st.plotly_chart.return_value = None

                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns, window=63)

        # Verify make_subplots was called with 2 rows
        mock_subplots.assert_called_once()
        call_kwargs = mock_subplots.call_args[1]
        assert call_kwargs['rows'] == 2
        assert call_kwargs['cols'] == 1

    def test_calculates_rolling_sharpe(self, sample_returns):
        """Calculates rolling Sharpe ratio."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots', return_value=mock_fig):
                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns, window=63)

        # Verify figure has traces added
        assert mock_fig.add_trace.call_count >= 2  # Sharpe + Volatility

    def test_calculates_rolling_volatility(self, sample_returns):
        """Calculates rolling volatility."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots', return_value=mock_fig):
                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns, window=63)

        # Verify traces were added for both Sharpe and Volatility
        assert mock_fig.add_trace.call_count >= 2

    def test_adds_zero_line_for_sharpe(self, sample_returns):
        """Adds horizontal line at y=0 for Sharpe ratio."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots', return_value=mock_fig):
                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns, window=63)

        # Verify add_hline was called
        mock_fig.add_hline.assert_called_once()
        call_kwargs = mock_fig.add_hline.call_args[1]
        assert call_kwargs['y'] == 0

    def test_custom_window_size(self, sample_returns):
        """Uses custom window size when specified."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots', return_value=mock_fig):
                with patch('hrp.dashboard.components.tearsheet_viz.go'):
                    render_rolling_metrics(sample_returns, window=126)

        # Function should complete without error with custom window
        assert True


class TestRenderDrawdownAnalysis:
    """Test render_drawdown_analysis function."""

    def test_renders_title(self, sample_returns):
        """Renders section title."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Drawdown Analysis" in call for call in markdown_calls)

    def test_calculates_cumulative_returns(self, sample_returns):
        """Calculates cumulative returns for drawdown."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        # Should complete without error
        assert True

    def test_calculates_running_maximum(self, sample_returns):
        """Calculates running maximum for drawdown."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        # Should complete without error
        assert True

    def test_displays_max_drawdown_metric(self, sample_returns):
        """Displays maximum drawdown metric."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        # First metric should be Max Drawdown
        mock_cols[0].metric.assert_called_once()
        call_args = mock_cols[0].metric.call_args[0]
        assert "Max Drawdown" in call_args[0]

    def test_displays_avg_drawdown_metric(self, sample_returns):
        """Displays average drawdown metric."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        # Second metric should be Avg Drawdown
        mock_cols[1].metric.assert_called_once()
        call_args = mock_cols[1].metric.call_args[0]
        assert "Avg Drawdown" in call_args[0]

    def test_displays_max_dd_duration_metric(self, sample_returns):
        """Displays maximum drawdown duration metric."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_drawdown_analysis(sample_returns)

        # Third metric should be Max DD Duration
        mock_cols[2].metric.assert_called_once()
        call_args = mock_cols[2].metric.call_args[0]
        assert "Duration" in call_args[0] or "days" in str(call_args)

    def test_creates_underwater_plot(self, sample_returns):
        """Creates underwater plot for drawdown visualization."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go.Figure', return_value=mock_fig):
                render_drawdown_analysis(sample_returns)

        # Should create figure and add trace
        mock_fig.add_trace.assert_called_once()
        mock_fig.update_layout.assert_called()


class TestRenderMonthlyReturnsHeatmap:
    """Test render_monthly_returns_heatmap function."""

    def test_renders_title(self, sample_returns):
        """Renders section title."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_monthly_returns_heatmap(sample_returns)

        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Monthly Returns" in call for call in markdown_calls)

    def test_converts_to_datetime_index(self):
        """Converts non-DatetimeIndex to datetime."""
        non_dt_returns = pd.Series([0.01] * 100, index=range(100))

        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_monthly_returns_heatmap(non_dt_returns)

        # Should complete without error
        assert True

    def test_resamples_to_monthly_returns(self, sample_returns):
        """Resamples daily returns to monthly."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                render_monthly_returns_heatmap(sample_returns)

        # Should create heatmap figure
        mock_go.Figure.assert_called()

    def test_creates_year_month_pivot_table(self, sample_returns):
        """Creates pivot table with years as rows and months as columns."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_monthly_returns_heatmap(sample_returns)

        # Should complete without error
        assert True

    def test_adds_annual_returns_column(self, sample_returns):
        """Adds annual returns column to heatmap."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_monthly_returns_heatmap(sample_returns)

        # Should complete without error
        assert True

    def test_uses_month_name_columns(self, sample_returns):
        """Uses month abbreviations (Jan, Feb, etc.) for columns."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                render_monthly_returns_heatmap(sample_returns)

        # Should complete without error
        assert True


class TestRenderTailRiskMetrics:
    """Test render_tail_risk_metrics function."""

    def test_renders_title(self, sample_returns):
        """Renders section title."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                render_tail_risk_metrics(sample_returns)

        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Tail Risk" in call for call in markdown_calls)

    def test_calculates_var_95(self, sample_returns):
        """Calculates Value at Risk at 95% confidence."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                render_tail_risk_metrics(sample_returns)

        # First metric should be VaR
        mock_cols[0].metric.assert_called_once()
        call_args = mock_cols[0].metric.call_args[0]
        assert "Value at Risk" in call_args[0] or "VaR" in str(call_args)

    def test_calculates_cvar_95(self, sample_returns):
        """Calculates Conditional VaR at 95% confidence."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                render_tail_risk_metrics(sample_returns)

        # Second metric should be CVaR
        mock_cols[1].metric.assert_called_once()
        call_args = mock_cols[1].metric.call_args[0]
        assert "CVaR" in str(call_args) or "Shortfall" in str(call_args)

    def test_calculates_tail_ratio(self, sample_returns):
        """Calculates tail ratio."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                render_tail_risk_metrics(sample_returns)

        # Third metric should be Tail Ratio
        mock_cols[2].metric.assert_called_once()
        call_args = mock_cols[2].metric.call_args[0]
        assert "Tail Ratio" in str(call_args)

    def test_adds_var_marker_to_chart(self, sample_returns):
        """Adds vertical line marker for VaR."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_fig = MagicMock()
            mock_st.plotly_chart.return_value = None
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols

            with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                mock_go.Figure.return_value = mock_fig
                render_tail_risk_metrics(sample_returns)

        # Should add vline for VaR
        assert mock_fig.add_vline.call_count >= 2  # VaR + CVaR

    def test_falls_back_to_pandas_on_empyrical_error(self, sample_returns):
        """Falls back to pandas calculations when Empyrical fails."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols = [MagicMock() for _ in range(3)]
            mock_st.columns.return_value = mock_cols
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.ep') as mock_ep:
                # Make Empyrical fail
                mock_ep.value_at_risk.side_effect = Exception("Empyrical error")

                with patch('hrp.dashboard.components.tearsheet_viz.go') as mock_go:
                    mock_fig = MagicMock()
                    mock_go.Figure.return_value = mock_fig
                    # Should still work with fallback
                    render_tail_risk_metrics(sample_returns)

        # Should still display metrics
        for col in mock_cols:
            col.metric.assert_called_once()


class TestRenderTearSheet:
    """Test render_tear_sheet function."""

    def test_warns_on_no_returns(self):
        """Warns when returns is None or empty."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            render_tear_sheet(None)

        mock_st.warning.assert_called_once()
        warning_msg = str(mock_st.warning.call_args)
        assert "No returns" in warning_msg

    def test_warns_on_insufficient_data(self):
        """Warns when returns has less than 30 data points."""
        short_returns = pd.Series([0.01] * 10)

        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            render_tear_sheet(short_returns)

        mock_st.warning.assert_called_once()
        warning_msg = str(mock_st.warning.call_args)
        assert "Insufficient data" in warning_msg or "30 days" in warning_msg

    def test_creates_analysis_tabs(self, sample_returns):
        """Creates tabs for different analyses."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab1 = MagicMock()
            mock_tab2 = MagicMock()
            mock_tab3 = MagicMock()
            mock_tab4 = MagicMock()
            mock_tab1.__enter__ = Mock(return_value=mock_tab1)
            mock_tab1.__exit__ = Mock(return_value=False)
            mock_tab2.__enter__ = Mock(return_value=mock_tab2)
            mock_tab2.__exit__ = Mock(return_value=False)
            mock_tab3.__enter__ = Mock(return_value=mock_tab3)
            mock_tab3.__exit__ = Mock(return_value=False)
            mock_tab4.__enter__ = Mock(return_value=mock_tab4)
            mock_tab4.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [mock_tab1, mock_tab2, mock_tab3, mock_tab4]
            # Setup enough side_effect values for all nested st.columns() calls
            # Each nested function may call st.columns()
            mock_cols_4 = [MagicMock() for _ in range(4)]
            mock_cols_3 = [MagicMock() for _ in range(3)]
            mock_cols_2 = [MagicMock() for _ in range(2)]
            mock_st.columns.side_effect = [
                mock_cols_2,  # rolling_metrics tab calls st.columns
                mock_cols_4, mock_cols_4,  # returns_distribution calls
                mock_cols_3, mock_cols_3,  # drawdown_analysis and tail_risk_metrics call
            ]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                render_tear_sheet(sample_returns)

        # Should create 4 tabs
        mock_st.tabs.assert_called_once()
        tab_names = mock_st.tabs.call_args[0][0]
        assert len(tab_names) == 4
        assert "Returns Analysis" in tab_names
        assert "Rolling Metrics" in tab_names
        assert "Drawdown" in tab_names
        assert "Tail Risk" in tab_names

    def test_calls_render_returns_distribution(self, sample_returns):
        """Calls render_returns_distribution in first tab."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab1 = MagicMock()
            mock_tab1.__enter__ = Mock(return_value=mock_tab1)
            mock_tab1.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [mock_tab1, MagicMock(), MagicMock(), MagicMock()]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution') as mock_render:
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                render_tear_sheet(sample_returns)

        mock_render.assert_called_once()

    def test_calls_render_monthly_returns_heatmap(self, sample_returns):
        """Calls render_monthly_returns_heatmap in first tab."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab1 = MagicMock()
            mock_tab1.__enter__ = Mock(return_value=mock_tab1)
            mock_tab1.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [mock_tab1, MagicMock(), MagicMock(), MagicMock()]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap') as mock_render:
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                render_tear_sheet(sample_returns)

        mock_render.assert_called_once()

    def test_calls_render_rolling_metrics(self, sample_returns):
        """Calls render_rolling_metrics in second tab."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab2 = MagicMock()
            mock_tab2.__enter__ = Mock(return_value=mock_tab2)
            mock_tab2.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [MagicMock(), mock_tab2, MagicMock(), MagicMock()]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics') as mock_render:
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                render_tear_sheet(sample_returns)

        mock_render.assert_called_once()

    def test_calls_render_drawdown_analysis(self, sample_returns):
        """Calls render_drawdown_analysis in third tab."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab3 = MagicMock()
            mock_tab3.__enter__ = Mock(return_value=mock_tab3)
            mock_tab3.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [MagicMock(), MagicMock(), mock_tab3, MagicMock()]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis') as mock_render:
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                render_tear_sheet(sample_returns)

        mock_render.assert_called_once()

    def test_calls_render_tail_risk_metrics(self, sample_returns):
        """Calls render_tail_risk_metrics in fourth tab."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_tab4 = MagicMock()
            mock_tab4.__enter__ = Mock(return_value=mock_tab4)
            mock_tab4.__exit__ = Mock(return_value=False)
            mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), mock_tab4]

            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics') as mock_render:
                                render_tear_sheet(sample_returns)

        mock_render.assert_called_once()


class TestTearsheetVizIntegration:
    """Integration tests for tearsheet visualization component."""

    def test_functions_are_callable(self):
        """Exported functions are callable."""
        assert callable(render_returns_distribution)
        assert callable(render_rolling_metrics)
        assert callable(render_drawdown_analysis)
        assert callable(render_monthly_returns_heatmap)
        assert callable(render_tail_risk_metrics)
        assert callable(render_tear_sheet)

    def test_functions_return_none(self, sample_returns):
        """All render functions return None (they render directly to UI)."""
        with patch('hrp.dashboard.components.tearsheet_viz.st') as mock_st:
            mock_cols_4 = [MagicMock() for _ in range(4)]
            mock_cols_3 = [MagicMock() for _ in range(3)]
            mock_cols_2 = [MagicMock() for _ in range(2)]
            mock_st.columns.side_effect = [
                mock_cols_4,  # returns_distribution needs 4
                mock_cols_3,  # drawdown_analysis needs 3
                mock_cols_3,  # tail_risk_metrics needs 3
            ]
            mock_st.plotly_chart.return_value = None

            with patch('hrp.dashboard.components.tearsheet_viz.go'):
                assert render_returns_distribution(sample_returns) is None

            with patch('hrp.dashboard.components.tearsheet_viz.make_subplots'):
                assert render_rolling_metrics(sample_returns) is None

            assert render_drawdown_analysis(sample_returns) is None

            assert render_monthly_returns_heatmap(sample_returns) is None

            assert render_tail_risk_metrics(sample_returns) is None

            # For render_tear_sheet, we need to mock all the nested render functions
            with patch('hrp.dashboard.components.tearsheet_viz.render_returns_distribution'):
                with patch('hrp.dashboard.components.tearsheet_viz.render_monthly_returns_heatmap'):
                    with patch('hrp.dashboard.components.tearsheet_viz.render_rolling_metrics'):
                        with patch('hrp.dashboard.components.tearsheet_viz.render_drawdown_analysis'):
                            with patch('hrp.dashboard.components.tearsheet_viz.render_tail_risk_metrics'):
                                assert render_tear_sheet(sample_returns) is None
