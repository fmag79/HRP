"""
Tests for Report Generator agent.

Tests the report generation functionality including data gathering,
insights generation, and report rendering.
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.report_generator import ReportGenerator, ReportGeneratorConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def report_config():
    """Create a test ReportGeneratorConfig."""
    return ReportGeneratorConfig(
        report_type="daily",
        report_dir="/tmp/test_reports",
    )


@pytest.fixture
def daily_generator(report_config):
    """Create a daily ReportGenerator for testing."""
    return ReportGenerator(report_type="daily", config=report_config)


@pytest.fixture
def weekly_generator():
    """Create a weekly ReportGenerator for testing."""
    return ReportGenerator(report_type="weekly")


# =============================================================================
# Test Config
# =============================================================================

class TestReportGeneratorConfig:
    """Tests for ReportGeneratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportGeneratorConfig()
        assert config.report_type == "daily"
        assert config.report_dir == "docs/reports"
        assert len(config.include_sections) == 6
        assert config.lookback_days == 7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReportGeneratorConfig(
            report_type="weekly",
            report_dir="custom/reports",
            lookback_days=14,
        )
        assert config.report_type == "weekly"
        assert config.report_dir == "custom/reports"
        assert config.lookback_days == 14


# =============================================================================
# Test Initialization
# =============================================================================

class TestReportGeneratorInit:
    """Tests for ReportGenerator initialization."""

    def test_daily_initialization(self):
        """Test daily report generator initialization."""
        generator = ReportGenerator(report_type="daily")
        assert generator.report_type == "daily"
        assert generator.ACTOR == "agent:report-generator"

    def test_weekly_initialization(self):
        """Test weekly report generator initialization."""
        generator = ReportGenerator(report_type="weekly")
        assert generator.report_type == "weekly"

    def test_custom_config_initialization(self, report_config):
        """Test initialization with custom config."""
        generator = ReportGenerator(
            report_type="weekly",
            config=report_config,
        )
        assert generator.report_type == "weekly"
        assert generator.config.report_dir == "/tmp/test_reports"

    def test_inherits_from_sdk_agent(self, daily_generator):
        """Test that ReportGenerator extends SDKAgent."""
        from hrp.agents.sdk_agent import SDKAgent
        assert isinstance(daily_generator, SDKAgent)

    def test_api_initialization(self, daily_generator):
        """Test that PlatformAPI is initialized."""
        assert daily_generator.api is not None


# =============================================================================
# Test Data Gathering (stubs for now)
# =============================================================================

class TestGatherHypothesisData:
    """Tests for hypothesis data gathering."""

    def test_gather_hypothesis_data_returns_dict(self, daily_generator):
        """Test that hypothesis data gathering returns a dict."""
        result = daily_generator._gather_hypothesis_data()
        assert isinstance(result, dict)
        assert "draft" in result
        assert "testing" in result
        assert "validated" in result
        assert "total_counts" in result


class TestGatherExperimentData:
    """Tests for experiment data gathering."""

    def test_gather_experiment_data_returns_dict(self, daily_generator):
        """Test that experiment data gathering returns a dict."""
        result = daily_generator._gather_experiment_data()
        assert isinstance(result, dict)
        assert "total_experiments" in result
        assert "top_experiments" in result
        assert "model_performance" in result


class TestGatherSignalData:
    """Tests for signal data gathering."""

    def test_gather_signal_data_returns_dict(self, daily_generator):
        """Test that signal data gathering returns a dict."""
        result = daily_generator._gather_signal_data()
        assert isinstance(result, dict)
        assert "recent_discoveries" in result
        assert "best_signals" in result


class TestGatherAgentActivity:
    """Tests for agent activity gathering."""

    def test_gather_agent_activity_returns_dict(self, daily_generator):
        """Test that agent activity gathering returns a dict."""
        result = daily_generator._gather_agent_activity()
        assert isinstance(result, dict)
        assert "signal_scientist" in result
        assert "alpha_researcher" in result
        assert "ml_scientist" in result
        assert "ml_quality_sentinel" in result
        assert "validation_analyst" in result


# =============================================================================
# Test Insights Generation (stubs for now)
# =============================================================================

class TestGenerateInsights:
    """Tests for insights generation."""

    def test_generate_insights_returns_list(self, daily_generator):
        """Test that insights generation returns a list."""
        context = {
            "hypothesis_data": {"total_counts": {"draft": 0, "testing": 0, "validated": 0, "deployed": 0}},
            "experiment_data": {"total_experiments": 0, "top_experiments": [], "model_performance": {}},
            "signal_data": {"best_signals": []},
            "agent_activity": {},
        }
        result = daily_generator._generate_insights(context)
        assert isinstance(result, list)


# =============================================================================
# Test Report Rendering (stubs for now)
# =============================================================================

class TestRenderDailyReport:
    """Tests for daily report rendering."""

    def test_render_daily_report_returns_string(self, daily_generator):
        """Test that daily report rendering returns a string."""
        context = {
            "hypothesis_data": {
                "draft": [],
                "testing": [],
                "validated": [],
                "deployed": [],
                "total_counts": {"draft": 0, "testing": 0, "validated": 0, "deployed": 0},
            },
            "experiment_data": {"total_experiments": 0, "top_experiments": [], "model_performance": {}},
            "signal_data": {"best_signals": []},
            "agent_activity": {"signal_scientist": {"status": "pending"}, "alpha_researcher": {"status": "pending"}, "ml_scientist": {"status": "pending"}, "ml_quality_sentinel": {"status": "pending"}, "validation_analyst": {"status": "pending"}},
            "report_type": "daily",
            "generated_at": datetime(2026, 1, 26, 7, 0),
        }
        insights = [{"priority": "low", "category": "general", "action": "No issues"}]
        result = daily_generator._render_daily_report(context, insights)
        assert isinstance(result, str)
        assert "# HRP Research Report" in result


class TestRenderWeeklyReport:
    """Tests for weekly report rendering."""

    def test_render_weekly_report_returns_string(self, weekly_generator):
        """Test that weekly report rendering returns a string."""
        context = {
            "hypothesis_data": {
                "draft": [],
                "testing": [],
                "validated": [],
                "deployed": [],
                "total_counts": {"draft": 0, "testing": 0, "validated": 0, "deployed": 0},
            },
            "experiment_data": {"total_experiments": 0, "top_experiments": [], "model_performance": {}},
            "signal_data": {"best_signals": []},
            "agent_activity": {},
            "report_type": "weekly",
            "generated_at": datetime(2026, 1, 26, 20, 0),
        }
        insights = [{"priority": "low", "category": "general", "action": "No issues"}]
        result = weekly_generator._render_weekly_report(context, insights)
        assert isinstance(result, str)
        assert "# HRP Research Report" in result


class TestGetReportFilename:
    """Tests for report filename generation."""

    @patch('hrp.agents.report_generator.datetime')
    def test_daily_report_filename(self, mock_datetime, daily_generator):
        """Test daily report filename format."""
        mock_datetime.now.return_value = datetime(2026, 1, 26, 7, 0)
        filename = daily_generator._get_report_filename()
        assert filename == "2026-01-26-07-00-daily.md"

    @patch('hrp.agents.report_generator.datetime')
    def test_weekly_report_filename(self, mock_datetime, weekly_generator):
        """Test weekly report filename format."""
        mock_datetime.now.return_value = datetime(2026, 1, 26, 20, 0)
        filename = weekly_generator._get_report_filename()
        assert filename == "2026-01-26-20-00-weekly.md"


# =============================================================================
# Test Report Writing (stubs for now)
# =============================================================================

class TestWriteReport:
    """Tests for report writing."""

    def test_write_report_returns_path(self, daily_generator):
        """Test that report writing returns a filepath."""
        result = daily_generator._write_report("# Test Report")
        # Stub returns empty string, will be implemented in Task 5
        assert isinstance(result, str)


# =============================================================================
# Test Execute
# =============================================================================

class TestExecute:
    """Tests for execute method."""

    @patch('hrp.agents.report_generator.ReportGenerator._write_report')
    @patch('hrp.agents.report_generator.ReportGenerator._render_daily_report')
    @patch('hrp.agents.report_generator.ReportGenerator._generate_insights')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_agent_activity')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_signal_data')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_experiment_data')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_hypothesis_data')
    def test_execute_daily_report(
        self,
        mock_hypothesis,
        mock_experiment,
        mock_signal,
        mock_activity,
        mock_insights,
        mock_render,
        mock_write,
        daily_generator,
    ):
        """Test daily report execution."""
        # Setup mocks
        mock_hypothesis.return_value = {"total_counts": {"draft": 2, "testing": 1}}
        mock_experiment.return_value = {"total_experiments": 5}
        mock_signal.return_value = {"recent_discoveries": []}
        mock_activity.return_value = {"signal_scientist": {"status": "success"}}
        mock_insights.return_value = []
        mock_render.return_value = "# Daily Report"
        mock_write.return_value = "/tmp/test/2026-01-26-07-00-daily.md"

        result = daily_generator.execute()

        assert result["report_type"] == "daily"
        assert result["report_path"] == "/tmp/test/2026-01-26-07-00-daily.md"
        assert "token_usage" in result

    @patch('hrp.agents.report_generator.ReportGenerator._write_report')
    @patch('hrp.agents.report_generator.ReportGenerator._render_weekly_report')
    @patch('hrp.agents.report_generator.ReportGenerator._generate_insights')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_agent_activity')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_signal_data')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_experiment_data')
    @patch('hrp.agents.report_generator.ReportGenerator._gather_hypothesis_data')
    def test_execute_weekly_report(
        self,
        mock_hypothesis,
        mock_experiment,
        mock_signal,
        mock_activity,
        mock_insights,
        mock_render,
        mock_write,
        weekly_generator,
    ):
        """Test weekly report execution."""
        # Setup mocks
        mock_hypothesis.return_value = {"total_counts": {}}
        mock_experiment.return_value = {}
        mock_signal.return_value = {}
        mock_activity.return_value = {}
        mock_insights.return_value = []
        mock_render.return_value = "# Weekly Report"
        mock_write.return_value = "/tmp/test/2026-01-26-20-00-weekly.md"

        result = weekly_generator.execute()

        assert result["report_type"] == "weekly"
        assert result["report_path"] == "/tmp/test/2026-01-26-20-00-weekly.md"


# =============================================================================
# Test System Prompt
# =============================================================================

class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_not_empty(self, daily_generator):
        """Test that system prompt is not empty."""
        prompt = daily_generator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Report Generator" in prompt


# =============================================================================
# Test Available Tools
# =============================================================================

class TestAvailableTools:
    """Tests for available tools."""

    def test_available_tools_returns_list(self, daily_generator):
        """Test that available tools returns a list."""
        tools = daily_generator._get_available_tools()
        assert isinstance(tools, list)
        # Report Generator doesn't need tools, returns empty list
