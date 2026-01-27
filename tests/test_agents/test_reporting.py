"""Tests for agent reporting utilities."""

from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from hrp.agents.reporting import AgentReport


class TestAgentReport:
    """Tests for AgentReport dataclass."""

    def test_create_basic_report(self):
        """Test creating a basic agent report."""
        report = AgentReport(
            agent_name="test_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="success",
            results={"records_processed": 100},
        )

        assert report.agent_name == "test_agent"
        assert report.status == "success"
        assert report.results == {"records_processed": 100}
        assert report.errors == []

    def test_create_report_with_errors(self):
        """Test creating a report with errors."""
        report = AgentReport(
            agent_name="failing_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="failed",
            results={},
            errors=["Error 1", "Error 2"],
        )

        assert report.status == "failed"
        assert len(report.errors) == 2
        assert "Error 1" in report.errors

    def test_to_markdown_generates_valid_markdown(self):
        """Test that to_markdown generates valid markdown."""
        report = AgentReport(
            agent_name="test_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="success",
            results={"records_processed": 100, "symbols": 10},
        )

        markdown = report.to_markdown()

        # Should contain key sections
        assert "# Agent Execution Report" in markdown
        assert "test_agent" in markdown
        assert "success" in markdown
        assert "records_processed" in markdown
        assert "100" in markdown

    def test_to_markdown_includes_errors_when_present(self):
        """Test that to_markdown includes error section."""
        report = AgentReport(
            agent_name="failing_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="failed",
            results={},
            errors=["Connection failed", "Timeout"],
        )

        markdown = report.to_markdown()

        assert "## Errors" in markdown
        assert "Connection failed" in markdown
        assert "Timeout" in markdown

    def test_save_to_file_writes_content(self):
        """Test that save_to_file writes markdown to file."""
        report = AgentReport(
            agent_name="test_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="success",
            results={"records_processed": 100},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            report.save_to_file(report_path)

            assert report_path.exists()
            content = report_path.read_text()
            assert "# Agent Execution Report" in content
            assert "test_agent" in content

    def test_to_markdown_includes_duration(self):
        """Test that to_markdown includes execution duration."""
        report = AgentReport(
            agent_name="test_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 30, 0),  # 1.5 hours
            status="success",
            results={},
        )

        markdown = report.to_markdown()

        assert "Duration" in markdown
        # Should show approximately 1.5h (5400 seconds)
        assert "1.5h" in markdown

    def test_report_with_nested_results(self):
        """Test report with nested dictionary results."""
        report = AgentReport(
            agent_name="complex_agent",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
            status="success",
            results={
                "stage1": {"processed": 100, "failed": 0},
                "stage2": {"processed": 95, "failed": 5},
            },
        )

        markdown = report.to_markdown()

        assert "stage1" in markdown
        assert "stage2" in markdown
