"""Tests for hrp/notifications/templates.py."""

import pytest

from hrp.notifications.templates import format_failure_email, format_summary_email


class TestFormatFailureEmail:
    """Tests for format_failure_email function."""

    def test_basic(self):
        """Verify job_name and error in output."""
        text, html = format_failure_email(
            job_name="PriceIngestionJob",
            error_message="Connection timeout",
        )

        assert "PriceIngestionJob" in text
        assert "Connection timeout" in text
        assert "PriceIngestionJob" in html
        assert "Connection timeout" in html

    def test_with_retry_count(self):
        """Verify retry count shown."""
        text, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
            retry_count=2,
            max_retries=3,
        )

        assert "Retry Attempt: 2/3" in text
        assert "2/3" in html

    def test_without_retry_count(self):
        """Verify retry count not shown when 0."""
        text, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
            retry_count=0,
            max_retries=3,
        )

        assert "Retry Attempt" not in text
        # HTML should not have retry section
        assert "Retry Attempt" not in html

    def test_with_timestamp(self):
        """Verify timestamp formatted."""
        text, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
            timestamp="2025-01-15 10:30:00",
        )

        assert "Timestamp: 2025-01-15 10:30:00" in text
        assert "2025-01-15 10:30:00" in html

    def test_without_timestamp(self):
        """Verify timestamp not shown when None."""
        text, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
            timestamp=None,
        )

        assert "Timestamp:" not in text

    def test_returns_tuple(self):
        """Verify returns (text, html) tuple."""
        result = format_failure_email(
            job_name="TestJob",
            error_message="Error",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        text, html = result
        assert isinstance(text, str)
        assert isinstance(html, str)

    def test_html_structure(self):
        """Verify HTML has proper tags."""
        _, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
        )

        assert "<html>" in html
        assert "</html>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "<h2" in html
        assert "<pre" in html

    def test_contains_dashboard_link(self):
        """Verify dashboard link included."""
        text, html = format_failure_email(
            job_name="TestJob",
            error_message="Error",
        )

        assert "http://localhost:8501" in text
        assert "http://localhost:8501" in html

    def test_full_output(self):
        """Verify complete output with all fields."""
        text, html = format_failure_email(
            job_name="FeatureComputationJob",
            error_message="Database connection failed: timeout after 30s",
            retry_count=1,
            max_retries=3,
            timestamp="2025-01-20 14:30:00",
        )

        # Text body checks
        assert "FeatureComputationJob" in text
        assert "Database connection failed: timeout after 30s" in text
        assert "Retry Attempt: 1/3" in text
        assert "Timestamp: 2025-01-20 14:30:00" in text
        assert "automated notification from HRP" in text

        # HTML body checks
        assert "Job Failed: FeatureComputationJob" in html
        assert "Database connection failed: timeout after 30s" in html


class TestFormatSummaryEmail:
    """Tests for format_summary_email function."""

    def test_basic(self):
        """Verify summary content rendered."""
        text, html = format_summary_email(
            subject="Daily Ingestion Complete",
            summary_data={"records": 1000, "errors": 0},
        )

        assert "Daily Ingestion Complete" in text
        assert "records: 1000" in text
        assert "errors: 0" in text

        assert "Daily Ingestion Complete" in html
        assert "1000" in html
        assert "0" in html

    def test_multiple_items(self):
        """Verify multiple items in list."""
        text, html = format_summary_email(
            subject="Report",
            summary_data={
                "symbols_processed": 500,
                "features_computed": 15000,
                "duration_seconds": 120,
                "status": "success",
            },
        )

        assert "symbols_processed: 500" in text
        assert "features_computed: 15000" in text
        assert "duration_seconds: 120" in text
        assert "status: success" in text

        # HTML should have list items
        assert "<li>" in html
        assert "symbols_processed" in html

    def test_empty_data(self):
        """Verify handles empty summary_data."""
        text, html = format_summary_email(
            subject="Empty Report",
            summary_data={},
        )

        assert "Empty Report" in text
        assert "Summary:" in text
        assert "Empty Report" in html

    def test_returns_tuple(self):
        """Verify returns (text, html) tuple."""
        result = format_summary_email(
            subject="Test",
            summary_data={"key": "value"},
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        text, html = result
        assert isinstance(text, str)
        assert isinstance(html, str)

    def test_html_structure(self):
        """Verify HTML has proper tags."""
        _, html = format_summary_email(
            subject="Test",
            summary_data={"key": "value"},
        )

        assert "<html>" in html
        assert "</html>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "<h2>" in html
        assert "<ul>" in html
        assert "</ul>" in html

    def test_contains_dashboard_link(self):
        """Verify dashboard link included."""
        text, html = format_summary_email(
            subject="Test",
            summary_data={},
        )

        assert "http://localhost:8501" in text
        assert "http://localhost:8501" in html

    def test_subject_used_as_heading(self):
        """Verify subject is used as HTML heading."""
        _, html = format_summary_email(
            subject="Weekly Performance Summary",
            summary_data={"sharpe": 1.5},
        )

        assert "<h2>Weekly Performance Summary</h2>" in html

    def test_numeric_values(self):
        """Verify numeric values rendered correctly."""
        text, html = format_summary_email(
            subject="Metrics",
            summary_data={
                "sharpe_ratio": 1.5,
                "total_return": 0.15,
                "max_drawdown": -0.08,
            },
        )

        assert "sharpe_ratio: 1.5" in text
        assert "total_return: 0.15" in text
        assert "max_drawdown: -0.08" in text
