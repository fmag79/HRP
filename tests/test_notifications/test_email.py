"""
Comprehensive tests for email notification service.

Tests cover:
- Quality alert sending
- Email formatting (subject, HTML body, text body)
- Configuration validation
- Error handling (missing config, API failures)
- Various severity levels and report structures
"""

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.notifications.email import (
    _format_html_body,
    _format_subject,
    _format_text_body,
    send_quality_alert,
)
from hrp.utils.config import reset_config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_report():
    """Create a sample quality check report."""
    return {
        "check_date": "2023-01-15",
        "overall_status": "warning",
        "checks": {
            "completeness": {
                "status": "pass",
                "issues": [],
            },
            "data_quality": {
                "status": "warning",
                "issues": [
                    "AAPL: Missing price data on 2023-01-10",
                    "MSFT: Volume anomaly detected (3x average)",
                ],
            },
            "timeliness": {
                "status": "fail",
                "issues": [
                    "Data last updated 5 hours ago (SLA: 1 hour)",
                ],
            },
        },
    }


@pytest.fixture
def critical_report():
    """Create a critical quality report with many issues."""
    issues = [f"Symbol_{i}: Critical data issue" for i in range(15)]
    return {
        "check_date": "2023-01-15",
        "overall_status": "fail",
        "checks": {
            "completeness": {
                "status": "fail",
                "issues": issues,
            },
        },
    }


@pytest.fixture
def clean_report():
    """Create a report with no issues."""
    return {
        "check_date": "2023-01-15",
        "overall_status": "pass",
        "checks": {
            "completeness": {"status": "pass", "issues": []},
            "data_quality": {"status": "pass", "issues": []},
            "timeliness": {"status": "pass", "issues": []},
        },
    }


@pytest.fixture
def mock_config():
    """Mock configuration with email settings."""

    class MockAPIConfig:
        resend_api_key = "re_test_key_123"

    class MockConfig:
        api = MockAPIConfig()
        notification_email = "test@example.com"

    return MockConfig()


@pytest.fixture(autouse=True)
def cleanup_config():
    """Reset config after each test."""
    yield
    reset_config()


# =============================================================================
# Test Email Sending
# =============================================================================


class TestSendQualityAlert:
    """Tests for send_quality_alert function."""

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_alert_success(self, mock_get_config, mock_resend, sample_report, mock_config):
        """Test successful email sending."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.return_value = {"id": "msg_123"}

        result = send_quality_alert(sample_report, recipient="user@example.com", severity="warning")

        assert result["status"] == "sent"
        assert result["message_id"] == "msg_123"
        assert result["recipient"] == "user@example.com"

        # Verify Resend was called correctly
        mock_resend.Emails.send.assert_called_once()
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["user@example.com"]
        assert "⚠️ Warning" in call_args["subject"]
        assert "Data Quality Report" in call_args["subject"]
        assert call_args["html"]
        assert call_args["text"]

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_alert_uses_config_recipient(
        self, mock_get_config, mock_resend, sample_report, mock_config
    ):
        """Test that config recipient is used when none provided."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.return_value = {"id": "msg_456"}

        result = send_quality_alert(sample_report)

        assert result["recipient"] == "test@example.com"
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert call_args["to"] == ["test@example.com"]

    @patch("hrp.notifications.email.get_config")
    def test_send_alert_missing_recipient(self, mock_get_config, sample_report):
        """Test error when no recipient configured."""

        class MockConfigNoEmail:
            api = MagicMock()
            notification_email = None

        mock_get_config.return_value = MockConfigNoEmail()

        with pytest.raises(ValueError, match="No recipient specified"):
            send_quality_alert(sample_report)

    @patch("hrp.notifications.email.get_config")
    def test_send_alert_missing_api_key(self, mock_get_config, sample_report):
        """Test error when Resend API key not configured."""

        class MockConfigNoKey:
            api = MagicMock()
            api.resend_api_key = None
            notification_email = "test@example.com"

        mock_get_config.return_value = MockConfigNoKey()

        with pytest.raises(RuntimeError, match="RESEND_API_KEY not configured"):
            send_quality_alert(sample_report, recipient="user@example.com")

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_alert_api_failure(self, mock_get_config, mock_resend, sample_report, mock_config):
        """Test handling of Resend API failures."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.side_effect = Exception("API rate limit exceeded")

        with pytest.raises(RuntimeError, match="Failed to send email via Resend"):
            send_quality_alert(sample_report, recipient="user@example.com")

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_alert_different_severities(
        self, mock_get_config, mock_resend, sample_report, mock_config
    ):
        """Test sending alerts with different severity levels."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.return_value = {"id": "msg_123"}

        for severity in ["info", "warning", "critical"]:
            result = send_quality_alert(sample_report, recipient="user@example.com", severity=severity)
            assert result["status"] == "sent"

            call_args = mock_resend.Emails.send.call_args[0][0]
            assert severity.upper() in call_args["text"]

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_alert_with_clean_report(
        self, mock_get_config, mock_resend, clean_report, mock_config
    ):
        """Test sending alert for report with no issues."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.return_value = {"id": "msg_789"}

        result = send_quality_alert(clean_report, recipient="user@example.com", severity="info")

        assert result["status"] == "sent"
        call_args = mock_resend.Emails.send.call_args[0][0]
        assert "No critical issues detected" in call_args["text"]


# =============================================================================
# Test Email Formatting
# =============================================================================


class TestFormatSubject:
    """Tests for _format_subject function."""

    def test_format_subject_info(self):
        """Test subject formatting for info severity."""
        report = {"check_date": "2023-01-15"}
        subject = _format_subject(report, "info")

        assert "ℹ️ Info" in subject
        assert "Data Quality Report" in subject
        assert "2023-01-15" in subject

    def test_format_subject_warning(self):
        """Test subject formatting for warning severity."""
        report = {"check_date": "2023-01-15"}
        subject = _format_subject(report, "warning")

        assert "⚠️ Warning" in subject
        assert "Data Quality Report" in subject

    def test_format_subject_critical(self):
        """Test subject formatting for critical severity."""
        report = {"check_date": "2023-01-15"}
        subject = _format_subject(report, "critical")

        assert "🚨 Critical" in subject
        assert "Data Quality Report" in subject

    def test_format_subject_unknown_severity(self):
        """Test subject formatting for unknown severity."""
        report = {"check_date": "2023-01-15"}
        subject = _format_subject(report, "unknown")

        assert "📊 Alert" in subject

    def test_format_subject_missing_date(self):
        """Test subject formatting when date is missing."""
        report = {}
        subject = _format_subject(report, "info")

        assert "Unknown" in subject


class TestFormatHTMLBody:
    """Tests for _format_html_body function."""

    def test_format_html_basic_structure(self, sample_report):
        """Test HTML body contains all required sections."""
        html = _format_html_body(sample_report, "warning")

        assert "<!DOCTYPE html>" in html
        assert "Data Quality Report" in html
        assert "2023-01-15" in html
        assert sample_report["overall_status"].upper() in html
        assert "Summary" in html
        assert "Issues Detected" in html

    def test_format_html_status_colors(self):
        """Test that different statuses use appropriate colors."""
        for status, expected_color in [
            ("pass", "#10b981"),
            ("warning", "#f59e0b"),
            ("fail", "#ef4444"),
            ("error", "#dc2626"),
        ]:
            report = {"check_date": "2023-01-15", "overall_status": status, "checks": {}}
            html = _format_html_body(report, "info")
            assert expected_color in html

    def test_format_html_includes_issues(self, sample_report):
        """Test that issues are included in HTML."""
        html = _format_html_body(sample_report, "warning")

        assert "Missing price data" in html
        assert "Volume anomaly" in html
        assert "Data last updated 5 hours ago" in html

    def test_format_html_limits_issues(self, critical_report):
        """Test that HTML limits issues to first 10."""
        html = _format_html_body(critical_report, "critical")

        # Should show "and X more issues" message
        assert "and 5 more issues" in html

    def test_format_html_no_issues(self, clean_report):
        """Test HTML for report with no issues."""
        html = _format_html_body(clean_report, "info")

        assert "No critical issues detected" in html

    def test_format_html_check_summary(self, sample_report):
        """Test that check summary includes all checks."""
        html = _format_html_body(sample_report, "warning")

        assert "Completeness" in html
        assert "Data Quality" in html
        assert "Timeliness" in html
        assert "(2 issues)" in html  # data_quality has 2 issues
        assert "(1 issues)" in html  # timeliness has 1 issue


class TestFormatTextBody:
    """Tests for _format_text_body function."""

    def test_format_text_basic_structure(self, sample_report):
        """Test text body contains all required sections."""
        text = _format_text_body(sample_report, "warning")

        assert "DATA QUALITY REPORT" in text
        assert "Check Date: 2023-01-15" in text
        assert "Overall Status: WARNING" in text
        assert "Severity: WARNING" in text
        assert "SUMMARY" in text
        assert "ISSUES DETECTED" in text

    def test_format_text_includes_issues(self, sample_report):
        """Test that issues are included in text body."""
        text = _format_text_body(sample_report, "warning")

        assert "Missing price data" in text
        assert "Volume anomaly" in text
        assert "Data last updated 5 hours ago" in text

    def test_format_text_limits_issues(self, critical_report):
        """Test that text body limits issues to first 10."""
        text = _format_text_body(critical_report, "critical")

        # Should show "and X more issues" message
        assert "and 5 more issues" in text

    def test_format_text_no_issues(self, clean_report):
        """Test text body for report with no issues."""
        text = _format_text_body(clean_report, "info")

        assert "No critical issues detected" in text

    def test_format_text_check_summary(self, sample_report):
        """Test that check summary includes all checks with counts."""
        text = _format_text_body(sample_report, "warning")

        assert "Completeness: PASS" in text
        assert "Data Quality: WARNING (2 issues)" in text
        assert "Timeliness: FAIL (1 issues)" in text

    def test_format_text_footer(self, sample_report):
        """Test that footer is included."""
        text = _format_text_body(sample_report, "warning")

        assert "automated alert from the HRP Data Quality Framework" in text
        assert "HRP dashboard" in text


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_report(self):
        """Test formatting with minimal/empty report."""
        report = {}

        subject = _format_subject(report, "info")
        assert subject

        html = _format_html_body(report, "info")
        assert html
        assert "Unknown" in html

        text = _format_text_body(report, "info")
        assert text
        assert "Unknown" in text

    def test_report_missing_checks(self):
        """Test report without checks section."""
        report = {"check_date": "2023-01-15", "overall_status": "pass"}

        html = _format_html_body(report, "info")
        assert "No critical issues detected" in html

        text = _format_text_body(report, "info")
        assert text

    def test_check_missing_issues(self):
        """Test check result without issues list."""
        report = {
            "check_date": "2023-01-15",
            "overall_status": "pass",
            "checks": {
                "completeness": {
                    "status": "pass",
                    # No 'issues' key
                },
            },
        }

        html = _format_html_body(report, "info")
        text = _format_text_body(report, "info")

        assert html
        assert text

    @patch("hrp.notifications.email.resend")
    @patch("hrp.notifications.email.get_config")
    def test_send_with_minimal_report(self, mock_get_config, mock_resend, mock_config):
        """Test sending alert with minimal report data."""
        mock_get_config.return_value = mock_config
        mock_resend.Emails.send.return_value = {"id": "msg_minimal"}

        minimal_report = {"check_date": "2023-01-15"}

        result = send_quality_alert(minimal_report, recipient="user@example.com")

        assert result["status"] == "sent"

    def test_special_characters_in_issues(self):
        """Test handling of special characters in issue messages."""
        report = {
            "check_date": "2023-01-15",
            "overall_status": "fail",
            "checks": {
                "data_quality": {
                    "status": "fail",
                    "issues": [
                        "Issue with <special> & 'characters'",
                        'Issue with "quotes" and symbols: $%^',
                    ],
                }
            },
        }

        html = _format_html_body(report, "critical")
        text = _format_text_body(report, "critical")

        # Both should handle special characters without crashing
        assert html
        assert text
        assert "special" in html.lower()
        assert "special" in text.lower()
