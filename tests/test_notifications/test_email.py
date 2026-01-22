"""
Tests for EmailNotifier.

Tests cover:
- Configuration checking
- Email sending with various parameters
- Graceful handling of missing configuration
- Failure notification formatting
- Summary email formatting
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestEmailNotifierConfiguration:
    """Tests for EmailNotifier configuration handling."""

    def test_configured_when_all_env_vars_set(self):
        """EmailNotifier should be configured when all env vars are set."""
        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "test@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                assert notifier.is_configured() is True

    def test_not_configured_without_api_key(self):
        """EmailNotifier should not be configured without API key."""
        env_without_key = {
            "NOTIFICATION_EMAIL": "test@example.com",
        }
        # Remove RESEND_API_KEY if it exists
        with patch.dict(os.environ, env_without_key, clear=False):
            if "RESEND_API_KEY" in os.environ:
                del os.environ["RESEND_API_KEY"]

            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                assert notifier.is_configured() is False

    def test_not_configured_without_notification_email(self):
        """EmailNotifier should not be configured without notification email."""
        env_without_email = {
            "RESEND_API_KEY": "test_api_key",
        }
        with patch.dict(os.environ, env_without_email, clear=False):
            if "NOTIFICATION_EMAIL" in os.environ:
                del os.environ["NOTIFICATION_EMAIL"]

            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                assert notifier.is_configured() is False

    def test_not_configured_when_resend_unavailable(self):
        """EmailNotifier should not be configured when resend package missing."""
        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "test@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", False):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                assert notifier.is_configured() is False

    def test_default_from_email(self):
        """EmailNotifier should use default from email when not specified."""
        with patch.dict(os.environ, {}, clear=False):
            if "NOTIFICATION_FROM_EMAIL" in os.environ:
                del os.environ["NOTIFICATION_FROM_EMAIL"]

            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            assert notifier.from_email == "noreply@hrp.local"

    def test_custom_from_email(self):
        """EmailNotifier should use custom from email when specified."""
        with patch.dict(
            os.environ,
            {"NOTIFICATION_FROM_EMAIL": "custom@example.com"},
        ):
            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            assert notifier.from_email == "custom@example.com"


class TestEmailNotifierSendEmail:
    """Tests for send_email method."""

    def test_send_email_when_not_configured_returns_false(self):
        """send_email should return False when not configured."""
        with patch("hrp.notifications.email.RESEND_AVAILABLE", False):
            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            result = notifier.send_email(
                subject="Test Subject",
                body="Test body",
            )
            assert result is False

    @patch("hrp.notifications.email.resend")
    def test_send_email_calls_resend_api(self, mock_resend):
        """send_email should call Resend API with correct parameters."""
        mock_resend.Emails.send.return_value = {"id": "email_123"}

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
                "NOTIFICATION_FROM_EMAIL": "sender@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_email(
                    subject="Test Subject",
                    body="Test body text",
                )

                assert result is True
                mock_resend.Emails.send.assert_called_once()
                call_args = mock_resend.Emails.send.call_args[0][0]
                assert call_args["subject"] == "Test Subject"
                assert call_args["text"] == "Test body text"
                assert call_args["to"] == ["recipient@example.com"]
                assert call_args["from"] == "sender@example.com"

    @patch("hrp.notifications.email.resend")
    def test_send_email_with_html_body(self, mock_resend):
        """send_email should include HTML body when provided."""
        mock_resend.Emails.send.return_value = {"id": "email_123"}

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_email(
                    subject="Test Subject",
                    body="Plain text",
                    html_body="<html><body>HTML content</body></html>",
                )

                assert result is True
                call_args = mock_resend.Emails.send.call_args[0][0]
                assert call_args["html"] == "<html><body>HTML content</body></html>"

    @patch("hrp.notifications.email.resend")
    def test_send_email_raises_on_api_error(self, mock_resend):
        """send_email should raise EmailNotificationError on API error."""
        mock_resend.Emails.send.side_effect = Exception("API Error")

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import (
                    EmailNotificationError,
                    EmailNotifier,
                )

                notifier = EmailNotifier()
                with pytest.raises(EmailNotificationError):
                    notifier.send_email(
                        subject="Test Subject",
                        body="Test body",
                    )


class TestEmailNotifierFailureNotification:
    """Tests for send_failure_notification method."""

    @patch("hrp.notifications.email.resend")
    def test_failure_notification_format(self, mock_resend):
        """send_failure_notification should format email correctly."""
        mock_resend.Emails.send.return_value = {"id": "email_123"}

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_failure_notification(
                    job_name="price_ingestion",
                    error_message="Connection timeout",
                    retry_count=2,
                    max_retries=3,
                    timestamp="2024-01-15T10:30:00",
                )

                assert result is True
                call_args = mock_resend.Emails.send.call_args[0][0]

                # Check subject contains job name
                assert "price_ingestion" in call_args["subject"]
                assert "Failed" in call_args["subject"]

                # Check body contains error details
                assert "Connection timeout" in call_args["text"]
                assert "2/3" in call_args["text"]  # Retry count

                # Check HTML body
                assert "price_ingestion" in call_args["html"]
                assert "Connection timeout" in call_args["html"]

    def test_failure_notification_returns_false_when_not_configured(self):
        """send_failure_notification should return False when not configured."""
        with patch("hrp.notifications.email.RESEND_AVAILABLE", False):
            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            result = notifier.send_failure_notification(
                job_name="test_job",
                error_message="Error",
            )
            assert result is False

    @patch("hrp.notifications.email.resend")
    def test_failure_notification_handles_api_error(self, mock_resend):
        """send_failure_notification should return False on API error."""
        mock_resend.Emails.send.side_effect = Exception("API Error")

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_failure_notification(
                    job_name="test_job",
                    error_message="Error",
                )
                assert result is False


class TestEmailNotifierSummaryEmail:
    """Tests for send_summary_email method."""

    @patch("hrp.notifications.email.resend")
    def test_summary_email_format(self, mock_resend):
        """send_summary_email should format email correctly."""
        mock_resend.Emails.send.return_value = {"id": "email_123"}

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_summary_email(
                    subject="Daily Ingestion Summary",
                    summary_data={
                        "jobs_run": 5,
                        "jobs_succeeded": 4,
                        "jobs_failed": 1,
                    },
                )

                assert result is True
                call_args = mock_resend.Emails.send.call_args[0][0]

                # Check subject
                assert call_args["subject"] == "Daily Ingestion Summary"

                # Check body contains summary data
                assert "jobs_run" in call_args["text"]
                assert "5" in call_args["text"]

                # Check HTML body
                assert "jobs_run" in call_args["html"]

    def test_summary_email_returns_false_when_not_configured(self):
        """send_summary_email should return False when not configured."""
        with patch("hrp.notifications.email.RESEND_AVAILABLE", False):
            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            result = notifier.send_summary_email(
                subject="Summary",
                summary_data={"key": "value"},
            )
            assert result is False

    @patch("hrp.notifications.email.resend")
    def test_summary_email_handles_api_error(self, mock_resend):
        """send_summary_email should return False on API error."""
        mock_resend.Emails.send.side_effect = Exception("API Error")

        with patch.dict(
            os.environ,
            {
                "RESEND_API_KEY": "test_api_key",
                "NOTIFICATION_EMAIL": "recipient@example.com",
            },
        ):
            with patch("hrp.notifications.email.RESEND_AVAILABLE", True):
                from hrp.notifications.email import EmailNotifier

                notifier = EmailNotifier()
                result = notifier.send_summary_email(
                    subject="Summary",
                    summary_data={"key": "value"},
                )
                assert result is False


class TestEmailNotificationExceptions:
    """Tests for email notification exception classes."""

    def test_email_notification_error_inheritance(self):
        """EmailNotificationError should be an Exception."""
        from hrp.notifications.email import EmailNotificationError

        error = EmailNotificationError("Test error")
        assert isinstance(error, Exception)

    def test_email_configuration_error_inheritance(self):
        """EmailConfigurationError should be an EmailNotificationError."""
        from hrp.notifications.email import (
            EmailConfigurationError,
            EmailNotificationError,
        )

        error = EmailConfigurationError("Config error")
        assert isinstance(error, EmailNotificationError)
