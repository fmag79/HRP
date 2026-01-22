"""
Email notification service for HRP.

Sends email notifications for job failures and summaries using Resend.
Gracefully handles missing configuration or unavailable service.
"""

import os
from typing import Any

from loguru import logger

# Import resend conditionally to handle missing package gracefully
try:
    import resend

    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False
    logger.warning("resend package not installed - email notifications will be disabled")


class EmailNotificationError(Exception):
    """Base exception for email notification errors."""

    pass


class EmailConfigurationError(EmailNotificationError):
    """Raised when email configuration is missing or invalid."""

    pass


class EmailNotifier:
    """
    Email notification service using Resend.

    Sends email notifications for job failures and status updates.
    Requires RESEND_API_KEY and NOTIFICATION_EMAIL environment variables.

    Usage:
        notifier = EmailNotifier()
        notifier.send_failure_notification(
            job_name="price_ingestion",
            error_message="Connection timeout",
            retry_count=3
        )
        notifier.send_summary_email(
            subject="Daily Ingestion Summary",
            body="All jobs completed successfully"
        )

    Configuration:
        Set these environment variables:
        - RESEND_API_KEY: Your Resend API key
        - NOTIFICATION_EMAIL: Email address to send notifications to
    """

    def __init__(self):
        """
        Initialize the EmailNotifier.

        Reads configuration from environment variables.
        Logs warnings if configuration is missing but does not raise errors.
        """
        self.api_key = os.getenv("RESEND_API_KEY")
        self.notification_email = os.getenv("NOTIFICATION_EMAIL")
        self.from_email = os.getenv("NOTIFICATION_FROM_EMAIL", "noreply@hrp.local")

        # Track configuration status
        self._configured = self._check_configuration()

        if self._configured:
            logger.debug("EmailNotifier initialized and configured")
        else:
            logger.warning(
                "EmailNotifier initialized but not configured - "
                "set RESEND_API_KEY and NOTIFICATION_EMAIL to enable notifications"
            )

    def _check_configuration(self) -> bool:
        """
        Check if the notifier is properly configured.

        Returns:
            True if all required configuration is present, False otherwise
        """
        if not RESEND_AVAILABLE:
            logger.debug("Email notifications disabled: resend package not installed")
            return False

        if not self.api_key:
            logger.debug("Email notifications disabled: RESEND_API_KEY not set")
            return False

        if not self.notification_email:
            logger.debug("Email notifications disabled: NOTIFICATION_EMAIL not set")
            return False

        return True

    def send_email(
        self,
        subject: str,
        body: str,
        html_body: str | None = None,
    ) -> bool:
        """
        Send an email using Resend.

        Args:
            subject: Email subject line
            body: Plain text email body
            html_body: Optional HTML email body

        Returns:
            True if email was sent successfully, False otherwise

        Raises:
            EmailNotificationError: If sending fails and notifications are configured
        """
        if not self._configured:
            logger.debug(f"Email not sent (not configured): {subject}")
            return False

        try:
            # Set API key for resend
            resend.api_key = self.api_key

            # Prepare email payload
            params = {
                "from": self.from_email,
                "to": [self.notification_email],
                "subject": subject,
                "text": body,
            }

            if html_body:
                params["html"] = html_body

            # Send email
            response = resend.Emails.send(params)

            logger.info(f"Email sent successfully: {subject} (id: {response.get('id', 'unknown')})")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise EmailNotificationError(f"Failed to send email: {e}") from e

    def send_failure_notification(
        self,
        job_name: str,
        error_message: str,
        retry_count: int = 0,
        max_retries: int = 0,
        timestamp: str | None = None,
    ) -> bool:
        """
        Send a job failure notification email.

        Args:
            job_name: Name of the failed job
            error_message: Error message describing the failure
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries configured
            timestamp: Optional timestamp of the failure

        Returns:
            True if email was sent successfully, False otherwise
        """
        subject = f"❌ HRP Job Failed: {job_name}"

        # Build plain text body
        body_parts = [
            f"Job '{job_name}' has failed.",
            "",
            "Error Details:",
            f"  {error_message}",
            "",
        ]

        if retry_count > 0:
            body_parts.append(f"Retry Attempt: {retry_count}/{max_retries}")
            body_parts.append("")

        if timestamp:
            body_parts.append(f"Timestamp: {timestamp}")
            body_parts.append("")

        body_parts.extend(
            [
                "This is an automated notification from HRP.",
                "Check the dashboard for more details: http://localhost:8501",
            ]
        )

        body = "\n".join(body_parts)

        # Build HTML body
        html_body = f"""
        <html>
        <body>
            <h2 style="color: #d32f2f;">Job Failed: {job_name}</h2>
            <p>Job <strong>{job_name}</strong> has failed.</p>

            <h3>Error Details</h3>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px;">{error_message}</pre>

            {f'<p><strong>Retry Attempt:</strong> {retry_count}/{max_retries}</p>' if retry_count > 0 else ''}
            {f'<p><strong>Timestamp:</strong> {timestamp}</p>' if timestamp else ''}

            <hr>
            <p style="color: #666;">
                This is an automated notification from HRP.<br>
                <a href="http://localhost:8501">Check the dashboard</a> for more details.
            </p>
        </body>
        </html>
        """

        try:
            return self.send_email(subject, body, html_body)
        except EmailNotificationError:
            # Already logged in send_email
            return False

    def send_summary_email(
        self,
        subject: str,
        summary_data: dict[str, Any],
    ) -> bool:
        """
        Send a summary email with job statistics.

        Args:
            subject: Email subject line
            summary_data: Dictionary containing summary information

        Returns:
            True if email was sent successfully, False otherwise
        """
        # Build plain text body
        body_parts = [subject, "", "Summary:"]

        for key, value in summary_data.items():
            body_parts.append(f"  {key}: {value}")

        body_parts.extend(
            [
                "",
                "This is an automated notification from HRP.",
                "Check the dashboard for more details: http://localhost:8501",
            ]
        )

        body = "\n".join(body_parts)

        # Build HTML body
        summary_html = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in summary_data.items()])

        html_body = f"""
        <html>
        <body>
            <h2>{subject}</h2>
            <h3>Summary</h3>
            <ul>
                {summary_html}
            </ul>
            <hr>
            <p style="color: #666;">
                This is an automated notification from HRP.<br>
                <a href="http://localhost:8501">Check the dashboard</a> for more details.
            </p>
        </body>
        </html>
        """

        try:
            return self.send_email(subject, body, html_body)
        except EmailNotificationError:
            # Already logged in send_email
            return False

    def is_configured(self) -> bool:
        """
        Check if email notifications are configured and ready to use.

        Returns:
            True if configured, False otherwise
        """
        return self._configured
