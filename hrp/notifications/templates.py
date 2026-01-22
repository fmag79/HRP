"""
Email templates for HRP notifications.

Provides formatting functions for failure notifications and summary emails.
Separates presentation logic from email sending logic.
"""

from typing import Any, Optional


def format_failure_email(
    job_name: str,
    error_message: str,
    retry_count: int = 0,
    max_retries: int = 0,
    timestamp: Optional[str] = None,
) -> tuple[str, str]:
    """
    Format a job failure notification email.

    Args:
        job_name: Name of the failed job
        error_message: Error message describing the failure
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries configured
        timestamp: Optional timestamp of the failure

    Returns:
        Tuple of (plain_text_body, html_body)
    """
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

    text_body = "\n".join(body_parts)

    # Build HTML body
    retry_section = (
        f'<p><strong>Retry Attempt:</strong> {retry_count}/{max_retries}</p>'
        if retry_count > 0
        else ""
    )

    timestamp_section = (
        f'<p><strong>Timestamp:</strong> {timestamp}</p>' if timestamp else ""
    )

    html_body = f"""
    <html>
    <body>
        <h2 style="color: #d32f2f;">Job Failed: {job_name}</h2>
        <p>Job <strong>{job_name}</strong> has failed.</p>

        <h3>Error Details</h3>
        <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px;">{error_message}</pre>

        {retry_section}
        {timestamp_section}

        <hr>
        <p style="color: #666;">
            This is an automated notification from HRP.<br>
            <a href="http://localhost:8501">Check the dashboard</a> for more details.
        </p>
    </body>
    </html>
    """

    return text_body, html_body


def format_summary_email(
    subject: str,
    summary_data: dict[str, Any],
) -> tuple[str, str]:
    """
    Format a summary email with job statistics.

    Args:
        subject: Email subject line (also used as heading)
        summary_data: Dictionary containing summary information

    Returns:
        Tuple of (plain_text_body, html_body)
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

    text_body = "\n".join(body_parts)

    # Build HTML body
    summary_html = "".join(
        [f"<li><strong>{k}:</strong> {v}</li>" for k, v in summary_data.items()]
    )

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

    return text_body, html_body
