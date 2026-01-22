"""
Email notification service using Resend API.

Sends quality alerts and other system notifications.
"""

from typing import Any

from loguru import logger

from hrp.utils.config import get_config


def send_quality_alert(
    report: dict[str, Any],
    recipient: str | None = None,
    severity: str = "warning",
) -> dict[str, Any]:
    """
    Send email alert for data quality issues.

    Args:
        report: Quality check report containing status and issues
        recipient: Email recipient (uses config if not provided)
        severity: Alert severity level (info, warning, critical)

    Returns:
        dict: Result with status and message_id if successful

    Raises:
        ValueError: If recipient not provided and not configured
        RuntimeError: If Resend API key not configured or send fails
    """
    config = get_config()

    # Validate recipient
    if recipient is None:
        recipient = config.notification_email
    if not recipient:
        raise ValueError("No recipient specified and NOTIFICATION_EMAIL not configured")

    # Validate API key
    if not config.api.resend_api_key:
        raise RuntimeError("RESEND_API_KEY not configured")

    # Format email content
    subject = _format_subject(report, severity)
    html_body = _format_html_body(report, severity)
    text_body = _format_text_body(report, severity)

    # Send via Resend
    try:
        import resend

        resend.api_key = config.api.resend_api_key

        params = {
            "from": "HRP Data Quality <alerts@resend.dev>",
            "to": [recipient],
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }

        response = resend.Emails.send(params)

        logger.info(f"Quality alert sent to {recipient} (severity={severity})")
        return {"status": "sent", "message_id": response.get("id"), "recipient": recipient}

    except Exception as e:
        logger.error(f"Failed to send quality alert: {e}")
        raise RuntimeError(f"Failed to send email via Resend: {e}") from e


def _format_subject(report: dict[str, Any], severity: str) -> str:
    """Format email subject line based on severity and issues."""
    severity_prefix = {
        "info": "ℹ️ Info",
        "warning": "⚠️ Warning",
        "critical": "🚨 Critical",
    }.get(severity, "📊 Alert")

    check_date = report.get("check_date", "Unknown")
    return f"{severity_prefix}: Data Quality Report - {check_date}"


def _format_html_body(report: dict[str, Any], severity: str) -> str:
    """Format HTML email body with quality report details."""
    check_date = report.get("check_date", "Unknown")
    overall_status = report.get("overall_status", "unknown")

    # Status color coding
    status_color = {
        "pass": "#10b981",
        "warning": "#f59e0b",
        "fail": "#ef4444",
        "error": "#dc2626",
    }.get(overall_status, "#6b7280")

    # Build issues section
    issues_html = ""
    checks = report.get("checks", {})

    for check_name, check_result in checks.items():
        if check_result.get("status") in ["warning", "fail", "error"]:
            issues = check_result.get("issues", [])
            if issues:
                issues_html += f"<h3>{check_name.replace('_', ' ').title()}</h3><ul>"
                for issue in issues[:10]:  # Limit to first 10 issues
                    issues_html += f"<li>{issue}</li>"
                if len(issues) > 10:
                    issues_html += f"<li><em>...and {len(issues) - 10} more issues</em></li>"
                issues_html += "</ul>"

    if not issues_html:
        issues_html = "<p>No critical issues detected.</p>"

    # Build summary section
    summary_html = "<ul>"
    for check_name, check_result in checks.items():
        status = check_result.get("status", "unknown")
        count = len(check_result.get("issues", []))
        summary_html += f"<li><strong>{check_name.replace('_', ' ').title()}</strong>: {status}"
        if count > 0:
            summary_html += f" ({count} issues)"
        summary_html += "</li>"
    summary_html += "</ul>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: {status_color}; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .summary {{ background-color: #f3f4f6; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            ul {{ padding-left: 20px; }}
            li {{ margin: 5px 0; }}
            .footer {{ color: #6b7280; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Quality Report</h1>
            <p>Check Date: {check_date}</p>
            <p>Overall Status: {overall_status.upper()}</p>
        </div>
        <div class="content">
            <div class="summary">
                <h2>Summary</h2>
                {summary_html}
            </div>
            <h2>Issues Detected</h2>
            {issues_html}
        </div>
        <div class="footer">
            <p>This is an automated alert from the HRP Data Quality Framework.</p>
            <p>For more details, check the quality metrics in your HRP dashboard.</p>
        </div>
    </body>
    </html>
    """
    return html


def _format_text_body(report: dict[str, Any], severity: str) -> str:
    """Format plain text email body with quality report details."""
    check_date = report.get("check_date", "Unknown")
    overall_status = report.get("overall_status", "unknown")

    text = f"DATA QUALITY REPORT\n"
    text += f"{'=' * 50}\n\n"
    text += f"Check Date: {check_date}\n"
    text += f"Overall Status: {overall_status.upper()}\n"
    text += f"Severity: {severity.upper()}\n\n"

    # Summary section
    text += "SUMMARY\n"
    text += f"{'-' * 50}\n"
    checks = report.get("checks", {})
    for check_name, check_result in checks.items():
        status = check_result.get("status", "unknown")
        count = len(check_result.get("issues", []))
        text += f"• {check_name.replace('_', ' ').title()}: {status.upper()}"
        if count > 0:
            text += f" ({count} issues)"
        text += "\n"

    # Issues section
    text += f"\nISSUES DETECTED\n"
    text += f"{'-' * 50}\n"

    issues_found = False
    for check_name, check_result in checks.items():
        if check_result.get("status") in ["warning", "fail", "error"]:
            issues = check_result.get("issues", [])
            if issues:
                issues_found = True
                text += f"\n{check_name.replace('_', ' ').title()}:\n"
                for issue in issues[:10]:  # Limit to first 10
                    text += f"  - {issue}\n"
                if len(issues) > 10:
                    text += f"  ...and {len(issues) - 10} more issues\n"

    if not issues_found:
        text += "\nNo critical issues detected.\n"

    text += f"\n{'=' * 50}\n"
    text += "This is an automated alert from the HRP Data Quality Framework.\n"
    text += "For more details, check the quality metrics in your HRP dashboard.\n"

    return text
