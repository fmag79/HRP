"""
Data quality alerting for HRP.

Sends email alerts for critical data quality issues
and daily summary reports.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger

from hrp.data.quality.checks import IssueSeverity, QualityIssue
from hrp.data.quality.report import QualityReport
from hrp.notifications.email import EmailNotifier


class QualityAlertManager:
    """
    Manages data quality alerts.

    Sends notifications for critical issues and daily summaries.
    Integrates with the existing EmailNotifier service.
    """

    def __init__(self):
        """Initialize the alert manager."""
        self._notifier = EmailNotifier()
        logger.info("Quality alert manager initialized")

    def send_critical_alert(
        self,
        issues: list[QualityIssue],
        report_date: date,
    ) -> bool:
        """
        Send an alert for critical quality issues.

        Args:
            issues: List of critical QualityIssue objects.
            report_date: Date of the quality report.

        Returns:
            True if alert was sent successfully.
        """
        if not issues:
            logger.debug("No critical issues to alert on")
            return False

        subject = f"[CRITICAL] Data Quality Alert - {report_date}"

        # Build HTML body
        issue_rows = ""
        for issue in issues[:20]:  # Limit to 20 issues
            issue_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{issue.symbol or 'N/A'}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{issue.check_name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{issue.description}</td>
            </tr>
            """

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #d32f2f;">Critical Data Quality Issues Detected</h2>
            <p><strong>Report Date:</strong> {report_date}</p>
            <p><strong>Critical Issues Found:</strong> {len(issues)}</p>

            <table style="border-collapse: collapse; width: 100%; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #f5f5f5;">
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Symbol</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Check</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Description</th>
                    </tr>
                </thead>
                <tbody>
                    {issue_rows}
                </tbody>
            </table>

            {"<p><em>Showing first 20 issues. See full report for details.</em></p>" if len(issues) > 20 else ""}

            <p style="margin-top: 20px; color: #666;">
                This is an automated alert from HRP Data Quality Framework.
            </p>
        </body>
        </html>
        """

        try:
            self._notifier.send_html_email(subject=subject, body=body)
            logger.info(f"Sent critical quality alert with {len(issues)} issues")
            return True
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")
            return False

    def send_daily_summary(self, report: QualityReport) -> bool:
        """
        Send a daily quality summary email.

        Args:
            report: QualityReport to summarize.

        Returns:
            True if email was sent successfully.
        """
        status_emoji = "✅" if report.passed else "❌"
        subject = f"{status_emoji} Daily Data Quality Report - {report.report_date}"

        # Build check results rows
        check_rows = ""
        for result in report.results:
            status_color = "#4caf50" if result.passed else "#f44336"
            status_text = "PASS" if result.passed else "FAIL"
            check_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{result.check_name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {status_color};">{status_text}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{result.critical_count}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{result.warning_count}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{result.run_time_ms:.0f}ms</td>
            </tr>
            """

        # Health score color
        score = report.health_score
        if score >= 90:
            score_color = "#4caf50"
        elif score >= 70:
            score_color = "#ff9800"
        else:
            score_color = "#f44336"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Daily Data Quality Report</h2>
            <p><strong>Date:</strong> {report.report_date}</p>

            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin-top: 0;">Health Score</h3>
                <div style="font-size: 48px; font-weight: bold; color: {score_color};">
                    {score:.0f}/100
                </div>
                <p style="color: #666; margin-bottom: 0;">
                    Status: <strong>{"PASSED" if report.passed else "FAILED"}</strong>
                </p>
            </div>

            <h3>Summary</h3>
            <ul>
                <li>Checks Run: {report.checks_run}</li>
                <li>Checks Passed: {report.checks_passed}</li>
                <li>Total Issues: {report.total_issues}</li>
                <li>Critical Issues: <span style="color: {'#f44336' if report.critical_issues > 0 else '#4caf50'};">{report.critical_issues}</span></li>
                <li>Warnings: {report.warning_issues}</li>
            </ul>

            <h3>Check Results</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #f5f5f5;">
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Check</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Status</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Critical</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Warnings</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Time</th>
                    </tr>
                </thead>
                <tbody>
                    {check_rows}
                </tbody>
            </table>

            <p style="margin-top: 20px; color: #666; font-size: 12px;">
                Generated at {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} |
                Total runtime: {report.total_run_time_ms:.0f}ms
            </p>
        </body>
        </html>
        """

        try:
            self._notifier.send_html_email(subject=subject, body=body)
            logger.info(f"Sent daily quality summary for {report.report_date}")
            return True
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False

    def process_report(self, report: QualityReport, send_summary: bool = True) -> dict[str, Any]:
        """
        Process a quality report and send appropriate alerts.

        Sends critical alert if there are critical issues,
        and optionally sends daily summary.

        Args:
            report: QualityReport to process.
            send_summary: Whether to send daily summary email.

        Returns:
            Dictionary with alert status.
        """
        result = {
            "critical_alert_sent": False,
            "summary_sent": False,
            "critical_issues": report.critical_issues,
        }

        # Send critical alert if needed
        if report.critical_issues > 0:
            critical_issues = report.get_critical_issues()
            result["critical_alert_sent"] = self.send_critical_alert(
                critical_issues, report.report_date
            )

        # Send daily summary
        if send_summary:
            result["summary_sent"] = self.send_daily_summary(report)

        return result


def run_quality_check_with_alerts(
    db_path: str | None = None,
    as_of_date: date | None = None,
    send_summary: bool = True,
    store_report: bool = True,
) -> dict[str, Any]:
    """
    Run quality checks and send alerts.

    Convenience function that:
    1. Generates a quality report
    2. Stores it in the database
    3. Sends alerts for critical issues
    4. Optionally sends daily summary

    Args:
        db_path: Optional database path.
        as_of_date: Date to check. Defaults to today.
        send_summary: Whether to send daily summary email.
        store_report: Whether to store report in database.

    Returns:
        Dictionary with report and alert status.
    """
    from hrp.data.quality.report import QualityReportGenerator

    as_of_date = as_of_date or date.today()

    # Generate report
    generator = QualityReportGenerator(db_path)
    report = generator.generate_report(as_of_date)

    # Store report
    report_id = None
    if store_report:
        report_id = generator.store_report(report)

    # Process alerts
    alert_manager = QualityAlertManager()
    alert_result = alert_manager.process_report(report, send_summary)

    return {
        "report_date": as_of_date,
        "report_id": report_id,
        "health_score": report.health_score,
        "passed": report.passed,
        "total_issues": report.total_issues,
        "critical_issues": report.critical_issues,
        "warning_issues": report.warning_issues,
        **alert_result,
    }
