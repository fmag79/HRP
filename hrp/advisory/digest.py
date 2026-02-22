"""
Weekly email digest for the advisory service.

Generates and sends a weekly recommendation email with new picks,
open position updates, recent outcomes, and track record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from hrp.advisory.recommendation_engine import Recommendation, RecommendationUpdate
    from hrp.advisory.track_record import TrackRecordSummary, WeeklyReport


@dataclass
class DigestContent:
    """Structured email content for the weekly digest."""

    subject: str
    html_body: str
    text_body: str


class WeeklyDigest:
    """Generates and sends the weekly recommendation email."""

    def generate(self, report: WeeklyReport) -> DigestContent:
        """Generate email content from a weekly report."""
        subject = f"Your Weekly Market Brief — {report.report_date.strftime('%b %d, %Y')}"

        html_parts = [self._html_header(report.report_date)]

        # New recommendations
        if report.new_recommendations:
            html_parts.append(self._html_new_recommendations(report.new_recommendations))

        # Open positions
        if report.open_positions:
            html_parts.append(self._html_open_positions(report.open_positions))

        # Closed this week
        if report.closed_this_week:
            html_parts.append(self._html_closed(report.closed_this_week))

        # Track record
        html_parts.append(self._html_track_record(report.track_record))

        html_parts.append(self._html_footer())

        html_body = "\n".join(html_parts)
        text_body = self._generate_text_version(report)

        return DigestContent(
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )

    def send(self, content: DigestContent, to_email: str) -> bool:
        """Send the digest email via the existing notification system."""
        try:
            from hrp.notifications.email import EmailNotifier
            notifier = EmailNotifier()
            notifier.send_summary_email(
                subject=content.subject,
                body=content.html_body,
            )
            logger.info(f"Weekly digest sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send weekly digest: {e}")
            return False

    # --- HTML generation ---

    def _html_header(self, report_date: date) -> str:
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    max-width: 600px; margin: 0 auto; color: #333;">
        <h1 style="color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px;">
            Weekly Market Brief
        </h1>
        <p style="color: #666; font-size: 14px;">
            {report_date.strftime('%B %d, %Y')}
        </p>
        """

    def _html_new_recommendations(self, recs: list[dict]) -> str:
        items = []
        for rec in recs:
            confidence_color = {
                "HIGH": "#27ae60", "MEDIUM": "#f39c12", "LOW": "#e74c3c"
            }.get(rec.get("confidence", ""), "#999")

            items.append(f"""
            <div style="background: #f8f9fa; border-left: 4px solid {confidence_color};
                        padding: 12px 16px; margin: 8px 0; border-radius: 4px;">
                <strong style="font-size: 16px;">
                    {rec.get('action', 'BUY')} {rec.get('symbol', '')}
                </strong>
                <span style="background: {confidence_color}; color: white;
                             padding: 2px 8px; border-radius: 12px; font-size: 12px;
                             margin-left: 8px;">
                    {rec.get('confidence', '')}
                </span>
                <p style="margin: 8px 0 4px; color: #555;">
                    {rec.get('thesis_plain', '')}
                </p>
                <p style="margin: 4px 0; color: #888; font-size: 13px;">
                    Risk: {rec.get('risk_plain', '')}
                </p>
                <p style="margin: 4px 0; font-size: 13px; color: #666;">
                    Entry: ${rec.get('entry_price', 0):.2f} |
                    Target: ${rec.get('target_price', 0):.2f} |
                    Stop: ${rec.get('stop_price', 0):.2f}
                </p>
            </div>
            """)

        return f"""
        <h2 style="color: #16213e; margin-top: 24px;">This Week's Recommendations</h2>
        {''.join(items)}
        """

    def _html_open_positions(self, positions: list[dict]) -> str:
        rows = []
        for pos in positions:
            rows.append(f"""
            <tr>
                <td style="padding: 6px 12px;">{pos.get('symbol', '')}</td>
                <td style="padding: 6px 12px;">${pos.get('entry_price', 0):.2f}</td>
                <td style="padding: 6px 12px;">
                    {float(pos.get('signal_strength', 0)):.2f}
                </td>
            </tr>
            """)

        return f"""
        <h2 style="color: #16213e; margin-top: 24px;">Open Positions ({len(positions)})</h2>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr style="background: #f0f0f0; font-weight: bold;">
                <td style="padding: 6px 12px;">Symbol</td>
                <td style="padding: 6px 12px;">Entry</td>
                <td style="padding: 6px 12px;">Signal</td>
            </tr>
            {''.join(rows)}
        </table>
        """

    def _html_closed(self, closed: list[dict]) -> str:
        items = []
        for rec in closed:
            ret = float(rec.get("realized_return", 0))
            icon = "+" if ret > 0 else ""
            color = "#27ae60" if ret > 0 else "#e74c3c"
            items.append(f"""
            <div style="padding: 4px 0; font-size: 14px;">
                <span style="color: {color}; font-weight: bold;">
                    {icon}{ret:.1%}
                </span>
                {rec.get('symbol', '')} — {rec.get('status', '')}
            </div>
            """)

        return f"""
        <h2 style="color: #16213e; margin-top: 24px;">Closed This Week</h2>
        {''.join(items)}
        """

    def _html_track_record(self, tr: TrackRecordSummary) -> str:
        excess_color = "#27ae60" if tr.excess_return > 0 else "#e74c3c"
        return f"""
        <h2 style="color: #16213e; margin-top: 24px;">Track Record</h2>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr>
                <td style="padding: 4px 0; color: #666;">Win Rate</td>
                <td style="padding: 4px 0; font-weight: bold;">{tr.win_rate:.0%}</td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #666;">Avg Return</td>
                <td style="padding: 4px 0; font-weight: bold;">{tr.avg_return:+.1%}</td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #666;">Avg Win / Avg Loss</td>
                <td style="padding: 4px 0;">{tr.avg_win:+.1%} / {tr.avg_loss:+.1%}</td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #666;">vs SPY</td>
                <td style="padding: 4px 0; color: {excess_color}; font-weight: bold;">
                    {tr.excess_return:+.1%}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #666;">Total Closed</td>
                <td style="padding: 4px 0;">{tr.closed_recommendations}</td>
            </tr>
        </table>
        """

    def _html_footer(self) -> str:
        return """
        <hr style="margin-top: 24px; border: none; border-top: 1px solid #ddd;">
        <p style="color: #999; font-size: 12px;">
            This is not financial advice. Past performance does not guarantee future results.
            All recommendations are generated by automated models and should be reviewed
            before acting. Consult a financial advisor for personalized advice.
        </p>
        </div>
        """

    def _generate_text_version(self, report: WeeklyReport) -> str:
        """Generate plain-text fallback for the email."""
        lines = [
            f"Weekly Market Brief — {report.report_date.strftime('%B %d, %Y')}",
            "=" * 50,
            "",
        ]

        if report.new_recommendations:
            lines.append("THIS WEEK'S RECOMMENDATIONS")
            lines.append("-" * 30)
            for rec in report.new_recommendations:
                lines.append(
                    f"  {rec.get('action', 'BUY')} {rec.get('symbol', '')} "
                    f"({rec.get('confidence', '')})"
                )
                lines.append(f"  {rec.get('thesis_plain', '')}")
                lines.append("")

        tr = report.track_record
        lines.extend([
            "TRACK RECORD",
            "-" * 30,
            f"  Win Rate: {tr.win_rate:.0%}",
            f"  Avg Return: {tr.avg_return:+.1%}",
            f"  vs SPY: {tr.excess_return:+.1%}",
            "",
            "This is not financial advice.",
        ])

        return "\n".join(lines)
