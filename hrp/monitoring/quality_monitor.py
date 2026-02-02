#!/usr/bin/env python
"""
Automated Data Quality Monitoring System for HRP.

This module provides a comprehensive monitoring system that:
1. Runs scheduled quality checks (daily)
2. Sends email alerts for critical issues
3. Tracks health score trends
4. Monitors data freshness
5. Detects anomalies and outliers

Alert Thresholds:
- Health Score < 90: Warning email
- Health Score < 70: Critical email + SMS if configured
- New critical issues: Immediate alert
- Data freshness > 3 days: Critical alert
- Anomaly detection spike: Alert

Usage:
    from hrp.monitoring.quality_monitor import DataQualityMonitor

    monitor = DataQualityMonitor()
    monitor.run_daily_check()

    # Or via scheduler
    scheduler = IngestionScheduler()
    scheduler.setup_quality_monitoring(
        check_time="06:00",  # 6 AM ET
        health_threshold=90,
        send_alerts=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from hrp.data.quality.alerts import QualityAlertManager
from hrp.data.quality.report import QualityReport, QualityReportGenerator


@dataclass
class MonitoringThresholds:
    """
    Thresholds for triggering monitoring alerts.

    Attributes:
        health_score_warning: Health score below this triggers warning (default: 90)
        health_score_critical: Health score below this triggers critical alert (default: 70)
        freshness_warning_days: Data older than this triggers warning (default: 3)
        freshness_critical_days: Data older than this triggers critical alert (default: 5)
        anomaly_count_critical: Number of anomalies triggering critical alert (default: 100)
    """

    health_score_warning: float = 90.0
    health_score_critical: float = 70.0
    freshness_warning_days: int = 3
    freshness_critical_days: int = 5
    anomaly_count_critical: int = 100


@dataclass
class MonitoringResult:
    """Result of a monitoring check."""

    timestamp: datetime
    health_score: float
    passed: bool
    critical_issues: int
    warning_issues: int
    total_issues: int
    alerts_sent: dict[str, bool] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    trend: str = "stable"  # "improving", "stable", "declining"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.health_score,
            "passed": self.passed,
            "critical_issues": self.critical_issues,
            "warning_issues": self.warning_issues,
            "total_issues": self.total_issues,
            "alerts_sent": self.alerts_sent,
            "recommendations": self.recommendations,
            "trend": self.trend,
        }


class DataQualityMonitor:
    """
    Automated data quality monitoring system.

    Features:
    - Scheduled daily quality checks
    - Health score trend tracking
    - Threshold-based alerting
    - Data freshness monitoring
    - Anomaly detection alerts

    Example:
        monitor = DataQualityMonitor(
            thresholds=MonitoringThresholds(health_score_warning=90),
            send_alerts=True,
        )
        result = monitor.run_daily_check()
    """

    def __init__(
        self,
        thresholds: MonitoringThresholds | None = None,
        send_alerts: bool = True,
        db_path: str | None = None,
    ):
        """
        Initialize the data quality monitor.

        Args:
            thresholds: Custom alert thresholds (uses defaults if None)
            send_alerts: Whether to send email alerts for issues
            db_path: Optional database path
        """
        self.thresholds = thresholds or MonitoringThresholds()
        self.send_alerts = send_alerts
        self.db_path = db_path
        self.alert_manager = QualityAlertManager() if send_alerts else None
        self.report_generator = QualityReportGenerator(db_path)

        logger.info(
            f"DataQualityMonitor initialized (thresholds: "
            f"health_warning={self.thresholds.health_score_warning}, "
            f"health_critical={self.thresholds.health_score_critical})"
        )

    def run_daily_check(self, as_of_date: date | None = None) -> MonitoringResult:
        """
        Run the daily quality monitoring check.

        This is the main entry point for scheduled monitoring. It:
        1. Generates a quality report
        2. Stores the report for historical tracking
        3. Checks health score against thresholds
        4. Analyzes trends
        5. Sends alerts if thresholds breached
        6. Provides recommendations

        Args:
            as_of_date: Date to check (defaults to today)

        Returns:
            MonitoringResult with check status and alerts sent
        """
        as_of_date = as_of_date or date.today()

        # Resolve to most recent trading day so checks don't flag
        # missing data on weekends/holidays when markets are closed
        from hrp.utils.calendar import get_previous_trading_day

        as_of_date = get_previous_trading_day(as_of_date)
        logger.info(f"Running daily quality check for {as_of_date}")

        # Generate quality report
        report = self.report_generator.generate_report(as_of_date)

        # Store report for historical tracking
        report_id = self.report_generator.store_report(report)
        logger.info(f"Quality report stored: {report_id}")

        # Calculate trend
        trend = self._calculate_trend(as_of_date)

        # Generate recommendations
        recommendations = self._generate_recommendations(report)

        # Initialize result
        result = MonitoringResult(
            timestamp=datetime.now(),
            health_score=report.health_score,
            passed=report.passed,
            critical_issues=report.critical_issues,
            warning_issues=report.warning_issues,
            total_issues=report.total_issues,
            recommendations=recommendations,
            trend=trend,
        )

        # Check thresholds and send alerts
        if self.send_alerts and self.alert_manager:
            result.alerts_sent = self._check_and_alert(report, as_of_date)

        # Log summary
        logger.info(
            f"Daily check complete: score={report.health_score:.0f}/100, "
            f"critical={report.critical_issues}, warnings={report.warning_issues}, "
            f"trend={trend}, alerts_sent={sum(result.alerts_sent.values())}"
        )

        return result

    def _calculate_trend(self, as_of_date: date) -> str:
        """
        Calculate health score trend over last 7 days.

        Args:
            as_of_date: Date to calculate trend for

        Returns:
            "improving", "stable", or "declining"
        """
        try:
            trend_data = self.report_generator.get_health_trend(days=7)
            if not trend_data or len(trend_data) < 2:
                return "stable"

            # Get recent scores
            recent_scores = [d["health_score"] for d in trend_data[-7:]]

            # Calculate simple linear regression
            if len(recent_scores) < 2:
                return "stable"

            avg_start = sum(recent_scores[:3]) / min(3, len(recent_scores))
            avg_end = sum(recent_scores[-3:]) / min(3, len(recent_scores))

            diff = avg_end - avg_start

            if diff > 5:
                return "improving"
            elif diff < -5:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.warning(f"Failed to calculate trend: {e}")
            return "stable"

    def _generate_recommendations(self, report: QualityReport) -> list[str]:
        """
        Generate actionable recommendations based on quality report.

        Args:
            report: Quality report to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Health score recommendations
        if report.health_score < self.thresholds.health_score_critical:
            recommendations.append(
                "URGENT: Health score critically low. Review all critical issues immediately."
            )
        elif report.health_score < self.thresholds.health_score_warning:
            recommendations.append(
                "Health score below target. Address warning issues before they become critical."
            )

        # Critical issues
        if report.critical_issues > 0:
            critical_issues = report.get_critical_issues()
            by_check = {}
            for issue in critical_issues:
                by_check[issue.check_name] = by_check.get(issue.check_name, 0) + 1

            for check_name, count in sorted(by_check.items(), key=lambda x: -x[1]):
                recommendations.append(
                    f"Fix {count} critical {check_name} issues"
                )

        # Warning issues
        if report.warning_issues > 50:
            recommendations.append(
                f"High warning count ({report.warning_issues}). Schedule maintenance to reduce noise."
            )

        # Stale data
        stale_count = 0
        for result in report.results:
            if result.check_name == "stale_data" and not result.passed:
                stale_count = result.critical_count + result.warning_count
                break

        if stale_count > 0:
            recommendations.append(
                f"{stale_count} symbols have stale data. Check ingestion pipeline."
            )

        # Data freshness
        try:
            from hrp.api.platform import PlatformAPI

            api = PlatformAPI()
            result = api.fetchone_readonly("SELECT MAX(date) FROM prices")
            if result and result[0]:
                last_date = result[0]
                if isinstance(last_date, str):
                    last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

                days_stale = (date.today() - last_date).days
                if days_stale > self.thresholds.freshness_critical_days:
                    recommendations.append(
                        f"CRITICAL: Price data is {days_stale} days stale. Ingestion may be failing."
                    )
                elif days_stale > self.thresholds.freshness_warning_days:
                    recommendations.append(
                        f"WARNING: Price data is {days_stale} days stale."
                    )
        except Exception as e:
            logger.warning(f"Failed to check data freshness: {e}")

        return recommendations

    def _check_and_alert(
        self,
        report: QualityReport,
        as_of_date: date,
    ) -> dict[str, bool]:
        """
        Check report against thresholds and send alerts.

        Args:
            report: Quality report to check
            as_of_date: Report date

        Returns:
            Dictionary of which alerts were sent
        """
        alerts_sent = {
            "health_warning": False,
            "health_critical": False,
            "critical_issues": False,
            "data_freshness": False,
            "anomaly_spike": False,
        }

        # Health score alerts
        if report.health_score < self.thresholds.health_score_critical:
            logger.warning(
                f"Health score {report.health_score:.0f} below critical threshold "
                f"{self.thresholds.health_score_critical}"
            )
            # Critical health score - immediate alert
            try:
                if self.alert_manager:
                    critical_issues = report.get_critical_issues()
                    alerts_sent["health_critical"] = self.alert_manager.send_critical_alert(
                        critical_issues, as_of_date
                    )
                    alerts_sent["critical_issues"] = alerts_sent["health_critical"]
            except Exception as e:
                logger.error(f"Failed to send critical health alert: {e}")

        elif report.health_score < self.thresholds.health_score_warning:
            logger.warning(
                f"Health score {report.health_score:.0f} below warning threshold "
                f"{self.thresholds.health_score_warning}"
            )
            # Warning health score - daily summary (includes warning in email)
            try:
                if self.alert_manager:
                    alerts_sent["health_warning"] = self.alert_manager.send_daily_summary(
                        report
                    )
            except Exception as e:
                logger.error(f"Failed to send warning health alert: {e}")

        # Critical issues alert (even if health score is okay)
        if report.critical_issues > 0:
            logger.warning(f"Found {report.critical_issues} critical issues")
            if not alerts_sent["critical_issues"] and self.alert_manager:
                try:
                    critical_issues = report.get_critical_issues()
                    alerts_sent["critical_issues"] = self.alert_manager.send_critical_alert(
                        critical_issues, as_of_date
                    )
                except Exception as e:
                    logger.error(f"Failed to send critical issues alert: {e}")

        # Data freshness alert
        try:
            from hrp.api.platform import PlatformAPI

            api = PlatformAPI()
            result = api.fetchone_readonly("SELECT MAX(date) FROM prices")
            if result and result[0]:
                last_date = result[0]
                if isinstance(last_date, str):
                    last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

                days_stale = (date.today() - last_date).days
                if days_stale > self.thresholds.freshness_critical_days:
                    logger.critical(
                        f"Price data is {days_stale} days stale (critical threshold: "
                        f"{self.thresholds.freshness_critical_days})"
                    )
                    # Send freshness alert
                    if self.alert_manager:
                        freshness_issues = [
                            QualityIssue.create(
                                check_name="data_freshness",
                                severity=IssueSeverity.CRITICAL,
                                description=f"Price data is {days_stale} days old",
                                details={"last_date": str(last_date), "days_stale": days_stale},
                            )
                        ]
                        alerts_sent["data_freshness"] = self.alert_manager.send_critical_alert(
                            freshness_issues, as_of_date
                        )
        except Exception as e:
            logger.warning(f"Failed to check data freshness for alerting: {e}")

        # Anomaly spike detection
        total_anomalies = report.total_issues
        if total_anomalies > self.thresholds.anomaly_count_critical:
            logger.warning(
                f"Anomaly count spike: {total_anomalies} anomalies "
                f"(threshold: {self.thresholds.anomaly_count_critical})"
            )
            if self.alert_manager:
                try:
                    # Send spike alert
                    spike_issues = [
                        QualityIssue.create(
                            check_name="anomaly_spike",
                            severity=IssueSeverity.WARNING,
                            description=f"Unusually high anomaly count: {total_anomalies}",
                            details={
                                "total_anomalies": total_anomalies,
                                "threshold": self.thresholds.anomaly_count_critical,
                            },
                        )
                    ]
                    alerts_sent["anomaly_spike"] = self.alert_manager.send_critical_alert(
                        spike_issues, as_of_date
                    )
                except Exception as e:
                    logger.error(f"Failed to send anomaly spike alert: {e}")

        return alerts_sent

    def get_health_summary(self, days: int = 30) -> dict[str, Any]:
        """
        Get health summary for the monitoring dashboard.

        Args:
            days: Number of days to include in summary

        Returns:
            Dictionary with health summary metrics
        """
        try:
            # Get current report
            current_report = self.report_generator.generate_report(date.today())

            # Get trend data
            trend_data = self.report_generator.get_health_trend(days=days)

            # Calculate trend direction
            if trend_data and len(trend_data) >= 2:
                recent_scores = [d["health_score"] for d in trend_data[-7:]]
                if len(recent_scores) >= 2:
                    avg_start = sum(recent_scores[:3]) / min(3, len(recent_scores))
                    avg_end = sum(recent_scores[-3:]) / min(3, len(recent_scores))
                    trend_direction = "improving" if avg_end > avg_start + 5 else "declining" if avg_end < avg_start - 5 else "stable"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "unknown"

            return {
                "current_health_score": current_report.health_score,
                "current_critical": current_report.critical_issues,
                "current_warnings": current_report.warning_issues,
                "trend_direction": trend_direction,
                "trend_days": days,
                "trend_data_points": len(trend_data) if trend_data else 0,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get health summary: {e}")
            return {
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }


# Import QualityIssue at the end to avoid circular dependency
from hrp.data.quality.checks import IssueSeverity, QualityIssue


def run_quality_monitor_with_alerts(
    as_of_date: date | None = None,
    send_alerts: bool = True,
    thresholds: MonitoringThresholds | None = None,
) -> MonitoringResult:
    """
    Run quality monitoring with alerts (convenience function).

    This is a convenience wrapper around DataQualityMonitor that:
    1. Runs the quality check
    2. Stores the report
    3. Sends appropriate alerts based on thresholds
    4. Returns the monitoring result

    Args:
        as_of_date: Date to check (defaults to today)
        send_alerts: Whether to send email alerts
        thresholds: Custom alert thresholds

    Returns:
        MonitoringResult with check status

    Example:
        result = run_quality_monitor_with_alerts()
        print(f"Health Score: {result.health_score}")
        print(f"Trend: {result.trend}")
        print(f"Alerts Sent: {sum(result.alerts_sent.values())}")
    """
    monitor = DataQualityMonitor(
        thresholds=thresholds,
        send_alerts=send_alerts,
    )
    return monitor.run_daily_check(as_of_date)
