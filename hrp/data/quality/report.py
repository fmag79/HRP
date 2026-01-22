"""
Data quality report generation for HRP.

Generates comprehensive quality reports by running all configured checks
and aggregating results. Supports historical tracking of quality metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db
from hrp.data.quality.checks import (
    DEFAULT_CHECKS,
    CheckResult,
    IssueSeverity,
    QualityCheck,
    QualityIssue,
)


@dataclass
class QualityReport:
    """Complete data quality report."""

    report_date: date
    generated_at: datetime
    checks_run: int
    checks_passed: int
    total_issues: int
    critical_issues: int
    warning_issues: int
    results: list[CheckResult] = field(default_factory=list)
    total_run_time_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Report passes if no critical issues."""
        return self.critical_issues == 0

    @property
    def health_score(self) -> float:
        """
        Calculate overall data health score (0-100).

        Score is reduced by critical and warning issues.
        """
        if self.checks_run == 0:
            return 100.0

        base_score = 100.0
        # Deduct 20 points per critical issue (max 60)
        critical_penalty = min(60, self.critical_issues * 20)
        # Deduct 5 points per warning (max 30)
        warning_penalty = min(30, self.warning_issues * 5)

        return max(0, base_score - critical_penalty - warning_penalty)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "report_date": str(self.report_date),
            "generated_at": self.generated_at.isoformat(),
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "warning_issues": self.warning_issues,
            "health_score": self.health_score,
            "passed": self.passed,
            "total_run_time_ms": self.total_run_time_ms,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "critical_count": r.critical_count,
                    "warning_count": r.warning_count,
                    "stats": r.stats,
                    "run_time_ms": r.run_time_ms,
                    "issues": [i.to_dict() for i in r.issues],
                }
                for r in self.results
            ],
        }

    def get_critical_issues(self) -> list[QualityIssue]:
        """Get all critical issues across all checks."""
        critical = []
        for result in self.results:
            critical.extend(
                i for i in result.issues if i.severity == IssueSeverity.CRITICAL
            )
        return critical

    def get_summary_text(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            f"Data Quality Report - {self.report_date}",
            "=" * 50,
            f"Health Score: {self.health_score:.0f}/100",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            "",
            f"Checks Run: {self.checks_run}",
            f"Checks Passed: {self.checks_passed}",
            f"Total Issues: {self.total_issues}",
            f"  Critical: {self.critical_issues}",
            f"  Warnings: {self.warning_issues}",
            "",
            "Check Results:",
            "-" * 30,
        ]

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(
                f"  {result.check_name}: {status} "
                f"({result.critical_count} critical, {result.warning_count} warnings)"
            )

        if self.critical_issues > 0:
            lines.extend(["", "Critical Issues:", "-" * 30])
            for issue in self.get_critical_issues()[:10]:  # First 10
                lines.append(f"  [{issue.symbol}] {issue.description}")

        return "\n".join(lines)


class QualityReportGenerator:
    """
    Generates data quality reports.

    Runs all configured quality checks and aggregates results
    into a comprehensive report. Supports storing reports for
    historical tracking.
    """

    def __init__(
        self,
        db_path: str | None = None,
        checks: list[type[QualityCheck]] | None = None,
    ):
        """
        Initialize the report generator.

        Args:
            db_path: Optional database path.
            checks: List of check classes to run. Defaults to DEFAULT_CHECKS.
        """
        self._db = get_db(db_path)
        self._db_path = db_path
        self._check_classes = checks or DEFAULT_CHECKS
        logger.info(f"Quality report generator initialized with {len(self._check_classes)} checks")

    def generate_report(self, as_of_date: date) -> QualityReport:
        """
        Generate a quality report for the given date.

        Args:
            as_of_date: Date to generate report for.

        Returns:
            QualityReport with all check results.
        """
        logger.info(f"Generating quality report for {as_of_date}")

        results = []
        total_run_time = 0.0

        for check_class in self._check_classes:
            try:
                check = check_class(self._db_path)
                result = check.run(as_of_date)
                results.append(result)
                total_run_time += result.run_time_ms
                logger.debug(
                    f"Check {check.name}: {'PASS' if result.passed else 'FAIL'} "
                    f"({len(result.issues)} issues, {result.run_time_ms:.1f}ms)"
                )
            except Exception as e:
                logger.error(f"Check {check_class.name} failed: {e}")
                # Create a failed result
                results.append(
                    CheckResult(
                        check_name=check_class.name,
                        passed=False,
                        issues=[
                            QualityIssue(
                                check_name=check_class.name,
                                severity=IssueSeverity.CRITICAL,
                                symbol=None,
                                date=as_of_date,
                                description=f"Check failed with error: {e}",
                            )
                        ],
                    )
                )

        # Aggregate stats
        checks_passed = sum(1 for r in results if r.passed)
        total_issues = sum(len(r.issues) for r in results)
        critical_issues = sum(r.critical_count for r in results)
        warning_issues = sum(r.warning_count for r in results)

        report = QualityReport(
            report_date=as_of_date,
            generated_at=datetime.now(),
            checks_run=len(results),
            checks_passed=checks_passed,
            total_issues=total_issues,
            critical_issues=critical_issues,
            warning_issues=warning_issues,
            results=results,
            total_run_time_ms=total_run_time,
        )

        logger.info(
            f"Report generated: score={report.health_score:.0f}, "
            f"critical={critical_issues}, warnings={warning_issues}"
        )

        return report

    def store_report(self, report: QualityReport) -> int:
        """
        Store a quality report in the database.

        Args:
            report: QualityReport to store.

        Returns:
            Report ID.
        """
        # Ensure quality_reports table exists
        self._ensure_quality_tables()

        with self._db.connection() as conn:
            # Get next report ID
            result = conn.execute(
                "SELECT COALESCE(MAX(report_id), 0) + 1 FROM quality_reports"
            ).fetchone()
            report_id = result[0]

            # Store report
            conn.execute(
                """
                INSERT INTO quality_reports (
                    report_id, report_date, generated_at, checks_run, checks_passed,
                    total_issues, critical_issues, warning_issues, health_score,
                    passed, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    report.report_date,
                    report.generated_at,
                    report.checks_run,
                    report.checks_passed,
                    report.total_issues,
                    report.critical_issues,
                    report.warning_issues,
                    report.health_score,
                    report.passed,
                    json.dumps(report.to_dict()),
                ),
            )

            logger.info(f"Stored quality report {report_id} for {report.report_date}")
            return report_id

    def get_historical_reports(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """
        Get historical quality reports.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            List of report summaries (without full JSON).
        """
        query = """
            SELECT
                report_id, report_date, generated_at, checks_run, checks_passed,
                total_issues, critical_issues, warning_issues, health_score, passed
            FROM quality_reports
            WHERE report_date BETWEEN ? AND ?
            ORDER BY report_date DESC
        """

        results = self._db.fetchall(query, (start_date, end_date))

        return [
            {
                "report_id": r[0],
                "report_date": r[1],
                "generated_at": r[2],
                "checks_run": r[3],
                "checks_passed": r[4],
                "total_issues": r[5],
                "critical_issues": r[6],
                "warning_issues": r[7],
                "health_score": r[8],
                "passed": r[9],
            }
            for r in results
        ]

    def get_health_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """
        Get health score trend over time.

        Args:
            days: Number of days to look back.

        Returns:
            List of {date, health_score, critical_issues} dicts.
        """
        query = """
            SELECT report_date, health_score, critical_issues
            FROM quality_reports
            WHERE report_date >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY report_date
        """

        results = self._db.fetchall(query, (days,))

        return [
            {
                "date": r[0],
                "health_score": r[1],
                "critical_issues": r[2],
            }
            for r in results
        ]

    def _ensure_quality_tables(self) -> None:
        """Ensure quality reporting tables exist."""
        with self._db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id INTEGER PRIMARY KEY,
                    report_date DATE NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    checks_run INTEGER NOT NULL,
                    checks_passed INTEGER NOT NULL,
                    total_issues INTEGER NOT NULL,
                    critical_issues INTEGER NOT NULL,
                    warning_issues INTEGER NOT NULL,
                    health_score DECIMAL(5,2) NOT NULL,
                    passed BOOLEAN NOT NULL,
                    report_json JSON
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_reports_date
                ON quality_reports(report_date)
                """
            )


def generate_daily_report(
    db_path: str | None = None,
    as_of_date: date | None = None,
    store: bool = True,
) -> QualityReport:
    """
    Convenience function to generate a daily quality report.

    Args:
        db_path: Optional database path.
        as_of_date: Date to generate for. Defaults to today.
        store: Whether to store the report in the database.

    Returns:
        QualityReport.
    """
    as_of_date = as_of_date or date.today()
    generator = QualityReportGenerator(db_path)
    report = generator.generate_report(as_of_date)

    if store:
        generator.store_report(report)

    return report
