"""
Data Quality Framework for HRP.

Provides comprehensive data quality monitoring including:
- Automated quality checks (anomaly, completeness, gaps, stale data)
- Daily quality report generation
- Health score tracking over time
- Email alerts for critical issues

Usage:
    from hrp.data.quality import generate_daily_report, run_quality_check_with_alerts

    # Generate a report
    report = generate_daily_report(as_of_date=date.today())
    print(f"Health Score: {report.health_score}")

    # Run checks with alerts
    result = run_quality_check_with_alerts()
"""

from hrp.data.quality.alerts import (
    QualityAlertManager,
    run_quality_check_with_alerts,
)
from hrp.data.quality.checks import (
    CheckResult,
    CompletenessCheck,
    GapDetectionCheck,
    IssueSeverity,
    PriceAnomalyCheck,
    QualityCheck,
    QualityIssue,
    StaleDataCheck,
    VolumeAnomalyCheck,
)
from hrp.data.quality.report import (
    QualityReport,
    QualityReportGenerator,
    generate_daily_report,
)

__all__ = [
    # Checks
    "QualityCheck",
    "CheckResult",
    "QualityIssue",
    "IssueSeverity",
    "PriceAnomalyCheck",
    "CompletenessCheck",
    "GapDetectionCheck",
    "StaleDataCheck",
    "VolumeAnomalyCheck",
    # Reports
    "QualityReport",
    "QualityReportGenerator",
    "generate_daily_report",
    # Alerts
    "QualityAlertManager",
    "run_quality_check_with_alerts",
]
