"""
Tests for Data Quality Framework.

Tests cover:
- Individual quality checks (anomaly, completeness, gaps, stale)
- Report generation
- Health score calculation
- Historical tracking
"""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.quality.checks import (
    CheckResult,
    CompletenessCheck,
    GapDetectionCheck,
    IssueSeverity,
    PriceAnomalyCheck,
    QualityIssue,
    StaleDataCheck,
    VolumeAnomalyCheck,
)
from hrp.data.quality.report import QualityReport, QualityReportGenerator
from hrp.data.schema import create_tables


class TestPriceAnomalyCheck:
    """Tests for price anomaly detection."""

    def test_detects_large_price_change(self, test_db):
        """Should flag price changes >50% without corporate action."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert prices with large change
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES
                    ('TEST', '2024-01-10', 100.00, 'test'),
                    ('TEST', '2024-01-11', 200.00, 'test')
                """
            )

        check = PriceAnomalyCheck(test_db, threshold=0.5, read_only=False)
        result = check.run(date(2024, 1, 11))

        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0].symbol == "TEST"
        assert result.issues[0].severity in (IssueSeverity.WARNING, IssueSeverity.CRITICAL)

    def test_ignores_change_with_corporate_action(self, test_db):
        """Should not flag price changes with corporate action."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            # Price doubles due to split
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES
                    ('SPLIT', '2024-01-10', 100.00, 'test'),
                    ('SPLIT', '2024-01-11', 50.00, 'test')
                """
            )
            # Record the split
            conn.execute(
                """
                INSERT INTO corporate_actions (symbol, date, action_type, factor)
                VALUES ('SPLIT', '2024-01-11', 'split', 2.0)
                """
            )

        check = PriceAnomalyCheck(test_db, threshold=0.5, read_only=False)
        result = check.run(date(2024, 1, 11))

        # Should not flag SPLIT because it has a corporate action
        split_issues = [i for i in result.issues if i.symbol == "SPLIT"]
        assert len(split_issues) == 0

    def test_passes_with_normal_changes(self, test_db):
        """Should pass when price changes are normal."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES
                    ('NORM', '2024-01-10', 100.00, 'test'),
                    ('NORM', '2024-01-11', 102.00, 'test')
                """
            )

        check = PriceAnomalyCheck(test_db, threshold=0.5, read_only=False)
        result = check.run(date(2024, 1, 11))

        norm_issues = [i for i in result.issues if i.symbol == "NORM"]
        assert len(norm_issues) == 0


class TestCompletenessCheck:
    """Tests for data completeness checking."""

    def test_detects_missing_prices(self, test_db):
        """Should flag symbols in universe without prices."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            # Add symbols to universe
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-15', TRUE, 'Technology'),
                    ('MSFT', '2024-01-15', TRUE, 'Technology')
                """
            )
            # Only add prices for AAPL
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES ('AAPL', '2024-01-15', 150.00, 'test')
                """
            )

        check = CompletenessCheck(test_db, read_only=False)
        result = check.run(date(2024, 1, 15))

        # Missing prices are flagged as WARNING, not CRITICAL
        # So passed is True (no critical issues), but there should be 1 warning issue
        assert len(result.issues) == 1
        assert result.issues[0].symbol == "MSFT"
        assert result.issues[0].severity == IssueSeverity.WARNING
        assert result.warning_count == 1

    def test_passes_when_complete(self, test_db):
        """Should pass when all universe symbols have prices."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES ('AAPL', '2024-01-15', TRUE, 'Technology')
                """
            )
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES ('AAPL', '2024-01-15', 150.00, 'test')
                """
            )

        check = CompletenessCheck(test_db, read_only=False)
        result = check.run(date(2024, 1, 15))

        assert result.passed


class TestGapDetectionCheck:
    """Tests for price history gap detection."""

    def test_detects_gaps_in_history(self, test_db):
        """Should flag symbols with missing trading days."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create prices for reference symbol (all days)
        with db.connection() as conn:
            for i in range(20):
                dt = date(2024, 1, 1) + timedelta(days=i)
                if dt.weekday() < 5:  # Weekdays only
                    conn.execute(
                        """
                        INSERT INTO prices (symbol, date, close, source)
                        VALUES ('FULL', ?, 100.00, 'test')
                        """,
                        (dt,),
                    )

            # Create prices for symbol with gaps (only some days)
            for i in [0, 5, 10, 15]:
                dt = date(2024, 1, 1) + timedelta(days=i)
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, source)
                    VALUES ('GAPS', ?, 100.00, 'test')
                    """,
                    (dt,),
                )

        check = GapDetectionCheck(test_db, lookback_days=20, read_only=False)
        result = check.run(date(2024, 1, 20))

        gap_issues = [i for i in result.issues if i.symbol == "GAPS"]
        assert len(gap_issues) > 0

    def test_ignores_weekends_and_holidays(self, test_db):
        """Should not flag gaps for weekends, NYSE holidays, or last trading day."""
        from hrp.data.db import get_db
        from hrp.utils.calendar import get_trading_days

        db = get_db(test_db)

        # Use a range that includes a weekend and a known NYSE holiday
        # January 15, 2024 (Martin Luther King Jr. Day) is a NYSE holiday
        # January 13-14, 2024 is Saturday-Sunday
        with db.connection() as conn:
            # Insert symbol first (required for foreign key)
            conn.execute(
                """
                INSERT INTO symbols (symbol, exchange, name)
                VALUES ('COMPLETE', 'NYSE', 'Complete Test Corp')
                """
            )

            # Insert prices for all weekdays EXCEPT the holiday and last trading day
            # Trading days in range: Jan 8-12, Jan 16-19 (holiday on Jan 15, weekend Jan 13-14)
            trading_days_to_insert = [
                date(2024, 1, 8),   # Monday
                date(2024, 1, 9),   # Tuesday
                date(2024, 1, 10),  # Wednesday
                date(2024, 1, 11),  # Thursday
                date(2024, 1, 12),  # Friday
                # Jan 13-14: weekend (skipped)
                # Jan 15: MLK Day holiday (skipped)
                date(2024, 1, 16),  # Tuesday
                date(2024, 1, 17),  # Wednesday
                date(2024, 1, 18),  # Thursday
                # Jan 19: Friday - this is the last trading day, exclude from expected
            ]

            for dt in trading_days_to_insert:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, source)
                    VALUES ('COMPLETE', ?, 100.00, 'test')
                    """,
                    (dt,),
                )

        # Get expected NYSE trading days (excluding last one)
        expected_trading_days = get_trading_days(date(2024, 1, 8), date(2024, 1, 19))
        # Exclude the last trading day from expectations
        if len(expected_trading_days) > 0:
            expected_trading_days = expected_trading_days[:-1]

        check = GapDetectionCheck(test_db, lookback_days=12, read_only=False)
        result = check.run(date(2024, 1, 19))

        # Should not flag gaps for COMPLETE symbol
        complete_issues = [i for i in result.issues if i.symbol == "COMPLETE"]
        assert len(complete_issues) == 0, (
            f"Should not flag gaps when all expected trading days have data. "
            f"Expected {len(expected_trading_days)} trading days, got issues: {complete_issues}"
        )

    def test_flags_missing_trading_day(self, test_db):
        """Should flag a symbol with significant gaps in actual trading day data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Use a date range with known trading days
        with db.connection() as conn:
            # Insert symbol first (required for foreign key)
            conn.execute(
                """
                INSERT INTO symbols (symbol, exchange, name)
                VALUES ('MISSING', 'NYSE', 'Missing Test Corp')
                """
            )

            # Insert for only 3 out of ~6 expected trading days
            # This should be flagged since 3/6 = 50% < 80% threshold
            trading_days = [
                date(2024, 1, 8),   # Monday
                date(2024, 1, 9),   # Tuesday
                # Skip Wednesday Jan 10
                # Skip Thursday Jan 11
                # Skip Friday Jan 12
                # Jan 13-14: weekend
                # Jan 15: MLK Day holiday
                date(2024, 1, 16),  # Tuesday
            ]

            for dt in trading_days:
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, close, source)
                    VALUES ('MISSING', ?, 100.00, 'test')
                    """,
                    (dt,),
                )

        check = GapDetectionCheck(test_db, lookback_days=10, read_only=False)
        result = check.run(date(2024, 1, 17))

        # Should flag the missing trading days (only has 2 out of ~6 expected)
        missing_issues = [i for i in result.issues if i.symbol == "MISSING"]
        assert len(missing_issues) > 0, "Should flag missing trading days"


class TestStaleDataCheck:
    """Tests for stale data detection."""

    def test_detects_stale_symbols(self, test_db):
        """Should flag symbols with old data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            # Add to universe
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES ('STALE', '2024-01-01', TRUE, 'Technology')
                """
            )
            # Add old price data
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES ('STALE', '2024-01-01', 100.00, 'test')
                """
            )

        check = StaleDataCheck(test_db, stale_threshold_days=3, read_only=False)
        result = check.run(date(2024, 1, 15))

        assert not result.passed
        stale_issues = [i for i in result.issues if i.symbol == "STALE"]
        assert len(stale_issues) == 1


class TestVolumeAnomalyCheck:
    """Tests for volume anomaly detection."""

    def test_detects_zero_volume(self, test_db):
        """Should flag zero volume days."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume, source)
                VALUES ('ZERO', '2024-01-15', 100.00, 0, 'test')
                """
            )

        check = VolumeAnomalyCheck(test_db, read_only=False)
        result = check.run(date(2024, 1, 15))

        zero_issues = [i for i in result.issues if i.symbol == "ZERO"]
        assert len(zero_issues) == 1


class TestQualityReport:
    """Tests for quality report generation."""

    def test_health_score_calculation(self):
        """Health score should decrease with issues."""
        from datetime import datetime

        # Perfect report - no issues
        perfect = QualityReport(
            report_date=date.today(),
            generated_at=datetime.now(),
            results=[
                CheckResult(check_name="check1", issues=[]),
                CheckResult(check_name="check2", issues=[]),
                CheckResult(check_name="check3", issues=[]),
                CheckResult(check_name="check4", issues=[]),
                CheckResult(check_name="check5", issues=[]),
            ],
        )
        assert perfect.health_score == 100.0
        assert perfect.passed
        assert perfect.checks_run == 5
        assert perfect.checks_passed == 5

        # Report with critical issues
        critical = QualityReport(
            report_date=date.today(),
            generated_at=datetime.now(),
            results=[
                CheckResult(check_name="check1", issues=[]),
                CheckResult(check_name="check2", issues=[]),
                CheckResult(
                    check_name="check3",
                    issues=[
                        QualityIssue(
                            check_name="check3",
                            severity=IssueSeverity.CRITICAL,
                            symbol="A",
                            date=date.today(),
                            description="Critical 1",
                        ),
                        QualityIssue(
                            check_name="check3",
                            severity=IssueSeverity.CRITICAL,
                            symbol="B",
                            date=date.today(),
                            description="Critical 2",
                        ),
                    ],
                ),
                CheckResult(
                    check_name="check4",
                    issues=[
                        QualityIssue(
                            check_name="check4",
                            severity=IssueSeverity.WARNING,
                            symbol="C",
                            date=date.today(),
                            description="Warning 1",
                        ),
                        QualityIssue(
                            check_name="check4",
                            severity=IssueSeverity.WARNING,
                            symbol="D",
                            date=date.today(),
                            description="Warning 2",
                        ),
                        QualityIssue(
                            check_name="check4",
                            severity=IssueSeverity.WARNING,
                            symbol="E",
                            date=date.today(),
                            description="Warning 3",
                        ),
                    ],
                ),
                CheckResult(check_name="check5", issues=[]),
            ],
        )
        assert critical.health_score < 100.0
        assert not critical.passed  # Has critical issues
        assert critical.critical_issues == 2
        assert critical.warning_issues == 3

        # Report with only warnings
        warnings_only = QualityReport(
            report_date=date.today(),
            generated_at=datetime.now(),
            results=[
                CheckResult(check_name="check1", issues=[]),
                CheckResult(check_name="check2", issues=[]),
                CheckResult(check_name="check3", issues=[]),
                CheckResult(check_name="check4", issues=[]),
                CheckResult(
                    check_name="check5",
                    issues=[
                        QualityIssue(
                            check_name="check5",
                            severity=IssueSeverity.WARNING,
                            symbol="A",
                            date=date.today(),
                            description="Warning 1",
                        ),
                        QualityIssue(
                            check_name="check5",
                            severity=IssueSeverity.WARNING,
                            symbol="B",
                            date=date.today(),
                            description="Warning 2",
                        ),
                        QualityIssue(
                            check_name="check5",
                            severity=IssueSeverity.WARNING,
                            symbol="C",
                            date=date.today(),
                            description="Warning 3",
                        ),
                    ],
                ),
            ],
        )
        assert warnings_only.passed  # No critical issues
        assert warnings_only.health_score > critical.health_score

    def test_report_to_dict(self):
        """Report should serialize to dictionary."""
        from datetime import datetime

        report = QualityReport(
            report_date=date(2024, 1, 15),
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
            results=[
                CheckResult(check_name="check1", issues=[]),
                CheckResult(check_name="check2", issues=[]),
                CheckResult(
                    check_name="check3",
                    issues=[
                        QualityIssue(
                            check_name="check3",
                            severity=IssueSeverity.WARNING,
                            symbol="A",
                            date=date(2024, 1, 15),
                            description="Warning",
                        ),
                    ],
                ),
            ],
        )

        d = report.to_dict()
        assert d["report_date"] == "2024-01-15"
        assert d["checks_run"] == 3
        assert d["health_score"] > 0

    def test_summary_text_generation(self):
        """Report should generate readable summary."""
        from datetime import datetime

        report = QualityReport(
            report_date=date(2024, 1, 15),
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
            results=[
                CheckResult(check_name="check1", issues=[]),
                CheckResult(check_name="check2", issues=[]),
                CheckResult(check_name="check3", issues=[]),
            ],
        )

        summary = report.get_summary_text()
        assert "2024-01-15" in summary
        assert "Health Score" in summary
        assert "PASSED" in summary


class TestQualityReportGenerator:
    """Tests for report generator."""

    def test_generate_report(self, test_db):
        """Should generate report with all checks."""
        generator = QualityReportGenerator(test_db, read_only=False)
        report = generator.generate_report(date(2024, 1, 15))

        assert report.checks_run > 0
        assert report.report_date == date(2024, 1, 15)

    def test_store_and_retrieve_report(self, test_db):
        """Should store report and retrieve it."""
        generator = QualityReportGenerator(test_db, read_only=False)
        report = generator.generate_report(date(2024, 1, 15))

        # Store
        report_id = generator.store_report(report)
        assert report_id > 0

        # Retrieve
        history = generator.get_historical_reports(
            date(2024, 1, 1), date(2024, 1, 31)
        )
        assert len(history) >= 1
        assert history[0]["report_date"] == date(2024, 1, 15)


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_critical_count(self):
        """Should count critical issues correctly."""
        # Note: passed is now a computed property, not a constructor arg
        result = CheckResult(
            check_name="test",
            issues=[
                QualityIssue(
                    check_name="test",
                    severity=IssueSeverity.CRITICAL,
                    symbol="A",
                    date=date.today(),
                    description="Critical",
                ),
                QualityIssue(
                    check_name="test",
                    severity=IssueSeverity.WARNING,
                    symbol="B",
                    date=date.today(),
                    description="Warning",
                ),
                QualityIssue(
                    check_name="test",
                    severity=IssueSeverity.CRITICAL,
                    symbol="C",
                    date=date.today(),
                    description="Critical 2",
                ),
            ],
        )

        assert result.critical_count == 2
        assert result.warning_count == 1
        assert not result.passed  # Has critical issues


class TestQualityIssue:
    """Tests for QualityIssue dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        issue = QualityIssue(
            check_name="test_check",
            severity=IssueSeverity.WARNING,
            symbol="AAPL",
            date=date(2024, 1, 15),
            description="Test issue",
            details={"value": 123},
        )

        d = issue.to_dict()
        assert d["check_name"] == "test_check"
        assert d["severity"] == "warning"
        assert d["symbol"] == "AAPL"
        assert d["date"] == "2024-01-15"
        assert d["details"]["value"] == 123


# =============================================================================
# Quality Alert Tests
# =============================================================================


class TestQualityAlertManager:
    """Tests for QualityAlertManager alert sending."""

    def test_send_critical_alert_with_issues(self, test_db):
        """send_critical_alert should send email when issues present."""
        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.return_value = True
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()

            issues = [
                QualityIssue(
                    check_name="price_anomaly",
                    severity=IssueSeverity.CRITICAL,
                    symbol="AAPL",
                    date=date(2024, 1, 15),
                    description="Price dropped 60%",
                ),
                QualityIssue(
                    check_name="price_anomaly",
                    severity=IssueSeverity.CRITICAL,
                    symbol="MSFT",
                    date=date(2024, 1, 15),
                    description="Price spiked 70%",
                ),
            ]

            result = alert_manager.send_critical_alert(issues, date(2024, 1, 15))

            assert result is True
            mock_instance.send_email.assert_called_once()
            call_kwargs = mock_instance.send_email.call_args[1]
            assert "CRITICAL" in call_kwargs["subject"]
            assert "2024-01-15" in call_kwargs["subject"]
            assert "AAPL" in call_kwargs["html_body"]
            assert "MSFT" in call_kwargs["html_body"]

    def test_send_critical_alert_no_issues(self, test_db):
        """send_critical_alert should not send when no issues."""
        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()
            result = alert_manager.send_critical_alert([], date(2024, 1, 15))

            assert result is False
            mock_instance.send_email.assert_not_called()

    def test_send_critical_alert_email_failure(self, test_db):
        """send_critical_alert should return False on email error."""
        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.side_effect = Exception("SMTP error")
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()

            issues = [
                QualityIssue(
                    check_name="test",
                    severity=IssueSeverity.CRITICAL,
                    symbol="TEST",
                    date=date(2024, 1, 15),
                    description="Test",
                )
            ]

            result = alert_manager.send_critical_alert(issues, date(2024, 1, 15))
            assert result is False

    def test_send_daily_summary(self, test_db):
        """send_daily_summary should send formatted email."""
        from datetime import datetime

        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.return_value = True
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()

            # Create a report with results (summary stats are computed from results)
            report = QualityReport(
                report_date=date(2024, 1, 15),
                generated_at=datetime(2024, 1, 15, 12, 0, 0),
                results=[
                    CheckResult(
                        check_name="price_anomaly",
                        issues=[],
                        run_time_ms=100.0,
                    ),
                    CheckResult(
                        check_name="completeness",
                        issues=[
                            QualityIssue(
                                check_name="completeness",
                                severity=IssueSeverity.WARNING,
                                symbol="TEST",
                                date=date(2024, 1, 15),
                                description="Missing",
                            )
                        ],
                        run_time_ms=50.0,
                    ),
                ],
            )

            result = alert_manager.send_daily_summary(report)

            assert result is True
            mock_instance.send_email.assert_called_once()
            call_kwargs = mock_instance.send_email.call_args[1]
            assert "Daily Data Quality Report" in call_kwargs["subject"]
            assert "2024-01-15" in call_kwargs["subject"]
            assert "Health Score" in call_kwargs["html_body"]

    def test_send_daily_summary_email_failure(self, test_db):
        """send_daily_summary should return False on email error."""
        from datetime import datetime

        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.side_effect = Exception("SMTP error")
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()

            # Create report with empty results list (summary stats computed from results)
            report = QualityReport(
                report_date=date(2024, 1, 15),
                generated_at=datetime(2024, 1, 15, 12, 0, 0),
                results=[],
            )

            result = alert_manager.send_daily_summary(report)
            assert result is False

    def test_process_report_with_critical_issues(self, test_db):
        """process_report should send critical alert when critical issues exist."""
        from datetime import datetime

        from hrp.data.quality.alerts import QualityAlertManager

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.return_value = True
            mock_notifier.return_value = mock_instance

            alert_manager = QualityAlertManager()

            # Create report with critical issues (summary stats computed from results)
            report = QualityReport(
                report_date=date(2024, 1, 15),
                generated_at=datetime(2024, 1, 15, 12, 0, 0),
                results=[
                    CheckResult(
                        check_name="price_anomaly",
                        issues=[
                            QualityIssue(
                                check_name="price_anomaly",
                                severity=IssueSeverity.CRITICAL,
                                symbol="TEST",
                                date=date(2024, 1, 15),
                                description="Critical issue",
                            ),
                            QualityIssue(
                                check_name="price_anomaly",
                                severity=IssueSeverity.CRITICAL,
                                symbol="TEST2",
                                date=date(2024, 1, 15),
                                description="Critical issue 2",
                            ),
                        ],
                    ),
                    CheckResult(check_name="check2", issues=[]),
                    CheckResult(check_name="check3", issues=[]),
                    CheckResult(check_name="check4", issues=[]),
                    CheckResult(check_name="check5", issues=[]),
                ],
            )

            result = alert_manager.process_report(report, send_summary=True)

            assert result["critical_alert_sent"] is True
            assert result["summary_sent"] is True
            # Two calls: critical alert + daily summary
            assert mock_instance.send_email.call_count == 2


class TestRunQualityCheckWithAlerts:
    """Tests for run_quality_check_with_alerts convenience function."""

    def test_run_with_store_report(self, test_db):
        """run_quality_check_with_alerts should store report when requested."""
        from hrp.data.quality.alerts import run_quality_check_with_alerts

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_notifier.return_value = mock_instance

            result = run_quality_check_with_alerts(
                db_path=test_db,
                as_of_date=date(2024, 1, 15),
                send_summary=False,
                store_report=True,
            )

            assert "report_date" in result
            assert "report_id" in result
            assert result["report_id"] is not None
            assert "health_score" in result
            assert "passed" in result

    def test_run_without_store_report(self, test_db):
        """run_quality_check_with_alerts should not store when store_report=False."""
        from hrp.data.quality.alerts import run_quality_check_with_alerts

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_notifier.return_value = mock_instance

            result = run_quality_check_with_alerts(
                db_path=test_db,
                as_of_date=date(2024, 1, 15),
                send_summary=False,
                store_report=False,
            )

            assert result["report_id"] is None

    def test_run_sends_summary(self, test_db):
        """run_quality_check_with_alerts should send summary when requested."""
        from hrp.data.quality.alerts import run_quality_check_with_alerts

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_instance.send_email.return_value = True
            mock_notifier.return_value = mock_instance

            result = run_quality_check_with_alerts(
                db_path=test_db,
                as_of_date=date(2024, 1, 15),
                send_summary=True,
                store_report=False,
            )

            assert result["summary_sent"] is True
            mock_instance.send_email.assert_called()

    def test_run_returns_issue_counts(self, test_db):
        """run_quality_check_with_alerts should return issue counts."""
        from hrp.data.quality.alerts import run_quality_check_with_alerts

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_notifier.return_value = mock_instance

            result = run_quality_check_with_alerts(
                db_path=test_db,
                as_of_date=date(2024, 1, 15),
                send_summary=False,
                store_report=False,
            )

            assert "total_issues" in result
            assert "critical_issues" in result
            assert "warning_issues" in result
            assert isinstance(result["total_issues"], int)
            assert isinstance(result["critical_issues"], int)
            assert isinstance(result["warning_issues"], int)
