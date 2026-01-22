"""
End-to-end integration test for the data quality framework.

Tests the complete workflow:
1. Create test database with sample data including anomalies
2. Run PlatformAPI.run_quality_checks()
3. Verify quality_metrics table has results
4. Verify anomalies detected correctly
5. Verify report generated successfully
"""

import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

from hrp.api.platform import PlatformAPI
from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables


@pytest.fixture
def integration_db():
    """Create a test database with HRP schema for integration testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any wal/tmp files
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def api_with_anomalies(integration_db):
    """
    Create PlatformAPI with test database containing intentional anomalies.

    Anomalies include:
    - Price spike >50% for AAPL on one day
    - Negative price for MSFT on one day
    - High < Low for GOOGL on one day
    - Date gaps in AMZN data
    - Zero volume for META on trading day
    """
    api = PlatformAPI(db_path=integration_db)
    db = get_db(integration_db)

    # Define test date range
    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 31)

    # Generate normal price data for multiple symbols
    symbols_data = {
        "AAPL": {"base_price": 150.0, "anomaly_date": date(2024, 1, 15), "anomaly_type": "spike"},
        "MSFT": {
            "base_price": 250.0,
            "anomaly_date": date(2024, 1, 20),
            "anomaly_type": "negative",
        },
        "GOOGL": {
            "base_price": 140.0,
            "anomaly_date": date(2024, 1, 12),
            "anomaly_type": "invalid_range",
        },
        "AMZN": {"base_price": 180.0, "anomaly_date": None, "anomaly_type": "gaps"},
        "META": {
            "base_price": 350.0,
            "anomaly_date": date(2024, 1, 18),
            "anomaly_type": "zero_volume",
        },
    }

    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        for symbol, config in symbols_data.items():
            base_price = config["base_price"]
            anomaly_date = config.get("anomaly_date")
            anomaly_type = config.get("anomaly_type")

            # Skip dates for AMZN to create gaps
            if symbol == "AMZN" and anomaly_type == "gaps":
                # Skip Jan 8-10 and Jan 22-24 to create gaps
                if current_date in [
                    date(2024, 1, 8),
                    date(2024, 1, 9),
                    date(2024, 1, 10),
                    date(2024, 1, 22),
                    date(2024, 1, 23),
                    date(2024, 1, 24),
                ]:
                    continue

            # Normal price with small daily variation
            price = base_price * (1 + 0.001 * (current_date - start_date).days)

            # Apply anomaly on specific date
            if current_date == anomaly_date:
                if anomaly_type == "spike":
                    # >50% price spike
                    price = price * 1.6
                elif anomaly_type == "negative":
                    # Negative close price (data error)
                    price = -10.0

            # Calculate OHLC based on close
            if current_date == anomaly_date and anomaly_type == "invalid_range":
                # High < Low (invalid data)
                open_price = price
                high = price * 0.99  # High is lower than close
                low = price * 1.01  # Low is higher than close
                close = price
            elif current_date == anomaly_date and anomaly_type == "negative":
                # Negative price
                open_price = price
                high = price
                low = price
                close = price
            else:
                # Normal OHLC
                open_price = price * 0.995
                high = price * 1.01
                low = price * 0.99
                close = price

            # Volume
            volume = 0 if (current_date == anomaly_date and anomaly_type == "zero_volume") else 1000000

            # Insert price record
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
            """,
                (
                    symbol,
                    current_date,
                    open_price,
                    high,
                    low,
                    close,
                    close,
                    volume,
                ),
            )

        current_date += timedelta(days=1)

    return api


class TestQualityWorkflowE2E:
    """End-to-end integration tests for quality check workflow."""

    def test_complete_quality_workflow(self, api_with_anomalies):
        """
        Test the complete quality check workflow end-to-end.

        Verifies:
        1. Quality checks run successfully
        2. Report has expected structure
        3. Results stored in quality_metrics table
        4. Anomalies detected correctly
        5. Overall status calculated correctly
        """
        # Run quality checks
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(
            check_date=check_date,
            symbols=None,  # Check all symbols
            send_alerts=False,  # Don't send emails in test
        )

        # Verify report structure
        assert "check_date" in report
        assert "overall_status" in report
        assert "checks" in report
        assert "summary" in report
        assert "critical_issues" in report
        assert "metrics_stored" in report

        assert report["check_date"] == str(check_date)
        assert report["overall_status"] in ["pass", "warning", "fail", "error"]

        # Verify all check types present
        assert "completeness" in report["checks"]
        assert "anomalies" in report["checks"]
        assert "gaps" in report["checks"]
        assert "freshness" in report["checks"]

        # Verify summary counts
        summary = report["summary"]
        assert "total_checks" in summary
        assert "passed" in summary
        assert "warnings" in summary
        assert "failed" in summary
        assert summary["total_checks"] == 4  # 4 check types

    def test_anomalies_detected(self, api_with_anomalies):
        """Verify that intentional anomalies are detected correctly."""
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(check_date=check_date)

        # Anomaly check should detect issues
        anomalies_check = report["checks"]["anomalies"]
        assert anomalies_check["status"] in ["warning", "fail"]
        assert anomalies_check["total_anomalies"] > 0

        # Verify specific anomaly types detected
        issues = anomalies_check["issues"]

        # Should detect AAPL spike (>50% change)
        aapl_issues = [i for i in issues if i["symbol"] == "AAPL"]
        assert len(aapl_issues) > 0
        assert any("Extreme price move" in i["issue"] for i in aapl_issues)

        # Should detect MSFT negative price
        msft_issues = [i for i in issues if i["symbol"] == "MSFT"]
        assert len(msft_issues) > 0
        assert any("Negative price" in i["issue"] for i in msft_issues)

        # Should detect GOOGL invalid range (high < low)
        googl_issues = [i for i in issues if i["symbol"] == "GOOGL"]
        assert len(googl_issues) > 0
        assert any("High < Low" in i["issue"] for i in googl_issues)

    def test_gaps_detected(self, api_with_anomalies):
        """Verify that date gaps are detected correctly."""
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(check_date=check_date)

        # Gaps check should detect missing dates in AMZN
        gaps_check = report["checks"]["gaps"]

        # AMZN has intentional gaps
        amzn_gaps = [g for g in gaps_check["issues"] if g["symbol"] == "AMZN"]
        assert len(amzn_gaps) > 0

        # Verify gap details
        for gap in amzn_gaps:
            assert "gap_start" in gap
            assert "gap_end" in gap
            assert "gap_days" in gap
            assert gap["gap_days"] > 0

    def test_metrics_stored_in_database(self, api_with_anomalies):
        """Verify that quality check results are stored in quality_metrics table."""
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(check_date=check_date)

        # Verify metrics were stored
        assert report["metrics_stored"] > 0

        # Query quality_metrics table directly
        db = api_with_anomalies._db
        metrics = db.fetchdf(
            """
            SELECT check_type, check_date, metric_name, metric_value, status
            FROM quality_metrics
            WHERE check_date = ?
            ORDER BY check_type, metric_name
        """,
            (check_date,),
        )

        assert not metrics.empty
        assert len(metrics) > 0

        # Verify all check types stored
        check_types = set(metrics["check_type"].unique())
        assert "completeness" in check_types
        assert "anomalies" in check_types
        assert "gaps" in check_types
        assert "freshness" in check_types

        # Verify status values are valid
        for status in metrics["status"]:
            assert status in ["pass", "warning", "fail", "error"]

    def test_get_quality_metrics(self, api_with_anomalies):
        """Test retrieving historical quality metrics."""
        # Run checks for multiple dates
        dates = [date(2024, 1, 31), date(2024, 1, 30), date(2024, 1, 29)]
        for check_date in dates:
            api_with_anomalies.run_quality_checks(check_date=check_date)

        # Retrieve metrics
        history = api_with_anomalies.get_quality_metrics(
            start_date=date(2024, 1, 29),
            end_date=date(2024, 1, 31),
        )

        assert "metrics" in history
        assert "summary" in history

        # Should have metrics from all 3 dates
        assert history["summary"]["total_records"] > 0

        # Verify date range in summary
        summary = history["summary"]
        assert "date_range" in summary
        assert summary["date_range"]["start"] is not None
        assert summary["date_range"]["end"] is not None

    def test_get_quality_report(self, api_with_anomalies):
        """Test retrieving a specific quality report."""
        check_date = date(2024, 1, 31)

        # Run checks
        original_report = api_with_anomalies.run_quality_checks(check_date=check_date)

        # Retrieve the same report
        retrieved_report = api_with_anomalies.get_quality_report(report_date=check_date)

        assert retrieved_report is not None
        assert retrieved_report["check_date"] == str(check_date)
        assert retrieved_report["overall_status"] == original_report["overall_status"]
        assert "checks" in retrieved_report
        assert "summary" in retrieved_report

    def test_quality_workflow_with_no_data(self, integration_db):
        """Test quality checks with empty database (no price data)."""
        api = PlatformAPI(db_path=integration_db)

        # Run checks on empty database
        report = api.run_quality_checks(check_date=date(2024, 1, 15))

        # Should complete without crashing
        assert "check_date" in report
        assert "overall_status" in report

        # May have warnings or errors due to no data
        assert report["overall_status"] in ["pass", "warning", "fail", "error"]

    def test_quality_workflow_with_filtered_symbols(self, api_with_anomalies):
        """Test quality checks with symbol filtering."""
        check_date = date(2024, 1, 31)

        # Run checks only for AAPL and MSFT
        report = api_with_anomalies.run_quality_checks(
            check_date=check_date,
            symbols=["AAPL", "MSFT"],
        )

        # Should complete successfully
        assert report["overall_status"] in ["pass", "warning", "fail", "error"]

        # Anomalies should only be for requested symbols
        if report["checks"]["anomalies"]["issues"]:
            for issue in report["checks"]["anomalies"]["issues"]:
                assert issue["symbol"] in ["AAPL", "MSFT"]

    def test_critical_issues_identified(self, api_with_anomalies):
        """Verify that critical issues are properly identified."""
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(check_date=check_date)

        # With anomalies present, should have critical issues
        critical_issues = report["critical_issues"]

        # Should be a list
        assert isinstance(critical_issues, list)

        # Given our test data with multiple anomalies, should have issues
        # (unless checks are very lenient, but anomaly detection should flag them)
        if report["overall_status"] == "fail":
            assert len(critical_issues) > 0

    def test_overall_status_calculation(self, api_with_anomalies):
        """Verify overall status is calculated correctly from individual checks."""
        check_date = date(2024, 1, 31)
        report = api_with_anomalies.run_quality_checks(check_date=check_date)

        overall_status = report["overall_status"]
        checks = report["checks"]

        # Extract individual check statuses
        check_statuses = [check["status"] for check in checks.values()]

        # Overall status logic:
        # - If any check is "fail", overall should be "fail"
        # - If any check is "error", overall should be "error"
        # - If any check is "warning", overall should be "warning"
        # - Otherwise, overall should be "pass"

        if "error" in check_statuses:
            assert overall_status == "error"
        elif "fail" in check_statuses:
            assert overall_status == "fail"
        elif "warning" in check_statuses:
            assert overall_status == "warning"
        else:
            assert overall_status == "pass"

    def test_metrics_persistence(self, api_with_anomalies):
        """Verify that metrics persist correctly across API instances."""
        check_date = date(2024, 1, 31)

        # Run checks
        api_with_anomalies.run_quality_checks(check_date=check_date)

        # Create new API instance pointing to same database
        db_path = api_with_anomalies._db._db_path
        new_api = PlatformAPI(db_path=db_path)

        # Retrieve metrics using new instance
        report = new_api.get_quality_report(report_date=check_date)

        assert report is not None
        assert report["check_date"] == str(check_date)
        assert "checks" in report
        assert len(report["checks"]) == 4  # All 4 check types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
