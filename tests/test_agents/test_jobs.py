"""
Tests for ingestion job classes.

Tests cover:
- IngestionJob base class (abstract)
- PriceIngestionJob execution and logging
- FeatureComputationJob execution and logging
- Dependency checking
- Retry logic
- Failure notifications
"""

import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.jobs import (
    FeatureComputationJob,
    IngestionJob,
    JobStatus,
    PriceIngestionJob,
)
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def job_test_db():
    """Create a temporary database with schema for job tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    # Insert data_sources entries for job IDs (required for FK constraint)
    from hrp.data.db import get_db

    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('price_ingestion', 'scheduled_job', 'active'),
                ('feature_computation', 'scheduled_job', 'active'),
                ('test_job', 'scheduled_job', 'active'),
                ('dep_job', 'scheduled_job', 'active')
            """
        )

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# Test Classes
# =============================================================================


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """JobStatus should have expected values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.SUCCESS.value == "success"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.RETRYING.value == "retrying"


class TestPriceIngestionJob:
    """Tests for PriceIngestionJob."""

    def test_init_defaults(self):
        """PriceIngestionJob should initialize with defaults."""
        job = PriceIngestionJob()
        assert job.job_id == "price_ingestion"
        assert job.max_retries == 3
        assert job.status == JobStatus.PENDING
        assert job.dependencies == []

    def test_init_with_symbols(self):
        """PriceIngestionJob should accept custom symbols."""
        symbols = ["AAPL", "MSFT"]
        job = PriceIngestionJob(symbols=symbols)
        assert job.symbols == symbols

    def test_init_with_dates(self):
        """PriceIngestionJob should accept custom dates."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        job = PriceIngestionJob(start=start, end=end)
        assert job.start == start
        assert job.end == end

    @patch("hrp.agents.jobs.ingest_prices")
    def test_execute_calls_ingest_prices(self, mock_ingest, job_test_db):
        """execute() should call ingest_prices with correct args."""
        mock_ingest.return_value = {
            "rows_fetched": 100,
            "rows_inserted": 95,
            "symbols_success": 5,
            "symbols_failed": 0,
            "failed_symbols": [],
        }

        symbols = ["AAPL", "MSFT"]
        job = PriceIngestionJob(symbols=symbols, source="yfinance")
        result = job.execute()

        mock_ingest.assert_called_once()
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["symbols"] == symbols
        assert call_kwargs["source"] == "yfinance"

        assert result["records_fetched"] == 100
        assert result["records_inserted"] == 95

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_success(self, mock_notifier, mock_ingest, job_test_db):
        """run() should log success to ingestion_log."""
        mock_ingest.return_value = {
            "rows_fetched": 50,
            "rows_inserted": 50,
            "symbols_success": 2,
            "symbols_failed": 0,
            "failed_symbols": [],
        }

        job = PriceIngestionJob(symbols=["AAPL"])
        result = job.run()

        assert result["records_inserted"] == 50
        assert job.status == JobStatus.SUCCESS

        # Verify logged to ingestion_log
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, records_inserted
                FROM ingestion_log
                WHERE source_id = 'price_ingestion'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "completed"
        assert log_row[1] == 50

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_failure(self, mock_notifier, mock_ingest, job_test_db):
        """run() should log failure to ingestion_log."""
        mock_ingest.side_effect = Exception("API error")
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = PriceIngestionJob(symbols=["AAPL"], max_retries=0)
        result = job.run()

        assert result["status"] == "failed"
        assert "API error" in result["error"]
        assert job.status == JobStatus.FAILED

        # Verify logged to ingestion_log
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, error_message
                FROM ingestion_log
                WHERE source_id = 'price_ingestion'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "failed"
        assert "API error" in log_row[1]

    def test_get_status(self):
        """get_status() should return job status dict."""
        job = PriceIngestionJob()
        status = job.get_status()

        assert status["job_id"] == "price_ingestion"
        assert status["status"] == "pending"
        assert status["retry_count"] == 0
        assert status["dependencies"] == []

    def test_repr(self):
        """Job repr should show id and status."""
        job = PriceIngestionJob()
        repr_str = repr(job)
        assert "PriceIngestionJob" in repr_str
        assert "price_ingestion" in repr_str
        assert "pending" in repr_str


class TestFeatureComputationJob:
    """Tests for FeatureComputationJob."""

    def test_init_defaults(self):
        """FeatureComputationJob should initialize with defaults."""
        job = FeatureComputationJob()
        assert job.job_id == "feature_computation"
        assert job.dependencies == ["price_ingestion"]  # Default dependency

    def test_init_with_custom_dependencies(self):
        """FeatureComputationJob should accept custom dependencies."""
        job = FeatureComputationJob(dependencies=["custom_job"])
        assert job.dependencies == ["custom_job"]

    @patch("hrp.agents.jobs.compute_features")
    def test_execute_calls_compute_features(self, mock_compute, job_test_db):
        """execute() should call compute_features with correct args."""
        mock_compute.return_value = {
            "features_computed": 500,
            "rows_inserted": 500,
            "symbols_success": 10,
            "symbols_failed": 0,
            "failed_symbols": [],
        }

        job = FeatureComputationJob(symbols=None, version="v2")
        result = job.execute()

        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args[1]
        assert call_kwargs["version"] == "v2"

        assert result["records_fetched"] == 500  # features_computed
        assert result["records_inserted"] == 500

    @patch("hrp.agents.jobs.compute_features")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_checks_dependency(self, mock_notifier, mock_compute, job_test_db):
        """run() should check dependencies before executing."""
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        # No successful price_ingestion in log = dependency not met
        job = FeatureComputationJob()
        result = job.run()

        assert result["status"] == "failed"
        assert "Dependencies not met" in result["error"]
        mock_compute.assert_not_called()

    @patch("hrp.agents.jobs.compute_features")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_with_met_dependency(self, mock_notifier, mock_compute, job_test_db):
        """run() should execute when dependencies are met."""
        mock_compute.return_value = {
            "features_computed": 100,
            "rows_inserted": 100,
            "symbols_success": 5,
            "symbols_failed": 0,
        }

        # Insert successful price_ingestion record
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (source_id, status, completed_at)
                VALUES ('price_ingestion', 'completed', CURRENT_TIMESTAMP)
                """
            )

        job = FeatureComputationJob()
        result = job.run()

        assert job.status == JobStatus.SUCCESS
        mock_compute.assert_called_once()


class TestDependencyChecking:
    """Tests for job dependency checking."""

    def test_no_dependencies_passes(self, job_test_db):
        """Job with no dependencies should pass check."""
        job = PriceIngestionJob()  # No dependencies
        assert job._check_dependencies() is True

    def test_dependency_never_run_fails(self, job_test_db):
        """Dependency that never ran should fail check."""

        class TestJob(IngestionJob):
            def execute(self):
                return {}

        job = TestJob(job_id="test_job", dependencies=["dep_job"])
        assert job._check_dependencies() is False

    def test_dependency_failed_fails(self, job_test_db):
        """Dependency with failed status should fail check."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (source_id, status, completed_at)
                VALUES ('dep_job', 'failed', CURRENT_TIMESTAMP)
                """
            )

        class TestJob(IngestionJob):
            def execute(self):
                return {}

        job = TestJob(job_id="test_job", dependencies=["dep_job"])
        assert job._check_dependencies() is False

    def test_dependency_success_passes(self, job_test_db):
        """Dependency with success status should pass check."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (source_id, status, completed_at)
                VALUES ('dep_job', 'completed', CURRENT_TIMESTAMP)
                """
            )

        class TestJob(IngestionJob):
            def execute(self):
                return {}

        job = TestJob(job_id="test_job", dependencies=["dep_job"])
        assert job._check_dependencies() is True


class TestRetryLogic:
    """Tests for job retry behavior."""

    @patch("hrp.agents.jobs.time.sleep")
    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_retry_on_failure(self, mock_notifier, mock_ingest, mock_sleep, job_test_db):
        """Job should retry on failure up to max_retries."""
        call_count = [0]

        def fail_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary error")
            return {
                "rows_fetched": 10,
                "rows_inserted": 10,
                "symbols_success": 1,
                "symbols_failed": 0,
            }

        mock_ingest.side_effect = fail_then_succeed

        job = PriceIngestionJob(symbols=["AAPL"], max_retries=3, retry_backoff=1.0)
        result = job.run()

        assert result["records_inserted"] == 10
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2  # Slept before retries 2 and 3

    @patch("hrp.agents.jobs.time.sleep")
    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_max_retries_exceeded(self, mock_notifier, mock_ingest, mock_sleep, job_test_db):
        """Job should fail after exceeding max_retries."""
        mock_ingest.side_effect = Exception("Persistent error")
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = PriceIngestionJob(symbols=["AAPL"], max_retries=2, retry_backoff=1.0)
        result = job.run()

        assert result["status"] == "failed"
        assert result["retry_count"] == 2
        # Called 3 times total: initial + 2 retries
        assert mock_ingest.call_count == 3


class TestFailureNotification:
    """Tests for failure notification sending."""

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_sends_failure_notification(self, mock_notifier, mock_ingest, job_test_db):
        """Job should send email notification on failure."""
        mock_ingest.side_effect = Exception("Critical error")
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = PriceIngestionJob(symbols=["AAPL"], max_retries=0)
        job.run()

        mock_notifier_instance.send_failure_notification.assert_called_once()
        call_kwargs = mock_notifier_instance.send_failure_notification.call_args[1]
        assert call_kwargs["job_name"] == "price_ingestion"
        assert "Critical error" in call_kwargs["error_message"]

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_notification_failure_does_not_break_job(
        self, mock_notifier, mock_ingest, job_test_db
    ):
        """Notification failure should not affect job status."""
        mock_ingest.side_effect = Exception("Job error")
        mock_notifier_instance = MagicMock()
        mock_notifier_instance.send_failure_notification.side_effect = Exception(
            "Email error"
        )
        mock_notifier.return_value = mock_notifier_instance

        job = PriceIngestionJob(symbols=["AAPL"], max_retries=0)
        result = job.run()

        # Job should still report failure correctly
        assert result["status"] == "failed"
        assert "Job error" in result["error"]


class TestJobLogging:
    """Tests for ingestion_log table updates."""

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_log_start_creates_entry(self, mock_notifier, mock_ingest, job_test_db):
        """_log_start should create ingestion_log entry."""
        mock_ingest.return_value = {
            "rows_fetched": 10,
            "rows_inserted": 10,
            "symbols_success": 1,
            "symbols_failed": 0,
        }

        job = PriceIngestionJob(symbols=["AAPL"])
        job.run()

        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM ingestion_log WHERE source_id = 'price_ingestion'"
            ).fetchone()[0]

        assert count >= 1

    @patch("hrp.agents.jobs.ingest_prices")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_get_last_successful_run(self, mock_notifier, mock_ingest, job_test_db):
        """get_last_successful_run should return last success timestamp."""
        mock_ingest.return_value = {
            "rows_fetched": 10,
            "rows_inserted": 10,
            "symbols_success": 1,
            "symbols_failed": 0,
        }

        job = PriceIngestionJob(symbols=["AAPL"])
        job.run()

        last_run = job.get_last_successful_run()
        assert last_run is not None

    def test_get_last_successful_run_no_history(self, job_test_db):
        """get_last_successful_run should return None if no history."""
        job = PriceIngestionJob()
        last_run = job.get_last_successful_run()
        assert last_run is None
