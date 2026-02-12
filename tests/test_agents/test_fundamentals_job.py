"""
Tests for FundamentalsIngestionJob.

Tests cover:
- Job initialization with defaults and custom params
- execute() calls ingest_fundamentals correctly
- run() logs success/failure to ingestion_log
- Failure sends email notification
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.jobs import JobStatus
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


@pytest.fixture
def fundamentals_job_test_db():
    """Create a temporary database with schema for fundamentals job tests."""
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
                ('fundamentals_ingestion', 'scheduled_job', 'active'),
                ('yfinance', 'api', 'active'),
                ('simfin', 'api', 'active')
            ON CONFLICT DO NOTHING
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


class TestFundamentalsIngestionJobInit:
    """Tests for FundamentalsIngestionJob initialization."""

    def test_init_defaults(self):
        """FundamentalsIngestionJob should initialize with defaults."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        job = FundamentalsIngestionJob()
        assert job.job_id == "fundamentals_ingestion"
        assert job.max_retries == 3
        assert job.status == JobStatus.PENDING
        assert job.dependencies == []
        assert job.source == "simfin"
        assert job.symbols is None  # Will fetch from universe
        assert job.metrics is None  # Will use defaults

    def test_init_with_symbols(self):
        """FundamentalsIngestionJob should accept custom symbols."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        symbols = ["AAPL", "MSFT"]
        job = FundamentalsIngestionJob(symbols=symbols)
        assert job.symbols == symbols

    def test_init_with_metrics(self):
        """FundamentalsIngestionJob should accept custom metrics."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        metrics = ["revenue", "eps"]
        job = FundamentalsIngestionJob(metrics=metrics)
        assert job.metrics == metrics

    def test_init_with_source(self):
        """FundamentalsIngestionJob should accept custom source."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        job = FundamentalsIngestionJob(source="yfinance")
        assert job.source == "yfinance"

    def test_init_with_dates(self):
        """FundamentalsIngestionJob should accept custom dates."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        job = FundamentalsIngestionJob(start_date=start, end_date=end)
        assert job.start_date == start
        assert job.end_date == end


class TestFundamentalsIngestionJobExecute:
    """Tests for FundamentalsIngestionJob.execute()."""

    @patch("hrp.agents.jobs.ingest_fundamentals")
    def test_execute_calls_ingest_fundamentals(self, mock_ingest, fundamentals_job_test_db):
        """execute() should call ingest_fundamentals with correct args."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        mock_ingest.return_value = {
            "records_fetched": 100,
            "records_inserted": 95,
            "symbols_success": 5,
            "symbols_failed": 0,
            "failed_symbols": [],
            "fallback_used": 0,
            "pit_violations_filtered": 2,
        }

        symbols = ["AAPL", "MSFT"]
        metrics = ["revenue", "eps"]
        job = FundamentalsIngestionJob(
            symbols=symbols,
            metrics=metrics,
            source="yfinance",
        )
        result = job.execute()

        mock_ingest.assert_called_once()
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["symbols"] == symbols
        assert call_kwargs["metrics"] == metrics
        assert call_kwargs["source"] == "yfinance"

        assert result["records_fetched"] == 100
        assert result["records_inserted"] == 95

    @patch("hrp.agents.jobs.ingest_fundamentals")
    @patch("hrp.agents.jobs.UniverseManager")
    def test_execute_gets_universe_symbols(self, mock_manager_class, mock_ingest, fundamentals_job_test_db):
        """execute() should get symbols from universe if not specified."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        mock_manager = MagicMock()
        mock_manager.get_universe_at_date.return_value = ["AAPL", "MSFT", "GOOGL"]
        mock_manager_class.return_value = mock_manager

        mock_ingest.return_value = {
            "records_fetched": 50,
            "records_inserted": 50,
            "symbols_success": 3,
            "symbols_failed": 0,
            "failed_symbols": [],
            "fallback_used": 0,
            "pit_violations_filtered": 0,
        }

        job = FundamentalsIngestionJob(symbols=None)  # Use universe
        result = job.execute()

        mock_manager.get_universe_at_date.assert_called_once()
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["symbols"] == ["AAPL", "MSFT", "GOOGL"]


class TestFundamentalsIngestionJobRun:
    """Tests for FundamentalsIngestionJob.run()."""

    @patch("hrp.agents.jobs.ingest_fundamentals")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_success(self, mock_notifier, mock_ingest, fundamentals_job_test_db):
        """run() should log success to ingestion_log."""
        from hrp.agents.jobs import FundamentalsIngestionJob
        from hrp.data.db import get_db

        mock_ingest.return_value = {
            "records_fetched": 50,
            "records_inserted": 50,
            "symbols_success": 2,
            "symbols_failed": 0,
            "failed_symbols": [],
            "fallback_used": 0,
            "pit_violations_filtered": 0,
        }

        job = FundamentalsIngestionJob(symbols=["AAPL"])
        result = job.run()

        assert result["records_inserted"] == 50
        assert job.status == JobStatus.SUCCESS

        # Verify logged to ingestion_log
        db = get_db(fundamentals_job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, records_inserted
                FROM ingestion_log
                WHERE source_id = 'fundamentals_ingestion'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "completed"
        assert log_row[1] == 50

    @patch("hrp.agents.jobs.ingest_fundamentals")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_failure(self, mock_notifier, mock_ingest, fundamentals_job_test_db):
        """run() should log failure to ingestion_log."""
        from hrp.agents.jobs import FundamentalsIngestionJob
        from hrp.data.db import get_db

        mock_ingest.side_effect = Exception("API error")
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = FundamentalsIngestionJob(symbols=["AAPL"], max_retries=0)
        result = job.run()

        assert result["status"] == "failed"
        assert "API error" in result["error"]
        assert job.status == JobStatus.FAILED

        # Verify logged to ingestion_log
        db = get_db(fundamentals_job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, error_message
                FROM ingestion_log
                WHERE source_id = 'fundamentals_ingestion'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "failed"
        assert "API error" in log_row[1]


class TestFundamentalsIngestionJobNotification:
    """Tests for failure notification sending."""

    @patch("hrp.agents.jobs.ingest_fundamentals")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_sends_failure_notification(self, mock_notifier, mock_ingest, fundamentals_job_test_db):
        """Job should send email notification on failure."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        mock_ingest.side_effect = Exception("Critical error")
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = FundamentalsIngestionJob(symbols=["AAPL"], max_retries=0)
        job.run()

        mock_notifier_instance.send_failure_notification.assert_called_once()
        call_kwargs = mock_notifier_instance.send_failure_notification.call_args[1]
        assert call_kwargs["job_name"] == "fundamentals_ingestion"
        assert "Critical error" in call_kwargs["error_message"]

    @patch("hrp.agents.jobs.ingest_fundamentals")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_notification_failure_does_not_break_job(
        self, mock_notifier, mock_ingest, fundamentals_job_test_db
    ):
        """Notification failure should not affect job status."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        mock_ingest.side_effect = Exception("Job error")
        mock_notifier_instance = MagicMock()
        mock_notifier_instance.send_failure_notification.side_effect = Exception(
            "Email error"
        )
        mock_notifier.return_value = mock_notifier_instance

        job = FundamentalsIngestionJob(symbols=["AAPL"], max_retries=0)
        result = job.run()

        # Job should still report failure correctly
        assert result["status"] == "failed"
        assert "Job error" in result["error"]


class TestFundamentalsIngestionJobHelpers:
    """Tests for helper methods."""

    def test_get_status(self, fundamentals_job_test_db):
        """get_status() should return job status dict."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        job = FundamentalsIngestionJob()
        status = job.get_status()

        assert status["job_id"] == "fundamentals_ingestion"
        assert status["status"] == "pending"
        assert status["retry_count"] == 0
        assert status["dependencies"] == []

    def test_repr(self):
        """Job repr should show id and status."""
        from hrp.agents.jobs import FundamentalsIngestionJob

        job = FundamentalsIngestionJob()
        repr_str = repr(job)
        assert "FundamentalsIngestionJob" in repr_str
        assert "fundamentals_ingestion" in repr_str
        assert "pending" in repr_str
