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
    DataRequirement,
    FeatureComputationJob,
    IngestionJob,
    JobStatus,
    PriceIngestionJob,
    UniverseUpdateJob,
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
                ('universe_update', 'scheduled_job', 'active'),
                ('test_job', 'scheduled_job', 'active'),
                ('dep_job', 'scheduled_job', 'active')
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

    def test_get_status(self, job_test_db):
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
        # Now uses data requirements instead of job dependencies
        assert job.dependencies == []
        assert len(job.data_requirements) == 1
        assert job.data_requirements[0].table == "prices"

    def test_init_with_custom_max_age(self):
        """FeatureComputationJob should accept custom max_price_age_days."""
        job = FeatureComputationJob(max_price_age_days=14)
        assert job.data_requirements[0].max_age_days == 14

    @patch("hrp.data.ingestion.features.compute_features_batch")
    def test_execute_calls_compute_features(self, mock_compute, job_test_db):
        """execute() should call compute_features_batch with correct args."""
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

    @patch("hrp.data.ingestion.features.compute_features_batch")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_checks_data_requirements(self, mock_notifier, mock_compute, job_test_db):
        """run() should check data requirements before executing."""
        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        # No price data = data requirement not met
        job = FeatureComputationJob()
        result = job.run()

        assert result["status"] == "failed"
        assert "Data requirements not met" in result["error"]
        mock_compute.assert_not_called()

    @patch("hrp.data.ingestion.features.compute_features_batch")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_with_met_data_requirements(self, mock_notifier, mock_compute, job_test_db):
        """run() should execute when data requirements are met."""
        from datetime import date, timedelta

        mock_compute.return_value = {
            "features_computed": 100,
            "rows_inserted": 100,
            "symbols_success": 5,
            "symbols_failed": 0,
        }

        # Insert sufficient price data to meet requirements
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbols first (FK constraint)
            for i in range(200):
                conn.execute(
                    f"""
                    INSERT INTO symbols (symbol, name, exchange)
                    VALUES ('SYM{i}', 'Symbol {i}', 'NYSE')
                    """
                )
            # Insert 1000+ rows of price data with RECENT dates (within max_price_age_days)
            # Use 200 symbols × 6 days = 1200 unique rows
            today = date.today()
            for symbol_idx in range(200):
                for day_offset in range(6):
                    # Ensure most recent date (day_offset=0) is today
                    price_date = today - timedelta(days=day_offset)
                    conn.execute(
                        f"""
                        INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                        VALUES ('SYM{symbol_idx}', '{price_date}', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
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
                INSERT INTO ingestion_log (log_id, source_id, status, completed_at)
                VALUES (1, 'dep_job', 'failed', CURRENT_TIMESTAMP)
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
                INSERT INTO ingestion_log (log_id, source_id, status, completed_at)
                VALUES (1, 'dep_job', 'completed', CURRENT_TIMESTAMP)
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
        """Job should retry on transient failures up to max_retries."""
        call_count = [0]

        def fail_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                # Use transient error (ConnectionError) to trigger retry
                raise ConnectionError("Temporary network error")
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
        """Job should fail after exceeding max_retries for transient errors."""
        # Use transient error (ConnectionError) to trigger retries
        mock_ingest.side_effect = ConnectionError("Persistent network error")
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


class TestUniverseUpdateJob:
    """Tests for UniverseUpdateJob."""

    def test_init_defaults(self):
        """UniverseUpdateJob should initialize with defaults."""
        job = UniverseUpdateJob()
        assert job.job_id == "universe_update"
        assert job.max_retries == 3
        assert job.status == JobStatus.PENDING
        assert job.dependencies == []
        assert job.as_of_date == date.today()
        assert job.actor == "system:scheduled_universe_update"

    def test_init_with_custom_date(self):
        """UniverseUpdateJob should accept custom date."""
        test_date = date(2024, 1, 15)
        job = UniverseUpdateJob(as_of_date=test_date)
        assert job.as_of_date == test_date

    def test_init_with_custom_actor(self):
        """UniverseUpdateJob should accept custom actor."""
        job = UniverseUpdateJob(actor="user:manual_test")
        assert job.actor == "user:manual_test"

    @patch("hrp.agents.jobs.UniverseManager")
    def test_execute_calls_universe_manager(self, mock_manager_class, job_test_db):
        """execute() should call UniverseManager.update_universe."""
        mock_manager = MagicMock()
        mock_manager.update_universe.return_value = {
            "date": date.today(),
            "total_constituents": 503,
            "included": 380,
            "excluded": 123,
            "added": 5,
            "removed": 2,
            "exclusion_reasons": {
                "excluded_sector": 80,
                "penny_stock": 43,
            },
        }
        mock_manager_class.return_value = mock_manager

        test_date = date(2024, 1, 15)
        job = UniverseUpdateJob(as_of_date=test_date, actor="test_actor")
        result = job.execute()

        # Verify UniverseManager was called correctly
        mock_manager.update_universe.assert_called_once_with(
            as_of_date=test_date,
            actor="test_actor",
        )

        # Verify result format
        assert result["records_fetched"] == 503  # total_constituents
        assert result["records_inserted"] == 380  # included
        assert result["symbols_added"] == 5
        assert result["symbols_removed"] == 2
        assert result["symbols_excluded"] == 123

    @patch("hrp.agents.jobs.UniverseManager")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_success(self, mock_notifier, mock_manager_class, job_test_db):
        """run() should log success to ingestion_log."""
        mock_manager = MagicMock()
        mock_manager.update_universe.return_value = {
            "date": date.today(),
            "total_constituents": 503,
            "included": 380,
            "excluded": 123,
            "added": 0,
            "removed": 0,
            "exclusion_reasons": {},
        }
        mock_manager_class.return_value = mock_manager

        job = UniverseUpdateJob()
        result = job.run()

        assert result["records_inserted"] == 380
        assert job.status == JobStatus.SUCCESS

        # Verify logged to ingestion_log
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, records_fetched, records_inserted
                FROM ingestion_log
                WHERE source_id = 'universe_update'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "completed"
        assert log_row[1] == 503  # total_constituents
        assert log_row[2] == 380  # included

    @patch("hrp.agents.jobs.UniverseManager")
    @patch("hrp.agents.jobs.EmailNotifier")
    def test_run_logs_failure(self, mock_notifier, mock_manager_class, job_test_db):
        """run() should log failure to ingestion_log."""
        mock_manager = MagicMock()
        mock_manager.update_universe.side_effect = Exception("Wikipedia fetch error")
        mock_manager_class.return_value = mock_manager

        mock_notifier_instance = MagicMock()
        mock_notifier.return_value = mock_notifier_instance

        job = UniverseUpdateJob(max_retries=0)
        result = job.run()

        assert result["status"] == "failed"
        assert "Wikipedia fetch error" in result["error"]
        assert job.status == JobStatus.FAILED

        # Verify logged to ingestion_log
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            log_row = conn.execute(
                """
                SELECT status, error_message
                FROM ingestion_log
                WHERE source_id = 'universe_update'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        assert log_row is not None
        assert log_row[0] == "failed"
        assert "Wikipedia fetch error" in log_row[1]


class TestDataRequirement:
    """Tests for DataRequirement class."""

    def test_init_defaults(self):
        """DataRequirement should initialize with defaults."""
        req = DataRequirement(table="prices")
        assert req.table == "prices"
        assert req.min_rows == 1
        assert req.max_age_days is None
        assert req.date_column == "date"
        assert req.description == "prices data"

    def test_init_with_custom_values(self):
        """DataRequirement should accept custom values."""
        req = DataRequirement(
            table="features",
            min_rows=1000,
            max_age_days=7,
            date_column="computed_at",
            description="Feature data",
        )
        assert req.table == "features"
        assert req.min_rows == 1000
        assert req.max_age_days == 7
        assert req.date_column == "computed_at"
        assert req.description == "Feature data"

    def test_check_empty_table_fails(self, job_test_db):
        """Check should fail when table has no data."""
        req = DataRequirement(table="prices", min_rows=1)
        is_met, message = req.check()
        assert is_met is False
        assert "found 0 rows" in message

    def test_check_sufficient_rows_passes(self, job_test_db):
        """Check should pass when table has sufficient rows."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbols first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES
                    ('AAPL', 'Apple Inc.', 'NASDAQ'),
                    ('MSFT', 'Microsoft Corporation', 'NASDAQ')
                """
            )
            # Insert test price data
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                VALUES
                    ('AAPL', '2026-01-20', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5),
                    ('AAPL', '2026-01-21', 100.5, 102.0, 100.0, 101.5, 1100000, 101.5),
                    ('MSFT', '2026-01-20', 200.0, 201.0, 199.0, 200.5, 500000, 200.5)
                """
            )

        req = DataRequirement(table="prices", min_rows=2)
        is_met, message = req.check()
        assert is_met is True
        assert "OK" in message

    def test_check_insufficient_rows_fails(self, job_test_db):
        """Check should fail when table has insufficient rows."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbol first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')
                """
            )
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                VALUES ('AAPL', '2026-01-20', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
                """
            )

        req = DataRequirement(table="prices", min_rows=100)
        is_met, message = req.check()
        assert is_met is False
        assert "found 1 rows, need 100" in message

    def test_check_recent_data_passes(self, job_test_db):
        """Check should pass when data is recent enough."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbol first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')
                """
            )
            # Insert recent data
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                VALUES ('AAPL', CURRENT_DATE - INTERVAL '1 day', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
                """
            )

        req = DataRequirement(table="prices", min_rows=1, max_age_days=7)
        is_met, message = req.check()
        assert is_met is True
        assert "OK" in message

    def test_check_stale_data_fails(self, job_test_db):
        """Check should fail when data is too old."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbol first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')
                """
            )
            # Insert old data
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                VALUES ('AAPL', '2020-01-01', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
                """
            )

        req = DataRequirement(table="prices", min_rows=1, max_age_days=7)
        is_met, message = req.check()
        assert is_met is False
        assert "days old" in message


class TestDataRequirementIntegration:
    """Tests for data requirement integration with jobs."""

    def test_job_with_data_requirements(self, job_test_db):
        """Job should check data requirements before running."""
        from hrp.data.db import get_db

        class TestJob(IngestionJob):
            def execute(self):
                return {"status": "success"}

        # Job with data requirement that won't be met
        job = TestJob(
            job_id="test_job",
            data_requirements=[
                DataRequirement(table="prices", min_rows=1000)
            ],
        )

        result = job.run()
        assert result["status"] == "failed"
        assert "Data requirements not met" in result["error"]

    def test_job_with_met_data_requirements(self, job_test_db):
        """Job should run when data requirements are met."""
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbol first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')
                """
            )
            # Insert enough test data
            for i in range(10):
                conn.execute(
                    f"""
                    INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                    VALUES ('AAPL', '2026-01-{10+i:02d}', 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
                    """
                )

        class TestJob(IngestionJob):
            def execute(self):
                return {"records_fetched": 10, "records_inserted": 10}

        job = TestJob(
            job_id="test_job",
            data_requirements=[
                DataRequirement(table="prices", min_rows=5)
            ],
        )

        result = job.run()
        assert result.get("records_inserted") == 10

    def test_data_requirements_skip_job_dependencies(self, job_test_db):
        """When data_requirements are specified, job dependencies should be skipped."""

        class TestJob(IngestionJob):
            def execute(self):
                return {"status": "success"}

        # Job with both data requirements and job dependencies
        # Data requirements are met, job dependency is not
        from hrp.data.db import get_db

        db = get_db(job_test_db)
        with db.connection() as conn:
            # Insert symbol first (FK constraint)
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange)
                VALUES ('AAPL', 'Apple Inc.', 'NASDAQ')
                """
            )
            conn.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, volume, adj_close)
                VALUES ('AAPL', CURRENT_DATE, 100.0, 101.0, 99.0, 100.5, 1000000, 100.5)
                """
            )

        job = TestJob(
            job_id="test_job",
            dependencies=["nonexistent_job"],  # This would fail
            data_requirements=[
                DataRequirement(table="prices", min_rows=1)  # This passes
            ],
        )

        # Should pass because data_requirements take precedence
        result = job.run()
        assert result.get("status") == "success"

    def test_feature_computation_uses_data_requirements(self, job_test_db):
        """FeatureComputationJob should use data requirements."""
        job = FeatureComputationJob()
        assert len(job.data_requirements) == 1
        assert job.data_requirements[0].table == "prices"
        assert job.dependencies == []  # No job dependencies
