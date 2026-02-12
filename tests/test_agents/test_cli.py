"""
Tests for CLI job management commands.

Tests cover:
- run_job_now for prices and features
- list_scheduled_jobs
- get_job_status
- clear_job_history
"""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.cli import (
    clear_job_history,
    get_job_status,
    list_scheduled_jobs,
    run_job_now,
)
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_test_db():
    """Create a temporary database with schema for CLI tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    # Insert data_sources entries for job IDs
    from hrp.data.db import get_db

    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('price_ingestion', 'scheduled_job', 'active'),
                ('feature_computation', 'scheduled_job', 'active')
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


class TestRunJobNow:
    """Tests for run_job_now CLI function."""

    @patch("hrp.agents.cli.PriceIngestionJob")
    def test_run_prices_job(self, mock_job_class, cli_test_db):
        """run_job_now('prices') should create and run PriceIngestionJob."""
        mock_job = MagicMock()
        mock_job.run.return_value = {
            "records_fetched": 100,
            "records_inserted": 95,
            "symbols_success": 5,
            "symbols_failed": 0,
        }
        mock_job_class.return_value = mock_job

        result = run_job_now("prices")

        mock_job_class.assert_called_once()
        mock_job.run.assert_called_once()
        assert result["records_inserted"] == 95

    @patch("hrp.agents.cli.PriceIngestionJob")
    def test_run_prices_with_symbols(self, mock_job_class, cli_test_db):
        """run_job_now('prices', symbols) should pass symbols to job."""
        mock_job = MagicMock()
        mock_job.run.return_value = {"records_inserted": 50}
        mock_job_class.return_value = mock_job

        symbols = ["AAPL", "MSFT"]
        run_job_now("prices", symbols=symbols)

        call_kwargs = mock_job_class.call_args[1]
        assert call_kwargs["symbols"] == symbols

    @patch("hrp.agents.cli.FeatureComputationJob")
    def test_run_features_job(self, mock_job_class, cli_test_db):
        """run_job_now('features') should create and run FeatureComputationJob."""
        mock_job = MagicMock()
        mock_job.run.return_value = {
            "records_fetched": 500,
            "records_inserted": 500,
            "symbols_success": 10,
            "symbols_failed": 0,
        }
        mock_job_class.return_value = mock_job

        result = run_job_now("features")

        mock_job_class.assert_called_once()
        mock_job.run.assert_called_once()
        assert result["records_inserted"] == 500

    @patch("hrp.agents.cli.FeatureComputationJob")
    def test_run_features_with_symbols(self, mock_job_class, cli_test_db):
        """run_job_now('features', symbols) should pass symbols to job."""
        mock_job = MagicMock()
        mock_job.run.return_value = {"records_inserted": 200}
        mock_job_class.return_value = mock_job

        symbols = ["AAPL", "GOOGL"]
        run_job_now("features", symbols=symbols)

        call_kwargs = mock_job_class.call_args[1]
        assert call_kwargs["symbols"] == symbols

    def test_run_unknown_job_raises(self, cli_test_db):
        """run_job_now with unknown job name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown job"):
            run_job_now("invalid_job_name")

    @patch("hrp.agents.cli.PriceIngestionJob")
    def test_run_job_returns_failure(self, mock_job_class, cli_test_db):
        """run_job_now should return failure result from job."""
        mock_job = MagicMock()
        mock_job.run.return_value = {"status": "failed", "error": "Network error"}
        mock_job_class.return_value = mock_job

        result = run_job_now("prices")

        assert result["status"] == "failed"
        assert result["error"] == "Network error"


class TestListScheduledJobs:
    """Tests for list_scheduled_jobs CLI function."""

    @patch("hrp.agents.cli.IngestionScheduler")
    def test_list_jobs_creates_scheduler(self, mock_scheduler_class):
        """list_scheduled_jobs should create scheduler and setup jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = [
            {"id": "price_ingestion", "name": "Daily Price Ingestion", "next_run": None},
            {"id": "feature_computation", "name": "Daily Feature Computation", "next_run": None},
        ]
        mock_scheduler_class.return_value = mock_scheduler

        result = list_scheduled_jobs()

        mock_scheduler.setup_daily_ingestion.assert_called_once()
        mock_scheduler.list_jobs.assert_called_once()
        assert len(result) == 2

    @patch("hrp.agents.cli.IngestionScheduler")
    def test_list_jobs_returns_info(self, mock_scheduler_class):
        """list_scheduled_jobs should return job info list."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = [
            {
                "id": "price_ingestion",
                "name": "Daily Price Ingestion",
                "next_run": datetime(2024, 1, 15, 18, 0),
                "trigger": "cron[hour='18', minute='0']",
            }
        ]
        mock_scheduler_class.return_value = mock_scheduler

        result = list_scheduled_jobs()

        assert len(result) == 1
        assert result[0]["id"] == "price_ingestion"
        assert "next_run" in result[0]

    @patch("hrp.agents.cli.IngestionScheduler")
    def test_list_jobs_empty(self, mock_scheduler_class):
        """list_scheduled_jobs should return empty list when no jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []
        mock_scheduler_class.return_value = mock_scheduler

        result = list_scheduled_jobs()

        assert result == []

    @patch("hrp.agents.cli.IngestionScheduler")
    def test_list_jobs_handles_setup_error(self, mock_scheduler_class):
        """list_scheduled_jobs should handle setup errors gracefully."""
        mock_scheduler = MagicMock()
        mock_scheduler.setup_daily_ingestion.side_effect = Exception("Setup failed")
        mock_scheduler.list_jobs.return_value = []
        mock_scheduler_class.return_value = mock_scheduler

        # Should not raise
        result = list_scheduled_jobs()
        assert result == []


class TestGetJobStatus:
    """Tests for get_job_status CLI function."""

    def test_get_status_empty(self, cli_test_db):
        """get_job_status should return empty list when no history."""
        result = get_job_status()
        assert result == []

    def test_get_status_with_history(self, cli_test_db):
        """get_job_status should return job history records."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status, records_fetched, records_inserted)
                VALUES (1, 'price_ingestion', 'completed', 100, 95)
                """
            )

        result = get_job_status()

        assert len(result) == 1
        assert result[0]["source_id"] == "price_ingestion"
        assert result[0]["status"] == "completed"
        assert result[0]["records_fetched"] == 100
        assert result[0]["records_inserted"] == 95

    def test_get_status_filter_by_job_id(self, cli_test_db):
        """get_job_status should filter by job_id."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status)
                VALUES
                    (1, 'price_ingestion', 'completed'),
                    (2, 'feature_computation', 'completed'),
                    (3, 'price_ingestion', 'failed')
                """
            )

        result = get_job_status(job_id="price_ingestion")

        assert len(result) == 2
        for record in result:
            assert record["source_id"] == "price_ingestion"

    def test_get_status_limit(self, cli_test_db):
        """get_job_status should respect limit parameter."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            for i in range(10):
                conn.execute(
                    f"""
                    INSERT INTO ingestion_log (log_id, source_id, status)
                    VALUES ({i + 1}, 'price_ingestion', 'completed')
                    """
                )

        result = get_job_status(limit=5)

        assert len(result) == 5

    def test_get_status_includes_error_message(self, cli_test_db):
        """get_job_status should include error_message for failed jobs."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status, error_message)
                VALUES (1, 'price_ingestion', 'failed', 'Connection timeout')
                """
            )

        result = get_job_status()

        assert len(result) == 1
        assert result[0]["error_message"] == "Connection timeout"


class TestClearJobHistory:
    """Tests for clear_job_history CLI function."""

    def test_clear_all_history(self, cli_test_db):
        """clear_job_history should delete all records when no filters."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            for i in range(5):
                conn.execute(
                    f"""
                    INSERT INTO ingestion_log (log_id, source_id, status)
                    VALUES ({i + 1}, 'price_ingestion', 'completed')
                    """
                )

        count = clear_job_history()

        assert count == 5
        result = get_job_status()
        assert len(result) == 0

    def test_clear_by_job_id(self, cli_test_db):
        """clear_job_history should filter by job_id."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status)
                VALUES
                    (1, 'price_ingestion', 'completed'),
                    (2, 'price_ingestion', 'completed'),
                    (3, 'feature_computation', 'completed')
                """
            )

        count = clear_job_history(job_id="price_ingestion")

        assert count == 2
        result = get_job_status()
        assert len(result) == 1
        assert result[0]["source_id"] == "feature_computation"

    def test_clear_by_status(self, cli_test_db):
        """clear_job_history should filter by status."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status)
                VALUES
                    (1, 'price_ingestion', 'completed'),
                    (2, 'price_ingestion', 'failed'),
                    (3, 'price_ingestion', 'failed')
                """
            )

        count = clear_job_history(status="failed")

        assert count == 2
        result = get_job_status()
        assert len(result) == 1
        assert result[0]["status"] == "completed"

    def test_clear_by_before_date(self, cli_test_db):
        """clear_job_history should filter by before date."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            # Insert old record
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, started_at, status)
                VALUES (1, 'price_ingestion', '2024-01-01 00:00:00', 'completed')
                """
            )
            # Insert recent record
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, started_at, status)
                VALUES (2, 'price_ingestion', '2024-06-01 00:00:00', 'completed')
                """
            )

        count = clear_job_history(before=datetime(2024, 3, 1))

        assert count == 1
        result = get_job_status()
        assert len(result) == 1

    def test_clear_combined_filters(self, cli_test_db):
        """clear_job_history should combine multiple filters."""
        from hrp.data.db import get_db

        db = get_db(cli_test_db)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status, started_at)
                VALUES
                    (1, 'price_ingestion', 'failed', '2024-01-01 00:00:00'),
                    (2, 'price_ingestion', 'completed', '2024-01-01 00:00:00'),
                    (3, 'feature_computation', 'failed', '2024-01-01 00:00:00')
                """
            )

        count = clear_job_history(job_id="price_ingestion", status="failed")

        assert count == 1
        result = get_job_status()
        assert len(result) == 2

    def test_clear_empty_history(self, cli_test_db):
        """clear_job_history on empty table should return 0."""
        count = clear_job_history()
        assert count == 0


class TestRunJobNowUniverse:
    """Additional tests for run_job_now with universe job."""

    @patch("hrp.agents.cli.UniverseUpdateJob")
    def test_run_universe_job(self, mock_job_class, cli_test_db):
        """run_job_now('universe') should create and run UniverseUpdateJob."""
        mock_job = MagicMock()
        mock_job.run.return_value = {
            "status": "success",
            "symbols_added": 5,
            "symbols_removed": 2,
            "total_symbols": 500,
        }
        mock_job_class.return_value = mock_job

        result = run_job_now("universe")

        mock_job_class.assert_called_once()
        call_kwargs = mock_job_class.call_args[1]
        assert "as_of_date" in call_kwargs
        assert call_kwargs["actor"] == "user:manual_cli"
        mock_job.run.assert_called_once()
        assert result["symbols_added"] == 5


class TestMainCLI:
    """Tests for main() CLI entry point."""

    @patch("hrp.agents.cli.run_job_now")
    def test_main_run_now_prices(self, mock_run, cli_test_db):
        """main() run-now --job prices should call run_job_now."""
        import sys
        from hrp.agents.cli import main

        mock_run.return_value = {"status": "success", "records_inserted": 100}

        with patch.object(sys, "argv", ["cli", "run-now", "--job", "prices"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once_with("prices", None)

    @patch("hrp.agents.cli.run_job_now")
    def test_main_run_now_with_symbols(self, mock_run, cli_test_db):
        """main() run-now with --symbols should pass symbols."""
        import sys
        from hrp.agents.cli import main

        mock_run.return_value = {"status": "success"}

        with patch.object(sys, "argv", ["cli", "run-now", "--job", "features", "--symbols", "AAPL", "MSFT"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_run.assert_called_once_with("features", ["AAPL", "MSFT"])

    @patch("hrp.agents.cli.run_job_now")
    def test_main_run_now_failed_job(self, mock_run, cli_test_db):
        """main() run-now should exit 1 on failed job."""
        import sys
        from hrp.agents.cli import main

        mock_run.return_value = {"status": "failed", "error": "Test error"}

        with patch.object(sys, "argv", ["cli", "run-now", "--job", "prices"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch("hrp.agents.cli.list_scheduled_jobs")
    def test_main_list_jobs(self, mock_list, cli_test_db, capsys):
        """main() list-jobs should call list_scheduled_jobs and print."""
        import sys
        from hrp.agents.cli import main

        mock_list.return_value = [
            {
                "id": "price_ingestion",
                "name": "Daily Prices",
                "next_run": datetime(2024, 1, 15, 18, 0),
                "trigger": "cron",
            }
        ]

        with patch.object(sys, "argv", ["cli", "list-jobs"]):
            main()

        mock_list.assert_called_once()
        captured = capsys.readouterr()
        assert "price_ingestion" in captured.out

    @patch("hrp.agents.cli.list_scheduled_jobs")
    def test_main_list_jobs_empty(self, mock_list, cli_test_db, capsys):
        """main() list-jobs with no jobs should print message."""
        import sys
        from hrp.agents.cli import main

        mock_list.return_value = []

        with patch.object(sys, "argv", ["cli", "list-jobs"]):
            main()

        captured = capsys.readouterr()
        assert "No scheduled jobs" in captured.out

    @patch("hrp.agents.cli.get_job_status")
    def test_main_job_status(self, mock_status, cli_test_db, capsys):
        """main() job-status should call get_job_status and print."""
        import sys
        from hrp.agents.cli import main

        mock_status.return_value = [
            {
                "log_id": 1,
                "source_id": "price_ingestion",
                "status": "success",
                "started_at": datetime(2024, 1, 15, 18, 0),
                "completed_at": datetime(2024, 1, 15, 18, 5),
                "records_fetched": 100,
                "records_inserted": 95,
                "error_message": None,
            }
        ]

        with patch.object(sys, "argv", ["cli", "job-status"]):
            main()

        mock_status.assert_called_once_with(None, 10)
        captured = capsys.readouterr()
        assert "price_ingestion" in captured.out

    @patch("hrp.agents.cli.get_job_status")
    def test_main_job_status_with_filters(self, mock_status, cli_test_db):
        """main() job-status with --job-id and --limit should pass args."""
        import sys
        from hrp.agents.cli import main

        mock_status.return_value = []

        with patch.object(sys, "argv", ["cli", "job-status", "--job-id", "feature_computation", "--limit", "5"]):
            main()

        mock_status.assert_called_once_with("feature_computation", 5)

    @patch("hrp.agents.cli.get_job_status")
    def test_main_job_status_empty(self, mock_status, cli_test_db, capsys):
        """main() job-status with no history should print message."""
        import sys
        from hrp.agents.cli import main

        mock_status.return_value = []

        with patch.object(sys, "argv", ["cli", "job-status"]):
            main()

        captured = capsys.readouterr()
        assert "No job history" in captured.out

    @patch("hrp.agents.cli.get_job_status")
    def test_main_job_status_with_error(self, mock_status, cli_test_db, capsys):
        """main() job-status should print error messages."""
        import sys
        from hrp.agents.cli import main

        mock_status.return_value = [
            {
                "log_id": 1,
                "source_id": "price_ingestion",
                "status": "failed",
                "started_at": datetime(2024, 1, 15, 18, 0),
                "completed_at": None,
                "records_fetched": 0,
                "records_inserted": 0,
                "error_message": "Connection refused",
            }
        ]

        with patch.object(sys, "argv", ["cli", "job-status"]):
            main()

        captured = capsys.readouterr()
        assert "Connection refused" in captured.out

    @patch("hrp.agents.cli.clear_job_history")
    def test_main_clear_history_with_confirm(self, mock_clear, cli_test_db, capsys):
        """main() clear-history --confirm should skip confirmation."""
        import sys
        from hrp.agents.cli import main

        mock_clear.return_value = 5

        with patch.object(sys, "argv", ["cli", "clear-history", "--confirm"]):
            main()

        mock_clear.assert_called_once()
        captured = capsys.readouterr()
        assert "Deleted 5 records" in captured.out

    @patch("hrp.agents.cli.clear_job_history")
    def test_main_clear_history_with_filters(self, mock_clear, cli_test_db):
        """main() clear-history with filters should pass args."""
        import sys
        from hrp.agents.cli import main

        mock_clear.return_value = 2

        with patch.object(sys, "argv", ["cli", "clear-history", "--job-id", "price_ingestion", "--status", "failed", "--confirm"]):
            main()

        call_args = mock_clear.call_args
        assert call_args[0][0] == "price_ingestion"
        assert call_args[0][2] == "failed"

    @patch("hrp.agents.cli.clear_job_history")
    def test_main_clear_history_with_before_date(self, mock_clear, cli_test_db):
        """main() clear-history with --before should parse date."""
        import sys
        from hrp.agents.cli import main

        mock_clear.return_value = 3

        with patch.object(sys, "argv", ["cli", "clear-history", "--before", "2024-06-01", "--confirm"]):
            main()

        call_args = mock_clear.call_args
        before_dt = call_args[0][1]
        assert before_dt.year == 2024
        assert before_dt.month == 6
        assert before_dt.day == 1

    def test_main_clear_history_invalid_date(self, cli_test_db):
        """main() clear-history with invalid date should exit 1."""
        import sys
        from hrp.agents.cli import main

        with patch.object(sys, "argv", ["cli", "clear-history", "--before", "not-a-date", "--confirm"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    @patch("builtins.input", return_value="n")
    @patch("hrp.agents.cli.clear_job_history")
    def test_main_clear_history_cancelled(self, mock_clear, mock_input, cli_test_db, capsys):
        """main() clear-history without --confirm should prompt and cancel."""
        import sys
        from hrp.agents.cli import main

        with patch.object(sys, "argv", ["cli", "clear-history"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_clear.assert_not_called()
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

    @patch("builtins.input", return_value="y")
    @patch("hrp.agents.cli.clear_job_history")
    def test_main_clear_history_confirmed(self, mock_clear, mock_input, cli_test_db, capsys):
        """main() clear-history with 'y' confirmation should proceed."""
        import sys
        from hrp.agents.cli import main

        mock_clear.return_value = 10

        with patch.object(sys, "argv", ["cli", "clear-history"]):
            main()

        mock_clear.assert_called_once()
        captured = capsys.readouterr()
        assert "Deleted 10 records" in captured.out

    def test_main_no_command(self, cli_test_db):
        """main() with no command should show help and exit 1."""
        import sys
        from hrp.agents.cli import main

        with patch.object(sys, "argv", ["cli"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
