"""
Tests for IngestionScheduler.

Tests cover:
- Scheduler initialization
- Job lifecycle (add, remove, list)
- Daily ingestion setup
- Start/shutdown behavior
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.scheduler import IngestionScheduler


class TestSchedulerInit:
    """Tests for scheduler initialization."""

    def test_init_creates_scheduler(self):
        """Scheduler should initialize with BackgroundScheduler."""
        scheduler = IngestionScheduler()
        assert scheduler.scheduler is not None
        assert not scheduler.running

    def test_init_empty_jobs(self):
        """Scheduler should start with no jobs."""
        scheduler = IngestionScheduler()
        jobs = scheduler.list_jobs()
        assert jobs == []

    def test_repr(self):
        """Scheduler repr should show status and job count."""
        scheduler = IngestionScheduler()
        repr_str = repr(scheduler)
        assert "IngestionScheduler" in repr_str
        assert "stopped" in repr_str
        assert "jobs=0" in repr_str


class TestSchedulerJobManagement:
    """Tests for job add/remove/list operations."""

    def test_add_job(self):
        """Should add a job to the scheduler."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(func=dummy_func, job_id="test_job", trigger="interval", seconds=60)

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test_job"

    def test_add_job_replace_existing(self):
        """Adding job with same ID should replace existing (when scheduler started)."""
        scheduler = IngestionScheduler()
        scheduler.start()  # Must start for replace_existing to work

        try:
            def func1():
                pass

            def func2():
                pass

            scheduler.add_job(func=func1, job_id="test_job", trigger="interval", seconds=60)
            scheduler.add_job(func=func2, job_id="test_job", trigger="interval", seconds=30)

            jobs = scheduler.list_jobs()
            assert len(jobs) == 1
        finally:
            scheduler.shutdown(wait=False)

    def test_remove_job(self):
        """Should remove a job from the scheduler."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(func=dummy_func, job_id="test_job", trigger="interval", seconds=60)
        assert len(scheduler.list_jobs()) == 1

        scheduler.remove_job("test_job")
        assert len(scheduler.list_jobs()) == 0

    def test_remove_nonexistent_job_raises(self):
        """Removing nonexistent job should raise."""
        scheduler = IngestionScheduler()

        with pytest.raises(Exception):
            scheduler.remove_job("nonexistent_job")

    def test_list_jobs_returns_info(self):
        """list_jobs should return job info dictionaries."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(
            func=dummy_func, job_id="test_job", trigger="interval", seconds=60, name="Test Job"
        )

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert "id" in job
        assert "name" in job
        assert "next_run" in job
        assert "trigger" in job
        assert job["name"] == "Test Job"

    def test_get_job_info(self):
        """get_job_info should return specific job details."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(
            func=dummy_func, job_id="my_job", trigger="interval", seconds=60, name="My Job"
        )

        info = scheduler.get_job_info("my_job")
        assert info is not None
        assert info["id"] == "my_job"
        assert info["name"] == "My Job"
        assert "func" in info

    def test_get_job_info_not_found(self):
        """get_job_info should return None for nonexistent job."""
        scheduler = IngestionScheduler()
        info = scheduler.get_job_info("nonexistent")
        assert info is None


class TestSchedulerStartStop:
    """Tests for scheduler start/shutdown."""

    def test_start_scheduler(self):
        """Starting scheduler should change running state."""
        scheduler = IngestionScheduler()
        assert not scheduler.running

        scheduler.start()
        assert scheduler.running

        scheduler.shutdown(wait=False)

    def test_start_already_running(self):
        """Starting already running scheduler should not error."""
        scheduler = IngestionScheduler()
        scheduler.start()
        scheduler.start()  # Should not raise
        assert scheduler.running
        scheduler.shutdown(wait=False)

    def test_shutdown_scheduler(self):
        """Shutdown should stop the scheduler."""
        scheduler = IngestionScheduler()
        scheduler.start()
        assert scheduler.running

        scheduler.shutdown(wait=False)
        assert not scheduler.running

    def test_shutdown_not_running(self):
        """Shutdown on non-running scheduler should not error."""
        scheduler = IngestionScheduler()
        scheduler.shutdown(wait=False)  # Should not raise


class TestSchedulerPauseResume:
    """Tests for job pause/resume."""

    def test_pause_job(self):
        """Should be able to pause a job."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(func=dummy_func, job_id="pausable", trigger="interval", seconds=60)

        # Should not raise
        scheduler.pause_job("pausable")

    def test_resume_job(self):
        """Should be able to resume a paused job."""
        scheduler = IngestionScheduler()

        def dummy_func():
            pass

        scheduler.add_job(func=dummy_func, job_id="resumable", trigger="interval", seconds=60)
        scheduler.pause_job("resumable")
        scheduler.resume_job("resumable")  # Should not raise


class TestSetupDailyIngestion:
    """Tests for daily ingestion pipeline setup."""

    @patch("hrp.agents.jobs.PriceIngestionJob")
    @patch("hrp.agents.jobs.FeatureComputationJob")
    def test_setup_daily_ingestion_creates_jobs(self, mock_feature_job, mock_price_job):
        """setup_daily_ingestion should create price and feature jobs."""
        scheduler = IngestionScheduler()

        # Mock job instances
        mock_price_instance = MagicMock()
        mock_feature_instance = MagicMock()
        mock_price_job.return_value = mock_price_instance
        mock_feature_job.return_value = mock_feature_instance

        scheduler.setup_daily_ingestion()

        jobs = scheduler.list_jobs()
        job_ids = [j["id"] for j in jobs]

        assert "price_ingestion" in job_ids
        assert "feature_computation" in job_ids
        assert len(jobs) == 2

    @patch("hrp.agents.jobs.PriceIngestionJob")
    @patch("hrp.agents.jobs.FeatureComputationJob")
    def test_setup_daily_ingestion_with_custom_times(self, mock_feature_job, mock_price_job):
        """setup_daily_ingestion should respect custom job times."""
        scheduler = IngestionScheduler()

        mock_price_instance = MagicMock()
        mock_feature_instance = MagicMock()
        mock_price_job.return_value = mock_price_instance
        mock_feature_job.return_value = mock_feature_instance

        scheduler.setup_daily_ingestion(price_job_time="19:00", feature_job_time="19:30")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    @patch("hrp.agents.jobs.PriceIngestionJob")
    @patch("hrp.agents.jobs.FeatureComputationJob")
    def test_setup_daily_ingestion_with_symbols(self, mock_feature_job, mock_price_job):
        """setup_daily_ingestion should pass symbols to jobs."""
        scheduler = IngestionScheduler()

        mock_price_instance = MagicMock()
        mock_feature_instance = MagicMock()
        mock_price_job.return_value = mock_price_instance
        mock_feature_job.return_value = mock_feature_instance

        symbols = ["AAPL", "MSFT"]
        scheduler.setup_daily_ingestion(symbols=symbols)

        # Verify PriceIngestionJob was called with symbols
        mock_price_job.assert_called_once_with(symbols=symbols)
