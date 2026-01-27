"""
Tests for IngestionScheduler and LineageEventWatcher.

Tests cover:
- Scheduler initialization
- Job lifecycle (add, remove, list)
- Daily ingestion setup
- Start/shutdown behavior
- LineageEventWatcher trigger registration
- LineageEventWatcher polling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.scheduler import IngestionScheduler, LineageEventWatcher, LineageTrigger


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
    @patch("hrp.agents.jobs.UniverseUpdateJob")
    @patch("hrp.agents.jobs.FeatureComputationJob")
    def test_setup_daily_ingestion_creates_jobs(
        self, mock_feature_job, mock_universe_job, mock_price_job
    ):
        """setup_daily_ingestion should create price, universe, and feature jobs."""
        scheduler = IngestionScheduler()

        # Mock job instances
        mock_price_instance = MagicMock()
        mock_universe_instance = MagicMock()
        mock_feature_instance = MagicMock()
        mock_price_job.return_value = mock_price_instance
        mock_universe_job.return_value = mock_universe_instance
        mock_feature_job.return_value = mock_feature_instance

        scheduler.setup_daily_ingestion()

        jobs = scheduler.list_jobs()
        job_ids = [j["id"] for j in jobs]

        assert "price_ingestion" in job_ids
        assert "universe_update" in job_ids
        assert "feature_computation" in job_ids
        assert len(jobs) == 3

    @patch("hrp.agents.jobs.PriceIngestionJob")
    @patch("hrp.agents.jobs.UniverseUpdateJob")
    @patch("hrp.agents.jobs.FeatureComputationJob")
    def test_setup_daily_ingestion_with_custom_times(
        self, mock_feature_job, mock_universe_job, mock_price_job
    ):
        """setup_daily_ingestion should respect custom job times."""
        scheduler = IngestionScheduler()

        mock_price_instance = MagicMock()
        mock_universe_instance = MagicMock()
        mock_feature_instance = MagicMock()
        mock_price_job.return_value = mock_price_instance
        mock_universe_job.return_value = mock_universe_instance
        mock_feature_job.return_value = mock_feature_instance

        scheduler.setup_daily_ingestion(
            price_job_time="19:00",
            universe_job_time="19:05",
            feature_job_time="19:30",
        )

        jobs = scheduler.list_jobs()
        assert len(jobs) == 3

        # Verify custom times are used
        price_job = [j for j in jobs if j["id"] == "price_ingestion"][0]
        assert "19" in price_job["trigger"] and "0" in price_job["trigger"]

        universe_job = [j for j in jobs if j["id"] == "universe_update"][0]
        assert "19" in universe_job["trigger"] and "5" in universe_job["trigger"]

        feature_job = [j for j in jobs if j["id"] == "feature_computation"][0]
        assert "19" in feature_job["trigger"] and "30" in feature_job["trigger"]

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


class TestSetupDailyBackup:
    """Tests for daily backup setup."""

    @patch("hrp.data.backup.BackupJob")
    def test_setup_daily_backup_creates_job(self, mock_backup_job):
        """setup_daily_backup should create a backup job."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_backup_job.return_value = mock_job_instance

        scheduler.setup_daily_backup()

        jobs = scheduler.list_jobs()
        job_ids = [j["id"] for j in jobs]

        assert "daily_backup" in job_ids
        assert len(jobs) == 1

    @patch("hrp.data.backup.BackupJob")
    def test_setup_daily_backup_with_custom_time(self, mock_backup_job):
        """setup_daily_backup should respect custom time."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_backup_job.return_value = mock_job_instance

        scheduler.setup_daily_backup(backup_time="03:00")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1

    @patch("hrp.data.backup.BackupJob")
    def test_setup_daily_backup_with_options(self, mock_backup_job):
        """setup_daily_backup should pass options to BackupJob."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_backup_job.return_value = mock_job_instance

        scheduler.setup_daily_backup(keep_days=14, include_mlflow=False)

        mock_backup_job.assert_called_once_with(
            include_mlflow=False,
            keep_days=14,
        )


class TestSetupWeeklySignalScan:
    """Tests for weekly signal scan setup."""

    @patch("hrp.agents.research_agents.SignalScientist")
    def test_setup_weekly_signal_scan_creates_job(self, mock_scientist):
        """setup_weekly_signal_scan should create a signal scientist job."""
        scheduler = IngestionScheduler()

        mock_agent_instance = MagicMock()
        mock_scientist.return_value = mock_agent_instance

        scheduler.setup_weekly_signal_scan()

        jobs = scheduler.list_jobs()
        job_ids = [j["id"] for j in jobs]

        assert "signal_scientist_weekly" in job_ids
        assert len(jobs) == 1

    @patch("hrp.agents.research_agents.SignalScientist")
    def test_setup_weekly_signal_scan_with_custom_day(self, mock_scientist):
        """setup_weekly_signal_scan should respect custom day of week."""
        scheduler = IngestionScheduler()

        mock_agent_instance = MagicMock()
        mock_scientist.return_value = mock_agent_instance

        scheduler.setup_weekly_signal_scan(day_of_week="fri")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert "fri" in jobs[0]["trigger"]

    @patch("hrp.agents.research_agents.SignalScientist")
    def test_setup_weekly_signal_scan_with_custom_time(self, mock_scientist):
        """setup_weekly_signal_scan should respect custom time."""
        scheduler = IngestionScheduler()

        mock_agent_instance = MagicMock()
        mock_scientist.return_value = mock_agent_instance

        scheduler.setup_weekly_signal_scan(scan_time="20:30")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        # Verify time is in trigger
        assert "20" in jobs[0]["trigger"] and "30" in jobs[0]["trigger"]

    @patch("hrp.agents.research_agents.SignalScientist")
    def test_setup_weekly_signal_scan_passes_params(self, mock_scientist):
        """setup_weekly_signal_scan should pass parameters to SignalScientist."""
        scheduler = IngestionScheduler()

        mock_agent_instance = MagicMock()
        mock_scientist.return_value = mock_agent_instance

        symbols = ["AAPL", "MSFT"]
        features = ["momentum_20d", "volatility_60d"]

        scheduler.setup_weekly_signal_scan(
            symbols=symbols,
            features=features,
            ic_threshold=0.05,
            create_hypotheses=False,
        )

        mock_scientist.assert_called_once_with(
            symbols=symbols,
            features=features,
            ic_threshold=0.05,
            create_hypotheses=False,
        )

    def test_setup_weekly_signal_scan_invalid_day_raises(self):
        """setup_weekly_signal_scan should raise for invalid day."""
        scheduler = IngestionScheduler()

        with pytest.raises(ValueError, match="day_of_week"):
            scheduler.setup_weekly_signal_scan(day_of_week="invalid")

    def test_setup_weekly_signal_scan_invalid_time_raises(self):
        """setup_weekly_signal_scan should raise for invalid time."""
        scheduler = IngestionScheduler()

        with pytest.raises(ValueError):
            scheduler.setup_weekly_signal_scan(scan_time="25:00")


class TestLineageTrigger:
    """Tests for LineageTrigger dataclass."""

    def test_trigger_creation(self):
        """Should create a LineageTrigger with all fields."""

        def dummy_callback(event):
            pass

        trigger = LineageTrigger(
            event_type="hypothesis_created",
            actor_filter="agent:signal-scientist",
            callback=dummy_callback,
            name="test-trigger",
        )

        assert trigger.event_type == "hypothesis_created"
        assert trigger.actor_filter == "agent:signal-scientist"
        assert trigger.callback == dummy_callback
        assert trigger.name == "test-trigger"

    def test_trigger_default_name(self):
        """Should use empty string as default name."""

        def dummy_callback(event):
            pass

        trigger = LineageTrigger(
            event_type="hypothesis_created",
            actor_filter=None,
            callback=dummy_callback,
        )

        assert trigger.name == ""


class TestLineageEventWatcherInit:
    """Tests for LineageEventWatcher initialization."""

    @patch("hrp.data.db.get_db")
    def test_init_default_interval(self, mock_get_db):
        """Should initialize with default poll interval."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        assert watcher.poll_interval_seconds == 60
        assert not watcher.running
        assert watcher.trigger_count == 0

    @patch("hrp.data.db.get_db")
    def test_init_custom_interval(self, mock_get_db):
        """Should accept custom poll interval."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher(poll_interval_seconds=30)

        assert watcher.poll_interval_seconds == 30

    @patch("hrp.data.db.get_db")
    def test_init_reads_last_lineage_id(self, mock_get_db):
        """Should initialize last_lineage_id from database."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (42,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        assert watcher.last_lineage_id == 42

    @patch("hrp.data.db.get_db")
    def test_init_handles_db_error(self, mock_get_db):
        """Should handle database error gracefully."""
        mock_get_db.side_effect = Exception("DB error")

        watcher = LineageEventWatcher()

        assert watcher.last_lineage_id == 0

    @patch("hrp.data.db.get_db")
    def test_repr(self, mock_get_db):
        """Should show status and trigger count in repr."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (10,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        repr_str = repr(watcher)
        assert "LineageEventWatcher" in repr_str
        assert "stopped" in repr_str
        assert "triggers=0" in repr_str
        assert "last_id=10" in repr_str


class TestLineageEventWatcherTriggers:
    """Tests for LineageEventWatcher trigger registration."""

    @patch("hrp.data.db.get_db")
    def test_register_trigger(self, mock_get_db):
        """Should register a trigger."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        def callback(event):
            pass

        watcher.register_trigger(
            event_type="hypothesis_created",
            callback=callback,
            actor_filter="agent:test",
            name="test-trigger",
        )

        assert watcher.trigger_count == 1

    @patch("hrp.data.db.get_db")
    def test_register_multiple_triggers(self, mock_get_db):
        """Should register multiple triggers."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        def callback(event):
            pass

        watcher.register_trigger("event1", callback)
        watcher.register_trigger("event2", callback)
        watcher.register_trigger("event3", callback)

        assert watcher.trigger_count == 3

    @patch("hrp.data.db.get_db")
    def test_unregister_trigger(self, mock_get_db):
        """Should unregister a trigger."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        def callback(event):
            pass

        watcher.register_trigger("event1", callback, actor_filter="agent:test")
        assert watcher.trigger_count == 1

        result = watcher.unregister_trigger("event1", actor_filter="agent:test")

        assert result is True
        assert watcher.trigger_count == 0

    @patch("hrp.data.db.get_db")
    def test_unregister_nonexistent_trigger(self, mock_get_db):
        """Should return False for nonexistent trigger."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        result = watcher.unregister_trigger("nonexistent", actor_filter=None)

        assert result is False

    @patch("hrp.data.db.get_db")
    def test_register_trigger_auto_name(self, mock_get_db):
        """Should generate name when not provided."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        def callback(event):
            pass

        watcher.register_trigger("event1", callback, actor_filter="agent:test")

        # Verify trigger was registered (name is generated internally)
        assert watcher.trigger_count == 1


class TestLineageEventWatcherPoll:
    """Tests for LineageEventWatcher polling."""

    @patch("hrp.data.db.get_db")
    def test_poll_no_triggers_returns_zero(self, mock_get_db):
        """Should return 0 when no triggers registered."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()
        result = watcher.poll()

        assert result == 0

    @patch("hrp.data.db.get_db")
    def test_poll_fires_matching_callback(self, mock_get_db):
        """Should fire callback for matching event."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_db.fetchall.return_value = [
            (1, "hypothesis_created", datetime.now(), "agent:signal-scientist", "HYP-001", None, None, None)
        ]
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        received_events = []

        def callback(event):
            received_events.append(event)

        watcher.register_trigger("hypothesis_created", callback, actor_filter="agent:signal-scientist")
        result = watcher.poll()

        assert result == 1
        assert len(received_events) == 1
        assert received_events[0]["hypothesis_id"] == "HYP-001"
        assert watcher.last_lineage_id == 1

    @patch("hrp.data.db.get_db")
    def test_poll_filters_by_event_type(self, mock_get_db):
        """Should not fire callback for non-matching event type."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_db.fetchall.return_value = [
            (1, "experiment_run", datetime.now(), "agent:test", None, "EXP-001", None, None)
        ]
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        received_events = []

        def callback(event):
            received_events.append(event)

        watcher.register_trigger("hypothesis_created", callback)
        result = watcher.poll()

        assert result == 0
        assert len(received_events) == 0

    @patch("hrp.data.db.get_db")
    def test_poll_filters_by_actor(self, mock_get_db):
        """Should not fire callback for non-matching actor."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_db.fetchall.return_value = [
            (1, "hypothesis_created", datetime.now(), "user", "HYP-001", None, None, None)
        ]
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        received_events = []

        def callback(event):
            received_events.append(event)

        watcher.register_trigger("hypothesis_created", callback, actor_filter="agent:signal-scientist")
        result = watcher.poll()

        assert result == 0
        assert len(received_events) == 0

    @patch("hrp.data.db.get_db")
    def test_poll_wildcard_actor(self, mock_get_db):
        """Should match any actor when actor_filter is None."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_db.fetchall.return_value = [
            (1, "hypothesis_created", datetime.now(), "any-actor", "HYP-001", None, None, None)
        ]
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        received_events = []

        def callback(event):
            received_events.append(event)

        watcher.register_trigger("hypothesis_created", callback, actor_filter=None)
        result = watcher.poll()

        assert result == 1
        assert len(received_events) == 1

    @patch("hrp.data.db.get_db")
    def test_poll_handles_callback_error(self, mock_get_db):
        """Should continue processing after callback error."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_db.fetchall.return_value = [
            (1, "event1", datetime.now(), "agent", None, None, None, None),
            (2, "event2", datetime.now(), "agent", None, None, None, None),
        ]
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        call_count = {"value": 0}

        def failing_callback(event):
            call_count["value"] += 1
            if event["lineage_id"] == 1:
                raise Exception("Test error")

        watcher.register_trigger("event1", failing_callback)
        watcher.register_trigger("event2", failing_callback)

        # Should not raise, and should process second event
        result = watcher.poll()

        # First callback fails, second succeeds
        assert call_count["value"] == 2
        assert watcher.last_lineage_id == 2

    @patch("hrp.data.db.get_db")
    def test_poll_handles_db_error(self, mock_get_db):
        """Should handle database error gracefully."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()

        def callback(event):
            pass

        watcher.register_trigger("event1", callback)

        # Simulate DB error on poll
        mock_db.fetchall.side_effect = Exception("DB error")

        result = watcher.poll()

        assert result == 0


class TestLineageEventWatcherStartStop:
    """Tests for LineageEventWatcher start/stop."""

    @patch("hrp.data.db.get_db")
    def test_start_creates_scheduler(self, mock_get_db):
        """Should create scheduler when not provided."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()
        watcher.start()

        try:
            assert watcher.running
        finally:
            watcher.stop()

    @patch("hrp.data.db.get_db")
    def test_start_uses_provided_scheduler(self, mock_get_db):
        """Should use provided scheduler."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        scheduler = IngestionScheduler()
        scheduler.start()

        try:
            watcher = LineageEventWatcher(scheduler=scheduler)
            watcher.start()

            assert watcher.running
            # Verify job was added to existing scheduler
            jobs = scheduler.list_jobs()
            job_ids = [j["id"] for j in jobs]
            assert "lineage_event_watcher" in job_ids
        finally:
            watcher.stop()
            scheduler.shutdown(wait=False)

    @patch("hrp.data.db.get_db")
    def test_start_already_running(self, mock_get_db):
        """Should not raise when starting already running watcher."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()
        watcher.start()

        try:
            watcher.start()  # Should not raise
            assert watcher.running
        finally:
            watcher.stop()

    @patch("hrp.data.db.get_db")
    def test_stop_watcher(self, mock_get_db):
        """Should stop running watcher."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()
        watcher.start()
        assert watcher.running

        watcher.stop()
        assert not watcher.running

    @patch("hrp.data.db.get_db")
    def test_stop_not_running(self, mock_get_db):
        """Should not raise when stopping non-running watcher."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        watcher = LineageEventWatcher()
        watcher.stop()  # Should not raise

        assert not watcher.running


class TestSetupWeeklySectors:
    """Tests for weekly sector ingestion setup."""

    @patch("hrp.data.ingestion.sectors.SectorIngestionJob")
    def test_setup_weekly_sectors_creates_job(self, mock_sector_job):
        """setup_weekly_sectors should create a sector ingestion job."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_sector_job.return_value = mock_job_instance

        scheduler.setup_weekly_sectors()

        jobs = scheduler.list_jobs()
        job_ids = [j["id"] for j in jobs]

        assert "sector_ingestion" in job_ids
        assert len(jobs) == 1

    @patch("hrp.data.ingestion.sectors.SectorIngestionJob")
    def test_setup_weekly_sectors_with_custom_day(self, mock_sector_job):
        """setup_weekly_sectors should respect custom day of week."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_sector_job.return_value = mock_job_instance

        scheduler.setup_weekly_sectors(day_of_week="sun")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert "sun" in jobs[0]["trigger"]

    @patch("hrp.data.ingestion.sectors.SectorIngestionJob")
    def test_setup_weekly_sectors_with_custom_time(self, mock_sector_job):
        """setup_weekly_sectors should respect custom time."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_sector_job.return_value = mock_job_instance

        scheduler.setup_weekly_sectors(sectors_time="09:30")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        # Verify time is in trigger
        assert "9" in jobs[0]["trigger"] and "30" in jobs[0]["trigger"]

    @patch("hrp.data.ingestion.sectors.SectorIngestionJob")
    def test_setup_weekly_sectors_passes_symbols(self, mock_sector_job):
        """setup_weekly_sectors should pass symbols to SectorIngestionJob."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_sector_job.return_value = mock_job_instance

        symbols = ["AAPL", "MSFT", "GOOGL"]
        scheduler.setup_weekly_sectors(symbols=symbols)

        mock_sector_job.assert_called_once_with(symbols=symbols)

    @patch("hrp.data.ingestion.sectors.SectorIngestionJob")
    def test_setup_weekly_sectors_default_symbols_none(self, mock_sector_job):
        """setup_weekly_sectors should pass None for symbols by default (all universe)."""
        scheduler = IngestionScheduler()

        mock_job_instance = MagicMock()
        mock_sector_job.return_value = mock_job_instance

        scheduler.setup_weekly_sectors()

        mock_sector_job.assert_called_once_with(symbols=None)

    def test_setup_weekly_sectors_invalid_day_raises(self):
        """setup_weekly_sectors should raise for invalid day."""
        scheduler = IngestionScheduler()

        with pytest.raises(ValueError, match="day_of_week"):
            scheduler.setup_weekly_sectors(day_of_week="invalid")

    def test_setup_weekly_sectors_invalid_time_raises(self):
        """setup_weekly_sectors should raise for invalid time."""
        scheduler = IngestionScheduler()

        with pytest.raises(ValueError):
            scheduler.setup_weekly_sectors(sectors_time="25:00")


class TestResearchAgentTriggers:
    """Tests for research agent trigger setup."""

    @patch("hrp.data.db.get_db")
    def test_setup_research_agent_triggers_includes_validation_analyst(self, mock_get_db):
        """setup_research_agent_triggers should include ValidationAnalyst trigger."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = (0,)
        mock_get_db.return_value = mock_db

        scheduler = IngestionScheduler()
        watcher = scheduler.setup_research_agent_triggers()

        # Should have 4 triggers:
        # 1. Signal Scientist → Alpha Researcher
        # 2. Alpha Researcher → ML Scientist
        # 3. ML Scientist → ML Quality Sentinel
        # 4. ML Quality Sentinel → Validation Analyst
        assert watcher.trigger_count == 4

        # Verify the Validation Analyst trigger is registered
        trigger_names = [t.name for t in watcher._triggers]
        assert "ml_quality_sentinel_to_validation_analyst" in trigger_names
