"""
Job scheduler for HRP data ingestion pipeline.

Uses APScheduler to orchestrate daily data ingestion with dependency management.
Includes LineageEventWatcher for event-driven agent coordination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Union

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from pytz import timezone


# Configure timezone for market close (6 PM ET)
ET_TIMEZONE = timezone("US/Eastern")


@dataclass
class LineageTrigger:
    """Configuration for a lineage event trigger."""

    event_type: str
    actor_filter: str | None
    callback: Callable[[dict], None]
    name: str = ""


class LineageEventWatcher:
    """
    Polls lineage table for events and triggers callbacks.

    Enables automatic agent coordination by watching for specific lineage events
    and triggering downstream agents when events match configured criteria.

    Example:
        watcher = LineageEventWatcher(poll_interval_seconds=60)

        # Trigger Alpha Researcher when Signal Scientist creates hypotheses
        watcher.register_trigger(
            event_type="hypothesis_created",
            actor_filter="agent:signal-scientist",
            callback=lambda event: alpha_researcher.run(event["hypothesis_id"]),
        )

        watcher.start()  # Starts polling in background
    """

    def __init__(
        self,
        poll_interval_seconds: int = 60,
        scheduler: "IngestionScheduler | None" = None,
    ):
        """
        Initialize the LineageEventWatcher.

        Args:
            poll_interval_seconds: How often to poll for new events (default: 60)
            scheduler: Optional existing scheduler to use for polling job
        """
        self.poll_interval_seconds = poll_interval_seconds
        self._triggers: list[LineageTrigger] = []
        self._last_lineage_id: int = 0
        self._running = False
        self._scheduler = scheduler
        self._owns_scheduler = scheduler is None

        # Initialize last_lineage_id from database
        self._init_last_lineage_id()

    def _init_last_lineage_id(self) -> None:
        """Initialize last_lineage_id from database to avoid reprocessing old events."""
        try:
            from hrp.data.db import get_db

            db = get_db()
            result = db.fetchone("SELECT COALESCE(MAX(lineage_id), 0) FROM lineage")
            self._last_lineage_id = result[0] if result else 0
            logger.debug(f"LineageEventWatcher initialized at lineage_id={self._last_lineage_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize last_lineage_id: {e}")
            self._last_lineage_id = 0

    def register_trigger(
        self,
        event_type: str,
        callback: Callable[[dict], None],
        actor_filter: str | None = None,
        name: str = "",
    ) -> None:
        """
        Register a callback for matching lineage events.

        Args:
            event_type: Event type to match (e.g., "hypothesis_created")
            callback: Function to call when event matches. Receives event dict.
            actor_filter: Optional actor to filter on (e.g., "agent:signal-scientist")
            name: Optional name for this trigger (for logging)
        """
        trigger = LineageTrigger(
            event_type=event_type,
            actor_filter=actor_filter,
            callback=callback,
            name=name or f"{event_type}:{actor_filter or '*'}",
        )
        self._triggers.append(trigger)
        logger.info(f"Registered lineage trigger: {trigger.name}")

    def unregister_trigger(self, event_type: str, actor_filter: str | None = None) -> bool:
        """
        Unregister a trigger by event_type and actor_filter.

        Args:
            event_type: Event type of trigger to remove
            actor_filter: Actor filter of trigger to remove

        Returns:
            True if trigger was found and removed, False otherwise
        """
        for i, trigger in enumerate(self._triggers):
            if trigger.event_type == event_type and trigger.actor_filter == actor_filter:
                removed = self._triggers.pop(i)
                logger.info(f"Unregistered lineage trigger: {removed.name}")
                return True
        return False

    def poll(self) -> int:
        """
        Check for new events and fire matching callbacks.

        Returns:
            Number of events processed
        """
        if not self._triggers:
            return 0

        try:
            from hrp.data.db import get_db

            db = get_db()

            # Get events newer than last processed
            query = """
                SELECT lineage_id, event_type, timestamp, actor,
                       hypothesis_id, experiment_id, details, parent_lineage_id
                FROM lineage
                WHERE lineage_id > ?
                ORDER BY lineage_id ASC
                LIMIT 100
            """
            rows = db.fetchall(query, (self._last_lineage_id,))

            events_processed = 0

            for row in rows:
                event = {
                    "lineage_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "actor": row[3],
                    "hypothesis_id": row[4],
                    "experiment_id": row[5],
                    "details": row[6],
                    "parent_lineage_id": row[7],
                }

                # Update last processed ID
                self._last_lineage_id = event["lineage_id"]

                # Check each trigger
                for trigger in self._triggers:
                    if self._matches_trigger(event, trigger):
                        try:
                            logger.info(
                                f"Firing trigger '{trigger.name}' for event "
                                f"{event['event_type']} (lineage_id={event['lineage_id']})"
                            )
                            trigger.callback(event)
                            events_processed += 1
                        except Exception as e:
                            logger.error(
                                f"Trigger '{trigger.name}' failed for event "
                                f"{event['lineage_id']}: {e}"
                            )

            if events_processed > 0:
                logger.debug(f"LineageEventWatcher processed {events_processed} events")

            return events_processed

        except Exception as e:
            logger.error(f"LineageEventWatcher poll failed: {e}")
            return 0

    def _matches_trigger(self, event: dict, trigger: LineageTrigger) -> bool:
        """Check if an event matches a trigger's criteria."""
        # Check event type
        if event["event_type"] != trigger.event_type:
            return False

        # Check actor filter if specified
        if trigger.actor_filter is not None:
            if event["actor"] != trigger.actor_filter:
                return False

        return True

    def start(self) -> None:
        """Start polling for events in background."""
        if self._running:
            logger.warning("LineageEventWatcher is already running")
            return

        # Create scheduler if needed
        if self._scheduler is None:
            self._scheduler = IngestionScheduler()
            self._owns_scheduler = True

        # Add polling job
        self._scheduler.add_job(
            func=self.poll,
            job_id="lineage_event_watcher",
            trigger=IntervalTrigger(seconds=self.poll_interval_seconds),
            name="Lineage Event Watcher",
        )

        # Start scheduler if we own it
        if self._owns_scheduler:
            self._scheduler.start()

        self._running = True
        logger.info(
            f"LineageEventWatcher started (poll interval: {self.poll_interval_seconds}s, "
            f"triggers: {len(self._triggers)})"
        )

    def stop(self) -> None:
        """Stop polling for events."""
        if not self._running:
            logger.warning("LineageEventWatcher is not running")
            return

        if self._scheduler:
            try:
                self._scheduler.remove_job("lineage_event_watcher")
            except Exception:
                pass  # Job may not exist

            if self._owns_scheduler:
                self._scheduler.shutdown()

        self._running = False
        logger.info("LineageEventWatcher stopped")

    @property
    def running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    @property
    def trigger_count(self) -> int:
        """Number of registered triggers."""
        return len(self._triggers)

    @property
    def last_lineage_id(self) -> int:
        """Last processed lineage ID."""
        return self._last_lineage_id

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self._running else "stopped"
        return (
            f"<LineageEventWatcher status={status} triggers={len(self._triggers)} "
            f"last_id={self._last_lineage_id}>"
        )


def _parse_time(time_str: str, param_name: str) -> tuple[int, int]:
    """
    Parse HH:MM time string with validation.

    Args:
        time_str: Time string in HH:MM format
        param_name: Parameter name for error messages

    Returns:
        Tuple of (hour, minute)

    Raises:
        ValueError: If time string is invalid
    """
    if not isinstance(time_str, str):
        raise ValueError(f"{param_name} must be a string, got {type(time_str).__name__}")

    parts = time_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"{param_name} must be in HH:MM format, got '{time_str}'")

    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"{param_name} contains non-numeric values: '{time_str}'")

    if not (0 <= hour <= 23):
        raise ValueError(f"Hour must be 0-23, got {hour} in '{time_str}'")
    if not (0 <= minute <= 59):
        raise ValueError(f"Minute must be 0-59, got {minute} in '{time_str}'")

    return hour, minute


class IngestionScheduler:
    """
    Background scheduler for automated data ingestion.

    Manages scheduled jobs for price ingestion and feature computation
    with dependency management and retry logic.
    """

    def __init__(self):
        """Initialize the scheduler with APScheduler BackgroundScheduler."""
        self.scheduler = BackgroundScheduler(timezone=ET_TIMEZONE)
        self._jobs = {}
        logger.info("Ingestion scheduler initialized")

    def add_job(
        self,
        func: Callable,
        job_id: str,
        trigger: Union[str, CronTrigger],
        **kwargs: Any,
    ) -> None:
        """
        Add a job to the scheduler.

        Args:
            func: Function to execute
            job_id: Unique identifier for the job
            trigger: APScheduler trigger (e.g., 'cron', CronTrigger instance)
            **kwargs: Additional APScheduler job configuration
        """
        try:
            job = self.scheduler.add_job(
                func,
                trigger,
                id=job_id,
                replace_existing=True,
                **kwargs,
            )
            self._jobs[job_id] = job
            logger.info(f"Added job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            raise

    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.

        Args:
            job_id: Job identifier to remove
        """
        try:
            self.scheduler.remove_job(job_id)
            self._jobs.pop(job_id, None)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            raise

    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
        else:
            logger.warning("Scheduler is already running")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.

        Args:
            wait: Wait for running jobs to complete before shutdown
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown complete")
        else:
            logger.warning("Scheduler is not running")

    def list_jobs(self) -> list[dict[str, Any]]:
        """
        List all scheduled jobs with their status.

        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = {
                "id": job.id,
                "name": getattr(job, "name", None) or job.id,
                "next_run": getattr(job, "next_run_time", None),
                "trigger": str(getattr(job, "trigger", "unknown")),
            }
            jobs.append(job_info)
            logger.debug(f"Job: {job_info}")
        return jobs

    def get_job_info(self, job_id: str) -> Union[dict[str, Any], None]:
        """
        Get information about a specific job.

        Args:
            job_id: Job identifier

        Returns:
            Job information dictionary or None if not found
        """
        job = self.scheduler.get_job(job_id)
        if job is None:
            logger.warning(f"Job not found: {job_id}")
            return None

        func = getattr(job, "func", None)
        return {
            "id": job.id,
            "name": getattr(job, "name", None) or job.id,
            "next_run": getattr(job, "next_run_time", None),
            "trigger": str(getattr(job, "trigger", "unknown")),
            "func": func.__name__ if func else None,
        }

    def pause_job(self, job_id: str) -> None:
        """
        Pause a scheduled job.

        Args:
            job_id: Job identifier to pause
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            raise

    def resume_job(self, job_id: str) -> None:
        """
        Resume a paused job.

        Args:
            job_id: Job identifier to resume
        """
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            raise

    @property
    def running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.running

    def setup_daily_ingestion(
        self,
        symbols: list[str] | None = None,
        price_job_time: str = "18:00",
        universe_job_time: str = "18:05",
        feature_job_time: str = "18:10",
    ) -> None:
        """
        Configure daily data ingestion pipeline with dependency chain.

        Sets up three scheduled jobs:
        1. Price ingestion at 6:00 PM ET (configurable)
        2. Universe update at 6:05 PM ET (configurable)
        3. Feature computation at 6:10 PM ET (configurable)

        The universe and feature jobs have dependencies on price ingestion completing successfully.

        Args:
            symbols: List of stock tickers to ingest (None = TEST_SYMBOLS for prices, all DB symbols for features)
            price_job_time: Time to run price ingestion (HH:MM format, default: 18:00)
            universe_job_time: Time to run universe update (HH:MM format, default: 18:05)
            feature_job_time: Time to run feature computation (HH:MM format, default: 18:10)
        """
        from hrp.agents.jobs import FeatureComputationJob, PriceIngestionJob, UniverseUpdateJob

        # Parse and validate time strings
        price_hour, price_minute = _parse_time(price_job_time, "price_job_time")
        universe_hour, universe_minute = _parse_time(universe_job_time, "universe_job_time")
        feature_hour, feature_minute = _parse_time(feature_job_time, "feature_job_time")

        # Create job instances
        price_job = PriceIngestionJob(symbols=symbols)
        universe_job = UniverseUpdateJob()
        feature_job = FeatureComputationJob(symbols=None)  # None = all symbols in DB

        # Schedule price ingestion job
        self.add_job(
            func=price_job.run,
            job_id="price_ingestion",
            trigger=CronTrigger(hour=price_hour, minute=price_minute, timezone=ET_TIMEZONE),
            name="Daily Price Ingestion",
        )
        logger.info(f"Scheduled price ingestion at {price_job_time} ET")

        # Schedule universe update job (depends on prices for price-based exclusions)
        self.add_job(
            func=universe_job.run,
            job_id="universe_update",
            trigger=CronTrigger(
                hour=universe_hour, minute=universe_minute, timezone=ET_TIMEZONE
            ),
            name="Daily Universe Update",
        )
        logger.info(f"Scheduled universe update at {universe_job_time} ET")

        # Schedule feature computation job (depends on prices)
        self.add_job(
            func=feature_job.run,
            job_id="feature_computation",
            trigger=CronTrigger(
                hour=feature_hour, minute=feature_minute, timezone=ET_TIMEZONE
            ),
            name="Daily Feature Computation",
        )
        logger.info(f"Scheduled feature computation at {feature_job_time} ET")

        logger.info(
            "Daily ingestion pipeline configured: prices → universe → features (dependency enforced)"
        )

    def setup_daily_backup(
        self,
        backup_time: str = "02:00",
        keep_days: int = 30,
        include_mlflow: bool = True,
    ) -> None:
        """
        Configure daily backup job.

        Schedules a backup job to run at the specified time (default 2 AM ET).
        The backup includes the DuckDB database and optionally MLflow artifacts.

        Args:
            backup_time: Time to run backup (HH:MM format, default: 02:00)
            keep_days: Number of days of backups to retain (default: 30)
            include_mlflow: Whether to include MLflow artifacts (default: True)
        """
        from hrp.data.backup import BackupJob

        # Parse and validate time
        hour, minute = _parse_time(backup_time, "backup_time")

        # Create backup job
        backup_job = BackupJob(
            include_mlflow=include_mlflow,
            keep_days=keep_days,
        )

        # Schedule backup job
        self.add_job(
            func=backup_job.run,
            job_id="daily_backup",
            trigger=CronTrigger(hour=hour, minute=minute, timezone=ET_TIMEZONE),
            name="Daily Backup",
        )
        logger.info(f"Scheduled daily backup at {backup_time} ET (keep {keep_days} days)")

    def setup_weekly_fundamentals(
        self,
        fundamentals_time: str = "10:00",
        day_of_week: str = "sat",
        source: str = "simfin",
    ) -> None:
        """
        Configure weekly fundamentals data ingestion.

        Schedules a job to fetch fundamental data (revenue, EPS, book value, etc.)
        on the specified day of the week. Default is Saturday at 10 AM ET.

        Args:
            fundamentals_time: Time to run fundamentals ingestion (HH:MM format, default: 10:00)
            day_of_week: Day of week to run (mon, tue, wed, thu, fri, sat, sun, default: sat)
            source: Data source ('simfin' or 'yfinance', default: 'simfin')
        """
        from hrp.agents.jobs import FundamentalsIngestionJob

        # Parse and validate time
        hour, minute = _parse_time(fundamentals_time, "fundamentals_time")

        # Validate day of week
        valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        day_lower = day_of_week.lower()
        if day_lower not in valid_days:
            raise ValueError(
                f"day_of_week must be one of {valid_days}, got '{day_of_week}'"
            )

        # Create job instance
        fundamentals_job = FundamentalsIngestionJob(
            symbols=None,  # Use universe symbols
            source=source,
        )

        # Schedule fundamentals ingestion job
        self.add_job(
            func=fundamentals_job.run,
            job_id="fundamentals_ingestion",
            trigger=CronTrigger(
                day_of_week=day_lower,
                hour=hour,
                minute=minute,
                timezone=ET_TIMEZONE,
            ),
            name="Weekly Fundamentals Ingestion",
        )
        logger.info(
            f"Scheduled weekly fundamentals ingestion at {fundamentals_time} ET "
            f"on {day_of_week.upper()} using {source}"
        )

    def setup_weekly_snapshot_fundamentals(
        self,
        snapshot_time: str = "10:30",
        day_of_week: str = "sat",
    ) -> None:
        """
        Configure weekly snapshot fundamentals ingestion.

        Schedules a job to fetch current fundamental metrics (P/E ratio, P/B ratio,
        market cap, dividend yield, EV/EBITDA) on the specified day of the week.
        Default is Saturday at 10:30 AM ET.

        These are point-in-time snapshots stored in the features table, unlike
        quarterly fundamentals which have report dates for backtesting.

        Args:
            snapshot_time: Time to run snapshot ingestion (HH:MM format, default: 10:30)
            day_of_week: Day of week to run (mon, tue, wed, thu, fri, sat, sun, default: sat)
        """
        from hrp.agents.jobs import SnapshotFundamentalsJob

        # Parse and validate time
        hour, minute = _parse_time(snapshot_time, "snapshot_time")

        # Validate day of week
        valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        day_lower = day_of_week.lower()
        if day_lower not in valid_days:
            raise ValueError(
                f"day_of_week must be one of {valid_days}, got '{day_of_week}'"
            )

        # Create job instance
        snapshot_job = SnapshotFundamentalsJob(
            symbols=None,  # Use universe symbols
        )

        # Schedule snapshot fundamentals ingestion job
        self.add_job(
            func=snapshot_job.run,
            job_id="snapshot_fundamentals",
            trigger=CronTrigger(
                day_of_week=day_lower,
                hour=hour,
                minute=minute,
                timezone=ET_TIMEZONE,
            ),
            name="Weekly Snapshot Fundamentals",
        )
        logger.info(
            f"Scheduled weekly snapshot fundamentals at {snapshot_time} ET "
            f"on {day_of_week.upper()}"
        )

    def setup_weekly_signal_scan(
        self,
        scan_time: str = "19:00",
        day_of_week: str = "mon",
        symbols: list[str] | None = None,
        features: list[str] | None = None,
        ic_threshold: float = 0.03,
        create_hypotheses: bool = True,
    ) -> None:
        """
        Configure weekly signal discovery scan.

        Schedules the Signal Scientist agent to run weekly, scanning features
        for predictive signals and creating draft hypotheses for promising ones.

        Args:
            scan_time: Time to run scan (HH:MM format, default: 19:00)
            day_of_week: Day of week to run (mon, tue, wed, thu, fri, sat, sun, default: mon)
            symbols: List of symbols to scan (None = universe symbols)
            features: List of features to scan (None = all 44 features)
            ic_threshold: Minimum IC to create hypothesis (default: 0.03)
            create_hypotheses: Whether to create draft hypotheses
        """
        from hrp.agents.research_agents import SignalScientist

        # Parse and validate time
        hour, minute = _parse_time(scan_time, "scan_time")

        # Validate day of week
        valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        day_lower = day_of_week.lower()
        if day_lower not in valid_days:
            raise ValueError(
                f"day_of_week must be one of {valid_days}, got '{day_of_week}'"
            )

        # Create agent instance
        agent = SignalScientist(
            symbols=symbols,
            features=features,
            ic_threshold=ic_threshold,
            create_hypotheses=create_hypotheses,
        )

        # Schedule signal scan job
        self.add_job(
            func=agent.run,
            job_id="signal_scientist_weekly",
            trigger=CronTrigger(
                day_of_week=day_lower,
                hour=hour,
                minute=minute,
                timezone=ET_TIMEZONE,
            ),
            name="Weekly Signal Scan",
        )
        logger.info(
            f"Scheduled weekly signal scan at {scan_time} ET on {day_of_week.upper()} "
            f"(IC threshold: {ic_threshold}, create_hypotheses: {create_hypotheses})"
        )

    def setup_weekly_sectors(
        self,
        sectors_time: str = "10:15",
        day_of_week: str = "sat",
        symbols: list[str] | None = None,
    ) -> None:
        """
        Configure weekly sector data ingestion.

        Schedules a job to fetch GICS sector classifications from Polygon.io
        (with Yahoo Finance fallback) on the specified day of the week.
        Default is Saturday at 10:15 AM ET.

        Args:
            sectors_time: Time to run sector ingestion (HH:MM format, default: 10:15)
            day_of_week: Day of week to run (mon, tue, wed, thu, fri, sat, sun, default: sat)
            symbols: List of symbols to update (None = all universe symbols)
        """
        from hrp.data.ingestion.sectors import SectorIngestionJob

        # Parse and validate time
        hour, minute = _parse_time(sectors_time, "sectors_time")

        # Validate day of week
        valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        day_lower = day_of_week.lower()
        if day_lower not in valid_days:
            raise ValueError(
                f"day_of_week must be one of {valid_days}, got '{day_of_week}'"
            )

        # Create job instance
        sector_job = SectorIngestionJob(symbols=symbols)

        # Schedule sector ingestion job
        self.add_job(
            func=sector_job.run,
            job_id="sector_ingestion",
            trigger=CronTrigger(
                day_of_week=day_lower,
                hour=hour,
                minute=minute,
                timezone=ET_TIMEZONE,
            ),
            name="Weekly Sector Ingestion",
        )
        logger.info(
            f"Scheduled weekly sector ingestion at {sectors_time} ET on {day_of_week.upper()}"
        )

    def setup_quality_sentinel(
        self,
        audit_time: str = "06:00",
        audit_window_days: int = 1,
        include_monitoring: bool = True,
        send_alerts: bool = True,
    ) -> None:
        """
        Configure daily ML Quality Sentinel audit.

        Schedules the ML Quality Sentinel to run daily, auditing recent experiments
        for overfitting, leakage, and other quality issues.

        Args:
            audit_time: Time to run audit (HH:MM format, default: 06:00 AM ET)
            audit_window_days: Days of recent experiments to audit (default: 1)
            include_monitoring: Whether to monitor deployed models (default: True)
            send_alerts: Whether to send email alerts for critical issues (default: True)
        """
        from hrp.agents.research_agents import MLQualitySentinel

        # Parse and validate time
        hour, minute = _parse_time(audit_time, "audit_time")

        # Create agent instance
        sentinel = MLQualitySentinel(
            audit_window_days=audit_window_days,
            include_monitoring=include_monitoring,
            send_alerts=send_alerts,
        )

        # Schedule quality sentinel job
        self.add_job(
            func=sentinel.run,
            job_id="ml_quality_sentinel",
            trigger=CronTrigger(hour=hour, minute=minute, timezone=ET_TIMEZONE),
            name="Daily ML Quality Sentinel",
        )
        logger.info(
            f"Scheduled ML Quality Sentinel at {audit_time} ET "
            f"(window: {audit_window_days} days, monitoring: {include_monitoring})"
        )

    def setup_daily_report(
        self,
        report_time: str = "07:00",
    ) -> None:
        """
        Schedule daily research report generation.

        Generates a daily research summary at the specified time, aggregating
        data from hypotheses, experiments, signals, and agent activity.

        Args:
            report_time: Time to generate daily report (HH:MM format, default 07:00)
        """
        from hrp.agents.report_generator import ReportGenerator

        # Parse and validate time
        hour, minute = _parse_time(report_time, "report_time")

        # Create a wrapper function that creates a fresh generator each run
        def run_daily_report():
            generator = ReportGenerator(report_type="daily")
            generator.run()

        # Schedule daily report job
        self.add_job(
            func=run_daily_report,
            job_id="daily_report",
            trigger=CronTrigger(hour=hour, minute=minute, timezone=ET_TIMEZONE),
            name="Daily Research Report",
        )
        logger.info(f"Scheduled daily research report at {report_time} ET")

    def setup_weekly_report(
        self,
        report_time: str = "20:00",
    ) -> None:
        """
        Schedule weekly research report generation.

        Generates a comprehensive weekly research summary at the specified time,
        including pipeline velocity, top hypotheses, and extended analysis.

        Args:
            report_time: Time to generate weekly report (HH:MM format, default 20:00)
        """
        from hrp.agents.report_generator import ReportGenerator

        # Parse and validate time
        hour, minute = _parse_time(report_time, "report_time")

        # Create a wrapper function that creates a fresh generator each run
        def run_weekly_report():
            generator = ReportGenerator(report_type="weekly")
            generator.run()

        # Schedule weekly report job (Sunday evening)
        self.add_job(
            func=run_weekly_report,
            job_id="weekly_report",
            trigger=CronTrigger(
                day_of_week="sun",
                hour=hour,
                minute=minute,
                timezone=ET_TIMEZONE,
            ),
            name="Weekly Research Report",
        )
        logger.info(f"Scheduled weekly research report on Sunday at {report_time} ET")

    def setup_research_agent_triggers(
        self,
        poll_interval_seconds: int = 60,
    ) -> LineageEventWatcher:
        """
        Set up event-driven triggers for research agent coordination.

        Creates a LineageEventWatcher that monitors the lineage table and
        automatically triggers downstream agents when upstream agents complete.

        Trigger chain:
        - Signal Scientist (hypothesis_created) → Alpha Researcher
        - Alpha Researcher (alpha_researcher_review, PROCEED) → ML Scientist
        - ML Scientist (experiment_completed) → ML Quality Sentinel
        - ML Quality Sentinel (ml_quality_sentinel_audit, passed) → Validation Analyst

        Args:
            poll_interval_seconds: How often to poll lineage table (default: 60s)

        Returns:
            The configured LineageEventWatcher instance
        """
        from hrp.agents.alpha_researcher import AlphaResearcher
        from hrp.agents.research_agents import MLQualitySentinel, MLScientist, ValidationAnalyst

        watcher = LineageEventWatcher(
            poll_interval_seconds=poll_interval_seconds,
            scheduler=self,
        )

        # Trigger 1: Signal Scientist → Alpha Researcher
        # When Signal Scientist creates a hypothesis, Alpha Researcher reviews it
        def on_hypothesis_created(event: dict) -> None:
            details = event.get("details", {})
            hypothesis_id = details.get("hypothesis_id") or event.get("hypothesis_id")
            if hypothesis_id:
                logger.info(f"Triggering Alpha Researcher for hypothesis {hypothesis_id}")
                try:
                    researcher = AlphaResearcher()
                    researcher.run()
                except Exception as e:
                    logger.error(f"Alpha Researcher trigger failed: {e}")

        watcher.register_trigger(
            event_type="hypothesis_created",
            callback=on_hypothesis_created,
            actor_filter="agent:signal-scientist",
            name="signal_scientist_to_alpha_researcher",
        )

        # Trigger 2: Alpha Researcher → ML Scientist
        # When Alpha Researcher promotes hypothesis to testing, ML Scientist validates it
        def on_alpha_review(event: dict) -> None:
            details = event.get("details", {})
            recommendation = details.get("recommendation", "")
            hypothesis_id = event.get("hypothesis_id")

            # Only trigger if Alpha Researcher recommended PROCEED
            if recommendation == "PROCEED" and hypothesis_id:
                logger.info(f"Triggering ML Scientist for hypothesis {hypothesis_id}")
                try:
                    scientist = MLScientist(hypothesis_id=hypothesis_id)
                    scientist.run()
                except Exception as e:
                    logger.error(f"ML Scientist trigger failed: {e}")

        watcher.register_trigger(
            event_type="alpha_researcher_review",
            callback=on_alpha_review,
            actor_filter="agent:alpha-researcher",
            name="alpha_researcher_to_ml_scientist",
        )

        # Trigger 3: ML Scientist → ML Quality Sentinel
        # When ML Scientist completes an experiment, Quality Sentinel audits it
        def on_experiment_completed(event: dict) -> None:
            details = event.get("details", {})
            experiment_id = details.get("experiment_id") or event.get("experiment_id")
            hypothesis_id = event.get("hypothesis_id")

            if experiment_id or hypothesis_id:
                logger.info(
                    f"Triggering ML Quality Sentinel for experiment {experiment_id}"
                )
                try:
                    sentinel = MLQualitySentinel(
                        audit_window_days=1,
                        send_alerts=True,
                    )
                    sentinel.run()
                except Exception as e:
                    logger.error(f"ML Quality Sentinel trigger failed: {e}")

        watcher.register_trigger(
            event_type="experiment_completed",
            callback=on_experiment_completed,
            actor_filter="agent:ml-scientist",
            name="ml_scientist_to_quality_sentinel",
        )

        # Trigger 4: ML Quality Sentinel → Validation Analyst
        # When Quality Sentinel completes audit, Validation Analyst stress tests
        def on_quality_audit(event: dict) -> None:
            details = event.get("details", {})
            hypothesis_id = event.get("hypothesis_id")
            overall_passed = details.get("overall_passed", False)

            # Only trigger if audit passed (no critical issues)
            if overall_passed and hypothesis_id:
                logger.info(
                    f"Triggering Validation Analyst for hypothesis {hypothesis_id}"
                )
                try:
                    analyst = ValidationAnalyst(
                        hypothesis_ids=[hypothesis_id],
                        send_alerts=True,
                    )
                    analyst.run()
                except Exception as e:
                    logger.error(f"Validation Analyst trigger failed: {e}")

        watcher.register_trigger(
            event_type="ml_quality_sentinel_audit",
            callback=on_quality_audit,
            actor_filter="agent:ml-quality-sentinel",
            name="ml_quality_sentinel_to_validation_analyst",
        )

        # Store watcher reference
        self._event_watcher = watcher

        logger.info(
            f"Research agent triggers configured (poll interval: {poll_interval_seconds}s, "
            f"triggers: {watcher.trigger_count})"
        )

        return watcher

    def start_with_triggers(self) -> None:
        """Start the scheduler with research agent triggers enabled."""
        if not hasattr(self, "_event_watcher"):
            self.setup_research_agent_triggers()

        self._event_watcher.start()
        self.start()
        logger.info("Scheduler started with research agent triggers")

    def __repr__(self) -> str:
        """String representation of the scheduler."""
        status = "running" if self.running else "stopped"
        job_count = len(self.scheduler.get_jobs())
        return f"<IngestionScheduler status={status} jobs={job_count}>"
