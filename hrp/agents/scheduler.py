"""
Job scheduler for HRP data ingestion pipeline.

Uses APScheduler to orchestrate daily data ingestion with dependency management.
"""

from datetime import datetime
from typing import Any, Callable, Union

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from pytz import timezone


# Configure timezone for market close (6 PM ET)
ET_TIMEZONE = timezone("US/Eastern")


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

    def __repr__(self) -> str:
        """String representation of the scheduler."""
        status = "running" if self.running else "stopped"
        job_count = len(self.scheduler.get_jobs())
        return f"<IngestionScheduler status={status} jobs={job_count}>"
