"""
Job orchestration for HRP data ingestion pipeline.

Base classes for scheduled jobs with dependency management, retry logic,
and logging to the ingestion_log table.
"""

import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger

from hrp.api.platform import PlatformAPI
from hrp.data.ingestion.features import compute_features
from hrp.data.ingestion.fundamentals import ingest_fundamentals
from hrp.data.ingestion.intraday import IntradayIngestionService
from hrp.data.ingestion.prices import ingest_prices
from hrp.data.universe import UniverseManager
from hrp.notifications.email import EmailNotifier


# Transient errors that should be retried (network/IO issues)
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)


class JobStatus(Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class DataRequirement:
    """
    Specifies a data availability requirement for job dependencies.

    Instead of checking if a job has run, this checks if the actual
    data needed by the job exists in the database.
    """

    def __init__(
        self,
        table: str,
        min_rows: int = 1,
        max_age_days: int | None = None,
        date_column: str = "date",
        description: str | None = None,
    ):
        """
        Initialize a data requirement.

        Args:
            table: Database table to check (e.g., "prices", "features")
            min_rows: Minimum number of rows required
            max_age_days: Maximum age in days for most recent data (None = no limit)
            date_column: Column to check for recency
            description: Human-readable description for error messages
        """
        self.table = table
        self.min_rows = min_rows
        self.max_age_days = max_age_days
        self.date_column = date_column
        self.description = description or f"{table} data"

    def check(self) -> tuple[bool, str]:
        """
        Check if the data requirement is met.

        Returns:
            Tuple of (is_met, message)
        """
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()

        # Check row count
        result = api.fetchone_readonly(
            f"SELECT COUNT(*) FROM {self.table}"
        )
        row_count = result[0] if result else 0

        if row_count < self.min_rows:
            return False, f"{self.description}: found {row_count} rows, need {self.min_rows}"

        # Check recency if max_age_days specified
        if self.max_age_days is not None:
            result = api.fetchone_readonly(
                f"SELECT MAX({self.date_column}) FROM {self.table}"
            )
            if result and result[0]:
                from datetime import datetime
                max_date = result[0]
                if isinstance(max_date, str):
                    max_date = datetime.strptime(max_date[:10], "%Y-%m-%d").date()
                elif hasattr(max_date, 'date'):
                    max_date = max_date.date()

                age_days = (date.today() - max_date).days
                if age_days > self.max_age_days:
                    return False, (
                        f"{self.description}: most recent data is {age_days} days old "
                        f"(max allowed: {self.max_age_days})"
                    )
            else:
                return False, f"{self.description}: no date data found"

        return True, f"{self.description}: OK ({row_count} rows)"


class IngestionJob(ABC):
    """
    Base class for scheduled data ingestion jobs.

    Features:
    - Automatic logging to ingestion_log table
    - Retry logic with exponential backoff
    - Dependency management (job-based or data-based)
    - Status tracking
    """

    def __init__(
        self,
        job_id: str,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
        data_requirements: list[DataRequirement] | None = None,
    ):
        """
        Initialize an ingestion job.

        Args:
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
                         (legacy - prefer data_requirements for new jobs)
            data_requirements: List of DataRequirement objects specifying what data
                              must exist before this job can run
        """
        self.job_id = job_id
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.dependencies = dependencies or []
        self.data_requirements = data_requirements or []
        self.api = PlatformAPI()
        self._status = JobStatus.PENDING
        self.log_id: int | None = None
        self.retry_count = 0
        self.last_error: str | None = None

    @property
    def status(self) -> JobStatus:
        """Get current job status (read-only from external code)."""
        return self._status

    def _set_status(self, new_status: JobStatus) -> None:
        """
        Internal method for controlled status transitions.

        Validates that the transition is allowed and logs warnings
        for invalid transitions.

        Args:
            new_status: The new status to set
        """
        valid_transitions = {
            JobStatus.PENDING: {JobStatus.RUNNING},
            JobStatus.RUNNING: {JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.RETRYING},
            JobStatus.RETRYING: {JobStatus.RUNNING, JobStatus.FAILED},
            JobStatus.SUCCESS: set(),  # Terminal state
            JobStatus.FAILED: set(),   # Terminal state
        }

        if new_status not in valid_transitions.get(self._status, set()):
            logger.warning(
                f"Invalid status transition for job {self.job_id}: "
                f"{self._status.value} -> {new_status.value}"
            )

        self._status = new_status

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """
        Execute the job's main logic.

        Must be implemented by subclasses.

        Returns:
            Dictionary with job execution stats (records_fetched, records_inserted, etc.)
        """
        pass

    def run(self) -> dict[str, Any]:
        """
        Run the job with retry logic and logging.

        Returns:
            Dictionary with job execution results
        """
        # Check data requirements first (preferred method)
        data_ok, data_error = self._check_data_requirements()
        if not data_ok:
            logger.error(data_error)
            self._log_failure(data_error)
            self._send_failure_notification(data_error)
            return {"status": "failed", "error": data_error}

        # Check job-based dependencies (legacy - only if no data_requirements)
        if not self.data_requirements and not self._check_dependencies():
            error_msg = f"Dependencies not met for job {self.job_id}"
            logger.error(error_msg)
            self._log_failure(error_msg)
            self._send_failure_notification(error_msg)
            return {"status": "failed", "error": error_msg}

        # Retry loop
        for attempt in range(self.max_retries + 1):
            self.retry_count = attempt
            try:
                # Start logging
                self._log_start()

                # Execute job
                logger.info(f"Starting job {self.job_id} (attempt {attempt + 1}/{self.max_retries + 1})")
                self._set_status(JobStatus.RUNNING)
                result = self.execute()

                # Success
                self._set_status(JobStatus.SUCCESS)
                self._log_success(result)
                logger.info(f"Job {self.job_id} completed successfully")
                return result

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                is_transient = isinstance(e, TRANSIENT_ERRORS)
                self.last_error = error_msg

                logger.error(
                    f"Job {self.job_id} failed (attempt {attempt + 1}): {error_type}: {e}",
                    extra={"error_type": error_type, "transient": is_transient},
                )

                # Only retry transient errors (network/IO issues)
                if attempt < self.max_retries and is_transient:
                    self._set_status(JobStatus.RETRYING)
                    backoff_time = self.retry_backoff ** attempt
                    logger.warning(f"Retrying job {self.job_id} in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                else:
                    # Final failure (or non-transient error - fail immediately)
                    self._set_status(JobStatus.FAILED)
                    self._log_failure(error_msg)
                    self._send_failure_notification(error_msg)
                    if is_transient:
                        logger.error(f"Job {self.job_id} failed after {self.max_retries + 1} attempts")
                    else:
                        logger.error(
                            f"Job {self.job_id} failed with non-transient error (no retry): {error_type}"
                        )
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "error_type": error_type,
                        "retry_count": self.retry_count,
                    }

        # Should not reach here, but just in case
        return {"status": "failed", "error": "Unknown error"}

    def _check_dependencies(self) -> bool:
        """
        Check if all dependency jobs have completed successfully.

        Returns:
            True if all dependencies are met, False otherwise
        """
        if not self.dependencies:
            return True

        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()

        for dep_job_id in self.dependencies:
            # Get the most recent run of the dependency job
            result = api.fetchone_readonly(
                """
                SELECT status, completed_at
                FROM ingestion_log
                WHERE source_id = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (dep_job_id,),
            )

            if not result:
                logger.warning(f"Dependency {dep_job_id} has never run")
                return False

            status, completed_at = result
            if status != "completed":
                logger.warning(f"Dependency {dep_job_id} did not complete successfully (status: {status})")
                return False

            if completed_at is None:
                logger.warning(f"Dependency {dep_job_id} has not completed yet")
                return False

        logger.info(f"All dependencies met for job {self.job_id}")
        return True

    def _check_data_requirements(self) -> tuple[bool, str | None]:
        """
        Check if all data requirements are met.

        This is the preferred way to check dependencies - it verifies
        that the actual data exists rather than checking if a job ran.

        Returns:
            Tuple of (all_met, error_message)
        """
        if not self.data_requirements:
            return True, None

        failed_requirements = []
        for req in self.data_requirements:
            is_met, message = req.check()
            if not is_met:
                logger.warning(f"Data requirement not met: {message}")
                failed_requirements.append(message)
            else:
                logger.debug(f"Data requirement met: {message}")

        if failed_requirements:
            error_msg = "Data requirements not met: " + "; ".join(failed_requirements)
            return False, error_msg

        logger.info(f"All data requirements met for job {self.job_id}")
        return True, None

    def _log_start(self) -> None:
        """Log job start to ingestion_log table."""
        try:
            self.log_id = self.api.log_job_start(self.job_id)
            if self.log_id:
                logger.debug(f"Created ingestion log entry {self.log_id} for job {self.job_id}")
            else:
                logger.error(f"INSERT returned no result for job {self.job_id}")
        except Exception as e:
            logger.error(f"Failed to create ingestion log for {self.job_id}: {e}")
            self.log_id = None

    def _log_success(self, result: dict[str, Any]) -> None:
        """
        Log successful job completion.

        Args:
            result: Job execution result dictionary
        """
        if self.log_id is None:
            logger.warning(f"No log_id for job {self.job_id}, cannot update log")
            return

        self.api.log_job_success(
            log_id=self.log_id,
            records_fetched=result.get("records_fetched", 0),
            records_inserted=result.get("records_inserted", 0),
        )
        logger.debug(f"Updated ingestion log {self.log_id} with success status")

    def _log_failure(self, error_msg: str) -> None:
        """
        Log job failure.

        Args:
            error_msg: Error message to log
        """
        if self.log_id is None:
            self.log_id = self.api.log_job_failure(error_msg, job_id=self.job_id)
            return

        self.api.log_job_failure(error_msg, log_id=self.log_id)
        logger.debug(f"Updated ingestion log {self.log_id} with failure status")

    def _send_failure_notification(self, error_msg: str) -> None:
        """
        Send email notification for job failure.

        Args:
            error_msg: Error message to include in notification
        """
        try:
            notifier = EmailNotifier()
            notifier.send_failure_notification(
                job_name=self.job_id,
                error_message=error_msg,
                retry_count=self.retry_count,
                max_retries=self.max_retries,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            # Don't let notification failures break the job, but log as error
            # so operators see it in monitoring
            logger.error(
                f"ALERT: Failed to send failure notification for job {self.job_id}: {e}",
                extra={"notification_failed": True, "job_id": self.job_id},
            )

    def get_last_successful_run(self) -> datetime | None:
        """
        Get the timestamp of the last successful run of this job.

        Returns:
            Datetime of last successful run, or None if never succeeded
        """
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        result = api.fetchone_readonly(
            """
            SELECT completed_at
            FROM ingestion_log
            WHERE source_id = ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            (self.job_id,),
        )

        if result and result[0]:
            return result[0]
        return None

    def get_status(self) -> dict[str, Any]:
        """
        Get current job status.

        Returns:
            Dictionary with job status information
        """
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "last_successful_run": self.get_last_successful_run(),
            "dependencies": self.dependencies,
        }

    def __repr__(self) -> str:
        """String representation of the job."""
        return f"<{self.__class__.__name__} id={self.job_id} status={self.status.value}>"


class PriceIngestionJob(IngestionJob):
    """
    Scheduled job for daily price data ingestion.

    Wraps the ingest_prices() function with retry logic and logging.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        source: str = "yfinance",
        job_id: str = "price_ingestion",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize price ingestion job.

        Args:
            symbols: List of stock tickers to ingest (default: universe symbols)
            start: Start date (default: yesterday)
            end: End date (default: today)
            source: Data source to use (default: 'yfinance')
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies)
        self._symbols_override = symbols
        self.symbols = symbols
        self.start = start or (date.today() - timedelta(days=1))
        self.end = end or date.today()
        self.source = source

    def execute(self) -> dict[str, Any]:
        """
        Execute price data ingestion.

        Returns:
            Dictionary with job execution stats
        """
        # Resolve symbols from universe if no explicit override was provided
        if self._symbols_override is None:
            um = UniverseManager()
            self.symbols = um.get_universe_at_date(date.today())
            if not self.symbols:
                raise RuntimeError("Universe is empty — cannot run price ingestion without symbols")
            logger.info(f"Price ingestion using {len(self.symbols)} symbols from universe")

        logger.info(
            f"Ingesting prices for {len(self.symbols)} symbols from {self.start} to {self.end}"
        )

        # Call the underlying ingest_prices function
        result = ingest_prices(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            source=self.source,
        )

        # Convert to standardized format expected by base class logging
        return {
            "records_fetched": result["rows_fetched"],
            "records_inserted": result["rows_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
        }


class FeatureComputationJob(IngestionJob):
    """
    Scheduled job for feature computation from price data.

    Wraps the compute_features() function with retry logic and logging.
    Requires recent price data to exist in the database.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        lookback_days: int = 252,
        version: str = "v1",
        job_id: str = "feature_computation",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        max_price_age_days: int = 7,
    ):
        """
        Initialize feature computation job.

        Args:
            symbols: List of stock tickers to compute features for (None = all symbols in database)
            start: Start date (default: 30 days ago)
            end: End date (default: today)
            lookback_days: Days of price history needed for computation (default: 252)
            version: Feature version identifier (default: 'v1')
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            max_price_age_days: Maximum age of price data in days (default: 7)
        """
        # Use data requirements instead of job-based dependencies
        data_requirements = [
            DataRequirement(
                table="prices",
                min_rows=1000,  # Need substantial price history
                max_age_days=max_price_age_days,
                date_column="date",
                description="Price data",
            )
        ]

        super().__init__(
            job_id,
            max_retries,
            retry_backoff,
            dependencies=None,  # No job-based dependencies
            data_requirements=data_requirements,
        )
        self.symbols = symbols
        self.start = start or (date.today() - timedelta(days=30))
        self.end = end or date.today()
        self.lookback_days = lookback_days
        self.version = version

    def execute(self) -> dict[str, Any]:
        """
        Execute feature computation using vectorized batch method.

        Returns:
            Dictionary with job execution stats
        """
        logger.info(
            f"Computing features (vectorized batch) from {self.start} to {self.end} "
            f"(lookback: {self.lookback_days} days)"
        )

        # Use the optimized batch computation function
        from hrp.data.ingestion.features import compute_features_batch

        result = compute_features_batch(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            lookback_days=self.lookback_days,
            version=self.version,
        )

        # Convert to standardized format expected by base class logging
        return {
            "records_fetched": result["features_computed"],
            "records_inserted": result["rows_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
        }


class UniverseUpdateJob(IngestionJob):
    """
    Scheduled job for S&P 500 universe updates.

    Fetches current S&P 500 constituents from Wikipedia, applies exclusion
    rules, and updates the universe table with membership changes.
    """

    def __init__(
        self,
        as_of_date: date | None = None,
        actor: str = "system:scheduled_universe_update",
        job_id: str = "universe_update",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize universe update job.

        Args:
            as_of_date: Date to record for this universe snapshot (default: today)
            actor: Actor to record in lineage (default: 'system:scheduled_universe_update')
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies)
        self.as_of_date = as_of_date or date.today()
        self.actor = actor

    def execute(self) -> dict[str, Any]:
        """
        Execute universe update.

        Returns:
            Dictionary with job execution stats
        """
        logger.info(f"Updating S&P 500 universe as of {self.as_of_date}")

        # Create universe manager and run update
        manager = UniverseManager()
        result = manager.update_universe(
            as_of_date=self.as_of_date,
            actor=self.actor,
        )

        # Convert to standardized format expected by base class logging
        # For universe updates, "fetched" = total constituents, "inserted" = included symbols
        return {
            "records_fetched": result["total_constituents"],
            "records_inserted": result["included"],
            "symbols_added": result["added"],
            "symbols_removed": result["removed"],
            "symbols_excluded": result["excluded"],
            "exclusion_breakdown": result["exclusion_reasons"],
        }


class FundamentalsIngestionJob(IngestionJob):
    """
    Scheduled job for weekly fundamentals data ingestion.

    Wraps the ingest_fundamentals() function with retry logic and logging.
    Fetches revenue, EPS, book value, and other fundamental data with
    point-in-time correctness for backtesting.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        metrics: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        source: str = "simfin",
        job_id: str = "fundamentals_ingestion",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize fundamentals ingestion job.

        Args:
            symbols: List of stock tickers to ingest (None = all universe symbols)
            metrics: List of metrics to fetch (None = all default metrics)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            source: Data source ('simfin' or 'yfinance', default: 'simfin')
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.metrics = metrics
        self.start_date = start_date
        self.end_date = end_date
        self.source = source

    def execute(self) -> dict[str, Any]:
        """
        Execute fundamentals data ingestion.

        Returns:
            Dictionary with job execution stats
        """
        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                raise RuntimeError("Universe is empty — cannot run fundamentals ingestion without symbols")
            logger.info(f"Using {len(symbols)} symbols from universe")

        logger.info(
            f"Ingesting fundamentals for {len(symbols)} symbols using {self.source}"
        )

        # Call the underlying ingest_fundamentals function
        result = ingest_fundamentals(
            symbols=symbols,
            metrics=self.metrics,
            start_date=self.start_date,
            end_date=self.end_date,
            source=self.source,
        )

        # Convert to standardized format expected by base class logging
        return {
            "records_fetched": result["records_fetched"],
            "records_inserted": result["records_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
            "fallback_used": result.get("fallback_used", 0),
            "pit_violations_filtered": result.get("pit_violations_filtered", 0),
        }


class SnapshotFundamentalsJob(IngestionJob):
    """
    Scheduled job for weekly snapshot fundamentals ingestion.

    Fetches current fundamental metrics (P/E ratio, P/B ratio, market cap,
    dividend yield, EV/EBITDA) from Yahoo Finance. Unlike quarterly fundamentals,
    these are point-in-time snapshots stored in the features table.

    Recommended schedule: Weekly (fundamentals don't change daily)
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        as_of_date: date | None = None,
        job_id: str = "snapshot_fundamentals",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize snapshot fundamentals job.

        Args:
            symbols: List of stock tickers to ingest (None = all universe symbols)
            as_of_date: Date to record for these values (default: today)
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.as_of_date = as_of_date or date.today()

    def execute(self) -> dict[str, Any]:
        """
        Execute snapshot fundamentals ingestion.

        Returns:
            Dictionary with job execution stats
        """
        from hrp.data.ingestion.fundamentals import ingest_snapshot_fundamentals

        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                raise RuntimeError("Universe is empty — cannot run snapshot fundamentals ingestion without symbols")
            logger.info(f"Using {len(symbols)} symbols from universe")

        logger.info(
            f"Ingesting snapshot fundamentals for {len(symbols)} symbols as of {self.as_of_date}"
        )

        # Call the underlying ingest_snapshot_fundamentals function
        result = ingest_snapshot_fundamentals(
            symbols=symbols,
            as_of_date=self.as_of_date,
        )

        # Convert to standardized format expected by base class logging
        return {
            "records_fetched": result["records_fetched"],
            "records_inserted": result["records_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
        }


class SnapshotFundamentalsBackfillJob(IngestionJob):
    """
    Job to backfill snapshot fundamentals across historical trading days.

    Replicates current snapshot fundamental values (market_cap, pe_ratio, etc.)
    across a date range to provide historical coverage for backtesting.

    Note: This uses current values as an approximation for historical values.
    For true point-in-time accuracy, use quarterly fundamentals.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        job_id: str = "snapshot_fundamentals_backfill",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize snapshot fundamentals backfill job.

        Args:
            symbols: List of stock tickers (None = all universe symbols)
            start_date: Start date for backfill (default: 365 days ago)
            end_date: End date for backfill (default: today)
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.start_date = start_date or (date.today() - timedelta(days=365))
        self.end_date = end_date or date.today()

    def execute(self) -> dict[str, Any]:
        """
        Execute snapshot fundamentals backfill.

        Returns:
            Dictionary with job execution stats
        """
        from hrp.data.ingestion.fundamentals import backfill_snapshot_fundamentals

        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                raise RuntimeError("Universe is empty — cannot run snapshot fundamentals backfill without symbols")
            logger.info(f"Using {len(symbols)} symbols from universe")

        logger.info(
            f"Backfilling snapshot fundamentals for {len(symbols)} symbols "
            f"from {self.start_date} to {self.end_date}"
        )

        result = backfill_snapshot_fundamentals(
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        return {
            "records_fetched": result["records_fetched"],
            "records_inserted": result["records_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
            "trading_days": result.get("trading_days", 0),
        }


class FundamentalsTimeSeriesJob(IngestionJob):
    """
    Scheduled job for daily fundamentals time-series ingestion.

    Runs weekly to update fundamental time-series with latest quarterly data.
    Recommended schedule: Sunday 6 AM ET (before market open).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        as_of_date: date | None = None,
        lookback_days: int = 90,
        include_valuation_metrics: bool = False,
        job_id: str = "fundamentals_timeseries",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize fundamentals time-series job.

        Args:
            symbols: List of stock tickers (None = all universe symbols)
            as_of_date: Date to compute time-series as of (default: today)
            lookback_days: Days to backfill for point-in-time correctness
            include_valuation_metrics: If True, also backfill valuation metrics
                (ts_market_cap, ts_pe_ratio, etc.)
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.as_of_date = as_of_date or date.today()
        self.lookback_days = lookback_days
        self.include_valuation_metrics = include_valuation_metrics

    def execute(self) -> dict[str, Any]:
        """
        Execute fundamentals time-series ingestion.

        Returns:
            Dictionary with job execution stats
        """
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries
        from hrp.data.universe import UniverseManager

        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                symbols = ["AAPL", "MSFT", "GOOGL"]
            logger.info(f"Using {len(symbols)} symbols from universe")

        # Compute time-series for lookback period
        start = self.as_of_date - timedelta(days=self.lookback_days)

        logger.info(
            f"Computing fundamentals time-series for {len(symbols)} symbols "
            f"from {start} to {self.as_of_date}"
        )

        result = backfill_fundamentals_timeseries(
            symbols=symbols,
            start=start,
            end=self.as_of_date,
            batch_size=10,
            include_valuation_metrics=self.include_valuation_metrics,
        )

        return {
            "records_fetched": result["rows_inserted"],
            "records_inserted": result["rows_inserted"],
            "symbols_success": result["symbols_success"],
            "symbols_failed": result["symbols_failed"],
            "failed_symbols": result.get("failed_symbols", []),
            "valuation_rows_inserted": result.get("valuation_rows_inserted", 0),
        }


class ComprehensiveFundamentalsBackfillJob(IngestionJob):
    """
    Comprehensive fundamentals backfill job that runs all fundamental
    data pipelines in sequence.

    Runs:
    1. Quarterly fundamentals ingestion (revenue, eps, book_value, etc.)
    2. Snapshot fundamentals backfill (market_cap, pe_ratio, etc.)
    3. Time-series generation (ts_revenue, ts_eps, ts_market_cap, etc.)

    This provides complete fundamental data coverage for backtesting.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 365,
        job_id: str = "comprehensive_fundamentals_backfill",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize comprehensive fundamentals backfill job.

        Args:
            symbols: List of stock tickers (None = all universe symbols)
            lookback_days: Days of history to backfill
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies or [])
        self.symbols = symbols
        self.lookback_days = lookback_days

    def execute(self) -> dict[str, Any]:
        """
        Execute comprehensive fundamentals backfill.

        Runs three stages in sequence:
        1. Quarterly fundamentals ingestion
        2. Snapshot fundamentals backfill
        3. Time-series generation (including valuation metrics)

        Returns:
            Dictionary with job execution stats for all stages
        """
        from hrp.data.ingestion.fundamentals import (
            backfill_snapshot_fundamentals,
            ingest_fundamentals,
            ingest_snapshot_fundamentals,
        )
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries

        # Get symbols from universe if not specified
        symbols = self.symbols
        if symbols is None:
            manager = UniverseManager()
            symbols = manager.get_universe_at_date(date.today())
            if not symbols:
                raise RuntimeError("Universe is empty — cannot run comprehensive fundamentals backfill without symbols")
            logger.info(f"Using {len(symbols)} symbols from universe")

        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_days)

        stats: dict[str, Any] = {
            "symbols_requested": len(symbols),
            "stages_completed": 0,
            "quarterly_stats": {},
            "snapshot_stats": {},
            "timeseries_stats": {},
            "records_inserted": 0,
        }

        # Stage 1: Quarterly fundamentals ingestion
        logger.info("Stage 1/3: Ingesting quarterly fundamentals")
        try:
            quarterly_result = ingest_fundamentals(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                source="yfinance",  # Use yfinance as SimFin may not be available
            )
            stats["quarterly_stats"] = quarterly_result
            stats["records_inserted"] += quarterly_result.get("records_inserted", 0)
            stats["stages_completed"] += 1
            logger.info(f"Quarterly fundamentals: {quarterly_result.get('records_inserted', 0)} records")
        except Exception as e:
            logger.error(f"Quarterly fundamentals failed: {e}")
            stats["quarterly_stats"] = {"error": str(e)}

        # Stage 2: Snapshot fundamentals (current + backfill)
        logger.info("Stage 2/3: Ingesting and backfilling snapshot fundamentals")
        try:
            # First get current snapshot values
            snapshot_current = ingest_snapshot_fundamentals(
                symbols=symbols,
                as_of_date=end_date,
            )
            logger.info(f"Current snapshot: {snapshot_current.get('records_inserted', 0)} records")

            # Then backfill historical
            snapshot_backfill = backfill_snapshot_fundamentals(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            stats["snapshot_stats"] = {
                "current_records": snapshot_current.get("records_inserted", 0),
                "backfill_records": snapshot_backfill.get("records_inserted", 0),
                "trading_days": snapshot_backfill.get("trading_days", 0),
            }
            stats["records_inserted"] += snapshot_backfill.get("records_inserted", 0)
            stats["stages_completed"] += 1
            logger.info(f"Snapshot backfill: {snapshot_backfill.get('records_inserted', 0)} records")
        except Exception as e:
            logger.error(f"Snapshot fundamentals failed: {e}")
            stats["snapshot_stats"] = {"error": str(e)}

        # Stage 3: Time-series generation (quarterly + valuation)
        logger.info("Stage 3/3: Generating fundamentals time-series")
        try:
            timeseries_result = backfill_fundamentals_timeseries(
                symbols=symbols,
                start=start_date,
                end=end_date,
                include_valuation_metrics=True,  # Include ts_market_cap, ts_pe_ratio, etc.
            )
            stats["timeseries_stats"] = timeseries_result
            stats["records_inserted"] += timeseries_result.get("rows_inserted", 0)
            stats["stages_completed"] += 1
            logger.info(
                f"Time-series: {timeseries_result.get('rows_inserted', 0)} records "
                f"(including {timeseries_result.get('valuation_rows_inserted', 0)} valuation)"
            )
        except Exception as e:
            logger.error(f"Time-series generation failed: {e}")
            stats["timeseries_stats"] = {"error": str(e)}

        logger.info(
            f"Comprehensive fundamentals backfill complete: "
            f"{stats['stages_completed']}/3 stages, {stats['records_inserted']} total records"
        )

        return stats


class IntradayIngestionJob(IngestionJob):
    """
    Scheduled job for real-time intraday data ingestion via WebSocket.

    Unlike other ingestion jobs, this is a long-running service that streams
    data during market hours (9:30 AM - 4:00 PM ET).

    Usage:
        - Schedule start at 9:25 AM ET (5 min before open)
        - Schedule stop at 4:05 PM ET (5 min after close)
        - Runs only on weekdays (market days)
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        channels: list[str] | None = None,
        job_id: str = "intraday_ingestion",
        max_retries: int = 0,  # No retries for long-running service
        data_requirements: list[DataRequirement] | None = None,
    ):
        """
        Initialize intraday ingestion job.

        Args:
            symbols: List of tickers to stream (None = top 50 from universe)
            channels: WebSocket channels to subscribe (default: ['AM'] for minute bars)
            job_id: Unique identifier for this job
            max_retries: Maximum retry attempts (default 0 for market-hours service)
            data_requirements: Data availability requirements (symbols table must exist)
        """
        # Default data requirement: symbols table must have rows
        if data_requirements is None:
            data_requirements = [
                DataRequirement(
                    table="symbols",
                    min_rows=1,
                    description="Universe symbols",
                )
            ]

        super().__init__(job_id, max_retries, retry_backoff=0, data_requirements=data_requirements)
        self._symbols_override = symbols
        self.symbols = symbols
        self.channels = channels or ["AM"]
        self.service: IntradayIngestionService | None = None

    def execute(self) -> dict[str, Any]:
        """
        Execute intraday data ingestion service.

        This method starts the WebSocket client and runs until market close
        or manual stop. It blocks during market hours.

        Returns:
            Dictionary with session stats (bars_received, bars_written, etc.)
        """
        # Resolve symbols from universe if not explicitly provided
        if self._symbols_override is None:
            um = UniverseManager()
            universe = um.get_universe_at_date(date.today())
            if not universe:
                raise RuntimeError("Universe is empty — cannot stream intraday data without symbols")

            # Get top 50 symbols by volume (config.realtime.symbols or default)
            from hrp.utils.config import get_config
            config = get_config()

            if config.realtime.symbols:
                self.symbols = config.realtime.symbols
            else:
                # Default: top 50 from universe (sorted by recent volume)
                # For simplicity, just take first 50 from universe
                self.symbols = universe[:50]

            logger.info(f"Intraday ingestion using {len(self.symbols)} symbols from universe")

        logger.info(
            f"Starting intraday ingestion for {len(self.symbols)} symbols, channels: {self.channels}"
        )

        # Create and start ingestion service
        from hrp.data.connection_pool import ConnectionPool
        from hrp.utils.config import get_config

        config = get_config()
        conn_pool = ConnectionPool(str(config.data.db_path))

        self.service = IntradayIngestionService(
            conn_pool=conn_pool,
            flush_interval=config.realtime.buffer_flush_interval_seconds,
            max_buffer_size=config.realtime.max_buffer_size,
        )

        # Start streaming
        self.service.start(symbols=self.symbols, channels=self.channels)

        # Block until market close or manual stop
        # In production, this would be stopped by scheduler at 4:05 PM ET
        # For now, we'll let the service run and rely on external stop signal
        logger.info(
            f"Intraday ingestion service running. "
            f"Stats: {self.service.get_stats()}"
        )

        # Return current stats (service still running)
        stats = self.service.get_stats()
        return {
            "bars_received": stats["bars_received"],
            "bars_written": stats["bars_written"],
            "session_start": stats["session_start"],
            "is_connected": stats["is_connected"],
        }

    def stop(self) -> dict[str, Any]:
        """
        Gracefully stop the intraday ingestion service.

        Should be called at market close (4:05 PM ET).

        Returns:
            Final session statistics
        """
        if self.service is None:
            logger.warning("Cannot stop intraday service — not started")
            return {}

        logger.info("Stopping intraday ingestion service...")
        final_stats = self.service.stop()

        # Log final stats to lineage (if needed, can be done in scheduler)
        logger.info(
            f"Intraday ingestion session complete: "
            f"{final_stats['bars_received']} bars received, "
            f"{final_stats['bars_written']} bars written, "
            f"{final_stats['session_duration_seconds']:.0f}s duration"
        )

        return final_stats

    def run(self) -> dict[str, Any]:
        """
        Override run() to handle long-running nature of this job.

        Unlike batch jobs, this doesn't use retry logic since it's a
        continuous service during market hours.
        """
        # Check data requirements
        data_ok, data_error = self._check_data_requirements()
        if not data_ok:
            logger.error(data_error)
            self._log_failure(data_error)
            return {"status": "failed", "error": data_error}

        try:
            # Start logging
            self._log_start()

            # Execute (starts WebSocket service)
            logger.info(f"Starting intraday ingestion service: {self.job_id}")
            self._set_status(JobStatus.RUNNING)
            result = self.execute()

            # Note: Service is still running at this point
            # Success logging will happen when stop() is called
            logger.info(f"Intraday ingestion service started successfully")
            return result

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            self.last_error = error_msg

            logger.error(
                f"Intraday ingestion job failed: {error_type}: {e}",
                exc_info=True,
            )

            self._set_status(JobStatus.FAILED)
            self._log_failure(error_msg)
            self._send_failure_notification(error_msg)

            return {"status": "failed", "error": error_msg}
