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

from hrp.data.db import get_db
from hrp.data.ingestion.features import compute_features
from hrp.data.ingestion.prices import TEST_SYMBOLS, ingest_prices
from hrp.notifications.email import EmailNotifier


class JobStatus(Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class IngestionJob(ABC):
    """
    Base class for scheduled data ingestion jobs.

    Features:
    - Automatic logging to ingestion_log table
    - Retry logic with exponential backoff
    - Dependency management
    - Status tracking
    """

    def __init__(
        self,
        job_id: str,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize an ingestion job.

        Args:
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        self.job_id = job_id
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.dependencies = dependencies or []
        self.status = JobStatus.PENDING
        self.log_id: int | None = None
        self.retry_count = 0
        self.last_error: str | None = None

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
        # Check dependencies
        if not self._check_dependencies():
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
                self.status = JobStatus.RUNNING
                result = self.execute()

                # Success
                self.status = JobStatus.SUCCESS
                self._log_success(result)
                logger.info(f"Job {self.job_id} completed successfully")
                return result

            except Exception as e:
                error_msg = str(e)
                self.last_error = error_msg
                logger.error(f"Job {self.job_id} failed (attempt {attempt + 1}): {e}")

                # Check if we should retry
                if attempt < self.max_retries:
                    self.status = JobStatus.RETRYING
                    backoff_time = self.retry_backoff ** attempt
                    logger.warning(f"Retrying job {self.job_id} in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                else:
                    # Final failure
                    self.status = JobStatus.FAILED
                    self._log_failure(error_msg)
                    self._send_failure_notification(error_msg)
                    logger.error(f"Job {self.job_id} failed after {self.max_retries + 1} attempts")
                    return {
                        "status": "failed",
                        "error": error_msg,
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

        db = get_db()
        with db.connection() as conn:
            for dep_job_id in self.dependencies:
                # Get the most recent run of the dependency job
                result = conn.execute(
                    """
                    SELECT status, completed_at
                    FROM ingestion_log
                    WHERE source_id = ?
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (dep_job_id,),
                ).fetchone()

                if not result:
                    logger.warning(f"Dependency {dep_job_id} has never run")
                    return False

                status, completed_at = result
                if status != "success":
                    logger.warning(f"Dependency {dep_job_id} did not complete successfully (status: {status})")
                    return False

                if completed_at is None:
                    logger.warning(f"Dependency {dep_job_id} has not completed yet")
                    return False

        logger.info(f"All dependencies met for job {self.job_id}")
        return True

    def _log_start(self) -> None:
        """Log job start to ingestion_log table."""
        db = get_db()
        with db.connection() as conn:
            result = conn.execute(
                """
                INSERT INTO ingestion_log (source_id, started_at, status)
                VALUES (?, CURRENT_TIMESTAMP, 'running')
                RETURNING log_id
                """,
                (self.job_id,),
            ).fetchone()

            if result:
                self.log_id = result[0]
                logger.debug(f"Created ingestion log entry {self.log_id} for job {self.job_id}")

    def _log_success(self, result: dict[str, Any]) -> None:
        """
        Log successful job completion.

        Args:
            result: Job execution result dictionary
        """
        if self.log_id is None:
            logger.warning(f"No log_id for job {self.job_id}, cannot update log")
            return

        db = get_db()
        with db.connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_log
                SET completed_at = CURRENT_TIMESTAMP,
                    status = 'success',
                    records_fetched = ?,
                    records_inserted = ?,
                    error_message = NULL
                WHERE log_id = ?
                """,
                (
                    result.get("records_fetched", 0),
                    result.get("records_inserted", 0),
                    self.log_id,
                ),
            )
            logger.debug(f"Updated ingestion log {self.log_id} with success status")

    def _log_failure(self, error_msg: str) -> None:
        """
        Log job failure.

        Args:
            error_msg: Error message to log
        """
        if self.log_id is None:
            # Create a new log entry if we don't have one
            db = get_db()
            with db.connection() as conn:
                result = conn.execute(
                    """
                    INSERT INTO ingestion_log (source_id, started_at, status, error_message)
                    VALUES (?, CURRENT_TIMESTAMP, 'failed', ?)
                    RETURNING log_id
                    """,
                    (self.job_id, error_msg),
                ).fetchone()
                if result:
                    self.log_id = result[0]
            return

        db = get_db()
        with db.connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_log
                SET completed_at = CURRENT_TIMESTAMP,
                    status = 'failed',
                    error_message = ?
                WHERE log_id = ?
                """,
                (error_msg, self.log_id),
            )
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
            # Don't let notification failures break the job
            logger.warning(f"Failed to send failure notification for job {self.job_id}: {e}")

    def get_last_successful_run(self) -> datetime | None:
        """
        Get the timestamp of the last successful run of this job.

        Returns:
            Datetime of last successful run, or None if never succeeded
        """
        db = get_db()
        with db.connection() as conn:
            result = conn.execute(
                """
                SELECT completed_at
                FROM ingestion_log
                WHERE source_id = ? AND status = 'success'
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (self.job_id,),
            ).fetchone()

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
            symbols: List of stock tickers to ingest (default: TEST_SYMBOLS)
            start: Start date (default: yesterday)
            end: End date (default: today)
            source: Data source to use (default: 'yfinance')
            job_id: Unique identifier for this job
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier (seconds)
            dependencies: List of job IDs that must complete before this job runs
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies)
        self.symbols = symbols or TEST_SYMBOLS
        self.start = start or (date.today() - timedelta(days=1))
        self.end = end or date.today()
        self.source = source

    def execute(self) -> dict[str, Any]:
        """
        Execute price data ingestion.

        Returns:
            Dictionary with job execution stats
        """
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
    Depends on price ingestion completing successfully.
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
        dependencies: list[str] | None = None,
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
            dependencies: List of job IDs that must complete before this job runs
                         (default: ['price_ingestion'])
        """
        # Default dependency on price ingestion
        if dependencies is None:
            dependencies = ["price_ingestion"]

        super().__init__(job_id, max_retries, retry_backoff, dependencies)
        self.symbols = symbols
        self.start = start or (date.today() - timedelta(days=30))
        self.end = end or date.today()
        self.lookback_days = lookback_days
        self.version = version

    def execute(self) -> dict[str, Any]:
        """
        Execute feature computation.

        Returns:
            Dictionary with job execution stats
        """
        logger.info(
            f"Computing features from {self.start} to {self.end} "
            f"(lookback: {self.lookback_days} days)"
        )

        # Call the underlying compute_features function
        result = compute_features(
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
