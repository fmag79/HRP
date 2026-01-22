"""
Job orchestration for HRP data ingestion pipeline.

Base classes for scheduled jobs with dependency management, retry logic,
and logging to the ingestion_log table.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from hrp.data.db import get_db


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
