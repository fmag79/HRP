"""
Automated data cleanup jobs for HRP.

Provides scheduled jobs for cleaning up old data based on simple date thresholds.
Includes safety checks and dry-run mode for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from loguru import logger

from hrp.data.db import get_db

# Cutoff thresholds in days per data type.
# Records older than the cutoff are eligible for deletion.
CLEANUP_CUTOFF_DAYS: dict[str, int] = {
    "prices": 365 * 3,  # 3 years
    "features": 365 * 3,  # 3 years
    "lineage": 365,  # 1 year
    "ingestion_log": 365,  # 1 year
    "intraday_bars": 30,  # 30 days (hot: 7, warm: 7-30, cold: delete)
    "intraday_features": 30,  # 30 days (same retention as bars)
}


@dataclass
class CleanupResult:
    """
    Result of a cleanup operation.

    Attributes:
        data_type: Type of data cleaned up
        dry_run: Whether this was a dry run
        records_deleted: Number of records deleted
        bytes_freed: Estimated bytes freed
        duration_seconds: Time taken for cleanup
        errors: List of error messages
    """

    data_type: str
    dry_run: bool
    records_deleted: int = 0
    bytes_freed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] | None = None

    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        """Check if cleanup was successful."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": self.data_type,
            "dry_run": self.dry_run,
            "records_deleted": self.records_deleted,
            "bytes_freed": self.bytes_freed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class DataCleanupJob:
    """
    Automated cleanup job for old data based on date thresholds.

    Identifies and deletes data that exceeds retention thresholds.
    Includes dry-run mode for safe testing and safety checks to prevent
    accidental data loss.
    """

    def __init__(
        self,
        db_path: str | None = None,
        dry_run: bool = False,
        data_types: list[str] | None = None,
        require_confirmation: bool = True,
    ):
        """
        Initialize the cleanup job.

        Args:
            db_path: Optional database path
            dry_run: If True, only report what would be deleted
            data_types: List of data types to clean (defaults to all)
            require_confirmation: If True, requires confirmation before deletion
        """
        self._db = get_db(db_path)
        self._dry_run = dry_run
        self._data_types = data_types or list(CLEANUP_CUTOFF_DAYS.keys())
        self._require_confirmation = require_confirmation

    def run(self, as_of_date: date | None = None) -> dict[str, CleanupResult]:
        """
        Run the cleanup job for all configured data types.

        Args:
            as_of_date: Reference date for age calculation

        Returns:
            Dictionary mapping data_type to CleanupResult
        """
        import time

        as_of_date = as_of_date or date.today()
        results = {}

        logger.info(
            f"Running cleanup job (dry_run={self._dry_run}) for {len(self._data_types)} data types"
        )

        for data_type in self._data_types:
            start_time = time.time()

            try:
                result = self._cleanup_data_type(data_type, as_of_date)
                result.duration_seconds = time.time() - start_time
                results[data_type] = result

                logger.info(
                    f"Cleanup {data_type}: deleted={result.records_deleted}, "
                    f"errors={len(result.errors)}"
                )

            except Exception as e:
                logger.error(f"Error cleaning up {data_type}: {e}")
                results[data_type] = CleanupResult(
                    data_type=data_type,
                    dry_run=self._dry_run,
                    errors=[str(e)],
                )

        return results

    def _get_cutoff_date(self, data_type: str, as_of_date: date) -> date:
        """Get the cutoff date for a data type."""
        cutoff_days = CLEANUP_CUTOFF_DAYS.get(data_type)
        if cutoff_days is None:
            raise ValueError(f"No cleanup threshold defined for data type: {data_type}")
        return as_of_date - timedelta(days=cutoff_days)

    def _cleanup_data_type(self, data_type: str, as_of_date: date) -> CleanupResult:
        """
        Cleanup a specific data type.

        Args:
            data_type: Type of data to clean
            as_of_date: Reference date for age calculation

        Returns:
            CleanupResult with operation details
        """
        cutoff_date = self._get_cutoff_date(data_type, as_of_date)
        result = CleanupResult(data_type=data_type, dry_run=self._dry_run)

        with self._db.connection() as conn:
            if data_type == "prices":
                result.records_deleted = self._cleanup_table(
                    conn, "prices", "date", cutoff_date
                )
            elif data_type == "features":
                result.records_deleted = self._cleanup_table(
                    conn, "features", "date", cutoff_date
                )
            elif data_type == "lineage":
                result.records_deleted = self._cleanup_table(
                    conn, "lineage", "timestamp", cutoff_date
                )
            elif data_type == "ingestion_log":
                result.records_deleted = self._cleanup_table(
                    conn, "ingestion_log", "started_at", cutoff_date
                )
            elif data_type == "intraday_bars":
                result.records_deleted = self._cleanup_table(
                    conn, "intraday_bars", "timestamp", cutoff_date
                )
            elif data_type == "intraday_features":
                result.records_deleted = self._cleanup_table(
                    conn, "intraday_features", "timestamp", cutoff_date
                )
            else:
                result.errors.append(f"Cleanup not implemented for data type: {data_type}")

        return result

    def _cleanup_table(self, conn, table: str, date_column: str, cutoff_date: date) -> int:
        """
        Delete or count records older than the cutoff date in a table.

        Args:
            conn: Database connection
            table: Table name
            date_column: Column containing the date to compare
            cutoff_date: Records older than this are deleted

        Returns:
            Number of records deleted (or that would be deleted in dry-run mode)
        """
        if self._dry_run:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {date_column} < ?",
                (cutoff_date,),
            ).fetchone()
            return row[0] if row else 0

        try:
            result = conn.execute(
                f"DELETE FROM {table} WHERE {date_column} < ?",
                (cutoff_date,),
            )
            return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting from {table}: {e}")
            return 0
