"""
Data cleanup for HRP.

Provides scheduled cleanup of orphaned records and old logs.
"""

from hrp.data.retention.cleanup import DataCleanupJob

__all__ = [
    "DataCleanupJob",
]
