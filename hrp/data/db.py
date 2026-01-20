"""
DuckDB connection management for HRP.

Provides thread-safe connection pooling and query helpers.
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import duckdb
from loguru import logger

# Default data directory
DEFAULT_DATA_DIR = Path.home() / "hrp-data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "hrp.duckdb"


class DatabaseManager:
    """
    Thread-safe DuckDB connection manager.

    Uses a connection pool to handle concurrent access safely.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Path | str | None = None):
        """Singleton pattern for database manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the database manager."""
        if self._initialized:
            return

        self.db_path = Path(db_path) if db_path else self._get_db_path()
        self._local = threading.local()
        self._initialized = True

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Database manager initialized: {self.db_path}")

    def _get_db_path(self) -> Path:
        """Get database path from environment or default."""
        data_dir = os.getenv("HRP_DATA_DIR", str(DEFAULT_DATA_DIR))
        return Path(data_dir).expanduser() / "hrp.duckdb"

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = duckdb.connect(str(self.db_path))
            logger.debug(f"Created new connection for thread {threading.current_thread().name}")
        return self._local.connection

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Context manager for database connection.

        Usage:
            with db.connection() as conn:
                result = conn.execute("SELECT * FROM prices").fetchall()
        """
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise

    def execute(self, query: str, params: tuple | None = None) -> duckdb.DuckDBPyRelation:
        """Execute a query and return the relation."""
        conn = self._get_connection()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)

    def fetchall(self, query: str, params: tuple | None = None) -> list[tuple]:
        """Execute a query and fetch all results."""
        return self.execute(query, params).fetchall()

    def fetchone(self, query: str, params: tuple | None = None) -> tuple | None:
        """Execute a query and fetch one result."""
        return self.execute(query, params).fetchone()

    def fetchdf(self, query: str, params: tuple | None = None) -> Any:
        """Execute a query and return a pandas DataFrame."""
        return self.execute(query, params).df()

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None
            logger.debug(f"Closed connection for thread {threading.current_thread().name}")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None


# Convenience function for quick access
def get_db(db_path: Path | str | None = None) -> DatabaseManager:
    """Get the database manager instance."""
    return DatabaseManager(db_path)
