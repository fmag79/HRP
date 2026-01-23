"""
DuckDB connection management for HRP.

Provides thread-safe connection pooling and query helpers.

Read-Only Support:
    The connection manager supports read-only connections via `get_db(read_only=True)`.
    However, DuckDB does not allow mixing read-only and read-write connections
    to the same database file within the same process. Use read-only mode:
    - In separate processes/scripts that only need to read data
    - To prevent accidental writes in read-only analysis scripts
    - Note: Read-only connections still use file-level locking and won't allow
      concurrent access from multiple processes (DuckDB limitation)

    Example:
        # Default: read-write (for normal operations)
        db = get_db()
        with db.connection() as conn:
            conn.execute("INSERT INTO prices VALUES (...)")

        # Read-only (use in separate process/script)
        db_ro = get_db(read_only=True)
        with db_ro.connection() as conn:
            result = conn.execute("SELECT * FROM prices").fetchall()
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Union

import duckdb
from loguru import logger

# Default data directory
DEFAULT_DATA_DIR = Path.home() / "hrp-data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "hrp.duckdb"


class MaterializedRelation:
    """
    A relation-like object that holds materialized query results.

    Used to return query results that don't depend on an active connection.
    Provides the same API as DuckDBPyRelation for backward compatibility.
    """

    def __init__(self, rows: List[Tuple[Any, ...]]):
        """Initialize with materialized rows."""
        self._rows = rows
        self._index = 0

    def fetchall(self) -> List[Tuple[Any, ...]]:
        """Fetch all rows."""
        return self._rows

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """Fetch one row."""
        if self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            return row
        return None

    def fetchmany(self, size: int = 1) -> List[Tuple[Any, ...]]:
        """Fetch multiple rows."""
        result = self._rows[self._index : self._index + size]
        self._index += len(result)
        return result


class ConnectionPool:
    """
    Thread-safe connection pool for DuckDB.

    Manages a pool of reusable connections with configurable max size.
    Connections are created on-demand up to max_size and reused when released.
    Supports both read-write and read-only connections.
    """

    def __init__(
        self, db_path: Union[str, Path], max_size: int = 5, idle_timeout: int = 300, read_only: bool = False
    ):
        """
        Initialize connection pool.

        Args:
            db_path: Path to the DuckDB database file
            max_size: Maximum number of connections in the pool
            idle_timeout: Seconds before idle connections are closed (default: 300)
            read_only: If True, all connections will be read-only (default: False)
        """
        self.db_path = str(db_path)
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.read_only = read_only
        self._pool: List[duckdb.DuckDBPyConnection] = []
        self._in_use: Set[duckdb.DuckDBPyConnection] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        mode_str = "read-only" if read_only else "read-write"
        logger.debug(f"ConnectionPool initialized: {self.db_path}, max_size={self.max_size}, mode={mode_str}")

    def _is_connection_valid(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """
        Check if a connection is still valid and usable.

        Args:
            conn: The connection to check

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try a simple query to verify connection works
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            logger.debug(f"Connection validation failed: {e}")
            return False

    def acquire(self) -> duckdb.DuckDBPyConnection:
        """
        Acquire a connection from the pool.

        Waits if all connections are in use and max_size is reached.
        Validates connections from pool and creates new ones if invalid.

        Returns:
            A DuckDB connection
        """
        with self._condition:
            # Wait if pool is exhausted and at max size
            while len(self._pool) == 0 and len(self._in_use) >= self.max_size:
                logger.debug("Connection pool exhausted, waiting...")
                self._condition.wait()

            # Get connection from pool or create new one
            conn = None
            if self._pool:
                # Try to get a valid connection from pool
                while self._pool:
                    candidate = self._pool.pop()
                    if self._is_connection_valid(candidate):
                        conn = candidate
                        logger.debug(f"Reusing connection from pool (pool size: {len(self._pool)})")
                        break
                    else:
                        # Connection is invalid, close it and try next
                        logger.debug("Discarding invalid connection from pool")
                        try:
                            candidate.close()
                        except Exception:
                            pass  # Ignore errors when closing invalid connection

            # Create new connection if we didn't get a valid one from pool
            if conn is None:
                if self.read_only:
                    conn = duckdb.connect(self.db_path, read_only=True)
                else:
                    conn = duckdb.connect(self.db_path)
                mode_str = "read-only" if self.read_only else "read-write"
                logger.debug(
                    f"Created new {mode_str} connection to {self.db_path} "
                    f"(in use: {len(self._in_use) + 1}/{self.max_size})"
                )

            self._in_use.add(conn)
            return conn

    def release(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Release a connection back to the pool.

        Validates connection health before returning to pool.
        Invalid connections are closed and discarded.

        Args:
            conn: The connection to release
        """
        with self._condition:
            if conn in self._in_use:
                self._in_use.remove(conn)

                # Check if connection is still valid
                if self._is_connection_valid(conn):
                    self._pool.append(conn)
                    logger.debug(f"Connection released to pool (pool size: {len(self._pool)})")
                else:
                    # Connection is invalid, close it instead of returning to pool
                    logger.debug("Discarding invalid connection on release")
                    try:
                        conn.close()
                    except Exception:
                        pass  # Ignore errors when closing invalid connection

                self._condition.notify()
            else:
                logger.warning("Attempted to release connection not in use")

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Context manager for pooled connection.

        Acquires a connection from the pool, yields it, and releases it back.

        Yields:
            A DuckDB connection from the pool

        Example:
            with pool.connection() as conn:
                conn.execute("SELECT * FROM table")
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    def get_stats(self) -> dict:
        """
        Get current pool statistics.

        Returns:
            Dictionary with pool statistics:
                - available: Number of connections available in pool
                - in_use: Number of connections currently in use
                - total: Total number of connections (available + in_use)
                - max_size: Maximum pool size
                - utilization: Pool utilization percentage (0-100)
        """
        with self._lock:
            available = len(self._pool)
            in_use = len(self._in_use)
            total = available + in_use
            utilization = (in_use / self.max_size * 100) if self.max_size > 0 else 0

            return {
                "available": available,
                "in_use": in_use,
                "total": total,
                "max_size": self.max_size,
                "utilization": round(utilization, 1),
            }

    def log_stats(self) -> None:
        """
        Log current pool statistics.

        Logs pool state including available connections, connections in use,
        total connections, and utilization percentage.
        """
        stats = self.get_stats()
        logger.info(
            f"Pool stats: {stats['in_use']}/{stats['max_size']} in use, "
            f"{stats['available']} available, "
            f"{stats['utilization']}% utilization"
        )

    def close_all(self) -> None:
        """
        Close all connections in the pool.

        Closes both idle connections in the pool and any currently in-use connections.
        Clears the pool completely. Should be called on shutdown or for cleanup.
        """
        with self._lock:
            for conn in self._pool:
                conn.close()
            for conn in self._in_use:
                conn.close()
            self._pool.clear()
            self._in_use.clear()
            logger.debug("All connections closed")


class DatabaseManager:
    """
    Thread-safe DuckDB connection manager with connection pooling.

    Uses a ConnectionPool to manage and reuse database connections efficiently.
    Singleton pattern ensures a single pool is shared across the application.
    All database operations automatically acquire and release pooled connections.

    Thread Safety:
        - Safe for concurrent use from multiple threads
        - Connections are automatically managed and reused
        - Blocks when pool is exhausted until a connection becomes available

    Connection Lifecycle:
        - Connections are created on-demand up to max_connections
        - Reused across operations for efficiency
        - Validated before reuse to ensure they're still healthy
        - Automatically cleaned up on close()
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Union[Path, str, None] = None, max_connections: int = 5, read_only: bool = False):
        """Singleton pattern for database manager."""
        # Create separate instances for read-only vs read-write
        instance_key = (db_path, read_only)
        if not hasattr(cls, "_instances"):
            cls._instances = {}
        
        if instance_key not in cls._instances:
            with cls._lock:
                if instance_key not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, db_path: Union[Path, str, None] = None, max_connections: int = 5, read_only: bool = False):
        """
        Initialize the database manager with connection pooling.

        Creates a ConnectionPool with the specified configuration. Due to singleton
        pattern, separate instances are created for read-only and read-write modes.

        Args:
            db_path: Path to DuckDB database file. Defaults to ~/hrp-data/hrp.duckdb
            max_connections: Maximum number of connections in the pool (default: 5)
            read_only: If True, creates a read-only connection manager (default: False)
        """
        if self._initialized:
            return

        self.db_path = Path(db_path) if db_path else self._get_db_path()
        self.max_connections = max_connections
        self.read_only = read_only
        self._initialized = True

        # Ensure data directory exists (only for read-write)
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool (single mode: read-only OR read-write, not both)
        self._pool = ConnectionPool(self.db_path, max_size=self.max_connections, read_only=read_only)

        mode_str = "read-only" if read_only else "read-write"
        logger.info(
            f"Database manager initialized: {self.db_path} "
            f"(mode: {mode_str}, max_connections: {self.max_connections})"
        )

    def _get_db_path(self) -> Path:
        """Get database path from environment or default.

        Checks for HRP_DB_PATH first (full path to database file),
        then HRP_DATA_DIR (directory containing hrp.duckdb),
        then falls back to default ~/hrp-data/hrp.duckdb.
        """
        # First check for explicit database path (used in tests)
        db_path = os.getenv("HRP_DB_PATH")
        if db_path:
            return Path(db_path).expanduser()

        # Then check for data directory
        data_dir = os.getenv("HRP_DATA_DIR", str(DEFAULT_DATA_DIR))
        return Path(data_dir).expanduser() / "hrp.duckdb"

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Context manager for pooled database connection.

        Acquires a connection from the pool, yields it for use, and automatically
        releases it back to the pool when done. The connection is reused by
        subsequent operations for efficiency.

        Blocks if all connections are in use until one becomes available.

        Yields:
            A DuckDB connection from the pool (read-only or read-write based on manager configuration)

        Example:
            # Read-write connection (for INSERT, UPDATE, DELETE)
            db = get_db()  # Default is read-write
            with db.connection() as conn:
                conn.execute("INSERT INTO prices VALUES (...)")

            # Read-only connection (for SELECT queries)
            db_ro = get_db(read_only=True)
            with db_ro.connection() as conn:
                result = conn.execute("SELECT * FROM prices").fetchall()
        """
        with self._pool.connection() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Database error: {e}")
                raise

    def execute(self, query: str, params: Union[tuple, None] = None) -> MaterializedRelation:
        """
        Execute a query and return a relation-like object with materialized results.

        Acquires a pooled connection, executes the query, materializes all results,
        and releases the connection back to the pool. The returned MaterializedRelation
        is independent of the connection, allowing the connection to be reused immediately.

        Args:
            query: SQL query to execute
            params: Optional tuple of query parameters

        Returns:
            MaterializedRelation with fetchall() and fetchone() methods

        Note:
            For better performance with large result sets, use fetchdf() instead.
            For write operations, ensure DatabaseManager is initialized with read_only=False.
        """
        with self.connection() as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)
            # Materialize the result before releasing connection
            rows = result.fetchall()
            return MaterializedRelation(rows)

    def fetchall(self, query: str, params: Union[tuple, None] = None) -> List[tuple]:
        """
        Execute a query and fetch all results.

        Acquires a pooled connection, executes the query, fetches all rows,
        and releases the connection back to the pool for reuse.

        Args:
            query: SQL query to execute
            params: Optional tuple of query parameters

        Returns:
            List of tuples containing query results
        """
        with self.connection() as conn:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            return result

    def fetchone(self, query: str, params: Union[tuple, None] = None) -> Union[tuple, None]:
        """
        Execute a query and fetch one result.

        Acquires a pooled connection, executes the query, fetches one row,
        and releases the connection back to the pool for reuse.

        Args:
            query: SQL query to execute
            params: Optional tuple of query parameters

        Returns:
            Single tuple containing query result, or None if no results
        """
        with self.connection() as conn:
            if params:
                result = conn.execute(query, params).fetchone()
            else:
                result = conn.execute(query).fetchone()
            return result

    def fetchdf(self, query: str, params: Union[tuple, None] = None) -> Any:
        """
        Execute a query and return a pandas DataFrame.

        Acquires a pooled connection, executes the query, converts to DataFrame,
        and releases the connection back to the pool for reuse. Recommended for
        large result sets due to efficient memory handling.

        Args:
            query: SQL query to execute
            params: Optional tuple of query parameters

        Returns:
            pandas DataFrame containing query results
        """
        with self.connection() as conn:
            if params:
                result = conn.execute(query, params).df()
            else:
                result = conn.execute(query).df()
            return result

    def close(self) -> None:
        """
        Close all connections in the pool.

        Closes both idle connections in the pool and any currently in-use connections.
        Should be called on application shutdown to clean up resources.
        """
        if hasattr(self, "_pool"):
            self._pool.close_all()
            mode_str = "read-only" if self.read_only else "read-write"
            logger.debug(f"Closed all {mode_str} pooled connections")

    @classmethod
    def reset(cls) -> None:
        """Reset all singleton instances (useful for testing)."""
        with cls._lock:
            if hasattr(cls, "_instances"):
                for instance in cls._instances.values():
                    instance.close()
                cls._instances.clear()


# Convenience function for quick access
def get_db(
    db_path: Union[Path, str, None] = None, max_connections: int = 5, read_only: bool = False
) -> DatabaseManager:
    """
    Get the database manager instance with connection pooling.

    Returns a DatabaseManager instance (singleton per configuration). Separate instances
    are created for read-only and read-write modes.

    Args:
        db_path: Path to DuckDB database file. Defaults to ~/hrp-data/hrp.duckdb
        max_connections: Maximum number of connections in the pool (default: 5)
        read_only: If True, returns a read-only database manager (default: False)

    Returns:
        DatabaseManager instance with connection pooling enabled

    Example:
        # Read-write manager (default, for writes)
        db = get_db()
        with db.connection() as conn:
            conn.execute("INSERT INTO prices VALUES (...)")

        # Read-only manager (for reads, prevents accidental writes)
        db_ro = get_db(read_only=True)
        with db_ro.connection() as conn:
            result = conn.execute("SELECT * FROM prices").fetchall()
    """
    return DatabaseManager(db_path, max_connections, read_only)
