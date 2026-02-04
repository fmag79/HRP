"""DuckDB connection pooling with retry logic.

Provides thread-safe connection pooling and automatic retry on transient errors.
"""

from __future__ import annotations

import functools
import queue
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import duckdb
from loguru import logger

T = TypeVar("T")


class ConnectionPool:
    """Thread-safe DuckDB connection pool."""

    def __init__(
        self,
        database: str,
        max_connections: int = 5,
        read_only: bool = False,
    ):
        """
        Initialize connection pool.

        Args:
            database: Path to DuckDB database
            max_connections: Maximum number of connections
            read_only: Whether connections should be read-only
        """
        self.database = database
        self.max_connections = max_connections
        self.read_only = read_only
        self._pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created = 0

    def get_connection(self, timeout: float = 30.0) -> duckdb.DuckDBPyConnection:
        """
        Get a connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection

        Returns:
            DuckDB connection

        Raises:
            TimeoutError: If no connection available within timeout
        """
        try:
            # Try to get existing connection
            conn = self._pool.get(block=False)
            return conn
        except queue.Empty:
            pass

        # Try to create new connection if under limit
        with self._lock:
            if self._created < self.max_connections:
                conn = duckdb.connect(self.database, read_only=self.read_only)
                self._created += 1
                logger.debug(f"Created new connection ({self._created}/{self.max_connections})")
                return conn

        # Wait for connection to be released
        try:
            conn = self._pool.get(block=True, timeout=timeout)
            return conn
        except queue.Empty:
            raise TimeoutError(f"No connection available within {timeout}s")

    def release_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put(conn, block=False)
        except queue.Full:
            # Pool is full, close the connection
            conn.close()

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except queue.Empty:
                break
        self._created = 0


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    exponential: bool = True,
    retryable_errors: tuple = ("database is locked", "BUSY"),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying operations on transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        exponential: Whether to use exponential backoff
        retryable_errors: Error messages that trigger retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_retryable = any(err.lower() in error_msg for err in retryable_errors)

                    if not is_retryable or attempt >= max_retries:
                        raise

                    last_exception = e
                    delay = base_delay * (2**attempt) if exponential else base_delay
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator
