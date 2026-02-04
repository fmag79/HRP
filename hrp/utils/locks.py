"""Job locking utilities for preventing concurrent execution.

Uses file-based locks with PID tracking for stale lock detection.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


class JobLock:
    """Context manager for file-based job locking."""

    def __init__(
        self,
        lock_file: str,
        timeout: float = 0.0,
        stale_threshold: float = 3600.0,
    ):
        """
        Initialize job lock.

        Args:
            lock_file: Path to lock file
            timeout: Time to wait for lock (0 = no wait)
            stale_threshold: Seconds before lock considered stale
        """
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.stale_threshold = stale_threshold
        self._acquired = False

    def __enter__(self) -> bool:
        """Attempt to acquire the lock."""
        start_time = time.time()

        while True:
            if self._try_acquire():
                self._acquired = True
                return True

            if self.timeout <= 0:
                return False

            if time.time() - start_time >= self.timeout:
                return False

            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the lock."""
        if self._acquired:
            try:
                self.lock_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to release lock {self.lock_file}: {e}")
            self._acquired = False

    def _try_acquire(self) -> bool:
        """Try to acquire the lock."""
        # Check if lock exists
        if self.lock_file.exists():
            if self._is_stale():
                logger.info(f"Cleaning stale lock: {self.lock_file}")
                self.lock_file.unlink(missing_ok=True)
            else:
                return False

        # Create lock file with our PID
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(str(os.getpid()))

            # Verify we got it (race condition check)
            time.sleep(0.01)
            if self.lock_file.read_text().strip() == str(os.getpid()):
                return True
        except Exception as e:
            logger.warning(f"Failed to acquire lock: {e}")

        return False

    def _is_stale(self) -> bool:
        """Check if the lock is stale."""
        try:
            # Check if PID is still running
            pid = int(self.lock_file.read_text().strip())

            # Check if process exists
            try:
                os.kill(pid, 0)
                return False  # Process exists, not stale
            except OSError:
                return True  # Process doesn't exist, stale

        except (ValueError, FileNotFoundError):
            return True  # Invalid or missing, treat as stale


def acquire_job_lock(
    lock_file: str,
    timeout: float = 0.0,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """
    Decorator to acquire job lock before execution.

    Args:
        lock_file: Path to lock file
        timeout: Time to wait for lock

    Returns:
        Decorated function that returns None if lock not acquired
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T | None:
            with JobLock(lock_file, timeout=timeout) as acquired:
                if not acquired:
                    logger.info(f"Skipping {func.__name__}: lock held by another process")
                    return None
                return func(*args, **kwargs)

        return wrapper

    return decorator
