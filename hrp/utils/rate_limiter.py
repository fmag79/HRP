"""
Rate limiter utility for API call throttling.

Implements token bucket algorithm for rate limiting with thread-safe operations.
"""

import threading
import time
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Allows a maximum number of calls within a given time period.
    Calls that exceed the limit will block until tokens become available.
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update

        # Add tokens proportional to elapsed time
        tokens_to_add = elapsed * (self.max_calls / self.period)
        self.tokens = min(self.max_calls, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default: 1)
            blocking: If True, wait until tokens are available
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if tokens were acquired, False if timeout occurred

        Raises:
            ValueError: If tokens exceeds max_calls
        """
        if tokens > self.max_calls:
            raise ValueError(f"Cannot acquire {tokens} tokens (max: {self.max_calls})")

        start_time = time.monotonic()

        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not blocking:
                    return False

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed * (self.period / self.max_calls)

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            # Sleep before retrying
            time.sleep(wait_time)

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self.lock:
            self.tokens = self.max_calls
            self.last_update = time.monotonic()

    @property
    def available_tokens(self) -> float:
        """Get the current number of available tokens."""
        with self.lock:
            self._refill_tokens()
            return self.tokens
