"""Rate limiting for Robinhood API calls."""
import logging
import re
import time
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    max_requests_per_window: int = 5  # Max requests per window
    window_seconds: float = 15.0  # Window duration
    order_cooldown: float = 2.0  # Min seconds between orders
    backoff_base: float = 2.0  # Exponential backoff base
    max_backoff: float = 60.0  # Max backoff seconds
    max_retries: int = 3  # Max retry attempts


class RateLimiter:
    """Token-bucket rate limiter for Robinhood API.

    Implements a token bucket algorithm to throttle API requests
    within configured limits. Thread-safe for concurrent usage.

    Example:
        >>> config = RateLimitConfig(max_requests_per_window=5, window_seconds=15.0)
        >>> limiter = RateLimiter(config)
        >>> limiter.acquire()  # Blocks if rate limit exceeded
        >>> # Make API call here
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration. Uses defaults if None.
        """
        self.config = config or RateLimitConfig()
        self._tokens = float(self.config.max_requests_per_window)
        self._last_update = time.monotonic()
        self._last_order_time: float | None = None
        self._lock = Lock()
        self._retry_count = 0

        logger.info(
            "RateLimiter initialized: %d requests per %.1fs window",
            self.config.max_requests_per_window,
            self.config.window_seconds,
        )

    def acquire(self, is_order: bool = False) -> None:
        """Block until a request slot is available.

        Implements token bucket: tokens refill at a constant rate,
        and each request consumes one token. If no tokens available,
        blocks until enough tokens have refilled.

        Args:
            is_order: If True, enforces order_cooldown between requests.
        """
        with self._lock:
            now = time.monotonic()

            # Enforce order cooldown
            if is_order and self._last_order_time is not None:
                elapsed_since_order = now - self._last_order_time
                if elapsed_since_order < self.config.order_cooldown:
                    sleep_time = self.config.order_cooldown - elapsed_since_order
                    logger.debug(
                        "Order cooldown: sleeping %.2fs (%.2fs since last order)",
                        sleep_time,
                        elapsed_since_order,
                    )
                    time.sleep(sleep_time)
                    now = time.monotonic()

            # Refill tokens based on elapsed time
            elapsed = now - self._last_update
            refill_rate = (
                self.config.max_requests_per_window / self.config.window_seconds
            )
            self._tokens = min(
                self.config.max_requests_per_window,
                self._tokens + (elapsed * refill_rate),
            )
            self._last_update = now

            # Wait if no tokens available
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / refill_rate
                logger.warning(
                    "Rate limit reached, waiting %.2fs for token refill", wait_time
                )
                time.sleep(wait_time)
                # Recalculate after sleep
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(
                    self.config.max_requests_per_window,
                    self._tokens + (elapsed * refill_rate),
                )
                self._last_update = now

            # Consume a token
            self._tokens -= 1.0

            # Update last order time
            if is_order:
                self._last_order_time = now

            logger.debug(
                "Rate limiter: consumed token, %.1f tokens remaining", self._tokens
            )

    def handle_throttle(self, response_text: str) -> float:
        """Parse throttle response and return wait seconds.

        Robinhood throttle responses typically contain:
        "Expected available in N seconds" or "Please wait N seconds"

        Args:
            response_text: Response text from Robinhood API.

        Returns:
            Seconds to wait before retrying.
        """
        # Try to parse wait time from response
        match = re.search(r"(\d+)\s*seconds?", response_text.lower())
        if match:
            wait_seconds = float(match.group(1))
        else:
            # Fallback: exponential backoff
            wait_seconds = min(
                self.config.backoff_base ** self._retry_count, self.config.max_backoff
            )

        self._retry_count += 1

        if self._retry_count > self.config.max_retries:
            logger.error(
                "Max retries (%d) exceeded, throttled response: %s",
                self.config.max_retries,
                response_text[:200],
            )
            raise RuntimeError(
                f"Rate limit max retries ({self.config.max_retries}) exceeded"
            )

        logger.warning(
            "Throttled by Robinhood (retry %d/%d), waiting %.1fs: %s",
            self._retry_count,
            self.config.max_retries,
            wait_seconds,
            response_text[:200],
        )

        return wait_seconds

    def reset(self) -> None:
        """Reset rate limiter state.

        Useful after successful operations to reset retry counter.
        """
        with self._lock:
            self._retry_count = 0
            logger.debug("Rate limiter reset: retry counter cleared")
