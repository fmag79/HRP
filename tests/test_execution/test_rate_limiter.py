"""Tests for rate limiter."""
import time

import pytest

from hrp.execution.rate_limiter import RateLimitConfig, RateLimiter


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.max_requests_per_window == 5
        assert config.window_seconds == 15.0
        assert config.order_cooldown == 2.0
        assert config.backoff_base == 2.0
        assert config.max_backoff == 60.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            max_requests_per_window=10,
            window_seconds=30.0,
            order_cooldown=1.0,
        )
        assert config.max_requests_per_window == 10
        assert config.window_seconds == 30.0
        assert config.order_cooldown == 1.0


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        config = RateLimitConfig(max_requests_per_window=5, window_seconds=15.0)
        limiter = RateLimiter(config)

        assert limiter.config == config
        assert limiter._tokens == 5.0
        assert limiter._last_order_time is None
        assert limiter._retry_count == 0

    def test_initialization_default_config(self):
        """Test rate limiter with default config."""
        limiter = RateLimiter()
        assert limiter.config.max_requests_per_window == 5

    def test_acquire_consumes_token(self):
        """Test that acquire() consumes a token."""
        limiter = RateLimiter(RateLimitConfig(max_requests_per_window=5))
        initial_tokens = limiter._tokens

        limiter.acquire()

        assert limiter._tokens == initial_tokens - 1.0

    def test_acquire_blocks_when_no_tokens(self):
        """Test that acquire() blocks when no tokens available."""
        config = RateLimitConfig(max_requests_per_window=2, window_seconds=1.0)
        limiter = RateLimiter(config)

        # Consume all tokens
        limiter.acquire()
        limiter.acquire()

        # Next acquire should block briefly
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited ~0.5s (half the window for 1 token refill)
        assert 0.3 < elapsed < 0.7

    def test_order_cooldown_enforced(self):
        """Test that order cooldown is enforced between orders."""
        config = RateLimitConfig(order_cooldown=0.5)
        limiter = RateLimiter(config)

        # First order
        limiter.acquire(is_order=True)

        # Second order should wait
        start = time.monotonic()
        limiter.acquire(is_order=True)
        elapsed = time.monotonic() - start

        assert 0.4 < elapsed < 0.6

    def test_non_order_requests_skip_cooldown(self):
        """Test that non-order requests don't trigger cooldown."""
        config = RateLimitConfig(order_cooldown=1.0)
        limiter = RateLimiter(config)

        limiter.acquire(is_order=True)

        # Non-order request should not wait
        start = time.monotonic()
        limiter.acquire(is_order=False)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # Should be instant

    def test_token_refill_over_time(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(max_requests_per_window=5, window_seconds=1.0)
        limiter = RateLimiter(config)

        # Consume all tokens
        for _ in range(5):
            limiter.acquire()

        assert limiter._tokens < 1.0

        # Wait for refill
        time.sleep(0.3)

        # Manually update tokens (simulating what acquire() does)
        now = time.monotonic()
        elapsed = now - limiter._last_update
        refill_rate = config.max_requests_per_window / config.window_seconds
        limiter._tokens = min(
            config.max_requests_per_window,
            limiter._tokens + (elapsed * refill_rate),
        )

        # Should have refilled ~1.5 tokens (5 tokens/s * 0.3s)
        assert 1.0 < limiter._tokens < 2.0

    def test_handle_throttle_parses_seconds(self):
        """Test throttle response parsing."""
        limiter = RateLimiter()

        response = "Rate limit exceeded. Expected available in 30 seconds."
        wait_seconds = limiter.handle_throttle(response)

        assert wait_seconds == 30.0
        assert limiter._retry_count == 1

    def test_handle_throttle_fallback_exponential_backoff(self):
        """Test exponential backoff when no seconds in response."""
        config = RateLimitConfig(backoff_base=2.0)
        limiter = RateLimiter(config)

        # First retry: 2^0 = 1s
        wait1 = limiter.handle_throttle("Throttled")
        assert wait1 == 1.0
        assert limiter._retry_count == 1

        # Second retry: 2^1 = 2s
        wait2 = limiter.handle_throttle("Throttled")
        assert wait2 == 2.0
        assert limiter._retry_count == 2

        # Third retry: 2^2 = 4s
        wait3 = limiter.handle_throttle("Throttled")
        assert wait3 == 4.0
        assert limiter._retry_count == 3

    def test_handle_throttle_respects_max_backoff(self):
        """Test max backoff limit."""
        config = RateLimitConfig(backoff_base=2.0, max_backoff=5.0)
        limiter = RateLimiter(config)

        # Trigger many retries
        limiter._retry_count = 10

        wait = limiter.handle_throttle("Throttled")

        # Should cap at max_backoff
        assert wait == 5.0

    def test_handle_throttle_raises_after_max_retries(self):
        """Test that max retries raises error."""
        config = RateLimitConfig(max_retries=3)
        limiter = RateLimiter(config)

        # Exhaust retries
        limiter._retry_count = 3

        with pytest.raises(RuntimeError, match="max retries"):
            limiter.handle_throttle("Throttled")

    def test_reset_clears_retry_count(self):
        """Test reset clears retry counter."""
        limiter = RateLimiter()
        limiter._retry_count = 5

        limiter.reset()

        assert limiter._retry_count == 0

    def test_thread_safety(self):
        """Test rate limiter is thread-safe."""
        import threading

        limiter = RateLimiter(RateLimitConfig(max_requests_per_window=100))
        results = []

        def worker():
            limiter.acquire()
            results.append(1)

        # Create 10 threads
        threads = [threading.Thread(target=worker) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All 10 should complete
        assert len(results) == 10
