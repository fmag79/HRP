"""
Comprehensive tests for the RateLimiter utility.

Tests cover:
- Basic token acquisition
- Token refill mechanism
- Blocking and non-blocking modes
- Timeout functionality
- Context manager usage
- Reset functionality
- Thread safety
- Edge cases and error handling
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hrp.utils.rate_limiter import RateLimiter


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_sets_parameters(self):
        """Test that initialization sets correct parameters."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        assert limiter.max_calls == 10
        assert limiter.period == 1.0
        assert limiter.tokens == 10

    def test_init_with_different_values(self):
        """Test initialization with various parameter values."""
        limiter1 = RateLimiter(max_calls=5, period=2.0)
        assert limiter1.max_calls == 5
        assert limiter1.period == 2.0

        limiter2 = RateLimiter(max_calls=100, period=60.0)
        assert limiter2.max_calls == 100
        assert limiter2.period == 60.0

    def test_init_starts_with_full_tokens(self):
        """Test that limiter starts with full token bucket."""
        limiter = RateLimiter(max_calls=20, period=5.0)
        assert limiter.tokens == limiter.max_calls


class TestBasicAcquisition:
    """Tests for basic token acquisition."""

    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = RateLimiter(max_calls=10, period=1.0)
        result = limiter.acquire(tokens=1, blocking=False)

        assert result is True
        assert limiter.tokens == 9

    def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(max_calls=10, period=1.0)
        result = limiter.acquire(tokens=5, blocking=False)

        assert result is True
        assert limiter.tokens == 5

    def test_acquire_all_tokens(self):
        """Test acquiring all available tokens."""
        limiter = RateLimiter(max_calls=10, period=1.0)
        result = limiter.acquire(tokens=10, blocking=False)

        assert result is True
        assert limiter.tokens == 0

    def test_acquire_default_is_one_token(self):
        """Test that acquire() defaults to 1 token."""
        limiter = RateLimiter(max_calls=10, period=1.0)
        result = limiter.acquire(blocking=False)

        assert result is True
        assert limiter.tokens == 9


class TestTokenRefill:
    """Tests for token refill mechanism."""

    def test_tokens_refill_over_time(self):
        """Test that tokens refill based on elapsed time."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)
        assert limiter.tokens == 0

        # Wait for half the period
        time.sleep(0.5)

        # Check available tokens (should be ~5)
        available = limiter.available_tokens
        assert 4.5 <= available <= 5.5

    def test_tokens_dont_exceed_max(self):
        """Test that tokens never exceed max_calls."""
        limiter = RateLimiter(max_calls=5, period=1.0)

        # Wait longer than the period
        time.sleep(2.0)

        # Tokens should be capped at max_calls
        assert limiter.available_tokens == 5

    def test_partial_refill(self):
        """Test that partial token refill works correctly."""
        limiter = RateLimiter(max_calls=100, period=10.0)

        # Use 50 tokens
        limiter.acquire(tokens=50, blocking=False)

        # Wait for 1 second (should add 10 tokens)
        time.sleep(1.0)

        available = limiter.available_tokens
        # Should be around 60 (50 + 10)
        assert 58 <= available <= 62


class TestNonBlockingMode:
    """Tests for non-blocking acquisition mode."""

    def test_nonblocking_returns_false_when_insufficient(self):
        """Test that non-blocking mode returns False when tokens unavailable."""
        limiter = RateLimiter(max_calls=10, period=10.0)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)

        # Try to acquire more
        result = limiter.acquire(tokens=1, blocking=False)
        assert result is False

    def test_nonblocking_does_not_wait(self):
        """Test that non-blocking mode returns immediately."""
        limiter = RateLimiter(max_calls=5, period=10.0)

        # Use all tokens
        limiter.acquire(tokens=5, blocking=False)

        # This should return immediately
        start_time = time.monotonic()
        result = limiter.acquire(tokens=1, blocking=False)
        elapsed = time.monotonic() - start_time

        assert result is False
        assert elapsed < 0.1  # Should be nearly instant


class TestBlockingMode:
    """Tests for blocking acquisition mode."""

    def test_blocking_waits_for_tokens(self):
        """Test that blocking mode waits until tokens are available."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)

        # This should block briefly then succeed
        start_time = time.monotonic()
        result = limiter.acquire(tokens=1, blocking=True)
        elapsed = time.monotonic() - start_time

        assert result is True
        assert 0.05 <= elapsed <= 0.3  # Should wait for token refill

    def test_blocking_acquires_when_available(self):
        """Test that blocking mode acquires tokens when they become available."""
        limiter = RateLimiter(max_calls=10, period=0.5)

        # Use most tokens
        limiter.acquire(tokens=9, blocking=False)

        # This should succeed with minimal wait
        result = limiter.acquire(tokens=1, blocking=True)
        assert result is True


class TestTimeoutFunctionality:
    """Tests for timeout parameter."""

    def test_timeout_returns_false_when_exceeded(self):
        """Test that timeout causes acquire to return False."""
        limiter = RateLimiter(max_calls=10, period=10.0)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)

        # Try to acquire with short timeout
        start_time = time.monotonic()
        result = limiter.acquire(tokens=5, blocking=True, timeout=0.2)
        elapsed = time.monotonic() - start_time

        assert result is False
        assert 0.15 <= elapsed <= 0.3

    def test_timeout_succeeds_before_expiry(self):
        """Test that acquisition succeeds if tokens available before timeout."""
        limiter = RateLimiter(max_calls=10, period=0.5)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)

        # Wait with generous timeout
        result = limiter.acquire(tokens=1, blocking=True, timeout=1.0)
        assert result is True

    def test_timeout_zero_behaves_like_nonblocking(self):
        """Test that timeout=0 behaves like non-blocking mode."""
        limiter = RateLimiter(max_calls=5, period=10.0)

        # Use all tokens
        limiter.acquire(tokens=5, blocking=False)

        # Timeout of 0 should return immediately
        start_time = time.monotonic()
        result = limiter.acquire(tokens=1, blocking=True, timeout=0)
        elapsed = time.monotonic() - start_time

        assert result is False
        assert elapsed < 0.1


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_acquires_token(self):
        """Test that context manager acquires a token."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        with limiter:
            # One token should be consumed
            assert limiter.tokens == 9

    def test_context_manager_multiple_uses(self):
        """Test multiple context manager uses."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        with limiter:
            pass

        with limiter:
            pass

        # Two tokens should be consumed
        assert limiter.tokens == 8

    def test_context_manager_blocks_if_needed(self):
        """Test that context manager blocks when tokens unavailable."""
        limiter = RateLimiter(max_calls=5, period=0.5)

        # Use all tokens
        limiter.acquire(tokens=5, blocking=False)

        # This should block briefly
        start_time = time.monotonic()
        with limiter:
            elapsed = time.monotonic() - start_time

        assert 0.05 <= elapsed <= 0.3


class TestResetFunctionality:
    """Tests for reset() method."""

    def test_reset_restores_full_tokens(self):
        """Test that reset restores tokens to maximum."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use some tokens
        limiter.acquire(tokens=7, blocking=False)
        assert limiter.tokens == 3

        # Reset
        limiter.reset()
        assert limiter.tokens == 10

    def test_reset_after_complete_depletion(self):
        """Test reset after all tokens are used."""
        limiter = RateLimiter(max_calls=20, period=5.0)

        # Use all tokens
        limiter.acquire(tokens=20, blocking=False)
        assert limiter.tokens == 0

        # Reset
        limiter.reset()
        assert limiter.tokens == 20

    def test_reset_updates_timestamp(self):
        """Test that reset updates the last_update timestamp."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        old_timestamp = limiter.last_update

        time.sleep(0.1)

        limiter.reset()
        new_timestamp = limiter.last_update

        assert new_timestamp > old_timestamp


class TestAvailableTokensProperty:
    """Tests for available_tokens property."""

    def test_available_tokens_initial(self):
        """Test that available_tokens returns max initially."""
        limiter = RateLimiter(max_calls=15, period=1.0)
        assert limiter.available_tokens == 15

    def test_available_tokens_after_acquisition(self):
        """Test available_tokens after consuming some."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        limiter.acquire(tokens=3, blocking=False)
        assert limiter.available_tokens == 7

    def test_available_tokens_accounts_for_refill(self):
        """Test that available_tokens includes refilled tokens."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use all tokens
        limiter.acquire(tokens=10, blocking=False)

        # Wait for partial refill
        time.sleep(0.5)

        available = limiter.available_tokens
        assert 4.5 <= available <= 5.5


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_acquire_more_than_max_raises_error(self):
        """Test that requesting more tokens than max raises ValueError."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        with pytest.raises(ValueError) as exc_info:
            limiter.acquire(tokens=15, blocking=False)

        assert "Cannot acquire 15 tokens (max: 10)" in str(exc_info.value)

    def test_acquire_zero_tokens(self):
        """Test acquiring zero tokens."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        result = limiter.acquire(tokens=0, blocking=False)
        assert result is True
        assert limiter.tokens == 10  # No tokens consumed

    def test_acquire_negative_tokens_handled(self):
        """Test that negative tokens are handled (adds tokens)."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use some tokens first
        limiter.acquire(tokens=5, blocking=False)

        # Negative acquisition adds tokens
        result = limiter.acquire(tokens=-2, blocking=False)
        assert result is True
        assert limiter.tokens == 7


class TestThreadSafety:
    """Tests for thread safety of RateLimiter."""

    def test_concurrent_acquisitions(self):
        """Test concurrent token acquisitions from multiple threads."""
        limiter = RateLimiter(max_calls=100, period=1.0)
        acquired_count = []
        failed_count = []

        def acquire_tokens(thread_id):
            for _ in range(10):
                if limiter.acquire(tokens=1, blocking=False):
                    acquired_count.append(thread_id)
                else:
                    failed_count.append(thread_id)

        threads = []
        for i in range(10):
            t = threading.Thread(target=acquire_tokens, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have acquired exactly 100 tokens
        assert len(acquired_count) == 100
        # Some threads should have failed to acquire
        assert len(failed_count) == 0  # All should succeed since we have 100 tokens

    def test_concurrent_blocking_acquisitions(self):
        """Test concurrent blocking acquisitions."""
        limiter = RateLimiter(max_calls=50, period=2.0)
        results = []

        def acquire_with_blocking(thread_id):
            result = limiter.acquire(tokens=1, blocking=True, timeout=3.0)
            results.append((thread_id, result))

        # Try to acquire more than available
        threads = []
        for i in range(60):
            t = threading.Thread(target=acquire_with_blocking, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Most should succeed (initial 50 + refilled tokens during wait)
        successful = sum(1 for _, result in results if result)
        assert successful >= 50

    def test_thread_pool_executor(self):
        """Test using RateLimiter with ThreadPoolExecutor."""
        limiter = RateLimiter(max_calls=20, period=1.0)

        def try_acquire(task_id):
            return limiter.acquire(tokens=1, blocking=False)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_acquire, i) for i in range(25)]
            results = [f.result() for f in futures]

        # First 20 should succeed, rest should fail
        successful = sum(1 for r in results if r)
        assert successful == 20

    def test_reset_thread_safety(self):
        """Test that reset is thread-safe."""
        limiter = RateLimiter(max_calls=50, period=1.0)
        errors = []

        def acquire_and_reset(thread_id):
            try:
                for _ in range(10):
                    limiter.acquire(tokens=1, blocking=False)
                    if thread_id % 3 == 0:
                        limiter.reset()
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(10):
            t = threading.Thread(target=acquire_and_reset, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_very_short_period(self):
        """Test rate limiter with very short period."""
        limiter = RateLimiter(max_calls=5, period=0.1)

        # Use all tokens
        limiter.acquire(tokens=5, blocking=False)

        # Wait for refill
        time.sleep(0.15)

        # Should have refilled
        assert limiter.available_tokens >= 4.5

    def test_very_long_period(self):
        """Test rate limiter with very long period."""
        limiter = RateLimiter(max_calls=100, period=3600.0)

        # Use some tokens
        limiter.acquire(tokens=10, blocking=False)

        # Short wait shouldn't refill much
        time.sleep(0.1)

        available = limiter.available_tokens
        assert 89.5 <= available <= 90.5  # Very small refill

    def test_fractional_tokens(self):
        """Test that fractional token refill works correctly."""
        limiter = RateLimiter(max_calls=3, period=1.0)

        # Use all tokens
        limiter.acquire(tokens=3, blocking=False)

        # Wait for fractional refill
        time.sleep(0.4)

        # Should have ~1.2 tokens
        available = limiter.available_tokens
        assert 1.0 <= available <= 1.5

    def test_multiple_rapid_acquisitions(self):
        """Test multiple rapid token acquisitions."""
        limiter = RateLimiter(max_calls=100, period=1.0)

        # Rapidly acquire tokens
        for _ in range(50):
            result = limiter.acquire(tokens=1, blocking=False)
            assert result is True

        # Should have 50 tokens left
        assert 49 <= limiter.tokens <= 51

    def test_large_max_calls(self):
        """Test rate limiter with very large max_calls."""
        limiter = RateLimiter(max_calls=10000, period=60.0)

        result = limiter.acquire(tokens=5000, blocking=False)
        assert result is True
        assert limiter.tokens == 5000

    def test_acquire_exact_available_tokens(self):
        """Test acquiring exactly the number of available tokens."""
        limiter = RateLimiter(max_calls=10, period=1.0)

        # Use 3 tokens
        limiter.acquire(tokens=3, blocking=False)

        # Acquire exactly what's left
        result = limiter.acquire(tokens=7, blocking=False)
        assert result is True
        assert limiter.tokens == 0
