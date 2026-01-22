"""
Comprehensive tests for the retry utility.

Tests cover:
- Retry on transient failures
- Non-transient error handling
- Max retries enforcement
- Exponential backoff delays
- Jitter functionality
- Retry callback mechanism
- retry_call function wrapper
- RetryError exception
- Edge cases and error handling
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from hrp.utils.retry import (
    RetryError,
    is_transient_error,
    retry_call,
    retry_with_backoff,
)


class MockHTTPError(Exception):
    """Mock HTTP error for testing."""

    def __init__(self, status_code: int):
        self.response = Mock()
        self.response.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class TestIsTransientError:
    """Tests for is_transient_error() helper function."""

    def test_timeout_error_is_transient(self):
        """Test that TimeoutError is considered transient."""
        error = TimeoutError("Request timed out")
        assert is_transient_error(error) is True

    def test_connection_error_is_transient(self):
        """Test that ConnectionError is considered transient."""
        error = ConnectionError("Connection failed")
        assert is_transient_error(error) is True

    def test_connection_reset_is_transient(self):
        """Test that ConnectionResetError is considered transient."""
        error = ConnectionResetError("Connection reset by peer")
        assert is_transient_error(error) is True

    def test_broken_pipe_is_transient(self):
        """Test that BrokenPipeError is considered transient."""
        error = BrokenPipeError("Broken pipe")
        assert is_transient_error(error) is True

    def test_http_500_is_transient(self):
        """Test that HTTP 5xx errors are considered transient."""
        for status in [500, 502, 503, 504]:
            error = MockHTTPError(status)
            assert is_transient_error(error) is True, f"HTTP {status} should be transient"

    def test_http_429_is_transient(self):
        """Test that HTTP 429 (rate limit) is considered transient."""
        error = MockHTTPError(429)
        assert is_transient_error(error) is True

    def test_http_4xx_is_not_transient(self):
        """Test that HTTP 4xx errors (except 429) are not transient."""
        for status in [400, 401, 403, 404, 422]:
            error = MockHTTPError(status)
            assert is_transient_error(error) is False, f"HTTP {status} should not be transient"

    def test_http_2xx_is_not_transient(self):
        """Test that HTTP 2xx success codes are not transient."""
        for status in [200, 201, 204]:
            error = MockHTTPError(status)
            assert is_transient_error(error) is False, f"HTTP {status} should not be transient"

    def test_value_error_is_not_transient(self):
        """Test that ValueError is not considered transient."""
        error = ValueError("Invalid value")
        assert is_transient_error(error) is False

    def test_key_error_is_not_transient(self):
        """Test that KeyError is not considered transient."""
        error = KeyError("missing_key")
        assert is_transient_error(error) is False

    def test_attribute_error_is_not_transient(self):
        """Test that AttributeError is not considered transient."""
        error = AttributeError("'NoneType' object has no attribute 'foo'")
        assert is_transient_error(error) is False


class TestRetryWithBackoffDecorator:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_transient_timeout(self):
        """Test retry on TimeoutError."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Timeout")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_on_transient_http_error(self):
        """Test retry on HTTP 5xx errors."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise MockHTTPError(503)
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 2

    def test_no_retry_on_non_transient_error(self):
        """Test that non-transient errors are not retried."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError) as exc_info:
            failing_func()

        assert "Invalid input" in str(exc_info.value)
        assert call_count[0] == 1  # Should not retry

    def test_max_retries_exhausted(self):
        """Test that RetryError is raised when max retries exhausted."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def always_fails():
            call_count[0] += 1
            raise TimeoutError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            always_fails()

        assert "Failed after 3 attempts" in str(exc_info.value)
        assert call_count[0] == 3  # Initial + 2 retries
        assert isinstance(exc_info.value.last_exception, TimeoutError)

    def test_exponential_backoff_delays(self):
        """Test that delays follow exponential backoff pattern."""
        call_times = []

        @retry_with_backoff(
            max_retries=3, base_delay=0.1, exponential_base=2.0, jitter=False
        )
        def failing_func():
            call_times.append(time.monotonic())
            if len(call_times) < 4:
                raise TimeoutError("Retry")
            return "success"

        failing_func()

        # Calculate delays between calls
        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        # Expected delays: 0.1, 0.2, 0.4 (exponential backoff)
        assert 0.08 <= delays[0] <= 0.15  # First delay ~0.1s
        assert 0.18 <= delays[1] <= 0.25  # Second delay ~0.2s
        assert 0.38 <= delays[2] <= 0.45  # Third delay ~0.4s

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        call_times = []

        @retry_with_backoff(
            max_retries=5,
            base_delay=1.0,
            max_delay=0.2,
            exponential_base=2.0,
            jitter=False,
        )
        def failing_func():
            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise TimeoutError("Retry")
            return "success"

        failing_func()

        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        # All delays should be capped at max_delay (0.2s)
        for delay in delays:
            assert delay <= 0.25  # Allow small tolerance

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        delays_run1 = []
        delays_run2 = []

        def run_with_jitter():
            call_times = []

            @retry_with_backoff(
                max_retries=2, base_delay=0.1, exponential_base=2.0, jitter=True
            )
            def failing_func():
                call_times.append(time.monotonic())
                if len(call_times) < 3:
                    raise TimeoutError("Retry")
                return "success"

            failing_func()
            return [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        delays_run1 = run_with_jitter()
        delays_run2 = run_with_jitter()

        # With jitter, delays should vary between runs
        # (Small chance they're identical, but very unlikely)
        assert delays_run1 != delays_run2 or len(delays_run1) == 0

    def test_on_retry_callback(self):
        """Test that on_retry callback is called on each retry."""
        callback_calls = []

        def on_retry_handler(exception, attempt, delay):
            callback_calls.append((exception, attempt, delay))

        call_count = [0]

        @retry_with_backoff(
            max_retries=3, base_delay=0.01, jitter=False, on_retry=on_retry_handler
        )
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Retry")
            return "success"

        failing_func()

        # Should have 2 retry callbacks (after 1st and 2nd failures)
        assert len(callback_calls) == 2

        # Check first callback
        exception1, attempt1, delay1 = callback_calls[0]
        assert isinstance(exception1, TimeoutError)
        assert attempt1 == 1
        assert 0.008 <= delay1 <= 0.015

        # Check second callback
        exception2, attempt2, delay2 = callback_calls[1]
        assert isinstance(exception2, TimeoutError)
        assert attempt2 == 2
        assert 0.018 <= delay2 <= 0.025

    def test_custom_exception_types(self):
        """Test specifying custom exception types to catch."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=2, base_delay=0.01, exceptions=(TimeoutError, ValueError)
        )
        def failing_func():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First error")
            if call_count[0] == 2:
                raise TimeoutError("Second error")
            return "success"

        # ValueError is in exceptions but not transient, so it won't be retried
        with pytest.raises(ValueError):
            failing_func()

        assert call_count[0] == 1

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @retry_with_backoff(max_retries=3)
        def documented_func():
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    def test_retry_with_function_arguments(self):
        """Test that retried functions receive their arguments correctly."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def func_with_args(x: int, y: int, z: int = 10) -> int:
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("Retry")
            return x + y + z

        result = func_with_args(5, 10, z=20)

        assert result == 35
        assert call_count[0] == 2

    def test_retry_with_return_values(self):
        """Test that return values are preserved through retries."""

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def returns_dict():
            return {"status": "ok", "data": [1, 2, 3]}

        result = returns_dict()

        assert result == {"status": "ok", "data": [1, 2, 3]}


class TestRetryCall:
    """Tests for retry_call() function wrapper."""

    def test_retry_call_successful(self):
        """Test retry_call with successful function."""
        call_count = [0]

        def successful_func(value):
            call_count[0] += 1
            return value * 2

        result = retry_call(successful_func, 5, max_retries=3)

        assert result == 10
        assert call_count[0] == 1

    def test_retry_call_with_retries(self):
        """Test retry_call with transient failures."""
        call_count = [0]

        def failing_func(value):
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Retry")
            return value + 10

        result = retry_call(failing_func, 5, max_retries=3, base_delay=0.01, jitter=False)

        assert result == 15
        assert call_count[0] == 3

    def test_retry_call_with_kwargs(self):
        """Test retry_call with keyword arguments."""
        call_count = [0]

        def func_with_kwargs(a, b, c=0):
            call_count[0] += 1
            return a + b + c

        result = retry_call(func_with_kwargs, 1, 2, c=3, max_retries=2)

        assert result == 6
        assert call_count[0] == 1

    def test_retry_call_exhausts_retries(self):
        """Test retry_call raises RetryError when retries exhausted."""

        def always_fails():
            raise TimeoutError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            retry_call(always_fails, max_retries=2, base_delay=0.01, jitter=False)

        assert "Failed after 3 attempts" in str(exc_info.value)

    def test_retry_call_with_callback(self):
        """Test retry_call with on_retry callback."""
        callback_calls = []

        def on_retry_handler(exception, attempt, delay):
            callback_calls.append(attempt)

        call_count = [0]

        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("Retry")
            return "success"

        retry_call(
            failing_func, max_retries=2, base_delay=0.01, on_retry=on_retry_handler
        )

        assert len(callback_calls) == 1
        assert callback_calls[0] == 1

    def test_retry_call_custom_parameters(self):
        """Test retry_call with custom backoff parameters."""
        call_times = []

        def failing_func():
            call_times.append(time.monotonic())
            if len(call_times) < 3:
                raise TimeoutError("Retry")
            return "success"

        retry_call(
            failing_func,
            max_retries=3,
            base_delay=0.05,
            exponential_base=3.0,
            jitter=False,
        )

        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        # Exponential base of 3: delays should be 0.05, 0.15, ...
        assert 0.04 <= delays[0] <= 0.08
        assert 0.13 <= delays[1] <= 0.18


class TestRetryError:
    """Tests for RetryError exception."""

    def test_retry_error_message(self):
        """Test RetryError stores message correctly."""
        error = RetryError("Test error message")
        assert str(error) == "Test error message"

    def test_retry_error_with_last_exception(self):
        """Test RetryError stores last exception."""
        last_exc = TimeoutError("Original error")
        error = RetryError("Retry failed", last_exception=last_exc)

        assert error.last_exception is last_exc
        assert isinstance(error.last_exception, TimeoutError)

    def test_retry_error_without_last_exception(self):
        """Test RetryError with no last exception."""
        error = RetryError("Failed")
        assert error.last_exception is None

    def test_retry_error_is_exception(self):
        """Test that RetryError is an Exception."""
        error = RetryError("Test")
        assert isinstance(error, Exception)


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_zero_max_retries(self):
        """Test with max_retries=0 (no retries, fail immediately)."""
        call_count = [0]

        @retry_with_backoff(max_retries=0, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            raise TimeoutError("Fail")

        with pytest.raises(RetryError):
            failing_func()

        assert call_count[0] == 1  # Only initial attempt

    def test_very_high_max_retries(self):
        """Test with very high max_retries."""
        call_count = [0]

        @retry_with_backoff(max_retries=100, base_delay=0.001, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 5:
                raise TimeoutError("Retry")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 5  # Should succeed before hitting max

    def test_very_small_base_delay(self):
        """Test with very small base delay."""

        @retry_with_backoff(max_retries=2, base_delay=0.001, jitter=False)
        def failing_func():
            raise TimeoutError("Fail")

        start_time = time.monotonic()
        with pytest.raises(RetryError):
            failing_func()
        elapsed = time.monotonic() - start_time

        # Should complete very quickly
        assert elapsed < 0.1

    def test_connection_error_retry(self):
        """Test retry on ConnectionError."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Connection failed")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 2

    def test_http_429_rate_limit_retry(self):
        """Test retry on HTTP 429 rate limit."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise MockHTTPError(429)
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 2

    def test_multiple_exception_types_in_sequence(self):
        """Test handling multiple transient exception types."""
        call_count = [0]

        @retry_with_backoff(max_retries=5, base_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("Timeout")
            if call_count[0] == 2:
                raise ConnectionError("Connection")
            if call_count[0] == 3:
                raise MockHTTPError(503)
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 4

    def test_no_jitter_produces_consistent_delays(self):
        """Test that jitter=False produces consistent delays."""
        delays1 = []
        delays2 = []

        def measure_delays():
            call_times = []

            @retry_with_backoff(max_retries=2, base_delay=0.05, jitter=False)
            def failing_func():
                call_times.append(time.monotonic())
                if len(call_times) < 3:
                    raise TimeoutError("Retry")
                return "success"

            failing_func()
            return [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        delays1 = measure_delays()
        delays2 = measure_delays()

        # Without jitter, delays should be very similar (within timing tolerance)
        for d1, d2 in zip(delays1, delays2):
            assert abs(d1 - d2) < 0.01

    def test_exception_chain_preserved(self):
        """Test that exception chain is preserved in RetryError."""

        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def failing_func():
            raise TimeoutError("Original timeout")

        with pytest.raises(RetryError) as exc_info:
            failing_func()

        # Check that the original exception is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, TimeoutError)
        assert "Original timeout" in str(exc_info.value.__cause__)

    def test_lambda_function_retry(self):
        """Test retry with lambda functions."""
        counter = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def use_lambda():
            counter[0] += 1
            func = lambda x: x * 2 if counter[0] >= 2 else (_ for _ in ()).throw(
                TimeoutError("Retry")
            )
            return func(5)

        result = use_lambda()
        assert result == 10
        assert counter[0] == 2
