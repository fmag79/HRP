"""
Retry utility with exponential backoff for handling transient failures.

Provides decorator and function wrapper for automatic retry logic with
exponential backoff and jitter to handle API rate limits and transient errors.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


def is_transient_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a transient error worth retrying.

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried
    """
    # Timeout errors
    if isinstance(exception, TimeoutError):
        return True

    # Connection errors
    if exception.__class__.__name__ in (
        "ConnectionError",
        "ConnectionResetError",
        "BrokenPipeError",
    ):
        return True

    # HTTP errors (checking by name to avoid importing requests/httpx)
    if hasattr(exception, "response"):
        response = getattr(exception, "response")
        if hasattr(response, "status_code"):
            status_code = response.status_code
            # Retry on 5xx server errors and 429 rate limit
            if status_code >= 500 or status_code == 429:
                return True

    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to retry a function with exponential backoff.

    Retries the decorated function on transient failures with exponentially
    increasing delays between attempts. Adds jitter to prevent thundering herd.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to delays (default: True)
        exceptions: Tuple of exception types to catch (default: (Exception,))
        on_retry: Optional callback called on each retry with (exception, attempt, delay)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def fetch_data(url: str) -> dict:
        ...     response = requests.get(url)
        ...     response.raise_for_status()
        ...     return response.json()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        break

                    # Only retry if it's a transient error
                    if not is_transient_error(e):
                        logger.debug(
                            f"Non-transient error in {func.__name__}, not retrying: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    time.sleep(delay)

            # All retries exhausted
            error_msg = f"Failed after {max_retries + 1} attempts: {func.__name__}"
            logger.error(f"{error_msg}. Last error: {last_exception}")
            raise RetryError(error_msg, last_exception) from last_exception

        return wrapper  # type: ignore

    return decorator


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Call a function with retry logic without using a decorator.

    Useful for one-off calls that need retry logic without decorating the function.

    Args:
        func: Function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to delays (default: True)
        exceptions: Tuple of exception types to catch (default: (Exception,))
        on_retry: Optional callback called on each retry with (exception, attempt, delay)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Example:
        >>> result = retry_call(
        ...     requests.get,
        ...     "https://api.example.com",
        ...     max_retries=5,
        ...     base_delay=2.0
        ... )
    """
    decorated_func = retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions,
        on_retry=on_retry,
    )(func)

    return decorated_func(*args, **kwargs)
