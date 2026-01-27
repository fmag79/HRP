"""Database query helpers and utilities.

Provides decorators and utilities for consistent database query logging
and error handling across the platform.
"""

from functools import wraps
from typing import Callable, Any

import pandas as pd
from loguru import logger


def log_query(operation: str) -> Callable:
    """
    Decorator to log database query results.

    This decorator wraps database query functions to automatically log:
    - Warning when no results are found (empty DataFrame)
    - Debug message with record count when results are found

    Args:
        operation: Description of the operation (e.g., "price data", "features")

    Returns:
        Decorated function with automatic logging

    Example:
        @log_query("price data")
        def get_prices(symbols: List[str]) -> pd.DataFrame:
            # Database query logic here
            return df

    Note:
        Non-DataFrame return values are passed through unchanged.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Only log DataFrame results
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    logger.warning(f"No {operation} results found")
                elif hasattr(result, '__len__'):
                    logger.debug(
                        f"Retrieved {len(result)} {operation} records"
                    )

            return result

        return wrapper

    return decorator
