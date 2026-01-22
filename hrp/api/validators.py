"""
Input validation functions for Platform API.

Provides reusable validation functions for symbols, dates, and parameters.
All validators raise ValueError with descriptive messages on validation failure.
"""

from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.data.db import DatabaseManager, get_db


def validate_symbols(
    symbols: list[str],
    as_of_date: date | None = None,
    db: DatabaseManager | None = None,
) -> None:
    """
    Validate that symbols exist in the universe.

    Args:
        symbols: List of ticker symbols to validate
        as_of_date: Optional date to check universe membership (uses latest if not provided)
        db: Optional database manager instance (creates new if not provided)

    Raises:
        ValueError: If symbols list is empty or contains invalid symbols

    Example:
        validate_symbols(['AAPL', 'MSFT'])  # Validates against current universe
        validate_symbols(['AAPL'], as_of_date=date(2023, 1, 1))  # Validates at specific date
    """
    if not symbols:
        raise ValueError("symbols list cannot be empty")

    if db is None:
        db = get_db()

    # Get valid symbols from universe
    if as_of_date:
        query = """
            SELECT DISTINCT symbol
            FROM universe
            WHERE date = ?
              AND in_universe = TRUE
        """
        valid_symbols = {row[0] for row in db.fetchall(query, (as_of_date,))}
    else:
        # Get all symbols that have ever been in universe
        query = """
            SELECT DISTINCT symbol
            FROM universe
            WHERE in_universe = TRUE
        """
        valid_symbols = {row[0] for row in db.fetchall(query)}

    # Check for invalid symbols
    invalid_symbols = [s for s in symbols if s not in valid_symbols]

    if invalid_symbols:
        if as_of_date:
            raise ValueError(
                f"Invalid symbols not in universe as of {as_of_date}: {', '.join(invalid_symbols)}"
            )
        else:
            raise ValueError(
                f"Invalid symbols not found in universe: {', '.join(invalid_symbols)}"
            )

    logger.debug(f"Validated {len(symbols)} symbols")


def validate_date(
    date_value: date,
    parameter_name: str = "date",
) -> None:
    """
    Validate that a date is not in the future.

    Args:
        date_value: Date to validate
        parameter_name: Name of the parameter for error messages (default 'date')

    Raises:
        ValueError: If date is in the future or invalid

    Example:
        validate_date(date(2023, 1, 1))  # Valid past date
        validate_date(date.today())  # Valid current date
        validate_date(date(2099, 1, 1))  # Raises ValueError
    """
    if not isinstance(date_value, date):
        raise ValueError(f"{parameter_name} must be a date object, got {type(date_value).__name__}")

    today = date.today()
    if date_value > today:
        raise ValueError(
            f"{parameter_name} cannot be in the future: {date_value} > {today}"
        )

    logger.debug(f"Validated {parameter_name}: {date_value}")


def validate_date_range(
    start: date,
    end: date,
) -> None:
    """
    Validate that a date range is valid.

    Checks that both dates are valid (not in future) and that start <= end.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Raises:
        ValueError: If either date is invalid or start > end

    Example:
        validate_date_range(date(2023, 1, 1), date(2023, 12, 31))  # Valid range
        validate_date_range(date(2023, 6, 1), date(2023, 1, 1))  # Raises ValueError
    """
    # Validate individual dates
    validate_date(start, "start date")
    validate_date(end, "end date")

    # Check that start <= end
    if start > end:
        raise ValueError(
            f"start date must be <= end date: {start} > {end}"
        )

    logger.debug(f"Validated date range: {start} to {end}")


def validate_positive_int(
    value: Any,
    parameter_name: str = "value",
    allow_zero: bool = False,
) -> None:
    """
    Validate that a parameter is a positive integer.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages (default 'value')
        allow_zero: Whether to allow zero (default False)

    Raises:
        ValueError: If value is not an integer or not positive

    Example:
        validate_positive_int(10, "limit")  # Valid positive integer
        validate_positive_int(0, "offset", allow_zero=True)  # Valid with allow_zero
        validate_positive_int(-5, "count")  # Raises ValueError
        validate_positive_int(3.14, "num")  # Raises ValueError (not an integer)
    """
    if not isinstance(value, int):
        raise ValueError(
            f"{parameter_name} must be an integer, got {type(value).__name__}"
        )

    if allow_zero:
        if value < 0:
            raise ValueError(
                f"{parameter_name} must be non-negative, got {value}"
            )
    else:
        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be positive, got {value}"
            )

    logger.debug(f"Validated {parameter_name}: {value}")


def validate_non_empty_string(
    value: Any,
    parameter_name: str = "value",
    strip: bool = True,
) -> None:
    """
    Validate that a parameter is a non-empty string.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages (default 'value')
        strip: Whether to check after stripping whitespace (default True)

    Raises:
        ValueError: If value is not a string or is empty

    Example:
        validate_non_empty_string("AAPL", "symbol")  # Valid non-empty string
        validate_non_empty_string("  content  ", "text")  # Valid (has content after strip)
        validate_non_empty_string("", "title")  # Raises ValueError
        validate_non_empty_string("   ", "name")  # Raises ValueError (empty after strip)
        validate_non_empty_string(123, "label")  # Raises ValueError (not a string)
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{parameter_name} must be a string, got {type(value).__name__}"
        )

    check_value = value.strip() if strip else value
    if not check_value:
        if strip:
            raise ValueError(
                f"{parameter_name} cannot be empty or whitespace-only"
            )
        else:
            raise ValueError(
                f"{parameter_name} cannot be empty"
            )

    logger.debug(f"Validated {parameter_name}: {value!r}")
