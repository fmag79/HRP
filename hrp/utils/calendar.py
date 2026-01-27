"""
NYSE trading calendar utilities.

Provides centralized access to NYSE (XNYS) trading calendar for filtering
backtests and feature computations to valid trading days only.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd
from loguru import logger


@lru_cache(maxsize=1)
def get_nyse_calendar() -> xcals.ExchangeCalendar:
    """
    Get the NYSE exchange calendar instance.

    Returns a cached instance to avoid repeated initialization.
    Supports date range from 1990-2050 minimum.

    Returns:
        ExchangeCalendar instance for NYSE (XNYS)
    """
    logger.debug("Initializing NYSE calendar (XNYS)")
    return xcals.get_calendar("XNYS")


def is_trading_day(trading_date: date) -> bool:
    """
    Check if a date is a valid NYSE trading day.

    Args:
        trading_date: Date to check

    Returns:
        True if the date is a weekday and not a NYSE holiday, False otherwise

    Examples:
        >>> is_trading_day(date(2022, 7, 4))  # Independence Day
        False
        >>> is_trading_day(date(2022, 7, 5))  # Regular Tuesday
        True
        >>> is_trading_day(date(2022, 7, 2))  # Saturday
        False
    """
    if not isinstance(trading_date, date):
        raise TypeError(f"Expected date object, got {type(trading_date)}")

    cal = get_nyse_calendar()
    # Convert date to pandas Timestamp for exchange_calendars
    ts = pd.Timestamp(trading_date)
    result = cal.is_session(ts)

    logger.debug(f"is_trading_day({trading_date}): {result}")
    return bool(result)


def get_trading_days(start: date, end: date) -> pd.DatetimeIndex:
    """
    Get all NYSE trading days within a date range (inclusive).

    Excludes weekends and NYSE holidays. Returns empty index if range
    contains no trading days.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DatetimeIndex of trading days in chronological order

    Raises:
        ValueError: If start > end

    Examples:
        >>> days = get_trading_days(date(2022, 1, 1), date(2022, 1, 31))
        >>> len(days)  # January 2022 has 20 trading days
        20
        >>> date(2022, 1, 17) in days.date  # MLK Day (holiday)
        False
    """
    if not isinstance(start, date):
        raise TypeError(f"start must be date object, got {type(start)}")
    if not isinstance(end, date):
        raise TypeError(f"end must be date object, got {type(end)}")
    if start > end:
        raise ValueError(f"start date {start} must be <= end date {end}")

    cal = get_nyse_calendar()

    # Convert to pandas Timestamps
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Get trading sessions in range
    sessions = cal.sessions_in_range(start_ts, end_ts)

    logger.debug(
        f"get_trading_days({start}, {end}): {len(sessions)} trading days "
        f"(first={sessions[0].date() if len(sessions) > 0 else None}, "
        f"last={sessions[-1].date() if len(sessions) > 0 else None})"
    )

    if len(sessions) == 0:
        logger.warning(f"No trading days found between {start} and {end}")

    return sessions


def get_next_trading_day(from_date: date) -> date:
    """
    Get the next trading day on or after the given date.

    Args:
        from_date: Starting date

    Returns:
        Next trading day (could be from_date itself if it's a trading day)

    Examples:
        >>> get_next_trading_day(date(2022, 7, 2))  # Saturday
        datetime.date(2022, 7, 5)  # Tuesday (Monday was July 4th holiday)
    """
    if not isinstance(from_date, date):
        raise TypeError(f"Expected date object, got {type(from_date)}")

    cal = get_nyse_calendar()
    ts = pd.Timestamp(from_date)

    # If it's already a trading day, return it
    if cal.is_session(ts):
        return from_date

    # Otherwise find the next session by getting trading days in a range
    # Look ahead up to 10 days (should cover any holiday period)
    end_date = from_date + pd.Timedelta(days=10)
    sessions = cal.sessions_in_range(ts, pd.Timestamp(end_date))
    
    if len(sessions) == 0:
        raise ValueError(f"No trading days found after {from_date}")
    
    result: date = sessions[0].date()

    logger.debug(f"get_next_trading_day({from_date}): {result}")
    return result


def get_previous_trading_day(from_date: date) -> date:
    """
    Get the previous trading day on or before the given date.

    Args:
        from_date: Starting date

    Returns:
        Previous trading day (could be from_date itself if it's a trading day)

    Examples:
        >>> get_previous_trading_day(date(2022, 7, 4))  # Independence Day (Monday)
        datetime.date(2022, 7, 1)  # Friday
    """
    if not isinstance(from_date, date):
        raise TypeError(f"Expected date object, got {type(from_date)}")

    cal = get_nyse_calendar()
    ts = pd.Timestamp(from_date)

    # If it's already a trading day, return it
    if cal.is_session(ts):
        return from_date

    # Otherwise find the previous session by getting trading days in a range
    # Look back up to 10 days (should cover any holiday period)
    start_date = from_date - pd.Timedelta(days=10)
    sessions = cal.sessions_in_range(pd.Timestamp(start_date), ts)

    if len(sessions) == 0:
        raise ValueError(f"No trading days found before {from_date}")

    result: date = sessions[-1].date()

    logger.debug(f"get_previous_trading_day({from_date}): {result}")
    return result


def filter_to_trading_days(start: date, end: date) -> tuple[date, date, pd.DatetimeIndex]:
    """
    Filter date range to NYSE trading days only.

    This utility function filters a date range to only include NYSE trading days,
    excluding weekends and holidays. It's useful for backtesting and feature computation
    where you need to ensure you're only analyzing valid trading days.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        Tuple of (filtered_start, filtered_end, trading_days)
        - filtered_start: First trading day in range (as date)
        - filtered_end: Last trading day in range (as date)
        - trading_days: DatetimeIndex of all trading days in range

    Raises:
        ValueError: If no trading days found in the range

    Examples:
        >>> # Monday to Sunday (excludes weekend)
        >>> start, end, days = filter_to_trading_days(date(2023, 1, 2), date(2023, 1, 8))
        >>> len(days)
        5
        >>> # Range with holiday
        >>> start, end, days = filter_to_trading_days(date(2023, 7, 3), date(2023, 7, 7))
        >>> len(days)  # Skips July 4 (Independence Day)
        4

    Note:
        This function is particularly useful when you want to ensure your analysis
        only includes days when the market was open. It automatically adjusts the
        start and end dates to the nearest trading days.
    """
    trading_days = get_trading_days(start, end)

    if len(trading_days) == 0:
        raise ValueError(f"No trading days found between {start} and {end}")

    # Extract first and last trading days as date objects
    filtered_start = trading_days[0].date()
    filtered_end = trading_days[-1].date()

    logger.debug(
        f"filter_to_trading_days({start}, {end}): "
        f"filtered to {len(trading_days)} trading days "
        f"({filtered_start} to {filtered_end})"
    )

    return filtered_start, filtered_end, trading_days
