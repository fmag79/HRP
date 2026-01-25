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
    
    result = sessions[0].date()

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
    
    result = sessions[-1].date()

    logger.debug(f"get_previous_trading_day({from_date}): {result}")
    return result
