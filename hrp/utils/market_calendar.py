"""
Market calendar utilities for US stock market.

Provides holiday detection and market hours information.
"""

from datetime import date, time

# US stock market holidays for 2026
# Source: NYSE calendar
US_MARKET_HOLIDAYS_2026 = [
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
]

# Half-day early close dates for 2026 (close at 1:00 PM ET)
US_MARKET_HALF_DAYS_2026 = [
    date(2026, 7, 2),   # Day before Independence Day
    date(2026, 11, 27), # Day after Thanksgiving (Black Friday)
    date(2026, 12, 24), # Christmas Eve
]


def is_market_holiday(check_date: date) -> bool:
    """
    Check if a date is a US stock market holiday.

    Args:
        check_date: Date to check

    Returns:
        True if market is closed for holiday, False otherwise
    """
    # Check if weekend
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return True

    # Check if in holiday list
    return check_date in US_MARKET_HOLIDAYS_2026


def is_market_half_day(check_date: date) -> bool:
    """
    Check if a date is a half-day (early close at 1:00 PM ET).

    Args:
        check_date: Date to check

    Returns:
        True if market closes early, False otherwise
    """
    return check_date in US_MARKET_HALF_DAYS_2026


def get_market_close_time(check_date: date) -> time:
    """
    Get market close time for a given date.

    Args:
        check_date: Date to check

    Returns:
        Market close time (16:00 for normal days, 13:00 for half-days)
    """
    if is_market_half_day(check_date):
        return time(13, 0)  # 1:00 PM ET
    return time(16, 0)  # 4:00 PM ET


def get_market_open_time(check_date: date) -> time:
    """
    Get market open time for a given date.

    Args:
        check_date: Date to check

    Returns:
        Market open time (always 9:30 AM ET)
    """
    return time(9, 30)  # 9:30 AM ET


def should_run_intraday_session(check_date: date) -> bool:
    """
    Check if intraday ingestion should run on a given date.

    Args:
        check_date: Date to check

    Returns:
        True if intraday session should run, False if market is closed
    """
    return not is_market_holiday(check_date)
