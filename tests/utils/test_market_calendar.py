"""
Tests for market calendar utilities.
"""

from datetime import date, time

import pytest

from hrp.utils.market_calendar import (
    get_market_close_time,
    get_market_open_time,
    is_market_half_day,
    is_market_holiday,
    should_run_intraday_session,
)


def test_is_market_holiday_weekday():
    """Test that regular weekdays are not holidays."""
    # Tuesday, Jan 6, 2026 - regular weekday
    assert not is_market_holiday(date(2026, 1, 6))


def test_is_market_holiday_weekend():
    """Test that weekends are considered holidays."""
    # Saturday, Jan 3, 2026
    assert is_market_holiday(date(2026, 1, 3))

    # Sunday, Jan 4, 2026
    assert is_market_holiday(date(2026, 1, 4))


def test_is_market_holiday_known_holidays():
    """Test known US market holidays for 2026."""
    # New Year's Day
    assert is_market_holiday(date(2026, 1, 1))

    # MLK Day
    assert is_market_holiday(date(2026, 1, 19))

    # Christmas
    assert is_market_holiday(date(2026, 12, 25))


def test_is_market_half_day():
    """Test half-day detection."""
    # Day before Independence Day
    assert is_market_half_day(date(2026, 7, 2))

    # Black Friday
    assert is_market_half_day(date(2026, 11, 27))

    # Christmas Eve
    assert is_market_half_day(date(2026, 12, 24))

    # Regular day - not a half day
    assert not is_market_half_day(date(2026, 6, 15))


def test_get_market_close_time_normal_day():
    """Test market close time on normal trading days."""
    close_time = get_market_close_time(date(2026, 6, 15))
    assert close_time == time(16, 0)


def test_get_market_close_time_half_day():
    """Test market close time on half-days."""
    # Black Friday
    close_time = get_market_close_time(date(2026, 11, 27))
    assert close_time == time(13, 0)


def test_get_market_open_time():
    """Test market open time (always 9:30 AM)."""
    open_time = get_market_open_time(date(2026, 6, 15))
    assert open_time == time(9, 30)


def test_should_run_intraday_session_weekday():
    """Test intraday session should run on regular weekdays."""
    # Tuesday, Jan 6, 2026
    assert should_run_intraday_session(date(2026, 1, 6))


def test_should_run_intraday_session_weekend():
    """Test intraday session should NOT run on weekends."""
    # Saturday
    assert not should_run_intraday_session(date(2026, 1, 3))

    # Sunday
    assert not should_run_intraday_session(date(2026, 1, 4))


def test_should_run_intraday_session_holiday():
    """Test intraday session should NOT run on holidays."""
    # New Year's Day
    assert not should_run_intraday_session(date(2026, 1, 1))

    # Christmas
    assert not should_run_intraday_session(date(2026, 12, 25))


def test_should_run_intraday_session_half_day():
    """Test intraday session SHOULD run on half-days (just closes early)."""
    # Black Friday - market open but closes early
    assert should_run_intraday_session(date(2026, 11, 27))
