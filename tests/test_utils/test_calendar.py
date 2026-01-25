"""
Tests for NYSE trading calendar utilities.
"""

from datetime import date

import pandas as pd
import pytest

from hrp.utils.calendar import (
    get_next_trading_day,
    get_nyse_calendar,
    get_previous_trading_day,
    get_trading_days,
    is_trading_day,
)


class TestGetNYSECalendar:
    """Tests for get_nyse_calendar()."""

    def test_returns_calendar_instance(self):
        """Should return an exchange_calendars instance for XNYS."""
        cal = get_nyse_calendar()
        assert cal is not None
        assert cal.name == "XNYS"

    def test_calendar_is_cached(self):
        """Multiple calls should return the same cached instance."""
        cal1 = get_nyse_calendar()
        cal2 = get_nyse_calendar()
        assert cal1 is cal2  # Same object in memory


class TestIsTradingDay:
    """Tests for is_trading_day()."""

    def test_regular_weekday(self):
        """Regular weekdays should be trading days."""
        # Tuesday, March 15, 2022
        assert is_trading_day(date(2022, 3, 15)) is True

        # Wednesday, June 8, 2022
        assert is_trading_day(date(2022, 6, 8)) is True

        # Friday, September 16, 2022
        assert is_trading_day(date(2022, 9, 16)) is True

    def test_weekend_saturday(self):
        """Saturdays should not be trading days."""
        # Saturday, July 2, 2022
        assert is_trading_day(date(2022, 7, 2)) is False

        # Saturday, December 31, 2022
        assert is_trading_day(date(2022, 12, 31)) is False

    def test_weekend_sunday(self):
        """Sundays should not be trading days."""
        # Sunday, July 3, 2022
        assert is_trading_day(date(2022, 7, 3)) is False

        # Sunday, January 1, 2023
        assert is_trading_day(date(2023, 1, 1)) is False

    def test_independence_day(self):
        """Independence Day (July 4) should not be a trading day."""
        # Monday, July 4, 2022
        assert is_trading_day(date(2022, 7, 4)) is False

    def test_christmas_observed(self):
        """Christmas observed should not be a trading day."""
        # Monday, December 26, 2022 (Christmas observed, as Dec 25 was Sunday)
        assert is_trading_day(date(2022, 12, 26)) is False

    def test_thanksgiving(self):
        """Thanksgiving should not be a trading day."""
        # Thursday, November 24, 2022
        assert is_trading_day(date(2022, 11, 24)) is False

    def test_mlk_day(self):
        """Martin Luther King Jr. Day should not be a trading day."""
        # Monday, January 17, 2022
        assert is_trading_day(date(2022, 1, 17)) is False

    def test_presidents_day(self):
        """Presidents' Day should not be a trading day."""
        # Monday, February 21, 2022
        assert is_trading_day(date(2022, 2, 21)) is False

    def test_good_friday(self):
        """Good Friday should not be a trading day."""
        # Friday, April 15, 2022
        assert is_trading_day(date(2022, 4, 15)) is False

    def test_memorial_day(self):
        """Memorial Day should not be a trading day."""
        # Monday, May 30, 2022
        assert is_trading_day(date(2022, 5, 30)) is False

    def test_juneteenth(self):
        """Juneteenth should not be a trading day."""
        # Monday, June 20, 2022 (Juneteenth observed, as June 19 was Sunday)
        assert is_trading_day(date(2022, 6, 20)) is False

    def test_labor_day(self):
        """Labor Day should not be a trading day."""
        # Monday, September 5, 2022
        assert is_trading_day(date(2022, 9, 5)) is False

    def test_new_years_day_observed(self):
        """New Year's Day observed should not be a trading day."""
        # Monday, January 2, 2023 (New Year's observed, as Jan 1 was Sunday)
        assert is_trading_day(date(2023, 1, 2)) is False

    def test_early_close_day_is_trading_day(self):
        """Early close days (e.g., day after Thanksgiving) should still be trading days."""
        # Friday, November 25, 2022 (day after Thanksgiving, early close)
        assert is_trading_day(date(2022, 11, 25)) is True

    def test_invalid_input_type(self):
        """Should raise TypeError for non-date input."""
        with pytest.raises(TypeError):
            is_trading_day("2022-07-04")

        with pytest.raises(TypeError):
            is_trading_day(20220704)


class TestGetTradingDays:
    """Tests for get_trading_days()."""

    def test_january_2022(self):
        """January 2022 should have 20 trading days."""
        days = get_trading_days(date(2022, 1, 1), date(2022, 1, 31))
        assert len(days) == 20
        assert isinstance(days, pd.DatetimeIndex)

        # Check first and last
        assert days[0].date() == date(2022, 1, 3)  # First Monday (Jan 1-2 were weekend)
        assert days[-1].date() == date(2022, 1, 31)  # Last Monday

    def test_excludes_mlk_day(self):
        """January 2022 range should exclude MLK Day (Jan 17)."""
        days = get_trading_days(date(2022, 1, 1), date(2022, 1, 31))
        dates = [d.date() for d in days]
        assert date(2022, 1, 17) not in dates  # MLK Day

    def test_excludes_memorial_day(self):
        """May 2022 range should exclude Memorial Day (May 30)."""
        days = get_trading_days(date(2022, 5, 1), date(2022, 5, 31))
        dates = [d.date() for d in days]
        assert date(2022, 5, 30) not in dates  # Memorial Day

    def test_july_4th_week(self):
        """July 1-8, 2022 should exclude weekend and July 4th."""
        days = get_trading_days(date(2022, 7, 1), date(2022, 7, 8))
        dates = [d.date() for d in days]

        # Should include: July 1 (Fri), 5 (Tue), 6 (Wed), 7 (Thu), 8 (Fri)
        assert date(2022, 7, 1) in dates
        assert date(2022, 7, 5) in dates
        assert date(2022, 7, 6) in dates
        assert date(2022, 7, 7) in dates
        assert date(2022, 7, 8) in dates

        # Should exclude: July 2-3 (weekend), July 4 (holiday)
        assert date(2022, 7, 2) not in dates
        assert date(2022, 7, 3) not in dates
        assert date(2022, 7, 4) not in dates

        assert len(days) == 5

    def test_thanksgiving_week(self):
        """Thanksgiving week 2022 should exclude Thursday (holiday) but include Friday."""
        days = get_trading_days(date(2022, 11, 21), date(2022, 11, 25))
        dates = [d.date() for d in days]

        # Should include: Nov 21 (Mon), 22 (Tue), 23 (Wed), 25 (Fri - early close)
        assert date(2022, 11, 21) in dates
        assert date(2022, 11, 22) in dates
        assert date(2022, 11, 23) in dates
        assert date(2022, 11, 25) in dates

        # Should exclude: Nov 24 (Thanksgiving)
        assert date(2022, 11, 24) not in dates

        assert len(days) == 4

    def test_full_year_2022(self):
        """Full year 2022 should have 251 trading days."""
        days = get_trading_days(date(2022, 1, 1), date(2022, 12, 31))
        assert len(days) == 251

    def test_single_day_trading(self):
        """Single trading day range should return one day."""
        days = get_trading_days(date(2022, 7, 5), date(2022, 7, 5))
        assert len(days) == 1
        assert days[0].date() == date(2022, 7, 5)

    def test_single_day_holiday(self):
        """Single day range on holiday should return empty."""
        days = get_trading_days(date(2022, 7, 4), date(2022, 7, 4))
        assert len(days) == 0

    def test_single_day_weekend(self):
        """Single day range on weekend should return empty."""
        days = get_trading_days(date(2022, 7, 2), date(2022, 7, 2))
        assert len(days) == 0

    def test_weekend_only_range(self):
        """Range containing only weekend should return empty."""
        days = get_trading_days(date(2022, 7, 2), date(2022, 7, 3))
        assert len(days) == 0

    def test_chronological_order(self):
        """Trading days should be in chronological order."""
        days = get_trading_days(date(2022, 1, 1), date(2022, 12, 31))
        dates = [d.date() for d in days]
        assert dates == sorted(dates)

    def test_start_greater_than_end_raises(self):
        """Should raise ValueError if start > end."""
        with pytest.raises(ValueError, match="must be <= end date"):
            get_trading_days(date(2022, 12, 31), date(2022, 1, 1))

    def test_invalid_start_type(self):
        """Should raise TypeError for non-date start."""
        with pytest.raises(TypeError):
            get_trading_days("2022-01-01", date(2022, 12, 31))

    def test_invalid_end_type(self):
        """Should raise TypeError for non-date end."""
        with pytest.raises(TypeError):
            get_trading_days(date(2022, 1, 1), "2022-12-31")


class TestGetNextTradingDay:
    """Tests for get_next_trading_day()."""

    def test_from_trading_day(self):
        """If input is already a trading day, should return same day."""
        # Tuesday, July 5, 2022
        result = get_next_trading_day(date(2022, 7, 5))
        assert result == date(2022, 7, 5)

    def test_from_saturday(self):
        """From Saturday should return next Monday (or Tuesday if Monday is holiday)."""
        # Saturday, July 2, 2022 -> Tuesday, July 5 (Monday July 4 is holiday)
        result = get_next_trading_day(date(2022, 7, 2))
        assert result == date(2022, 7, 5)

    def test_from_sunday(self):
        """From Sunday should return next Monday (or Tuesday if Monday is holiday)."""
        # Sunday, July 3, 2022 -> Tuesday, July 5 (Monday July 4 is holiday)
        result = get_next_trading_day(date(2022, 7, 3))
        assert result == date(2022, 7, 5)

    def test_from_holiday(self):
        """From holiday should return next trading day."""
        # Monday, July 4, 2022 (Independence Day) -> Tuesday, July 5
        result = get_next_trading_day(date(2022, 7, 4))
        assert result == date(2022, 7, 5)

    def test_from_friday_before_holiday_weekend(self):
        """From Friday before holiday weekend should return next trading day."""
        # Friday, December 23, 2022 -> Tuesday, December 27
        # (Dec 24-25 weekend, Dec 26 Christmas observed)
        result = get_next_trading_day(date(2022, 12, 23))
        assert result == date(2022, 12, 23)  # Friday is a trading day

    def test_invalid_input_type(self):
        """Should raise TypeError for non-date input."""
        with pytest.raises(TypeError):
            get_next_trading_day("2022-07-04")


class TestGetPreviousTradingDay:
    """Tests for get_previous_trading_day()."""

    def test_from_trading_day(self):
        """If input is already a trading day, should return same day."""
        # Tuesday, July 5, 2022
        result = get_previous_trading_day(date(2022, 7, 5))
        assert result == date(2022, 7, 5)

    def test_from_saturday(self):
        """From Saturday should return previous Friday."""
        # Saturday, July 2, 2022 -> Friday, July 1
        result = get_previous_trading_day(date(2022, 7, 2))
        assert result == date(2022, 7, 1)

    def test_from_sunday(self):
        """From Sunday should return previous Friday."""
        # Sunday, July 3, 2022 -> Friday, July 1
        result = get_previous_trading_day(date(2022, 7, 3))
        assert result == date(2022, 7, 1)

    def test_from_holiday(self):
        """From holiday should return previous trading day."""
        # Monday, July 4, 2022 (Independence Day) -> Friday, July 1
        result = get_previous_trading_day(date(2022, 7, 4))
        assert result == date(2022, 7, 1)

    def test_from_monday_after_holiday_weekend(self):
        """From Monday after holiday weekend should return previous trading day."""
        # Tuesday, July 5, 2022 -> Tuesday, July 5 (it's a trading day)
        result = get_previous_trading_day(date(2022, 7, 5))
        assert result == date(2022, 7, 5)

    def test_invalid_input_type(self):
        """Should raise TypeError for non-date input."""
        with pytest.raises(TypeError):
            get_previous_trading_day("2022-07-04")
