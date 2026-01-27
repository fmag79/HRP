"""Tests for filter_to_trading_days function."""

import pytest
from datetime import date, timedelta
import pandas as pd

from hrp.utils.calendar import filter_to_trading_days


class TestFilterToTradingDays:
    """Tests for filter_to_trading_days function."""

    def test_filter_normal_range(self):
        """Test filtering a normal date range to trading days."""
        start = date(2023, 1, 1)  # Sunday (New Year's Day observed Jan 2)
        end = date(2023, 1, 7)   # Saturday

        filtered_start, filtered_end, trading_days = filter_to_trading_days(start, end)

        # First trading day is Tuesday Jan 3 (Monday Jan 2 was holiday)
        assert filtered_start == date(2023, 1, 3)
        # Last trading day is Friday Jan 6
        assert filtered_end == date(2023, 1, 6)
        # Should have 4 trading days (Tue-Fri)
        assert len(trading_days) == 4

    def test_filter_range_with_holidays(self):
        """Test filtering range that includes holidays."""
        # July 4, 2023 is Independence Day (Tuesday)
        start = date(2023, 7, 3)  # Monday
        end = date(2023, 7, 7)   # Friday

        filtered_start, filtered_end, trading_days = filter_to_trading_days(start, end)

        # Should skip July 4 (holiday)
        assert len(trading_days) == 4
        # First trading day is Monday July 3
        assert filtered_start == date(2023, 7, 3)
        # Last trading day is Friday July 7
        assert filtered_end == date(2023, 7, 7)

    def test_filter_single_trading_day(self):
        """Test filtering when start and end are same trading day."""
        trading_day = date(2023, 1, 3)  # Tuesday

        filtered_start, filtered_end, trading_days = filter_to_trading_days(trading_day, trading_day)

        assert filtered_start == trading_day
        assert filtered_end == trading_day
        assert len(trading_days) == 1

    def test_filter_weekend_only_range_raises_error(self):
        """Test that filtering weekend-only range raises ValueError."""
        # Saturday to Sunday (no trading days)
        start = date(2023, 1, 7)  # Saturday
        end = date(2023, 1, 8)    # Sunday

        with pytest.raises(ValueError, match="No trading days found"):
            filter_to_trading_days(start, end)

    def test_filter_returns_datetime_index(self):
        """Test that function returns DatetimeIndex."""
        start = date(2023, 1, 1)
        end = date(2023, 1, 7)

        filtered_start, filtered_end, trading_days = filter_to_trading_days(start, end)

        # Should be a DatetimeIndex
        import pandas as pd
        assert isinstance(trading_days, pd.DatetimeIndex)
