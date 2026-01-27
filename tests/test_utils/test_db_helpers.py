"""Tests for database query logging utilities."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from hrp.utils.db_helpers import log_query


class TestLogQueryDecorator:
    """Tests for log_query decorator."""

    def test_logs_empty_dataframe_warning(self, caplog):
        """Test that empty DataFrame triggers warning."""
        @log_query("test operation")
        def query_func() -> pd.DataFrame:
            return pd.DataFrame()

        with patch("hrp.utils.db_helpers.logger") as mock_logger:
            result = query_func()

            # Should return the empty DataFrame
            assert result.empty
            # Should log warning
            mock_logger.warning.assert_called_once_with(
                "No test operation results found"
            )

    def test_logs_dataframe_with_records(self, caplog):
        """Test that DataFrame with records logs count."""
        @log_query("price data")
        def query_func() -> pd.DataFrame:
            return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with patch("hrp.utils.db_helpers.logger") as mock_logger:
            result = query_func()

            # Should return the DataFrame
            assert len(result) == 3
            # Should log debug message with count
            mock_logger.debug.assert_called_once()
            # Check that the message contains the count
            args = mock_logger.debug.call_args
            assert "3" in str(args) or "price data" in str(args)

    def test_logs_non_dataframe_results_unchanged(self):
        """Test that non-DataFrame results are returned unchanged."""
        @log_query("test operation")
        def query_func() -> str:
            return "string result"

        result = query_func()

        assert result == "string result"

    def test_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @log_query("custom operation")
        def my_query() -> pd.DataFrame:
            return pd.DataFrame({"data": [1]})

        # Function name should be preserved
        assert my_query.__name__ == "my_query"
        # Should work normally
        result = my_query()
        assert len(result) == 1

    def test_decorated_function_returns_correct_value(self):
        """Test that decorated function returns correct value."""
        expected_df = pd.DataFrame({"test": [1, 2, 3]})

        @log_query("test operation")
        def query_func() -> pd.DataFrame:
            return expected_df

        result = query_func()

        # Should return the exact same DataFrame object
        assert result is expected_df
        assert len(result) == 3
