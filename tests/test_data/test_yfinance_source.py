"""
Comprehensive tests for the YFinance data source.

Tests cover:
- get_corporate_actions method with various scenarios
- Dividend data handling
- Stock split data handling
- Empty results handling
- Date filtering
- Error handling
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.sources.yfinance_source import YFinanceSource


class TestGetCorporateActions:
    """Tests for the get_corporate_actions method."""

    def test_get_corporate_actions_with_dividends(self):
        """Test fetching corporate actions with dividend data."""
        source = YFinanceSource()

        # Create mock actions DataFrame with dividends
        mock_actions = pd.DataFrame({
            'Dividends': [0.25, 0.30],
            'Stock Splits': [0.0, 0.0],
        }, index=pd.DatetimeIndex(['2023-03-15', '2023-06-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['symbol', 'date', 'action_type', 'value', 'source']
        assert all(result['symbol'] == 'AAPL')
        assert all(result['action_type'] == 'dividend')
        assert result['value'].tolist() == [0.25, 0.30]
        assert all(result['source'] == 'yfinance')

    def test_get_corporate_actions_with_splits(self):
        """Test fetching corporate actions with stock split data."""
        source = YFinanceSource()

        # Create mock actions DataFrame with splits
        mock_actions = pd.DataFrame({
            'Dividends': [0.0, 0.0],
            'Stock Splits': [2.0, 3.0],
        }, index=pd.DatetimeIndex(['2023-04-20', '2023-08-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'TSLA',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert all(result['symbol'] == 'TSLA')
        assert all(result['action_type'] == 'split')
        assert result['value'].tolist() == [2.0, 3.0]
        assert all(result['source'] == 'yfinance')

    def test_get_corporate_actions_with_both(self):
        """Test fetching corporate actions with both dividends and splits."""
        source = YFinanceSource()

        # Create mock actions DataFrame with both dividends and splits
        mock_actions = pd.DataFrame({
            'Dividends': [0.22, 0.0, 0.23],
            'Stock Splits': [0.0, 4.0, 0.0],
        }, index=pd.DatetimeIndex(['2023-03-15', '2023-06-20', '2023-09-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result.iloc[0]['action_type'] == 'dividend'
        assert result.iloc[0]['value'] == 0.22
        assert result.iloc[1]['action_type'] == 'split'
        assert result.iloc[1]['value'] == 4.0
        assert result.iloc[2]['action_type'] == 'dividend'
        assert result.iloc[2]['value'] == 0.23

    def test_get_corporate_actions_empty_result(self):
        """Test get_corporate_actions returns empty DataFrame when no actions exist."""
        source = YFinanceSource()

        # Create empty mock actions DataFrame
        mock_actions = pd.DataFrame()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'MSFT',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_corporate_actions_outside_date_range(self):
        """Test filtering actions outside the requested date range."""
        source = YFinanceSource()

        # Create mock actions DataFrame with dates outside range
        mock_actions = pd.DataFrame({
            'Dividends': [0.25, 0.30],
            'Stock Splits': [0.0, 0.0],
        }, index=pd.DatetimeIndex(['2022-12-15', '2024-01-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should be empty since all actions are outside the date range
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_corporate_actions_partial_date_range(self):
        """Test filtering actions with some inside and some outside date range."""
        source = YFinanceSource()

        # Create mock actions DataFrame with mixed dates
        mock_actions = pd.DataFrame({
            'Dividends': [0.20, 0.25, 0.30],
            'Stock Splits': [0.0, 0.0, 0.0],
        }, index=pd.DatetimeIndex(['2022-12-15', '2023-06-15', '2024-01-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should only include the middle action
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['value'] == 0.25
        assert result.iloc[0]['date'] == date(2023, 6, 15)

    def test_get_corporate_actions_zero_dividend_filtered(self):
        """Test that zero dividends are filtered out."""
        source = YFinanceSource()

        # Create mock actions DataFrame with zero dividends
        mock_actions = pd.DataFrame({
            'Dividends': [0.0, 0.25, 0.0],
            'Stock Splits': [0.0, 0.0, 0.0],
        }, index=pd.DatetimeIndex(['2023-03-15', '2023-06-15', '2023-09-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should only include the non-zero dividend
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['value'] == 0.25

    def test_get_corporate_actions_zero_split_filtered(self):
        """Test that zero splits (no split) are filtered out."""
        source = YFinanceSource()

        # Create mock actions DataFrame with zero splits
        mock_actions = pd.DataFrame({
            'Dividends': [0.0, 0.0, 0.0],
            'Stock Splits': [0.0, 2.0, 0.0],
        }, index=pd.DatetimeIndex(['2023-03-15', '2023-06-15', '2023-09-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'TSLA',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should only include the non-zero split
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['action_type'] == 'split'
        assert result.iloc[0]['value'] == 2.0

    def test_get_corporate_actions_date_conversion(self):
        """Test that dates are properly converted to date objects."""
        source = YFinanceSource()

        # Create mock actions DataFrame
        mock_actions = pd.DataFrame({
            'Dividends': [0.25],
            'Stock Splits': [0.0],
        }, index=pd.DatetimeIndex(['2023-06-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        assert len(result) == 1
        # Verify the date is a date object, not datetime
        assert isinstance(result.iloc[0]['date'], date)
        assert result.iloc[0]['date'] == date(2023, 6, 15)

    def test_get_corporate_actions_column_order(self):
        """Test that returned DataFrame has correct column order."""
        source = YFinanceSource()

        # Create mock actions DataFrame
        mock_actions = pd.DataFrame({
            'Dividends': [0.25],
            'Stock Splits': [0.0],
        }, index=pd.DatetimeIndex(['2023-06-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Verify column order matches specification
        expected_columns = ['symbol', 'date', 'action_type', 'value', 'source']
        assert list(result.columns) == expected_columns

    def test_get_corporate_actions_raises_on_error(self):
        """Test that exceptions are raised when yfinance fails."""
        source = YFinanceSource()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API Error")

            with pytest.raises(Exception) as exc_info:
                source.get_corporate_actions(
                    'INVALID',
                    date(2023, 1, 1),
                    date(2023, 12, 31)
                )

            assert "API Error" in str(exc_info.value)

    def test_get_corporate_actions_multiple_actions_same_date(self):
        """Test handling when both dividend and split occur on same date."""
        source = YFinanceSource()

        # Create mock actions DataFrame with both on same date
        mock_actions = pd.DataFrame({
            'Dividends': [0.25],
            'Stock Splits': [2.0],
        }, index=pd.DatetimeIndex(['2023-06-15'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should have two rows for the same date
        assert len(result) == 2
        assert result.iloc[0]['date'] == result.iloc[1]['date']
        assert result.iloc[0]['action_type'] == 'dividend'
        assert result.iloc[1]['action_type'] == 'split'

    def test_get_corporate_actions_boundary_dates(self):
        """Test corporate actions on exact start and end dates."""
        source = YFinanceSource()

        # Create mock actions DataFrame with dates on boundaries
        mock_actions = pd.DataFrame({
            'Dividends': [0.20, 0.25, 0.30],
            'Stock Splits': [0.0, 0.0, 0.0],
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-06-15', '2023-12-31'], name='Date'))

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.actions = mock_actions
            mock_ticker.return_value = mock_instance

            result = source.get_corporate_actions(
                'AAPL',
                date(2023, 1, 1),
                date(2023, 12, 31)
            )

        # Should include all three (boundary dates are inclusive)
        assert len(result) == 3
        dates = result['date'].tolist()
        assert date(2023, 1, 1) in dates
        assert date(2023, 6, 15) in dates
        assert date(2023, 12, 31) in dates
