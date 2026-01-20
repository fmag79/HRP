"""
Base class for data source adapters.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class DataSourceBase(ABC):
    """
    Abstract base class for data sources.

    All data source adapters should inherit from this class.
    """

    source_name: str = "base"

    def __init__(self):
        """Initialize the data source."""
        pass

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Stock ticker
            start: Start date
            end: End date

        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume, source
        """
        pass

    def get_multiple_symbols(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch daily bars for multiple symbols.

        Default implementation calls get_daily_bars for each symbol.
        Override for more efficient batch fetching.
        """
        all_data = []
        for symbol in symbols:
            df = self.get_daily_bars(symbol, start, end)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid."""
        pass
