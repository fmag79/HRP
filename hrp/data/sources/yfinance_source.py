"""
Yahoo Finance data source adapter for HRP.

Uses yfinance library to fetch price data. Free but unofficial API.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from hrp.data.sources.base import DataSourceBase


class YFinanceSource(DataSourceBase):
    """
    Yahoo Finance data source.

    Free, unofficial API. Good for development and backup.
    May break without notice as it's not an official API.
    """

    source_name = "yfinance"

    def __init__(self):
        """Initialize the Yahoo Finance source."""
        super().__init__()
        logger.info("YFinance data source initialized")

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start: Start date
            end: End date (inclusive)

        Returns:
            DataFrame with columns: date, open, high, low, close, adj_close, volume
        """
        try:
            ticker = yf.Ticker(symbol)

            # yfinance end date is exclusive, so add 1 day
            end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)

            df = ticker.history(
                start=start,
                end=end_exclusive,
                auto_adjust=False,
                actions=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol} from {start} to {end}")
                return pd.DataFrame()

            # Rename columns to our standard format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
            })

            # Convert date to date type (not datetime)
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.source_name

            # Select and order columns
            columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'source']
            df = df[columns]

            logger.debug(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise

    def get_multiple_symbols(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch daily bars for multiple symbols.

        Args:
            symbols: List of stock tickers
            start: Start date
            end: End date

        Returns:
            DataFrame with all symbols' data combined
        """
        all_data = []

        for symbol in symbols:
            try:
                df = self.get_daily_bars(symbol, start, end)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Skipping {symbol} due to error: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def get_info(self, symbol: str) -> dict[str, Any]:
        """
        Get basic info about a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with symbol info (sector, market cap, etc.)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'exchange': info.get('exchange', ''),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol}

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data.

        Args:
            symbol: Stock ticker to validate

        Returns:
            True if symbol exists and has recent data
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to fetch last 5 days
            df = ticker.history(period="5d")
            return not df.empty
        except Exception:
            return False
