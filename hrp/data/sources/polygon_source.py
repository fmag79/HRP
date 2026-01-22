"""
Polygon.io data source adapter for HRP.

Official data source with rate limiting and retry logic.
Requires POLYGON_API_KEY environment variable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from polygon import RESTClient
from polygon.exceptions import BadResponse

from hrp.data.sources.base import DataSourceBase
from hrp.utils.config import get_config
from hrp.utils.rate_limiter import RateLimiter
from hrp.utils.retry import retry_with_backoff


class PolygonSource(DataSourceBase):
    """
    Polygon.io data source.

    Official, high-quality market data. Requires API key.
    Includes rate limiting (5 calls/min for Basic tier) and automatic retry.
    """

    source_name = "polygon"

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Polygon.io source.

        Args:
            api_key: Polygon.io API key. If None, loads from config/environment.
        """
        super().__init__()

        # Get API key from parameter or config
        if api_key is None:
            api_key = get_config().api.polygon_api_key

        if not api_key:
            raise ValueError(
                "POLYGON_API_KEY not found. Set it in .env or pass to constructor."
            )

        self.client = RESTClient(api_key)

        # Rate limiter: 5 calls per minute for Basic tier
        # Polygon's API uses per-minute limits, so period=60
        self.rate_limiter = RateLimiter(max_calls=5, period=60)

        logger.info("Polygon.io data source initialized")

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
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
            DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume, source
        """
        # Acquire rate limit token before making request
        self.rate_limiter.acquire()

        try:
            # Polygon uses inclusive dates, format as YYYY-MM-DD
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            logger.debug(f"Fetching {symbol} from {start_str} to {end_str}")

            # Fetch aggregates (bars) from Polygon
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str,
                adjusted=True,  # Get adjusted prices
                limit=50000,  # Max results
            )

            if not aggs:
                logger.warning(f"No data returned for {symbol} from {start} to {end}")
                return pd.DataFrame()

            # Convert to DataFrame
            rows = []
            for agg in aggs:
                rows.append({
                    'date': datetime.fromtimestamp(agg.timestamp / 1000).date(),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    # Polygon provides VWAP, we use close as adj_close
                    # (adjusted=True means close is already adjusted)
                    'adj_close': agg.close,
                })

            df = pd.DataFrame(rows)

            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.source_name

            # Select and order columns
            columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'source']
            df = df[columns]

            logger.debug(f"Fetched {len(df)} rows for {symbol}")
            return df

        except BadResponse as e:
            logger.error(f"Polygon API error for {symbol}: {e}")
            raise
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

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def get_ticker_details(self, symbol: str) -> dict[str, Any]:
        """
        Get detailed information about a ticker.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with ticker details
        """
        self.rate_limiter.acquire()

        try:
            details = self.client.get_ticker_details(symbol)

            return {
                'symbol': symbol,
                'name': details.name if hasattr(details, 'name') else '',
                'market': details.market if hasattr(details, 'market') else '',
                'locale': details.locale if hasattr(details, 'locale') else '',
                'primary_exchange': details.primary_exchange if hasattr(details, 'primary_exchange') else '',
                'type': details.type if hasattr(details, 'type') else '',
                'currency_name': details.currency_name if hasattr(details, 'currency_name') else '',
                'market_cap': details.market_cap if hasattr(details, 'market_cap') else None,
            }
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {e}")
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
            # Try to get ticker details
            self.rate_limiter.acquire()
            details = self.client.get_ticker_details(symbol)

            # Check if ticker is active and is a stock
            if hasattr(details, 'active') and not details.active:
                return False

            return True

        except BadResponse:
            # Symbol doesn't exist
            return False
        except Exception as e:
            logger.warning(f"Error validating {symbol}: {e}")
            return False

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def get_splits(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Fetch stock splits for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start: Start date
            end: End date (inclusive)

        Returns:
            DataFrame with columns: symbol, date, action_type, factor, source
        """
        self.rate_limiter.acquire()

        try:
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            logger.debug(f"Fetching splits for {symbol} from {start_str} to {end_str}")

            # Fetch splits from Polygon
            splits = self.client.list_splits(
                ticker=symbol,
                execution_date_gte=start_str,
                execution_date_lte=end_str,
                limit=1000,
            )

            if not splits:
                logger.debug(f"No splits found for {symbol} from {start} to {end}")
                return pd.DataFrame()

            # Convert to DataFrame
            rows = []
            for split in splits:
                # execution_date is the ex-date when split takes effect
                split_date = datetime.strptime(split.execution_date, "%Y-%m-%d").date()

                # Split factor (e.g., 2.0 for 2-for-1 split)
                # Polygon returns split_from and split_to (e.g., 1 and 2 for 2:1 split)
                factor = split.split_to / split.split_from if split.split_from else 1.0

                rows.append({
                    'symbol': symbol,
                    'date': split_date,
                    'action_type': 'split',
                    'factor': factor,
                    'source': self.source_name,
                })

            df = pd.DataFrame(rows)
            logger.debug(f"Fetched {len(df)} splits for {symbol}")
            return df

        except BadResponse as e:
            logger.error(f"Polygon API error fetching splits for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            raise

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def get_dividends(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Fetch dividends for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start: Start date
            end: End date (inclusive)

        Returns:
            DataFrame with columns: symbol, date, action_type, factor, source
        """
        self.rate_limiter.acquire()

        try:
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            logger.debug(f"Fetching dividends for {symbol} from {start_str} to {end_str}")

            # Fetch dividends from Polygon
            dividends = self.client.list_dividends(
                ticker=symbol,
                ex_dividend_date_gte=start_str,
                ex_dividend_date_lte=end_str,
                limit=1000,
            )

            if not dividends:
                logger.debug(f"No dividends found for {symbol} from {start} to {end}")
                return pd.DataFrame()

            # Convert to DataFrame
            rows = []
            for div in dividends:
                # ex_dividend_date is when stock trades without dividend
                div_date = datetime.strptime(div.ex_dividend_date, "%Y-%m-%d").date()

                # Cash amount per share
                cash_amount = div.cash_amount if hasattr(div, 'cash_amount') else 0.0

                rows.append({
                    'symbol': symbol,
                    'date': div_date,
                    'action_type': 'dividend',
                    'factor': cash_amount,
                    'source': self.source_name,
                })

            df = pd.DataFrame(rows)
            logger.debug(f"Fetched {len(df)} dividends for {symbol}")
            return df

        except BadResponse as e:
            logger.error(f"Polygon API error fetching dividends for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            raise

    def get_corporate_actions(
        self,
        symbol: str,
        start: date,
        end: date,
        action_types: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Fetch corporate actions for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start: Start date
            end: End date (inclusive)
            action_types: List of action types to fetch ('split', 'dividend').
                         If None, fetches all types.

        Returns:
            DataFrame with columns: symbol, date, action_type, factor, source
        """
        if action_types is None:
            action_types = ['split', 'dividend']

        all_actions = []

        # Fetch splits
        if 'split' in action_types:
            try:
                splits_df = self.get_splits(symbol, start, end)
                if not splits_df.empty:
                    all_actions.append(splits_df)
            except Exception as e:
                logger.warning(f"Error fetching splits for {symbol}: {e}")

        # Fetch dividends
        if 'dividend' in action_types:
            try:
                dividends_df = self.get_dividends(symbol, start, end)
                if not dividends_df.empty:
                    all_actions.append(dividends_df)
            except Exception as e:
                logger.warning(f"Error fetching dividends for {symbol}: {e}")

        if not all_actions:
            return pd.DataFrame(columns=['symbol', 'date', 'action_type', 'factor', 'source'])

        # Combine and sort by date
        df = pd.concat(all_actions, ignore_index=True)
        df = df.sort_values(['date', 'action_type']).reset_index(drop=True)

        return df
