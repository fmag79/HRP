"""
SimFin data source adapter for HRP.

Provides fundamental data (revenue, EPS, book value, etc.) with point-in-time correctness
via the publish_date field.
"""

import time
from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.sources.base import DataSourceBase
from hrp.utils.config import get_config


# SimFin metric mapping: our name -> SimFin indicator name
METRIC_MAPPING = {
    "revenue": "Revenue",
    "net_income": "Net Income",
    "eps": "Diluted EPS",
    "book_value": "Book Value",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities",
    "operating_cash_flow": "Operating Cash Flow",
    "free_cash_flow": "Free Cash Flow",
}

# Reverse mapping for lookups
SIMFIN_TO_METRIC = {v: k for k, v in METRIC_MAPPING.items()}

# Default metrics to fetch
DEFAULT_METRICS = list(METRIC_MAPPING.keys())


class RateLimiter:
    """
    Simple rate limiter for API calls.

    SimFin free tier allows 60 requests per hour.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_times: list[float] = []

    def wait_if_needed(self) -> None:
        """Block if rate limit would be exceeded."""
        now = time.time()

        # Remove requests outside the window
        self._request_times = [
            t for t in self._request_times
            if now - t < self.window_seconds
        ]

        if len(self._request_times) >= self.max_requests:
            # Calculate how long to wait
            oldest = min(self._request_times)
            wait_time = self.window_seconds - (now - oldest) + 1
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    @property
    def remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        now = time.time()
        active = [t for t in self._request_times if now - t < self.window_seconds]
        return max(0, self.max_requests - len(active))


class SimFinSource(DataSourceBase):
    """
    SimFin data source for fundamental data.

    Provides quarterly/annual fundamental data with point-in-time correctness
    using the publish_date field (date when data became publicly available).

    Free tier: 60 requests/hour, limited history.
    """

    source_name = "simfin"

    def __init__(self, api_key: str | None = None):
        """
        Initialize the SimFin source.

        Args:
            api_key: SimFin API key (falls back to SIMFIN_API_KEY env var)

        Raises:
            ValueError: If no API key is available
        """
        super().__init__()

        config = get_config()
        self.api_key = api_key or config.api.simfin_api_key

        if not self.api_key:
            raise ValueError(
                "SimFin API key required. Set SIMFIN_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._rate_limiter = RateLimiter(max_requests=60, window_seconds=3600)

        # Import simfin lazily to avoid import errors if not installed
        try:
            import simfin as sf
            sf.set_api_key(self.api_key)
            sf.set_data_dir("~/hrp-data/simfin/")
            self._sf = sf
        except ImportError:
            raise ImportError(
                "simfin package required. Install with: pip install simfin"
            )

        logger.info("SimFin data source initialized")

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Not implemented - SimFin is for fundamentals, not price data.

        Use YFinanceSource or PolygonSource for price data.
        """
        raise NotImplementedError(
            "SimFin does not provide price data. Use YFinanceSource or PolygonSource."
        )

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is available in SimFin.

        Args:
            symbol: Stock ticker to validate

        Returns:
            True if symbol has fundamental data available
        """
        try:
            self._rate_limiter.wait_if_needed()
            df = self._sf.load_income(variant="quarterly", market="us")
            return symbol in df.index.get_level_values("Ticker").unique()
        except Exception as e:
            logger.warning(f"Failed to validate symbol {symbol}: {e}")
            return False

    def get_fundamentals(
        self,
        symbol: str,
        metrics: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for a single symbol.

        Args:
            symbol: Stock ticker
            metrics: List of metrics to fetch (default: all available)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with columns: symbol, report_date, period_end, metric, value, source
            report_date is the publish_date (point-in-time)
            period_end is the fiscal period end date
        """
        metrics = metrics or DEFAULT_METRICS

        self._rate_limiter.wait_if_needed()

        try:
            # Load income statement, balance sheet, and cash flow statement data
            income_df = self._sf.load_income(variant="quarterly", market="us")
            balance_df = self._sf.load_balance(variant="quarterly", market="us")
            cashflow_df = self._sf.load_cashflow(variant="quarterly", market="us")

            # Filter by symbol
            if symbol not in income_df.index.get_level_values("Ticker").unique():
                logger.warning(f"Symbol {symbol} not found in SimFin data")
                return pd.DataFrame()

            income_data = income_df.loc[symbol] if symbol in income_df.index.get_level_values("Ticker") else pd.DataFrame()
            balance_data = balance_df.loc[symbol] if symbol in balance_df.index.get_level_values("Ticker") else pd.DataFrame()
            cashflow_data = cashflow_df.loc[symbol] if symbol in cashflow_df.index.get_level_values("Ticker") else pd.DataFrame()

            # Build result rows
            rows = []

            for metric in metrics:
                simfin_name = METRIC_MAPPING.get(metric)
                if not simfin_name:
                    logger.warning(f"Unknown metric: {metric}")
                    continue

                # Determine source dataframe based on metric
                if metric in ["revenue", "net_income", "eps"]:
                    source_df = income_data
                elif metric in ["operating_cash_flow", "free_cash_flow"]:
                    source_df = cashflow_data
                else:
                    source_df = balance_data

                if source_df.empty or simfin_name not in source_df.columns:
                    continue

                for idx, row in source_df.iterrows():
                    # Get the publish date (point-in-time correctness)
                    # SimFin uses 'Publish Date' for when data was released
                    publish_date = row.get("Publish Date")
                    if pd.isna(publish_date):
                        # Fall back to Report Date if Publish Date not available
                        publish_date = row.get("Report Date")

                    if pd.isna(publish_date):
                        continue

                    # Get period end date (fiscal quarter end)
                    if isinstance(idx, tuple):
                        period_end = idx[-1]  # Last element of multi-index
                    else:
                        period_end = idx

                    # Convert to date if needed
                    if isinstance(publish_date, pd.Timestamp):
                        publish_date = publish_date.date()
                    if isinstance(period_end, pd.Timestamp):
                        period_end = period_end.date()

                    value = row.get(simfin_name)
                    if pd.isna(value):
                        continue

                    # Apply date filters
                    if start_date and publish_date < start_date:
                        continue
                    if end_date and publish_date > end_date:
                        continue

                    rows.append({
                        "symbol": symbol,
                        "report_date": publish_date,
                        "period_end": period_end,
                        "metric": metric,
                        "value": float(value),
                        "source": self.source_name,
                    })

            if not rows:
                logger.warning(f"No fundamental data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            # Ensure point-in-time validity
            df = df[df["period_end"] <= df["report_date"]]

            logger.debug(f"Fetched {len(df)} fundamental records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            raise

    def get_fundamentals_batch(
        self,
        symbols: list[str],
        metrics: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of stock tickers
            metrics: List of metrics to fetch (default: all available)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            Combined DataFrame for all symbols
        """
        all_data = []
        failed_symbols = []

        for symbol in symbols:
            try:
                df = self.get_fundamentals(
                    symbol=symbol,
                    metrics=metrics,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        if failed_symbols:
            logger.warning(f"Failed symbols: {failed_symbols}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    @property
    def remaining_requests(self) -> int:
        """Get remaining API requests in current window."""
        return self._rate_limiter.remaining_requests
