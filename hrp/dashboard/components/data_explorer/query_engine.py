"""
Query Engine for Data Explorer

Provides cached SQL queries for each data type with pagination
and performance optimization.

NOTE: All query functions are static with @st.cache_data decorator.
The QueryEngine class serves as a namespace for organization.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from hrp.api.platform import PlatformAPI

def _get_api():
    return PlatformAPI()


class QueryEngine:
    """Efficient SQL query builder for data explorer.

    This class serves as a namespace for query functions.
    All methods are static to work with Streamlit's @st.cache_data decorator.
    """

    # -------------------------------------------------------------------------
    # Price Data Queries
    # -------------------------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=300)
    def get_price_ohlcv(
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Get OHLCV price data for visualization.

        Args:
            symbols: List of symbols (None = all)
            start_date: Start date filter
            end_date: End date filter
            limit: Max rows to return

        Returns:
            DataFrame with OHLCV data
        """
        api = _get_api()
        conditions = []
        params = []

        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        if start_date:
            conditions.append("date >= ?")
            params.append(str(start_date))

        if end_date:
            conditions.append("date <= ?")
            params.append(str(end_date))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume
            FROM prices
            {where_clause}
            ORDER BY symbol, date
            LIMIT {limit}
        """

        return api.query_readonly(query, tuple(params) if params else ())

    @staticmethod
    @st.cache_data(ttl=300)
    def get_price_summary_stats(_symbols: tuple[str, ...] | None = None) -> pd.DataFrame:
        """
        Get summary statistics for price data.

        Args:
            _symbols: Tuple of symbols (None = all) - underscored for cache

        Returns:
            DataFrame with per-symbol stats
        """
        api = _get_api()
        symbols = list(_symbols) if _symbols else None
        symbol_filter = ""
        params = ()

        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            symbol_filter = f"WHERE symbol IN ({placeholders})"
            params = tuple(symbols)

        query = f"""
            SELECT
                symbol,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(close) as avg_close,
                MAX(close) as max_close,
                MIN(close) as min_close,
                STDDEV(close) as std_close,
                AVG(volume) as avg_volume
            FROM prices
            {symbol_filter}
            GROUP BY symbol
            ORDER BY symbol
        """

        return api.query_readonly(query, params)

    # -------------------------------------------------------------------------
    # Feature Data Queries
    # -------------------------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=300)
    def get_available_features() -> list[str]:
        """Get list of all available feature names."""
        api = _get_api()
        query = """
            SELECT DISTINCT feature_name
            FROM features
            ORDER BY feature_name
        """
        result = api.query_readonly(query)
        return result["feature_name"].tolist() if not result.empty else []

    @staticmethod
    @st.cache_data(ttl=300)
    def get_feature_values(
        _features: tuple[str, ...],
        _symbols: tuple[str, ...] | None = None,
        _as_of_date: date | None = None,
        _limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Get feature values for visualization.

        Args:
            _features: Tuple of feature names - underscored for cache
            _symbols: Tuple of symbols to filter (None = all)
            _as_of_date: Get values as of specific date
            _limit: Max rows

        Returns:
            DataFrame in wide format (one row per date/symbol)
        """
        api = _get_api()
        features = list(_features)
        symbols = list(_symbols) if _symbols else None
        as_of_date = _as_of_date
        limit = _limit

        feature_list = ",".join([f"'{f}'" for f in features])

        conditions = [f"feature_name IN ({feature_list})"]
        params = []

        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        if as_of_date:
            conditions.append("date <= ?")
            params.append(str(as_of_date))

        where_clause = f"WHERE {' AND '.join(conditions)}"

        query = f"""
            SELECT
                symbol,
                date,
                feature_name,
                value
            FROM features
            {where_clause}
            ORDER BY symbol, date, feature_name
            LIMIT {limit}
        """

        df = api.query_readonly(query, tuple(params) if params else ())

        # Pivot to wide format
        if not df.empty:
            return df.pivot(index=["symbol", "date"], columns="feature_name", values="value").reset_index()
        return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=600)
    def get_feature_distribution(_feature_name: str) -> pd.DataFrame:
        """
        Get distribution statistics for a single feature.

        Args:
            _feature_name: Name of feature - underscored for cache

        Returns:
            DataFrame with distribution stats
        """
        api = _get_api()
        feature_name = _feature_name
        query = """
            SELECT
                feature_name,
                COUNT(*) as count,
                AVG(value) as mean,
                STDDEV(value) as std,
                MIN(value) as min,
                MAX(value) as max,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q75
            FROM features
            WHERE feature_name = ?
            GROUP BY feature_name
        """

        return api.query_readonly(query, (feature_name,))

    @staticmethod
    @st.cache_data(ttl=300)
    def get_feature_correlation(
        _features: tuple[str, ...],
        _symbols: tuple[str, ...] | None = None,
        _recent_days: int = 252,
    ) -> pd.DataFrame:
        """
        Get correlation matrix for features.

        Args:
            _features: Tuple of features to correlate - underscored for cache
            _symbols: Tuple of symbols to filter (None = all)
            _recent_days: Use last N days

        Returns:
            Correlation matrix DataFrame
        """
        api = _get_api()
        features = list(_features)
        symbols = list(_symbols) if _symbols else None
        recent_days = _recent_days

        feature_list = ",".join([f"'{f}'" for f in features])

        conditions = [
            f"feature_name IN ({feature_list})",
            f"date >= CURRENT_DATE - INTERVAL '{recent_days}' day",
        ]
        params = []

        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_clause = f"WHERE {' AND '.join(conditions)}"

        query = f"""
            SELECT symbol, date, feature_name, value
            FROM features
            {where_clause}
        """

        df = api.query_readonly(query, tuple(params) if params else ())

        if df.empty:
            return pd.DataFrame()

        # Pivot and compute correlation
        wide_df = df.pivot(index="date", columns="feature_name", values="value")
        return wide_df.corr()

    # -------------------------------------------------------------------------
    # Fundamentals Queries
    # -------------------------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=600)
    def get_fundamentals_history(
        _symbols: tuple[str, ...],
        _metrics: tuple[str, ...] | None = None,
        _start_date: date | None = None,
        _end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Get fundamentals history for symbols.

        Args:
            _symbols: Tuple of symbols - underscored for cache
            _metrics: Tuple of metrics (None = all)
            _start_date: Start date
            _end_date: End date

        Returns:
            DataFrame with fundamentals data
        """
        api = _get_api()
        symbols = list(_symbols)
        metrics = list(_metrics) if _metrics else None
        start_date = _start_date
        end_date = _end_date

        metric_filter = ""
        params = list(symbols)

        if metrics:
            placeholders = ",".join(["?" for _ in metrics])
            metric_filter = f"AND metric IN ({placeholders})"
            params.extend(metrics)

        date_filter = ""
        if start_date:
            date_filter += " AND as_of_date >= ?"
            params.append(str(start_date))
        if end_date:
            date_filter += " AND as_of_date <= ?"
            params.append(str(end_date))

        placeholders = ",".join(["?" for _ in symbols])
        query = f"""
            SELECT
                symbol,
                as_of_date,
                metric,
                value
            FROM fundamentals
            WHERE symbol IN ({placeholders})
                {metric_filter}
                {date_filter}
            ORDER BY symbol, as_of_date, metric
        """

        df = api.query_readonly(query, tuple(params))

        if not df.empty:
            return df.pivot(index=["symbol", "as_of_date"], columns="metric", values="value").reset_index()
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Universe & Metadata Queries
    # -------------------------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=600)
    def get_universe_symbols(_active_only: bool = True) -> list[str]:
        """
        Get list of universe symbols.

        Gets the most recent snapshot of universe symbols.

        Args:
            _active_only: Only include active symbols - underscored for cache

        Returns:
            List of symbol strings
        """
        api = _get_api()
        active_only = _active_only

        if active_only:
            # Get symbols that are currently in the universe (most recent date)
            query = """
                SELECT DISTINCT symbol
                FROM universe
                WHERE date = (SELECT MAX(date) FROM universe)
                  AND in_universe = TRUE
                ORDER BY symbol
            """
        else:
            # Get all distinct symbols that have ever been in the universe
            query = """
                SELECT DISTINCT symbol
                FROM universe
                ORDER BY symbol
            """

        result = api.query_readonly(query)
        return result["symbol"].tolist() if not result.empty else []

    @staticmethod
    @st.cache_data(ttl=60)
    def get_data_freshness() -> dict[str, Any]:
        """
        Get data freshness metrics.

        Returns:
            Dict with freshness info
        """
        api = _get_api()
        query = """
            SELECT
                MAX(date) as latest_date,
                COUNT(DISTINCT symbol) as symbol_count,
                COUNT(*) as total_records
            FROM prices
        """

        result = api.fetchone_readonly(query)

        if result and result[0]:
            latest = result[0]
            today = datetime.now().date()
            if isinstance(latest, str):
                latest = datetime.strptime(latest, "%Y-%m-%d").date()

            days_stale = (today - latest).days

            return {
                "latest_date": latest,
                "days_stale": days_stale,
                "is_fresh": days_stale <= 3,
                "symbol_count": result[1],
                "total_records": result[2],
            }

        return {
            "latest_date": None,
            "days_stale": None,
            "is_fresh": False,
            "symbol_count": 0,
            "total_records": 0,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def get_date_presets() -> dict[str, date]:
        """Get common date range presets."""
        today = datetime.now().date()
        return {
            "1M": today - timedelta(days=30),
            "3M": today - timedelta(days=90),
            "6M": today - timedelta(days=180),
            "YTD": date(today.year, 1, 1),
            "1Y": today - timedelta(days=365),
            "5Y": today - timedelta(days=365 * 5),
            "ALL": date(2000, 1, 1),
        }
