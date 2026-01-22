"""
S&P 500 Universe Management.

Provides functionality to:
- Fetch current S&P 500 constituents from Wikipedia
- Track historical membership with add/remove dates
- Apply automatic exclusions (financials, REITs, penny stocks)
- Query point-in-time universe membership
- Log all changes to lineage

Usage:
    from hrp.data.universe import UniverseManager

    manager = UniverseManager()
    manager.update_universe()  # Fetch and update S&P 500 constituents
    symbols = manager.get_universe_at_date(date(2024, 1, 15))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db


# Sectors to exclude from the trading universe
EXCLUDED_SECTORS = frozenset({
    "Financials",
    "Financial Services",
    "Real Estate",
})

# GICS sub-industries that are REITs (for more precise exclusion)
REIT_SUBINDUSTRIES = frozenset({
    "Equity Real Estate Investment Trusts (REITs)",
    "Mortgage Real Estate Investment Trusts (REITs)",
    "Real Estate Management & Development",
    "Diversified REITs",
    "Industrial REITs",
    "Hotel & Resort REITs",
    "Office REITs",
    "Health Care REITs",
    "Residential REITs",
    "Retail REITs",
    "Specialized REITs",
})

# Minimum price threshold (penny stock exclusion)
MIN_PRICE_THRESHOLD = Decimal("5.00")


@dataclass
class ConstituentInfo:
    """Information about an S&P 500 constituent."""

    symbol: str
    name: str
    sector: str
    sub_industry: str
    headquarters: str
    date_added: date | None
    cik: str | None
    founded: str | None


class UniverseManager:
    """
    Manages the S&P 500 trading universe.

    Handles fetching constituents, tracking membership changes,
    applying exclusion rules, and providing point-in-time queries.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the universe manager.

        Args:
            db_path: Optional path to database. Uses default if not specified.
        """
        self._db = get_db(db_path)
        logger.info("Universe manager initialized")

    def fetch_sp500_constituents(self) -> list[ConstituentInfo]:
        """
        Fetch current S&P 500 constituents from Wikipedia.

        Returns:
            List of ConstituentInfo objects for all current S&P 500 members.

        Raises:
            RuntimeError: If unable to fetch or parse Wikipedia data.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        logger.info(f"Fetching S&P 500 constituents from {url}")

        try:
            # Wikipedia table has constituents in the first table
            tables = pd.read_html(url)
            df = tables[0]

            constituents = []
            for _, row in df.iterrows():
                # Parse date added - handle various formats
                date_added = None
                date_str = str(row.get("Date added", ""))
                if date_str and date_str != "nan":
                    try:
                        # Try parsing common date formats
                        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%Y"]:
                            try:
                                date_added = datetime.strptime(date_str.split()[0], fmt).date()
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                constituent = ConstituentInfo(
                    symbol=str(row["Symbol"]).replace(".", "-"),  # BRK.B -> BRK-B
                    name=str(row.get("Security", "")),
                    sector=str(row.get("GICS Sector", "")),
                    sub_industry=str(row.get("GICS Sub-Industry", "")),
                    headquarters=str(row.get("Headquarters Location", "")),
                    date_added=date_added,
                    cik=str(row.get("CIK", "")) if pd.notna(row.get("CIK")) else None,
                    founded=str(row.get("Founded", "")) if pd.notna(row.get("Founded")) else None,
                )
                constituents.append(constituent)

            logger.info(f"Fetched {len(constituents)} S&P 500 constituents")
            return constituents

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 constituents: {e}")
            raise RuntimeError(f"Failed to fetch S&P 500 constituents: {e}") from e

    def get_exclusion_reason(
        self,
        sector: str,
        sub_industry: str,
        price: Decimal | None = None,
    ) -> str | None:
        """
        Determine if a symbol should be excluded and why.

        Args:
            sector: GICS sector
            sub_industry: GICS sub-industry
            price: Current stock price (optional)

        Returns:
            Exclusion reason string, or None if not excluded.
        """
        # Check sector exclusions
        if sector in EXCLUDED_SECTORS:
            return f"excluded_sector:{sector.lower().replace(' ', '_')}"

        # Check REIT sub-industries
        if sub_industry in REIT_SUBINDUSTRIES:
            return "excluded_sector:reit"

        # Check penny stock threshold
        if price is not None and price < MIN_PRICE_THRESHOLD:
            return f"penny_stock:price_below_{MIN_PRICE_THRESHOLD}"

        return None

    def update_universe(
        self,
        as_of_date: date | None = None,
        prices: dict[str, Decimal] | None = None,
        actor: str = "system:universe_update",
    ) -> dict[str, Any]:
        """
        Update the universe with current S&P 500 constituents.

        Fetches current constituents, applies exclusion rules, and updates
        the database. Tracks membership changes and logs to lineage.

        Args:
            as_of_date: Date to record for this universe snapshot. Defaults to today.
            prices: Optional dict of symbol -> price for penny stock filtering.
            actor: Actor to record in lineage.

        Returns:
            Dictionary with update statistics.
        """
        as_of_date = as_of_date or date.today()
        logger.info(f"Updating universe as of {as_of_date}")

        # Fetch current constituents
        constituents = self.fetch_sp500_constituents()

        # Get previous universe for change detection
        previous_symbols = set(self.get_universe_at_date(as_of_date))

        stats = {
            "date": as_of_date,
            "total_constituents": len(constituents),
            "included": 0,
            "excluded": 0,
            "added": 0,
            "removed": 0,
            "exclusion_reasons": {},
        }

        current_symbols = set()

        with self._db.connection() as conn:
            for constituent in constituents:
                symbol = constituent.symbol
                current_symbols.add(symbol)

                # Determine exclusion status
                price = prices.get(symbol) if prices else None
                exclusion_reason = self.get_exclusion_reason(
                    constituent.sector,
                    constituent.sub_industry,
                    price,
                )

                in_universe = exclusion_reason is None

                if in_universe:
                    stats["included"] += 1
                else:
                    stats["excluded"] += 1
                    reason_key = exclusion_reason.split(":")[0]
                    stats["exclusion_reasons"][reason_key] = (
                        stats["exclusion_reasons"].get(reason_key, 0) + 1
                    )

                # Upsert universe record
                conn.execute(
                    """
                    INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector, market_cap)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        in_universe = EXCLUDED.in_universe,
                        exclusion_reason = EXCLUDED.exclusion_reason,
                        sector = EXCLUDED.sector
                    """,
                    (symbol, as_of_date, in_universe, exclusion_reason, constituent.sector),
                )

            # Calculate additions and removals
            included_current = {c.symbol for c in constituents if self.get_exclusion_reason(
                c.sector, c.sub_industry) is None}

            stats["added"] = len(included_current - previous_symbols)
            stats["removed"] = len(previous_symbols - included_current)

            # Mark removed symbols as not in universe
            removed_symbols = previous_symbols - current_symbols
            for symbol in removed_symbols:
                conn.execute(
                    """
                    INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector)
                    VALUES (?, ?, FALSE, 'removed_from_index', NULL)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        in_universe = FALSE,
                        exclusion_reason = 'removed_from_index'
                    """,
                    (symbol, as_of_date),
                )

            # Log to lineage
            self._log_universe_change(conn, as_of_date, stats, actor)

        logger.info(
            f"Universe updated: {stats['included']} included, "
            f"{stats['excluded']} excluded, {stats['added']} added, "
            f"{stats['removed']} removed"
        )

        return stats

    def get_universe_at_date(self, as_of_date: date) -> list[str]:
        """
        Get the trading universe as of a specific date (point-in-time).

        Uses the most recent universe snapshot on or before the given date.
        This prevents look-ahead bias in backtests.

        Args:
            as_of_date: Date to get universe for.

        Returns:
            List of ticker symbols in the universe.
        """
        query = """
            WITH latest_per_symbol AS (
                SELECT
                    symbol,
                    in_universe,
                    date,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY date DESC
                    ) as rn
                FROM universe
                WHERE date <= ?
            )
            SELECT symbol
            FROM latest_per_symbol
            WHERE rn = 1 AND in_universe = TRUE
            ORDER BY symbol
        """

        result = self._db.fetchall(query, (as_of_date,))
        symbols = [row[0] for row in result]

        logger.debug(f"Universe contains {len(symbols)} symbols as of {as_of_date}")
        return symbols

    def get_universe_changes(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get universe membership changes between two dates.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            DataFrame with columns: date, symbol, change_type, exclusion_reason
        """
        query = """
            WITH changes AS (
                SELECT
                    symbol,
                    date,
                    in_universe,
                    exclusion_reason,
                    LAG(in_universe) OVER (
                        PARTITION BY symbol
                        ORDER BY date
                    ) as prev_in_universe
                FROM universe
                WHERE date BETWEEN ? AND ?
            )
            SELECT
                date,
                symbol,
                CASE
                    WHEN prev_in_universe IS NULL AND in_universe THEN 'added'
                    WHEN prev_in_universe AND NOT in_universe THEN 'removed'
                    WHEN NOT prev_in_universe AND in_universe THEN 'readded'
                    ELSE NULL
                END as change_type,
                exclusion_reason
            FROM changes
            WHERE (prev_in_universe IS NULL AND in_universe)
               OR (prev_in_universe != in_universe)
            ORDER BY date, symbol
        """

        result = self._db.fetchall(query, (start_date, end_date))

        return pd.DataFrame(
            result,
            columns=["date", "symbol", "change_type", "exclusion_reason"],
        )

    def get_historical_membership(self, symbol: str) -> pd.DataFrame:
        """
        Get full membership history for a symbol.

        Args:
            symbol: Ticker symbol to look up.

        Returns:
            DataFrame with columns: date, in_universe, exclusion_reason, sector
        """
        query = """
            SELECT date, in_universe, exclusion_reason, sector
            FROM universe
            WHERE symbol = ?
            ORDER BY date
        """

        result = self._db.fetchall(query, (symbol,))

        return pd.DataFrame(
            result,
            columns=["date", "in_universe", "exclusion_reason", "sector"],
        )

    def apply_price_exclusions(
        self,
        as_of_date: date,
        prices: dict[str, Decimal],
        actor: str = "system:price_exclusion",
    ) -> dict[str, Any]:
        """
        Apply penny stock exclusions based on current prices.

        This can be run separately from update_universe() to apply
        price-based exclusions to existing universe data.

        Args:
            as_of_date: Date for the exclusion check.
            prices: Dict of symbol -> price.
            actor: Actor to record in lineage.

        Returns:
            Dictionary with exclusion statistics.
        """
        stats = {"date": as_of_date, "excluded": 0, "symbols": []}

        with self._db.connection() as conn:
            for symbol, price in prices.items():
                if price < MIN_PRICE_THRESHOLD:
                    # Check if symbol exists and is currently in universe
                    check = conn.execute(
                        """
                        SELECT 1 FROM universe
                        WHERE symbol = ? AND date = ? AND in_universe = TRUE
                        """,
                        (symbol, as_of_date),
                    ).fetchone()

                    if check:
                        conn.execute(
                            """
                            UPDATE universe
                            SET in_universe = FALSE,
                                exclusion_reason = ?
                            WHERE symbol = ? AND date = ? AND in_universe = TRUE
                            """,
                            (f"penny_stock:price_{price}", symbol, as_of_date),
                        )
                        stats["excluded"] += 1
                        stats["symbols"].append(symbol)

            if stats["excluded"] > 0:
                self._log_universe_change(
                    conn,
                    as_of_date,
                    {"price_exclusions": stats["excluded"], "symbols": stats["symbols"]},
                    actor,
                )

        logger.info(f"Applied price exclusions: {stats['excluded']} symbols excluded")
        return stats

    def _log_universe_change(
        self,
        conn,
        as_of_date: date,
        details: dict[str, Any],
        actor: str,
    ) -> None:
        """Log universe change to lineage table."""
        import json

        # Convert any non-serializable types
        def json_serializer(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        serializable_details = {"date": str(as_of_date), **details}

        # Get next lineage_id (DuckDB doesn't auto-increment without sequence)
        result = conn.execute("SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage").fetchone()
        next_id = result[0]

        conn.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, timestamp, actor, details)
            VALUES (?, 'universe_update', CURRENT_TIMESTAMP, ?, ?)
            """,
            (next_id, actor, json.dumps(serializable_details, default=json_serializer)),
        )

    def get_sector_breakdown(self, as_of_date: date) -> dict[str, int]:
        """
        Get breakdown of universe by sector.

        Args:
            as_of_date: Date to get breakdown for.

        Returns:
            Dictionary mapping sector names to symbol counts.
        """
        query = """
            WITH latest_per_symbol AS (
                SELECT
                    symbol,
                    sector,
                    in_universe,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY date DESC
                    ) as rn
                FROM universe
                WHERE date <= ?
            )
            SELECT sector, COUNT(*) as count
            FROM latest_per_symbol
            WHERE rn = 1 AND in_universe = TRUE
            GROUP BY sector
            ORDER BY count DESC
        """

        result = self._db.fetchall(query, (as_of_date,))
        return {row[0]: row[1] for row in result}


def update_sp500_universe(
    db_path: str | None = None,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """
    Convenience function to update the S&P 500 universe.

    Args:
        db_path: Optional path to database.
        as_of_date: Date for the update. Defaults to today.

    Returns:
        Update statistics dictionary.
    """
    manager = UniverseManager(db_path)
    return manager.update_universe(as_of_date)
