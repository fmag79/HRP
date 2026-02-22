"""
Historical S&P 500 universe ingestion.

Tracks S&P 500 membership changes over time to enable survivorship-bias-free
backtesting. Uses Wikipedia's "Selected changes" table and/or a local CSV
for historical additions and removals.

Usage:
    from hrp.data.ingestion.historical_universe import HistoricalUniverseIngestion
    ingestion = HistoricalUniverseIngestion()
    result = ingestion.ingest_from_wikipedia()
    symbols = ingestion.get_universe_as_of(date(2020, 6, 15))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db


@dataclass
class MembershipChange:
    """A single S&P 500 membership change."""

    effective_date: date
    symbol: str
    action: str  # 'ADDED' or 'REMOVED'
    reason: str | None = None


class HistoricalUniverseIngestion:
    """
    Track S&P 500 membership changes over time.

    Populates the `universe_history` table with historical additions/removals,
    enabling point-in-time universe queries for survivorship-bias-free analysis.
    """

    def __init__(self, db_path: str | None = None):
        self._db = get_db(db_path)

    def ingest_from_wikipedia(self) -> dict[str, Any]:
        """
        Fetch historical S&P 500 changes from Wikipedia.

        The Wikipedia "List of S&P 500 companies" page has a second table
        titled "Selected changes to the list of S&P 500 components"
        with columns: Date, Added (Ticker, Security), Removed (Ticker, Security), Reason.

        Returns:
            Summary dict with counts of changes ingested.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        logger.info(f"Fetching historical S&P 500 changes from Wikipedia")

        try:
            import urllib.request
            req = urllib.request.Request(url)
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36",
            )
            with urllib.request.urlopen(req) as response:
                html = response.read()
            tables = pd.read_html(html)
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia tables: {e}")
            return {"status": "error", "message": str(e), "changes_ingested": 0}

        if len(tables) < 2:
            logger.warning("Wikipedia page does not have a second (changes) table")
            return {"status": "no_data", "changes_ingested": 0}

        changes_df = tables[1]
        changes = self._parse_wikipedia_changes(changes_df)
        count = self._persist_changes(changes)

        logger.info(f"Ingested {count} historical universe changes from Wikipedia")
        return {
            "status": "success",
            "changes_parsed": len(changes),
            "changes_ingested": count,
        }

    def ingest_from_csv(self, csv_path: str | Path) -> dict[str, Any]:
        """
        Ingest historical changes from a CSV file.

        Expected CSV format:
            effective_date,symbol,action,reason
            2020-06-22,AMCR,ADDED,Replaced XRX
            2020-06-22,XRX,REMOVED,Replaced by AMCR

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Summary dict with counts.
        """
        path = Path(csv_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {csv_path}"}

        df = pd.read_csv(path)
        required_cols = {"effective_date", "symbol", "action"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            return {"status": "error", "message": f"Missing columns: {missing}"}

        changes = []
        for _, row in df.iterrows():
            try:
                eff_date = pd.to_datetime(row["effective_date"]).date()
                action = str(row["action"]).upper()
                if action not in ("ADDED", "REMOVED"):
                    continue
                changes.append(MembershipChange(
                    effective_date=eff_date,
                    symbol=str(row["symbol"]).strip().upper(),
                    action=action,
                    reason=str(row.get("reason", "")) or None,
                ))
            except (ValueError, TypeError):
                continue

        count = self._persist_changes(changes)
        return {
            "status": "success",
            "changes_parsed": len(changes),
            "changes_ingested": count,
        }

    def get_universe_as_of(self, as_of_date: date) -> list[str]:
        """
        Reconstruct the S&P 500 membership as it existed on a specific date.

        Uses the `universe_history` table to replay additions/removals forward
        from the earliest recorded change.

        Args:
            as_of_date: The target date for point-in-time universe.

        Returns:
            List of ticker symbols in the S&P 500 on that date.
        """
        # Get all changes up to and including as_of_date
        df = self._db.fetchdf(
            "SELECT symbol, action, effective_date "
            "FROM universe_history "
            "WHERE effective_date <= ? "
            "ORDER BY effective_date, action",
            (as_of_date,),
        )

        if df.empty:
            # Fall back to current universe table
            logger.debug(
                f"No universe_history data for {as_of_date}, "
                "falling back to universe table"
            )
            fallback = self._db.fetchdf(
                "SELECT DISTINCT symbol FROM universe "
                "WHERE in_universe = TRUE "
                "AND date <= ? "
                "ORDER BY date DESC",
                (as_of_date,),
            )
            return fallback["symbol"].tolist() if not fallback.empty else []

        # Replay changes: ADDED adds to set, REMOVED removes
        members: set[str] = set()
        for _, row in df.iterrows():
            symbol = row["symbol"]
            if row["action"] == "ADDED":
                members.add(symbol)
            elif row["action"] == "REMOVED":
                members.discard(symbol)

        return sorted(members)

    def get_changes_between(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Get all membership changes between two dates.

        Returns:
            DataFrame with columns: effective_date, symbol, action, reason
        """
        return self._db.fetchdf(
            "SELECT effective_date, symbol, action, reason "
            "FROM universe_history "
            "WHERE effective_date >= ? AND effective_date <= ? "
            "ORDER BY effective_date, action",
            (start_date, end_date),
        )

    def get_total_changes(self) -> int:
        """Get total number of recorded changes."""
        result = self._db.fetchone(
            "SELECT COUNT(*) FROM universe_history"
        )
        return result[0] if result else 0

    # --- Private helpers ---

    def _parse_wikipedia_changes(
        self, df: pd.DataFrame
    ) -> list[MembershipChange]:
        """Parse Wikipedia's S&P 500 changes table into MembershipChange objects."""
        changes = []

        # Wikipedia table columns vary but typically include:
        # Date, Added (Ticker, Security), Removed (Ticker, Security), Reason
        # The table may use multi-level headers
        cols = [str(c).lower() for c in df.columns]
        df.columns = cols

        for _, row in df.iterrows():
            try:
                # Parse date - first column is usually the date
                date_val = row.iloc[0]
                eff_date = self._parse_date(date_val)
                if eff_date is None:
                    continue

                # Parse added ticker
                added_ticker = self._extract_ticker(row, "added")
                if added_ticker:
                    reason = self._extract_reason(row)
                    changes.append(MembershipChange(
                        effective_date=eff_date,
                        symbol=added_ticker,
                        action="ADDED",
                        reason=reason,
                    ))

                # Parse removed ticker
                removed_ticker = self._extract_ticker(row, "removed")
                if removed_ticker:
                    reason = self._extract_reason(row)
                    changes.append(MembershipChange(
                        effective_date=eff_date,
                        symbol=removed_ticker,
                        action="REMOVED",
                        reason=reason,
                    ))

            except Exception:
                continue

        logger.info(f"Parsed {len(changes)} membership changes from Wikipedia")
        return changes

    def _parse_date(self, val: Any) -> date | None:
        """Parse a date from various formats."""
        if pd.isna(val):
            return None
        try:
            return pd.to_datetime(str(val)).date()
        except (ValueError, TypeError):
            return None

    def _extract_ticker(self, row: pd.Series, direction: str) -> str | None:
        """Extract ticker symbol from a row for 'added' or 'removed' direction."""
        # Look for column containing the direction keyword
        for i, col in enumerate(row.index):
            col_str = str(col).lower()
            if direction in col_str and "ticker" in col_str:
                val = row.iloc[i]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip().upper()

        # Fallback: look for the direction in any column name
        for i, col in enumerate(row.index):
            if direction in str(col).lower():
                val = row.iloc[i]
                if pd.notna(val):
                    ticker = str(val).strip().upper()
                    # Basic validation: tickers are short uppercase strings
                    if 1 <= len(ticker) <= 5 and ticker.isalpha():
                        return ticker

        return None

    def _extract_reason(self, row: pd.Series) -> str | None:
        """Extract the reason from a row."""
        for col in row.index:
            if "reason" in str(col).lower():
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
        return None

    def _persist_changes(self, changes: list[MembershipChange]) -> int:
        """Persist changes to the universe_history table."""
        count = 0
        for change in changes:
            try:
                self._db.execute(
                    "INSERT INTO universe_history "
                    "(effective_date, symbol, action, reason) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (effective_date, symbol, action) DO NOTHING",
                    (change.effective_date, change.symbol, change.action, change.reason),
                )
                count += 1
            except Exception as e:
                logger.debug(f"Skipping duplicate change: {change.symbol} {change.action} on {change.effective_date}: {e}")

        return count
