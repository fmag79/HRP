"""
Tests for S&P 500 Universe Management.

Tests cover:
- Point-in-time universe queries
- Automatic exclusion rules (financials, REITs, penny stocks)
- Historical membership tracking
- Universe change detection
- Lineage logging
"""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.schema import create_tables
from hrp.data.universe import (
    EXCLUDED_SECTORS,
    MIN_PRICE_THRESHOLD,
    REIT_SUBINDUSTRIES,
    ConstituentInfo,
    UniverseManager,
)


class TestExclusionRules:
    """Test automatic exclusion logic."""

    def test_excluded_sector_financials(self, test_db):
        """Financials sector should be excluded."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason("Financials", "Banks")
        assert reason is not None
        assert "excluded_sector" in reason
        assert "financials" in reason.lower()

    def test_excluded_sector_real_estate(self, test_db):
        """Real Estate sector should be excluded."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason("Real Estate", "Residential REITs")
        assert reason is not None
        assert "excluded_sector" in reason

    def test_excluded_reit_subindustry(self, test_db):
        """REIT sub-industries should be excluded."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason(
            "Information Technology",  # Non-excluded sector
            "Equity Real Estate Investment Trusts (REITs)",  # But REIT sub-industry
        )
        assert reason is not None
        assert "reit" in reason.lower()

    def test_penny_stock_exclusion(self, test_db):
        """Stocks below $5 should be excluded."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason(
            "Information Technology",
            "Software",
            price=Decimal("4.99"),
        )
        assert reason is not None
        assert "penny_stock" in reason

    def test_penny_stock_at_threshold(self, test_db):
        """Stocks at exactly $5 should NOT be excluded."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason(
            "Information Technology",
            "Software",
            price=Decimal("5.00"),
        )
        assert reason is None

    def test_included_sector(self, test_db):
        """Technology sector should be included."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason("Information Technology", "Software")
        assert reason is None

    def test_health_care_included(self, test_db):
        """Health Care sector should be included."""
        manager = UniverseManager(test_db)
        reason = manager.get_exclusion_reason("Health Care", "Pharmaceuticals")
        assert reason is None


class TestPointInTimeQuery:
    """Test point-in-time universe queries."""

    def test_get_universe_at_date_exact_match(self, test_db):
        """Should return symbols with exact date match."""
        manager = UniverseManager(test_db)

        # Insert test data
        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-15', TRUE, 'Technology'),
                    ('MSFT', '2024-01-15', TRUE, 'Technology'),
                    ('JPM', '2024-01-15', FALSE, 'Financials')
                """
            )

        result = manager.get_universe_at_date(date(2024, 1, 15))
        assert set(result) == {"AAPL", "MSFT"}
        assert "JPM" not in result

    def test_get_universe_at_date_uses_most_recent(self, test_db):
        """Should use most recent snapshot on or before the date."""
        manager = UniverseManager(test_db)

        # Insert data for multiple dates
        with manager._db.connection() as conn:
            # Earlier snapshot
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-01', TRUE, 'Technology'),
                    ('MSFT', '2024-01-01', TRUE, 'Technology')
                """
            )
            # Later snapshot - MSFT removed
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-10', TRUE, 'Technology'),
                    ('MSFT', '2024-01-10', FALSE, 'Technology'),
                    ('GOOGL', '2024-01-10', TRUE, 'Technology')
                """
            )

        # Query for date after Jan 10 - should use Jan 10 data
        result = manager.get_universe_at_date(date(2024, 1, 15))
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "MSFT" not in result  # Removed in Jan 10 snapshot

    def test_get_universe_at_date_before_any_data(self, test_db):
        """Should return empty list if querying before any data."""
        manager = UniverseManager(test_db)

        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES ('AAPL', '2024-06-01', TRUE, 'Technology')
                """
            )

        result = manager.get_universe_at_date(date(2024, 1, 1))
        assert result == []

    def test_get_universe_at_date_sorted(self, test_db):
        """Results should be sorted alphabetically."""
        manager = UniverseManager(test_db)

        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('MSFT', '2024-01-15', TRUE, 'Technology'),
                    ('AAPL', '2024-01-15', TRUE, 'Technology'),
                    ('GOOGL', '2024-01-15', TRUE, 'Technology')
                """
            )

        result = manager.get_universe_at_date(date(2024, 1, 15))
        assert result == ["AAPL", "GOOGL", "MSFT"]


class TestHistoricalMembership:
    """Test historical membership tracking."""

    def test_get_historical_membership(self, test_db):
        """Should return full membership history for a symbol."""
        manager = UniverseManager(test_db)

        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, exclusion_reason, sector)
                VALUES
                    ('AAPL', '2024-01-01', TRUE, NULL, 'Technology'),
                    ('AAPL', '2024-02-01', FALSE, 'penny_stock:price_4.50', 'Technology'),
                    ('AAPL', '2024-03-01', TRUE, NULL, 'Technology')
                """
            )

        history = manager.get_historical_membership("AAPL")
        assert len(history) == 3
        assert history.iloc[0]["in_universe"] == True  # noqa: E712 - numpy bool comparison
        assert history.iloc[1]["in_universe"] == False  # noqa: E712
        assert "penny_stock" in str(history.iloc[1]["exclusion_reason"])
        assert history.iloc[2]["in_universe"] == True  # noqa: E712

    def test_get_universe_changes(self, test_db):
        """Should detect membership changes between dates."""
        manager = UniverseManager(test_db)

        with manager._db.connection() as conn:
            # Initial state
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-01', TRUE, 'Technology'),
                    ('MSFT', '2024-01-01', TRUE, 'Technology')
                """
            )
            # GOOGL added, MSFT removed
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-02-01', TRUE, 'Technology'),
                    ('MSFT', '2024-02-01', FALSE, 'Technology'),
                    ('GOOGL', '2024-02-01', TRUE, 'Technology')
                """
            )

        changes = manager.get_universe_changes(date(2024, 1, 1), date(2024, 2, 28))

        # Should have initial additions + GOOGL added + MSFT removed
        added = changes[changes["change_type"] == "added"]
        removed = changes[changes["change_type"] == "removed"]

        assert "GOOGL" in added["symbol"].values
        assert "MSFT" in removed["symbol"].values


class TestSectorBreakdown:
    """Test sector breakdown functionality."""

    def test_get_sector_breakdown(self, test_db):
        """Should return correct sector counts."""
        manager = UniverseManager(test_db)

        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-15', TRUE, 'Technology'),
                    ('MSFT', '2024-01-15', TRUE, 'Technology'),
                    ('GOOGL', '2024-01-15', TRUE, 'Technology'),
                    ('JNJ', '2024-01-15', TRUE, 'Health Care'),
                    ('PFE', '2024-01-15', TRUE, 'Health Care'),
                    ('JPM', '2024-01-15', FALSE, 'Financials')
                """
            )

        breakdown = manager.get_sector_breakdown(date(2024, 1, 15))
        assert breakdown.get("Technology") == 3
        assert breakdown.get("Health Care") == 2
        assert breakdown.get("Financials") is None  # Excluded


class TestLineageLogging:
    """Test that universe changes are logged to lineage."""

    def test_universe_update_logs_to_lineage(self, test_db):
        """Universe updates should create lineage entries."""
        manager = UniverseManager(test_db)

        # Mock the fetch to avoid network call
        mock_constituents = [
            ConstituentInfo(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Information Technology",
                sub_industry="Technology Hardware",
                headquarters="Cupertino, CA",
                date_added=date(1980, 1, 1),
                cik="320193",
                founded="1976",
            ),
        ]

        with patch.object(manager, "fetch_sp500_constituents", return_value=mock_constituents):
            manager.update_universe(as_of_date=date(2024, 1, 15))

        # Check lineage table
        result = manager._db.fetchall(
            "SELECT event_type, actor, details FROM lineage WHERE event_type = 'universe_update'"
        )
        assert len(result) >= 1
        assert result[0][0] == "universe_update"


class TestFetchConstituents:
    """Test S&P 500 constituent fetching."""

    def test_fetch_sp500_constituents_mock(self, test_db):
        """Test constituent fetching with mocked data."""
        manager = UniverseManager(test_db)

        # Create mock DataFrame matching Wikipedia structure
        mock_df = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "JPM"],
            "Security": ["Apple Inc.", "Microsoft Corp.", "JPMorgan Chase"],
            "GICS Sector": ["Information Technology", "Information Technology", "Financials"],
            "GICS Sub-Industry": ["Technology Hardware", "Systems Software", "Diversified Banks"],
            "Headquarters Location": ["Cupertino, CA", "Redmond, WA", "New York, NY"],
            "Date added": ["1980-01-01", "1994-06-01", "1991-05-01"],
            "CIK": ["320193", "789019", "19617"],
            "Founded": ["1976", "1975", "2000"],
        })

        with patch("pandas.read_html", return_value=[mock_df]):
            constituents = manager.fetch_sp500_constituents()

        assert len(constituents) == 3
        assert constituents[0].symbol == "AAPL"
        assert constituents[0].sector == "Information Technology"
        assert constituents[2].sector == "Financials"

    def test_symbol_normalization(self, test_db):
        """Symbols with dots should be normalized to dashes."""
        manager = UniverseManager(test_db)

        mock_df = pd.DataFrame({
            "Symbol": ["BRK.B", "BF.B"],
            "Security": ["Berkshire Hathaway", "Brown-Forman"],
            "GICS Sector": ["Financials", "Consumer Staples"],
            "GICS Sub-Industry": ["Multi-Sector Holdings", "Distillers & Vintners"],
            "Headquarters Location": ["Omaha, NE", "Louisville, KY"],
            "Date added": ["2010-01-01", "2010-01-01"],
            "CIK": ["1067983", "14693"],
            "Founded": ["1839", "1870"],
        })

        with patch("pandas.read_html", return_value=[mock_df]):
            constituents = manager.fetch_sp500_constituents()

        symbols = [c.symbol for c in constituents]
        assert "BRK-B" in symbols
        assert "BF-B" in symbols
        assert "BRK.B" not in symbols


class TestPriceExclusions:
    """Test price-based exclusion updates."""

    def test_apply_price_exclusions(self, test_db):
        """Should exclude symbols below price threshold."""
        manager = UniverseManager(test_db)

        # Setup initial universe
        with manager._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, sector)
                VALUES
                    ('AAPL', '2024-01-15', TRUE, 'Technology'),
                    ('PENNY', '2024-01-15', TRUE, 'Technology')
                """
            )

        # Apply price exclusions
        prices = {"AAPL": Decimal("150.00"), "PENNY": Decimal("3.50")}
        stats = manager.apply_price_exclusions(date(2024, 1, 15), prices)

        assert stats["excluded"] == 1
        assert "PENNY" in stats["symbols"]

        # Verify database was updated
        result = manager._db.fetchone(
            "SELECT in_universe FROM universe WHERE symbol = 'PENNY' AND date = '2024-01-15'"
        )
        assert result[0] is False


class TestUpdateUniverse:
    """Test full universe update workflow."""

    def test_update_universe_includes_and_excludes(self, test_db):
        """Update should correctly include/exclude based on rules."""
        manager = UniverseManager(test_db)

        mock_constituents = [
            ConstituentInfo(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Information Technology",
                sub_industry="Technology Hardware",
                headquarters="Cupertino, CA",
                date_added=None,
                cik=None,
                founded=None,
            ),
            ConstituentInfo(
                symbol="JPM",
                name="JPMorgan Chase",
                sector="Financials",
                sub_industry="Diversified Banks",
                headquarters="New York, NY",
                date_added=None,
                cik=None,
                founded=None,
            ),
        ]

        with patch.object(manager, "fetch_sp500_constituents", return_value=mock_constituents):
            stats = manager.update_universe(as_of_date=date(2024, 1, 15))

        assert stats["total_constituents"] == 2
        assert stats["included"] == 1  # AAPL
        assert stats["excluded"] == 1  # JPM (Financials)

        # Verify in database
        universe = manager.get_universe_at_date(date(2024, 1, 15))
        assert "AAPL" in universe
        assert "JPM" not in universe
