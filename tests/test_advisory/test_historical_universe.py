"""Tests for historical universe ingestion."""

import csv
import tempfile
from datetime import date

import pytest

from hrp.data.ingestion.historical_universe import (
    HistoricalUniverseIngestion,
    MembershipChange,
)


@pytest.fixture
def hist_universe(test_db):
    """HistoricalUniverseIngestion with test database."""
    return HistoricalUniverseIngestion()


class TestHistoricalUniverseIngestion:
    def test_persist_and_retrieve(self, hist_universe):
        changes = [
            MembershipChange(date(2020, 1, 1), "AAPL", "ADDED", "Original member"),
            MembershipChange(date(2020, 1, 1), "MSFT", "ADDED", "Original member"),
            MembershipChange(date(2020, 1, 1), "GOOGL", "ADDED", "Original member"),
            MembershipChange(date(2020, 6, 15), "TSLA", "ADDED", "Replaced XRX"),
            MembershipChange(date(2020, 6, 15), "XRX", "REMOVED", "Replaced by TSLA"),
        ]
        count = hist_universe._persist_changes(changes)
        assert count == 5

    def test_get_universe_as_of_before_changes(self, hist_universe):
        changes = [
            MembershipChange(date(2020, 1, 1), "AAPL", "ADDED"),
            MembershipChange(date(2020, 1, 1), "MSFT", "ADDED"),
            MembershipChange(date(2020, 6, 15), "TSLA", "ADDED"),
            MembershipChange(date(2020, 6, 15), "MSFT", "REMOVED"),
        ]
        hist_universe._persist_changes(changes)

        # Before TSLA was added and MSFT removed
        universe = hist_universe.get_universe_as_of(date(2020, 3, 1))
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "TSLA" not in universe

    def test_get_universe_as_of_after_changes(self, hist_universe):
        changes = [
            MembershipChange(date(2020, 1, 1), "AAPL", "ADDED"),
            MembershipChange(date(2020, 1, 1), "MSFT", "ADDED"),
            MembershipChange(date(2020, 6, 15), "TSLA", "ADDED"),
            MembershipChange(date(2020, 6, 15), "MSFT", "REMOVED"),
        ]
        hist_universe._persist_changes(changes)

        # After changes
        universe = hist_universe.get_universe_as_of(date(2020, 7, 1))
        assert "AAPL" in universe
        assert "TSLA" in universe
        assert "MSFT" not in universe

    def test_get_changes_between(self, hist_universe):
        changes = [
            MembershipChange(date(2020, 1, 1), "AAPL", "ADDED"),
            MembershipChange(date(2020, 6, 15), "TSLA", "ADDED"),
            MembershipChange(date(2021, 3, 1), "NVDA", "ADDED"),
        ]
        hist_universe._persist_changes(changes)

        df = hist_universe.get_changes_between(date(2020, 1, 1), date(2020, 12, 31))
        assert len(df) == 2
        assert set(df["symbol"].tolist()) == {"AAPL", "TSLA"}

    def test_get_total_changes(self, hist_universe):
        changes = [
            MembershipChange(date(2020, 1, 1), "AAPL", "ADDED"),
            MembershipChange(date(2020, 1, 1), "MSFT", "ADDED"),
        ]
        hist_universe._persist_changes(changes)
        assert hist_universe.get_total_changes() == 2

    def test_dedup_on_persist(self, hist_universe):
        change = MembershipChange(date(2020, 1, 1), "AAPL", "ADDED")
        hist_universe._persist_changes([change])
        hist_universe._persist_changes([change])  # Duplicate
        assert hist_universe.get_total_changes() == 1

    def test_ingest_from_csv(self, hist_universe):
        # Create a temp CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["effective_date", "symbol", "action", "reason"])
            writer.writerow(["2020-01-15", "AAPL", "ADDED", "Original member"])
            writer.writerow(["2020-06-22", "TSLA", "ADDED", "Replaced XRX"])
            writer.writerow(["2020-06-22", "XRX", "REMOVED", "Replaced by TSLA"])
            csv_path = f.name

        result = hist_universe.ingest_from_csv(csv_path)
        assert result["status"] == "success"
        assert result["changes_parsed"] == 3
        assert result["changes_ingested"] == 3

        # Verify point-in-time query
        universe = hist_universe.get_universe_as_of(date(2020, 2, 1))
        assert "AAPL" in universe
        assert "TSLA" not in universe

    def test_ingest_from_csv_missing_file(self, hist_universe):
        result = hist_universe.ingest_from_csv("/nonexistent/file.csv")
        assert result["status"] == "error"

    def test_ingest_from_csv_bad_columns(self, hist_universe):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "ticker"])
            writer.writerow(["2020-01-01", "AAPL"])
            csv_path = f.name

        result = hist_universe.ingest_from_csv(csv_path)
        assert result["status"] == "error"
        assert "Missing columns" in result["message"]

    def test_empty_universe_fallback(self, hist_universe):
        """When no history data, falls back to universe table."""
        universe = hist_universe.get_universe_as_of(date(2020, 1, 1))
        assert isinstance(universe, list)
