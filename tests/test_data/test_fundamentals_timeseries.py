"""Tests for fundamentals time-series backfill."""

import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


@pytest.fixture
def fundamentals_db():
    """Create a temporary DuckDB database with schema for fundamentals testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any wal/tmp files
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def populated_fundamentals_db(fundamentals_db):
    """Populate database with sample fundamentals data."""
    from hrp.data.db import get_db

    db = get_db(fundamentals_db)

    # Insert data source first (due to FK constraints)
    db.execute(
        "INSERT INTO data_sources (source_id, source_type) VALUES (?, ?) ON CONFLICT DO NOTHING",
        ("test", "fundamentals"),
    )

    # Insert symbols first (due to FK constraints)
    for symbol in ["AAPL", "MSFT"]:
        db.execute(
            "INSERT INTO symbols (symbol, name) VALUES (?, ?)",
            (symbol, symbol),
        )

    # Insert sample fundamentals data (quarterly reports)
    # Using smaller numbers that fit in DECIMAL(18,4)
    # Q2 2023 (reported before Oct 1) and Q3 2023 (reported in Nov)
    fundamentals_data = [
        # Q2 2023 (reported Aug 2023, before our test range)
        ("AAPL", date(2023, 8, 1), date(2023, 6, 30), "revenue", 81.8),
        ("AAPL", date(2023, 8, 1), date(2023, 6, 30), "eps", 1.26),
        ("AAPL", date(2023, 8, 1), date(2023, 6, 30), "book_value", 58.0),
        # Q3 2023 (reported Nov 2023, during our test range)
        ("AAPL", date(2023, 11, 1), date(2023, 9, 30), "revenue", 89.5),
        ("AAPL", date(2023, 11, 1), date(2023, 9, 30), "eps", 1.46),
        ("AAPL", date(2023, 11, 1), date(2023, 9, 30), "book_value", 62.0),
        # Q2 2023
        ("MSFT", date(2023, 8, 1), date(2023, 6, 30), "revenue", 56.2),
        ("MSFT", date(2023, 8, 1), date(2023, 6, 30), "eps", 2.69),
        ("MSFT", date(2023, 8, 1), date(2023, 6, 30), "book_value", 220.0),
        # Q3 2023
        ("MSFT", date(2023, 11, 1), date(2023, 9, 30), "revenue", 56.5),
        ("MSFT", date(2023, 11, 1), date(2023, 9, 30), "eps", 2.76),
        ("MSFT", date(2023, 11, 1), date(2023, 9, 30), "book_value", 230.0),
    ]

    for symbol, report_date, period_end, metric, value in fundamentals_data:
        db.execute(
            """
            INSERT INTO fundamentals (symbol, report_date, period_end, metric, value, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, report_date, period_end, metric, value, "test"),
        )

    # Insert price data for the trading days
    dates = pd.date_range("2023-10-01", "2023-12-31", freq="B")
    for symbol in ["AAPL", "MSFT"]:
        base_price = 150.0 if symbol == "AAPL" else 350.0
        for i, d in enumerate(dates):
            price = base_price * (1 + 0.001 * i)
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, d.date(), price * 0.99, price * 1.02, price * 0.98, price, price, 1000000, "test"),
            )

    return fundamentals_db


class TestBackfillFundamentalsTimeSeries:
    """Tests for backfill_fundamentals_timeseries function."""

    def test_backfill_fundamentals_timeseries_basic(self, populated_fundamentals_db):
        """Test fundamentals time-series backfill."""
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries
        from hrp.data.db import get_db

        result = backfill_fundamentals_timeseries(
            symbols=["AAPL", "MSFT"],
            start=date(2023, 10, 1),
            end=date(2023, 12, 31),
            db_path=populated_fundamentals_db,
        )

        assert result["symbols_success"] == 2
        assert result["rows_inserted"] > 0

        # Verify time-series fundamentals exist
        db = get_db(populated_fundamentals_db)
        count = db.fetchone("""
            SELECT COUNT(*) FROM features
            WHERE feature_name LIKE 'ts_%'
        """)[0]
        assert count > 0

    def test_backfill_fundamentals_timeseries_point_in_time(self, populated_fundamentals_db):
        """Test that time-series fundamentals have point-in-time correctness."""
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries
        from hrp.data.db import get_db

        result = backfill_fundamentals_timeseries(
            symbols=["AAPL"],
            start=date(2023, 10, 1),
            end=date(2023, 12, 31),
            metrics=["revenue", "eps"],
            db_path=populated_fundamentals_db,
        )

        assert result["symbols_success"] == 1

        # Verify point-in-time correctness: values should change at quarter boundaries
        # Q2 data (revenue=81.8) should be used for Oct 1-31
        # Q3 data (revenue=89.5) should be used for Nov 1 onwards
        db = get_db(populated_fundamentals_db)
        values = db.fetchall("""
            SELECT date, value
            FROM features
            WHERE symbol = 'AAPL' AND feature_name = 'ts_revenue'
              AND date BETWEEN '2023-10-01' AND '2023-12-31'
            ORDER BY date
        """)

        # Should have multiple different values (not constant)
        distinct_values = set(v[1] for v in values)
        assert len(distinct_values) > 1, f"Time-series should have different values across dates, got {len(distinct_values)} distinct value(s): {distinct_values}"

    def test_backfill_fundamentals_timeseries_no_fundamentals(self, fundamentals_db):
        """Test handling symbols with no fundamentals data."""
        from hrp.data.ingestion.fundamentals_timeseries import backfill_fundamentals_timeseries

        result = backfill_fundamentals_timeseries(
            symbols=["NONEXISTENT"],
            start=date(2023, 10, 1),
            end=date(2023, 12, 31),
            db_path=fundamentals_db,
        )

        assert result["symbols_failed"] == 1
        assert "NONEXISTENT" in result["failed_symbols"]
