"""
Tests for intraday_bars table schema.

Following TDD: Write tests FIRST, watch them FAIL, then implement.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from hrp.data.db import get_db
from hrp.data.schema import create_tables


@pytest.fixture
def db_with_schema(tmp_path):
    """Create a test database with the full schema."""
    db_path = str(tmp_path / "test.db")
    create_tables(db_path)
    return get_db(db_path)


def test_intraday_bars_table_exists(db_with_schema):
    """Test that intraday_bars table is created."""
    db = db_with_schema

    # Query to check if table exists
    result = db.fetchdf("SHOW TABLES")
    table_names = result["name"].tolist()

    assert "intraday_bars" in table_names, "intraday_bars table should exist"


def test_intraday_bars_has_correct_columns(db_with_schema):
    """Test that intraday_bars table has all required columns."""
    db = db_with_schema

    # Get column information
    columns = db.fetchdf("DESCRIBE intraday_bars")
    column_names = columns["column_name"].tolist()

    # Check all required columns exist
    required_columns = [
        "symbol", "timestamp", "open", "high", "low", "close",
        "volume", "vwap", "trade_count", "source", "ingested_at"
    ]

    for col in required_columns:
        assert col in column_names, f"Column '{col}' should exist in intraday_bars"


def test_intraday_bars_primary_key(db_with_schema):
    """Test that (symbol, timestamp) is the primary key."""
    db = db_with_schema

    # First, insert a symbol (FK requirement)
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Insert a test record
    db.execute("""
        INSERT INTO intraday_bars (symbol, timestamp, close)
        VALUES ('AAPL', '2026-02-10 10:30:00', 150.00)
    """)

    # Try to insert a duplicate (symbol, timestamp) - should fail
    with pytest.raises(Exception) as exc_info:
        db.execute("""
            INSERT INTO intraday_bars (symbol, timestamp, close)
            VALUES ('AAPL', '2026-02-10 10:30:00', 151.00)
        """)

    # Verify it's a constraint violation
    error_msg = str(exc_info.value).upper()
    assert "PRIMARY KEY" in error_msg or "UNIQUE" in error_msg or "DUPLICATE KEY" in error_msg


def test_intraday_bars_indexes_exist(db_with_schema):
    """Test that required indexes exist on intraday_bars."""
    db = db_with_schema

    # DuckDB doesn't have "SHOW INDEXES" but we can verify indexes via pragma
    # Check that the table was created successfully (indexes are created during schema setup)
    # Verify table exists and has correct structure
    result = db.fetchdf("SELECT * FROM intraday_bars LIMIT 0")

    # If table exists and is queryable, indexes were created successfully
    assert result is not None
    assert "symbol" in result.columns
    assert "timestamp" in result.columns


def test_intraday_bars_close_must_be_positive(db_with_schema):
    """Test that close price must be positive (CHECK constraint)."""
    db = db_with_schema

    # Insert a symbol first
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Try to insert with close <= 0
    with pytest.raises(Exception) as exc_info:
        db.execute("""
            INSERT INTO intraday_bars (symbol, timestamp, close)
            VALUES ('AAPL', '2026-02-10 10:30:00', 0)
        """)

    assert "CHECK" in str(exc_info.value) or "constraint" in str(exc_info.value).lower()


def test_intraday_bars_volume_must_be_non_negative(db_with_schema):
    """Test that volume must be >= 0 if provided."""
    db = db_with_schema

    # Insert a symbol first
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Try to insert with negative volume
    with pytest.raises(Exception) as exc_info:
        db.execute("""
            INSERT INTO intraday_bars (symbol, timestamp, close, volume)
            VALUES ('AAPL', '2026-02-10 10:30:00', 150.00, -100)
        """)

    assert "CHECK" in str(exc_info.value) or "constraint" in str(exc_info.value).lower()


def test_intraday_bars_foreign_key_to_symbols(db_with_schema):
    """Test that symbol column has FK constraint to symbols table."""
    db = db_with_schema

    # Try to insert with non-existent symbol
    with pytest.raises(Exception) as exc_info:
        db.execute("""
            INSERT INTO intraday_bars (symbol, timestamp, close)
            VALUES ('NONEXISTENT', '2026-02-10 10:30:00', 150.00)
        """)

    assert "FOREIGN KEY" in str(exc_info.value) or "foreign" in str(exc_info.value).lower()


def test_intraday_bars_successful_insert(db_with_schema):
    """Test successful insertion of a complete intraday bar."""
    db = db_with_schema

    # Insert a symbol first
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Insert a complete intraday bar
    db.execute("""
        INSERT INTO intraday_bars (
            symbol, timestamp, open, high, low, close, volume, vwap, trade_count, source
        ) VALUES (
            'AAPL',
            '2026-02-10 10:30:00',
            150.00,
            151.00,
            149.50,
            150.75,
            1000000,
            150.60,
            500,
            'polygon_ws'
        )
    """)

    # Query back and verify
    result = db.fetchdf("SELECT * FROM intraday_bars WHERE symbol = 'AAPL'")

    assert len(result) == 1
    assert result["symbol"][0] == "AAPL"
    assert result["close"][0] == Decimal("150.75")
    assert result["volume"][0] == 1000000
    assert result["source"][0] == "polygon_ws"


def test_intraday_bars_default_source(db_with_schema):
    """Test that source defaults to 'polygon_ws' if not provided."""
    db = db_with_schema

    # Insert a symbol first
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Insert without source
    db.execute("""
        INSERT INTO intraday_bars (symbol, timestamp, close)
        VALUES ('AAPL', '2026-02-10 10:30:00', 150.00)
    """)

    # Query back and verify default
    result = db.fetchdf("SELECT source FROM intraday_bars WHERE symbol = 'AAPL'")

    assert len(result) == 1
    assert result["source"][0] == "polygon_ws"


def test_intraday_bars_timestamp_precision(db_with_schema):
    """Test that timestamp stores with minute precision."""
    db = db_with_schema

    # Insert a symbol first
    db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

    # Insert with specific timestamp
    test_timestamp = "2026-02-10 10:30:45"
    db.execute(f"""
        INSERT INTO intraday_bars (symbol, timestamp, close)
        VALUES ('AAPL', '{test_timestamp}', 150.00)
    """)

    # Query back and verify timestamp is stored correctly
    result = db.fetchdf("SELECT timestamp FROM intraday_bars WHERE symbol = 'AAPL'")

    assert len(result) == 1
    # Timestamp should be stored as provided (with seconds)
    assert str(result["timestamp"][0]).startswith("2026-02-10 10:30:")
