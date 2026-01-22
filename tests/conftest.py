"""
Pytest configuration and fixtures for HRP tests.
"""

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Generator

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def temp_db():
    """Create a temporary DuckDB database path for testing.

    Note: We create then delete the file so DuckDB can create a fresh database.
    NamedTemporaryFile creates an empty file that DuckDB can't open.
    """
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the DatabaseManager singleton to ensure fresh state for each test
    from hrp.data.db import DatabaseManager
    DatabaseManager.reset()

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also clean up WAL and other DuckDB files
    for ext in [".wal", "-journal", "-shm"]:
        wal_path = db_path + ext
        if os.path.exists(wal_path):
            os.remove(wal_path)


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    data = []
    for symbol in symbols:
        base_price = {"AAPL": 100, "MSFT": 150, "GOOGL": 1000}[symbol]
        for i, d in enumerate(dates):
            # Simple random walk
            price = base_price * (1 + 0.001 * i)
            data.append(
                {
                    "symbol": symbol,
                    "date": d.date(),
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000000,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def test_symbols():
    """Standard test symbols."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


@pytest.fixture
def test_date_range() -> tuple[date, date]:
    """Standard test date range."""
    return (date(2020, 1, 1), date(2023, 12, 31))


@pytest.fixture
def test_db(temp_db: str) -> Generator[str, None, None]:
    """Create a test database with HRP schema initialized."""
    os.environ["HRP_DB_PATH"] = temp_db

    from hrp.data.schema import create_tables

    create_tables(temp_db)

    yield temp_db

    # Reset env var
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]


@pytest.fixture
def test_api(test_db: str):
    """Get a PlatformAPI instance for testing."""
    from hrp.api.platform import PlatformAPI

    return PlatformAPI()


@pytest.fixture
def populated_db(test_db: str, sample_prices: pd.DataFrame) -> Generator[str, None, None]:
    """Test database with sample price data loaded."""
    from hrp.data.db import get_db

    db = get_db(test_db)

    # Insert sample prices
    for _, row in sample_prices.iterrows():
        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
        """,
            (
                row["symbol"],
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["adj_close"],
                row["volume"],
            ),
        )

    yield test_db


@pytest.fixture
def sample_hypothesis() -> dict:
    """Sample hypothesis data for testing."""
    return {
        "title": "Momentum predicts returns",
        "thesis": "Stocks with high 12-month returns continue outperforming",
        "prediction": "Top decile momentum > SPY by 3% annually",
        "falsification": "Sharpe < SPY or p-value > 0.05",
        "actor": "user",
        "status": "draft",
        "tags": ["momentum", "factor"],
    }


@pytest.fixture
def test_db_with_sources(test_db: str) -> Generator[str, None, None]:
    """
    Test database with data_sources entries for scheduled jobs.

    Required for tests that use ingestion_log table (FK constraint).
    """
    from hrp.data.db import get_db

    db = get_db(test_db)

    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('price_ingestion', 'scheduled_job', 'active'),
                ('feature_computation', 'scheduled_job', 'active'),
                ('yfinance', 'api', 'active'),
                ('polygon', 'api', 'active'),
                ('test', 'test', 'active')
            """
        )

    yield test_db


@pytest.fixture
def populated_db_with_sources(
    test_db_with_sources: str, sample_prices: pd.DataFrame
) -> Generator[str, None, None]:
    """Test database with sample price data and data_sources loaded."""
    from hrp.data.db import get_db

    db = get_db(test_db_with_sources)

    # Insert sample prices
    for _, row in sample_prices.iterrows():
        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
        """,
            (
                row["symbol"],
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["adj_close"],
                row["volume"],
            ),
        )

    yield test_db_with_sources
