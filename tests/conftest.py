"""
Pytest configuration and fixtures for HRP tests.
"""

import os
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def temp_db():
    """Create a temporary DuckDB database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


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
def test_date_range():
    """Standard test date range."""
    return (date(2020, 1, 1), date(2023, 12, 31))
