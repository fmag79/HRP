"""
Comprehensive tests for the HRP Platform API.

Tests cover all major functionality:
- Input validation (symbols, dates, strings, numeric parameters)
- Initialization
- Health check
- Data operations (universe, prices, features)
- Hypothesis management (CRUD operations)
- Lineage/audit trail
- Deployment permissions (user vs agent)
- Experiment linking
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.api.platform import (
    NotFoundError,
    PermissionError,
    PlatformAPI,
    PlatformAPIError,
)
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db():
    """Create a temporary DuckDB database with schema for testing."""
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
def test_api(test_db):
    """Create a PlatformAPI instance with a test database."""
    return PlatformAPI(db_path=test_db)


@pytest.fixture
def populated_db(test_api):
    """
    Populate the test database with sample data.

    Returns the API instance for convenience.
    """
    # Insert sample universe data
    test_api._db.execute(
        """
        INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
        VALUES
            ('AAPL', '2023-01-01', TRUE, 'Technology', 2500000000000),
            ('MSFT', '2023-01-01', TRUE, 'Technology', 2400000000000),
            ('GOOGL', '2023-01-01', TRUE, 'Technology', 1500000000000),
            ('JPM', '2023-01-01', FALSE, 'Financials', 400000000000),
            ('AAPL', '2023-01-05', TRUE, 'Technology', 2500000000000),
            ('MSFT', '2023-01-05', TRUE, 'Technology', 2400000000000),
            ('GOOGL', '2023-01-05', TRUE, 'Technology', 1500000000000),
            ('AAPL', '2023-06-01', TRUE, 'Technology', 2800000000000),
            ('MSFT', '2023-06-01', TRUE, 'Technology', 2600000000000)
        """
    )

    # Insert sample price data
    prices_data = []
    base_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")

    for symbol, base in base_prices.items():
        for i, d in enumerate(dates):
            price = base * (1 + 0.01 * i)
            prices_data.append(
                (
                    symbol,
                    d.date(),
                    price * 0.99,
                    price * 1.02,
                    price * 0.98,
                    price,
                    price,
                    1000000 + i * 10000,
                )
            )

    test_api._db.execute(
        """
        INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        prices_data[0],
    )
    for row in prices_data[1:]:
        test_api._db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )

    # Insert sample features
    test_api._db.execute(
        """
        INSERT INTO features (symbol, date, feature_name, value, version)
        VALUES
            ('AAPL', '2023-01-05', 'momentum_20d', 0.05, 'v1'),
            ('AAPL', '2023-01-05', 'volatility_60d', 0.25, 'v1'),
            ('MSFT', '2023-01-05', 'momentum_20d', 0.03, 'v1'),
            ('MSFT', '2023-01-05', 'volatility_60d', 0.22, 'v1'),
            ('GOOGL', '2023-01-05', 'momentum_20d', -0.02, 'v1'),
            ('GOOGL', '2023-01-05', 'volatility_60d', 0.30, 'v1')
        """
    )

    return test_api


@pytest.fixture
def hypothesis_api(test_api):
    """
    Create a test API with some pre-existing hypotheses.

    Returns the API instance and created hypothesis IDs.
    """
    hyp_ids = []

    # Create several hypotheses with different statuses
    hyp_ids.append(
        test_api.create_hypothesis(
            title="Momentum Factor Test",
            thesis="Stocks with high momentum continue to outperform",
            prediction="Top decile momentum > SPY by 3% annually",
            falsification="Sharpe < SPY or p-value > 0.05",
            actor="user",
        )
    )

    hyp_ids.append(
        test_api.create_hypothesis(
            title="Value Factor Test",
            thesis="Low P/E stocks outperform over time",
            prediction="Bottom quintile P/E > market by 2%",
            falsification="Underperforms market over 5 years",
            actor="agent:discovery",
        )
    )

    # Update one to testing status
    test_api.update_hypothesis(hyp_ids[0], status="testing", actor="user")

    # Create a validated hypothesis
    hyp_ids.append(
        test_api.create_hypothesis(
            title="Volatility Factor Test",
            thesis="Low volatility stocks have better risk-adjusted returns",
            prediction="Low vol quintile Sharpe > high vol quintile",
            falsification="No significant difference in Sharpe",
            actor="user",
        )
    )
    test_api.update_hypothesis(hyp_ids[2], status="validated", actor="user")

    return test_api, hyp_ids


# =============================================================================
# Test Classes
# =============================================================================


class TestPlatformAPIValidation:
    """Tests for input validation in Platform API methods."""

    def test_get_prices_with_invalid_symbols(self, populated_db):
        """get_prices should reject invalid symbols."""
        with pytest.raises(ValueError, match="Invalid symbols not found in universe: INVALID"):
            populated_db.get_prices(
                symbols=['INVALID'],
                start=date(2023, 1, 1),
                end=date(2023, 1, 10)
            )

    def test_get_prices_with_future_dates(self, populated_db):
        """get_prices should reject future dates."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="date cannot be in the future"):
            populated_db.get_prices(
                symbols=['AAPL'],
                start=future_date,
                end=future_date
            )

    def test_get_prices_with_invalid_date_range(self, populated_db):
        """get_prices should reject end_date before start_date."""
        with pytest.raises(ValueError, match="start date must be <= end date"):
            populated_db.get_prices(
                symbols=['AAPL'],
                start=date(2023, 1, 10),
                end=date(2023, 1, 1)
            )

    def test_get_features_with_invalid_symbols(self, populated_db):
        """get_features should reject invalid symbols."""
        with pytest.raises(ValueError, match="Invalid symbols not in universe"):
            populated_db.get_features(
                symbols=['INVALID'],
                features=['momentum_20d'],
                as_of_date=date(2023, 1, 5)
            )

    def test_get_features_with_empty_symbols(self, populated_db):
        """get_features should reject empty symbols list."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            populated_db.get_features(
                symbols=[],
                features=['momentum_20d'],
                as_of_date=date(2023, 1, 5)
            )

    def test_get_features_with_empty_features(self, populated_db):
        """get_features should reject empty features list."""
        with pytest.raises(ValueError, match="features list cannot be empty"):
            populated_db.get_features(
                symbols=['AAPL'],
                features=[],
                as_of_date=date(2023, 1, 5)
            )

    def test_get_features_with_future_date(self, populated_db):
        """get_features should reject future dates."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="as_of_date cannot be in the future"):
            populated_db.get_features(
                symbols=['AAPL'],
                features=['momentum_20d'],
                as_of_date=future_date
            )

    def test_get_universe_with_future_date(self, populated_db):
        """get_universe should reject future dates."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="as_of_date cannot be in the future"):
            populated_db.get_universe(as_of_date=future_date)

    def test_create_hypothesis_with_empty_title(self, test_api):
        """create_hypothesis should reject empty title."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            test_api.create_hypothesis(
                title="",
                thesis="Some thesis",
                prediction="Some prediction",
                falsification="Some falsification",
                actor="user"
            )

    def test_create_hypothesis_with_empty_thesis(self, test_api):
        """create_hypothesis should reject empty thesis."""
        with pytest.raises(ValueError, match="thesis cannot be empty"):
            test_api.create_hypothesis(
                title="Test Title",
                thesis="",
                prediction="Some prediction",
                falsification="Some falsification",
                actor="user"
            )

    def test_create_hypothesis_with_empty_prediction(self, test_api):
        """create_hypothesis should reject empty prediction."""
        with pytest.raises(ValueError, match="prediction cannot be empty"):
            test_api.create_hypothesis(
                title="Test Title",
                thesis="Some thesis",
                prediction="",
                falsification="Some falsification",
                actor="user"
            )

    def test_create_hypothesis_with_empty_falsification(self, test_api):
        """create_hypothesis should reject empty falsification."""
        with pytest.raises(ValueError, match="falsification cannot be empty"):
            test_api.create_hypothesis(
                title="Test Title",
                thesis="Some thesis",
                prediction="Some prediction",
                falsification="",
                actor="user"
            )

    def test_create_hypothesis_with_empty_actor(self, test_api):
        """create_hypothesis should reject empty actor."""
        with pytest.raises(ValueError, match="actor cannot be empty"):
            test_api.create_hypothesis(
                title="Test Title",
                thesis="Some thesis",
                prediction="Some prediction",
                falsification="Some falsification",
                actor=""
            )

    def test_create_hypothesis_with_whitespace_only_strings(self, test_api):
        """create_hypothesis should reject whitespace-only strings."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            test_api.create_hypothesis(
                title="   ",
                thesis="Some thesis",
                prediction="Some prediction",
                falsification="Some falsification",
                actor="user"
            )

    def test_update_hypothesis_with_empty_hypothesis_id(self, hypothesis_api):
        """update_hypothesis should reject empty hypothesis_id."""
        api, hyp_ids = hypothesis_api
        with pytest.raises(ValueError, match="hypothesis_id cannot be empty"):
            api.update_hypothesis(
                hypothesis_id="",
                status="testing",
                actor="user"
            )

    def test_update_hypothesis_with_empty_status(self, hypothesis_api):
        """update_hypothesis should reject empty status."""
        api, hyp_ids = hypothesis_api
        with pytest.raises(ValueError, match="status cannot be empty"):
            api.update_hypothesis(
                hypothesis_id=hyp_ids[0],
                status="",
                actor="user"
            )

    def test_update_hypothesis_with_empty_actor(self, hypothesis_api):
        """update_hypothesis should reject empty actor."""
        api, hyp_ids = hypothesis_api
        with pytest.raises(ValueError, match="actor cannot be empty"):
            api.update_hypothesis(
                hypothesis_id=hyp_ids[0],
                status="testing",
                actor=""
            )

    def test_list_hypotheses_with_invalid_limit(self, test_api):
        """list_hypotheses should reject non-positive limit."""
        with pytest.raises(ValueError, match="limit must be positive"):
            test_api.list_hypotheses(limit=0)

        with pytest.raises(ValueError, match="limit must be positive"):
            test_api.list_hypotheses(limit=-5)

    def test_get_hypothesis_with_empty_id(self, test_api):
        """get_hypothesis should reject empty hypothesis_id."""
        with pytest.raises(ValueError, match="hypothesis_id cannot be empty"):
            test_api.get_hypothesis("")

    def test_mixed_valid_invalid_symbols(self, populated_db):
        """Mix of valid and invalid symbols should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid symbols not found in universe"):
            populated_db.get_prices(
                symbols=['AAPL', 'INVALID1', 'INVALID2'],
                start=date(2023, 1, 1),
                end=date(2023, 1, 10)
            )

    def test_excluded_symbol_validation(self, populated_db):
        """Symbols explicitly excluded from universe should be rejected."""
        # JPM has in_universe = FALSE in the test data
        with pytest.raises(ValueError, match="Invalid symbols not found in universe: JPM"):
            populated_db.get_prices(
                symbols=['JPM'],
                start=date(2023, 1, 1),
                end=date(2023, 1, 10)
            )

    def test_symbol_not_in_universe_at_specific_date(self, populated_db):
        """Symbols not in universe at specific date should be rejected."""
        # Insert TSLA only for 2023-06-01
        populated_db._db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES ('TSLA', '2023-06-01', TRUE, 'Consumer Discretionary', 800000000000)
            """
        )

        # TSLA should not be valid for 2023-01-05
        with pytest.raises(ValueError, match="Invalid symbols not in universe as of 2023-01-05: TSLA"):
            populated_db.get_features(
                symbols=['TSLA'],
                features=['momentum_20d'],
                as_of_date=date(2023, 1, 5)
            )


class TestPlatformAPIInit:
    """Tests for PlatformAPI initialization."""

    def test_init_with_db_path(self, test_db):
        """Test initialization with explicit database path."""
        api = PlatformAPI(db_path=test_db)
        assert api is not None
        assert api._db is not None

    def test_init_creates_connection(self, test_db):
        """Test that initialization creates a database connection."""
        api = PlatformAPI(db_path=test_db)
        # Verify we can execute queries
        result = api._db.fetchone("SELECT 1")
        assert result == (1,)

    def test_multiple_apis_same_db(self, test_db):
        """Test multiple API instances can share the same database."""
        api1 = PlatformAPI(db_path=test_db)
        api2 = PlatformAPI(db_path=test_db)

        # Create hypothesis with one API
        hyp_id = api1.create_hypothesis(
            title="Test",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
            actor="user",
        )

        # Should be visible from other API
        hyp = api2.get_hypothesis(hyp_id)
        assert hyp is not None
        assert hyp["title"] == "Test"


class TestPlatformAPIHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_api_ok(self, test_api):
        """Test that API reports healthy status."""
        health = test_api.health_check()
        assert health["api"] == "ok"

    def test_health_check_database_ok(self, test_api):
        """Test that database connectivity is checked."""
        health = test_api.health_check()
        assert health["database"] == "ok"

    def test_health_check_includes_tables(self, test_api):
        """Test that table status is included in health check."""
        health = test_api.health_check()
        assert "tables" in health
        assert isinstance(health["tables"], dict)

    def test_health_check_table_counts(self, test_api):
        """Test that table counts are included."""
        health = test_api.health_check()
        # Check that expected tables are present
        assert "prices" in health["tables"]
        assert "hypotheses" in health["tables"]
        assert "lineage" in health["tables"]
        # Check structure
        for table_info in health["tables"].values():
            assert "status" in table_info
            assert "count" in table_info


class TestPlatformAPIUniverse:
    """Tests for universe data operations."""

    def test_get_universe_empty(self, test_api):
        """Test getting universe when no data exists."""
        result = test_api.get_universe(date(2023, 1, 1))
        assert result == []

    def test_get_universe_with_data(self, populated_db):
        """Test getting universe with populated data."""
        result = populated_db.get_universe(date(2023, 1, 1))
        assert len(result) == 3  # AAPL, MSFT, GOOGL (JPM is excluded)
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result
        assert "JPM" not in result  # Excluded from universe

    def test_get_universe_different_date(self, populated_db):
        """Test universe can vary by date."""
        result = populated_db.get_universe(date(2023, 6, 1))
        assert len(result) == 2  # Only AAPL and MSFT for this date

    def test_get_universe_sorted(self, populated_db):
        """Test that universe symbols are sorted alphabetically."""
        result = populated_db.get_universe(date(2023, 1, 1))
        assert result == sorted(result)


class TestPlatformAPIPrices:
    """Tests for price data operations."""

    def test_get_prices_empty_symbols_raises(self, test_api):
        """Test that empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            test_api.get_prices([], date(2023, 1, 1), date(2023, 1, 10))

    def test_get_prices_no_data(self, test_api):
        """Test getting prices when no data exists."""
        # Add symbol to universe so validation passes
        test_api._db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES ('AAPL', '2023-01-01', TRUE, 'Technology', 2500000000000)
            """
        )
        result = test_api.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_prices_with_data(self, populated_db):
        """Test getting prices with populated data."""
        result = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        assert not result.empty
        assert "symbol" in result.columns
        assert "date" in result.columns
        assert "close" in result.columns
        assert all(result["symbol"] == "AAPL")

    def test_get_prices_multiple_symbols(self, populated_db):
        """Test getting prices for multiple symbols."""
        result = populated_db.get_prices(
            ["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 1, 10)
        )
        assert not result.empty
        symbols = result["symbol"].unique()
        assert len(symbols) == 2
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_prices_date_range(self, populated_db):
        """Test that date range filtering works."""
        # Get subset of dates
        result = populated_db.get_prices(["AAPL"], date(2023, 1, 3), date(2023, 1, 5))
        dates = result["date"].unique()
        for d in dates:
            # Convert to date if it's a Timestamp
            d_date = d.date() if hasattr(d, "date") else d
            assert d_date >= date(2023, 1, 3)
            assert d_date <= date(2023, 1, 5)

    def test_get_prices_returns_all_columns(self, populated_db):
        """Test that all expected columns are returned."""
        result = populated_db.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 1, 10))
        expected_columns = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        for col in expected_columns:
            assert col in result.columns


class TestPlatformAPIFeatures:
    """Tests for feature data operations."""

    def test_get_features_empty_symbols_raises(self, test_api):
        """Test that empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            test_api.get_features([], ["momentum_20d"], date(2023, 1, 5))

    def test_get_features_empty_features_raises(self, test_api):
        """Test that empty features list raises ValueError."""
        with pytest.raises(ValueError, match="features list cannot be empty"):
            test_api.get_features(["AAPL"], [], date(2023, 1, 5))

    def test_get_features_no_data(self, test_api):
        """Test getting features when no data exists."""
        # Add symbol to universe so validation passes
        test_api._db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES ('AAPL', '2023-01-05', TRUE, 'Technology', 2500000000000)
            """
        )
        result = test_api.get_features(["AAPL"], ["momentum_20d"], date(2023, 1, 5))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_features_with_data(self, populated_db):
        """Test getting features with populated data."""
        result = populated_db.get_features(
            ["AAPL"], ["momentum_20d", "volatility_60d"], date(2023, 1, 5)
        )
        assert not result.empty
        assert "symbol" in result.columns
        assert "momentum_20d" in result.columns
        assert "volatility_60d" in result.columns

    def test_get_features_multiple_symbols(self, populated_db):
        """Test getting features for multiple symbols."""
        result = populated_db.get_features(
            ["AAPL", "MSFT"], ["momentum_20d"], date(2023, 1, 5)
        )
        assert len(result) == 2
        symbols = result["symbol"].tolist()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_features_pivoted_format(self, populated_db):
        """Test that features are returned in pivoted format."""
        result = populated_db.get_features(
            ["AAPL", "MSFT", "GOOGL"],
            ["momentum_20d", "volatility_60d"],
            date(2023, 1, 5),
        )
        # Should have symbols as rows and features as columns
        assert len(result) == 3
        assert "momentum_20d" in result.columns
        assert "volatility_60d" in result.columns


class TestPlatformAPIHypothesis:
    """Tests for hypothesis management operations."""

    def test_create_hypothesis_returns_id(self, test_api):
        """Test that creating a hypothesis returns a valid ID."""
        hyp_id = test_api.create_hypothesis(
            title="Test Hypothesis",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
            actor="user",
        )
        assert hyp_id is not None
        assert hyp_id.startswith("HYP-")

    def test_create_hypothesis_id_format(self, test_api):
        """Test hypothesis ID format is HYP-YYYY-NNN."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        current_year = datetime.now().year
        assert hyp_id == f"HYP-{current_year}-001"

    def test_create_hypothesis_increments_counter(self, test_api):
        """Test that hypothesis IDs increment correctly."""
        hyp1 = test_api.create_hypothesis(
            title="Test 1",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        hyp2 = test_api.create_hypothesis(
            title="Test 2",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        current_year = datetime.now().year
        assert hyp1 == f"HYP-{current_year}-001"
        assert hyp2 == f"HYP-{current_year}-002"

    def test_create_hypothesis_logs_event(self, test_api):
        """Test that creating a hypothesis logs a lineage event."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        assert len(lineage) > 0
        assert lineage[0]["event_type"] == "hypothesis_created"
        assert lineage[0]["actor"] == "user"

    def test_create_hypothesis_by_agent(self, test_api):
        """Test that agents can create hypotheses."""
        hyp_id = test_api.create_hypothesis(
            title="Agent Hypothesis",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="agent:discovery",
        )
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["created_by"] == "agent:discovery"

    def test_get_hypothesis_exists(self, test_api):
        """Test retrieving an existing hypothesis."""
        hyp_id = test_api.create_hypothesis(
            title="Test Hypothesis",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
            actor="user",
        )
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp is not None
        assert hyp["hypothesis_id"] == hyp_id
        assert hyp["title"] == "Test Hypothesis"
        assert hyp["thesis"] == "Test thesis"
        assert hyp["prediction"] == "Test prediction"
        assert hyp["falsification"] == "Test falsification"
        assert hyp["status"] == "draft"

    def test_get_hypothesis_not_found(self, test_api):
        """Test retrieving a non-existent hypothesis returns None."""
        result = test_api.get_hypothesis("HYP-9999-999")
        assert result is None

    def test_list_hypotheses_empty(self, test_api):
        """Test listing hypotheses when none exist."""
        result = test_api.list_hypotheses()
        assert result == []

    def test_list_hypotheses_all(self, hypothesis_api):
        """Test listing all hypotheses."""
        api, hyp_ids = hypothesis_api
        result = api.list_hypotheses()
        assert len(result) == 3

    def test_list_hypotheses_by_status(self, hypothesis_api):
        """Test filtering hypotheses by status."""
        api, hyp_ids = hypothesis_api

        draft = api.list_hypotheses(status="draft")
        assert len(draft) == 1
        assert draft[0]["title"] == "Value Factor Test"

        testing = api.list_hypotheses(status="testing")
        assert len(testing) == 1
        assert testing[0]["title"] == "Momentum Factor Test"

        validated = api.list_hypotheses(status="validated")
        assert len(validated) == 1
        assert validated[0]["title"] == "Volatility Factor Test"

    def test_list_hypotheses_limit(self, test_api):
        """Test that limit parameter works."""
        # Create several hypotheses
        for i in range(5):
            test_api.create_hypothesis(
                title=f"Hypothesis {i}",
                thesis="Test",
                prediction="Test",
                falsification="Test",
                actor="user",
            )

        result = test_api.list_hypotheses(limit=3)
        assert len(result) == 3

    def test_list_hypotheses_ordered_by_created_at(self, test_api):
        """Test that hypotheses are ordered by creation time descending."""
        for i in range(3):
            test_api.create_hypothesis(
                title=f"Hypothesis {i}",
                thesis="Test",
                prediction="Test",
                falsification="Test",
                actor="user",
            )

        result = test_api.list_hypotheses()
        # Most recent should be first
        assert result[0]["title"] == "Hypothesis 2"
        assert result[2]["title"] == "Hypothesis 0"

    def test_update_hypothesis_status(self, test_api):
        """Test updating hypothesis status."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api.update_hypothesis(hyp_id, status="testing", actor="user")

        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "testing"

    def test_update_hypothesis_outcome(self, test_api):
        """Test updating hypothesis with outcome."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api.update_hypothesis(
            hyp_id,
            status="validated",
            outcome="Hypothesis confirmed with Sharpe of 1.5",
            actor="user",
        )

        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "validated"
        assert hyp["outcome"] == "Hypothesis confirmed with Sharpe of 1.5"

    def test_update_hypothesis_not_found(self, test_api):
        """Test updating non-existent hypothesis raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            test_api.update_hypothesis("HYP-9999-999", status="testing", actor="user")

    def test_update_hypothesis_logs_event(self, test_api):
        """Test that updating hypothesis logs a lineage event."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api.update_hypothesis(hyp_id, status="testing", actor="user")

        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        update_events = [e for e in lineage if e["event_type"] == "hypothesis_updated"]
        assert len(update_events) == 1
        assert update_events[0]["details"]["old_status"] == "draft"
        assert update_events[0]["details"]["new_status"] == "testing"


class TestPlatformAPIPermissions:
    """Tests for deployment permissions (user vs agent)."""

    def test_approve_deployment_user_allowed(self, hypothesis_api):
        """Test that users can approve deployments."""
        api, hyp_ids = hypothesis_api
        # hyp_ids[2] is validated
        result = api.approve_deployment(hyp_ids[2], actor="user")
        assert result is True

        hyp = api.get_hypothesis(hyp_ids[2])
        assert hyp["status"] == "deployed"

    def test_approve_deployment_agent_denied(self, hypothesis_api):
        """Test that agents cannot approve deployments."""
        api, hyp_ids = hypothesis_api

        with pytest.raises(PermissionError, match="Agents cannot approve deployments"):
            api.approve_deployment(hyp_ids[2], actor="agent:test")

    def test_approve_deployment_agent_variants_denied(self, hypothesis_api):
        """Test various agent actor patterns are denied."""
        api, hyp_ids = hypothesis_api

        agent_actors = [
            "agent:discovery",
            "agent:backtest",
            "agent:analysis",
            "agent:",
        ]

        for actor in agent_actors:
            with pytest.raises(PermissionError):
                api.approve_deployment(hyp_ids[2], actor=actor)

    def test_approve_deployment_not_found(self, test_api):
        """Test deploying non-existent hypothesis raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            test_api.approve_deployment("HYP-9999-999", actor="user")

    def test_approve_deployment_invalid_status(self, test_api):
        """Test deploying hypothesis with invalid status returns False."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        # Status is 'draft', not 'validated' or 'testing'
        result = test_api.approve_deployment(hyp_id, actor="user")
        assert result is False

    def test_approve_deployment_from_testing(self, test_api):
        """Test that hypothesis in 'testing' status can be deployed."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api.update_hypothesis(hyp_id, status="testing", actor="user")

        result = test_api.approve_deployment(hyp_id, actor="user")
        assert result is True

    def test_approve_deployment_logs_event(self, hypothesis_api):
        """Test that deployment approval logs a lineage event."""
        api, hyp_ids = hypothesis_api
        api.approve_deployment(hyp_ids[2], actor="user")

        lineage = api.get_lineage(hypothesis_id=hyp_ids[2])
        deploy_events = [e for e in lineage if e["event_type"] == "deployment_approved"]
        assert len(deploy_events) == 1
        assert deploy_events[0]["actor"] == "user"

    def test_get_deployed_strategies(self, hypothesis_api):
        """Test listing deployed strategies."""
        api, hyp_ids = hypothesis_api
        api.approve_deployment(hyp_ids[2], actor="user")

        deployed = api.get_deployed_strategies()
        assert len(deployed) == 1
        assert deployed[0]["hypothesis_id"] == hyp_ids[2]


class TestPlatformAPILineage:
    """Tests for lineage/audit trail operations."""

    def test_log_event_returns_id(self, test_api):
        """Test that logging an event returns a lineage ID."""
        lineage_id = test_api.log_event(
            event_type="test_event", actor="user", details={"test": "data"}
        )
        assert lineage_id is not None
        assert isinstance(lineage_id, int)
        assert lineage_id > 0

    def test_log_event_increments_id(self, test_api):
        """Test that lineage IDs increment."""
        id1 = test_api.log_event(event_type="event1", actor="user")
        id2 = test_api.log_event(event_type="event2", actor="user")
        assert id2 > id1

    def test_log_event_with_details(self, test_api):
        """Test logging event with details dictionary."""
        details = {"key1": "value1", "key2": 42, "nested": {"a": 1}}
        test_api.log_event(event_type="test", actor="user", details=details)

        lineage = test_api.get_lineage()
        assert len(lineage) == 1
        assert lineage[0]["details"] == details

    def test_log_event_with_hypothesis_id(self, test_api):
        """Test logging event linked to hypothesis."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api.log_event(
            event_type="custom_event", actor="user", hypothesis_id=hyp_id
        )

        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        custom_events = [e for e in lineage if e["event_type"] == "custom_event"]
        assert len(custom_events) == 1

    def test_log_event_with_experiment_id(self, test_api):
        """Test logging event linked to experiment."""
        test_api.log_event(
            event_type="experiment_complete",
            actor="user",
            experiment_id="exp-123",
        )

        lineage = test_api.get_lineage(experiment_id="exp-123")
        assert len(lineage) == 1
        assert lineage[0]["experiment_id"] == "exp-123"

    def test_log_event_with_parent_lineage_id(self, test_api):
        """Test logging event with parent lineage ID."""
        parent_id = test_api.log_event(event_type="parent", actor="user")
        child_id = test_api.log_event(
            event_type="child", actor="user", parent_lineage_id=parent_id
        )

        lineage = test_api.get_lineage()
        child_event = [e for e in lineage if e["lineage_id"] == child_id][0]
        assert child_event["parent_lineage_id"] == parent_id

    def test_get_lineage_empty(self, test_api):
        """Test getting lineage when empty."""
        result = test_api.get_lineage()
        assert result == []

    def test_get_lineage_all(self, test_api):
        """Test getting all lineage events."""
        test_api.log_event(event_type="event1", actor="user")
        test_api.log_event(event_type="event2", actor="agent:test")
        test_api.log_event(event_type="event3", actor="user")

        result = test_api.get_lineage()
        assert len(result) == 3

    def test_get_lineage_filter_by_hypothesis(self, test_api):
        """Test filtering lineage by hypothesis ID."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        # Create an unrelated event
        test_api.log_event(event_type="unrelated", actor="user")

        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        for event in lineage:
            assert event["hypothesis_id"] == hyp_id

    def test_get_lineage_filter_by_experiment(self, test_api):
        """Test filtering lineage by experiment ID."""
        test_api.log_event(
            event_type="exp_event", actor="user", experiment_id="exp-123"
        )
        test_api.log_event(
            event_type="other_event", actor="user", experiment_id="exp-456"
        )

        lineage = test_api.get_lineage(experiment_id="exp-123")
        assert len(lineage) == 1
        assert lineage[0]["experiment_id"] == "exp-123"

    def test_get_lineage_limit(self, test_api):
        """Test lineage limit parameter."""
        for i in range(10):
            test_api.log_event(event_type=f"event_{i}", actor="user")

        result = test_api.get_lineage(limit=5)
        assert len(result) == 5

    def test_get_lineage_ordered_by_timestamp_desc(self, test_api):
        """Test that lineage is ordered by timestamp descending."""
        test_api.log_event(event_type="first", actor="user")
        test_api.log_event(event_type="second", actor="user")
        test_api.log_event(event_type="third", actor="user")

        result = test_api.get_lineage()
        assert result[0]["event_type"] == "third"
        assert result[2]["event_type"] == "first"


class TestPlatformAPIExperimentLinking:
    """Tests for experiment-hypothesis linking."""

    def test_link_experiment_to_hypothesis(self, test_api):
        """Test linking an experiment to a hypothesis."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-123")

        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert "exp-123" in experiments

    def test_link_multiple_experiments(self, test_api):
        """Test linking multiple experiments to one hypothesis."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-1")
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-2")
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-3")

        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert len(experiments) == 3

    def test_link_experiment_duplicate_ignored(self, test_api):
        """Test that duplicate links are ignored."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-123")
        test_api._link_experiment_to_hypothesis(hyp_id, "exp-123")  # Duplicate

        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert len(experiments) == 1

    def test_get_experiments_empty(self, test_api):
        """Test getting experiments when none linked."""
        hyp_id = test_api.create_hypothesis(
            title="Test",
            thesis="Test",
            prediction="Test",
            falsification="Test",
            actor="user",
        )
        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert experiments == []


class TestPlatformAPIExceptions:
    """Tests for custom exception classes."""

    def test_platform_api_error_base(self):
        """Test PlatformAPIError is the base exception."""
        assert issubclass(PermissionError, PlatformAPIError)
        assert issubclass(NotFoundError, PlatformAPIError)

    def test_permission_error_message(self):
        """Test PermissionError preserves message."""
        error = PermissionError("Test permission error")
        assert str(error) == "Test permission error"

    def test_not_found_error_message(self):
        """Test NotFoundError preserves message."""
        error = NotFoundError("Resource not found")
        assert str(error) == "Resource not found"


class TestPlatformAPICompareExperiments:
    """Tests for experiment comparison functionality."""

    def test_compare_experiments_empty_list(self, test_api):
        """Test comparing empty experiment list returns empty DataFrame."""
        result = test_api.compare_experiments([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("hrp.api.platform.PlatformAPI.get_experiment")
    def test_compare_experiments_with_data(self, mock_get_exp, test_api):
        """Test comparing experiments returns proper DataFrame."""
        mock_get_exp.side_effect = [
            {
                "experiment_id": "exp-1",
                "metrics": {"sharpe_ratio": 1.5, "total_return": 0.25},
            },
            {
                "experiment_id": "exp-2",
                "metrics": {"sharpe_ratio": 1.2, "total_return": 0.20},
            },
        ]

        result = test_api.compare_experiments(
            ["exp-1", "exp-2"], metrics=["sharpe_ratio", "total_return"]
        )
        assert len(result) == 2
        assert "sharpe_ratio" in result.columns
        assert "total_return" in result.columns

    @patch("hrp.api.platform.PlatformAPI.get_experiment")
    def test_compare_experiments_not_found(self, mock_get_exp, test_api):
        """Test comparing experiments handles missing experiments."""
        mock_get_exp.return_value = None

        result = test_api.compare_experiments(["exp-1", "exp-2"])
        assert result.empty


class TestPlatformAPIIntegration:
    """Integration tests for multi-step workflows."""

    def test_full_hypothesis_lifecycle(self, test_api):
        """Test complete hypothesis lifecycle from creation to deployment."""
        # Create
        hyp_id = test_api.create_hypothesis(
            title="Integration Test Hypothesis",
            thesis="Test thesis for integration",
            prediction="Prediction",
            falsification="Falsification criteria",
            actor="user",
        )
        assert hyp_id.startswith("HYP-")

        # Verify created
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "draft"

        # Move to testing
        test_api.update_hypothesis(hyp_id, status="testing", actor="user")
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "testing"

        # Move to validated
        test_api.update_hypothesis(
            hyp_id, status="validated", outcome="Validated successfully", actor="user"
        )
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "validated"
        assert hyp["outcome"] == "Validated successfully"

        # Deploy
        result = test_api.approve_deployment(hyp_id, actor="user")
        assert result is True
        hyp = test_api.get_hypothesis(hyp_id)
        assert hyp["status"] == "deployed"

        # Verify in deployed list
        deployed = test_api.get_deployed_strategies()
        assert any(d["hypothesis_id"] == hyp_id for d in deployed)

        # Check lineage trail
        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        event_types = [e["event_type"] for e in lineage]
        assert "hypothesis_created" in event_types
        assert "hypothesis_updated" in event_types
        assert "deployment_approved" in event_types

    def test_hypothesis_with_experiments(self, test_api):
        """Test hypothesis workflow with linked experiments."""
        # Create hypothesis
        hyp_id = test_api.create_hypothesis(
            title="Experiment Linked Hypothesis",
            thesis="Test thesis",
            prediction="Prediction",
            falsification="Falsification",
            actor="user",
        )

        # Simulate running experiments
        for i in range(3):
            exp_id = f"exp-{i}"
            test_api._link_experiment_to_hypothesis(hyp_id, exp_id)
            test_api.log_event(
                event_type="experiment_run",
                actor="user",
                hypothesis_id=hyp_id,
                experiment_id=exp_id,
                details={"sharpe": 1.0 + i * 0.1},
            )

        # Verify experiments linked
        experiments = test_api.get_experiments_for_hypothesis(hyp_id)
        assert len(experiments) == 3

        # Verify lineage
        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        exp_events = [e for e in lineage if e["event_type"] == "experiment_run"]
        assert len(exp_events) == 3

    def test_agent_creates_user_deploys(self, test_api):
        """Test workflow where agent creates hypothesis but user deploys."""
        # Agent creates hypothesis
        hyp_id = test_api.create_hypothesis(
            title="Agent Discovery",
            thesis="Discovered pattern",
            prediction="Pattern holds",
            falsification="Pattern fails",
            actor="agent:discovery",
        )

        # Agent moves to testing
        test_api.update_hypothesis(hyp_id, status="testing", actor="agent:backtest")

        # Agent validates
        test_api.update_hypothesis(
            hyp_id, status="validated", actor="agent:analysis", outcome="Confirmed"
        )

        # Agent tries to deploy - should fail
        with pytest.raises(PermissionError):
            test_api.approve_deployment(hyp_id, actor="agent:deploy")

        # User deploys - should succeed
        result = test_api.approve_deployment(hyp_id, actor="user")
        assert result is True

        # Check lineage shows mixed actors
        lineage = test_api.get_lineage(hypothesis_id=hyp_id)
        actors = set(e["actor"] for e in lineage)
        assert "agent:discovery" in actors
        assert "agent:backtest" in actors
        assert "agent:analysis" in actors
        assert "user" in actors
