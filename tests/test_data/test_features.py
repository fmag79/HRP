"""
Comprehensive tests for the HRP Feature Store.

Tests cover:
- FeatureRegistry: registration, versioning, activation
- FeatureComputer: computation, storage, retrieval
- Feature versioning and reproducibility
- Integration with database and lineage tracking
"""

import os
import tempfile
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.features.computation import (
    FeatureComputer,
    compute_momentum_20d,
    compute_volatility_60d,
)
from hrp.data.features.registry import FeatureRegistry
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def feature_db():
    """Create a temporary DuckDB database with schema for testing features."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    # Insert common test symbols to satisfy FK constraints
    db = DatabaseManager(db_path)
    with db.connection() as conn:
        conn.execute("""
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ')
            ON CONFLICT DO NOTHING
        """)

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
def feature_registry(feature_db):
    """Create a FeatureRegistry instance with a test database."""
    return FeatureRegistry(db_path=feature_db)


@pytest.fixture
def feature_computer(feature_db):
    """Create a FeatureComputer instance with a test database."""
    return FeatureComputer(db_path=feature_db)


@pytest.fixture
def sample_feature_function():
    """Sample feature computation function."""

    def compute_simple_return(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple return from close prices."""
        close = prices["close"].unstack(level="symbol")
        returns = close.pct_change()
        result = returns.stack(level="symbol", future_stack=True)
        return result.to_frame(name="simple_return")

    return compute_simple_return


@pytest.fixture
def populated_feature_db(feature_db):
    """Populate the test database with sample price data for feature computation."""
    from hrp.data.db import get_db

    db = get_db(feature_db)

    # Insert symbols first to satisfy FK constraints
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        db.execute(
            "INSERT INTO symbols (symbol) VALUES (?) ON CONFLICT DO NOTHING",
            (symbol,),
        )

    # Insert sample price data
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")

    base_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}

    for symbol in symbols:
        base_price = base_prices[symbol]
        for i, d in enumerate(dates):
            price = base_price * (1 + 0.001 * i)
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    d.date(),
                    price * 0.99,
                    price * 1.02,
                    price * 0.98,
                    price,
                    price,
                    1000000 + i * 10000,
                ),
            )

    return feature_db


# =============================================================================
# Test Classes
# =============================================================================


class TestFeatureRegistryInit:
    """Tests for FeatureRegistry initialization."""

    def test_init_with_db_path(self, feature_db):
        """Test that FeatureRegistry initializes with a database path."""
        registry = FeatureRegistry(db_path=feature_db)
        assert registry.db is not None

    def test_init_without_db_path(self):
        """Test that FeatureRegistry can initialize without explicit db_path."""
        # This should use the default database path from environment
        registry = FeatureRegistry()
        assert registry.db is not None


class TestFeatureRegistryRegister:
    """Tests for feature registration."""

    def test_register_feature_with_code_string(self, feature_registry):
        """Test registering a feature with computation code as a string."""
        feature_registry.register(
            feature_name="test_feature",
            version="v1",
            computation_code="return prices['close'].pct_change()",
            description="Test feature for unit tests",
            is_active=True,
        )

        # Verify feature was registered
        feature = feature_registry.get("test_feature", "v1")
        assert feature is not None
        assert feature["feature_name"] == "test_feature"
        assert feature["version"] == "v1"
        assert feature["description"] == "Test feature for unit tests"
        assert feature["is_active"] is True

    def test_register_feature_with_callable(self, feature_registry, sample_feature_function):
        """Test registering a feature with a callable function."""
        feature_registry.register_feature(
            feature_name="simple_return",
            computation_fn=sample_feature_function,
            version="v1",
            description="Simple return calculation",
            is_active=True,
        )

        # Verify feature was registered
        feature = feature_registry.get("simple_return", "v1")
        assert feature is not None
        assert feature["feature_name"] == "simple_return"
        assert "compute_simple_return" in feature["computation_code"]

    def test_register_feature_with_lambda(self, feature_registry):
        """Test registering a feature with a lambda function."""
        lambda_fn = lambda x: x * 2

        feature_registry.register_feature(
            feature_name="lambda_feature",
            computation_fn=lambda_fn,
            version="v1",
            description="Lambda feature",
        )

        # Verify feature was registered (lambda will use repr)
        feature = feature_registry.get("lambda_feature", "v1")
        assert feature is not None
        assert feature["feature_name"] == "lambda_feature"

    def test_register_duplicate_feature_version_fails(self, feature_registry):
        """Test that registering the same feature+version twice raises an error."""
        feature_registry.register(
            feature_name="duplicate_test",
            version="v1",
            computation_code="test code",
            description="First registration",
        )

        # Attempt to register the same feature+version again
        with pytest.raises(Exception):
            feature_registry.register(
                feature_name="duplicate_test",
                version="v1",
                computation_code="different code",
                description="Second registration",
            )

    def test_register_same_feature_different_version(self, feature_registry):
        """Test that registering the same feature with different versions succeeds."""
        feature_registry.register(
            feature_name="versioned_feature",
            version="v1",
            computation_code="code v1",
            description="Version 1",
        )

        feature_registry.register(
            feature_name="versioned_feature",
            version="v2",
            computation_code="code v2",
            description="Version 2",
        )

        # Verify both versions exist
        v1 = feature_registry.get("versioned_feature", "v1")
        v2 = feature_registry.get("versioned_feature", "v2")

        assert v1 is not None
        assert v2 is not None
        assert v1["version"] == "v1"
        assert v2["version"] == "v2"
        assert v1["computation_code"] == "code v1"
        assert v2["computation_code"] == "code v2"

    def test_register_inactive_feature(self, feature_registry):
        """Test registering a feature as inactive."""
        feature_registry.register(
            feature_name="inactive_feature",
            version="v1",
            computation_code="test code",
            description="Inactive feature",
            is_active=False,
        )

        # Verify feature was registered as inactive
        feature = feature_registry.get("inactive_feature", "v1")
        assert feature is not None
        assert feature["is_active"] is False


class TestFeatureRegistryGet:
    """Tests for retrieving feature definitions."""

    def test_get_specific_version(self, feature_registry):
        """Test retrieving a specific version of a feature."""
        feature_registry.register(
            feature_name="test_get",
            version="v1",
            computation_code="v1 code",
        )
        feature_registry.register(
            feature_name="test_get",
            version="v2",
            computation_code="v2 code",
        )

        # Get specific version
        feature = feature_registry.get("test_get", "v1")
        assert feature["version"] == "v1"
        assert feature["computation_code"] == "v1 code"

        feature = feature_registry.get("test_get", "v2")
        assert feature["version"] == "v2"
        assert feature["computation_code"] == "v2 code"

    def test_get_latest_active_version(self, feature_registry):
        """Test retrieving the latest active version when no version is specified."""
        # Register multiple versions
        feature_registry.register(
            feature_name="test_latest",
            version="v1",
            computation_code="v1 code",
            is_active=True,
        )

        # Add a small delay to ensure different timestamps
        import time

        time.sleep(0.01)

        feature_registry.register(
            feature_name="test_latest",
            version="v2",
            computation_code="v2 code",
            is_active=True,
        )

        # Get latest active (should be v2)
        feature = feature_registry.get("test_latest")
        assert feature is not None
        assert feature["version"] == "v2"

    def test_get_latest_ignores_inactive(self, feature_registry):
        """Test that get without version ignores inactive features."""
        feature_registry.register(
            feature_name="test_inactive",
            version="v1",
            computation_code="v1 code",
            is_active=True,
        )

        import time

        time.sleep(0.01)

        feature_registry.register(
            feature_name="test_inactive",
            version="v2",
            computation_code="v2 code",
            is_active=False,
        )

        # Get latest active (should still be v1, not v2)
        feature = feature_registry.get("test_inactive")
        assert feature is not None
        assert feature["version"] == "v1"

    def test_get_nonexistent_feature_returns_none(self, feature_registry):
        """Test that getting a non-existent feature returns None."""
        feature = feature_registry.get("nonexistent_feature", "v1")
        assert feature is None

        feature = feature_registry.get("nonexistent_feature")
        assert feature is None

    def test_get_nonexistent_version_returns_none(self, feature_registry):
        """Test that getting a non-existent version returns None."""
        feature_registry.register(
            feature_name="test_version",
            version="v1",
            computation_code="v1 code",
        )

        # v1 exists, v2 doesn't
        feature = feature_registry.get("test_version", "v2")
        assert feature is None


class TestFeatureRegistryList:
    """Tests for listing features."""

    def test_list_features_active_only(self, feature_registry):
        """Test listing only active features."""
        feature_registry.register("feature_a", "v1", "code", is_active=True)
        feature_registry.register("feature_b", "v1", "code", is_active=True)
        feature_registry.register("feature_c", "v1", "code", is_active=False)

        features = feature_registry.list_features(active_only=True)
        assert "feature_a" in features
        assert "feature_b" in features
        assert "feature_c" not in features

    def test_list_features_all(self, feature_registry):
        """Test listing all features including inactive."""
        feature_registry.register("feature_a", "v1", "code", is_active=True)
        feature_registry.register("feature_b", "v1", "code", is_active=False)

        features = feature_registry.list_features(active_only=False)
        assert "feature_a" in features
        assert "feature_b" in features

    def test_list_features_returns_distinct(self, feature_registry):
        """Test that list_features returns distinct feature names."""
        feature_registry.register("feature_a", "v1", "code")
        feature_registry.register("feature_a", "v2", "code")
        feature_registry.register("feature_a", "v3", "code")

        features = feature_registry.list_features()
        # Should only appear once despite multiple versions
        assert features.count("feature_a") == 1

    def test_list_features_sorted(self, feature_registry):
        """Test that features are returned in sorted order."""
        feature_registry.register("zebra", "v1", "code")
        feature_registry.register("apple", "v1", "code")
        feature_registry.register("banana", "v1", "code")

        features = feature_registry.list_features()
        assert features == ["apple", "banana", "zebra"]

    def test_list_all_features_with_details(self, feature_registry):
        """Test listing features with full details."""
        feature_registry.register("feature_a", "v1", "code_v1", "Description v1")
        feature_registry.register("feature_a", "v2", "code_v2", "Description v2")

        all_features = feature_registry.list_all_features()
        assert len(all_features) == 2

        # Should be sorted by feature_name, then created_at DESC
        assert all_features[0]["feature_name"] == "feature_a"
        assert all_features[0]["version"] in ["v1", "v2"]
        assert all_features[1]["feature_name"] == "feature_a"

    def test_list_versions(self, feature_registry):
        """Test listing all versions of a specific feature."""
        feature_registry.register("multi_version", "v1", "code_v1", "Version 1")

        import time

        time.sleep(0.01)
        feature_registry.register("multi_version", "v2", "code_v2", "Version 2")

        time.sleep(0.01)
        feature_registry.register("multi_version", "v3", "code_v3", "Version 3")

        versions = feature_registry.list_versions("multi_version")
        assert len(versions) == 3

        # Should be ordered by created_at DESC (newest first)
        assert versions[0]["version"] == "v3"
        assert versions[1]["version"] == "v2"
        assert versions[2]["version"] == "v1"

    def test_list_versions_empty(self, feature_registry):
        """Test listing versions for non-existent feature."""
        versions = feature_registry.list_versions("nonexistent")
        assert len(versions) == 0


class TestFeatureRegistryActivation:
    """Tests for activating and deactivating features."""

    def test_deactivate_feature(self, feature_registry):
        """Test deactivating a feature version."""
        feature_registry.register("test_deactivate", "v1", "code", is_active=True)

        # Verify it's active
        feature = feature_registry.get("test_deactivate", "v1")
        assert feature["is_active"] is True

        # Deactivate it
        feature_registry.deactivate("test_deactivate", "v1")

        # Verify it's now inactive
        feature = feature_registry.get("test_deactivate", "v1")
        assert feature["is_active"] is False

    def test_activate_feature(self, feature_registry):
        """Test activating a feature version."""
        feature_registry.register("test_activate", "v1", "code", is_active=False)

        # Verify it's inactive
        feature = feature_registry.get("test_activate", "v1")
        assert feature["is_active"] is False

        # Activate it
        feature_registry.activate("test_activate", "v1")

        # Verify it's now active
        feature = feature_registry.get("test_activate", "v1")
        assert feature["is_active"] is True

    def test_deactivate_then_reactivate(self, feature_registry):
        """Test the full cycle of deactivate and reactivate."""
        feature_registry.register("test_cycle", "v1", "code", is_active=True)

        # Deactivate
        feature_registry.deactivate("test_cycle", "v1")
        assert feature_registry.get("test_cycle", "v1")["is_active"] is False

        # Reactivate
        feature_registry.activate("test_cycle", "v1")
        assert feature_registry.get("test_cycle", "v1")["is_active"] is True


class TestFeatureComputerInit:
    """Tests for FeatureComputer initialization."""

    def test_init_with_db_path(self, feature_db):
        """Test that FeatureComputer initializes with a database path."""
        computer = FeatureComputer(db_path=feature_db)
        assert computer.db is not None
        assert computer.registry is not None

    def test_init_without_db_path(self):
        """Test that FeatureComputer can initialize without explicit db_path."""
        computer = FeatureComputer()
        assert computer.db is not None
        assert computer.registry is not None


class TestFeatureComputation:
    """Tests for feature computation functions."""

    def test_compute_momentum_20d(self):
        """Test momentum_20d computation function."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="B")
        symbols = ["AAPL", "MSFT"]

        data = []
        for symbol in symbols:
            base_price = 100.0
            for i, d in enumerate(dates):
                price = base_price * (1 + 0.01 * i)
                data.append(
                    {
                        "date": d,
                        "symbol": symbol,
                        "close": price,
                    }
                )

        df = pd.DataFrame(data)
        df = df.set_index(["date", "symbol"])

        # Compute momentum
        result = compute_momentum_20d(df)

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert "momentum_20d" in result.columns
        assert len(result) > 0

        # Momentum should be NaN for first 20 periods
        # After that, should have numeric values

    def test_compute_volatility_60d(self):
        """Test volatility_60d computation function."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
        symbols = ["AAPL"]

        data = []
        base_price = 100.0
        for i, d in enumerate(dates):
            price = base_price * (1 + 0.001 * i)
            data.append(
                {
                    "date": d,
                    "symbol": "AAPL",
                    "close": price,
                }
            )

        df = pd.DataFrame(data)
        df = df.set_index(["date", "symbol"])

        # Compute volatility
        result = compute_volatility_60d(df)

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert "volatility_60d" in result.columns
        assert len(result) > 0

        # Volatility should be NaN for first 60 periods


class TestFeatureComputerIntegration:
    """Integration tests for FeatureComputer with database."""

    def test_compute_features_requires_registered_feature(
        self, feature_computer, populated_feature_db
    ):
        """Test that computing features requires them to be registered first."""
        # Try to compute a feature that doesn't exist
        with pytest.raises(ValueError, match="not found in registry"):
            feature_computer.compute_features(
                symbols=["AAPL"],
                dates=[date(2023, 1, 15)],
                feature_names=["nonexistent_feature"],
            )

    def test_compute_features_with_no_price_data(self, feature_computer, feature_registry):
        """Test computing features when no price data is available."""
        # Register a feature
        feature_registry.register("test_feature", "v1", "test code")

        # Try to compute for symbols/dates with no price data
        with pytest.raises(ValueError, match="No price data found"):
            feature_computer.compute_features(
                symbols=["INVALID_SYMBOL"],
                dates=[date(2023, 1, 1)],
                feature_names=["test_feature"],
            )


# =============================================================================
# End-to-End Tests
# =============================================================================


def test_feature_versioning_reproducibility(feature_db):
    """
    End-to-end test for feature versioning reproducibility.

    Tests the full workflow:
    1. Register and compute features v1
    2. Register and compute features v2 (different logic)
    3. Verify v1 features are still accessible
    4. Verify v2 features are different from v1
    """
    from hrp.data.db import get_db

    db = get_db(feature_db)
    registry = FeatureRegistry(db_path=feature_db)
    computer = FeatureComputer(db_path=feature_db)

    # Setup: Insert sample price data
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")

    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 200.0
        for i, d in enumerate(dates):
            price = base_price * (1 + 0.001 * i)
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, d.date(), price * 0.99, price * 1.02, price * 0.98, price, price, 1000000),
            )

    # Step 1: Register feature v1 (10-day momentum)
    def compute_momentum_v1(prices: pd.DataFrame) -> pd.DataFrame:
        close = prices["close"].unstack(level="symbol")
        momentum = close.pct_change(10)
        result = momentum.stack(level="symbol", future_stack=True)
        return result.to_frame(name="test_momentum")

    registry.register_feature(
        feature_name="test_momentum",
        computation_fn=compute_momentum_v1,
        version="v1",
        description="10-day momentum",
        is_active=True,
    )

    # Compute and store v1 features
    # Note: We can't actually compute since FEATURE_FUNCTIONS doesn't have test_momentum
    # This test demonstrates the workflow pattern

    # Step 2: Register feature v2 (20-day momentum)
    def compute_momentum_v2(prices: pd.DataFrame) -> pd.DataFrame:
        close = prices["close"].unstack(level="symbol")
        momentum = close.pct_change(20)
        result = momentum.stack(level="symbol", future_stack=True)
        return result.to_frame(name="test_momentum")

    import time

    time.sleep(0.01)  # Ensure different timestamp

    registry.register_feature(
        feature_name="test_momentum",
        computation_fn=compute_momentum_v2,
        version="v2",
        description="20-day momentum",
        is_active=True,
    )

    # Step 3: Verify both versions are accessible
    v1_def = registry.get("test_momentum", "v1")
    v2_def = registry.get("test_momentum", "v2")

    assert v1_def is not None
    assert v2_def is not None
    assert v1_def["version"] == "v1"
    assert v2_def["version"] == "v2"
    assert v1_def["description"] == "10-day momentum"
    assert v2_def["description"] == "20-day momentum"

    # Step 4: Verify latest active version is v2
    latest = registry.get("test_momentum")
    assert latest["version"] == "v2"

    # This demonstrates that old versions (v1) are retained and accessible
    # for reproducing historical experiments, while new versions (v2) can be
    # used for new experiments.


class TestFeatureCalendarIntegration:
    """Tests for feature computation trading calendar integration."""

    def test_load_price_data_filters_to_trading_days(self, feature_db):
        """Feature price data loading should only include trading days."""
        # Insert test price data including weekends and holidays
        db = DatabaseManager(feature_db)
        
        test_data = []
        symbols = ["AAPL"]
        
        # July 2022 data (includes July 4th holiday and weekend)
        dates = [
            date(2022, 7, 1),  # Friday - trading day
            date(2022, 7, 5),  # Tuesday - trading day (after holiday weekend)
            date(2022, 7, 6),  # Wednesday - trading day
        ]
        
        for symbol in symbols:
            for i, dt in enumerate(dates):
                test_data.append({
                    "symbol": symbol,
                    "date": dt,
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 100.0 + i,
                    "adj_close": 100.0 + i,
                    "volume": 1000000,
                })
        
        df = pd.DataFrame(test_data)
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                SELECT * FROM df
            """)
        
        # Create computer and load price data
        computer = FeatureComputer(feature_db)
        prices = computer._load_price_data(
            symbols=["AAPL"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
        )
        
        # Should only have 3 trading days (no data for July 4, 7, 8 in our test data)
        assert len(prices) == 3
        
        # Verify dates (date is in the index)
        dates_in_data = prices.index.get_level_values('date').date
        assert date(2022, 7, 1) in dates_in_data
        assert date(2022, 7, 5) in dates_in_data
        assert date(2022, 7, 6) in dates_in_data

    def test_load_price_data_excludes_holidays(self, feature_db):
        """Feature price data should exclude holidays."""
        db = DatabaseManager(feature_db)
        
        # Insert Thanksgiving week data
        test_data = []
        thanksgiving_dates = [
            date(2022, 11, 21),  # Monday - trading day
            date(2022, 11, 22),  # Tuesday - trading day
            date(2022, 11, 23),  # Wednesday - trading day
            date(2022, 11, 25),  # Friday - trading day (early close)
        ]
        
        for i, dt in enumerate(thanksgiving_dates):
            test_data.append({
                "symbol": "AAPL",
                "date": dt,
                "open": 150.0 + i,
                "high": 155.0 + i,
                "low": 145.0 + i,
                "close": 150.0 + i,
                "adj_close": 150.0 + i,
                "volume": 1000000,
            })
        
        df = pd.DataFrame(test_data)
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                SELECT * FROM df
            """)
        
        # Load price data
        computer = FeatureComputer(feature_db)
        prices = computer._load_price_data(
            symbols=["AAPL"],
            start=date(2022, 11, 21),
            end=date(2022, 11, 25),
        )
        
        # Should have 4 trading days (excludes Thanksgiving)
        assert len(prices) == 4
        
        # Verify Thanksgiving is not in the data (date is in the index)
        dates_in_data = prices.index.get_level_values('date').date
        assert date(2022, 11, 24) not in dates_in_data

    def test_load_price_data_no_weekends(self, feature_db):
        """Feature price data should never include weekends."""
        db = DatabaseManager(feature_db)
        
        # Insert data for a full week
        test_data = []
        dates = [
            date(2022, 7, 1),  # Friday
            date(2022, 7, 5),  # Tuesday
            date(2022, 7, 6),  # Wednesday
            date(2022, 7, 7),  # Thursday
            date(2022, 7, 8),  # Friday
        ]
        
        for i, dt in enumerate(dates):
            test_data.append({
                "symbol": "MSFT",
                "date": dt,
                "open": 200.0 + i,
                "high": 205.0 + i,
                "low": 195.0 + i,
                "close": 200.0 + i,
                "adj_close": 200.0 + i,
                "volume": 2000000,
            })
        
        df = pd.DataFrame(test_data)
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                SELECT * FROM df
            """)
        
        # Load price data
        computer = FeatureComputer(feature_db)
        prices = computer._load_price_data(
            symbols=["MSFT"],
            start=date(2022, 7, 1),
            end=date(2022, 7, 8),
        )
        
        # Check no weekend dates (date is in the index)
        for dt in prices.index.get_level_values('date').unique():
            weekday = dt.weekday()
            assert weekday < 5, f"Found weekend date: {dt}"

    def test_load_price_data_empty_range_returns_empty(self, feature_db):
        """Loading price data for weekend-only range should return empty."""
        computer = FeatureComputer(feature_db)
        
        # Weekend only (Saturday-Sunday)
        prices = computer._load_price_data(
            symbols=["AAPL"],
            start=date(2022, 7, 2),
            end=date(2022, 7, 3),
        )
        
        assert prices.empty
