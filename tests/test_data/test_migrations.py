"""Tests for database schema migrations and integrity."""

import os
import tempfile

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import TABLES, create_tables, verify_schema


@pytest.fixture
def fresh_db():
    """Create a fresh temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()

    yield db_path

    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        if os.path.exists(db_path + ext):
            os.remove(db_path + ext)


class TestDatabaseMigrations:
    """Tests for database schema creation and migrations."""

    def test_create_tables_fresh_db(self, fresh_db):
        """create_tables successfully creates schema on empty database."""
        create_tables(fresh_db)

        db = get_db(fresh_db)
        result = db.fetchone("SELECT 1")
        assert result == (1,)

    def test_create_tables_idempotent(self, fresh_db):
        """Running create_tables twice doesn't error."""
        create_tables(fresh_db)
        create_tables(fresh_db)  # Should not raise

        assert verify_schema(fresh_db)

    def test_all_tables_exist(self, fresh_db):
        """All expected tables are created."""
        create_tables(fresh_db)

        db = get_db(fresh_db)
        for table_name in TABLES.keys():
            result = db.fetchone(f"SELECT COUNT(*) FROM {table_name}")
            assert result is not None, f"Table {table_name} does not exist"

    def test_verify_schema_returns_true(self, fresh_db):
        """verify_schema returns True when all tables exist."""
        create_tables(fresh_db)
        assert verify_schema(fresh_db) is True

    def test_verify_schema_returns_false_missing_table(self, fresh_db):
        """verify_schema returns False when tables are missing."""
        # Don't create tables - just get a fresh db
        db = get_db(fresh_db)
        # Create just one table to have a valid db file
        db.execute("CREATE TABLE dummy (id INTEGER)")
        DatabaseManager.reset()

        assert verify_schema(fresh_db) is False

    def test_hypothesis_experiments_no_fk_constraint(self, fresh_db):
        """hypothesis_experiments FK constraint intentionally removed (DuckDB 1.4.3 limitation).

        FK constraints on tables that reference hypotheses were removed because DuckDB 1.4.3
        prevents UPDATE on parent tables when FK constraints exist. Referential integrity
        is validated via SQL JOIN checks in migrate_constraints.py instead.
        """
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Can insert experiment link without hypothesis (FK not enforced)
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
            VALUES ('HYP-FAKE-001', 'exp-123')
            """
        )

        # Verify insert succeeded
        result = db.fetchone(
            "SELECT hypothesis_id FROM hypothesis_experiments WHERE hypothesis_id = 'HYP-FAKE-001'"
        )
        assert result is not None
        assert result[0] == 'HYP-FAKE-001'

    def test_unique_constraint_hypothesis_id(self, fresh_db):
        """Unique constraint on hypothesis_id is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Insert a hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
            VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction', 'draft')
            """
        )

        # Try to insert duplicate - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-2023-001', 'Test2', 'Thesis2', 'Prediction2', 'draft')
                """
            )

    def test_unique_constraint_prices(self, fresh_db):
        """Unique constraint on prices (symbol, date) is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # First need to add symbol
        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

        # Insert a price
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, source)
            VALUES ('AAPL', '2023-01-01', 150.0, 'test')
            """
        )

        # Try to insert duplicate - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES ('AAPL', '2023-01-01', 151.0, 'test')
                """
            )

    def test_check_constraint_hypothesis_status(self, fresh_db):
        """Check constraint on hypothesis status is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Try to insert invalid status - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction', 'invalid_status')
                """
            )

    def test_check_constraint_positive_price(self, fresh_db):
        """Check constraint on positive close price is enforced."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")

        # Try to insert negative price - should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close, source)
                VALUES ('AAPL', '2023-01-01', -10.0, 'test')
                """
            )

    def test_table_creation_order(self, fresh_db):
        """Tables are created in correct FK dependency order."""
        create_tables(fresh_db)
        db = get_db(fresh_db)

        # Verify we can insert in dependency order
        db.execute("INSERT INTO symbols (symbol) VALUES ('AAPL')")
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
            VALUES ('HYP-2023-001', 'Test', 'Thesis', 'Prediction', 'draft')
            """
        )
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
            VALUES ('HYP-2023-001', 'exp-001')
            """
        )

        # Verify data was inserted
        result = db.fetchone("SELECT COUNT(*) FROM hypothesis_experiments")
        assert result[0] == 1
