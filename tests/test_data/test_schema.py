"""
Comprehensive tests for database schema management.

Tests cover:
- TABLES and INDEXES constants
- create_tables function
- drop_all_tables function
- get_table_counts function
- verify_schema function
- main() CLI entry point
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.schema import (
    TABLES,
    INDEXES,
    create_tables,
    drop_all_tables,
    get_table_counts,
    verify_schema,
    main,
    migrate_add_sector_columns,
)
from hrp.data.db import DatabaseManager, get_db


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def schema_test_db():
    """Create a temporary database for schema tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    os.environ["HRP_DB_PATH"] = db_path

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def initialized_db(schema_test_db):
    """Create a temporary database with schema initialized."""
    create_tables(schema_test_db)
    return schema_test_db


# =============================================================================
# TABLES and INDEXES Constants Tests
# =============================================================================


class TestTablesConstant:
    """Tests for TABLES constant."""

    def test_tables_not_empty(self):
        """TABLES should contain table definitions."""
        assert len(TABLES) > 0

    def test_tables_is_dict(self):
        """TABLES should be a dictionary."""
        assert isinstance(TABLES, dict)

    def test_tables_have_string_keys(self):
        """TABLES keys should be table names (strings)."""
        for key in TABLES.keys():
            assert isinstance(key, str)

    def test_tables_have_string_values(self):
        """TABLES values should be SQL CREATE statements (strings)."""
        for value in TABLES.values():
            assert isinstance(value, str)

    def test_tables_contain_create_table(self):
        """TABLES values should contain CREATE TABLE statements."""
        for table_name, sql in TABLES.items():
            assert "CREATE TABLE" in sql, f"{table_name} missing CREATE TABLE"

    def test_expected_tables_exist(self):
        """TABLES should contain all expected core tables."""
        expected_tables = [
            "symbols",
            "prices",
            "features",
            "universe",
            "hypotheses",
            "lineage",
            "ingestion_log",
            "data_sources",
            "fundamentals",
            "corporate_actions",
        ]
        for table in expected_tables:
            assert table in TABLES, f"Missing expected table: {table}"

    def test_symbols_table_has_primary_key(self):
        """symbols table should have symbol as primary key."""
        assert "symbol VARCHAR PRIMARY KEY" in TABLES["symbols"]

    def test_prices_table_has_composite_primary_key(self):
        """prices table should have composite primary key (symbol, date)."""
        assert "PRIMARY KEY (symbol, date)" in TABLES["prices"]

    def test_prices_table_has_check_constraints(self):
        """prices table should have CHECK constraints."""
        assert "CHECK (close > 0)" in TABLES["prices"]
        assert "CHECK (volume IS NULL OR volume >= 0)" in TABLES["prices"]

    def test_hypotheses_table_has_status_check(self):
        """hypotheses table should have status CHECK constraint."""
        assert "CHECK (status IN" in TABLES["hypotheses"]

    def test_lineage_table_has_event_type_check(self):
        """lineage table should have event_type CHECK constraint."""
        assert "CHECK (event_type IN" in TABLES["lineage"]


class TestIndexesConstant:
    """Tests for INDEXES constant."""

    def test_indexes_not_empty(self):
        """INDEXES should contain index definitions."""
        assert len(INDEXES) > 0

    def test_indexes_is_list(self):
        """INDEXES should be a list."""
        assert isinstance(INDEXES, list)

    def test_indexes_are_strings(self):
        """INDEXES entries should be SQL strings."""
        for idx in INDEXES:
            assert isinstance(idx, str)

    def test_indexes_contain_create_index(self):
        """INDEXES entries should contain CREATE INDEX."""
        for idx in INDEXES:
            assert "CREATE INDEX" in idx

    def test_expected_indexes_exist(self):
        """INDEXES should contain performance-critical indexes."""
        index_names = " ".join(INDEXES)
        assert "idx_prices_symbol_date" in index_names
        assert "idx_features_symbol_date" in index_names
        assert "idx_lineage_hypothesis" in index_names


# =============================================================================
# create_tables Tests
# =============================================================================


class TestCreateTables:
    """Tests for create_tables function."""

    def test_create_tables_creates_all_tables(self, schema_test_db):
        """create_tables should create all defined tables."""
        create_tables(schema_test_db)

        db = get_db(schema_test_db)
        with db.connection() as conn:
            for table_name in TABLES.keys():
                # Query should not raise
                result = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                assert result is not None

    def test_create_tables_idempotent(self, schema_test_db):
        """create_tables should be idempotent (can run multiple times)."""
        create_tables(schema_test_db)
        create_tables(schema_test_db)  # Should not raise

        db = get_db(schema_test_db)
        with db.connection() as conn:
            result = conn.execute("SELECT 1 FROM symbols LIMIT 1")
            assert result is not None

    def test_create_tables_creates_indexes(self, schema_test_db):
        """create_tables should create all indexes."""
        create_tables(schema_test_db)

        db = get_db(schema_test_db)
        with db.connection() as conn:
            # Query DuckDB's index information
            result = conn.execute("""
                SELECT index_name FROM duckdb_indexes()
            """).fetchall()
            index_names = [r[0] for r in result]

            # Check some expected indexes exist
            assert any("prices" in name.lower() for name in index_names)

    def test_create_tables_with_none_path(self, schema_test_db):
        """create_tables with None should use default db path."""
        # Since HRP_DB_PATH is set in fixture, this should work
        create_tables(None)

        db = get_db()
        with db.connection() as conn:
            result = conn.execute("SELECT 1 FROM symbols LIMIT 1")
            assert result is not None


# =============================================================================
# drop_all_tables Tests
# =============================================================================


class TestDropAllTables:
    """Tests for drop_all_tables function."""

    def test_drop_all_tables_removes_tables(self, initialized_db):
        """drop_all_tables should remove all tables."""
        drop_all_tables(initialized_db)

        db = get_db(initialized_db)
        with db.connection() as conn:
            # Tables should no longer exist
            result = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
            """).fetchall()
            table_names = [r[0] for r in result]

            for table_name in TABLES.keys():
                assert table_name not in table_names

    def test_drop_all_tables_on_empty_db(self, schema_test_db):
        """drop_all_tables should not raise on empty database."""
        # Don't initialize, just drop
        drop_all_tables(schema_test_db)  # Should not raise

    def test_drop_all_tables_respects_fk_order(self, initialized_db):
        """drop_all_tables should drop in correct order for FK deps."""
        # Insert data with FK relationships
        db = get_db(initialized_db)
        with db.connection() as conn:
            conn.execute("INSERT INTO symbols (symbol, name) VALUES ('AAPL', 'Apple')")
            conn.execute("""
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2024-01-01', 150.0)
            """)

        # Should not raise FK violation
        drop_all_tables(initialized_db)


# =============================================================================
# get_table_counts Tests
# =============================================================================


class TestGetTableCounts:
    """Tests for get_table_counts function."""

    def test_get_table_counts_empty_tables(self, initialized_db):
        """get_table_counts should return 0 for empty tables."""
        counts = get_table_counts(initialized_db)

        assert isinstance(counts, dict)
        for table_name in TABLES.keys():
            assert table_name in counts
            assert counts[table_name] == 0

    def test_get_table_counts_with_data(self, initialized_db):
        """get_table_counts should return correct counts."""
        db = get_db(initialized_db)
        with db.connection() as conn:
            conn.execute("INSERT INTO symbols (symbol, name) VALUES ('AAPL', 'Apple')")
            conn.execute("INSERT INTO symbols (symbol, name) VALUES ('MSFT', 'Microsoft')")

        counts = get_table_counts(initialized_db)

        assert counts["symbols"] == 2

    def test_get_table_counts_missing_table(self, schema_test_db):
        """get_table_counts should return -1 for missing tables."""
        # Don't initialize schema
        counts = get_table_counts(schema_test_db)

        for table_name in TABLES.keys():
            assert counts[table_name] == -1

    def test_get_table_counts_returns_all_tables(self, initialized_db):
        """get_table_counts should return counts for all defined tables."""
        counts = get_table_counts(initialized_db)

        assert len(counts) == len(TABLES)


# =============================================================================
# verify_schema Tests
# =============================================================================


class TestVerifySchema:
    """Tests for verify_schema function."""

    def test_verify_schema_valid(self, initialized_db):
        """verify_schema should return True for valid schema."""
        result = verify_schema(initialized_db)
        assert result is True

    def test_verify_schema_missing_tables(self, schema_test_db):
        """verify_schema should return False for missing tables."""
        # Don't initialize schema
        result = verify_schema(schema_test_db)
        assert result is False

    def test_verify_schema_partial_tables(self, schema_test_db):
        """verify_schema should return False if some tables missing."""
        db = get_db(schema_test_db)
        with db.connection() as conn:
            # Only create symbols table
            conn.execute(TABLES["symbols"])

        result = verify_schema(schema_test_db)
        assert result is False


# =============================================================================
# main() CLI Tests
# =============================================================================


class TestMainCLI:
    """Tests for main() CLI entry point."""

    def test_main_init(self, schema_test_db):
        """main --init should create tables."""
        with patch.object(sys, "argv", ["schema", "--init", "--db", schema_test_db]):
            main()

        assert verify_schema(schema_test_db) is True

    @patch("builtins.input", return_value="yes")
    def test_main_drop_confirmed(self, mock_input, initialized_db):
        """main --drop with 'yes' should drop tables."""
        with patch.object(sys, "argv", ["schema", "--drop", "--db", initialized_db]):
            main()

        assert verify_schema(initialized_db) is False

    @patch("builtins.input", return_value="no")
    def test_main_drop_aborted(self, mock_input, initialized_db, capsys):
        """main --drop with 'no' should abort."""
        with patch.object(sys, "argv", ["schema", "--drop", "--db", initialized_db]):
            main()

        captured = capsys.readouterr()
        assert "Aborted" in captured.out
        # Tables should still exist
        assert verify_schema(initialized_db) is True

    def test_main_verify_valid(self, initialized_db, capsys):
        """main --verify should print 'Schema OK' for valid schema."""
        with patch.object(sys, "argv", ["schema", "--verify", "--db", initialized_db]):
            main()

        captured = capsys.readouterr()
        assert "Schema OK" in captured.out

    def test_main_verify_invalid(self, schema_test_db, capsys):
        """main --verify should exit 1 for invalid schema."""
        with patch.object(sys, "argv", ["schema", "--verify", "--db", schema_test_db]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "verification failed" in captured.out

    def test_main_counts(self, initialized_db, capsys):
        """main --counts should print table counts."""
        # Add some data
        db = get_db(initialized_db)
        with db.connection() as conn:
            conn.execute("INSERT INTO symbols (symbol, name) VALUES ('AAPL', 'Apple')")

        with patch.object(sys, "argv", ["schema", "--counts", "--db", initialized_db]):
            main()

        captured = capsys.readouterr()
        assert "Table Row Counts" in captured.out
        assert "symbols" in captured.out

    def test_main_no_args(self, capsys):
        """main with no args should print help."""
        with patch.object(sys, "argv", ["schema"]):
            main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "--init" in captured.out


# =============================================================================
# FK Constraint Tests
# =============================================================================


class TestForeignKeyConstraints:
    """Tests for foreign key relationships in schema."""

    def test_prices_requires_symbol(self, initialized_db):
        """prices table should require symbol to exist in symbols."""
        db = get_db(initialized_db)
        with db.connection() as conn:
            # Try to insert price for non-existent symbol
            with pytest.raises(Exception):
                conn.execute("""
                    INSERT INTO prices (symbol, date, close)
                    VALUES ('INVALID', '2024-01-01', 100.0)
                """)

    def test_features_requires_symbol(self, initialized_db):
        """features table should require symbol to exist in symbols."""
        db = get_db(initialized_db)
        with db.connection() as conn:
            # Try to insert feature for non-existent symbol
            with pytest.raises(Exception):
                conn.execute("""
                    INSERT INTO features (symbol, date, feature_name, value)
                    VALUES ('INVALID', '2024-01-01', 'momentum_20d', 0.05)
                """)

    def test_universe_requires_symbol(self, initialized_db):
        """universe table should require symbol to exist in symbols."""
        db = get_db(initialized_db)
        with db.connection() as conn:
            # Try to insert universe entry for non-existent symbol
            with pytest.raises(Exception):
                conn.execute("""
                    INSERT INTO universe (symbol, date, in_universe)
                    VALUES ('INVALID', '2024-01-01', TRUE)
                """)

    def test_valid_fk_insert(self, initialized_db):
        """Valid FK insertions should succeed."""
        db = get_db(initialized_db)
        with db.connection() as conn:
            # First insert symbol
            conn.execute("INSERT INTO symbols (symbol, name) VALUES ('AAPL', 'Apple')")

            # Then insert related data
            conn.execute("""
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2024-01-01', 150.0)
            """)
            conn.execute("""
                INSERT INTO features (symbol, date, feature_name, value)
                VALUES ('AAPL', '2024-01-01', 'momentum_20d', 0.05)
            """)
            conn.execute("""
                INSERT INTO universe (symbol, date, in_universe)
                VALUES ('AAPL', '2024-01-01', TRUE)
            """)

        # Verify data was inserted
        counts = get_table_counts(initialized_db)
        assert counts["symbols"] == 1
        assert counts["prices"] == 1
        assert counts["features"] == 1
        assert counts["universe"] == 1


# =============================================================================
# Sector Schema Migration Tests
# =============================================================================


class TestSymbolsSchema:
    """Tests for symbols table schema sector columns."""

    def test_symbols_has_sector_column(self, initialized_db):
        """Symbols table has sector column after migration."""
        from hrp.data.schema import migrate_add_sector_columns
        db = get_db(initialized_db)
        migrate_add_sector_columns()
        result = db.fetchdf("DESCRIBE symbols")
        columns = result["column_name"].tolist()
        assert "sector" in columns

    def test_symbols_has_industry_column(self, initialized_db):
        """Symbols table has industry column after migration."""
        from hrp.data.schema import migrate_add_sector_columns
        db = get_db(initialized_db)
        migrate_add_sector_columns()
        result = db.fetchdf("DESCRIBE symbols")
        columns = result["column_name"].tolist()
        assert "industry" in columns

    def test_sector_index_exists(self, initialized_db):
        """Index on sector column exists after migration."""
        from hrp.data.schema import migrate_add_sector_columns
        db = get_db(initialized_db)
        migrate_add_sector_columns()
        # Check for index
        result = db.fetchdf("""
            SELECT index_name FROM duckdb_indexes()
            WHERE table_name = 'symbols'
        """)
        index_names = result["index_name"].tolist()
        assert any("sector" in idx.lower() for idx in index_names)
