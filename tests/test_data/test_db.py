"""
Comprehensive tests for the HRP Database module.

Tests cover:
- DatabaseManager class initialization and singleton behavior
- Connection context manager
- Query execution methods (execute, fetchone, fetchall, fetchdf)
- Thread safety
- get_db() factory function
- Schema integration
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest


class TestDatabaseManagerInit:
    """Tests for DatabaseManager initialization."""

    def test_init_creates_directory_and_file(self, temp_db):
        """Test that initialization creates the database file and parent directory."""
        from hrp.data.db import DatabaseManager

        # Reset singleton to ensure fresh instance
        DatabaseManager.reset()

        # Use a nested path to test directory creation
        nested_path = Path(temp_db).parent / "nested" / "test.duckdb"

        dm = DatabaseManager(nested_path)

        # Parent directory should be created
        assert nested_path.parent.exists()

        # Execute a query to create the database file
        dm.execute("SELECT 1")
        assert nested_path.exists()

        # Cleanup
        DatabaseManager.reset()
        if nested_path.exists():
            nested_path.unlink()
        if nested_path.parent.exists():
            nested_path.parent.rmdir()

    def test_init_with_string_path(self, temp_db):
        """Test initialization with string path."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        dm = DatabaseManager(temp_db)  # temp_db is a string
        assert isinstance(dm.db_path, Path)
        assert str(dm.db_path) == temp_db

        DatabaseManager.reset()

    def test_init_with_path_object(self, temp_db):
        """Test initialization with Path object."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        path_obj = Path(temp_db)
        dm = DatabaseManager(path_obj)
        assert dm.db_path == path_obj

        DatabaseManager.reset()

    def test_init_uses_env_var_when_no_path(self):
        """Test that initialization uses HRP_DATA_DIR environment variable."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HRP_DATA_DIR": tmpdir}):
                dm = DatabaseManager()
                expected_path = Path(tmpdir) / "hrp.duckdb"
                assert dm.db_path == expected_path

        DatabaseManager.reset()

    def test_init_uses_default_when_no_env(self):
        """Test that initialization uses default path when no env var is set."""
        from hrp.data.db import DEFAULT_DATA_DIR, DatabaseManager

        DatabaseManager.reset()

        # Temporarily remove HRP_DATA_DIR if it exists
        original_env = os.environ.get("HRP_DATA_DIR")
        if "HRP_DATA_DIR" in os.environ:
            del os.environ["HRP_DATA_DIR"]

        try:
            dm = DatabaseManager()
            expected_path = Path(DEFAULT_DATA_DIR).expanduser() / "hrp.duckdb"
            assert dm.db_path == expected_path
        finally:
            if original_env is not None:
                os.environ["HRP_DATA_DIR"] = original_env
            DatabaseManager.reset()


class TestDatabaseManagerSingleton:
    """Tests for DatabaseManager singleton pattern."""

    def test_singleton_returns_same_instance(self, temp_db):
        """Test that multiple instantiations return the same instance."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        dm1 = DatabaseManager(temp_db)
        dm2 = DatabaseManager(temp_db)

        assert dm1 is dm2

        DatabaseManager.reset()

    def test_singleton_ignores_subsequent_paths(self, temp_db):
        """Test that singleton ignores paths after first initialization."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
            other_path = f.name

        try:
            dm1 = DatabaseManager(temp_db)
            dm2 = DatabaseManager(other_path)

            # Both should point to original path
            assert dm1.db_path == dm2.db_path
            assert str(dm1.db_path) == temp_db
        finally:
            DatabaseManager.reset()
            if os.path.exists(other_path):
                os.remove(other_path)

    def test_reset_allows_new_instance(self, temp_db):
        """Test that reset() allows creating a new instance with different path."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
            other_path = f.name

        try:
            dm1 = DatabaseManager(temp_db)
            original_path = dm1.db_path

            DatabaseManager.reset()

            dm2 = DatabaseManager(other_path)

            assert dm1.db_path != dm2.db_path
            assert str(dm2.db_path) == other_path
        finally:
            DatabaseManager.reset()
            if os.path.exists(other_path):
                os.remove(other_path)


class TestConnectionContextManager:
    """Tests for the connection() context manager."""

    def test_connection_context_manager_basic(self, temp_db):
        """Test basic context manager usage."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        with dm.connection() as conn:
            assert conn is not None
            assert isinstance(conn, duckdb.DuckDBPyConnection)

        DatabaseManager.reset()

    def test_connection_context_manager_persists_data(self, temp_db):
        """Test that data persists across context manager calls."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        # Create and insert in first context
        with dm.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
            conn.execute("INSERT INTO test VALUES (1, 'first')")

        # Verify in second context
        with dm.connection() as conn:
            result = conn.execute("SELECT * FROM test WHERE id = 1").fetchone()
            assert result is not None
            assert result[0] == 1
            assert result[1] == "first"

        DatabaseManager.reset()

    def test_connection_context_manager_reraises_exception(self, temp_db):
        """Test that exceptions are re-raised after logging."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        with pytest.raises(duckdb.CatalogException):
            with dm.connection() as conn:
                # Query non-existent table
                conn.execute("SELECT * FROM nonexistent_table")

        DatabaseManager.reset()

    def test_connection_reuses_thread_local_connection(self, temp_db):
        """Test that the same connection is reused within a thread."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        connections = []

        with dm.connection() as conn1:
            connections.append(conn1)

        with dm.connection() as conn2:
            connections.append(conn2)

        # Should be the same connection object
        assert connections[0] is connections[1]

        DatabaseManager.reset()


class TestExecuteMethod:
    """Tests for the execute() method."""

    def test_execute_create_table(self, temp_db):
        """Test executing CREATE TABLE statement."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        result = dm.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name VARCHAR)")

        # Verify table exists
        tables = dm.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'test'"
        )
        assert len(tables) == 1

        DatabaseManager.reset()

    def test_execute_insert_without_params(self, temp_db):
        """Test executing INSERT without parameters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        dm.execute("INSERT INTO test VALUES (1, 'test')")

        result = dm.fetchone("SELECT * FROM test")
        assert result == (1, "test")

        DatabaseManager.reset()

    def test_execute_insert_with_params(self, temp_db):
        """Test executing INSERT with parameters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, name VARCHAR, value DECIMAL(10,2))")
        dm.execute("INSERT INTO test VALUES (?, ?, ?)", (42, "answer", 3.14))

        result = dm.fetchone("SELECT * FROM test WHERE id = ?", (42,))
        assert result[0] == 42
        assert result[1] == "answer"
        assert float(result[2]) == pytest.approx(3.14)

        DatabaseManager.reset()

    def test_execute_returns_relation(self, temp_db):
        """Test that execute returns a DuckDB relation."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER)")
        dm.execute("INSERT INTO test VALUES (1), (2), (3)")

        result = dm.execute("SELECT * FROM test")

        # Should be able to call fetchall on the result
        rows = result.fetchall()
        assert len(rows) == 3

        DatabaseManager.reset()

    def test_execute_with_none_params(self, temp_db):
        """Test execute with params=None (explicit)."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        result = dm.execute("SELECT 1 + 1 AS sum", None)
        assert result.fetchone()[0] == 2

        DatabaseManager.reset()


class TestFetchoneMethod:
    """Tests for the fetchone() method."""

    def test_fetchone_returns_tuple(self, temp_db):
        """Test that fetchone returns a tuple."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        dm.execute("INSERT INTO test VALUES (1, 'first'), (2, 'second')")

        result = dm.fetchone("SELECT * FROM test ORDER BY id")

        assert isinstance(result, tuple)
        assert result == (1, "first")

        DatabaseManager.reset()

    def test_fetchone_returns_none_for_empty_result(self, temp_db):
        """Test that fetchone returns None when no rows match."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER)")

        result = dm.fetchone("SELECT * FROM test WHERE id = 999")

        assert result is None

        DatabaseManager.reset()

    def test_fetchone_with_params(self, temp_db):
        """Test fetchone with query parameters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        dm.execute("INSERT INTO test VALUES (1, 'one'), (2, 'two'), (3, 'three')")

        result = dm.fetchone("SELECT name FROM test WHERE id = ?", (2,))

        assert result[0] == "two"

        DatabaseManager.reset()

    def test_fetchone_with_date_types(self, temp_db):
        """Test fetchone with DATE column types."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, trade_date DATE)")
        dm.execute("INSERT INTO test VALUES (1, '2024-01-15')")

        result = dm.fetchone("SELECT trade_date FROM test WHERE id = 1")

        assert result[0] == date(2024, 1, 15)

        DatabaseManager.reset()


class TestFetchallMethod:
    """Tests for the fetchall() method."""

    def test_fetchall_returns_list_of_tuples(self, temp_db):
        """Test that fetchall returns a list of tuples."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER)")
        for i in range(5):
            dm.execute("INSERT INTO test VALUES (?)", (i,))

        results = dm.fetchall("SELECT * FROM test ORDER BY id")

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, tuple) for r in results)
        assert [r[0] for r in results] == [0, 1, 2, 3, 4]

        DatabaseManager.reset()

    def test_fetchall_returns_empty_list_for_no_results(self, temp_db):
        """Test that fetchall returns empty list when no rows match."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER)")

        results = dm.fetchall("SELECT * FROM test")

        assert results == []

        DatabaseManager.reset()

    def test_fetchall_with_params(self, temp_db):
        """Test fetchall with query parameters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (category VARCHAR, value INTEGER)")
        dm.execute("INSERT INTO test VALUES ('A', 1), ('A', 2), ('B', 3), ('A', 4)")

        results = dm.fetchall("SELECT value FROM test WHERE category = ? ORDER BY value", ("A",))

        assert len(results) == 3
        assert [r[0] for r in results] == [1, 2, 4]

        DatabaseManager.reset()

    def test_fetchall_large_result_set(self, temp_db):
        """Test fetchall with a larger result set."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, value DECIMAL(10,2))")

        # Insert 1000 rows
        for i in range(1000):
            dm.execute("INSERT INTO test VALUES (?, ?)", (i, i * 1.5))

        results = dm.fetchall("SELECT * FROM test")

        assert len(results) == 1000

        DatabaseManager.reset()


class TestFetchdfMethod:
    """Tests for the fetchdf() method."""

    def test_fetchdf_returns_dataframe(self, temp_db):
        """Test that fetchdf returns a pandas DataFrame."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, value DECIMAL(10,2))")
        dm.execute("INSERT INTO test VALUES (1, 100.5), (2, 200.75)")

        df = dm.fetchdf("SELECT * FROM test ORDER BY id")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "value"]

        DatabaseManager.reset()

    def test_fetchdf_empty_result(self, temp_db):
        """Test fetchdf with empty result set."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")

        df = dm.fetchdf("SELECT * FROM test")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["id", "name"]

        DatabaseManager.reset()

    def test_fetchdf_preserves_column_types(self, temp_db):
        """Test that fetchdf preserves appropriate column types."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("""
            CREATE TABLE test (
                int_col INTEGER,
                float_col DECIMAL(10,2),
                str_col VARCHAR,
                date_col DATE,
                bool_col BOOLEAN
            )
        """)
        dm.execute("INSERT INTO test VALUES (1, 3.14, 'hello', '2024-01-15', true)")

        df = dm.fetchdf("SELECT * FROM test")

        assert df["int_col"].dtype in ["int64", "int32"]
        assert df["str_col"].dtype == "object"
        assert df["bool_col"].dtype == "bool"

        DatabaseManager.reset()

    def test_fetchdf_with_params(self, temp_db):
        """Test fetchdf with query parameters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (symbol VARCHAR, price DECIMAL(10,2))")
        dm.execute("INSERT INTO test VALUES ('AAPL', 150.0), ('MSFT', 300.0), ('AAPL', 155.0)")

        df = dm.fetchdf("SELECT * FROM test WHERE symbol = ?", ("AAPL",))

        assert len(df) == 2
        assert all(df["symbol"] == "AAPL")

        DatabaseManager.reset()

    def test_fetchdf_with_aggregation(self, temp_db):
        """Test fetchdf with aggregate functions."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (category VARCHAR, value INTEGER)")
        dm.execute("""
            INSERT INTO test VALUES
            ('A', 10), ('A', 20), ('A', 30),
            ('B', 100), ('B', 200)
        """)

        df = dm.fetchdf("""
            SELECT category, SUM(value) as total, AVG(value) as average
            FROM test
            GROUP BY category
            ORDER BY category
        """)

        assert len(df) == 2
        assert df.loc[df["category"] == "A", "total"].values[0] == 60
        assert df.loc[df["category"] == "B", "average"].values[0] == 150.0

        DatabaseManager.reset()


class TestCloseMethod:
    """Tests for the close() method."""

    def test_close_closes_connection(self, temp_db):
        """Test that close() closes the thread-local connection."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        # Force connection creation
        dm.execute("SELECT 1")

        # Close should not raise
        dm.close()

        # Should be able to get a new connection
        result = dm.fetchone("SELECT 2")
        assert result[0] == 2

        DatabaseManager.reset()

    def test_close_idempotent(self, temp_db):
        """Test that calling close() multiple times is safe."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("SELECT 1")

        # Multiple closes should not raise
        dm.close()
        dm.close()
        dm.close()

        DatabaseManager.reset()


class TestThreadSafety:
    """Tests for thread safety of DatabaseManager."""

    def test_concurrent_reads(self, temp_db):
        """Test concurrent read operations from multiple threads."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        # Setup test data
        dm.execute("CREATE TABLE test (id INTEGER, value INTEGER)")
        for i in range(100):
            dm.execute("INSERT INTO test VALUES (?, ?)", (i, i * 10))

        errors = []
        results = []

        def read_data(thread_id):
            try:
                result = dm.fetchall("SELECT * FROM test WHERE id >= ? AND id < ?", (thread_id * 10, (thread_id + 1) * 10))
                results.append((thread_id, len(result)))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(10):
            t = threading.Thread(target=read_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(count == 10 for _, count in results)

        DatabaseManager.reset()

    def test_concurrent_writes(self, temp_db):
        """Test concurrent write operations from multiple threads."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (thread_id INTEGER, value INTEGER)")

        errors = []

        def write_data(thread_id):
            try:
                for i in range(10):
                    dm.execute("INSERT INTO test VALUES (?, ?)", (thread_id, i))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all data was written
        total = dm.fetchone("SELECT COUNT(*) FROM test")
        assert total[0] == 50  # 5 threads * 10 rows each

        DatabaseManager.reset()

    def test_thread_pool_executor(self, temp_db):
        """Test using DatabaseManager with ThreadPoolExecutor."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

        def insert_row(row_id):
            dm.execute("INSERT INTO test VALUES (?)", (row_id,))
            return dm.fetchone("SELECT * FROM test WHERE id = ?", (row_id,))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(insert_row, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 20

        # Verify all rows exist
        count = dm.fetchone("SELECT COUNT(*) FROM test")
        assert count[0] == 20

        DatabaseManager.reset()


class TestGetDB:
    """Tests for the get_db() factory function."""

    def test_get_db_returns_database_manager(self, temp_db):
        """Test that get_db returns a DatabaseManager instance."""
        from hrp.data.db import DatabaseManager, get_db

        DatabaseManager.reset()

        db = get_db(temp_db)

        assert isinstance(db, DatabaseManager)

        DatabaseManager.reset()

    def test_get_db_singleton(self, temp_db):
        """Test that get_db returns the same singleton instance."""
        from hrp.data.db import DatabaseManager, get_db

        DatabaseManager.reset()

        db1 = get_db(temp_db)
        db2 = get_db()

        assert db1 is db2

        DatabaseManager.reset()

    def test_get_db_with_explicit_path(self, temp_db):
        """Test get_db with explicit database path."""
        from hrp.data.db import DatabaseManager, get_db

        DatabaseManager.reset()

        db = get_db(temp_db)
        db.execute("CREATE TABLE test (id INTEGER)")
        db.execute("INSERT INTO test VALUES (42)")

        result = db.fetchone("SELECT * FROM test")
        assert result[0] == 42

        DatabaseManager.reset()

    def test_get_db_with_env_var(self):
        """Test get_db using HRP_DATA_DIR environment variable."""
        from hrp.data.db import DatabaseManager, get_db

        DatabaseManager.reset()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HRP_DATA_DIR": tmpdir}):
                db = get_db()
                expected_path = Path(tmpdir) / "hrp.duckdb"
                assert db.db_path == expected_path

        DatabaseManager.reset()


class TestSchemaIntegration:
    """Integration tests with schema module."""

    def test_create_tables(self, temp_db):
        """Test creating all schema tables."""
        from hrp.data.db import DatabaseManager
        from hrp.data.schema import TABLES, create_tables

        DatabaseManager.reset()

        create_tables(temp_db)

        db = DatabaseManager(temp_db)

        # Verify all tables exist
        for table_name in TABLES.keys():
            result = db.fetchone(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                (table_name,),
            )
            assert result[0] == 1, f"Table {table_name} was not created"

        DatabaseManager.reset()

    def test_verify_schema(self, temp_db):
        """Test schema verification."""
        from hrp.data.db import DatabaseManager
        from hrp.data.schema import create_tables, verify_schema

        DatabaseManager.reset()

        # Schema should not exist initially
        assert verify_schema(temp_db) is False

        # Create tables
        DatabaseManager.reset()
        create_tables(temp_db)

        # Now verification should pass
        DatabaseManager.reset()
        assert verify_schema(temp_db) is True

        DatabaseManager.reset()

    def test_get_table_counts(self, temp_db):
        """Test getting table row counts."""
        from hrp.data.db import DatabaseManager
        from hrp.data.schema import create_tables, get_table_counts

        DatabaseManager.reset()
        create_tables(temp_db)

        # Insert some test data
        DatabaseManager.reset()
        db = DatabaseManager(temp_db)
        db.execute("INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction) VALUES ('HYP-001', 'Test', 'Test thesis', 'Test prediction')")
        db.execute("INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction) VALUES ('HYP-002', 'Test 2', 'Test thesis 2', 'Test prediction 2')")

        DatabaseManager.reset()
        counts = get_table_counts(temp_db)

        assert counts["hypotheses"] == 2
        assert counts["prices"] == 0  # Empty table

        DatabaseManager.reset()

    def test_drop_all_tables(self, temp_db):
        """Test dropping all tables."""
        from hrp.data.db import DatabaseManager
        from hrp.data.schema import create_tables, drop_all_tables, verify_schema

        DatabaseManager.reset()
        create_tables(temp_db)

        DatabaseManager.reset()
        assert verify_schema(temp_db) is True

        DatabaseManager.reset()
        drop_all_tables(temp_db)

        DatabaseManager.reset()
        assert verify_schema(temp_db) is False

        DatabaseManager.reset()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_query_with_special_characters(self, temp_db):
        """Test queries with special characters in data."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, text VARCHAR)")
        dm.execute("INSERT INTO test VALUES (?, ?)", (1, "Hello 'World'"))
        dm.execute("INSERT INTO test VALUES (?, ?)", (2, 'Say "Hello"'))
        dm.execute("INSERT INTO test VALUES (?, ?)", (3, "Line1\nLine2"))

        result = dm.fetchone("SELECT text FROM test WHERE id = 1")
        assert result[0] == "Hello 'World'"

        result = dm.fetchone("SELECT text FROM test WHERE id = 2")
        assert result[0] == 'Say "Hello"'

        result = dm.fetchone("SELECT text FROM test WHERE id = 3")
        assert result[0] == "Line1\nLine2"

        DatabaseManager.reset()

    def test_null_values(self, temp_db):
        """Test handling of NULL values."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, optional_value VARCHAR)")
        dm.execute("INSERT INTO test VALUES (1, NULL)")
        dm.execute("INSERT INTO test VALUES (?, ?)", (2, None))

        result = dm.fetchone("SELECT optional_value FROM test WHERE id = 1")
        assert result[0] is None

        result = dm.fetchone("SELECT optional_value FROM test WHERE id = 2")
        assert result[0] is None

        df = dm.fetchdf("SELECT * FROM test")
        assert pd.isna(df["optional_value"].iloc[0])

        DatabaseManager.reset()

    def test_large_decimal_precision(self, temp_db):
        """Test handling of large decimal values."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, value DECIMAL(18,6))")
        dm.execute("INSERT INTO test VALUES (1, 123456789012.123456)")

        result = dm.fetchone("SELECT value FROM test WHERE id = 1")
        assert float(result[0]) == pytest.approx(123456789012.123456, rel=1e-6)

        DatabaseManager.reset()

    def test_empty_string_vs_null(self, temp_db):
        """Test distinction between empty string and NULL."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, value VARCHAR)")
        dm.execute("INSERT INTO test VALUES (1, '')")
        dm.execute("INSERT INTO test VALUES (2, NULL)")

        result1 = dm.fetchone("SELECT value FROM test WHERE id = 1")
        result2 = dm.fetchone("SELECT value FROM test WHERE id = 2")

        assert result1[0] == ""
        assert result2[0] is None

        DatabaseManager.reset()

    def test_unicode_characters(self, temp_db):
        """Test handling of Unicode characters."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, text VARCHAR)")
        dm.execute("INSERT INTO test VALUES (?, ?)", (1, "Hello World"))
        dm.execute("INSERT INTO test VALUES (?, ?)", (2, "Cafe au lait"))
        dm.execute("INSERT INTO test VALUES (?, ?)", (3, "Chinese text here"))

        results = dm.fetchall("SELECT text FROM test ORDER BY id")
        assert results[0][0] == "Hello World"
        assert results[1][0] == "Cafe au lait"
        assert results[2][0] == "Chinese text here"

        DatabaseManager.reset()

    def test_timestamp_handling(self, temp_db):
        """Test handling of TIMESTAMP values."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE test (id INTEGER, created_at TIMESTAMP)")
        dm.execute("INSERT INTO test VALUES (1, '2024-01-15 10:30:45')")

        result = dm.fetchone("SELECT created_at FROM test WHERE id = 1")
        assert isinstance(result[0], datetime)
        assert result[0].year == 2024
        assert result[0].month == 1
        assert result[0].day == 15

        DatabaseManager.reset()


class TestComplexQueries:
    """Tests for complex query scenarios."""

    def test_join_query(self, temp_db):
        """Test JOIN queries."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR)")
        dm.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, amount DECIMAL(10,2))")

        dm.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
        dm.execute("INSERT INTO orders VALUES (1, 1, 100.00), (2, 1, 150.00), (3, 2, 200.00)")

        df = dm.fetchdf("""
            SELECT u.name, SUM(o.amount) as total
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name
            ORDER BY total DESC
        """)

        assert len(df) == 2
        assert df.iloc[0]["name"] == "Alice"
        assert float(df.iloc[0]["total"]) == 250.00

        DatabaseManager.reset()

    def test_subquery(self, temp_db):
        """Test subqueries."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE prices (symbol VARCHAR, date DATE, close DECIMAL(10,2))")
        dm.execute("""
            INSERT INTO prices VALUES
            ('AAPL', '2024-01-01', 100),
            ('AAPL', '2024-01-02', 105),
            ('MSFT', '2024-01-01', 300),
            ('MSFT', '2024-01-02', 295)
        """)

        result = dm.fetchdf("""
            SELECT symbol, close
            FROM prices p
            WHERE date = (SELECT MAX(date) FROM prices)
            ORDER BY symbol
        """)

        assert len(result) == 2
        assert float(result[result["symbol"] == "AAPL"]["close"].values[0]) == 105

        DatabaseManager.reset()

    def test_window_function(self, temp_db):
        """Test window functions."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE prices (symbol VARCHAR, date DATE, close DECIMAL(10,2))")
        dm.execute("""
            INSERT INTO prices VALUES
            ('AAPL', '2024-01-01', 100),
            ('AAPL', '2024-01-02', 105),
            ('AAPL', '2024-01-03', 103)
        """)

        df = dm.fetchdf("""
            SELECT
                symbol,
                date,
                close,
                LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
            FROM prices
            ORDER BY date
        """)

        assert len(df) == 3
        assert pd.isna(df.iloc[0]["prev_close"])  # First row has no previous
        assert float(df.iloc[1]["prev_close"]) == 100
        assert float(df.iloc[2]["prev_close"]) == 105

        DatabaseManager.reset()

    def test_cte_query(self, temp_db):
        """Test Common Table Expressions (CTEs)."""
        from hrp.data.db import DatabaseManager

        DatabaseManager.reset()
        dm = DatabaseManager(temp_db)

        dm.execute("CREATE TABLE sales (id INTEGER, product VARCHAR, amount DECIMAL(10,2))")
        dm.execute("""
            INSERT INTO sales VALUES
            (1, 'A', 100), (2, 'A', 200), (3, 'B', 150), (4, 'B', 250), (5, 'B', 100)
        """)

        df = dm.fetchdf("""
            WITH product_totals AS (
                SELECT product, SUM(amount) as total
                FROM sales
                GROUP BY product
            )
            SELECT product, total
            FROM product_totals
            WHERE total > 300
            ORDER BY total DESC
        """)

        assert len(df) == 1
        assert df.iloc[0]["product"] == "B"
        assert float(df.iloc[0]["total"]) == 500

        DatabaseManager.reset()
