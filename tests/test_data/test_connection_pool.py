"""Test connection pool with retry logic."""

import pytest


class TestConnectionPool:
    """Tests for ConnectionPool class."""

    def test_get_connection_returns_connection(self, tmp_path):
        """get_connection should return a DuckDB connection."""
        from hrp.data.connection_pool import ConnectionPool

        db_path = tmp_path / "test.duckdb"
        pool = ConnectionPool(str(db_path), max_connections=2)

        conn = pool.get_connection()
        assert conn is not None

        # Verify it's a working connection
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1

        pool.release_connection(conn)
        pool.close_all()

    def test_pool_limits_connections(self, tmp_path):
        """Pool should limit number of connections."""
        from hrp.data.connection_pool import ConnectionPool

        db_path = tmp_path / "test.duckdb"
        pool = ConnectionPool(str(db_path), max_connections=2)

        conn1 = pool.get_connection()
        conn2 = pool.get_connection()

        # Third connection should block/timeout
        with pytest.raises(TimeoutError):
            pool.get_connection(timeout=0.1)

        pool.release_connection(conn1)
        pool.release_connection(conn2)
        pool.close_all()

    def test_release_returns_to_pool(self, tmp_path):
        """Released connections should be reused."""
        from hrp.data.connection_pool import ConnectionPool

        db_path = tmp_path / "test.duckdb"
        pool = ConnectionPool(str(db_path), max_connections=1)

        conn1 = pool.get_connection()
        pool.release_connection(conn1)

        conn2 = pool.get_connection()  # Should not block
        assert conn2 is not None

        pool.release_connection(conn2)
        pool.close_all()


class TestRetryLogic:
    """Tests for retry with backoff."""

    def test_retry_on_database_locked(self, tmp_path):
        """Should retry on database locked error."""
        from hrp.data.connection_pool import with_retry

        call_count = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def flaky_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("database is locked")
            return "success"

        result = flaky_operation()
        assert result == "success"
        assert call_count[0] == 3

    def test_raises_after_max_retries(self):
        """Should raise after max retries exhausted."""
        from hrp.data.connection_pool import with_retry

        @with_retry(max_retries=2, base_delay=0.01)
        def always_fails():
            raise Exception("database is locked")

        with pytest.raises(Exception, match="database is locked"):
            always_fails()
