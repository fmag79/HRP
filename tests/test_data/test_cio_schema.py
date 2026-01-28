"""Tests for CIO Agent database schema."""

import pytest
from hrp.data.db import DatabaseManager


class TestCIOSchema:
    """Test CIO tables are created correctly."""

    @pytest.fixture
    def db(self):
        """Get database connection."""
        return DatabaseManager()

    def test_paper_portfolio_table_exists(self, db):
        """Test paper_portfolio table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio'
        """)
        assert result[0][0] == 1

    def test_cio_decisions_table_exists(self, db):
        """Test cio_decisions table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'cio_decisions'
        """)
        assert result[0][0] == 1

    def test_model_cemetery_table_exists(self, db):
        """Test model_cemetery table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'model_cemetery'
        """)
        assert result[0][0] == 1

    def test_paper_portfolio_history_table_exists(self, db):
        """Test paper_portfolio_history table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio_history'
        """)
        assert result[0][0] == 1

    def test_paper_portfolio_trades_table_exists(self, db):
        """Test paper_portfolio_trades table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'paper_portfolio_trades'
        """)
        assert result[0][0] == 1

    def test_cio_threshold_history_table_exists(self, db):
        """Test cio_threshold_history table exists."""
        result = db.fetchall("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'cio_threshold_history'
        """)
        assert result[0][0] == 1
