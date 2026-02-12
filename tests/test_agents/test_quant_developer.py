"""
Tests for Quant Developer agent.

Tests cover:
- Backtest generation
- Parameter variations
- Pre-Backtest Review
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def quant_test_db():
    """Create a temporary database with schema for Quant Developer tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    from hrp.data.db import get_db

    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES ('test_quant_developer', 'quant_developer', 'active')
            ON CONFLICT DO NOTHING
            """
        )

    yield db_path

    # Cleanup
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]


# =============================================================================
# Tests
# ==============================================================================


class TestPreBacktestReview:
    """Tests for Pre-Backtest Review functionality."""

    def test_pre_backtest_review_warnings(self, quant_test_db):
        """Pre-backtest review generates warnings."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()

        # Create a test hypothesis first
        from hrp.api.platform import PlatformAPI
        api = PlatformAPI()
        hypothesis_id = api.create_hypothesis(
            title="Test Hypothesis",
            thesis="Test thesis",
            prediction="Test prediction",
            falsification="Test falsification",
            actor="agent:test",
        )

        result = developer._pre_backtest_review(hypothesis_id)

        assert result["passed"] is True  # Always passes (warnings only)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["data_issues"], list)
        assert "reviewed_at" in result

    def test_pre_backtest_review_nonexistent_hypothesis(self, quant_test_db):
        """Pre-backtest review handles nonexistent hypothesis."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        result = developer._pre_backtest_review("NONEXISTENT")

        assert result["passed"] is False
        assert "Hypothesis not found" in result["warnings"]

    def test_check_data_availability_empty_inputs(self, quant_test_db):
        """Check data availability with empty inputs generates warnings."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        warnings = developer._check_data_availability(
            symbols=[],
            features=[],
            start_date="",
        )

        # Should warn about missing inputs
        assert len(warnings) >= 3
        assert any("No universe symbols" in w for w in warnings)
        assert any("No features" in w for w in warnings)

    def test_check_execution_frequency_intraday_warning(self, quant_test_db):
        """Intraday rebalancing generates warning."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        notes = developer._check_execution_frequency(
            {"rebalance_cadence": "intraday"}
        )

        assert "intraday" in notes[0].lower()
        assert "not supported" in notes[0]

    def test_check_execution_frequency_daily_note(self, quant_test_db):
        """Daily rebalancing generates note about costs."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        notes = developer._check_execution_frequency(
            {"rebalance_cadence": "daily"}
        )

        assert any("daily" in note.lower() and "cost" in note.lower() for note in notes)

    def test_check_universe_liquidity_large_universe(self, quant_test_db):
        """Large universe generates liquidity warning."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        warnings = developer._check_universe_liquidity(
            list(range(600))  # 600 symbols
        )

        assert len(warnings) > 0
        assert any("large universe" in w.lower() for w in warnings)

    def test_check_point_in_time_validity_long_lookback(self, quant_test_db):
        """Long lookback period generates warning."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        warnings = developer._check_point_in_time_validity(
            {"lookback_days": 1000}  # > 756 days
        )

        assert len(warnings) > 0
        assert "lookback" in warnings[0].lower()

    def test_check_cost_model_applicability_daily_rebalance(self, quant_test_db):
        """Daily rebalancing generates cost warning."""
        from hrp.agents.research_agents import QuantDeveloper

        developer = QuantDeveloper()
        warnings = developer._check_cost_model_applicability(
            {"rebalance_cadence": "daily"}
        )

        assert len(warnings) > 0
        assert "cost" in warnings[0].lower()
