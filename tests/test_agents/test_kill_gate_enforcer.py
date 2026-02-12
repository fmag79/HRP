"""
Tests for Kill Gate Enforcer agent.

Tests cover:
- Baseline execution
- Experiment queue building
- Early kill gates
- Structural regime scenarios
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def enforcer_test_db():
    """Create a temporary database with schema for Kill Gate Enforcer tests."""
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
            VALUES ('test_orchestrator', 'kill_gate_enforcer', 'active')
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


class TestStructuralRegimeScenarios:
    """Tests for structural regime scenario generation."""

    def test_generate_structural_regime_scenarios_returns_periods(
        self, enforcer_test_db
    ):
        """Generate structural regime scenarios returns regime periods."""
        from hrp.agents.kill_gate_enforcer import KillGateEnforcer
        from hrp.ml.regime_detection import StructuralRegime

        enforcer = KillGateEnforcer()

        # Create synthetic price data
        import numpy as np

        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range("2020-01-01", periods=n_samples)
        prices = pd.DataFrame(
            {"close": 100 + np.cumsum(np.random.normal(0.1, 1, n_samples))},
            index=dates,
        )

        scenarios = enforcer._generate_structural_regime_scenarios(prices)

        # Should return dict with 4 regime types
        assert isinstance(scenarios, dict)
        for regime in [
            "low_vol_bull",
            "low_vol_bear",
            "high_vol_bull",
            "high_vol_bear",
        ]:
            assert regime in scenarios
            assert isinstance(scenarios[regime], list)

    def test_generate_structural_regime_scenarios_min_days_filter(
        self, enforcer_test_db
    ):
        """Generate structural regime scenarios filters by min_days."""
        from hrp.agents.kill_gate_enforcer import KillGateEnforcer

        enforcer = KillGateEnforcer()

        # Create synthetic price data
        import numpy as np

        np.random.seed(42)
        n_samples = 200  # Short period
        dates = pd.date_range("2020-01-01", periods=n_samples)
        prices = pd.DataFrame(
            {"close": 100 + np.cumsum(np.random.normal(0.1, 1, n_samples))},
            index=dates,
        )

        # With high min_days, should return fewer or no periods
        scenarios = enforcer._generate_structural_regime_scenarios(
            prices, min_days=100
        )

        # All periods should be >= 100 days
        for regime, periods in scenarios.items():
            for start, end in periods:
                days = (end - start).days
                assert days >= 100

    def test_generate_structural_regime_requires_fit_first(
        self, enforcer_test_db
    ):
        """StructuralRegimeClassifier requires fit before predict."""
        from hrp.agents.kill_gate_enforcer import KillGateEnforcer

        enforcer = KillGateEnforcer()

        # Create simple price data
        prices = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        # Should not raise error - method handles fitting internally
        scenarios = enforcer._generate_structural_regime_scenarios(prices)

        # May return empty dict for too-short data
        assert isinstance(scenarios, dict)
