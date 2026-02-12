"""
Tests for ResearchAgent base class.

Tests cover:
- Base class inheritance from IngestionJob
- Actor tracking
- Lineage event logging
- API access
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
# =============================================================================


@pytest.fixture
def research_test_db():
    """Create a temporary database with schema for research agent tests."""
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
            VALUES ('test_research_agent', 'research_agent', 'active')
            ON CONFLICT DO NOTHING
            """
        )

    yield db_path

    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# ResearchAgent Base Class Tests
# =============================================================================


class TestResearchAgentBase:
    """Tests for ResearchAgent base class."""

    def test_inherits_from_ingestion_job(self, research_test_db):
        """ResearchAgent should inherit from IngestionJob."""
        from hrp.agents.jobs import IngestionJob
        from hrp.agents.research_agents import ResearchAgent

        assert issubclass(ResearchAgent, IngestionJob)

    def test_is_abstract_class(self, research_test_db):
        """ResearchAgent should be abstract (cannot be instantiated directly)."""
        from hrp.agents.research_agents import ResearchAgent

        with pytest.raises(TypeError):
            # Should fail because execute() is abstract
            ResearchAgent(job_id="test", actor="test:actor")

    def test_concrete_subclass_can_be_instantiated(self, research_test_db):
        """Concrete subclass of ResearchAgent should be instantiable."""
        from hrp.agents.research_agents import ResearchAgent

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {"status": "success"}

        agent = ConcreteAgent(job_id="test_research_agent", actor="test:agent")
        assert agent is not None
        assert agent.actor == "test:agent"

    def test_actor_is_tracked(self, research_test_db):
        """ResearchAgent should track actor identity."""
        from hrp.agents.research_agents import ResearchAgent

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {}

        agent = ConcreteAgent(job_id="test_research_agent", actor="agent:discovery")
        assert agent.actor == "agent:discovery"

    def test_has_platform_api_access(self, research_test_db):
        """ResearchAgent should have access to PlatformAPI."""
        from hrp.agents.research_agents import ResearchAgent
        from hrp.api.platform import PlatformAPI

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {}

        agent = ConcreteAgent(job_id="test_research_agent", actor="agent:test")
        assert hasattr(agent, "api")
        assert isinstance(agent.api, PlatformAPI)

    def test_log_agent_event_logs_to_lineage(self, research_test_db):
        """_log_agent_event should log event to lineage table."""
        from hrp.agents.research_agents import ResearchAgent
        from hrp.research.lineage import EventType

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {}

        agent = ConcreteAgent(job_id="test_research_agent", actor="agent:test")

        with patch("hrp.agents.base.log_event") as mock_log:
            mock_log.return_value = 1

            lineage_id = agent._log_agent_event(
                event_type=EventType.AGENT_RUN_COMPLETE,
                details={"test": "data"},
                hypothesis_id="HYP-2025-001",
            )

            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["actor"] == "agent:test"
            assert call_kwargs["event_type"] == EventType.AGENT_RUN_COMPLETE
            assert call_kwargs["hypothesis_id"] == "HYP-2025-001"

    def test_can_set_dependencies(self, research_test_db):
        """ResearchAgent should support job dependencies."""
        from hrp.agents.research_agents import ResearchAgent

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {}

        agent = ConcreteAgent(
            job_id="test_research_agent",
            actor="agent:test",
            dependencies=["feature_computation", "price_ingestion"],
        )

        assert "feature_computation" in agent.dependencies
        assert "price_ingestion" in agent.dependencies

    def test_run_method_inherited(self, research_test_db):
        """ResearchAgent should inherit run() method from IngestionJob."""
        from hrp.agents.research_agents import ResearchAgent

        class ConcreteAgent(ResearchAgent):
            def execute(self):
                return {"records_fetched": 0, "records_inserted": 0}

        agent = ConcreteAgent(job_id="test_research_agent", actor="agent:test")

        # Patch notification to avoid side effects
        with patch("hrp.agents.jobs.EmailNotifier"):
            result = agent.run()

        assert "records_fetched" in result or "status" in result


# =============================================================================
# SignalScanResult Dataclass Tests
# =============================================================================


class TestSignalScanResult:
    """Tests for SignalScanResult dataclass."""

    def test_signal_scan_result_fields(self):
        """SignalScanResult should have all required fields."""
        from hrp.agents.research_agents import SignalScanResult

        result = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.045,
            ic_std=0.08,
            ic_ir=0.56,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert result.feature_name == "momentum_20d"
        assert result.forward_horizon == 20
        assert result.ic == 0.045
        assert result.ic_std == 0.08
        assert result.ic_ir == 0.56
        assert result.sample_size == 500
        assert result.start_date == date(2023, 1, 1)
        assert result.end_date == date(2024, 1, 1)
        assert result.is_combination is False
        assert result.combination_method is None

    def test_signal_scan_result_combination(self):
        """SignalScanResult should support combination flags."""
        from hrp.agents.research_agents import SignalScanResult

        result = SignalScanResult(
            feature_name="momentum_20d + volatility_60d",
            forward_horizon=20,
            ic=0.05,
            ic_std=0.07,
            ic_ir=0.71,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            is_combination=True,
            combination_method="additive",
        )

        assert result.is_combination is True
        assert result.combination_method == "additive"


# =============================================================================
# SignalScanReport Dataclass Tests
# =============================================================================


class TestSignalScanReport:
    """Tests for SignalScanReport dataclass."""

    def test_signal_scan_report_fields(self):
        """SignalScanReport should have all required fields."""
        from hrp.agents.research_agents import SignalScanReport, SignalScanResult

        results = [
            SignalScanResult(
                feature_name="momentum_20d",
                forward_horizon=20,
                ic=0.04,
                ic_std=0.08,
                ic_ir=0.5,
                sample_size=500,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
            )
        ]

        report = SignalScanReport(
            scan_date=date(2024, 6, 1),
            total_features_scanned=45,
            promising_signals=results,
            hypotheses_created=["HYP-2025-001"],
            mlflow_run_id="abc123",
            duration_seconds=127.5,
        )

        assert report.scan_date == date(2024, 6, 1)
        assert report.total_features_scanned == 45
        assert len(report.promising_signals) == 1
        assert report.hypotheses_created == ["HYP-2025-001"]
        assert report.mlflow_run_id == "abc123"
        assert report.duration_seconds == 127.5
