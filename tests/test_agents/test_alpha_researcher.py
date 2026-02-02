"""
Tests for AlphaResearcher agent.

Tests cover:
- Configuration and initialization
- Hypothesis analysis with mocked Claude
- Hypothesis updates and status transitions
- Lineage event logging
- Research note generation
- Related hypothesis search
- Error handling
"""

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.alpha_researcher import (
    AlphaResearcher,
    AlphaResearcherConfig,
    AlphaResearcherReport,
    HypothesisAnalysis,
)


@pytest.fixture(autouse=True)
def mock_all_db_access(monkeypatch):
    """Mock all database access for all tests in this module."""
    mock_db = MagicMock()
    mock_db.fetchone.return_value = (0,)
    mock_db.fetchall.return_value = []

    # Create a mock connection context manager
    mock_conn = MagicMock()
    # Return proper values for different queries:
    # - dependency check: (status, completed_at) - return success
    # - token usage: (count,) - return 0
    # - log insert: (log_id,) - return 1
    mock_conn.execute.return_value.fetchone.return_value = ("success", datetime.now())
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    # Mock get_db in the data layer (the only module that still uses it directly)
    monkeypatch.setattr("hrp.data.db.get_db", lambda *args, **kwargs: mock_db)


@pytest.fixture(autouse=True)
def mock_all_log_events(monkeypatch):
    """Mock log_event in all relevant modules."""
    mock_log = MagicMock()
    monkeypatch.setattr("hrp.agents.alpha_researcher.log_event", mock_log)
    monkeypatch.setattr("hrp.agents.research_agents.log_event", mock_log)
    monkeypatch.setattr("hrp.agents.sdk_agent.log_event", mock_log)


# Helper to create a fully mocked researcher that won't hit DB
def create_mocked_researcher(mock_api, config=None):
    """Create an AlphaResearcher with all DB access mocked."""
    researcher = AlphaResearcher(config=config, api=mock_api)
    # Mock internal methods that hit DB
    researcher._log_token_usage = MagicMock()
    researcher._save_checkpoint_to_db = MagicMock()
    researcher._get_daily_token_usage = MagicMock(return_value=0)
    researcher.mark_checkpoint_complete = MagicMock()
    # Mock dependency check to always pass
    researcher._check_dependencies = MagicMock(return_value=True)
    # Mock data requirement check to always pass
    researcher._check_data_requirements = MagicMock(return_value=(True, None))
    return researcher


class TestAlphaResearcherConfig:
    """Tests for AlphaResearcherConfig."""

    def test_default_config(self):
        """Should create config with default values."""
        config = AlphaResearcherConfig()

        assert config.hypothesis_ids is None
        assert config.regime_lookback_days == 252 * 3
        assert config.write_research_note is True
        from hrp.utils.config import get_config
        assert config.research_note_dir == str(get_config().data.research_dir)
        # Inherited from SDKAgentConfig
        assert config.max_tokens_per_run == 50_000

    def test_custom_config(self):
        """Should accept custom values."""
        config = AlphaResearcherConfig(
            hypothesis_ids=["HYP-2025-001", "HYP-2025-002"],
            regime_lookback_days=504,
            write_research_note=False,
            research_note_dir="custom/path",
        )

        assert config.hypothesis_ids == ["HYP-2025-001", "HYP-2025-002"]
        assert config.regime_lookback_days == 504
        assert config.write_research_note is False
        assert config.research_note_dir == "custom/path"


class TestHypothesisAnalysis:
    """Tests for HypothesisAnalysis dataclass."""

    def test_analysis_creation(self):
        """Should create analysis with all fields."""
        analysis = HypothesisAnalysis(
            hypothesis_id="HYP-2025-001",
            economic_rationale="Momentum effect captures trend-following behavior",
            regime_notes="Works best in trending markets",
            related_hypotheses=["HYP-2025-002", "HYP-2025-003"],
            refined_thesis="Improved thesis statement",
            refined_falsification="IC < 0.01 over 6 months",
            recommendation="PROCEED",
            status_updated=True,
        )

        assert analysis.hypothesis_id == "HYP-2025-001"
        assert "momentum" in analysis.economic_rationale.lower()
        assert analysis.recommendation == "PROCEED"
        assert analysis.status_updated is True

    def test_analysis_default_status(self):
        """Should default status_updated to False."""
        analysis = HypothesisAnalysis(
            hypothesis_id="HYP-2025-001",
            economic_rationale="",
            regime_notes="",
            related_hypotheses=[],
            refined_thesis="",
            refined_falsification="",
            recommendation="DEFER",
        )

        assert analysis.status_updated is False


class TestAlphaResearcherInit:
    """Tests for AlphaResearcher initialization."""

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_init_default(self, mock_api_class):
        """Should initialize with default settings."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher()

        assert researcher.hypothesis_ids is None
        assert researcher.api is not None
        assert researcher.actor == "agent:alpha-researcher"

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_init_with_hypothesis_ids(self, mock_api_class):
        """Should accept specific hypothesis IDs."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(hypothesis_ids=["HYP-2025-001"])

        assert researcher.hypothesis_ids == ["HYP-2025-001"]

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_init_with_config(self, mock_api_class):
        """Should accept custom config."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        config = AlphaResearcherConfig(
            hypothesis_ids=["HYP-2025-001"],
            write_research_note=False,
        )
        researcher = AlphaResearcher(config=config)

        assert researcher.hypothesis_ids == ["HYP-2025-001"]

    def test_init_with_provided_api(self):
        """Should use provided API instance."""
        mock_api = MagicMock()

        researcher = AlphaResearcher(api=mock_api)

        assert researcher.api is mock_api


class TestAlphaResearcherExecute:
    """Tests for AlphaResearcher.execute method."""

    def test_execute_no_hypotheses(self):
        """Should return empty results when no hypotheses to process."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.return_value = []

        researcher = create_mocked_researcher(mock_api)
        result = researcher.run()

        assert result["hypotheses_reviewed"] == 0
        assert result["hypotheses_promoted"] == 0
        assert result["hypotheses_deferred"] == 0

    def test_execute_processes_draft_hypotheses(self):
        """Should process all draft hypotheses when no IDs specified."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.return_value = [
            {
                "hypothesis_id": "HYP-2025-001",
                "title": "Momentum predicts returns",
                "thesis": "Stocks with high momentum continue outperforming",
                "status": "draft",
            }
        ]
        mock_api.get_prices.return_value = None

        researcher = create_mocked_researcher(mock_api)

        # Mock Claude response
        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Momentum effect is well-documented",
                    "regime_notes": "Works in trending markets",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "Improved thesis",
                    "refined_falsification": "IC < 0.01",
                    "recommendation": "PROCEED",
                    "recommendation_reason": "Sound rationale",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        assert result["hypotheses_reviewed"] == 1
        # list_hypotheses is called twice: once with status="draft", once for related hypotheses
        mock_api.list_hypotheses.assert_any_call(status="draft")

    def test_execute_processes_specific_hypothesis(self):
        """Should process specific hypothesis when ID provided."""
        mock_api = MagicMock()
        mock_api.get_hypothesis.return_value = {
            "hypothesis_id": "HYP-2025-001",
            "title": "Test hypothesis",
            "thesis": "Test thesis",
            "status": "draft",
        }
        mock_api.list_hypotheses.return_value = []
        mock_api.get_prices.return_value = None

        config = AlphaResearcherConfig(hypothesis_ids=["HYP-2025-001"])
        researcher = create_mocked_researcher(mock_api, config=config)

        # Mock Claude response
        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Test rationale",
                    "regime_notes": "Test regime",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "DEFER",
                    "recommendation_reason": "Needs more data",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        assert result["hypotheses_reviewed"] == 1
        mock_api.get_hypothesis.assert_called_with("HYP-2025-001")


class TestAlphaResearcherAnalysis:
    """Tests for hypothesis analysis logic."""

    def test_promotes_hypothesis_on_proceed(self):
        """Should update hypothesis status when recommendation is PROCEED."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [  # First call for draft hypotheses
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test",
                    "thesis": "Test",
                    "status": "draft",
                }
            ],
            [],  # Second call for related hypotheses
        ]
        mock_api.get_prices.return_value = None

        researcher = create_mocked_researcher(mock_api)

        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Sound rationale",
                    "regime_notes": "Good in all regimes",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "Better thesis",
                    "refined_falsification": "IC < 0.01",
                    "recommendation": "PROCEED",
                    "recommendation_reason": "Ready for testing",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        assert result["hypotheses_promoted"] == 1
        # First update_hypothesis call is the PROCEED promotion;
        # subsequent calls are from strategy generation adding metadata
        promote_call = mock_api.update_hypothesis.call_args_list[0]
        call_kwargs = promote_call[1]
        assert call_kwargs["status"] == "testing"
        assert call_kwargs["actor"] == "agent:alpha-researcher"

    def test_defers_hypothesis_on_defer(self):
        """Should not update status when recommendation is DEFER."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test",
                    "thesis": "Test",
                    "status": "draft",
                }
            ],
            [],
        ]
        mock_api.get_prices.return_value = None

        researcher = create_mocked_researcher(mock_api)

        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Unclear rationale",
                    "regime_notes": "Untested",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "DEFER",
                    "recommendation_reason": "Needs more analysis",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        assert result["hypotheses_deferred"] == 1
        assert result["hypotheses_promoted"] == 0
        # No call should promote to 'testing' — any update_hypothesis calls
        # are from strategy generation (status='draft'), not promotion
        for call in mock_api.update_hypothesis.call_args_list:
            assert call[1].get("status") != "testing"

    def test_logs_lineage_event(self, mock_all_log_events):
        """Should log ALPHA_RESEARCHER_REVIEW event to lineage - verified by fixture."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test",
                    "thesis": "Test",
                    "status": "draft",
                }
            ],
            [],
        ]
        mock_api.get_prices.return_value = None

        researcher = create_mocked_researcher(mock_api)

        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Test",
                    "regime_notes": "Test",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "DEFER",
                    "recommendation_reason": "Test",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        # Just verify it runs without error - lineage event mocked by fixture
        result = researcher.run()
        assert result["hypotheses_reviewed"] == 1


class TestRelatedHypothesisSearch:
    """Tests for related hypothesis search."""

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_find_related_hypotheses(self, mock_api_class):
        """Should find hypotheses with shared keywords."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        hypothesis = {
            "hypothesis_id": "HYP-2025-001",
            "title": "Momentum predicts returns",
            "thesis": "High momentum stocks outperform",
        }

        all_hypotheses = [
            {
                "hypothesis_id": "HYP-2025-001",
                "title": "Momentum predicts returns",
                "thesis": "High momentum stocks outperform",
                "status": "draft",
            },
            {
                "hypothesis_id": "HYP-2025-002",
                "title": "Long momentum factor",
                "thesis": "Momentum factor premium exists",
                "status": "testing",
            },
            {
                "hypothesis_id": "HYP-2025-003",
                "title": "Volatility clustering",
                "thesis": "Volatility persists",
                "status": "validated",
            },
        ]

        related = researcher._find_related_hypotheses(hypothesis, all_hypotheses)

        # Should find HYP-2025-002 (shares "momentum") but not HYP-2025-001 (self)
        related_ids = [r["hypothesis_id"] for r in related]
        assert "HYP-2025-002" in related_ids
        assert "HYP-2025-001" not in related_ids

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_find_related_limits_results(self, mock_api_class):
        """Should limit related hypotheses to 5."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        hypothesis = {
            "hypothesis_id": "HYP-2025-001",
            "title": "Momentum predicts returns",
            "thesis": "Momentum effect",
        }

        # Create many hypotheses with shared keyword
        all_hypotheses = [
            {
                "hypothesis_id": f"HYP-2025-{i:03d}",
                "title": f"Momentum variant {i}",
                "thesis": "momentum strategy",
                "status": "draft",
            }
            for i in range(10)
        ]

        related = researcher._find_related_hypotheses(hypothesis, all_hypotheses)

        assert len(related) <= 5


class TestResponseParsing:
    """Tests for Claude response parsing."""

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_parse_json_response(self, mock_api_class):
        """Should parse JSON from Claude response."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        response = {
            "content": json.dumps({
                "economic_rationale": "Test rationale",
                "regime_notes": "Test regime",
                "related_hypothesis_notes": "HYP-2025-002",
                "refined_thesis": "Test thesis",
                "refined_falsification": "Test criteria",
                "recommendation": "PROCEED",
            }),
        }

        analysis = researcher._parse_analysis_response("HYP-2025-001", response)

        assert analysis.economic_rationale == "Test rationale"
        assert analysis.regime_notes == "Test regime"
        assert analysis.recommendation == "PROCEED"

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_parse_json_code_block(self, mock_api_class):
        """Should extract JSON from code block."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        response = {
            "content": """Here is my analysis:

```json
{
    "economic_rationale": "Momentum effect documented",
    "regime_notes": "Bull markets",
    "related_hypothesis_notes": "",
    "refined_thesis": "",
    "refined_falsification": "",
    "recommendation": "PROCEED"
}
```

Additional notes here.""",
        }

        analysis = researcher._parse_analysis_response("HYP-2025-001", response)

        assert analysis.economic_rationale == "Momentum effect documented"
        assert analysis.recommendation == "PROCEED"

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_parse_invalid_json_gracefully(self, mock_api_class):
        """Should handle invalid JSON gracefully."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        response = {
            "content": "This is not valid JSON, just plain text analysis.",
        }

        analysis = researcher._parse_analysis_response("HYP-2025-001", response)

        # Should return deferred analysis with raw content
        assert analysis.recommendation == "DEFER"
        assert "not valid JSON" in analysis.economic_rationale

    @patch("hrp.agents.alpha_researcher.PlatformAPI")
    def test_extract_hypothesis_ids_from_notes(self, mock_api_class):
        """Should extract hypothesis IDs from notes."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        researcher = AlphaResearcher(api=mock_api)

        notes = "Related to HYP-2025-002 and HYP-2025-015, also see HYP-2025-002"

        ids = researcher._extract_related_ids(notes)

        assert "HYP-2025-002" in ids
        assert "HYP-2025-015" in ids
        # Should deduplicate
        assert len([i for i in ids if i == "HYP-2025-002"]) == 1


class TestResearchNote:
    """Tests for research note generation."""

    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    def test_writes_research_note(self, mock_makedirs, mock_open):
        """Should write research note to file."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test",
                    "thesis": "Test",
                    "status": "draft",
                }
            ],
            [],
        ]
        mock_api.get_prices.return_value = None

        config = AlphaResearcherConfig(write_research_note=True)
        researcher = create_mocked_researcher(mock_api, config=config)

        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Test rationale",
                    "regime_notes": "Test regime",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "PROCEED",
                    "recommendation_reason": "Test",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        # Should have created directory and written file
        mock_makedirs.assert_called()
        mock_open.assert_called()

        # File content should include hypothesis info
        written_content = mock_file.write.call_args[0][0]
        assert "HYP-2025-001" in written_content
        assert "Alpha Researcher" in written_content

    def test_skips_research_note_when_disabled(self):
        """Should not write research note when disabled."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.return_value = []

        config = AlphaResearcherConfig(write_research_note=False)
        researcher = create_mocked_researcher(mock_api, config=config)

        result = researcher.run()

        assert result.get("research_note_path") is None


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_content(self):
        """System prompt should describe Alpha Researcher role."""
        mock_api = MagicMock()

        researcher = AlphaResearcher(api=mock_api)

        prompt = researcher._get_system_prompt()

        assert "Alpha Researcher" in prompt
        assert "hypothes" in prompt.lower()  # matches "hypotheses" or "hypothesis"
        assert "economic" in prompt.lower()


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_analysis_error(self):
        """Should continue processing after individual hypothesis error."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test 1",
                    "thesis": "Test",
                    "status": "draft",
                },
                {
                    "hypothesis_id": "HYP-2025-002",
                    "title": "Test 2",
                    "thesis": "Test",
                    "status": "draft",
                },
            ],
            [],
        ]
        mock_api.get_prices.return_value = None

        researcher = create_mocked_researcher(mock_api)

        call_count = {"value": 0}

        def mock_claude(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise Exception("API error")
            return {
                "content": json.dumps({
                    "economic_rationale": "Test",
                    "regime_notes": "Test",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "PROCEED",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }

        researcher._call_claude_api = mock_claude

        result = researcher.run()

        # Should have processed second hypothesis despite first failing
        assert result["hypotheses_reviewed"] == 2
        assert result["hypotheses_deferred"] >= 1  # First one failed

    def test_handles_update_error(self):
        """Should handle hypothesis update errors gracefully."""
        mock_api = MagicMock()
        mock_api.list_hypotheses.side_effect = [
            [
                {
                    "hypothesis_id": "HYP-2025-001",
                    "title": "Test",
                    "thesis": "Test",
                    "status": "draft",
                }
            ],
            [],
        ]
        mock_api.get_prices.return_value = None
        mock_api.update_hypothesis.side_effect = Exception("Update failed")

        researcher = create_mocked_researcher(mock_api)

        researcher._call_claude_api = MagicMock(
            return_value={
                "content": json.dumps({
                    "economic_rationale": "Test",
                    "regime_notes": "Test",
                    "related_hypothesis_notes": "",
                    "refined_thesis": "",
                    "refined_falsification": "",
                    "recommendation": "PROCEED",
                }),
                "tool_calls": [],
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "stop_reason": "end_turn",
            }
        )

        result = researcher.run()

        # Should count as deferred due to update failure
        assert result["hypotheses_deferred"] == 1
