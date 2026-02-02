"""
Tests for SDKAgent base class.

Tests cover:
- Configuration and initialization
- Token usage tracking
- Claude API invocation (mocked)
- Checkpointing and resume
- Budget enforcement
- Run lifecycle
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hrp.agents.sdk_agent import (
    AgentCheckpoint,
    SDKAgent,
    SDKAgentConfig,
    TokenUsage,
)


class TestSDKAgentConfig:
    """Tests for SDKAgentConfig dataclass."""

    def test_default_config(self):
        """Should create config with default values."""
        config = SDKAgentConfig()

        assert config.max_tokens_per_run == 50_000
        assert config.max_output_tokens == 10_000
        assert config.model == "claude-sonnet-4-20250514"
        assert config.daily_budget_tokens is None
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.checkpoint_enabled is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = SDKAgentConfig(
            max_tokens_per_run=100_000,
            max_output_tokens=20_000,
            model="claude-3-opus-20240229",
            daily_budget_tokens=500_000,
            max_retries=5,
            retry_delay_seconds=2.0,
            checkpoint_enabled=False,
        )

        assert config.max_tokens_per_run == 100_000
        assert config.max_output_tokens == 20_000
        assert config.model == "claude-3-opus-20240229"
        assert config.daily_budget_tokens == 500_000
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.checkpoint_enabled is False


class TestTokenUsage:
    """Tests for TokenUsage tracking."""

    def test_default_values(self):
        """Should start with zero usage."""
        usage = TokenUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost_usd == 0.0

    def test_add_tokens(self):
        """Should accumulate token usage."""
        usage = TokenUsage()

        usage.add(100, 50)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_add_multiple_times(self):
        """Should accumulate across multiple calls."""
        usage = TokenUsage()

        usage.add(100, 50)
        usage.add(200, 100)
        usage.add(50, 25)

        assert usage.input_tokens == 350
        assert usage.output_tokens == 175
        assert usage.total_tokens == 525

    def test_cost_estimation(self):
        """Should estimate cost based on Sonnet pricing."""
        usage = TokenUsage()

        # 1000 input tokens at $3/MTok = $0.003
        # 1000 output tokens at $15/MTok = $0.015
        usage.add(1000, 1000)

        # (1000 * 0.003 + 1000 * 0.015) / 1000 = 0.018
        assert usage.estimated_cost_usd == pytest.approx(0.018, rel=0.01)


class TestAgentCheckpoint:
    """Tests for AgentCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Should create checkpoint with all fields."""
        now = datetime.now()
        usage = TokenUsage()
        usage.add(100, 50)

        checkpoint = AgentCheckpoint(
            agent_type="TestAgent",
            run_id="abc123",
            created_at=now,
            state={"step": 5, "data": [1, 2, 3]},
            token_usage=usage,
            completed=False,
        )

        assert checkpoint.agent_type == "TestAgent"
        assert checkpoint.run_id == "abc123"
        assert checkpoint.created_at == now
        assert checkpoint.state == {"step": 5, "data": [1, 2, 3]}
        assert checkpoint.token_usage == usage
        assert checkpoint.completed is False

    def test_checkpoint_default_completed(self):
        """Should default completed to False."""
        checkpoint = AgentCheckpoint(
            agent_type="TestAgent",
            run_id="abc123",
            created_at=datetime.now(),
            state={},
            token_usage=TokenUsage(),
        )

        assert checkpoint.completed is False


class ConcreteSDKAgent(SDKAgent):
    """Concrete implementation for testing."""

    def execute(self):
        """Test implementation."""
        return {"status": "success", "result": "test"}


class TestSDKAgentInit:
    """Tests for SDKAgent initialization."""

    def test_init_default_config(self):
        """Should use default config when not provided."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        assert agent.config is not None
        assert agent.config.max_tokens_per_run == 50_000
        assert agent.job_id == "test-job"
        assert agent.actor == "agent:test"

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = SDKAgentConfig(max_tokens_per_run=10_000)

        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            config=config,
        )

        assert agent.config.max_tokens_per_run == 10_000

    def test_init_with_dependencies(self):
        """Should accept dependencies list."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            dependencies=["prices", "features"],
        )

        assert agent.dependencies == ["prices", "features"]

    def test_token_usage_property(self):
        """Should expose token usage via property."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        assert agent.token_usage is not None
        assert agent.token_usage.total_tokens == 0


class TestSDKAgentTrackCost:
    """Tests for SDKAgent.track_cost method."""

    def test_track_cost_updates_usage(self):
        """Should update internal token usage."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.track_cost(100, 50)

        assert agent.token_usage.input_tokens == 100
        assert agent.token_usage.output_tokens == 50
        assert agent.token_usage.total_tokens == 150

    @patch("hrp.agents.sdk_agent.PlatformAPI")
    def test_track_cost_logs_to_db(self, mock_platform_api_cls):
        """Should log usage to database via PlatformAPI."""
        mock_api = MagicMock()
        mock_platform_api_cls.return_value = mock_api

        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )
        agent._run_id = "test-run-123"

        agent.track_cost(100, 50)

        mock_api.log_token_usage.assert_called_once()

    def test_track_cost_handles_db_error(self):
        """Should handle database errors gracefully."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Mock the api to raise on log_token_usage
        agent.api = MagicMock()
        agent.api.log_token_usage.side_effect = Exception("DB error")

        # Should not raise
        agent.track_cost(100, 50)

        # Usage should still be tracked
        assert agent.token_usage.total_tokens == 150


class TestSDKAgentInvokeClaude:
    """Tests for SDKAgent.invoke_claude method."""

    def test_invoke_claude_basic(self):
        """Should call Claude API with configured settings."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Mock daily token usage check
        agent.api = MagicMock()
        agent.api.get_daily_token_usage.return_value = 0

        # Mock the internal API call method
        agent._call_claude_api = MagicMock(
            return_value={
                "content": "Test response",
                "tool_calls": [],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "stop_reason": "end_turn",
            }
        )

        result = agent.invoke_claude("Test prompt")

        assert result["content"] == "Test response"
        assert agent.token_usage.total_tokens == 150

    def test_invoke_claude_with_tools(self):
        """Should pass tools to Claude API."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.api = MagicMock()
        agent.api.get_daily_token_usage.return_value = 0

        tools = [{"name": "test_tool", "description": "A test tool"}]

        agent._call_claude_api = MagicMock(
            return_value={
                "content": "",
                "tool_calls": [{"id": "call1", "name": "test_tool", "input": {}}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "stop_reason": "tool_use",
            }
        )

        result = agent.invoke_claude("Test prompt", tools=tools)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "test_tool"

    def test_invoke_claude_respects_per_run_limit(self):
        """Should raise when per-run token limit exceeded."""
        config = SDKAgentConfig(max_tokens_per_run=100)
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            config=config,
        )

        # Manually set usage above limit
        agent._token_usage.add(150, 0)

        with pytest.raises(RuntimeError, match="Per-run token limit exceeded"):
            agent.invoke_claude("Test prompt")

    def test_invoke_claude_respects_daily_budget(self):
        """Should raise when daily budget exceeded."""
        config = SDKAgentConfig(daily_budget_tokens=50_000)
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            config=config,
        )

        # Mock daily token usage to return above budget
        agent.api = MagicMock()
        agent.api.get_daily_token_usage.return_value = 100_000

        with pytest.raises(RuntimeError, match="Daily token budget exceeded"):
            agent.invoke_claude("Test prompt")

    @patch("time.sleep")
    def test_invoke_claude_retries_on_error(self, mock_sleep):
        """Should retry on API errors."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.api = MagicMock()
        agent.api.get_daily_token_usage.return_value = 0

        # Fail twice, succeed on third try
        call_count = {"value": 0}

        def mock_api(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise Exception("API error")
            return {
                "content": "Success",
                "tool_calls": [],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "stop_reason": "end_turn",
            }

        agent._call_claude_api = mock_api

        result = agent.invoke_claude("Test prompt")

        assert result["content"] == "Success"
        assert call_count["value"] == 3

    @patch("time.sleep")
    def test_invoke_claude_raises_after_max_retries(self, mock_sleep):
        """Should raise after all retries exhausted."""
        config = SDKAgentConfig(max_retries=3)
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            config=config,
        )

        agent.api = MagicMock()
        agent.api.get_daily_token_usage.return_value = 0

        agent._call_claude_api = MagicMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            agent.invoke_claude("Test prompt")


class TestSDKAgentCheckpoint:
    """Tests for SDKAgent checkpointing."""

    @patch("hrp.agents.sdk_agent.log_event")
    def test_checkpoint_saves_state(self, mock_log_event):
        """Should save state to checkpoint."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )
        agent._run_id = "test-run-123"

        state = {"step": 5, "processed_ids": [1, 2, 3]}
        agent.checkpoint(state)

        assert agent._checkpoint is not None
        assert agent._checkpoint.state == state
        assert agent._checkpoint.run_id == "test-run-123"
        assert agent._checkpoint.completed is False

    @patch("hrp.agents.sdk_agent.log_event")
    def test_checkpoint_logs_lineage_event(self, mock_log_event):
        """Should log checkpoint to lineage."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )
        agent._run_id = "test-run-123"

        agent.checkpoint({"step": 5})

        mock_log_event.assert_called_once()
        call_kwargs = mock_log_event.call_args[1]
        assert call_kwargs["actor"] == "agent:test"
        assert call_kwargs["details"]["checkpoint"] is True

    @patch("hrp.agents.sdk_agent.log_event")
    @patch("hrp.agents.sdk_agent.PlatformAPI")
    def test_checkpoint_saves_to_db(self, mock_platform_api_cls, mock_log_event):
        """Should save checkpoint to database via PlatformAPI."""
        mock_api = MagicMock()
        mock_platform_api_cls.return_value = mock_api

        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )
        agent._run_id = "test-run-123"

        agent.checkpoint({"step": 5})

        # Verify PlatformAPI save was called
        mock_api.save_agent_checkpoint.assert_called_once()

    def test_checkpoint_disabled(self):
        """Should skip checkpointing when disabled."""
        config = SDKAgentConfig(checkpoint_enabled=False)
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
            config=config,
        )

        agent.checkpoint({"step": 5})

        assert agent._checkpoint is None


class TestSDKAgentResumeCheckpoint:
    """Tests for SDKAgent checkpoint resume."""

    def test_resume_from_checkpoint(self):
        """Should restore state from checkpoint."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Mock api.resume_agent_checkpoint to return checkpoint data
        agent.api = MagicMock()
        agent.api.resume_agent_checkpoint.return_value = {
            "state_json": json.dumps({"step": 5, "data": [1, 2, 3]}),
            "input_tokens": 100,
            "output_tokens": 50,
        }

        state = agent.resume_from_checkpoint()

        assert state == {"step": 5, "data": [1, 2, 3]}
        assert agent.token_usage.input_tokens == 100
        assert agent.token_usage.output_tokens == 50

    def test_resume_from_specific_run(self):
        """Should resume from specific run_id."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.api = MagicMock()
        agent.api.resume_agent_checkpoint.return_value = {
            "state_json": json.dumps({"step": 10}),
            "input_tokens": 200,
            "output_tokens": 100,
        }

        state = agent.resume_from_checkpoint(run_id="specific-run")

        assert state == {"step": 10}
        # Verify run_id was passed
        agent.api.resume_agent_checkpoint.assert_called_once_with(
            "ConcreteSDKAgent", "specific-run"
        )

    def test_resume_no_checkpoint(self):
        """Should return None when no checkpoint found."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.api = MagicMock()
        agent.api.resume_agent_checkpoint.return_value = None

        state = agent.resume_from_checkpoint()

        assert state is None

    def test_resume_handles_db_error(self):
        """Should handle database errors gracefully."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.api = MagicMock()
        agent.api.resume_agent_checkpoint.side_effect = Exception("DB error")

        state = agent.resume_from_checkpoint()

        assert state is None


class TestSDKAgentMarkComplete:
    """Tests for SDKAgent.mark_checkpoint_complete."""

    @patch("hrp.agents.sdk_agent.log_event")
    @patch("hrp.agents.sdk_agent.PlatformAPI")
    def test_mark_checkpoint_complete(self, mock_platform_api_cls, mock_log_event):
        """Should mark checkpoint as complete via PlatformAPI."""
        mock_api = MagicMock()
        mock_platform_api_cls.return_value = mock_api

        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )
        agent._run_id = "test-run-123"

        # First create a checkpoint
        agent.checkpoint({"step": 5})

        # Then mark it complete
        agent.mark_checkpoint_complete()

        assert agent._checkpoint.completed is True

        # Verify PlatformAPI was called
        mock_api.complete_agent_checkpoint.assert_called_once()

    def test_mark_complete_no_checkpoint(self):
        """Should handle no checkpoint gracefully."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Should not raise
        agent.mark_checkpoint_complete()


class TestSDKAgentRun:
    """Tests for SDKAgent.run method."""

    def test_run_creates_run_id(self):
        """Should create unique run_id."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        agent.run()

        assert agent._run_id is not None
        assert len(agent._run_id) == 8  # UUID[:8]

    def test_run_resets_token_usage(self):
        """Should reset token usage for new run."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Pre-existing usage
        agent._token_usage.add(100, 50)

        agent.run()

        assert agent.token_usage.total_tokens == 0

    def test_run_returns_execute_result(self):
        """Should return result from execute method."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        result = agent.run()

        assert result["status"] == "success"
        assert result["result"] == "test"

    @patch("hrp.agents.sdk_agent.log_event")
    @patch("hrp.agents.sdk_agent.PlatformAPI")
    def test_run_marks_checkpoint_complete(self, mock_platform_api_cls, mock_log_event):
        """Should mark checkpoint complete on success."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        # Create checkpoint during execute
        original_execute = agent.execute

        def execute_with_checkpoint():
            agent.checkpoint({"step": 1})
            return original_execute()

        agent.execute = execute_with_checkpoint

        agent.run()

        assert agent._checkpoint.completed is True


class TestSDKAgentSystemPrompt:
    """Tests for SDKAgent._get_system_prompt."""

    def test_default_system_prompt(self):
        """Should return default system prompt with agent name."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        prompt = agent._get_system_prompt()

        assert "ConcreteSDKAgent" in prompt
        assert "research agent" in prompt.lower()


class TestSDKAgentAvailableTools:
    """Tests for SDKAgent._get_available_tools."""

    def test_default_no_tools(self):
        """Should return empty list by default."""
        agent = ConcreteSDKAgent(
            job_id="test-job",
            actor="agent:test",
        )

        tools = agent._get_available_tools()

        assert tools == []
