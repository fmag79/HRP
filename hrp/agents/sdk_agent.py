"""
SDK Agent base class for Claude-powered research agents.

Provides infrastructure for agents that use Claude API for reasoning,
including token tracking, checkpointing, and cost management.
"""

import json
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.agents.research_agents import ResearchAgent
from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType, log_event


@dataclass
class SDKAgentConfig:
    """Configuration for SDK-powered agents."""

    # Token limits
    max_tokens_per_run: int = 50_000
    max_output_tokens: int = 10_000

    # Model configuration
    model: str = "claude-sonnet-4-20250514"

    # Budget controls
    daily_budget_tokens: int | None = None

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Checkpoint settings
    checkpoint_enabled: bool = True


@dataclass
class TokenUsage:
    """Track token usage for a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        # Estimate cost (Sonnet pricing as of 2024)
        self.estimated_cost_usd += (input_tokens * 0.003 + output_tokens * 0.015) / 1000


@dataclass
class AgentCheckpoint:
    """Checkpoint for resumable agent execution."""

    agent_type: str
    run_id: str
    created_at: datetime
    state: dict[str, Any]
    token_usage: TokenUsage
    completed: bool = False


class SDKAgent(ResearchAgent):
    """
    Base class for Claude-powered research agents.

    Extends ResearchAgent with Claude API integration, including:
    - Token usage tracking and budget enforcement
    - Checkpointing for resumable execution
    - Cost tracking and logging
    - Structured tool calling

    Subclasses should implement:
    - execute(): Main agent logic
    - _get_system_prompt(): System prompt for Claude
    - _get_available_tools(): Tools available to Claude
    """

    # Pricing per 1K tokens (as of 2024, update as needed)
    INPUT_COST_PER_1K = 0.003  # $3/MTok for Sonnet
    OUTPUT_COST_PER_1K = 0.015  # $15/MTok for Sonnet

    def __init__(
        self,
        job_id: str,
        actor: str,
        config: SDKAgentConfig | None = None,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize the SDK Agent.

        Args:
            job_id: Unique job identifier
            actor: Actor identity (e.g., "agent:alpha-researcher")
            config: SDK agent configuration
            dependencies: List of data requirements
        """
        super().__init__(
            job_id=job_id,
            actor=actor,
            dependencies=dependencies or [],
        )
        self.config = config or SDKAgentConfig()
        self._platform_api = PlatformAPI()
        self._token_usage = TokenUsage()
        self._run_id: str | None = None
        self._checkpoint: AgentCheckpoint | None = None

    @property
    def token_usage(self) -> TokenUsage:
        """Current token usage for this run."""
        return self._token_usage

    def invoke_claude(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Call Claude API with configured settings.

        Args:
            prompt: User message/prompt
            tools: Optional list of tool definitions
            system_prompt: Optional system prompt (uses _get_system_prompt() if not provided)
            max_tokens: Max output tokens (uses config default if not provided)

        Returns:
            Dict with 'content', 'tool_calls', 'usage' keys

        Raises:
            RuntimeError: If token budget exceeded
            Exception: On API errors after retries
        """
        # Check budget
        if self.config.daily_budget_tokens is not None:
            daily_usage = self._get_daily_token_usage()
            if daily_usage >= self.config.daily_budget_tokens:
                raise RuntimeError(
                    f"Daily token budget exceeded: {daily_usage} >= {self.config.daily_budget_tokens}"
                )

        # Check per-run limit
        if self._token_usage.total_tokens >= self.config.max_tokens_per_run:
            raise RuntimeError(
                f"Per-run token limit exceeded: {self._token_usage.total_tokens} >= "
                f"{self.config.max_tokens_per_run}"
            )

        # Build request
        system = system_prompt or self._get_system_prompt()
        max_out = max_tokens or self.config.max_output_tokens

        # Attempt API call with retries
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_claude_api(
                    prompt=prompt,
                    system_prompt=system,
                    tools=tools,
                    max_tokens=max_out,
                )

                # Track usage
                usage = response.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                self.track_cost(input_tokens, output_tokens)

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Claude API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise last_error or RuntimeError("Claude API call failed")

    def _call_claude_api(
        self,
        prompt: str,
        system_prompt: str,
        tools: list[dict] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Make actual API call to Claude.

        This method can be mocked in tests. In production, uses anthropic SDK.

        Returns:
            Response dict with 'content', 'tool_calls', 'usage' keys
        """
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Build messages
            messages = [{"role": "user", "content": prompt}]

            # Make API call
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            }

            if tools:
                kwargs["tools"] = tools

            response = client.messages.create(**kwargs)

            # Parse response
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

        except ImportError:
            # anthropic not installed - return mock response for testing
            logger.warning("anthropic package not installed, returning mock response")
            return {
                "content": "Mock response (anthropic not installed)",
                "tool_calls": [],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "stop_reason": "end_turn",
            }

    def track_cost(self, input_tokens: int, output_tokens: int) -> None:
        """
        Log token usage for cost tracking.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        self._token_usage.add(input_tokens, output_tokens)

        # Log to database for daily tracking
        self._log_token_usage(input_tokens, output_tokens)

        logger.debug(
            f"Token usage: +{input_tokens} in, +{output_tokens} out "
            f"(total: {self._token_usage.total_tokens}, "
            f"cost: ${self._token_usage.estimated_cost_usd:.4f})"
        )

    def _log_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Log token usage to database."""
        try:
            self._platform_api.log_token_usage(
                agent_type=self.__class__.__name__,
                run_id=self._run_id or "unknown",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            # Table may not exist yet
            logger.debug(f"Failed to log token usage: {e}")

    def _get_daily_token_usage(self) -> int:
        """Get total tokens used today by this agent type."""
        try:
            return self.api.get_daily_token_usage(self.__class__.__name__)
        except Exception:
            return 0

    def checkpoint(self, state: dict[str, Any]) -> None:
        """
        Save state to lineage for resume capability.

        Args:
            state: Current state to checkpoint
        """
        if not self.config.checkpoint_enabled:
            return

        self._checkpoint = AgentCheckpoint(
            agent_type=self.__class__.__name__,
            run_id=self._run_id or "unknown",
            created_at=datetime.now(),
            state=state,
            token_usage=self._token_usage,
            completed=False,
        )

        # Log to lineage
        log_event(
            event_type=EventType.AGENT_RUN_COMPLETE.value,
            actor=self.actor,
            details={
                "checkpoint": True,
                "agent_type": self._checkpoint.agent_type,
                "run_id": self._checkpoint.run_id,
                "state_keys": list(state.keys()),
                "token_usage": {
                    "input": self._token_usage.input_tokens,
                    "output": self._token_usage.output_tokens,
                    "total": self._token_usage.total_tokens,
                },
            },
        )

        # Also save to checkpoint table
        self._save_checkpoint_to_db()

        logger.info(f"Checkpoint saved for {self.__class__.__name__} (run_id={self._run_id})")

    def _save_checkpoint_to_db(self) -> None:
        """Save checkpoint to database."""
        if self._checkpoint is None:
            return

        try:
            self._platform_api.save_agent_checkpoint(
                agent_type=self._checkpoint.agent_type,
                run_id=self._checkpoint.run_id,
                state_json=json.dumps(self._checkpoint.state),
                input_tokens=self._checkpoint.token_usage.input_tokens,
                output_tokens=self._checkpoint.token_usage.output_tokens,
                completed=self._checkpoint.completed,
            )
        except Exception as e:
            logger.debug(f"Failed to save checkpoint to DB: {e}")

    def resume_from_checkpoint(self, run_id: str | None = None) -> dict[str, Any] | None:
        """
        Resume from last checkpoint if exists.

        Args:
            run_id: Optional specific run_id to resume from

        Returns:
            Checkpoint state dict or None if no checkpoint found
        """
        try:
            checkpoint = self.api.resume_agent_checkpoint(
                self.__class__.__name__, run_id
            )

            if checkpoint is None:
                return None

            state_json = checkpoint["state_json"]
            input_tokens = checkpoint["input_tokens"]
            output_tokens = checkpoint["output_tokens"]
            state = json.loads(state_json)

            # Restore token usage
            self._token_usage.input_tokens = input_tokens
            self._token_usage.output_tokens = output_tokens
            self._token_usage.total_tokens = input_tokens + output_tokens

            logger.info(
                f"Resumed from checkpoint: {len(state)} state keys, "
                f"{self._token_usage.total_tokens} tokens used"
            )

            return state

        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}")
            return None

    def mark_checkpoint_complete(self) -> None:
        """Mark current checkpoint as complete."""
        if self._checkpoint is None or self._run_id is None:
            return

        try:
            self._platform_api.complete_agent_checkpoint(
                agent_type=self.__class__.__name__,
                run_id=self._run_id,
            )
            self._checkpoint.completed = True
        except Exception as e:
            logger.debug(f"Failed to mark checkpoint complete: {e}")

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for Claude.

        Subclasses should override this to provide agent-specific prompts.

        Returns:
            System prompt string
        """
        return (
            f"You are {self.__class__.__name__}, a research agent in the HRP quantitative "
            "research platform. You analyze financial data, hypotheses, and experiments "
            "to support systematic trading research. Be precise, data-driven, and thorough."
        )

    def _get_available_tools(self) -> list[dict]:
        """
        Get tools available to Claude.

        Subclasses should override this to provide agent-specific tools.

        Returns:
            List of tool definitions in Claude tool format
        """
        return []

    def run(self) -> dict[str, Any]:
        """
        Execute the agent with tracking and checkpoint support.

        Wraps execute() with token tracking and checkpoint management.
        """
        import uuid

        self._run_id = str(uuid.uuid4())[:8]
        self._token_usage = TokenUsage()

        logger.info(f"Starting {self.__class__.__name__} run (id={self._run_id})")

        try:
            result = super().run()

            # Mark checkpoint complete on success
            self.mark_checkpoint_complete()

            # Log final token usage
            logger.info(
                f"{self.__class__.__name__} completed: "
                f"{self._token_usage.total_tokens} tokens, "
                f"${self._token_usage.estimated_cost_usd:.4f} estimated cost"
            )

            return result

        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed: {e}")
            raise
