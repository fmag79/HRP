"""
Base class for research agents.

Research agents extend the IngestionJob pattern to provide automated
research capabilities with actor tracking and lineage logging.
"""

from abc import ABC, abstractmethod
from typing import Any

from hrp.agents.jobs import IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.research.lineage import EventType, log_event


class ResearchAgent(IngestionJob, ABC):
    """
    Base class for research agents (extends IngestionJob pattern).

    Research agents perform automated analysis and can create draft
    hypotheses. They track their actor identity for lineage purposes
    and have access to the PlatformAPI.
    """

    def __init__(
        self,
        job_id: str,
        actor: str,
        dependencies: list[str] | None = None,
        data_requirements: list | None = None,
        max_retries: int = 2,
    ):
        """
        Initialize a research agent.

        Args:
            job_id: Unique identifier for this job
            actor: Actor identity for lineage (e.g., 'agent:signal-scientist')
            dependencies: List of job IDs that must complete before this job runs
                         (legacy - prefer data_requirements)
            data_requirements: List of DataRequirement objects specifying what data
                              must exist before this agent can run
            max_retries: Maximum number of retry attempts
        """
        super().__init__(
            job_id=job_id,
            dependencies=dependencies or [],
            data_requirements=data_requirements or [],
            max_retries=max_retries,
        )
        self.actor = actor
        self.api = PlatformAPI()

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """
        Implement research logic.

        Must be implemented by subclasses.

        Returns:
            Dictionary with execution results
        """
        pass

    def _log_agent_event(
        self,
        event_type: str | EventType,
        details: dict,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
    ) -> int:
        """
        Log event to lineage with agent actor.

        Args:
            event_type: Type of event (EventType enum or string)
            details: Event-specific details
            hypothesis_id: Optional associated hypothesis
            experiment_id: Optional associated experiment

        Returns:
            lineage_id of the created event
        """
        # Convert EventType to string if needed
        if isinstance(event_type, EventType):
            event_type = event_type.value

        return log_event(
            event_type=event_type,
            actor=self.actor,
            details=details,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )
