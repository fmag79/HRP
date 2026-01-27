"""Agent reporting utilities for standardized execution reports."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AgentReport:
    """Standardized agent execution report.

    Provides consistent reporting format for all agent types with
    markdown generation and file persistence.
    """

    agent_name: str
    start_time: datetime
    end_time: datetime
    status: str  # "success", "failed", "running"
    results: dict[str, Any]
    errors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        duration = self.end_time - self.start_time
        duration_seconds = duration.total_seconds()
        duration_str = _format_duration(duration_seconds)

        lines = [
            "# Agent Execution Report",
            "",
            f"**Agent**: {self.agent_name}",
            f"**Status**: {self.status}",
            f"**Start**: {self.start_time.isoformat()}",
            f"**End**: {self.end_time.isoformat()}",
            f"**Duration**: {duration_str}",
            "",
        ]

        # Add results section
        if self.results:
            lines.extend([
                "## Results",
                "",
                _format_dict(self.results),
                "",
            ])

        # Add errors section if any
        if self.errors:
            lines.extend([
                "## Errors",
                "",
            ])
            for i, error in enumerate(self.errors, 1):
                lines.append(f"{i}. {error}")
            lines.append("")

        return "\n".join(lines)

    def save_to_file(self, path: Path) -> None:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _format_dict(d: dict[str, Any], indent: int = 0) -> str:
    """Format dictionary as markdown."""
    lines = []
    prefix = "  " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- **{key}**:")
            lines.append(_format_dict(value, indent + 1))
        else:
            lines.append(f"{prefix}- **{key}**: {value}")

    return "\n".join(lines)
