"""Job result dataclasses for type-safe job execution results."""

from dataclasses import dataclass, field


@dataclass
class JobResult:
    """Standardized result from job execution.

    Supports both attribute access (result.status) and dictionary-style access
    (result["status"]) for backward compatibility.

    Attributes:
        status: Job execution status ("success", "failed")
        records_fetched: Number of records fetched/computed
        records_inserted: Number of records inserted into database
        symbols_success: Count of successful symbols
        symbols_failed: Count of failed symbols
        failed_symbols: List of failed symbol names
        error: Error message if status is "failed"
    """

    status: str  # "success", "failed"
    records_fetched: int = 0
    records_inserted: int = 0
    symbols_success: int = 0
    symbols_failed: int = 0
    failed_symbols: list[str] = field(default_factory=list)
    error: str | None = None

    def __getitem__(self, key: str):
        """Support dictionary-style access for backward compatibility."""
        return getattr(self, key)

    def get(self, key: str, default=None):
        """Support dictionary-style get() for backward compatibility."""
        return getattr(self, key, default)


@dataclass
class PriceIngestionResult(JobResult):
    """Price ingestion specific result.

    Attributes:
        fallback_used: Number of symbols retrieved using fallback source
    """

    fallback_used: int = 0


@dataclass
class UniverseUpdateResult(JobResult):
    """Universe update specific result.

    Attributes:
        symbols_added: Number of symbols added to universe
        symbols_removed: Number of symbols removed from universe
        symbols_excluded: Number of symbols excluded by filters
        exclusion_breakdown: Breakdown of exclusions by reason
    """

    symbols_added: int = 0
    symbols_removed: int = 0
    symbols_excluded: int = 0
    exclusion_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class FundamentalsIngestionResult(JobResult):
    """Fundamentals ingestion specific result.

    Attributes:
        fallback_used: Number of symbols retrieved using fallback source
        pit_violations_filtered: Number of records filtered due to PIT violations
    """

    fallback_used: int = 0
    pit_violations_filtered: int = 0
