"""
HRP Agent module.

Research agents for automated hypothesis discovery and signal analysis.
"""

from hrp.agents.alpha_researcher import (
    AlphaResearcher,
    AlphaResearcherConfig,
    AlphaResearcherReport,
    HypothesisAnalysis,
)
from hrp.agents.jobs import (
    FeatureComputationJob,
    FundamentalsIngestionJob,
    IngestionJob,
    JobStatus,
    PriceIngestionJob,
    SnapshotFundamentalsJob,
    UniverseUpdateJob,
)
from hrp.agents.research_agents import (
    AuditCheck,
    AuditSeverity,
    ExperimentAudit,
    MLQualitySentinel,
    MLScientist,
    MLScientistReport,
    ModelExperimentResult,
    MonitoringAlert,
    QualitySentinelReport,
    ResearchAgent,
    SignalScanReport,
    SignalScanResult,
    SignalScientist,
)
from hrp.agents.scheduler import IngestionScheduler, LineageEventWatcher, LineageTrigger
from hrp.agents.sdk_agent import AgentCheckpoint, SDKAgent, SDKAgentConfig, TokenUsage

__all__ = [
    # Jobs
    "IngestionJob",
    "JobStatus",
    "PriceIngestionJob",
    "FeatureComputationJob",
    "UniverseUpdateJob",
    "FundamentalsIngestionJob",
    "SnapshotFundamentalsJob",
    # Research Agents
    "ResearchAgent",
    "SignalScientist",
    "SignalScanResult",
    "SignalScanReport",
    "MLScientist",
    "MLScientistReport",
    "ModelExperimentResult",
    "MLQualitySentinel",
    "AuditSeverity",
    "AuditCheck",
    "ExperimentAudit",
    "MonitoringAlert",
    "QualitySentinelReport",
    # SDK Agents
    "SDKAgent",
    "SDKAgentConfig",
    "TokenUsage",
    "AgentCheckpoint",
    "AlphaResearcher",
    "AlphaResearcherConfig",
    "AlphaResearcherReport",
    "HypothesisAnalysis",
    # Scheduler
    "IngestionScheduler",
    "LineageEventWatcher",
    "LineageTrigger",
]
