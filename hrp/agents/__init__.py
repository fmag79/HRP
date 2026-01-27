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
from hrp.agents.report_generator import ReportGenerator, ReportGeneratorConfig
from hrp.agents.research_agents import (
    AuditCheck,
    AuditSeverity,
    ExperimentAudit,
    HypothesisValidation,
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
    ValidationAnalyst,
    ValidationAnalystReport,
    ValidationCheck,
    ValidationSeverity,
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
    # Report Generator
    "ReportGenerator",
    "ReportGeneratorConfig",
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
    # Validation Analyst
    "ValidationAnalyst",
    "ValidationCheck",
    "ValidationSeverity",
    "HypothesisValidation",
    "ValidationAnalystReport",
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
