"""
HRP Agent module.

Research agents for automated hypothesis discovery and signal analysis.
"""

from hrp.agents.alpha_researcher import (
    AlphaResearcher,
    AlphaResearcherConfig,
    AlphaResearcherReport,
    HypothesisAnalysis,
    StrategySpec,
)
from hrp.agents.cio import CIOAgent, CIODecision, CIOReport, CIOScore
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
    ParameterVariation,
    PortfolioRiskAssessment,
    QualitySentinelReport,
    QuantDeveloper,
    QuantDeveloperReport,
    ResearchAgent,
    RiskManager,
    RiskManagerReport,
    RiskVeto,
    SignalScanReport,
    SignalScanResult,
    SignalScientist,
    ValidationAnalyst,
    ValidationAnalystReport,
    ValidationCheck,
    ValidationSeverity,
)
from hrp.agents.pipeline_orchestrator import (
    BaselineResult,
    BaselineType,
    ExperimentConfig,
    ExperimentResult,
    KillGateReason,
    OrchestratorResult,
    PipelineOrchestrator,
    PipelineOrchestratorConfig,
    PipelineOrchestratorReport,
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
    # Risk Manager
    "RiskManager",
    "RiskVeto",
    "PortfolioRiskAssessment",
    "RiskManagerReport",
    # Quant Developer
    "QuantDeveloper",
    "QuantDeveloperReport",
    "ParameterVariation",
    # SDK Agents
    "SDKAgent",
    "SDKAgentConfig",
    "TokenUsage",
    "AgentCheckpoint",
    "AlphaResearcher",
    "AlphaResearcherConfig",
    "AlphaResearcherReport",
    "HypothesisAnalysis",
    # CIO Agent
    "CIOAgent",
    "CIOScore",
    "CIODecision",
    "CIOReport",
    # Scheduler
    "IngestionScheduler",
    "LineageEventWatcher",
    "LineageTrigger",
    # Pipeline Orchestrator
    "PipelineOrchestrator",
    "PipelineOrchestratorConfig",
    "PipelineOrchestratorReport",
    "OrchestratorResult",
    "BaselineResult",
    "BaselineType",
    "ExperimentConfig",
    "ExperimentResult",
    "KillGateReason",
]
