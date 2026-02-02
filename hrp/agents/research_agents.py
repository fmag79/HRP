"""Research agents — split into per-agent modules.

This file re-exports all agents for backwards compatibility.
Prefer importing from individual modules directly.
"""
from hrp.agents.base import ResearchAgent
from hrp.research.lineage import log_event  # noqa: F401 — re-exported for backward compat
from hrp.agents.constants import IC_THRESHOLDS, get_ic_thresholds
from hrp.agents.signal_scientist import SignalScientist, SignalScanResult, SignalScanReport
from hrp.agents.ml_scientist import MLScientist, ModelExperimentResult, MLScientistReport
from hrp.agents.ml_quality_sentinel import (
    MLQualitySentinel, AuditSeverity, AuditCheck, ExperimentAudit,
    MonitoringAlert, QualitySentinelReport,
)
from hrp.agents.validation_analyst import (
    ValidationAnalyst, ValidationCheck, ValidationSeverity,
    HypothesisValidation, ValidationAnalystReport,
)
from hrp.agents.risk_manager import RiskManager, RiskVeto, PortfolioRiskAssessment, RiskManagerReport
from hrp.agents.quant_developer import QuantDeveloper, QuantDeveloperReport, ParameterVariation

__all__ = [
    "ResearchAgent",
    "IC_THRESHOLDS", "get_ic_thresholds",
    "SignalScientist", "SignalScanResult", "SignalScanReport",
    "MLScientist", "ModelExperimentResult", "MLScientistReport",
    "MLQualitySentinel", "AuditSeverity", "AuditCheck", "ExperimentAudit",
    "MonitoringAlert", "QualitySentinelReport",
    "ValidationAnalyst", "ValidationCheck", "ValidationSeverity",
    "HypothesisValidation", "ValidationAnalystReport",
    "RiskManager", "RiskVeto", "PortfolioRiskAssessment", "RiskManagerReport",
    "QuantDeveloper", "QuantDeveloperReport", "ParameterVariation",
]
