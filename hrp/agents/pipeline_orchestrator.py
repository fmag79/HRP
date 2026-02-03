"""
Pipeline Orchestrator Agent - Coordinates parallel experiment execution.

Orchestrates multiple experiments with intelligent resource management and
early stopping to save compute and focus resources on promising hypotheses.

Performs:
1. Runs baseline experiments first (sequential)
2. Queues parallel experiments (hyperparameters, feature subsets)
3. Applies early kill gates to save compute
4. Logs all artifacts to MLflow
5. Generates kill gate reports

Type: Orchestration & Coordination (deterministic workflow)

Trigger: Event-driven (QUANT_DEVELOPER_COMPLETE) or Scheduled (daily)
"""

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from enum import Enum

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from hrp.agents.jobs import IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.ml.regime_detection import StructuralRegimeClassifier, StructuralRegime
from hrp.research.lineage import EventType, log_event


class BaselineType(str, Enum):
    """Types of baseline experiments."""

    EQUAL_WEIGHT_LONG_SHORT = "equal_weight_long_short"
    BUY_AND_HOLD_SPY = "buy_and_hold_spy"
    MARKET_CAP_WEIGHTED = "market_cap_weighted"


class KillGateReason(str, Enum):
    """Reasons for kill gate triggering."""

    BASELINE_SHARPE_TOO_LOW = "baseline_sharpe_too_low"
    TRAIN_SHARPE_TOO_HIGH = "train_sharpe_too_high"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    FEATURE_COUNT_TOO_HIGH = "feature_count_too_high"
    INSTABILITY_TOO_HIGH = "instability_too_high"


@dataclass
class PipelineOrchestratorConfig:
    """Configuration for Pipeline Orchestrator agent."""

    # Which hypotheses to orchestrate
    hypothesis_ids: list[str] | None = None  # None = all ready for orchestration

    # Baseline settings
    run_baselines_first: bool = True
    baseline_types: list[str] = field(default_factory=lambda: [
        "equal_weight_long_short",
        "buy_and_hold_spy",
    ])

    # Parallel execution
    max_parallel_experiments: int = 4
    experiment_queue_size: int = 20

    # Early kill gates
    enable_early_kill: bool = True
    min_baseline_sharpe: float = 0.5
    max_train_sharpe: float = 3.0
    max_drawdown_threshold: float = 0.30
    max_feature_count: int = 50
    max_instability_score: float = 1.5

    # Resource tracking
    log_resource_usage: bool = True
    kill_gate_report_dir: str = ""  # Set from config in __post_init__

    def __post_init__(self):
        if not self.kill_gate_report_dir:
            from hrp.utils.config import get_config
            self.kill_gate_report_dir = str(get_config().data.research_dir)


@dataclass
class BaselineResult:
    """Result from a baseline experiment."""

    baseline_type: BaselineType
    sharpe: float
    total_return: float
    max_drawdown: float
    volatility: float
    mlflow_run_id: str


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in the queue."""

    experiment_id: str
    config_name: str  # e.g., "lookback_10_top_5pct"
    params: dict[str, Any]
    priority: int = 0  # Higher priority runs first


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    experiment_id: str
    config_name: str
    sharpe: float
    total_return: float
    max_drawdown: float
    volatility: float
    killed_early: bool = False
    kill_reason: KillGateReason | None = None
    mlflow_run_id: str = ""


@dataclass
class OrchestratorResult:
    """Result from orchestrating a single hypothesis."""

    hypothesis_id: str
    baselines: dict[str, BaselineResult]
    experiments: list[ExperimentResult]
    killed_by_gate: bool = False
    gate_trigger_reason: KillGateReason | None = None
    time_saved_seconds: float = 0.0
    total_duration_seconds: float = 0.0


@dataclass
class PipelineOrchestratorReport:
    """Complete Pipeline Orchestrator run report."""

    report_date: date
    hypotheses_processed: int
    hypotheses_killed: int
    baselines_run: int
    experiments_run: int
    experiments_killed: int
    time_saved_seconds: float
    kill_gate_report_path: str | None


class PipelineOrchestrator(IngestionJob):
    """
    Orchestrates parallel experiment execution with early kill gates.

    This agent coordinates the research pipeline by:
    1. Running simple baselines first to establish performance floor
    2. Building experiment queue from validated hypotheses
    3. Running experiments in parallel with resource management
    4. Applying early kill gates to save compute on bad ideas
    5. Logging all artifacts to MLflow for traceability

    The orchestrator does NOT:
    - Create strategies (Alpha Researcher)
    - Implement backtests (Quant Developer)
    - Train ML models (ML Scientist)
    - Judge performance (Validation Analyst, CIO)

    Its role is purely coordination and resource optimization.
    """

    DEFAULT_JOB_ID = "pipeline_orchestrator"
    ACTOR = "agent:pipeline-orchestrator"

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        config: PipelineOrchestratorConfig | None = None,
        api: PlatformAPI | None = None,
    ):
        """
        Initialize the Pipeline Orchestrator.

        Args:
            hypothesis_ids: Specific hypotheses to orchestrate (None = all ready)
            config: Orchestrator configuration
            api: PlatformAPI instance (created if not provided)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            dependencies=[],  # Triggered by lineage events
        )

        self.hypothesis_ids = hypothesis_ids
        self.config = config or PipelineOrchestratorConfig()
        self.api = api or PlatformAPI()
        self._results: list[OrchestratorResult] = []

    def execute(self) -> dict[str, Any]:
        """
        Execute the pipeline orchestration.

        Returns:
            Dict with orchestration results
        """
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_process()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "report": {
                    "hypotheses_processed": 0,
                    "hypotheses_killed": 0,
                    "baselines_run": 0,
                    "experiments_run": 0,
                    "experiments_killed": 0,
                    "time_saved_seconds": 0.0,
                },
            }

        logger.info(f"Orchestrating {len(hypotheses)} hypotheses")

        # 2. Process each hypothesis
        hypotheses_killed = 0
        baselines_run = 0
        experiments_run = 0
        experiments_killed = 0
        total_time_saved = 0.0

        for hypothesis in hypotheses:
            try:
                result = self._orchestrate_hypothesis(hypothesis)
                self._results.append(result)

                baselines_run += len(result.baselines)
                experiments_run += len(result.experiments)
                experiments_killed += sum(1 for e in result.experiments if e.killed_early)

                if result.killed_by_gate:
                    hypotheses_killed += 1

                total_time_saved += result.time_saved_seconds

            except Exception as e:
                logger.error(f"Failed to orchestrate {hypothesis.get('hypothesis_id')}: {e}")

        # 3. Generate report
        duration = time.time() - start_time
        report = PipelineOrchestratorReport(
            report_date=date.today(),
            hypotheses_processed=len(hypotheses),
            hypotheses_killed=hypotheses_killed,
            baselines_run=baselines_run,
            experiments_run=experiments_run,
            experiments_killed=experiments_killed,
            time_saved_seconds=total_time_saved,
            kill_gate_report_path=self._write_kill_gate_report() if self.config.enable_early_kill else None,
        )

        # 4. Log completion event
        log_event(
            event_type=EventType.PIPELINE_ORCHESTRATOR_COMPLETE.value,
            actor=self.ACTOR,
            details={
                "hypotheses_processed": report.hypotheses_processed,
                "hypotheses_killed": report.hypotheses_killed,
                "baselines_run": report.baselines_run,
                "experiments_run": report.experiments_run,
                "experiments_killed": report.experiments_killed,
                "time_saved_seconds": report.time_saved_seconds,
            },
        )

        return {
            "status": "complete",
            "report": {
                "hypotheses_processed": report.hypotheses_processed,
                "hypotheses_killed": report.hypotheses_killed,
                "baselines_run": report.baselines_run,
                "experiments_run": report.experiments_run,
                "experiments_killed": report.experiments_killed,
                "time_saved_seconds": report.time_saved_seconds,
                "duration_seconds": duration,
            },
        }

    def _get_hypotheses_to_process(self) -> list[dict[str, Any]]:
        """
        Get hypotheses ready for orchestration.

        Fetches hypotheses that have completed Quant Developer backtesting
        but have not yet been orchestrated.

        Returns:
            List of hypothesis dicts
        """
        if self.hypothesis_ids:
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis_with_metadata(hid)
                if hyp:
                    hypotheses.append(hyp)
        else:
            # Hypotheses with Quant Developer backtests but no orchestration
            hypotheses = []
            for status in ('backtested',):
                hypotheses.extend(self.api.list_hypotheses_with_metadata(
                    status=status,
                    metadata_filter='%quant_developer_backtest%',
                    metadata_exclude='%pipeline_orchestrator%',
                    limit=10,
                ))

        return hypotheses

    def _orchestrate_hypothesis(self, hypothesis: dict[str, Any]) -> OrchestratorResult:
        """
        Orchestrate experiments for a single hypothesis.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            OrchestratorResult with all experiment results
        """
        start_time = time.time()
        hypothesis_id = hypothesis["hypothesis_id"]

        logger.info(f"Orchestrating hypothesis {hypothesis_id}")

        # Step 1: Run baselines (sequential)
        baselines: dict[str, BaselineResult] = {}
        if self.config.run_baselines_first:
            baselines = self._run_baselines(hypothesis_id, hypothesis)

            # Check baseline kill gate
            if self.config.enable_early_kill and not self._check_baseline_kill_gate(baselines):
                logger.info(f"Hypothesis {hypothesis_id} killed by baseline gate")
                log_event(
                    event_type=EventType.KILL_GATE_TRIGGERED.value,
                    actor=self.ACTOR,
                    hypothesis_id=hypothesis_id,
                    details={
                        "reason": KillGateReason.BASELINE_SHARPE_TOO_LOW.value,
                        "baseline_sharpe": max(b.sharpe for b in baselines.values()),
                        "min_required": self.config.min_baseline_sharpe,
                    },
                )

                return OrchestratorResult(
                    hypothesis_id=hypothesis_id,
                    baselines=baselines,
                    experiments=[],
                    killed_by_gate=True,
                    gate_trigger_reason=KillGateReason.BASELINE_SHARPE_TOO_LOW,
                    total_duration_seconds=time.time() - start_time,
                )

        # Step 2: Build experiment queue
        queue = self._build_experiment_queue(hypothesis)

        if not queue:
            logger.info(f"No experiments to run for {hypothesis_id}")

        # Step 3: Run experiments in parallel
        experiments = self._run_parallel_experiments(hypothesis_id, queue, hypothesis)

        duration = time.time() - start_time

        return OrchestratorResult(
            hypothesis_id=hypothesis_id,
            baselines=baselines,
            experiments=experiments,
            killed_by_gate=False,
            total_duration_seconds=duration,
        )

    def _run_baselines(
        self, hypothesis_id: str, hypothesis: dict[str, Any]
    ) -> dict[str, BaselineResult]:
        """
        Run baseline experiments (sequential).

        Args:
            hypothesis_id: Hypothesis identifier
            hypothesis: Hypothesis dict with metadata

        Returns:
            Dict of baseline results
        """
        baselines: dict[str, BaselineResult] = {}

        # Extract backtest results from metadata
        metadata = hypothesis.get("metadata", {})
        backtest_results = metadata.get("quant_developer_backtest", {})

        # If we have existing backtest results, use those as baseline
        if backtest_results and "baseline" in backtest_results:
            base = backtest_results["baseline"]
            baselines["baseline"] = BaselineResult(
                baseline_type=BaselineType.EQUAL_WEIGHT_LONG_SHORT,
                sharpe=base.get("sharpe", 0.0),
                total_return=base.get("total_return", 0.0),
                max_drawdown=base.get("max_drawdown", 0.0),
                volatility=base.get("volatility", 0.0),
                mlflow_run_id=backtest_results.get("mlflow_run_id", ""),
            )

        return baselines

    def _check_baseline_kill_gate(self, baselines: dict[str, BaselineResult]) -> bool:
        """
        Check if baseline meets minimum requirements.

        Args:
            baselines: Dict of baseline results

        Returns:
            True if baseline passes, False if should be killed
        """
        if not baselines:
            return True  # No baselines to check, proceed

        max_sharpe = max(b.sharpe for b in baselines.values())
        return max_sharpe >= self.config.min_baseline_sharpe

    def _build_experiment_queue(self, hypothesis: dict[str, Any]) -> list[ExperimentConfig]:
        """
        Build experiment queue from hypothesis metadata.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            List of ExperimentConfig objects
        """
        queue: list[ExperimentConfig] = []
        metadata = hypothesis.get("metadata", {})

        # Extract parameter variations from Quant Developer results
        backtest_results = metadata.get("quant_developer_backtest", {})
        parameter_variations = backtest_results.get("parameter_variations", [])

        for i, variation in enumerate(parameter_variations):
            config = ExperimentConfig(
                experiment_id=f"{hypothesis['hypothesis_id']}_var_{i}",
                config_name=variation.get("variation_name", f"variation_{i}"),
                params=variation.get("params", {}),
                priority=variation.get("priority", 0),
            )
            queue.append(config)

        # Sort by priority (highest first)
        queue.sort(key=lambda x: x.priority, reverse=True)

        return queue[: self.config.experiment_queue_size]

    def _generate_structural_regime_scenarios(
        self,
        prices: pd.DataFrame,
        min_days: int = 60,
    ) -> dict[StructuralRegime, list[tuple]]:
        """
        Generate structural regime scenarios using HMM-based detection.

        Classifies market periods into 4 structural regimes:
        - low_vol_bull: Low volatility, positive returns
        - low_vol_bear: Low volatility, negative returns
        - high_vol_bull: High volatility, positive returns
        - high_vol_bear: High volatility, negative returns

        Args:
            prices: DataFrame with 'close' column and DatetimeIndex
            min_days: Minimum days for a period to be included

        Returns:
            Dict mapping regime to list of (start_date, end_date) tuples
        """
        classifier = StructuralRegimeClassifier()

        # Fit classifier to price data
        try:
            classifier.fit(prices)
        except Exception as e:
            logger.warning(f"Failed to fit structural regime classifier: {e}")
            # Return empty dict if fitting fails
            return {
                "low_vol_bull": [],
                "low_vol_bear": [],
                "high_vol_bull": [],
                "high_vol_bear": [],
            }

        # Get scenario periods
        periods = classifier.get_scenario_periods(prices, min_days=min_days)

        logger.info(
            f"Generated structural regime scenarios: "
            f"{sum(len(p) for p in periods.values())} periods"
        )

        return periods

    def _run_parallel_experiments(
        self,
        hypothesis_id: str,
        queue: list[ExperimentConfig],
        hypothesis: dict[str, Any],
    ) -> list[ExperimentResult]:
        """
        Run experiments in parallel with kill gates.

        Args:
            hypothesis_id: Hypothesis identifier
            queue: List of experiment configs
            hypothesis: Hypothesis dict with metadata

        Returns:
            List of ExperimentResult objects
        """
        results: list[ExperimentResult] = []

        # For now, run sequentially (could be parallelized later)
        for config in queue:
            try:
                result = self._run_single_experiment(hypothesis_id, config, hypothesis)
                results.append(result)

                # Apply kill gates
                if self.config.enable_early_kill:
                    if self._apply_kill_gates(result):
                        logger.info(f"Experiment {config.experiment_id} killed by gate")
                        results[-1].killed_early = True

            except Exception as e:
                logger.error(f"Failed to run experiment {config.experiment_id}: {e}")

        return results

    def _run_single_experiment(
        self,
        hypothesis_id: str,
        config: ExperimentConfig,
        hypothesis: dict[str, Any],
    ) -> ExperimentResult:
        """
        Run a single experiment.

        In production, this would launch an actual MLflow experiment.
        For now, we simulate with metadata from Quant Developer.

        Args:
            hypothesis_id: Hypothesis identifier
            config: Experiment configuration
            hypothesis: Hypothesis dict with metadata

        Returns:
            ExperimentResult
        """
        # Extract results from metadata (from Quant Developer)
        metadata = hypothesis.get("metadata", {})
        backtest_results = metadata.get("quant_developer_backtest", {})
        parameter_variations = backtest_results.get("parameter_variations", [])

        # Find matching variation
        variation = next(
            (v for v in parameter_variations if v.get("variation_name") == config.config_name),
            None,
        )

        if variation:
            return ExperimentResult(
                experiment_id=config.experiment_id,
                config_name=config.config_name,
                sharpe=variation.get("sharpe", 0.0),
                total_return=variation.get("total_return", 0.0),
                max_drawdown=variation.get("max_drawdown", 0.0),
                volatility=variation.get("volatility", 0.0),
                killed_early=False,
                mlflow_run_id=backtest_results.get("mlflow_run_id", ""),
            )
        else:
            # Default result if not found
            return ExperimentResult(
                experiment_id=config.experiment_id,
                config_name=config.config_name,
                sharpe=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                volatility=0.0,
            )

    def _apply_kill_gates(self, result: ExperimentResult) -> bool:
        """
        Apply kill gates to experiment result.

        Args:
            result: Experiment result to check

        Returns:
            True if should be killed, False otherwise
        """
        # Kill if train Sharpe is too high (overfitting)
        if result.sharpe > self.config.max_train_sharpe:
            result.kill_reason = KillGateReason.TRAIN_SHARPE_TOO_HIGH
            return True

        # Kill if max drawdown exceeded
        if result.max_drawdown > self.config.max_drawdown_threshold:
            result.kill_reason = KillGateReason.MAX_DRAWDOWN_EXCEEDED
            return True

        return False

    def _write_kill_gate_report(self) -> str | None:
        """
        Write kill gate report to docs/research/.

        Returns:
            Filepath if successful, None otherwise
        """
        if not self._results:
            return None

        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges, render_risk_limits,
            render_section_divider, render_progress_bar, format_metric,
        )

        from hrp.agents.output_paths import research_note_path

        today = date.today().isoformat()
        total = len(self._results)
        killed_count = sum(1 for r in self._results if r.killed_by_gate)
        passed_count = total - killed_count
        baselines_run = sum(len(r.baselines) for r in self._results)
        experiments_run = sum(len(r.experiments) for r in self._results)
        experiments_killed = sum(sum(1 for e in r.experiments if e.killed_early) for r in self._results)
        time_saved = sum(r.time_saved_seconds for r in self._results)

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Pipeline Kill Gate Report",
            report_type="kill-gates",
            date_str=today,
            subtitle=f"⚔️ {total} hypotheses processed | {killed_count} killed | {format_metric(time_saved, 'int')}s saved",
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Processed", "value": total, "detail": "hypotheses"},
            {"icon": "⚔️", "label": "Killed", "value": killed_count, "detail": "by gates"},
            {"icon": "🧪", "label": "Experiments", "value": experiments_run, "detail": f"{experiments_killed} killed"},
            {"icon": "⏱️", "label": "Time Saved", "value": f"{time_saved:.0f}s", "detail": "compute"},
        ]))

        # ── Alert banner ──
        if killed_count > 0:
            kill_pct = killed_count / max(total, 1) * 100
            parts.append(render_alert_banner(
                [f"{killed_count} of {total} hypotheses killed at gates ({kill_pct:.0f}%)",
                 f"⏱️  Estimated {time_saved:.0f}s of compute time saved"],
                severity="warning" if kill_pct < 50 else "critical",
            ))

        # ── Health Gauges ──
        survival_rate = (passed_count / max(total, 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Gate Survival Rate", "value": survival_rate, "max_val": 100,
             "trend": "up" if survival_rate > 50 else "down"},
            {"label": "Experiment Efficiency", "value": (experiments_run - experiments_killed), "max_val": max(experiments_run, 1),
             "trend": "stable"},
        ]))

        # ── Kill Gate Settings ──
        parts.append(render_risk_limits({
            "Min Baseline Sharpe": str(self.config.min_baseline_sharpe),
            "Max Train Sharpe": str(self.config.max_train_sharpe),
            "Max Drawdown": f"{self.config.max_drawdown_threshold:.1%}",
            "Max Feature Count": str(self.config.max_feature_count),
        }))

        # ── Hypothesis Details ──
        parts.append(render_section_divider("📊 Hypothesis Details"))

        for result in self._results:
            if result.killed_by_gate:
                status_emoji = "🔴"
                status_label = "KILLED"
            else:
                status_emoji = "✅"
                status_label = "PASSED"

            parts.append(f"### {status_emoji} {result.hypothesis_id} — **{status_label}**")
            parts.append("")

            # Gate trigger info
            gate_reason = result.gate_trigger_reason.value if result.gate_trigger_reason else "N/A"
            parts.append(f"| Field | Detail |")
            parts.append(f"|-------|--------|")
            parts.append(f"| **Gate Trigger** | {gate_reason} |")
            parts.append(f"| **Duration** | {result.total_duration_seconds:.1f}s |")
            parts.append(f"| **Experiments** | {len(result.experiments)} run |")
            parts.append("")

            # Baselines
            if result.baselines:
                parts.append("**Baselines:**")
                parts.append("```")
                for name, baseline in result.baselines.items():
                    bar = render_progress_bar(max(baseline.sharpe, 0), 2.0, width=10, show_pct=False)
                    parts.append(f"  {name.ljust(15)} Sharpe={baseline.sharpe:+.2f}  Return={baseline.total_return:+.2%}  {bar}")
                parts.append("```")
                parts.append("")

            # Experiments summary
            if result.experiments:
                exp_killed = sum(1 for e in result.experiments if e.killed_early)
                best_exp = max(result.experiments, key=lambda e: e.sharpe)
                parts.append(f"**Experiments:** {exp_killed}/{len(result.experiments)} killed early")
                parts.append(f"**Best:** {best_exp.config_name} (Sharpe={best_exp.sharpe:.2f})")
                parts.append("")

            parts.append("─" * 60)
            parts.append("")

        # ── Footer ──
        parts.append(render_footer(agent_name="pipeline-orchestrator"))

        content = "\n".join(parts)

        # Write to file
        filepath = str(research_note_path("06-kill-gates"))

        try:
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Kill gate report written to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write kill gate report: {e}")
            return None
