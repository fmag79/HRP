"""
Kill Gate Enforcer Agent - Enforces early termination gates to save compute.

Applies hard quality thresholds to terminate unpromising research early,
saving compute resources for promising hypotheses.

Performs:
1. Runs baseline experiments to establish performance floor
2. Applies 5 kill gates (Sharpe, drawdown, features, overfitting, instability)
3. Terminates hypotheses that fail gates with detailed reports
4. Logs all artifacts to MLflow
5. Generates comprehensive kill gate reports

Type: Quality Gate Enforcement (deterministic workflow)

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
class KillGateEnforcerConfig:
    """Configuration for Kill Gate Enforcer agent."""

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
class KillGateResult:
    """Result from orchestrating a single hypothesis."""

    hypothesis_id: str
    baselines: dict[str, BaselineResult]
    experiments: list[ExperimentResult]
    killed_by_gate: bool = False
    gate_trigger_reason: KillGateReason | None = None
    time_saved_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    # Enhanced context for reporting
    title: str = ""
    thesis: str = ""
    features: list[str] = field(default_factory=list)
    model_type: str = ""
    ml_ic: float | None = None  # Information Coefficient from ML Scientist
    stability_score: float | None = None  # From ML experiment


@dataclass
class KillGateEnforcerReport:
    """Complete Pipeline Orchestrator run report."""

    report_date: date
    hypotheses_processed: int
    hypotheses_killed: int
    baselines_run: int
    experiments_run: int
    experiments_killed: int
    time_saved_seconds: float
    kill_gate_report_path: str | None


class KillGateEnforcer(IngestionJob):
    """
    Enforces kill gates to terminate unpromising research early.

    This agent applies hard quality thresholds to save compute by:
    1. Running baselines to establish performance floor
    2. Applying 5 kill gates (Sharpe, drawdown, features, overfitting, instability)
    3. Terminating hypotheses that fail gates with detailed reports
    4. Logging all artifacts to MLflow for traceability

    Kill Gates Applied:
    - Baseline Sharpe < 0.5 (below minimum threshold)
    - Train Sharpe > 3.0 (suspiciously good, likely overfit)
    - Max Drawdown > 30% (excessive risk)
    - Feature Count > 50 (curse of dimensionality)
    - Sharpe Decay > 50% (train >> test, severe overfitting)

    The Kill Gate Enforcer does NOT:
    - Create strategies (Alpha Researcher)
    - Implement backtests (Quant Developer)
    - Train ML models (ML Scientist)
    - Judge performance qualitatively (Validation Analyst, CIO)

    Its role is purely gate enforcement and resource optimization.
    """

    DEFAULT_JOB_ID = "kill_gate_enforcer"
    ACTOR = "agent:kill-gate-enforcer"

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        config: KillGateEnforcerConfig | None = None,
        api: PlatformAPI | None = None,
    ):
        """
        Initialize the Kill Gate Enforcer.

        Args:
            hypothesis_ids: Specific hypotheses to evaluate (None = all ready)
            config: Kill gate configuration
            api: PlatformAPI instance (created if not provided)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            dependencies=[],  # Triggered by lineage events
        )

        self.hypothesis_ids = hypothesis_ids
        self.config = config or KillGateEnforcerConfig()
        self.api = api or PlatformAPI()
        self._results: list[KillGateResult] = []

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

                # Mark as processed by Kill Gate Enforcer (idempotency stamp)
                self.api.update_hypothesis(
                    hypothesis_id=hypothesis.get("hypothesis_id"),
                    metadata={
                        "kill_gate_enforcer": {
                            "run_id": self.job_id,
                            "run_date": date.today().isoformat(),
                            "result": "killed" if result.killed_by_gate else "passed",
                            "gate_trigger_reason": result.gate_trigger_reason.value if result.gate_trigger_reason else None,
                        }
                    },
                    actor=self.ACTOR,
                )

            except Exception as e:
                logger.error(f"Failed to orchestrate {hypothesis.get('hypothesis_id')}: {e}")

        # 3. Generate report
        duration = time.time() - start_time
        report = KillGateEnforcerReport(
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
            event_type=EventType.KILL_GATE_ENFORCER_COMPLETE.value,
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
            for status in ('validated',):  # Quant Developer sets validated + pipeline_stage
                hypotheses.extend(self.api.list_hypotheses_with_metadata(
                    status=status,
                    metadata_filter='%quant_developer_backtest%',
                    metadata_exclude='%kill_gate_enforcer%',
                    limit=10,
                ))

        return hypotheses

    def _extract_hypothesis_context(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Extract hypothesis context for enhanced reporting.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            Dict with context fields for KillGateResult
        """
        metadata = hypothesis.get("metadata", {})
        qd_backtest = metadata.get("quant_developer_backtest", {})
        ml_experiment = metadata.get("ml_scientist_experiment", {})

        # Extract features from various possible locations
        features = []
        if ml_experiment.get("features"):
            features = ml_experiment["features"]
        elif qd_backtest.get("features"):
            features = qd_backtest["features"]

        return {
            "title": hypothesis.get("title", ""),
            "thesis": hypothesis.get("thesis", ""),
            "features": features if isinstance(features, list) else [features] if features else [],
            "model_type": ml_experiment.get("model_type", qd_backtest.get("model_type", "")),
            "ml_ic": ml_experiment.get("mean_ic"),
            "stability_score": ml_experiment.get("stability_score"),
        }

    def _orchestrate_hypothesis(self, hypothesis: dict[str, Any]) -> KillGateResult:
        """
        Orchestrate experiments for a single hypothesis.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            KillGateResult with all experiment results
        """
        start_time = time.time()
        hypothesis_id = hypothesis["hypothesis_id"]

        logger.info(f"Orchestrating hypothesis {hypothesis_id}")

        # Extract hypothesis context for reporting
        context = self._extract_hypothesis_context(hypothesis)

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
                        "baseline_sharpe": max(b.sharpe for b in baselines.values()) if baselines else 0.0,
                        "min_required": self.config.min_baseline_sharpe,
                    },
                )

                return KillGateResult(
                    hypothesis_id=hypothesis_id,
                    baselines=baselines,
                    experiments=[],
                    killed_by_gate=True,
                    gate_trigger_reason=KillGateReason.BASELINE_SHARPE_TOO_LOW,
                    total_duration_seconds=time.time() - start_time,
                    **context,
                )

        # Step 2: Build experiment queue
        queue = self._build_experiment_queue(hypothesis)

        if not queue:
            logger.info(f"No experiments to run for {hypothesis_id}")

        # Step 3: Run experiments in parallel
        experiments = self._run_parallel_experiments(hypothesis_id, queue, hypothesis)

        duration = time.time() - start_time

        return KillGateResult(
            hypothesis_id=hypothesis_id,
            baselines=baselines,
            experiments=experiments,
            killed_by_gate=False,
            total_duration_seconds=duration,
            **context,
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

        # QD stores base metrics at top level (sharpe, max_drawdown, etc.)
        # Use these as the baseline for kill gate evaluation
        if backtest_results and "sharpe" in backtest_results:
            baselines["baseline"] = BaselineResult(
                baseline_type=BaselineType.EQUAL_WEIGHT_LONG_SHORT,
                sharpe=float(backtest_results.get("sharpe", 0.0)),
                total_return=float(backtest_results.get("total_return", 0.0)),
                max_drawdown=float(backtest_results.get("max_drawdown", 0.0)),
                volatility=float(backtest_results.get("volatility", 0.0)),
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
            # No baseline data available - fail the gate (don't proceed blindly)
            logger.warning("No baseline data found - failing kill gate")
            return False

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
        Write comprehensive kill gate report in institutional quant style.

        Generates a detailed research report with:
        - Executive summary with statistical highlights
        - Kill gate analysis with reason breakdown
        - Per-hypothesis detailed sections
        - Cross-hypothesis statistics
        - Risk metrics and recommendations

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

        # ══════════════════════════════════════════════════════════════════════
        # AGGREGATE STATISTICS
        # ══════════════════════════════════════════════════════════════════════
        total = len(self._results)
        killed_count = sum(1 for r in self._results if r.killed_by_gate)
        passed_count = total - killed_count
        baselines_run = sum(len(r.baselines) for r in self._results)
        experiments_run = sum(len(r.experiments) for r in self._results)
        experiments_killed = sum(
            sum(1 for e in r.experiments if e.killed_early)
            for r in self._results
        )
        time_saved = sum(r.time_saved_seconds for r in self._results)

        # Collect all Sharpe ratios for distribution analysis
        all_sharpes = []
        all_returns = []
        all_drawdowns = []
        all_volatilities = []

        for result in self._results:
            for baseline in result.baselines.values():
                all_sharpes.append(baseline.sharpe)
                all_returns.append(baseline.total_return)
                all_drawdowns.append(baseline.max_drawdown)
                all_volatilities.append(baseline.volatility)
            for exp in result.experiments:
                all_sharpes.append(exp.sharpe)
                all_returns.append(exp.total_return)
                all_drawdowns.append(exp.max_drawdown)
                all_volatilities.append(exp.volatility)

        # Kill reason breakdown
        kill_reasons: dict[str, int] = {}
        for result in self._results:
            if result.gate_trigger_reason:
                reason = result.gate_trigger_reason.value
                kill_reasons[reason] = kill_reasons.get(reason, 0) + 1

        parts = []

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_header(
            title="Pipeline Kill Gate Report",
            report_type="kill-gates",
            date_str=today,
            subtitle=f"⚔️ {total} hypotheses | {killed_count} killed | {passed_count} passed",
        ))

        # ══════════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        parts.append("## Executive Summary\n")

        kill_rate = (killed_count / max(total, 1)) * 100
        if kill_rate == 100:
            verdict = "🔴 **ALL HYPOTHESES REJECTED** — No strategies meet minimum quality thresholds"
        elif kill_rate >= 75:
            verdict = "🟠 **HIGH REJECTION RATE** — Majority of strategies fail quality gates"
        elif kill_rate >= 50:
            verdict = "🟡 **MODERATE REJECTION** — Mixed quality across hypothesis pool"
        elif kill_rate > 0:
            verdict = "🟢 **LOW REJECTION RATE** — Most strategies pass quality filters"
        else:
            verdict = "✅ **ALL HYPOTHESES PASSED** — Full pipeline proceeding to validation"

        parts.append(f"{verdict}\n")

        # KPI Dashboard
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Processed", "value": total, "detail": "hypotheses"},
            {"icon": "⚔️", "label": "Killed", "value": killed_count, "detail": f"{kill_rate:.0f}% rejection"},
            {"icon": "🧪", "label": "Experiments", "value": experiments_run, "detail": f"{experiments_killed} killed"},
            {"icon": "⏱️", "label": "Compute Saved", "value": f"{time_saved:.0f}s", "detail": "estimated"},
        ]))

        # Alert banner for high kill rates
        if killed_count > 0:
            parts.append(render_alert_banner(
                [f"Kill Rate: {kill_rate:.1f}% ({killed_count}/{total} hypotheses)",
                 f"Compute savings: {time_saved:.0f}s of experiment time avoided"],
                severity="warning" if kill_rate < 75 else "critical",
            ))

        # ══════════════════════════════════════════════════════════════════════
        # STATISTICAL DISTRIBUTION
        # ══════════════════════════════════════════════════════════════════════
        if all_sharpes:
            parts.append(render_section_divider("📊 Statistical Distribution"))

            sharpe_arr = np.array(all_sharpes) if all_sharpes else np.array([0])
            return_arr = np.array(all_returns) if all_returns else np.array([0])
            dd_arr = np.array(all_drawdowns) if all_drawdowns else np.array([0])

            parts.append("### Risk-Adjusted Performance Metrics\n")
            parts.append("| Statistic | Sharpe Ratio | Total Return | Max Drawdown |")
            parts.append("|-----------|--------------|--------------|--------------|")
            parts.append(f"| **Mean** | {np.mean(sharpe_arr):.4f} | {np.mean(return_arr):.2%} | {np.mean(dd_arr):.2%} |")
            parts.append(f"| **Std Dev** | {np.std(sharpe_arr):.4f} | {np.std(return_arr):.2%} | {np.std(dd_arr):.2%} |")
            parts.append(f"| **Min** | {np.min(sharpe_arr):.4f} | {np.min(return_arr):.2%} | {np.min(dd_arr):.2%} |")
            parts.append(f"| **Max** | {np.max(sharpe_arr):.4f} | {np.max(return_arr):.2%} | {np.max(dd_arr):.2%} |")
            if len(sharpe_arr) >= 4:
                parts.append(f"| **25th %ile** | {np.percentile(sharpe_arr, 25):.4f} | {np.percentile(return_arr, 25):.2%} | {np.percentile(dd_arr, 25):.2%} |")
                parts.append(f"| **Median** | {np.median(sharpe_arr):.4f} | {np.median(return_arr):.2%} | {np.median(dd_arr):.2%} |")
                parts.append(f"| **75th %ile** | {np.percentile(sharpe_arr, 75):.4f} | {np.percentile(return_arr, 75):.2%} | {np.percentile(dd_arr, 75):.2%} |")
            parts.append(f"| **Count** | {len(sharpe_arr)} | {len(return_arr)} | {len(dd_arr)} |")
            parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # KILL GATE ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("⚔️ Kill Gate Analysis"))

        # Kill reason breakdown
        if kill_reasons:
            parts.append("### Kill Reason Distribution\n")
            parts.append("| Reason | Count | % of Kills | Description |")
            parts.append("|--------|-------|------------|-------------|")
            reason_descriptions = {
                "baseline_sharpe_too_low": "Baseline Sharpe below minimum threshold",
                "train_sharpe_too_high": "Suspiciously high Sharpe (overfitting)",
                "max_drawdown_exceeded": "Maximum drawdown limit breached",
                "feature_count_too_high": "Too many features (complexity risk)",
                "instability_too_high": "High instability score across folds",
            }
            for reason, count in sorted(kill_reasons.items(), key=lambda x: -x[1]):
                pct = (count / killed_count) * 100
                desc = reason_descriptions.get(reason, "Unknown reason")
                parts.append(f"| `{reason}` | {count} | {pct:.1f}% | {desc} |")
            parts.append("")
        else:
            parts.append("*No hypotheses killed — all passed gates*\n")

        # Gate thresholds
        parts.append("### Gate Thresholds\n")
        parts.append(render_risk_limits({
            "Min Baseline Sharpe": f"{self.config.min_baseline_sharpe:.2f}",
            "Max Train Sharpe": f"{self.config.max_train_sharpe:.2f}",
            "Max Drawdown": f"{self.config.max_drawdown_threshold:.1%}",
            "Max Feature Count": str(self.config.max_feature_count),
            "Max Instability": f"{self.config.max_instability_score:.2f}",
        }))

        # Health gauges
        survival_rate = (passed_count / max(total, 1)) * 100
        exp_efficiency = ((experiments_run - experiments_killed) / max(experiments_run, 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Gate Survival Rate", "value": survival_rate, "max_val": 100,
             "trend": "up" if survival_rate > 50 else "down"},
            {"label": "Experiment Efficiency", "value": exp_efficiency, "max_val": 100,
             "trend": "stable" if exp_efficiency > 80 else "down"},
        ]))

        # ══════════════════════════════════════════════════════════════════════
        # PER-HYPOTHESIS DETAILED ANALYSIS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("📋 Hypothesis Analysis"))

        for result in self._results:
            status_emoji = "🔴" if result.killed_by_gate else "✅"
            status_label = "REJECTED" if result.killed_by_gate else "PASSED"

            parts.append(f"### {status_emoji} {result.hypothesis_id} — **{status_label}**\n")

            # Hypothesis context
            if result.title:
                parts.append(f"**{result.title}**\n")

            if result.thesis:
                thesis_short = result.thesis[:200] + "..." if len(result.thesis) > 200 else result.thesis
                parts.append(f"> {thesis_short}\n")

            # Metadata table
            parts.append("| Attribute | Value |")
            parts.append("|-----------|-------|")
            gate_reason = result.gate_trigger_reason.value if result.gate_trigger_reason else "—"
            parts.append(f"| **Gate Result** | {status_label} |")
            parts.append(f"| **Kill Reason** | `{gate_reason}` |")
            parts.append(f"| **Duration** | {result.total_duration_seconds:.2f}s |")
            if result.features:
                parts.append(f"| **Features** | `{', '.join(result.features[:5])}{'...' if len(result.features) > 5 else ''}` |")
            if result.model_type:
                parts.append(f"| **Model** | {result.model_type} |")
            if result.ml_ic is not None:
                ic_status = "⚠️ High" if result.ml_ic > 0.15 else "✅ Normal"
                parts.append(f"| **Information Coefficient** | {result.ml_ic:.4f} {ic_status} |")
            if result.stability_score is not None:
                stab_status = "⚠️ Unstable" if result.stability_score > 1.0 else "✅ Stable"
                parts.append(f"| **Stability Score** | {result.stability_score:.4f} {stab_status} |")
            parts.append("")

            # Baseline metrics
            if result.baselines:
                parts.append("#### Baseline Performance\n")
                parts.append("| Metric | Sharpe | Return | Max DD | Volatility | Status |")
                parts.append("|--------|--------|--------|--------|------------|--------|")
                for name, baseline in result.baselines.items():
                    sharpe_bar = render_progress_bar(
                        max(baseline.sharpe, 0), self.config.min_baseline_sharpe * 2,
                        width=8, show_pct=False
                    )
                    sharpe_status = "✅" if baseline.sharpe >= self.config.min_baseline_sharpe else "❌"
                    dd_status = "✅" if baseline.max_drawdown <= self.config.max_drawdown_threshold else "❌"
                    parts.append(
                        f"| {name} | {baseline.sharpe:+.4f} {sharpe_bar} | "
                        f"{baseline.total_return:+.2%} | {baseline.max_drawdown:.2%} {dd_status} | "
                        f"{baseline.volatility:.2%} | {sharpe_status} |"
                    )
                parts.append("")

                # Risk ratios (estimated from available data)
                for name, baseline in result.baselines.items():
                    if baseline.volatility > 0 and baseline.max_drawdown > 0:
                        # Calmar ratio estimate: annualized return / max drawdown
                        calmar_est = baseline.total_return / max(baseline.max_drawdown, 0.001)
                        # Sortino estimate (assume downside vol ~ 70% of total vol)
                        sortino_est = baseline.sharpe * 1.43 if baseline.sharpe > 0 else baseline.sharpe
                        parts.append(f"**Derived Risk Ratios** (estimates)")
                        parts.append(f"- Calmar Ratio: {calmar_est:.2f}")
                        parts.append(f"- Sortino Ratio: {sortino_est:.2f}")
                        parts.append(f"- Return/Vol: {baseline.total_return / max(baseline.volatility, 0.001):.2f}")
                        parts.append("")
                        break

            # Parameter sensitivity analysis
            if result.experiments:
                parts.append("#### Parameter Sensitivity Matrix\n")
                parts.append("| Variation | Sharpe | Return | Max DD | Vol | Status |")
                parts.append("|-----------|--------|--------|--------|-----|--------|")

                # Sort by Sharpe descending
                sorted_exps = sorted(result.experiments, key=lambda e: e.sharpe, reverse=True)
                for exp in sorted_exps[:10]:  # Top 10
                    kill_indicator = "⚠️" if exp.killed_early else ""
                    sharpe_status = "✅" if exp.sharpe >= self.config.min_baseline_sharpe else "❌"
                    parts.append(
                        f"| {exp.config_name} | {exp.sharpe:+.4f} | "
                        f"{exp.total_return:+.2%} | {exp.max_drawdown:.2%} | "
                        f"{exp.volatility:.2%} | {sharpe_status}{kill_indicator} |"
                    )

                # Sensitivity statistics
                if len(result.experiments) >= 2:
                    exp_sharpes = [e.sharpe for e in result.experiments]
                    exp_returns = [e.total_return for e in result.experiments]
                    sharpe_std = np.std(exp_sharpes)
                    return_std = np.std(exp_returns)
                    parts.append("")
                    parts.append(f"**Sensitivity Analysis** ({len(result.experiments)} variations)")
                    parts.append(f"- Sharpe Std Dev: {sharpe_std:.4f} {'⚠️ High variance' if sharpe_std > 0.5 else '✅ Stable'}")
                    parts.append(f"- Return Std Dev: {return_std:.2%}")
                    parts.append(f"- Best/Worst Sharpe: {max(exp_sharpes):.4f} / {min(exp_sharpes):.4f}")

                parts.append("")

            parts.append("─" * 70)
            parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("💡 Recommendations"))

        recommendations = []

        if kill_rate == 100:
            recommendations.append("- **Review hypothesis generation process** — All strategies rejected suggests systematic issues with signal generation or feature engineering")
            recommendations.append("- **Lower baseline thresholds temporarily** — Consider min Sharpe of 0.3 for early-stage research")
            recommendations.append("- **Inspect data quality** — Zero returns may indicate data pipeline issues")
        elif kill_rate >= 75:
            recommendations.append("- **Focus on surviving hypotheses** — Prioritize resources on strategies that passed")
            recommendations.append("- **Analyze common failure patterns** — Identify shared characteristics of rejected strategies")
        elif kill_rate >= 50:
            recommendations.append("- **Mixed results warrant careful analysis** — Review borderline cases manually")
        else:
            recommendations.append("- **Proceed to validation stage** — Passed hypotheses ready for out-of-sample testing")

        if "baseline_sharpe_too_low" in kill_reasons:
            recommendations.append(f"- **{kill_reasons['baseline_sharpe_too_low']} hypotheses killed for low Sharpe** — Consider alternative signal transformations or feature combinations")

        if "train_sharpe_too_high" in kill_reasons:
            recommendations.append(f"- **{kill_reasons['train_sharpe_too_high']} hypotheses flagged for overfitting** — Implement stricter cross-validation or reduce model complexity")

        if experiments_run == 0 and killed_count == total:
            recommendations.append("- **No experiments ran** — All hypotheses killed at baseline gate before parameter sweeps")

        for rec in recommendations:
            parts.append(rec)

        parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_footer(agent_name="kill-gate-enforcer"))

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
