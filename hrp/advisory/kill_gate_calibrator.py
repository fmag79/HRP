"""
Kill Gate Calibration for continuous improvement.

Analyzes historical kill gate decisions to identify:
- False positives: Hypotheses that were killed but similar ones later succeeded
- False negatives: Hypotheses that passed kill gates but were later rejected
- Threshold drift: Whether current thresholds are too strict or too loose

Uses closed recommendations and hypothesis outcomes to recommend threshold adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class CalibrationReport:
    """Results of a kill gate calibration analysis."""

    lookback_days: int
    total_hypotheses: int
    killed: int
    passed: int
    false_positives: int
    false_negatives: int
    # Fields with defaults must come after non-default fields
    false_positive_ids: list[str] = field(default_factory=list)
    false_negative_ids: list[str] = field(default_factory=list)
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    gate_stats: dict[str, dict] = field(default_factory=dict)


@dataclass
class ThresholdRecommendation:
    """Suggested threshold adjustment."""

    gate_name: str
    current_value: float
    suggested_value: float
    direction: str  # "tighten" or "loosen"
    reason: str
    confidence: str  # "high", "medium", "low"


class KillGateCalibrator:
    """
    Analyzes kill gate performance and recommends threshold adjustments.

    Uses historical lineage data and hypothesis outcomes to compute:
    1. False positive rate (killing good hypotheses)
    2. False negative rate (passing bad hypotheses)
    3. Per-gate trigger frequency
    4. Threshold adjustment recommendations
    """

    def __init__(self, api: PlatformAPI):
        self.api = api

    def calibrate(self, lookback_days: int = 180) -> CalibrationReport:
        """
        Run a full calibration analysis over the lookback period.

        Args:
            lookback_days: How far back to look for kill gate decisions.

        Returns:
            CalibrationReport with false positive/negative analysis and recommendations.
        """
        cutoff = date.today() - timedelta(days=lookback_days)

        # Get all hypotheses that went through kill gates
        killed_hyps = self._get_killed_hypotheses(cutoff)
        passed_hyps = self._get_passed_hypotheses(cutoff)

        total = len(killed_hyps) + len(passed_hyps)
        if total == 0:
            return CalibrationReport(
                lookback_days=lookback_days,
                total_hypotheses=0,
                killed=0,
                passed=0,
                false_positives=0,
                false_negatives=0,
                recommendations=["Insufficient data: no kill gate decisions found"],
            )

        # Identify false positives: killed hypotheses that may have been good
        fp_ids = self._find_false_positives(killed_hyps)

        # Identify false negatives: passed hypotheses that later failed
        fn_ids = self._find_false_negatives(passed_hyps)

        fp_rate = len(fp_ids) / len(killed_hyps) if killed_hyps else 0.0
        fn_rate = len(fn_ids) / len(passed_hyps) if passed_hyps else 0.0

        # Per-gate breakdown
        gate_stats = self._compute_gate_stats(killed_hyps, cutoff)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            fp_rate, fn_rate, gate_stats, len(killed_hyps), len(passed_hyps)
        )

        return CalibrationReport(
            lookback_days=lookback_days,
            total_hypotheses=total,
            killed=len(killed_hyps),
            passed=len(passed_hyps),
            false_positives=len(fp_ids),
            false_positive_ids=fp_ids,
            false_negatives=len(fn_ids),
            false_negative_ids=fn_ids,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            gate_stats=gate_stats,
            recommendations=recommendations,
        )

    def suggest_thresholds(
        self, report: CalibrationReport
    ) -> list[ThresholdRecommendation]:
        """
        Suggest concrete threshold adjustments based on calibration report.

        Returns list of ThresholdRecommendation with specific new values.
        """
        suggestions = []

        # If FP rate is high, gates are too strict — loosen them
        if report.false_positive_rate > 0.20:
            for gate, stats in report.gate_stats.items():
                trigger_rate = stats.get("trigger_rate", 0)
                if trigger_rate > 0.30:
                    current = stats.get("threshold", 0)
                    suggested = self._compute_loosened_threshold(gate, current)
                    if suggested is not None:
                        suggestions.append(ThresholdRecommendation(
                            gate_name=gate,
                            current_value=current,
                            suggested_value=suggested,
                            direction="loosen",
                            reason=(
                                f"Gate '{gate}' triggers {trigger_rate:.0%} of the time "
                                f"with FP rate {report.false_positive_rate:.0%}"
                            ),
                            confidence="medium" if trigger_rate > 0.40 else "low",
                        ))

        # If FN rate is high, gates are too loose — tighten them
        if report.false_negative_rate > 0.30:
            for gate, stats in report.gate_stats.items():
                trigger_rate = stats.get("trigger_rate", 0)
                if trigger_rate < 0.10:
                    current = stats.get("threshold", 0)
                    suggested = self._compute_tightened_threshold(gate, current)
                    if suggested is not None:
                        suggestions.append(ThresholdRecommendation(
                            gate_name=gate,
                            current_value=current,
                            suggested_value=suggested,
                            direction="tighten",
                            reason=(
                                f"Gate '{gate}' rarely triggers ({trigger_rate:.0%}) "
                                f"while FN rate is {report.false_negative_rate:.0%}"
                            ),
                            confidence="medium",
                        ))

        return suggestions

    # --- Private helpers ---

    def _get_killed_hypotheses(self, since: date) -> list[dict]:
        """Get hypotheses that were killed by kill gates."""
        df = self.api.query_readonly(
            "SELECT l.hypothesis_id, l.details, l.timestamp "
            "FROM lineage l "
            "WHERE l.event_type = 'kill_gate_triggered' "
            "AND l.timestamp >= ? "
            "ORDER BY l.timestamp DESC",
            [since],
        )
        if df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            results.append({
                "hypothesis_id": row["hypothesis_id"],
                "killed_at": row["timestamp"],
                "reason": details.get("reason", "unknown"),
                "details": details,
            })
        return results

    def _get_passed_hypotheses(self, since: date) -> list[dict]:
        """Get hypotheses that passed kill gates."""
        df = self.api.query_readonly(
            "SELECT l.hypothesis_id, l.details, l.timestamp "
            "FROM lineage l "
            "WHERE l.event_type = 'kill_gate_enforcer_complete' "
            "AND l.timestamp >= ? "
            "ORDER BY l.timestamp DESC",
            [since],
        )
        if df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            details = row.get("details", {})
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            # Only include hypotheses that actually passed (not killed)
            hyp_id = row.get("hypothesis_id")
            if hyp_id:
                results.append({
                    "hypothesis_id": hyp_id,
                    "passed_at": row["timestamp"],
                    "details": details,
                })
        return results

    def _find_false_positives(self, killed_hyps: list[dict]) -> list[str]:
        """
        Find killed hypotheses that were likely false positives.

        A false positive is detected when:
        - A hypothesis was killed, but a later hypothesis with similar
          features/thesis eventually succeeded (validated or deployed)
        - OR the hypothesis was later reopened and succeeded

        For simplicity, we check if any killed hypothesis was later reopened
        to draft/testing/validated status.
        """
        fp_ids = []
        for hyp in killed_hyps:
            hyp_id = hyp.get("hypothesis_id")
            if not hyp_id:
                continue

            # Check if hypothesis was reopened after being killed
            result = self.api.fetchone_readonly(
                "SELECT status FROM hypotheses WHERE hypothesis_id = ?",
                [hyp_id],
            )
            if result and result[0] in ("testing", "validated", "deployed"):
                fp_ids.append(hyp_id)
                continue

            # Check lineage for reopening events after kill
            reopen_events = self.api.query_readonly(
                "SELECT event_type, timestamp FROM lineage "
                "WHERE hypothesis_id = ? "
                "AND event_type = 'hypothesis_updated' "
                "AND timestamp > ? "
                "ORDER BY timestamp",
                [hyp_id, hyp["killed_at"]],
            )
            if not reopen_events.empty:
                # Hypothesis was modified after being killed — possible FP
                fp_ids.append(hyp_id)

        return fp_ids

    def _find_false_negatives(self, passed_hyps: list[dict]) -> list[str]:
        """
        Find passed hypotheses that later failed.

        A false negative is a hypothesis that:
        - Passed kill gates but was later rejected by downstream agents
        - Or had poor recommendation outcomes (negative track record)
        """
        fn_ids = []
        for hyp in passed_hyps:
            hyp_id = hyp.get("hypothesis_id")
            if not hyp_id:
                continue

            # Check if hypothesis was rejected after passing kill gates
            result = self.api.fetchone_readonly(
                "SELECT status FROM hypotheses WHERE hypothesis_id = ?",
                [hyp_id],
            )
            if result and result[0] == "rejected":
                fn_ids.append(hyp_id)
                continue

            # Check recommendation outcomes for this hypothesis's model
            recs = self.api.query_readonly(
                "SELECT realized_return FROM recommendations "
                "WHERE hypothesis_id = ? "
                "AND status IN ('closed_profit', 'closed_loss', 'closed_stopped')",
                [hyp_id],
            )
            if not recs.empty:
                avg_return = float(recs["realized_return"].astype(float).mean())
                if avg_return < -0.02:
                    fn_ids.append(hyp_id)

        return fn_ids

    def _compute_gate_stats(
        self, killed_hyps: list[dict], since: date
    ) -> dict[str, dict]:
        """Compute per-gate trigger statistics."""
        gate_counts: dict[str, int] = {}
        total = len(killed_hyps)

        for hyp in killed_hyps:
            reason = hyp.get("reason", "unknown")
            gate_counts[reason] = gate_counts.get(reason, 0) + 1

        # Map gate reasons to thresholds
        gate_thresholds = {
            "baseline_sharpe_too_low": ("min_baseline_sharpe", 0.5),
            "train_sharpe_too_high": ("max_train_sharpe", 3.0),
            "max_drawdown_exceeded": ("max_drawdown_threshold", 0.30),
            "feature_count_too_high": ("max_feature_count", 50),
            "instability_too_high": ("max_instability_score", 1.5),
        }

        stats = {}
        for reason, count in gate_counts.items():
            threshold_info = gate_thresholds.get(reason, (reason, 0))
            stats[reason] = {
                "trigger_count": count,
                "trigger_rate": count / total if total > 0 else 0,
                "threshold_name": threshold_info[0],
                "threshold": threshold_info[1],
            }

        return stats

    def _generate_recommendations(
        self,
        fp_rate: float,
        fn_rate: float,
        gate_stats: dict[str, dict],
        n_killed: int,
        n_passed: int,
    ) -> list[str]:
        """Generate human-readable threshold adjustment recommendations."""
        recs = []

        total = n_killed + n_passed
        kill_rate = n_killed / total if total > 0 else 0

        if total < 10:
            recs.append(
                f"Limited data ({total} hypotheses). "
                "Recommendations will be more reliable with 20+ observations."
            )

        if fp_rate > 0.20:
            recs.append(
                f"High false positive rate ({fp_rate:.0%}): "
                "Kill gates may be too strict. Consider loosening thresholds."
            )

        if fn_rate > 0.30:
            recs.append(
                f"High false negative rate ({fn_rate:.0%}): "
                "Kill gates may be too lenient. Consider tightening thresholds."
            )

        if fp_rate <= 0.10 and fn_rate <= 0.10:
            recs.append(
                f"Kill gates well-calibrated: FP={fp_rate:.0%}, FN={fn_rate:.0%}. "
                "No threshold changes recommended."
            )

        if kill_rate > 0.70:
            recs.append(
                f"Kill rate is very high ({kill_rate:.0%}). "
                "Most hypotheses are being killed — may indicate signals are weak "
                "or thresholds are too aggressive."
            )
        elif kill_rate < 0.10:
            recs.append(
                f"Kill rate is very low ({kill_rate:.0%}). "
                "Kill gates may not be filtering effectively."
            )

        # Per-gate specific recommendations
        for gate, stats in gate_stats.items():
            rate = stats.get("trigger_rate", 0)
            if rate > 0.50:
                recs.append(
                    f"Gate '{gate}' triggers on {rate:.0%} of kills. "
                    f"Consider reviewing threshold ({stats.get('threshold_name', '')}="
                    f"{stats.get('threshold', 'N/A')})."
                )

        return recs

    def _compute_loosened_threshold(
        self, gate: str, current: float
    ) -> float | None:
        """Compute a loosened threshold (more permissive)."""
        adjustments = {
            "baseline_sharpe_too_low": lambda v: max(0.2, v - 0.1),
            "train_sharpe_too_high": lambda v: v + 0.5,
            "max_drawdown_exceeded": lambda v: min(0.40, v + 0.05),
            "instability_too_high": lambda v: v + 0.3,
            "feature_count_too_high": lambda v: v + 10,
        }
        fn = adjustments.get(gate)
        return fn(current) if fn else None

    def _compute_tightened_threshold(
        self, gate: str, current: float
    ) -> float | None:
        """Compute a tightened threshold (more restrictive)."""
        adjustments = {
            "baseline_sharpe_too_low": lambda v: v + 0.1,
            "train_sharpe_too_high": lambda v: max(1.5, v - 0.5),
            "max_drawdown_exceeded": lambda v: max(0.15, v - 0.05),
            "instability_too_high": lambda v: max(0.5, v - 0.3),
            "feature_count_too_high": lambda v: max(10, v - 10),
        }
        fn = adjustments.get(gate)
        return fn(current) if fn else None

    def persist_calibration(self, report: CalibrationReport) -> None:
        """Log calibration results to lineage for audit trail."""
        from hrp.research.lineage import log_event

        log_event(
            event_type="other",
            actor="agent:kill-gate-calibrator",
            details={
                "action": "kill_gate_calibration",
                "lookback_days": report.lookback_days,
                "total_hypotheses": report.total_hypotheses,
                "killed": report.killed,
                "passed": report.passed,
                "false_positive_rate": report.false_positive_rate,
                "false_negative_rate": report.false_negative_rate,
                "false_positives": report.false_positive_ids,
                "false_negatives": report.false_negative_ids,
                "recommendations": report.recommendations,
            },
        )
        logger.info(
            f"Kill gate calibration: {report.total_hypotheses} hypotheses, "
            f"FP={report.false_positive_rate:.0%}, FN={report.false_negative_rate:.0%}"
        )
