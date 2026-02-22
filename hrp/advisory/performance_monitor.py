"""
Model performance monitor for the advisory service.

Tracks whether model predictions remain accurate over time and
triggers retraining when performance degrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class AccuracyReport:
    """Report on model prediction accuracy."""

    model_name: str
    lookback_days: int
    total_predictions: int
    directional_accuracy: float  # % of correct direction predictions
    avg_prediction_error: float  # Mean absolute error
    correlation: float | None  # Correlation between predictions and actual
    degraded: bool
    message: str


@dataclass
class StabilityReport:
    """Report on feature importance stability."""

    model_name: str
    feature_drift_detected: bool
    top_features_changed: int  # How many of top 5 features changed
    message: str


class ModelPerformanceMonitor:
    """Tracks model accuracy and triggers retraining when needed."""

    def __init__(self, api: PlatformAPI):
        self.api = api
        self.min_accuracy = 0.50  # Below 50% = coin flip
        self.min_correlation = 0.02  # Below 0.02 IC = no signal

    def check_prediction_accuracy(
        self, model_name: str, lookback_days: int = 60
    ) -> AccuracyReport:
        """
        Compare recent predictions to actual outcomes.

        Uses closed recommendations to evaluate whether the model's
        directional calls were correct.
        """
        cutoff = date.today() - timedelta(days=lookback_days)

        closed = self.api.query_readonly(
            "SELECT symbol, signal_strength, entry_price, close_price, realized_return "
            "FROM recommendations "
            "WHERE model_name = ? AND closed_at IS NOT NULL AND created_at >= ?",
            [model_name, cutoff],
        )

        if closed.empty or len(closed) < 5:
            return AccuracyReport(
                model_name=model_name,
                lookback_days=lookback_days,
                total_predictions=len(closed),
                directional_accuracy=0.0,
                avg_prediction_error=0.0,
                correlation=None,
                degraded=False,
                message=f"Insufficient closed recommendations ({len(closed)}) for accuracy check",
            )

        # Directional accuracy
        signals = closed["signal_strength"].astype(float)
        returns = closed["realized_return"].astype(float)
        correct = ((signals > 0) & (returns > 0)) | ((signals < 0) & (returns < 0))
        accuracy = float(correct.mean())

        # Prediction error
        avg_error = float((signals - returns).abs().mean())

        # Correlation (IC)
        correlation = None
        if signals.std() > 0 and returns.std() > 0:
            correlation = float(signals.corr(returns))

        # Degradation check
        degraded = accuracy < self.min_accuracy
        if correlation is not None and correlation < self.min_correlation:
            degraded = True

        message = f"Accuracy: {accuracy:.0%}, IC: {correlation:.3f}" if correlation else f"Accuracy: {accuracy:.0%}"
        if degraded:
            message += " — DEGRADED, retraining recommended"

        return AccuracyReport(
            model_name=model_name,
            lookback_days=lookback_days,
            total_predictions=len(closed),
            directional_accuracy=accuracy,
            avg_prediction_error=avg_error,
            correlation=correlation,
            degraded=degraded,
            message=message,
        )

    def check_all_models(self, lookback_days: int = 60) -> list[AccuracyReport]:
        """Check accuracy for all models with closed recommendations."""
        models = self.api.query_readonly(
            "SELECT DISTINCT model_name FROM recommendations "
            "WHERE model_name IS NOT NULL AND model_name != ''"
        )
        if models.empty:
            return []

        reports = []
        for _, row in models.iterrows():
            report = self.check_prediction_accuracy(row["model_name"], lookback_days)
            reports.append(report)
        return reports

    def trigger_retraining(self, model_name: str, reason: str) -> None:
        """
        Log a RETRAIN_TRIGGERED event to the lineage table.

        The ML Scientist agent picks this up via the event watcher
        and initiates retraining.
        """
        from hrp.research.lineage import log_event

        log_event(
            event_type="other",  # Use 'other' until we add RETRAIN_TRIGGERED to the enum
            actor="agent:performance-monitor",
            details={
                "action": "retrain_triggered",
                "model_name": model_name,
                "reason": reason,
            },
        )
        logger.warning(f"Retraining triggered for {model_name}: {reason}")

    def run_monitoring_cycle(self, lookback_days: int = 60) -> list[AccuracyReport]:
        """
        Run a full monitoring cycle: check all models, trigger retraining if needed.
        """
        reports = self.check_all_models(lookback_days)

        for report in reports:
            if report.degraded:
                self.trigger_retraining(
                    report.model_name,
                    f"Performance degraded: {report.message}",
                )

        return reports
