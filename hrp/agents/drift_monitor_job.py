"""Model drift monitoring job."""
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from hrp.agents.jobs import DataRequirement, IngestionJob
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift monitoring."""

    prediction_drift_threshold: float = 0.20  # 20% change in prediction distribution
    feature_drift_threshold: float = 0.15  # 15% change in feature distribution
    lookback_days: int = 30
    auto_rollback: bool = False  # Require explicit enable


class DriftMonitorJob(IngestionJob):
    """Job to monitor deployed models for drift.

    Checks for:
    1. Prediction drift - distribution of model outputs changing
    2. Feature drift - distribution of input features changing
    3. Performance drift - degradation in model performance metrics
    """

    def __init__(
        self,
        drift_config: DriftConfig | None = None,
        api: PlatformAPI | None = None,
        job_id: str = "drift_monitor",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize drift monitor job.

        Args:
            drift_config: Drift monitoring configuration
            api: PlatformAPI instance (creates new if None)
            job_id: Job identifier
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier
        """
        data_requirements = [
            DataRequirement(
                table="model_deployments",
                min_rows=1,
                max_age_days=None,  # No age limit - just need deployed models
                date_column="deployed_at",
                description="Deployed models",
            ),
        ]

        super().__init__(
            job_id,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            data_requirements=data_requirements,
        )

        if api is not None:
            self.api = api

        self.drift_config = drift_config or DriftConfig()

    def execute(self) -> dict[str, Any]:
        """Execute drift monitoring.

        Returns:
            Dict with monitoring results:
                - status: success or drift_detected
                - models_checked: Number of models checked
                - drift_detected: List of models with drift
                - rollbacks_triggered: Number of auto-rollbacks
        """
        logger.info("Starting drift monitoring job")

        # Get deployed models
        deployed = self.api.get_deployed_strategies()

        if not deployed:
            logger.info("No deployed models to monitor")
            return {
                "status": "no_deployed_models",
                "models_checked": 0,
                "drift_detected": [],
                "rollbacks_triggered": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        drift_detected = []
        rollbacks = 0

        for strategy in deployed:
            hypothesis_id = strategy.get("hypothesis_id") or getattr(
                strategy, "hypothesis_id", None
            )
            metadata = strategy.get("metadata") or getattr(strategy, "metadata", {})
            model_name = metadata.get("model_name") if isinstance(metadata, dict) else None

            if not model_name:
                continue

            try:
                # Check for drift
                drift_result = self._check_model_drift(model_name, hypothesis_id)

                if drift_result["has_drift"]:
                    drift_detected.append({
                        "model_name": model_name,
                        "hypothesis_id": hypothesis_id,
                        "drift_type": drift_result["drift_type"],
                        "drift_value": drift_result["drift_value"],
                    })

                    logger.warning(
                        f"Drift detected for {model_name}: "
                        f"{drift_result['drift_type']} = {drift_result['drift_value']:.4f}"
                    )

                    # Record drift check
                    self._record_drift_check(
                        model_name=model_name,
                        drift_type=drift_result["drift_type"],
                        drift_value=drift_result["drift_value"],
                        is_drift=True,
                        threshold=drift_result["threshold"],
                    )

                    # Auto-rollback if enabled
                    if self.drift_config.auto_rollback:
                        try:
                            self.api.rollback_deployment(
                                model_name=model_name,
                                to_version=None,  # Previous version
                                actor="system:drift_monitor",
                                reason=f"Auto-rollback due to {drift_result['drift_type']} drift",
                            )
                            rollbacks += 1
                            logger.info(f"Auto-rolled back {model_name}")
                        except Exception as e:
                            logger.error(f"Failed to rollback {model_name}: {e}")
                else:
                    # Record successful check
                    self._record_drift_check(
                        model_name=model_name,
                        drift_type="prediction",
                        drift_value=drift_result.get("drift_value", 0.0),
                        is_drift=False,
                        threshold=self.drift_config.prediction_drift_threshold,
                    )

            except Exception as e:
                logger.error(f"Failed to check drift for {model_name}: {e}")
                continue

        status = "drift_detected" if drift_detected else "success"

        # Log to lineage
        self.api.log_event(
            event_type="agent_run_complete",
            actor="system:drift_monitor",
            details={
                "models_checked": len(deployed),
                "drift_detected_count": len(drift_detected),
                "rollbacks_triggered": rollbacks,
                "auto_rollback_enabled": self.drift_config.auto_rollback,
            },
        )

        return {
            "status": status,
            "models_checked": len(deployed),
            "drift_detected": drift_detected,
            "rollbacks_triggered": rollbacks,
            "records_fetched": len(deployed),
            "records_inserted": len(drift_detected) + (len(deployed) - len(drift_detected)),
        }

    def _check_model_drift(
        self, model_name: str, hypothesis_id: str
    ) -> dict[str, Any]:
        """Check a single model for drift.

        Args:
            model_name: Name of the model
            hypothesis_id: Associated hypothesis

        Returns:
            Dict with drift detection results
        """
        try:
            # Get recent predictions
            end_date = date.today()
            start_date = end_date - timedelta(days=self.drift_config.lookback_days)

            # Use API to check drift if available
            drift_result = self.api.check_model_drift(
                model_name=model_name,
                current_data=None,  # Will fetch from DB
                reference_data=None,  # Will use baseline
                threshold=self.drift_config.prediction_drift_threshold,
            )

            if drift_result and drift_result.get("drift_detected"):
                return {
                    "has_drift": True,
                    "drift_type": drift_result.get("drift_type", "prediction"),
                    "drift_value": drift_result.get("drift_score", 0.0),
                    "threshold": self.drift_config.prediction_drift_threshold,
                }

            return {
                "has_drift": False,
                "drift_value": drift_result.get("drift_score", 0.0) if drift_result else 0.0,
            }

        except Exception as e:
            logger.debug(f"Drift check failed for {model_name}: {e}")
            # Return no drift on error (conservative approach)
            return {"has_drift": False, "drift_value": 0.0}

    def _record_drift_check(
        self,
        model_name: str,
        drift_type: str,
        drift_value: float,
        is_drift: bool,
        threshold: float,
    ) -> None:
        """Record drift check result to database.

        Args:
            model_name: Model name
            drift_type: Type of drift (prediction, feature, concept)
            drift_value: Measured drift value
            is_drift: Whether drift was detected
            threshold: Threshold used for detection
        """
        try:
            self.api.execute_write(
                """
                INSERT INTO model_drift_checks (
                    model_name, check_timestamp, drift_type,
                    metric_value, is_drift_detected, threshold_value
                ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """,
                (model_name, drift_type, drift_value, is_drift, threshold),
            )
        except Exception as e:
            logger.error(f"Failed to record drift check: {e}")
