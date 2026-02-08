"""Daily prediction job for deployed strategies."""
import logging
from datetime import date
from typing import Any

from hrp.agents.jobs import DataRequirement, IngestionJob
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)


class DailyPredictionJob(IngestionJob):
    """Generate daily predictions for all deployed strategies."""

    def __init__(
        self,
        job_id: str = "daily_predictions",
        api: PlatformAPI | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize daily prediction job.

        Args:
            job_id: Job identifier
            api: PlatformAPI instance (creates new if None)
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier
        """
        data_requirements = [
            DataRequirement(
                table="features",
                min_rows=100,
                max_age_days=3,
                date_column="date",
                description="Recent feature data",
            ),
            DataRequirement(
                table="prices",
                min_rows=1000,
                max_age_days=3,
                date_column="date",
                description="Recent price data",
            ),
        ]

        super().__init__(
            job_id,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            data_requirements=data_requirements,
        )
        # Override the base class api if one was provided
        if api is not None:
            self.api = api

    def execute(self) -> dict[str, Any]:
        """Execute daily prediction job.

        Returns:
            Dict with execution stats:
                - status: success, partial_failure, or no_deployed_strategies
                - predictions_generated: Count of predictions
                - strategies_processed: Count of strategies
                - errors: Count of errors
        """
        logger.info("Starting daily prediction job")

        # Get all deployed strategies
        deployed = self.api.get_deployed_strategies()

        if not deployed:
            logger.warning("No deployed strategies found")
            return {
                "status": "no_deployed_strategies",
                "predictions_generated": 0,
                "strategies_processed": 0,
                "errors": 0,
                "records_fetched": 0,
                "records_inserted": 0,
            }

        # Get universe for predictions
        universe = self.api.get_universe(as_of_date=date.today())

        total_predictions = 0
        errors = 0

        for strategy in deployed:
            hypothesis_id = strategy.get("hypothesis_id") or strategy.hypothesis_id

            try:
                # Get model name from metadata
                metadata = strategy.get("metadata") or getattr(strategy, "metadata", {})
                model_name = metadata.get("model_name") if isinstance(metadata, dict) else None

                if not model_name:
                    logger.warning(
                        f"Strategy {hypothesis_id} has no model_name in metadata, skipping"
                    )
                    errors += 1
                    continue

                # Generate predictions
                predictions = self.api.predict_model(
                    model_name=model_name,
                    symbols=universe,
                    as_of_date=date.today(),
                    model_version=None,  # Use production version
                )

                num_predictions = len(predictions) if predictions is not None else 0
                total_predictions += num_predictions

                logger.info(
                    f"Generated {num_predictions} predictions for {hypothesis_id} "
                    f"(model={model_name})"
                )

                # Log to lineage
                self.api.log_event(
                    event_type="agent_run_complete",
                    actor="system:prediction_job",
                    hypothesis_id=hypothesis_id,
                    details={
                        "model_name": model_name,
                        "num_predictions": num_predictions,
                        "as_of_date": str(date.today()),
                    },
                )

            except Exception as e:
                logger.error(f"Failed to generate predictions for {hypothesis_id}: {e}")
                errors += 1
                continue

        status = "success" if errors == 0 else "partial_failure"

        return {
            "status": status,
            "predictions_generated": total_predictions,
            "strategies_processed": len(deployed) - errors,
            "errors": errors,
            "records_fetched": total_predictions,
            "records_inserted": total_predictions,
        }
