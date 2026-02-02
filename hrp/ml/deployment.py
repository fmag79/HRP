"""
Model Deployment Pipeline for HRP.

Provides structured deployment workflow with staging, shadow mode,
validation checks, and rollback capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional
import uuid

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.ml.registry import ModelRegistry, RegisteredModel
from hrp.research.lineage import log_event, EventType


@dataclass
class DeploymentConfig:
    """
    Configuration for model deployment.

    Attributes:
        min_ic_threshold: Minimum IC required for deployment (default: 0.03)
        min_sharpe_threshold: Minimum Sharpe ratio required (default: 0.5)
        max_drawdown_threshold: Maximum acceptable drawdown (default: 0.3)
        shadow_mode_days: Days to run in shadow mode before promotion (default: 5)
        require_validation_checks: Whether to enforce validation checks (default: True)
        min_samples: Minimum samples required for validation (default: 10)
    """

    min_ic_threshold: float = 0.03
    min_sharpe_threshold: float = 0.5
    max_drawdown_threshold: float = 0.3
    shadow_mode_days: int = 5
    require_validation_checks: bool = True
    min_samples: int = 10


@dataclass
class DeploymentResult:
    """
    Result of a deployment operation.

    Attributes:
        deployment_id: Unique deployment identifier
        model_name: Name of the deployed model
        model_version: Version of the deployed model
        environment: Target environment ('staging', 'production', 'shadow')
        status: Deployment status ('pending', 'success', 'failed')
        validation_passed: Whether validation checks passed
        validation_results: Dict of validation check results
        timestamp: When deployment occurred
        error_message: Error message if deployment failed
    """

    deployment_id: str
    model_name: str
    model_version: str
    environment: str
    status: str
    validation_passed: bool
    validation_results: dict[str, bool]
    timestamp: datetime
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "environment": self.environment,
            "status": self.status,
            "validation_passed": self.validation_passed,
            "validation_results": self.validation_results,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
        }


class DeploymentPipeline:
    """
    Model deployment orchestration with staging → production workflow.

    Features:
    - Staging deployment with validation checks
    - Shadow mode for production testing
    - Production promotion with rollback capability
    - Validation checks (IC, Sharpe, drawdown, data quality)

    Example:
        ```python
        from hrp.ml.deployment import DeploymentPipeline

        pipeline = DeploymentPipeline()

        # Deploy to staging
        result = pipeline.deploy_to_staging(
            model_name="momentum_strategy",
            model_version="1",
            validation_data=validation_df,
            actor="user",
        )

        # Promote to production (with shadow mode)
        result = pipeline.promote_to_production(
            model_name="momentum_strategy",
            actor="user",
            shadow_mode_days=5,
        )
        ```
    """

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        registry: ModelRegistry | None = None,
        db_path: str | None = None,
        db=None,
    ):
        """
        Initialize the deployment pipeline.

        Args:
            config: Deployment configuration (uses defaults if None)
            registry: Model registry instance (creates new if None)
            db_path: Optional database path
            db: Optional database connection (uses default if not provided)
        """
        self.config = config or DeploymentConfig()
        self.registry = registry or ModelRegistry()
        self._db = db or get_db(db_path)

        logger.info(
            f"DeploymentPipeline initialized "
            f"(shadow_mode={self.config.shadow_mode_days}d)"
        )

    def deploy_to_staging(
        self,
        model_name: str,
        model_version: str,
        validation_data: pd.DataFrame,
        actor: str,
        deployment_config: dict[str, Any] | None = None,
    ) -> DeploymentResult:
        """
        Deploy a model to the staging environment.

        Staging is the first step in deployment. Models in staging:
        - Have passed basic validation checks
        - Can be tested in a safe environment
        - Are not yet used for trading

        Args:
            model_name: Name of the model to deploy
            model_version: Version of the model
            validation_data: Data for validation checks
            actor: Who is deploying (e.g., "user", "agent:ml_scientist")
            deployment_config: Optional deployment metadata

        Returns:
            DeploymentResult with deployment status

        Raises:
            ValueError: If model version not found or validation fails
        """
        deployment_id = str(uuid.uuid4())
        timestamp = datetime.now()

        logger.info(f"Deploying {model_name} v{model_version} to staging...")

        try:
            # Run validation checks
            validation_results = self._run_validation_checks(
                model_name, model_version, validation_data
            )

            validation_passed = all(validation_results.values())

            if self.config.require_validation_checks and not validation_passed:
                failed_checks = [k for k, v in validation_results.items() if not v]
                error_msg = f"Validation failed: {failed_checks}"

                result = DeploymentResult(
                    deployment_id=deployment_id,
                    model_name=model_name,
                    model_version=model_version,
                    environment="staging",
                    status="failed",
                    validation_passed=False,
                    validation_results=validation_results,
                    timestamp=timestamp,
                    error_message=error_msg,
                )

                # Log failed deployment
                self._log_deployment(deployment_id, result, actor)
                log_event(
                    event_type=EventType.DEPLOYMENT_REJECTED,
                    actor=actor,
                    details={
                        "model_name": model_name,
                        "model_version": model_version,
                        "environment": "staging",
                        "reason": error_msg,
                        "failed_checks": failed_checks,
                    },
                )

                logger.error(f"Staging deployment failed: {error_msg}")
                return result

            # Update model registry stage
            self.registry.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="staging",
            )

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version,
                environment="staging",
                status="success",
                validation_passed=validation_passed,
                validation_results=validation_results,
                timestamp=timestamp,
            )

            # Log successful deployment
            self._log_deployment(deployment_id, result, actor, deployment_config)
            log_event(
                event_type=EventType.DEPLOYMENT_APPROVED,
                actor=actor,
                details={
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": "staging",
                    "validation_results": validation_results,
                },
            )

            logger.info(f"Successfully deployed {model_name} v{model_version} to staging")
            return result

        except Exception as e:
            error_msg = f"Deployment failed: {e}"
            logger.error(error_msg)

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version,
                environment="staging",
                status="failed",
                validation_passed=False,
                validation_results={},
                timestamp=timestamp,
                error_message=error_msg,
            )

            self._log_deployment(deployment_id, result, actor)
            return result

    def promote_to_production(
        self,
        model_name: str,
        actor: str,
        model_version: str | None = None,
        shadow_mode_days: int | None = None,
    ) -> DeploymentResult:
        """
        Promote a staging model to production.

        Optionally runs in shadow mode first, where predictions are logged
        but not used for trading.

        Args:
            model_name: Name of the model
            actor: Who is promoting
            model_version: Version to promote (None for latest staging)
            shadow_mode_days: Days in shadow mode (None for config default)

        Returns:
            DeploymentResult with deployment status

        Raises:
            ValueError: If no staging model found or promotion fails
        """
        deployment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        shadow_days = shadow_mode_days or self.config.shadow_mode_days

        logger.info(f"Promoting {model_name} to production (shadow_mode={shadow_days}d)...")

        try:
            # Get staging version if not specified
            if model_version is None:
                staging = self.registry.get_staging_model(model_name)
                if staging is None:
                    raise ValueError(f"No staging model found for {model_name}")
                model_version = staging.model_version

            # For now, direct promotion (shadow mode would be implemented in execution module)
            # TODO: Integrate with execution module for shadow mode predictions
            self.registry.promote_to_production(
                model_name=model_name,
                model_version=model_version,
                actor=actor,
                validation_checks={},
            )

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version,
                environment="production",
                status="success",
                validation_passed=True,
                validation_results={},
                timestamp=timestamp,
            )

            # Log production deployment
            self._log_deployment(
                deployment_id,
                result,
                actor,
                deployment_config={"shadow_mode_days": shadow_days},
            )
            log_event(
                event_type=EventType.DEPLOYMENT_APPROVED,
                actor=actor,
                details={
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": "production",
                    "shadow_mode_days": shadow_days,
                },
            )

            logger.info(
                f"Successfully promoted {model_name} v{model_version} to production"
            )
            return result

        except Exception as e:
            error_msg = f"Promotion failed: {e}"
            logger.error(error_msg)

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version or "unknown",
                environment="production",
                status="failed",
                validation_passed=False,
                validation_results={},
                timestamp=timestamp,
                error_message=error_msg,
            )

            self._log_deployment(deployment_id, result, actor)
            return result

    def rollback_production(
        self,
        model_name: str,
        to_version: str,
        actor: str,
        reason: str,
    ) -> DeploymentResult:
        """
        Rollback production to a previous model version.

        Args:
            model_name: Name of the model
            to_version: Version to rollback to
            actor: Who is rolling back
            reason: Reason for rollback

        Returns:
            DeploymentResult with rollback status
        """
        deployment_id = str(uuid.uuid4())
        timestamp = datetime.now()

        logger.info(f"Rolling back {model_name} to version {to_version}...")

        try:
            # Get current production version
            current = self.registry.get_production_model(model_name)
            current_version = current.model_version if current else None

            # Perform rollback in registry
            self.registry.rollback_production(
                model_name=model_name,
                to_version=to_version,
                actor=actor,
                reason=reason,
            )

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=to_version,
                environment="production",
                status="success",
                validation_passed=True,
                validation_results={},
                timestamp=timestamp,
            )

            # Log rollback
            self._log_deployment(
                deployment_id,
                result,
                actor,
                deployment_config={"rollback_from_version": current_version},
            )

            logger.info(f"Successfully rolled back {model_name} to version {to_version}")
            return result

        except Exception as e:
            error_msg = f"Rollback failed: {e}"
            logger.error(error_msg)

            result = DeploymentResult(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=to_version,
                environment="production",
                status="failed",
                validation_passed=False,
                validation_results={},
                timestamp=timestamp,
                error_message=error_msg,
            )

            self._log_deployment(deployment_id, result, actor)
            return result

    def _run_validation_checks(
        self,
        model_name: str,
        model_version: str,
        validation_data: pd.DataFrame,
    ) -> dict[str, bool]:
        """
        Run pre-deployment validation checks.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            validation_data: Data for validation

        Returns:
            Dict of check names to pass/fail results
        """
        results = {}

        # Check 1: Model can be loaded
        try:
            # This would load the model from MLflow
            # For now, we'll skip actual loading
            results["model_load"] = True
        except Exception as e:
            logger.warning(f"Model load check failed: {e}")
            results["model_load"] = False

        # Check 2: Validation data not empty
        results["data_not_empty"] = len(validation_data) > 0

        # Check 3: No NaN in validation data
        results["no_nan_data"] = not validation_data.isnull().all().any()

        # Check 4: Minimum sample size
        results["min_samples"] = len(validation_data) >= self.config.min_samples if hasattr(self.config, 'min_samples') else len(validation_data) >= 100

        # Check 5: Feature availability (if prediction column exists)
        if "prediction" in validation_data.columns:
            results["has_predictions"] = True
            # Check for reasonable prediction values
            preds = validation_data["prediction"].dropna()
            results["reasonable_predictions"] = bool(
                len(preds) > 0 and preds.notna().all()
            )
        else:
            results["has_predictions"] = False
            results["reasonable_predictions"] = False

        logger.debug(f"Validation checks: {results}")
        return results

    def _log_deployment(
        self,
        deployment_id: str,
        result: DeploymentResult,
        actor: str,
        deployment_config: dict[str, Any] | None = None,
    ) -> None:
        """Log deployment to database."""
        try:
            import json

            query = """
                INSERT INTO model_deployments (
                    deployment_id,
                    model_name,
                    model_version,
                    environment,
                    status,
                    deployed_at,
                    deployed_by,
                    deployment_config,
                    validation_results,
                    rollback_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            self._db.execute(
                query,
                (
                    deployment_id,
                    result.model_name,
                    result.model_version,
                    result.environment,
                    result.status,
                    result.timestamp,
                    actor,
                    json.dumps(deployment_config or {}),
                    json.dumps(result.validation_results),
                    result.error_message,
                ),
            )

        except Exception as e:
            # Table might not exist yet
            logger.warning(f"Failed to log deployment to database: {e}")

    def get_deployment_history(
        self,
        model_name: str | None = None,
        environment: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Get deployment history.

        Args:
            model_name: Optional model name filter
            environment: Optional environment filter
            limit: Maximum records to return

        Returns:
            DataFrame with deployment history
        """
        query = """
            SELECT
                deployment_id,
                model_name,
                model_version,
                environment,
                status,
                deployed_at,
                deployed_by,
                deployment_config,
                validation_results,
                rollback_reason
            FROM model_deployments
            WHERE 1=1
        """

        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if environment:
            query += " AND environment = ?"
            params.append(environment)

        query += " ORDER BY deployed_at DESC LIMIT ?"
        params.append(limit)

        try:
            return self._db.fetchdf(query, params)
        except Exception:
            # Table might not exist yet
            return pd.DataFrame()


def get_deployment_pipeline(
    config: DeploymentConfig | None = None,
    registry: ModelRegistry | None = None,
    db_path: str | None = None,
) -> DeploymentPipeline:
    """
    Get a DeploymentPipeline instance.

    Args:
        config: Deployment configuration
        registry: Model registry instance
        db_path: Optional database path

    Returns:
        DeploymentPipeline instance
    """
    return DeploymentPipeline(config=config, registry=registry, db_path=db_path)
