"""
Risk Configuration API for HRP.

Provides UI access to Risk Manager limits and risk assessment preview.
"""

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any

from loguru import logger

from hrp.api.platform import PlatformAPI


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    max_drawdown: float  # Maximum allowed drawdown (e.g., 0.25 for 25%)
    max_drawdown_duration_days: int  # Maximum days to recover from drawdown
    max_position_correlation: float  # Max correlation with existing positions
    max_sector_exposure: float  # Max allocation to any sector
    max_single_position: float  # Max allocation to single position
    min_diversification: int  # Minimum number of positions
    target_positions: int  # Target number of positions


@dataclass
class ImpactPreview:
    """Preview of risk limit changes on current hypotheses."""

    total_hypotheses: int
    hypotheses_passed: int
    hypotheses_vetoed: int
    veto_details: list[dict[str, Any]]


class RiskConfigAPI:
    """
    API for risk limit configuration and preview.

    Provides methods to:
    - Get current risk limits
    - Set new risk limits
    - Preview impact of limit changes without committing
    - Reset to defaults
    """

    # Default conservative limits
    DEFAULT_LIMITS = RiskLimits(
        max_drawdown=0.25,
        max_drawdown_duration_days=126,
        max_position_correlation=0.70,
        max_sector_exposure=0.30,
        max_single_position=0.10,
        min_diversification=10,
        target_positions=20,
    )

    def __init__(self, api: PlatformAPI):
        """
        Initialize RiskConfigAPI.

        Args:
            api: PlatformAPI instance for database access
        """
        self.api = api
        self._limits_cache: RiskLimits | None = None

    def get_limits(self, use_cache: bool = True) -> RiskLimits:
        """
        Get current risk limits from database or defaults.

        Args:
            use_cache: If True, return cached limits if available

        Returns:
            RiskLimits with current configuration
        """
        if use_cache and self._limits_cache is not None:
            return self._limits_cache

        try:
            # Try to get from database metadata
            limits_json = self.api._db.fetchone(
                """
                SELECT value FROM metadata
                WHERE key = 'risk_limits'
                """
            )

            if limits_json and limits_json[0]:
                limits_dict = json.loads(limits_json[0])
                limits = RiskLimits(**limits_dict)
                self._limits_cache = limits
                logger.info("Retrieved risk limits from database")
                return limits

        except Exception as e:
            logger.warning(f"Failed to retrieve risk limits from database: {e}")

        # Return defaults if not found
        logger.info("Using default risk limits")
        self._limits_cache = self.DEFAULT_LIMITS
        return self.DEFAULT_LIMITS

    def set_limits(self, limits: RiskLimits, actor: str = "dashboard") -> bool:
        """
        Set new risk limits in database.

        Args:
            limits: New RiskLimits configuration
            actor: Who is making the change (e.g., 'user', 'dashboard')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate limits
            self._validate_limits(limits)

            # Store in database
            limits_json = json.dumps(asdict(limits))

            self.api._db.execute(
                """
                INSERT INTO metadata (key, value, updated_at)
                VALUES ('risk_limits', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (limits_json, datetime.now()),
            )

            # Update cache
            self._limits_cache = limits

            logger.info(f"Risk limits updated by {actor}")
            return True

        except Exception as e:
            logger.error(f"Failed to set risk limits: {e}")
            return False

    def preview_impact(self, limits: RiskLimits) -> ImpactPreview:
        """
        Preview impact of risk limit changes without committing.

        Assesses all validated hypotheses against the new limits
        to show what would pass/fail.

        Args:
            limits: Proposed RiskLimits configuration

        Returns:
            ImpactPreview with assessment results
        """
        from hrp.agents.risk_manager import (
            RiskManager,
            RiskVeto,
            PortfolioRiskAssessment,
        )

        # Create RiskManager with proposed limits
        rm = RiskManager(
            hypothesis_ids=None,  # Assess all validated hypotheses
            max_drawdown=limits.max_drawdown,
            max_correlation=limits.max_position_correlation,
            max_sector_exposure=limits.max_sector_exposure,
            send_alerts=False,  # No alerts for preview
        )

        # Override other limits
        rm.MAX_MAX_DRAWDOWN = limits.max_drawdown
        rm.MAX_DRAWDOWN_DURATION_DAYS = limits.max_drawdown_duration_days
        rm.MAX_POSITION_CORRELATION = limits.max_position_correlation
        rm.MAX_SECTOR_EXPOSURE = limits.max_sector_exposure
        rm.MAX_SINGLE_POSITION = limits.max_single_position
        rm.MIN_DIVERSIFICATION = limits.min_diversification
        rm.TARGET_POSITIONS = limits.target_positions

        # Get hypotheses to assess
        hypotheses = rm._get_hypotheses_to_assess()

        if not hypotheses:
            return ImpactPreview(
                total_hypotheses=0,
                hypotheses_passed=0,
                hypotheses_vetoed=0,
                veto_details=[],
            )

        # Assess each hypothesis (without logging)
        passed_count = 0
        vetoed_count = 0
        veto_details: list[dict[str, Any]] = []

        for hypothesis in hypotheses:
            # Get metrics
            metadata_str = hypothesis.get("metadata") or "{}"
            metadata = (
                json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            )
            experiment_data = rm._get_experiment_metrics(hypothesis["hypothesis_id"], metadata)

            # Run checks
            vetos = []

            # Check drawdown
            dd_veto = rm._check_drawdown_risk(hypothesis["hypothesis_id"], experiment_data)
            if dd_veto:
                vetos.append(dd_veto)

            # Check concentration
            conc_vetos, _ = rm._check_concentration_risk(
                hypothesis["hypothesis_id"], experiment_data, metadata
            )
            vetos.extend(conc_vetos)

            # Check correlation
            corr_veto = rm._check_correlation_risk(hypothesis["hypothesis_id"], metadata)
            if corr_veto:
                vetos.append(corr_veto)

            # Check other limits
            limits_vetos = rm._check_risk_limits(hypothesis["hypothesis_id"], experiment_data)
            vetos.extend(limits_vetos)

            # Count
            critical_vetos = [v for v in vetos if v.severity == "critical"]
            if critical_vetos:
                vetoed_count += 1
                # Add to veto details (just the first critical veto)
                veto_details.append(
                    {
                        "hypothesis_id": hypothesis["hypothesis_id"],
                        "title": hypothesis.get("title", ""),
                        "veto_reason": critical_vetos[0].veto_reason,
                        "veto_type": critical_vetos[0].veto_type,
                    }
                )
            else:
                passed_count += 1

        return ImpactPreview(
            total_hypotheses=len(hypotheses),
            hypotheses_passed=passed_count,
            hypotheses_vetoed=vetoed_count,
            veto_details=veto_details,
        )

    def reset_to_defaults(self, actor: str = "dashboard") -> RiskLimits:
        """
        Reset risk limits to defaults.

        Args:
            actor: Who is resetting (e.g., 'user', 'dashboard')

        Returns:
            RiskLimits with default configuration
        """
        logger.info(f"Risk limits reset to defaults by {actor}")
        self.set_limits(self.DEFAULT_LIMITS, actor)
        return self.DEFAULT_LIMITS

    def _validate_limits(self, limits: RiskLimits) -> None:
        """
        Validate risk limits configuration.

        Args:
            limits: RiskLimits to validate

        Raises:
            ValueError: If limits are invalid
        """
        if limits.max_drawdown <= 0 or limits.max_drawdown > 1.0:
            raise ValueError("max_drawdown must be between 0 and 1.0")

        if limits.max_drawdown_duration_days <= 0:
            raise ValueError("max_drawdown_duration_days must be positive")

        if limits.max_position_correlation < 0 or limits.max_position_correlation > 1.0:
            raise ValueError("max_position_correlation must be between 0 and 1.0")

        if limits.max_sector_exposure <= 0 or limits.max_sector_exposure > 1.0:
            raise ValueError("max_sector_exposure must be between 0 and 1.0")

        if limits.max_single_position <= 0 or limits.max_single_position > 1.0:
            raise ValueError("max_single_position must be between 0 and 1.0")

        if limits.min_diversification < 1:
            raise ValueError("min_diversification must be at least 1")

        if limits.target_positions < limits.min_diversification:
            raise ValueError("target_positions must be >= min_diversification")

        if limits.max_single_position > limits.max_sector_exposure:
            raise ValueError("max_single_position cannot exceed max_sector_exposure")
