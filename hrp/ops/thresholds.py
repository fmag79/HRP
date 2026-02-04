"""Configurable alert thresholds for HRP ops.

Thresholds can be configured via:
1. Environment variables (HRP_THRESHOLD_*)
2. YAML config file (~~/hrp-data/config/thresholds.yaml)
3. Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from loguru import logger


@dataclass
class OpsThresholds:
    """Alert thresholds for monitoring."""

    # Health score thresholds (0-100)
    health_score_warning: float = 90.0
    health_score_critical: float = 70.0

    # Data freshness thresholds (days)
    freshness_warning_days: int = 3
    freshness_critical_days: int = 5

    # Anomaly thresholds
    anomaly_count_warning: int = 50
    anomaly_count_critical: int = 100

    # Drift thresholds
    kl_divergence_threshold: float = 0.2
    psi_threshold: float = 0.2
    ic_decay_threshold: float = 0.2

    # Ingestion thresholds (percentage)
    ingestion_success_rate_warning: float = 95.0
    ingestion_success_rate_critical: float = 80.0


def load_thresholds(config_path: str | None = None) -> OpsThresholds:
    """
    Load thresholds with priority: Environment > YAML > Defaults.

    Args:
        config_path: Path to YAML config. Defaults to ~/hrp-data/config/thresholds.yaml

    Returns:
        OpsThresholds instance
    """
    thresholds = OpsThresholds()

    # Load from YAML if exists
    if config_path is None:
        config_path = os.path.expanduser("~/hrp-data/config/thresholds.yaml")

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}

            for key, value in yaml_config.items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)

            logger.debug(f"Loaded thresholds from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load thresholds from {config_path}: {e}")

    # Apply environment variable overrides
    for field in fields(thresholds):
        env_key = f"HRP_THRESHOLD_{field.name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                # Get current value to determine type (handles string type annotations)
                current_value = getattr(thresholds, field.name)
                if isinstance(current_value, float):
                    setattr(thresholds, field.name, float(env_value))
                elif isinstance(current_value, int):
                    setattr(thresholds, field.name, int(env_value))
                else:
                    setattr(thresholds, field.name, env_value)
                logger.debug(f"Threshold override: {field.name}={env_value}")
            except ValueError as e:
                logger.warning(f"Invalid env value for {env_key}: {e}")

    return thresholds
