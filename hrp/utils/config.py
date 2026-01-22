"""
Configuration management for HRP.

Loads settings from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


@dataclass
class DataConfig:
    """Data layer configuration."""

    data_dir: Path = field(default_factory=lambda: Path.home() / "hrp-data")
    db_name: str = "hrp.duckdb"

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    @property
    def mlflow_dir(self) -> Path:
        return self.data_dir / "mlflow"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@dataclass
class APIConfig:
    """API keys and external service configuration."""

    polygon_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    tiingo_api_key: str | None = None
    anthropic_api_key: str | None = None
    resend_api_key: str | None = None


@dataclass
class BacktestConfig:
    """Default backtest configuration."""

    # Position sizing
    max_position_pct: float = 0.10
    max_positions: int = 20
    min_position_pct: float = 0.02

    # Costs (IBKR realistic)
    commission_pct: float = 0.0005
    slippage_pct: float = 0.001

    # Risk limits
    max_gross_exposure: float = 1.0
    strategy_stop_loss: float = 0.15
    portfolio_stop_loss: float = 0.20


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # General settings
    log_level: str = "INFO"
    notification_email: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        data_dir = os.getenv("HRP_DATA_DIR")

        return cls(
            data=DataConfig(
                data_dir=Path(data_dir).expanduser() if data_dir else Path.home() / "hrp-data",
            ),
            api=APIConfig(
                polygon_api_key=os.getenv("POLYGON_API_KEY"),
                alpaca_api_key=os.getenv("ALPACA_API_KEY"),
                alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
                tiingo_api_key=os.getenv("TIINGO_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                resend_api_key=os.getenv("RESEND_API_KEY"),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            notification_email=os.getenv("NOTIFICATION_EMAIL"),
        )


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
