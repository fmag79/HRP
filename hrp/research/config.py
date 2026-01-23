"""
Configuration classes for backtesting and research.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Any

import pandas as pd


@dataclass
class CostModel:
    """Transaction cost model (IBKR realistic)."""

    commission_per_share: float = 0.005
    commission_min: float = 1.00
    commission_max_pct: float = 0.01
    spread_bps: float = 5.0
    slippage_bps: float = 5.0

    def total_cost_pct(self) -> float:
        """Total estimated cost as percentage (one-way)."""
        return (self.spread_bps + self.slippage_bps) / 10000


@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""

    # Universe
    symbols: list[str] = field(default_factory=list)
    start_date: date = None
    end_date: date = None

    # Position sizing
    sizing_method: str = "equal"  # equal, volatility, signal_scaled
    max_position_pct: float = 0.10
    max_positions: int = 20
    min_position_pct: float = 0.02

    # Costs
    costs: CostModel = field(default_factory=CostModel)

    # Validation splits
    train_end: date = None
    test_start: date = None

    # Metadata
    name: str = ""
    description: str = ""

    # Return type
    total_return: bool = False  # If True, include dividend reinvestment

    def __post_init__(self):
        if self.costs is None:
            self.costs = CostModel()


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    metrics: dict[str, float]
    equity_curve: pd.Series
    trades: pd.DataFrame
    benchmark_metrics: dict[str, float] = None

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", 0.0)
