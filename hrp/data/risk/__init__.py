"""
Risk metrics module for HRP.

Provides Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) calculations
using multiple methodologies: parametric, historical simulation, and Monte Carlo.
"""

from hrp.data.risk.risk_config import VaRConfig
from hrp.data.risk.var_calculator import VaRCalculator, VaRResult

__all__ = ["VaRConfig", "VaRCalculator", "VaRResult"]
