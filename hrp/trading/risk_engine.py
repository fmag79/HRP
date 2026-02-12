"""Risk engine for pre-trade risk checks.

This module provides risk management integration with TASK-006 VaR/CVaR
features. It performs pre-trade risk checks and position sizing based
on risk limits.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from hrp.data.features.risk_features import (
    compute_var_95_1d,
    compute_cvar_95_1d,
    compute_var_99_1d,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_portfolio_var: Decimal = Decimal("0.02")  # 2% of portfolio value
    max_single_position_var: Decimal = Decimal("0.005")  # 0.5% of portfolio value
    max_single_position_size: Decimal = Decimal("0.10")  # 10% of portfolio value
    var_confidence: float = 0.95
    use_cvar: bool = True  # Use CVaR for more conservative limits
    enable_hard_stops: bool = True  # Block orders exceeding risk limits

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_portfolio_var <= 0:
            raise ValueError("max_portfolio_var must be positive")
        if self.max_single_position_var <= 0:
            raise ValueError("max_single_position_var must be positive")
        if self.max_single_position_size <= 0:
            raise ValueError("max_single_position_size must be positive")
        if self.max_single_position_size > 1:
            raise ValueError("max_single_position_size cannot exceed 1.0 (100%)")


@dataclass
class RiskResult:
    """Result of risk check."""

    passed: bool
    var_95: Optional[Decimal] = None
    cvar_95: Optional[Decimal] = None
    var_99: Optional[Decimal] = None
    portfolio_value: Optional[Decimal] = None
    position_value: Optional[Decimal] = None
    max_allowed_size: Optional[int] = None
    reason: Optional[str] = None


class RiskEngine:
    """Risk engine for pre-trade risk checks.

    Integrates with TASK-006 VaR/CVaR features to perform
    risk-based order validation and position sizing.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialize risk engine.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Risk engine initialized (max_var={config.max_portfolio_var}, "
            f"max_pos_var={config.max_single_position_var}, "
            f"use_cvar={config.use_cvar})"
        )

    def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        portfolio_value: Decimal,
        current_position: Optional[int] = None,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> RiskResult:
        """Perform pre-trade risk check on order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Current price
            portfolio_value: Total portfolio value
            current_position: Current position size (optional)
            prices_df: Historical prices DataFrame for VaR calculation

        Returns:
            RiskResult with check outcome
        """
        logger.debug(
            f"Risk check: {side.upper()} {quantity} {symbol} @ {price} "
            f"(portfolio=${portfolio_value:,.2f})"
        )

        # Calculate position value after order
        if current_position is None:
            current_position = 0

        if side.lower() == "buy":
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity

        position_value = new_position * price

        # Check single position size limit
        max_pos_value = portfolio_value * self.config.max_single_position_size

        if position_value > max_pos_value:
            return RiskResult(
                passed=False,
                portfolio_value=portfolio_value,
                position_value=position_value,
                max_allowed_size=int(max_pos_value // price),
                reason=f"Position exceeds maximum size ({self.config.max_single_position_size*100:.1f}% of portfolio)",
            )

        # Calculate VaR/CVaR if prices available
        if prices_df is not None:
            try:
                var_95, cvar_95, var_99 = self._calculate_var(
                    symbol, prices_df
                )
            except Exception as e:
                logger.warning(f"VaR calculation failed for {symbol}: {e}")
                var_95 = cvar_95 = var_99 = None
        else:
            var_95 = cvar_95 = var_99 = None

        # Check VaR limit
        if var_95 is not None:
            # Use CVaR if enabled (more conservative)
            var_to_check = cvar_95 if self.config.use_cvar else var_95
            var_limit = portfolio_value * self.config.max_single_position_var

            # VaR is expressed as potential loss (positive number)
            # Compare against limit
            if var_to_check > var_limit:
                return RiskResult(
                    passed=False,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    var_99=var_99,
                    portfolio_value=portfolio_value,
                    position_value=position_value,
                    max_allowed_size=int(var_limit // price),
                    reason=f"VaR/CVaR exceeds limit (var={var_to_check:.2f}, limit={var_limit:.2f})",
                )

        # Check portfolio-level VaR
        # This would require calculating VaR for the entire portfolio
        # For now, we'll do a simple check:
        # New position VaR should not exceed X% of portfolio
        if var_95 is not None:
            var_to_check = cvar_95 if self.config.use_cvar else var_95
            portfolio_var_limit = portfolio_value * self.config.max_portfolio_var

            if var_to_check > portfolio_var_limit:
                return RiskResult(
                    passed=False,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    var_99=var_99,
                    portfolio_value=portfolio_value,
                    position_value=position_value,
                    max_allowed_size=int(portfolio_var_limit // price),
                    reason=f"Position VaR exceeds portfolio limit (var={var_to_check:.2f}, limit={portfolio_var_limit:.2f})",
                )

        # All checks passed
        return RiskResult(
            passed=True,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            portfolio_value=portfolio_value,
            position_value=position_value,
        )

    def calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        portfolio_value: Decimal,
        current_position: Optional[int] = None,
        prices_df: Optional[pd.DataFrame] = None,
        risk_target: Optional[Decimal] = None,
    ) -> int:
        """Calculate optimal position size based on risk limits.

        Args:
            symbol: Stock symbol
            price: Current price
            portfolio_value: Total portfolio value
            current_position: Current position size (optional)
            prices_df: Historical prices DataFrame for VaR calculation
            risk_target: Target risk as % of portfolio (optional)

        Returns:
            Recommended position size

        Raises:
            ValueError: If parameters are invalid
        """
        if price <= 0:
            raise ValueError("price must be positive")

        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")

        # Default to configured single position size limit
        if risk_target is None:
            risk_target = self.config.max_single_position_size

        # Calculate size based on risk target
        target_value = portfolio_value * risk_target
        target_size = int(target_value // price)

        logger.info(
            f"Position sizing: {symbol} - "
            f"target_value=${target_value:,.2f}, target_size={target_size}"
        )

        # Adjust for current position
        if current_position is not None and prices_df is not None:
            # Calculate VaR for proposed new position
            try:
                if current_position < target_size:
                    # Adding to position
                    new_size = target_size
                else:
                    # Reducing position
                    new_size = target_size

                # Validate with risk check
                risk_result = self.check_order_risk(
                    symbol=symbol,
                    side="buy" if new_size > current_position else "sell",
                    quantity=abs(new_size - current_position),
                    price=price,
                    portfolio_value=portfolio_value,
                    current_position=current_position,
                    prices_df=prices_df,
                )

                if not risk_result.passed:
                    if risk_result.max_allowed_size is not None:
                        target_size = risk_result.max_allowed_size
                        logger.warning(
                            f"Risk limit exceeded, adjusted size to {target_size}"
                        )

            except Exception as e:
                logger.warning(f"Risk-based sizing failed: {e}")

        return target_size

    def _calculate_var(
        self, symbol: str, prices_df: pd.DataFrame
    ) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate VaR and CVaR for symbol.

        Args:
            symbol: Stock symbol
            prices_df: Historical prices DataFrame

        Returns:
            Tuple of (var_95, cvar_95, var_99) or (None, None, None)
        """
        # Check cache
        cache_key = f"{symbol}_{prices_df.index.max().isoformat()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return (
                cached["var_95"],
                cached["cvar_95"],
                cached["var_99"],
            )

        try:
            # Filter prices for symbol
            if isinstance(prices_df.index, pd.MultiIndex):
                # MultiIndex with (date, symbol)
                symbol_prices = prices_df.xs(symbol, level="symbol")
            else:
                # Already filtered
                symbol_prices = prices_df

            if symbol_prices.empty or len(symbol_prices) < 60:
                logger.warning(f"Insufficient price data for {symbol}")
                return None, None, None

            # Calculate VaR and CVaR
            var_95_df = compute_var_95_1d(symbol_prices)
            cvar_95_df = compute_cvar_95_1d(symbol_prices)
            var_99_df = compute_var_99_1d(symbol_prices)

            # Get latest values
            var_95 = Decimal(str(var_95_df.iloc[-1, 0]))
            cvar_95 = Decimal(str(cvar_95_df.iloc[-1, 0]))
            var_99 = Decimal(str(var_99_df.iloc[-1, 0]))

            # Cache results
            self._cache[cache_key] = {
                "var_95": var_95,
                "cvar_95": cvar_95,
                "var_99": var_99,
            }

            logger.debug(
                f"VaR for {symbol}: 95%={var_95:.4f}, "
                f"CVaR={cvar_95:.4f}, 99%={var_99:.4f}"
            )

            return var_95, cvar_95, var_99

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return None, None, None

    def clear_cache(self) -> None:
        """Clear VaR calculation cache."""
        self._cache.clear()
        logger.debug("VaR cache cleared")

    def get_risk_summary(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get portfolio risk summary.

        Args:
            positions: List of position dictionaries

        Returns:
            Risk summary dictionary
        """
        total_var_95 = Decimal("0")
        total_cvar_95 = Decimal("0")
        total_value = Decimal("0")

        for pos in positions:
            if "var_95" in pos:
                total_var_95 += Decimal(str(pos["var_95"]))
            if "cvar_95" in pos:
                total_cvar_95 += Decimal(str(pos["cvar_95"]))
            if "value" in pos:
                total_value += Decimal(str(pos["value"]))

        # Use CVaR if enabled
        total_risk = total_cvar_95 if self.config.use_cvar else total_var_95

        risk_ratio = (
            (total_risk / total_value) if total_value > 0 else Decimal("0")
        )

        return {
            "total_value": float(total_value),
            "total_var_95": float(total_var_95),
            "total_cvar_95": float(total_cvar_95),
            "risk_ratio": float(risk_ratio),
            "max_portfolio_var": float(self.config.max_portfolio_var),
            "risk_limit_exceeded": risk_ratio > self.config.max_portfolio_var,
            "positions_count": len(positions),
        }
