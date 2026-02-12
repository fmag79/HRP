"""Position sizing based on VaR/CVaR risk limits.

This module provides intelligent position sizing that respects risk budgets
and VaR thresholds. It integrates with TASK-006 VaR features
to size positions proportionally to their risk contribution.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, Any, List
import math

from hrp.data.risk.var_calculator import VaRCalculator, VaRConfig
from hrp.data.risk.risk_config import VaRMethod

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""

    max_position_var: Decimal = Decimal("0.02")  # 2% of portfolio per position
    max_portfolio_var: Decimal = Decimal("0.05")  # 5% total portfolio VaR
    min_position_size: Decimal = Decimal("1")  # Minimum 1 share
    max_position_size: Decimal = Decimal("0.20")  # Max 20% in one position
    risk_target: Decimal = Decimal("0.01")  # Target 1% VaR per position
    confidence_level: float = 0.95
    time_horizon: int = 1  # 1-day VaR
    var_method: VaRMethod = VaRMethod.PARAMETRIC
    use_cvar: bool = False  # Use CVaR for more conservative sizing

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_position_var <= 0:
            raise ValueError("max_position_var must be positive")
        if self.max_portfolio_var <= 0:
            raise ValueError("max_portfolio_var must be positive")
        if self.min_position_size <= 0:
            raise ValueError("min_position_size must be positive")
        if self.max_position_size > 1:
            raise ValueError("max_position_size cannot exceed 1.0 (100%)")
        if self.max_position_var > self.max_portfolio_var:
            raise ValueError("max_position_var cannot exceed max_portfolio_var")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    recommended_size: int
    var_at_size: Optional[Decimal] = None
    cvar_at_size: Optional[Decimal] = None
    var_ratio: Optional[Decimal] = None  # % of portfolio Var budget
    position_value: Optional[Decimal] = None
    portfolio_var_used: Optional[Decimal] = None
    limit_reason: Optional[str] = None  # If size was limited


class VaRPositionSizer:
    """Position sizing engine based on VaR/CVaR.

    Calculates optimal position sizes that respect risk budgets
    and allocate VaR capacity across positions proportionally.
    """

    def __init__(self, config: PositionSizingConfig) -> None:
        """Initialize position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config
        self._var_cache: Dict[str, Dict[str, Decimal]] = {}

        # Initialize VaR calculator
        self._var_config = VaRConfig(
            confidence_level=config.confidence_level,
            time_horizon=config.time_horizon,
            method=config.var_method,
        )
        self._var_calculator = VaRCalculator(self._var_config)

        logger.info(
            f"VaR Position Sizer initialized "
            f"(max_pos_var={config.max_position_var}, "
            f"max_port_var={config.max_portfolio_var}, "
            f"use_cvar={config.use_cvar})"
        )

    def max_position_size(
        self,
        price: Decimal,
        portfolio_value: Decimal,
        current_position_var: Optional[Decimal] = None,
    ) -> int:
        """Calculate maximum position size based on VaR limits.

        Args:
            price: Current share price
            portfolio_value: Total portfolio value
            current_position_var: Current position's VaR contribution (optional)

        Returns:
            Maximum allowed position size in shares

        Raises:
            ValueError: If parameters are invalid
        """
        if price <= 0:
            raise ValueError("price must be positive")

        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")

        # Calculate maximum position value based on VaR limit
        max_var_value = portfolio_value * self.config.max_position_var

        # If we know the VaR at 1 share, we can be precise
        # Otherwise, fall back to percentage-based sizing
        if current_position_var is None:
            # Conservative estimate: use max position size % directly
            max_value = portfolio_value * self.config.max_position_size
            max_size = int(max_value // price)
        else:
            # VaR at 1 share = current_position_var / current_size
            # But we don't have current_size, so estimate
            # Assume VaR scales roughly linearly with size
            max_size = int(max_var_value // current_position_var)

        # Apply minimum
        max_size = max(max_size, int(self.config.min_position_size))

        logger.debug(
            f"Max position size for price ${price:.2f}: {max_size} shares"
        )

        return max_size

    def adjust_for_var(
        self,
        proposed_size: int,
        price: Decimal,
        returns: List[float],
        portfolio_value: Decimal,
        existing_positions_var: Optional[Decimal] = None,
    ) -> PositionSizeResult:
        """Adjust proposed position size to respect VaR limits.

        Args:
            proposed_size: Desired position size in shares
            price: Current share price
            returns: List of historical returns for VaR calculation
            portfolio_value: Total portfolio value
            existing_positions_var: VaR of existing positions (optional)

        Returns:
            PositionSizeResult with adjusted size

        Raises:
            ValueError: If parameters are invalid
        """
        if proposed_size <= 0:
            raise ValueError("proposed_size must be positive")

        if price <= 0:
            raise ValueError("price must be positive")

        if not returns or len(returns) < 60:
            logger.warning("Insufficient return data for VaR adjustment")
            # Fall back to simple percentage-based sizing
            return self._percentage_based_sizing(
                proposed_size, price, portfolio_value
            )

        # Calculate VaR for proposed position
        try:
            var_result = self._var_calculator.calculate(returns, self._var_config)
            var_at_size = var_result.var
            cvar_at_size = var_result.cvar if self.config.use_cvar else None

            # Use CVaR if enabled (more conservative)
            var_to_check = cvar_at_size if self.config.use_cvar else var_at_size

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            var_at_size = cvar_at_size = var_to_check = None

        if var_to_check is None:
            # Fall back to simple sizing
            return self._percentage_based_sizing(
                proposed_size, price, portfolio_value
            )

        # Calculate VaR budget for this position
        position_var_budget = portfolio_value * self.config.max_position_var

        # Check if exceeds VaR limit
        if var_to_check > position_var_budget:
            # Calculate size that fits within budget
            # Assume VaR scales roughly linearly with size
            adjustment_factor = position_var_budget / var_to_check
            adjusted_size = max(
                int(proposed_size * adjustment_factor),
                int(self.config.min_position_size),
            )

            position_value = adjusted_size * price
            var_ratio = self.config.max_position_var

            logger.info(
                f"Position size adjusted for VaR: {proposed_size} -> {adjusted_size} "
                f"(var={var_to_check:.4f} > budget={position_var_budget:.4f})"
            )

            return PositionSizeResult(
                recommended_size=adjusted_size,
                var_at_size=Decimal(str(var_to_check * adjustment_factor)),
                cvar_at_size=Decimal(str(cvar_at_size * adjustment_factor))
                if cvar_at_size
                else None,
                var_ratio=var_ratio,
                position_value=position_value,
                portfolio_var_used=Decimal(str(var_to_check * adjustment_factor)),
                limit_reason="VaR limit exceeded",
            )

        # Check against portfolio-level VaR budget
        if existing_positions_var is not None:
            portfolio_var_budget = portfolio_value * self.config.max_portfolio_var
            total_var = existing_positions_var + var_to_check

            if total_var > portfolio_var_budget:
                # Need to reduce to fit within portfolio budget
                available_var_budget = portfolio_var_budget - existing_positions_var
                adjustment_factor = available_var_budget / var_to_check
                adjusted_size = max(
                    int(proposed_size * adjustment_factor),
                    int(self.config.min_position_size),
                )

                position_value = adjusted_size * price
                var_ratio = (var_to_check * adjustment_factor) / portfolio_value

                logger.info(
                    f"Position size adjusted for portfolio VaR: "
                    f"{proposed_size} -> {adjusted_size} "
                    f"(total_var={total_var:.4f} > budget={portfolio_var_budget:.4f})"
                )

                return PositionSizeResult(
                    recommended_size=adjusted_size,
                    var_at_size=Decimal(str(var_to_check * adjustment_factor)),
                    cvar_at_size=Decimal(str(cvar_at_size * adjustment_factor))
                    if cvar_at_size
                    else None,
                    var_ratio=var_ratio,
                    position_value=position_value,
                    portfolio_var_used=Decimal(str(
                        existing_positions_var + var_to_check * adjustment_factor
                    )),
                    limit_reason="Portfolio VaR budget exceeded",
                )

        # No adjustment needed
        position_value = proposed_size * price
        var_ratio = var_to_check / portfolio_value

        return PositionSizeResult(
            recommended_size=proposed_size,
            var_at_size=Decimal(str(var_at_size)),
            cvar_at_size=Decimal(str(cvar_at_size)) if cvar_at_size else None,
            var_ratio=var_ratio,
            position_value=position_value,
            portfolio_var_used=existing_positions_var + var_to_check
            if existing_positions_var
            else var_to_check,
        )

    def portfolio_var_budget(
        self,
        portfolio_value: Decimal,
        existing_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Decimal]:
        """Calculate portfolio VaR budget allocation.

        Args:
            portfolio_value: Total portfolio value
            existing_positions: List of position dicts with 'var' field (optional)

        Returns:
            Dictionary with budget information:
            - total_budget: Total VaR budget
            - used_budget: VaR used by existing positions
            - available_budget: Remaining VaR budget
            - utilization: % of budget used
        """
        total_budget = portfolio_value * self.config.max_portfolio_var

        used_budget = Decimal("0")
        if existing_positions:
            for pos in existing_positions:
                if "var" in pos and pos["var"] is not None:
                    used_budget += Decimal(str(pos["var"]))

        available_budget = total_budget - used_budget
        utilization = (
            (used_budget / total_budget) if total_budget > 0 else Decimal("0")
        )

        budget_info = {
            "total_budget": total_budget,
            "used_budget": used_budget,
            "available_budget": max(available_budget, Decimal("0")),
            "utilization": utilization,
        }

        logger.debug(
            f"Portfolio VaR budget: ${used_budget:.2f} / ${total_budget:.2f} "
            f"({utilization*100:.1f}%)"
        )

        return budget_info

    def calculate_optimal_size(
        self,
        price: Decimal,
        returns: List[float],
        portfolio_value: Decimal,
        existing_positions: Optional[List[Dict[str, Any]]] = None,
        target_var_ratio: Optional[Decimal] = None,
    ) -> PositionSizeResult:
        """Calculate optimal position size balancing risk and return.

        Args:
            price: Current share price
            returns: List of historical returns
            portfolio_value: Total portfolio value
            existing_positions: List of existing positions (optional)
            target_var_ratio: Target VaR as % of portfolio (optional)

        Returns:
            PositionSizeResult with recommended size

        Raises:
            ValueError: If parameters are invalid
        """
        if price <= 0 or portfolio_value <= 0:
            raise ValueError("price and portfolio_value must be positive")

        if not returns or len(returns) < 60:
            logger.warning("Insufficient return data, using fallback sizing")
            return self._fallback_sizing(price, portfolio_value)

        # Use target VaR ratio if provided, else use config
        var_target = target_var_ratio or self.config.risk_target

        # Calculate VaR per share
        try:
            var_result = self._var_calculator.calculate(
                [r for r in returns if not math.isnan(r)],
                self._var_config,
            )
            var_per_share = var_result.var
            cvar_per_share = var_result.cvar

            # Use CVaR if enabled
            var_to_use = cvar_per_share if self.config.use_cvar else var_per_share

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return self._fallback_sizing(price, portfolio_value)

        # Calculate position size to meet target VaR
        target_var_value = portfolio_value * var_target

        # Estimate shares needed to reach target VaR
        # VaR scales roughly linearly with size
        if var_to_use > 0:
            target_size = int(target_var_value // var_to_use)
        else:
            # No risk data, use fallback
            target_size = self._fallback_sizing_size(price, portfolio_value)

        # Get budget info
        budget = self.portfolio_var_budget(portfolio_value, existing_positions)

        # Ensure we have budget available
        if var_to_use * target_size > budget["available_budget"]:
            # Adjust to fit in available budget
            target_size = max(
                int(budget["available_budget"] // var_to_use),
                int(self.config.min_position_size),
            )

        # Apply maximum position size limit
        max_size = self.max_position_size(price, portfolio_value)
        final_size = min(target_size, max_size)

        # Apply minimum
        final_size = max(final_size, int(self.config.min_position_size))

        position_value = final_size * price
        var_at_size = var_to_use * final_size

        return PositionSizeResult(
            recommended_size=final_size,
            var_at_size=Decimal(str(var_at_size)),
            cvar_at_size=Decimal(str(cvar_per_share * final_size))
            if self.config.use_cvar
            else None,
            var_ratio=var_at_size / portfolio_value,
            position_value=position_value,
            portfolio_var_used=budget["used_budget"] + var_at_size,
            limit_reason="Max position size limit" if final_size < target_size else None,
        )

    def _percentage_based_sizing(
        self,
        proposed_size: int,
        price: Decimal,
        portfolio_value: Decimal,
    ) -> PositionSizeResult:
        """Fallback to simple percentage-based sizing.

        Used when VaR calculation fails.
        """
        # Apply max position size limit
        max_value = portfolio_value * self.config.max_position_size
        max_size = int(max_value // price)

        adjusted_size = min(proposed_size, max_size)
        adjusted_size = max(adjusted_size, int(self.config.min_position_size))

        position_value = adjusted_size * price

        return PositionSizeResult(
            recommended_size=adjusted_size,
            position_value=position_value,
            var_ratio=Decimal(str(adjusted_size * price / portfolio_value)),
            limit_reason="Percentage-based fallback",
        )

    def _fallback_sizing(
        self, price: Decimal, portfolio_value: Decimal
    ) -> PositionSizeResult:
        """Fallback sizing when insufficient data.

        Uses simple percentage-based approach.
        """
        size = self._fallback_sizing_size(price, portfolio_value)
        position_value = size * price

        return PositionSizeResult(
            recommended_size=size,
            position_value=position_value,
            var_ratio=Decimal(str(size * price / portfolio_value)),
            limit_reason="Insufficient data for VaR",
        )

    def _fallback_sizing_size(
        self, price: Decimal, portfolio_value: Decimal
    ) -> int:
        """Calculate fallback position size.

        Uses target risk ratio or conservative default.
        """
        target_value = portfolio_value * self.config.risk_target
        size = int(target_value // price)

        # Apply limits
        max_value = portfolio_value * self.config.max_position_size
        max_size = int(max_value // price)

        size = min(size, max_size)
        size = max(size, int(self.config.min_position_size))

        return size

    def clear_cache(self) -> None:
        """Clear VaR calculation cache."""
        self._var_cache.clear()
        logger.debug("VaR position sizer cache cleared")
