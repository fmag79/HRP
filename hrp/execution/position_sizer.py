"""VaR/CVaR-aware position sizing engine.

This module provides position sizing logic that respects portfolio-level
Value-at-Risk (VaR) and Conditional VaR constraints. It integrates with the
risk features calculated in TASK-006 to ensure positions stay within risk limits.
"""

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from hrp.api.platform import PlatformAPI
from hrp.data.risk.risk_config import Distribution, VaRConfig, VaRMethod
from hrp.data.risk.var_calculator import VaRCalculator

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """VaR-aware position sizing configuration."""

    portfolio_value: Decimal
    max_portfolio_var_pct: float = 0.02  # 2% daily portfolio VaR limit
    max_position_var_pct: float = 0.005  # 0.5% daily VaR per position
    confidence_level: float = 0.95
    use_cvar: bool = False  # Use CVaR for more conservative sizing
    min_position_value: Decimal = Decimal("100.00")
    max_position_pct: float = 0.10  # Hard cap: 10% per position
    lookback_days: int = 252  # VaR calculation lookback
    fallback_var_pct: float = 0.02  # Fallback VaR if no data (2% daily)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_portfolio_var_pct <= 0 or self.max_portfolio_var_pct > 1:
            raise ValueError("max_portfolio_var_pct must be between 0 and 1")
        if self.max_position_var_pct <= 0 or self.max_position_var_pct > 1:
            raise ValueError("max_position_var_pct must be between 0 and 1")
        if self.max_position_pct <= 0 or self.max_position_pct > 1:
            raise ValueError("max_position_pct must be between 0 and 1")
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")


class PositionSizer:
    """VaR/CVaR-aware position sizing engine.

    This class calculates position sizes that respect portfolio-level VaR
    constraints. It reads VaR/CVaR features from the feature store and
    allocates position sizes based on signal strength and risk budgets.

    Example:
        >>> config = PositionSizingConfig(
        ...     portfolio_value=Decimal("100000"),
        ...     max_portfolio_var_pct=0.02,
        ...     max_position_var_pct=0.005
        ... )
        >>> api = PlatformAPI()
        >>> sizer = PositionSizer(config, api)
        >>> positions = sizer.size_all_positions(signals, current_positions)
    """

    def __init__(self, config: PositionSizingConfig, api: PlatformAPI) -> None:
        """Initialize position sizer.

        Args:
            config: Position sizing configuration
            api: Platform API for accessing features and prices
        """
        self.config = config
        self.api = api
        self._var_calculator: VaRCalculator | None = None

        logger.info(
            f"Initialized PositionSizer: portfolio=${config.portfolio_value:,.2f}, "
            f"max_portfolio_var={config.max_portfolio_var_pct:.2%}, "
            f"max_position_var={config.max_position_var_pct:.2%}"
        )

    def _get_var_calculator(self) -> VaRCalculator:
        """Get or create VaR calculator for fallback calculations."""
        if self._var_calculator is None:
            var_config = VaRConfig(
                confidence_level=self.config.confidence_level,
                time_horizon=1,
                method=VaRMethod.PARAMETRIC,
                distribution=Distribution.NORMAL,
            )
            self._var_calculator = VaRCalculator(var_config)
        return self._var_calculator

    def _get_symbol_var(
        self,
        symbol: str,
        as_of_date: date,
        price: Decimal,
    ) -> float:
        """Get VaR for a symbol from feature store.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to get VaR for
            price: Current price for fallback calculation

        Returns:
            VaR as a decimal (e.g., 0.02 for 2% daily VaR)
        """
        # Determine which feature to use
        feature_name = "cvar_95_1d" if self.config.use_cvar else "var_95_1d"

        try:
            # Get VaR feature from feature store
            features_df = self.api.get_features(
                symbols=[symbol],
                features=[feature_name],
                as_of_date=as_of_date,
                version="v1",
            )

            if not features_df.empty and feature_name in features_df.columns:
                var_value = features_df[feature_name].iloc[0]

                if pd.notna(var_value) and var_value > 0:
                    logger.debug(f"{symbol}: {feature_name}={var_value:.4f}")
                    return float(var_value)

            logger.warning(
                f"{symbol}: No {feature_name} feature found for {as_of_date}, "
                f"using fallback VaR={self.config.fallback_var_pct:.2%}"
            )

        except Exception as e:
            logger.warning(
                f"{symbol}: Error retrieving {feature_name}: {e}, "
                f"using fallback VaR={self.config.fallback_var_pct:.2%}"
            )

        # Fallback: use configured fallback VaR
        return self.config.fallback_var_pct

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: Decimal,
        as_of_date: date,
        current_portfolio_var: float | None = None,
    ) -> int:
        """Calculate position size constrained by VaR limits.

        Algorithm:
        1. Get symbol's VaR from feature store (or use fallback)
        2. Calculate max $ allocation: max_position_var / symbol_var
        3. Apply signal-strength scaling (stronger signal → larger position)
        4. Apply hard caps (max_position_pct, portfolio-level VaR budget)
        5. Convert to shares using current price

        Args:
            symbol: Ticker symbol
            signal_strength: Signal strength (0.0 to 1.0), where 1.0 is strongest
            current_price: Current price per share
            as_of_date: Date for VaR feature lookup
            current_portfolio_var: Current portfolio VaR (used for budget check)

        Returns:
            Position size in shares (0 if constraints violated)
        """
        if signal_strength <= 0:
            logger.debug(f"{symbol}: signal_strength={signal_strength} <= 0, size=0")
            return 0

        if current_price <= 0:
            logger.warning(f"{symbol}: invalid price={current_price}, size=0")
            return 0

        # Get symbol VaR
        symbol_var = self._get_symbol_var(symbol, as_of_date, current_price)

        if symbol_var <= 0:
            logger.warning(f"{symbol}: VaR={symbol_var} <= 0, using fallback")
            symbol_var = self.config.fallback_var_pct

        # Calculate max position value based on VaR constraint
        # max_position_var_pct / symbol_var = max position size as % of portfolio
        # Example: 0.005 / 0.02 = 0.25 = 25% of portfolio
        max_var_position_pct = self.config.max_position_var_pct / symbol_var

        # Apply hard cap
        max_position_pct = min(max_var_position_pct, self.config.max_position_pct)

        # Scale by signal strength
        # Strong signals (0.8-1.0) get closer to max, weak signals (0.1-0.3) get less
        scaled_position_pct = max_position_pct * signal_strength

        # Calculate position value
        position_value = self.config.portfolio_value * Decimal(str(scaled_position_pct))

        # Check minimum
        if position_value < self.config.min_position_value:
            logger.debug(
                f"{symbol}: position_value={position_value:.2f} "
                f"below minimum={self.config.min_position_value:.2f}, size=0"
            )
            return 0

        # Convert to shares
        quantity = int(position_value / current_price)

        if quantity <= 0:
            logger.debug(f"{symbol}: calculated quantity={quantity} <= 0, size=0")
            return 0

        # Log sizing decision
        actual_value = Decimal(quantity) * current_price
        actual_var = float(actual_value / self.config.portfolio_value) * symbol_var

        logger.debug(
            f"{symbol}: size={quantity} shares (value=${actual_value:,.2f}, "
            f"{100*float(actual_value/self.config.portfolio_value):.2f}% of portfolio, "
            f"signal={signal_strength:.3f}, VaR={symbol_var:.4f}, "
            f"position_var={actual_var:.4f})"
        )

        return quantity

    def calculate_portfolio_var_budget(
        self,
        current_positions: dict[str, int],
        current_prices: dict[str, Decimal],
        as_of_date: date,
    ) -> float:
        """Calculate remaining VaR budget for new positions.

        Args:
            current_positions: Dict of symbol -> current quantity
            current_prices: Dict of symbol -> current price
            as_of_date: Date for VaR feature lookup

        Returns:
            VaR capacity available (portfolio_limit - current_var)
        """
        if not current_positions:
            logger.debug(
                f"No current positions, full budget available: "
                f"{self.config.max_portfolio_var_pct:.2%}"
            )
            return self.config.max_portfolio_var_pct

        # Calculate VaR contribution from each position
        total_var = 0.0

        for symbol, quantity in current_positions.items():
            if quantity <= 0:
                continue

            # Get price
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(f"{symbol}: missing/invalid price, skipping VaR calculation")
                continue

            # Get VaR
            symbol_var = self._get_symbol_var(symbol, as_of_date, price)

            # Calculate position value and VaR contribution
            position_value = Decimal(quantity) * price
            position_pct = float(position_value / self.config.portfolio_value)
            position_var = position_pct * symbol_var

            total_var += position_var

            logger.debug(
                f"{symbol}: {quantity} shares @ ${price:.2f} = ${position_value:,.2f} "
                f"({position_pct:.2%} of portfolio), VaR={symbol_var:.4f}, "
                f"contribution={position_var:.4f}"
            )

        # Calculate remaining budget
        remaining = self.config.max_portfolio_var_pct - total_var

        logger.info(
            f"Portfolio VaR: current={total_var:.4f}, "
            f"limit={self.config.max_portfolio_var_pct:.4f}, "
            f"remaining={remaining:.4f}"
        )

        return remaining

    def size_all_positions(
        self,
        signals: pd.DataFrame,
        current_positions: dict[str, int],
        current_prices: dict[str, Decimal],
        as_of_date: date,
    ) -> dict[str, int]:
        """Size all positions from signals, respecting portfolio VaR limit.

        Allocates VaR budget across signals, strongest first.

        Args:
            signals: DataFrame with columns [symbol, signal, prediction]
                    signal: 1.0 (long), 0.0 (no position), -1.0 (short, not supported)
                    prediction: signal strength (higher = more confident)
            current_positions: Dict of symbol -> current quantity held
            current_prices: Dict of symbol -> current price
            as_of_date: Date for VaR feature lookup

        Returns:
            Dict of symbol -> target quantity (0 to exit position)
        """
        # Filter for buy signals
        buy_signals = signals[signals["signal"] == 1.0].copy()

        if buy_signals.empty:
            logger.info("No buy signals to size")
            return {}

        # Sort by prediction strength (strongest first)
        buy_signals = buy_signals.sort_values("prediction", ascending=False)

        # Normalize predictions to [0, 1] range for signal strength
        if len(buy_signals) > 1:
            min_pred = buy_signals["prediction"].min()
            max_pred = buy_signals["prediction"].max()

            if max_pred > min_pred:
                buy_signals["signal_strength"] = (
                    buy_signals["prediction"] - min_pred
                ) / (max_pred - min_pred)
            else:
                # All predictions equal
                buy_signals["signal_strength"] = 1.0
        else:
            buy_signals["signal_strength"] = 1.0

        # Calculate current portfolio VaR budget
        var_budget_remaining = self.calculate_portfolio_var_budget(
            current_positions, current_prices, as_of_date
        )

        # Size each position
        target_positions = {}
        var_used = 0.0

        for _, row in buy_signals.iterrows():
            symbol = row["symbol"]
            signal_strength = row["signal_strength"]

            # Skip if already at position limit
            if len(target_positions) >= self.config.max_position_pct * 100:
                logger.debug(f"{symbol}: position limit reached, skipping")
                continue

            # Get price
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(f"{symbol}: missing/invalid price, skipping")
                continue

            # Calculate position size
            quantity = self.calculate_position_size(
                symbol=symbol,
                signal_strength=signal_strength,
                current_price=price,
                as_of_date=as_of_date,
                current_portfolio_var=var_used,
            )

            if quantity <= 0:
                continue

            # Calculate VaR impact of this position
            symbol_var = self._get_symbol_var(symbol, as_of_date, price)
            position_value = Decimal(quantity) * price
            position_pct = float(position_value / self.config.portfolio_value)
            position_var = position_pct * symbol_var

            # Check if we have VaR budget for this position
            if var_used + position_var > var_budget_remaining:
                logger.info(
                    f"{symbol}: VaR budget exhausted "
                    f"(would use {position_var:.4f}, only {var_budget_remaining - var_used:.4f} remaining)"
                )
                break

            # Add to target positions
            target_positions[symbol] = quantity
            var_used += position_var

            logger.debug(
                f"{symbol}: allocated {quantity} shares "
                f"(VaR={position_var:.4f}, total_used={var_used:.4f})"
            )

        logger.info(
            f"Sized {len(target_positions)} positions from {len(buy_signals)} signals, "
            f"VaR used={var_used:.4f}/{var_budget_remaining:.4f}"
        )

        return target_positions
