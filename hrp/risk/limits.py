"""
Portfolio risk limits for pre-trade validation.

Defines limit configurations and validation reporting structures.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import numpy as np
from loguru import logger

from hrp.risk.costs import MarketImpactModel


@dataclass
class LimitViolation:
    """Record of a single limit violation or clip."""

    limit_name: str
    symbol: str | None
    limit_value: float
    actual_value: float
    action: Literal["clipped", "rejected", "warned"]
    details: str | None = None


@dataclass
class ValidationReport:
    """Report from pre-trade validation."""

    violations: list[LimitViolation] = field(default_factory=list)
    clips: list[LimitViolation] = field(default_factory=list)
    warnings: list[LimitViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no hard violations (clips and warnings are ok)."""
        return len(self.violations) == 0


@dataclass
class RiskLimits:
    """
    Portfolio risk limits for pre-trade validation.

    Conservative institutional defaults for long-only equity.
    """

    # Position limits
    max_position_pct: float = 0.05      # Max 5% in any single position
    min_position_pct: float = 0.01      # Min 1% (avoid tiny positions)
    max_position_adv_pct: float = 0.10  # Max 10% of daily volume

    # Sector limits
    max_sector_pct: float = 0.25        # Max 25% in any sector
    max_unknown_sector_pct: float = 0.10  # Max 10% in unknown sectors

    # Portfolio limits
    max_gross_exposure: float = 1.00    # 100% = no leverage
    min_gross_exposure: float = 0.80    # Stay 80%+ invested
    max_net_exposure: float = 1.00      # Long-only: net = gross

    # Turnover limits
    max_turnover_pct: float = 0.20      # Max 20% turnover per rebalance

    # Concentration limits
    max_top_n_concentration: float = 0.40  # Top 5 holdings < 40%
    top_n_for_concentration: int = 5

    # Liquidity
    min_adv_dollars: float = 1_000_000  # Min $1M daily volume


class RiskLimitViolationError(Exception):
    """Raised in strict mode when limits are violated."""
    pass


class PreTradeValidator:
    """
    Validates and adjusts signals against risk limits.

    Modes:
        - clip: Adjust weights to satisfy limits (default)
        - strict: Raise exception on any violation
        - warn: Log warnings but allow violations
    """

    def __init__(
        self,
        limits: RiskLimits,
        cost_model: MarketImpactModel,
        mode: Literal["clip", "strict", "warn"] = "clip",
    ):
        self.limits = limits
        self.cost_model = cost_model
        self.mode = mode

    def validate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        sectors: pd.Series,
        adv: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate signals against risk limits.

        Checks applied in order:
        1. Liquidity filter (remove illiquid symbols)
        2. Position sizing (max)
        3. Position sizing (min)
        4. Sector exposure
        5. Concentration (top N)
        6. Gross exposure

        Args:
            signals: Raw signals (weights per symbol per date)
            prices: Price data for position sizing
            sectors: Symbol → sector mapping
            adv: Average daily volume (shares)

        Returns:
            Tuple of (validated_signals, ValidationReport)
        """
        validated = signals.copy()
        report = ValidationReport()

        # Check 1: Liquidity filter
        validated, report = self._check_liquidity(validated, prices, adv, report)

        # Check 2: Position sizing (max)
        validated, report = self._check_max_position(validated, report)

        # Check 3: Position sizing (min)
        validated, report = self._check_min_position(validated, report)

        # Check 4: Sector exposure
        validated, report = self._check_sector_exposure(validated, sectors, report)

        # Check 5: Concentration
        validated, report = self._check_concentration(validated, report)

        # Check 6: Gross exposure
        validated, report = self._check_gross_exposure(validated, report)

        # In strict mode, raise if any violations
        if self.mode == "strict" and (report.clips or report.warnings):
            violations = report.clips + report.warnings
            raise RiskLimitViolationError(
                f"{len(violations)} risk limit violations detected"
            )

        return validated, report

    def _check_max_position(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Clip positions above max_position_pct."""
        max_pct = self.limits.max_position_pct

        for col in signals.columns:
            mask = signals[col] > max_pct
            if mask.any():
                violation = LimitViolation(
                    limit_name="max_position_pct",
                    symbol=col,
                    limit_value=max_pct,
                    actual_value=float(signals[col][mask].max()),
                    action="clipped" if self.mode == "clip" else "warned",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = max_pct
                    report.clips.append(violation)
                    logger.debug(f"Clipped {col} from {violation.actual_value:.2%} to {max_pct:.2%}")
                else:  # warn mode
                    report.warnings.append(violation)
                    logger.warning(f"{col} exceeds max position: {violation.actual_value:.2%} > {max_pct:.2%}")

        return signals, report

    def _check_min_position(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Zero out positions below min_position_pct."""
        min_pct = self.limits.min_position_pct

        for col in signals.columns:
            mask = (signals[col] > 0) & (signals[col] < min_pct)
            if mask.any():
                violation = LimitViolation(
                    limit_name="min_position_pct",
                    symbol=col,
                    limit_value=min_pct,
                    actual_value=float(signals[col][mask].min()),
                    action="clipped" if self.mode == "clip" else "warned",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = 0.0
                    report.clips.append(violation)
                    logger.debug(f"Zeroed {col}: {violation.actual_value:.2%} below min {min_pct:.2%}")
                else:  # warn mode
                    report.warnings.append(violation)
                    logger.warning(f"{col} below min position: {violation.actual_value:.2%} < {min_pct:.2%}")

        return signals, report

    def _check_liquidity(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        adv: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Filter out symbols below minimum ADV threshold."""
        min_adv = self.limits.min_adv_dollars

        for col in signals.columns:
            if col not in adv.columns or col not in prices.columns:
                continue

            # ADV in dollars = shares * price
            adv_dollars = adv[col] * prices[col]

            mask = adv_dollars < min_adv
            if mask.any() and (signals.loc[mask, col] > 0).any():
                actual_adv = float(adv_dollars[mask].mean())
                violation = LimitViolation(
                    limit_name="min_adv_dollars",
                    symbol=col,
                    limit_value=min_adv,
                    actual_value=actual_adv,
                    action="clipped" if self.mode == "clip" else "warned",
                    details=f"ADV ${actual_adv:,.0f} below minimum ${min_adv:,.0f}",
                )

                if self.mode == "clip":
                    signals.loc[mask, col] = 0.0
                    report.clips.append(violation)
                    logger.debug(f"Filtered {col}: ADV ${actual_adv:,.0f} < ${min_adv:,.0f}")
                else:
                    report.warnings.append(violation)

        return signals, report

    def _check_sector_exposure(
        self,
        signals: pd.DataFrame,
        sectors: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Pro-rata reduce sector exposure above max."""
        max_sector = self.limits.max_sector_pct

        for idx in signals.index:
            row = signals.loc[idx]

            # Group by sector
            sector_exposure = {}
            for symbol in row.index:
                sector = sectors.get(symbol, "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + row[symbol]

            # Check each sector
            for sector, exposure in sector_exposure.items():
                limit = (
                    self.limits.max_unknown_sector_pct
                    if sector == "Unknown"
                    else max_sector
                )

                if exposure > limit:
                    # Calculate scale factor
                    scale = limit / exposure

                    # Apply to all symbols in this sector
                    for symbol in row.index:
                        if sectors.get(symbol, "Unknown") == sector:
                            old_weight = signals.loc[idx, symbol]
                            if old_weight > 0:
                                new_weight = old_weight * scale
                                violation = LimitViolation(
                                    limit_name="max_sector_pct",
                                    symbol=symbol,
                                    limit_value=limit,
                                    actual_value=exposure,
                                    action="clipped" if self.mode == "clip" else "warned",
                                    details=f"{sector} sector: {exposure:.2%} -> {limit:.2%}",
                                )

                                if self.mode == "clip":
                                    signals.loc[idx, symbol] = new_weight
                                    report.clips.append(violation)
                                else:
                                    report.warnings.append(violation)

        return signals, report

    def _check_concentration(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Reduce top N concentration if exceeded."""
        max_conc = self.limits.max_top_n_concentration
        top_n = self.limits.top_n_for_concentration

        for idx in signals.index:
            row = signals.loc[idx]
            sorted_weights = row.sort_values(ascending=False)
            top_n_exposure = sorted_weights.head(top_n).sum()

            if top_n_exposure > max_conc:
                # Scale down top N positions
                scale = max_conc / top_n_exposure
                top_symbols = sorted_weights.head(top_n).index

                for symbol in top_symbols:
                    old_weight = signals.loc[idx, symbol]
                    new_weight = old_weight * scale

                    violation = LimitViolation(
                        limit_name="max_top_n_concentration",
                        symbol=symbol,
                        limit_value=max_conc,
                        actual_value=top_n_exposure,
                        action="clipped" if self.mode == "clip" else "warned",
                        details=f"Top {top_n}: {top_n_exposure:.2%} -> {max_conc:.2%}",
                    )

                    if self.mode == "clip":
                        signals.loc[idx, symbol] = new_weight
                        report.clips.append(violation)
                    else:
                        report.warnings.append(violation)

        return signals, report

    def _check_gross_exposure(
        self,
        signals: pd.DataFrame,
        report: ValidationReport,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Scale down all positions if gross exposure exceeded."""
        max_gross = self.limits.max_gross_exposure

        for idx in signals.index:
            gross = signals.loc[idx].sum()

            if gross > max_gross:
                scale = max_gross / gross

                for symbol in signals.columns:
                    old_weight = signals.loc[idx, symbol]
                    if old_weight > 0:
                        new_weight = old_weight * scale

                        violation = LimitViolation(
                            limit_name="max_gross_exposure",
                            symbol=symbol,
                            limit_value=max_gross,
                            actual_value=gross,
                            action="clipped" if self.mode == "clip" else "warned",
                            details=f"Gross: {gross:.2%} -> {max_gross:.2%}",
                        )

                        if self.mode == "clip":
                            signals.loc[idx, symbol] = new_weight
                            report.clips.append(violation)
                        else:
                            report.warnings.append(violation)

        return signals, report
