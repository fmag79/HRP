"""
Portfolio construction with optimization and constraints.

Replaces equal-weight sizing with proper portfolio optimization using
shrinkage estimators and risk-aware constraints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio construction."""

    max_positions: int = 20
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_correlation: float = 0.70
    max_weekly_turnover_pct: float = 0.20
    min_holding_days: int = 5

    @classmethod
    def from_env(cls) -> PortfolioConstraints:
        """Load constraints from environment variables."""
        return cls(
            max_positions=int(os.getenv("HRP_MAX_POSITIONS", "20")),
            max_position_pct=float(os.getenv("HRP_MAX_POSITION_PCT", "0.10")),
            max_sector_pct=float(os.getenv("HRP_PORTFOLIO_MAX_SECTOR_PCT", "0.30")),
            max_correlation=float(os.getenv("HRP_PORTFOLIO_MAX_CORRELATION", "0.70")),
            max_weekly_turnover_pct=float(
                os.getenv("HRP_PORTFOLIO_MAX_WEEKLY_TURNOVER", "0.20")
            ),
        )


@dataclass
class PortfolioAllocation:
    """Result of portfolio construction."""

    weights: dict[str, float]  # symbol -> weight (0 to 1)
    expected_return: float
    expected_risk: float
    sector_exposures: dict[str, float]
    turnover_pct: float
    dropped_symbols: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)


class CovarianceEstimator:
    """Robust covariance estimation for portfolio optimization."""

    def estimate(
        self,
        returns: pd.DataFrame,
        method: str = "ledoit_wolf",
    ) -> np.ndarray:
        """
        Estimate covariance matrix with shrinkage.

        Args:
            returns: DataFrame with columns = symbols, rows = dates
            method: 'ledoit_wolf', 'sample', or 'exponential'

        Returns:
            Covariance matrix as numpy array
        """
        if method == "ledoit_wolf":
            return self._ledoit_wolf(returns)
        elif method == "exponential":
            return self._exponential(returns)
        else:
            return returns.cov().values

    def _ledoit_wolf(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator.

        Shrinks the sample covariance toward a structured target
        (identity scaled by average variance). Produces more stable
        estimates especially when n_samples < n_features.
        """
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns.dropna())
            return lw.covariance_
        except ImportError:
            logger.warning("sklearn not available, falling back to sample covariance")
            return returns.cov().values

    def _exponential(
        self, returns: pd.DataFrame, halflife: int = 60
    ) -> np.ndarray:
        """Exponentially weighted covariance (more recent data = more weight)."""
        return returns.ewm(halflife=halflife).cov().iloc[-len(returns.columns):].values


class PortfolioConstructor:
    """Optimized portfolio construction with constraints."""

    def __init__(
        self,
        api: PlatformAPI,
        constraints: PortfolioConstraints | None = None,
    ):
        self.api = api
        self.constraints = constraints or PortfolioConstraints()
        self.cov_estimator = CovarianceEstimator()

    def construct(
        self,
        signals: dict[str, float],
        current_weights: dict[str, float] | None = None,
        as_of_date: date | None = None,
        lookback_days: int = 252,
    ) -> PortfolioAllocation:
        """
        Construct an optimized portfolio from signals.

        Steps:
        1. Get historical returns for covariance estimation
        2. Estimate covariance (Ledoit-Wolf shrinkage)
        3. Filter by correlation constraint
        4. Filter by sector constraint
        5. Compute weights (inverse-variance weighted by signal strength)
        6. Apply position limits
        7. Apply turnover constraint

        Args:
            signals: symbol -> signal strength mapping
            current_weights: current portfolio weights (for turnover limit)
            as_of_date: Date for price lookback
            lookback_days: Days of history for covariance estimation

        Returns:
            PortfolioAllocation with optimized weights
        """
        if not signals:
            return PortfolioAllocation(
                weights={}, expected_return=0.0, expected_risk=0.0,
                sector_exposures={}, turnover_pct=0.0,
            )

        current_weights = current_weights or {}
        as_of_date = as_of_date or date.today()
        symbols = list(signals.keys())

        # 1. Get returns
        returns = self._get_returns(symbols, as_of_date, lookback_days)
        valid_symbols = [s for s in symbols if s in returns.columns]
        if not valid_symbols:
            logger.warning("No valid symbols with return data")
            return PortfolioAllocation(
                weights={}, expected_return=0.0, expected_risk=0.0,
                sector_exposures={}, turnover_pct=0.0,
                dropped_symbols=symbols,
            )

        # 2. Estimate covariance
        returns_clean = returns[valid_symbols].dropna()
        if len(returns_clean) < 30:
            logger.warning(f"Only {len(returns_clean)} days of data, need at least 30")
            return self._equal_weight_fallback(valid_symbols, signals)

        cov_matrix = self.cov_estimator.estimate(returns_clean)

        # 3. Filter by correlation
        dropped = {}
        filtered_symbols = self._filter_by_correlation(
            valid_symbols, returns_clean, signals, dropped
        )

        # 4. Filter by sector
        sector_map = self._get_sector_map(filtered_symbols)
        filtered_symbols = self._filter_by_sector(
            filtered_symbols, signals, sector_map, dropped
        )

        if not filtered_symbols:
            return PortfolioAllocation(
                weights={}, expected_return=0.0, expected_risk=0.0,
                sector_exposures={}, turnover_pct=0.0,
                dropped_symbols=list(dropped.keys()),
                drop_reasons=dropped,
            )

        # 5. Compute weights: signal-weighted inverse variance
        weights = self._compute_weights(filtered_symbols, signals, returns_clean)

        # 6. Apply position limits
        weights = self._apply_position_limits(weights)

        # 7. Apply turnover constraint
        turnover = self._compute_turnover(weights, current_weights)
        if turnover > self.constraints.max_weekly_turnover_pct:
            weights = self._constrain_turnover(
                weights, current_weights, self.constraints.max_weekly_turnover_pct
            )
            turnover = self._compute_turnover(weights, current_weights)

        # Compute portfolio metrics
        w = np.array([weights.get(s, 0) for s in filtered_symbols])
        sub_cov = returns_clean[filtered_symbols].cov().values
        expected_risk = float(np.sqrt(w @ sub_cov @ w) * np.sqrt(252))
        expected_return = float(
            sum(signals.get(s, 0) * weights.get(s, 0) for s in filtered_symbols)
        )

        sector_exposures = {}
        for s, wt in weights.items():
            sector = sector_map.get(s, "Unknown")
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + wt

        return PortfolioAllocation(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sector_exposures=sector_exposures,
            turnover_pct=turnover,
            dropped_symbols=list(dropped.keys()),
            drop_reasons=dropped,
        )

    # --- Private helpers ---

    def _get_returns(
        self, symbols: list[str], as_of_date: date, lookback_days: int
    ) -> pd.DataFrame:
        """Get daily returns for symbols."""
        start = pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days * 1.5)
        prices_dict = {}
        for symbol in symbols:
            try:
                df = self.api.get_prices([symbol], start.date(), as_of_date)
                if not df.empty:
                    prices_dict[symbol] = df.set_index("date")["close"].astype(float)
            except Exception:
                pass

        if not prices_dict:
            return pd.DataFrame()

        prices = pd.DataFrame(prices_dict)
        returns = prices.pct_change().dropna()
        return returns

    def _filter_by_correlation(
        self,
        symbols: list[str],
        returns: pd.DataFrame,
        signals: dict[str, float],
        dropped: dict[str, str],
    ) -> list[str]:
        """Remove highly correlated pairs, keeping the one with stronger signal."""
        corr_matrix = returns[symbols].corr()
        to_drop = set()

        for i, s1 in enumerate(symbols):
            if s1 in to_drop:
                continue
            for s2 in symbols[i + 1:]:
                if s2 in to_drop:
                    continue
                if abs(corr_matrix.loc[s1, s2]) > self.constraints.max_correlation:
                    # Drop the one with weaker signal
                    if abs(signals.get(s1, 0)) >= abs(signals.get(s2, 0)):
                        to_drop.add(s2)
                        dropped[s2] = f"High correlation ({corr_matrix.loc[s1, s2]:.2f}) with {s1}"
                    else:
                        to_drop.add(s1)
                        dropped[s1] = f"High correlation ({corr_matrix.loc[s1, s2]:.2f}) with {s2}"

        return [s for s in symbols if s not in to_drop]

    def _filter_by_sector(
        self,
        symbols: list[str],
        signals: dict[str, float],
        sector_map: dict[str, str],
        dropped: dict[str, str],
    ) -> list[str]:
        """Limit sector concentration."""
        sector_counts: dict[str, list[str]] = {}
        for s in symbols:
            sector = sector_map.get(s, "Unknown")
            sector_counts.setdefault(sector, []).append(s)

        max_per_sector = max(
            1,
            int(self.constraints.max_positions * self.constraints.max_sector_pct),
        )

        result = []
        for sector, syms in sector_counts.items():
            # Sort by signal strength and keep top N
            sorted_syms = sorted(syms, key=lambda s: abs(signals.get(s, 0)), reverse=True)
            kept = sorted_syms[:max_per_sector]
            result.extend(kept)
            for s in sorted_syms[max_per_sector:]:
                dropped[s] = f"Sector {sector} limit ({max_per_sector} max)"

        return result

    def _compute_weights(
        self,
        symbols: list[str],
        signals: dict[str, float],
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute signal-weighted inverse-variance weights."""
        variances = returns[symbols].var()

        raw_weights = {}
        for s in symbols:
            signal = abs(signals.get(s, 0))
            var = variances.get(s, 1.0)
            if var > 0:
                raw_weights[s] = signal / np.sqrt(var)
            else:
                raw_weights[s] = signal

        total = sum(raw_weights.values())
        if total <= 0:
            # Equal weight fallback
            n = len(symbols)
            return {s: 1.0 / n for s in symbols}

        return {s: w / total for s, w in raw_weights.items()}

    def _apply_position_limits(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip weights to position limit and redistribute excess."""
        max_w = self.constraints.max_position_pct
        clipped = {}
        excess = 0.0

        for s, w in weights.items():
            if w > max_w:
                excess += w - max_w
                clipped[s] = max_w
            else:
                clipped[s] = w

        # Redistribute excess proportionally to uncapped positions
        if excess > 0:
            uncapped = {s: w for s, w in clipped.items() if w < max_w}
            if uncapped:
                total_uncapped = sum(uncapped.values())
                for s in uncapped:
                    clipped[s] += excess * (uncapped[s] / total_uncapped)
                    clipped[s] = min(clipped[s], max_w)

        # Normalize
        total = sum(clipped.values())
        if total > 0:
            clipped = {s: w / total for s, w in clipped.items()}

        return clipped

    @staticmethod
    def _compute_turnover(
        new_weights: dict[str, float], old_weights: dict[str, float]
    ) -> float:
        """Compute one-way turnover between old and new portfolio."""
        all_symbols = set(list(new_weights.keys()) + list(old_weights.keys()))
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in all_symbols
        )
        return turnover / 2  # One-way

    @staticmethod
    def _constrain_turnover(
        new_weights: dict[str, float],
        old_weights: dict[str, float],
        max_turnover: float,
    ) -> dict[str, float]:
        """Blend old and new weights to meet turnover constraint."""
        # Binary search for blending factor
        low, high = 0.0, 1.0
        for _ in range(20):
            mid = (low + high) / 2
            blended = {}
            all_symbols = set(list(new_weights.keys()) + list(old_weights.keys()))
            for s in all_symbols:
                blended[s] = mid * new_weights.get(s, 0) + (1 - mid) * old_weights.get(s, 0)

            turnover = sum(
                abs(blended.get(s, 0) - old_weights.get(s, 0)) for s in all_symbols
            ) / 2

            if turnover > max_turnover:
                high = mid
            else:
                low = mid

        # Use the converged blend
        result = {}
        all_symbols = set(list(new_weights.keys()) + list(old_weights.keys()))
        for s in all_symbols:
            w = low * new_weights.get(s, 0) + (1 - low) * old_weights.get(s, 0)
            if w > 0.001:  # Drop dust positions
                result[s] = w

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {s: w / total for s, w in result.items()}
        return result

    def _get_sector_map(self, symbols: list[str]) -> dict[str, str]:
        """Get sector for each symbol."""
        sector_map = {}
        for symbol in symbols:
            try:
                result = self.api.fetchone_readonly(
                    "SELECT sector FROM symbols WHERE symbol = ?", [symbol]
                )
                if result and result[0]:
                    sector_map[symbol] = result[0]
                else:
                    sector_map[symbol] = "Unknown"
            except Exception:
                sector_map[symbol] = "Unknown"
        return sector_map

    def _equal_weight_fallback(
        self, symbols: list[str], signals: dict[str, float]
    ) -> PortfolioAllocation:
        """Fallback to equal weight when insufficient data for optimization."""
        n = min(len(symbols), self.constraints.max_positions)
        sorted_symbols = sorted(
            symbols, key=lambda s: abs(signals.get(s, 0)), reverse=True
        )[:n]
        weight = 1.0 / n
        return PortfolioAllocation(
            weights={s: weight for s in sorted_symbols},
            expected_return=0.0,
            expected_risk=0.0,
            sector_exposures={},
            turnover_pct=1.0,
        )
