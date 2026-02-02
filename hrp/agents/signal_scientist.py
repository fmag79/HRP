"""
Signal Scientist agent for automated signal discovery.

Scans feature universe for predictive signals using IC analysis
and creates draft hypotheses for promising signals.
"""

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from hrp.agents.base import ResearchAgent
from hrp.agents.jobs import DataRequirement
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType


@dataclass
class SignalScanResult:
    """Result of a single feature scan."""

    feature_name: str
    forward_horizon: int  # days
    ic: float
    ic_std: float  # Standard deviation across rolling windows
    ic_ir: float  # Information ratio (ic / ic_std)
    sample_size: int
    start_date: date
    end_date: date
    is_combination: bool = False  # True if two-factor combination
    combination_method: str | None = None  # "additive" or "subtractive"


@dataclass
class SignalScanReport:
    """Complete scan report."""

    scan_date: date
    total_features_scanned: int
    promising_signals: list[SignalScanResult]
    hypotheses_created: list[str]  # hypothesis_ids
    mlflow_run_id: str
    duration_seconds: float


class SignalScientist(ResearchAgent):
    """
    Scans feature universe for predictive signals.

    The Signal Scientist performs systematic IC (Information Coefficient)
    analysis across all features to identify those with predictive power
    for forward returns. When promising signals are found, it creates
    draft hypotheses for review.

    Features:
    - Rolling IC calculation for robust signal detection
    - Multi-horizon analysis (5, 10, 20 day returns)
    - Two-factor combination scanning
    - Automatic hypothesis creation for strong signals
    - MLflow logging for reproducibility
    - Email notifications with scan results
    """

    DEFAULT_JOB_ID = "signal_scientist_scan"
    ACTOR = "agent:signal-scientist"

    # IC thresholds
    IC_WEAK = 0.02
    IC_MODERATE = 0.03
    IC_STRONG = 0.05

    # Pre-defined two-factor combinations (theoretically motivated)
    FACTOR_PAIRS = [
        ("momentum_20d", "volatility_60d"),  # Momentum + Low Vol
        ("momentum_20d", "rsi_14d"),  # Momentum + Oversold
        ("returns_252d", "volatility_60d"),  # Annual momentum + Vol
        ("price_to_sma_200d", "rsi_14d"),  # Trend + Mean reversion
        ("volume_ratio", "momentum_20d"),  # Volume confirmation
    ]

    # All available features (39 technical + 5 fundamental)
    ALL_FEATURES = [
        # Returns
        "returns_1d",
        "returns_5d",
        "returns_20d",
        "returns_60d",
        "returns_252d",
        # Momentum
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        # Volatility
        "volatility_20d",
        "volatility_60d",
        # Volume
        "volume_20d",
        "volume_ratio",
        "obv",
        # Oscillators
        "rsi_14d",
        "cci_20d",
        "roc_10d",
        "stoch_k_14d",
        "stoch_d_14d",
        "williams_r_14d",
        "mfi_14d",
        # Trend
        "atr_14d",
        "adx_14d",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "trend",
        # Moving Averages
        "sma_20d",
        "sma_50d",
        "sma_200d",
        "ema_12d",
        "ema_26d",
        # EMA Signals
        "ema_crossover",
        # Price Ratios
        "price_to_sma_20d",
        "price_to_sma_50d",
        "price_to_sma_200d",
        # Bollinger Bands
        "bb_upper_20d",
        "bb_lower_20d",
        "bb_width_20d",
        # VWAP
        "vwap_20d",
        # Fundamental
        "market_cap",
        "pe_ratio",
        "pb_ratio",
        "dividend_yield",
        "ev_ebitda",
    ]

    def __init__(
        self,
        symbols: list[str] | None = None,
        features: list[str] | None = None,
        forward_horizons: list[int] | None = None,
        lookback_days: int = 756,  # 3 years
        ic_threshold: float = 0.03,
        create_hypotheses: bool = True,
        as_of_date: date | None = None,
        max_feature_age_days: int = 7,
    ):
        """
        Initialize the Signal Scientist agent.

        Args:
            symbols: List of symbols to analyze (None = all universe)
            features: List of features to scan (None = all 44 features)
            forward_horizons: Return horizons in days (default: [5, 10, 20])
            lookback_days: Days of history to use for IC calculation
            ic_threshold: Minimum IC to create hypothesis (default: 0.03)
            create_hypotheses: Whether to create draft hypotheses
            as_of_date: Date to run scan as of (default: today)
            max_feature_age_days: Maximum age of feature data in days (default: 7)
        """
        # Use data requirements instead of job-based dependencies
        data_requirements = [
            DataRequirement(
                table="features",
                min_rows=1000,  # Need substantial feature history
                max_age_days=max_feature_age_days,
                date_column="date",
                description="Feature data",
            )
        ]

        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=None,  # No job-based dependencies
            data_requirements=data_requirements,
        )
        self.symbols = symbols  # None = all universe
        self.features = features  # None = all features
        self.forward_horizons = forward_horizons or [5, 10, 20]
        self.lookback_days = lookback_days
        self.ic_threshold = ic_threshold
        self.create_hypotheses = create_hypotheses
        self.as_of_date = as_of_date or date.today()

    def execute(self) -> dict[str, Any]:
        """
        Run signal scan.

        Scans all features for IC against forward returns, creates
        hypotheses for promising signals, and sends notification.

        Returns:
            Dictionary with scan results
        """
        start_time = time.time()

        # 1. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        if not symbols:
            logger.warning("No symbols to scan")
            return {
                "scan_date": self.as_of_date.isoformat(),
                "features_scanned": 0,
                "signals_found": 0,
                "hypotheses_created": [],
                "error": "No symbols in universe",
            }

        # 2. Get features to scan
        features = self.features or self.ALL_FEATURES

        # 3. Pre-load ALL data once (reduces ~22,800 queries to 2)
        all_prices_df = self._load_prices(symbols)
        forward_returns_df = self._compute_forward_returns(all_prices_df)
        all_features_df = self._load_all_features(symbols, features)

        # 4. Scan each single feature
        results: list[SignalScanResult] = []
        for feature in features:
            for horizon in self.forward_horizons:
                try:
                    result = self._scan_feature(
                        feature, horizon, symbols, all_features_df, forward_returns_df
                    )
                    if result and abs(result.ic) >= self.IC_WEAK:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to scan {feature}/{horizon}d: {e}")

        # 5. Scan two-factor combinations
        for feature_a, feature_b in self.FACTOR_PAIRS:
            # Check if both features are in our scan list
            if self.features and (
                feature_a not in self.features or feature_b not in self.features
            ):
                continue

            for method in ["additive", "subtractive"]:
                for horizon in self.forward_horizons:
                    try:
                        result = self._scan_combination(
                            feature_a, feature_b, method, horizon, symbols,
                            all_features_df, forward_returns_df
                        )
                        if result and abs(result.ic) >= self.IC_WEAK:
                            # Only keep if combo beats both individual features
                            ic_a = self._get_single_feature_ic(feature_a, horizon, results)
                            ic_b = self._get_single_feature_ic(feature_b, horizon, results)
                            if abs(result.ic) > max(abs(ic_a), abs(ic_b)) + 0.005:
                                results.append(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to scan {feature_a}+{feature_b}/{method}: {e}"
                        )

        # 6. Log to MLflow
        mlflow_run_id = self._log_to_mlflow(results)

        # 7. Create hypotheses for promising signals
        hypotheses_created = []
        if self.create_hypotheses:
            promising = [
                r
                for r in results
                if abs(r.ic) >= self.ic_threshold and r.ic_ir >= 0.3
            ]
            for signal in promising:
                try:
                    hyp_id = self._create_hypothesis(signal)
                    if hyp_id:
                        hypotheses_created.append(hyp_id)
                except Exception as e:
                    logger.error(f"Failed to create hypothesis for {signal.feature_name}: {e}")

        # 8. Log agent completion event
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "features_scanned": len(features),
                "combinations_scanned": len(self.FACTOR_PAIRS) * 2,
                "signals_found": len(results),
                "hypotheses_created": hypotheses_created,
                "mlflow_run_id": mlflow_run_id,
            },
        )

        duration = time.time() - start_time

        # 9. Send email notification
        self._send_email_notification(results, hypotheses_created, mlflow_run_id, duration)

        return {
            "scan_date": self.as_of_date.isoformat(),
            "features_scanned": len(features),
            "combinations_scanned": len(self.FACTOR_PAIRS) * 2,
            "signals_found": len(results),
            "promising_signals": len(
                [r for r in results if abs(r.ic) >= self.ic_threshold]
            ),
            "hypotheses_created": hypotheses_created,
            "mlflow_run_id": mlflow_run_id,
            "duration_seconds": duration,
        }

    def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the current universe."""
        try:
            return self.api.get_universe(self.as_of_date)
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}")
            # Fallback to symbols with features
            result = self.api.fetchall_readonly(
                """
                SELECT DISTINCT symbol
                FROM features
                WHERE date >= ?
                ORDER BY symbol
                """,
                (self.as_of_date - timedelta(days=30),),
            )
            return [row[0] for row in result]

    def _load_prices(self, symbols: list[str]) -> pd.DataFrame:
        """Load price data for forward return calculation.

        Extends end date by max horizon + buffer to enable forward return calculation.
        """
        start = self.as_of_date - timedelta(days=self.lookback_days)
        # Extend end date to compute forward returns for all horizons
        end = self.as_of_date + timedelta(days=max(self.forward_horizons) + 10)

        try:
            return self.api.get_prices(symbols, start, end)
        except Exception as e:
            logger.warning(f"Failed to load prices via API: {e}")
            # Direct query fallback
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            df = self.api.query_readonly(
                f"""
                SELECT symbol, date, adj_close as close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date
                """,
                (start, end),
            )
            return df

    def _load_all_features(self, symbols: list[str], features: list[str]) -> pd.DataFrame:
        """Load all feature data in a single query.

        Args:
            symbols: List of symbols to load features for
            features: List of feature names to load

        Returns:
            DataFrame with columns: symbol, date, feature_name, value
        """
        start_date = self.as_of_date - timedelta(days=self.lookback_days)
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        features_str = ",".join(f"'{f}'" for f in features)

        return self.api.query_readonly(
            f"""
            SELECT symbol, date, feature_name, value
            FROM features
            WHERE feature_name IN ({features_str})
              AND symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY feature_name, symbol, date
            """,
            (start_date, self.as_of_date),
        )

    def _compute_forward_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute forward returns for all horizons.

        Args:
            prices_df: DataFrame with columns: symbol, date, close (or adj_close)

        Returns:
            DataFrame with added columns: fwd_ret_{horizon}d for each horizon
        """
        if prices_df.empty:
            return prices_df

        result = prices_df.copy()

        # Normalize column name
        if "adj_close" in result.columns:
            result = result.rename(columns={"adj_close": "close"})

        for horizon in self.forward_horizons:
            result[f"fwd_ret_{horizon}d"] = (
                result.groupby("symbol")["close"]
                .transform(lambda x: x.shift(-horizon) / x - 1)
            )
        return result

    def _calculate_rolling_ic(
        self,
        feature_values: np.ndarray,
        forward_returns: np.ndarray,
        window_size: int = 60,
    ) -> dict[str, float]:
        """
        Calculate rolling IC (Information Coefficient) using Spearman correlation.

        Args:
            feature_values: Array of feature values
            forward_returns: Array of forward returns
            window_size: Rolling window size in observations

        Returns:
            Dictionary with mean_ic, ic_std, ic_ir, sample_size
        """
        if len(feature_values) < window_size:
            return {
                "mean_ic": 0.0,
                "ic_std": 1.0,
                "ic_ir": 0.0,
                "sample_size": len(feature_values),
            }

        rolling_ics = []

        for i in range(window_size, len(feature_values)):
            window_features = feature_values[i - window_size : i]
            window_returns = forward_returns[i - window_size : i]

            # Skip if insufficient valid data
            valid_mask = ~(np.isnan(window_features) | np.isnan(window_returns))
            if valid_mask.sum() < window_size * 0.5:  # Require at least 50% valid
                continue

            # Calculate Spearman correlation (IC)
            try:
                ic, _ = stats.spearmanr(
                    window_features[valid_mask], window_returns[valid_mask]
                )
                if not np.isnan(ic):
                    rolling_ics.append(ic)
            except Exception:
                continue

        if not rolling_ics:
            return {
                "mean_ic": 0.0,
                "ic_std": 1.0,
                "ic_ir": 0.0,
                "sample_size": len(feature_values),
            }

        mean_ic = np.mean(rolling_ics)
        ic_std = np.std(rolling_ics) if len(rolling_ics) > 1 else 1.0

        # Avoid division by zero
        if ic_std < 1e-8:
            ic_std = 1e-8

        ic_ir = mean_ic / ic_std

        return {
            "mean_ic": float(mean_ic),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_ir),
            "sample_size": len(feature_values),
        }

    def _scan_feature(
        self,
        feature: str,
        horizon: int,
        symbols: list[str],
        all_features_df: pd.DataFrame | None = None,
        forward_returns_df: pd.DataFrame | None = None,
    ) -> SignalScanResult | None:
        """
        Calculate IC for a single feature/horizon combination.

        Args:
            feature: Feature name to scan
            horizon: Forward return horizon in days
            symbols: Symbols to include in scan
            all_features_df: Pre-loaded feature data (optional, for optimization)
            forward_returns_df: Pre-computed forward returns (optional, for optimization)

        Returns:
            SignalScanResult or None if insufficient data
        """
        start_date = self.as_of_date - timedelta(days=self.lookback_days)

        # Filter feature data from pre-loaded DataFrame or query DB
        if all_features_df is not None and not all_features_df.empty:
            feature_df = all_features_df[
                all_features_df["feature_name"] == feature
            ][["symbol", "date", "value"]].copy()
        else:
            # Fallback to DB query (for backward compatibility with tests)
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            feature_df = self.api.query_readonly(
                f"""
                SELECT symbol, date, value
                FROM features
                WHERE feature_name = ?
                  AND symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date
                """,
                (feature, start_date, self.as_of_date),
            )

        if feature_df.empty:
            return None

        # Use pre-computed forward returns or compute from DB
        if forward_returns_df is not None and not forward_returns_df.empty:
            fwd_ret_col = f"fwd_ret_{horizon}d"
            if fwd_ret_col not in forward_returns_df.columns:
                return None
            price_df = forward_returns_df[["symbol", "date", fwd_ret_col]].copy()
            price_df = price_df.rename(columns={fwd_ret_col: "forward_return"})
        else:
            # Fallback to DB query (for backward compatibility with tests)
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            price_df = self.api.query_readonly(
                f"""
                SELECT symbol, date, adj_close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date
                """,
                (start_date, self.as_of_date + timedelta(days=horizon + 10)),
            )
            if price_df.empty:
                return None
            # Compute forward returns per symbol
            price_df["forward_return"] = (
                price_df.groupby("symbol")["adj_close"]
                .transform(lambda x: x.shift(-horizon) / x - 1)
            )
            price_df = price_df[["symbol", "date", "forward_return"]]

        if price_df.empty:
            return None

        # Merge features with forward returns
        merged = pd.merge(
            feature_df,
            price_df,
            on=["symbol", "date"],
            how="inner",
        )

        # Drop rows with NaN forward returns
        merged = merged.dropna(subset=["forward_return"])

        if len(merged) < 100:  # Minimum sample size
            return None

        # Calculate rolling IC
        ic_result = self._calculate_rolling_ic(
            merged["value"].values,
            merged["forward_return"].values,
            window_size=60,
        )

        return SignalScanResult(
            feature_name=feature,
            forward_horizon=horizon,
            ic=ic_result["mean_ic"],
            ic_std=ic_result["ic_std"],
            ic_ir=ic_result["ic_ir"],
            sample_size=int(ic_result["sample_size"]),
            start_date=start_date,
            end_date=self.as_of_date,
        )

    def _scan_combination(
        self,
        feature_a: str,
        feature_b: str,
        method: str,
        horizon: int,
        symbols: list[str],
        all_features_df: pd.DataFrame | None = None,
        forward_returns_df: pd.DataFrame | None = None,
    ) -> SignalScanResult | None:
        """
        Scan a two-factor combination for IC.

        Args:
            feature_a: First feature name
            feature_b: Second feature name
            method: Combination method ('additive' or 'subtractive')
            horizon: Forward return horizon in days
            symbols: Symbols to include
            all_features_df: Pre-loaded feature data (optional, for optimization)
            forward_returns_df: Pre-computed forward returns (optional, for optimization)

        Returns:
            SignalScanResult or None if insufficient data
        """
        start_date = self.as_of_date - timedelta(days=self.lookback_days)

        # Get features from pre-loaded data or query DB
        if all_features_df is not None and not all_features_df.empty:
            features_df = all_features_df[
                all_features_df["feature_name"].isin([feature_a, feature_b])
            ].copy()
        else:
            # Fallback to DB query (for backward compatibility with tests)
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            features_df = self.api.query_readonly(
                f"""
                SELECT symbol, date, feature_name, value
                FROM features
                WHERE feature_name IN (?, ?)
                  AND symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date, feature_name
                """,
                (feature_a, feature_b, start_date, self.as_of_date),
            )

        if features_df.empty:
            return None

        # Pivot to wide format
        features_wide = features_df.pivot_table(
            index=["symbol", "date"], columns="feature_name", values="value"
        ).reset_index()

        if feature_a not in features_wide.columns or feature_b not in features_wide.columns:
            return None

        # Rank transform within each date (vectorized)
        features_wide["rank_a"] = features_wide.groupby("date")[feature_a].rank(pct=True)
        features_wide["rank_b"] = features_wide.groupby("date")[feature_b].rank(pct=True)

        # Combine ranks
        if method == "additive":
            features_wide["composite"] = features_wide["rank_a"] + features_wide["rank_b"]
        else:  # subtractive
            features_wide["composite"] = features_wide["rank_a"] - features_wide["rank_b"]

        # Filter dates with enough stocks for meaningful ranking
        date_counts = features_wide.groupby("date").size()
        valid_dates = date_counts[date_counts >= 5].index
        features_wide = features_wide[features_wide["date"].isin(valid_dates)]

        if features_wide.empty:
            return None

        # Get forward returns from pre-loaded data or query DB
        fwd_ret_col = f"fwd_ret_{horizon}d"
        if forward_returns_df is not None and not forward_returns_df.empty:
            if fwd_ret_col not in forward_returns_df.columns:
                return None
            price_df = forward_returns_df[["symbol", "date", fwd_ret_col]].copy()
            price_df = price_df.rename(columns={fwd_ret_col: "forward_return"})
        else:
            # Fallback: query DB and compute forward returns
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            price_df = self.api.query_readonly(
                f"""
                SELECT symbol, date, adj_close
                FROM prices
                WHERE symbol IN ({symbols_str})
                  AND date >= ?
                  AND date <= ?
                ORDER BY symbol, date
                """,
                (start_date, self.as_of_date + timedelta(days=horizon + 10)),
            )
            if price_df.empty:
                return None
            price_df["forward_return"] = (
                price_df.groupby("symbol")["adj_close"]
                .transform(lambda x: x.shift(-horizon) / x - 1)
            )
            price_df = price_df[["symbol", "date", "forward_return"]]

        # Merge composite scores with forward returns (vectorized)
        merged = pd.merge(
            features_wide[["symbol", "date", "composite"]],
            price_df,
            on=["symbol", "date"],
            how="inner",
        )

        # Drop rows with NaN forward returns
        merged = merged.dropna(subset=["forward_return"])

        if len(merged) < 100:
            return None

        ic_result = self._calculate_rolling_ic(
            merged["composite"].values,
            merged["forward_return"].values,
            window_size=60,
        )

        combo_name = f"{feature_a} {'+' if method == 'additive' else '-'} {feature_b}"

        return SignalScanResult(
            feature_name=combo_name,
            forward_horizon=horizon,
            ic=ic_result["mean_ic"],
            ic_std=ic_result["ic_std"],
            ic_ir=ic_result["ic_ir"],
            sample_size=int(ic_result["sample_size"]),
            start_date=start_date,
            end_date=self.as_of_date,
            is_combination=True,
            combination_method=method,
        )

    def _get_single_feature_ic(
        self, feature: str, horizon: int, results: list[SignalScanResult]
    ) -> float:
        """Get IC for a single feature from results list."""
        for r in results:
            if r.feature_name == feature and r.forward_horizon == horizon:
                return r.ic
        return 0.0

    def _create_hypothesis(self, signal: SignalScanResult) -> str | None:
        """
        Create draft hypothesis from promising signal.

        Args:
            signal: SignalScanResult with promising IC

        Returns:
            hypothesis_id or None if creation failed
        """
        direction = "positively" if signal.ic > 0 else "negatively"
        horizon_name = {5: "weekly", 10: "bi-weekly", 20: "monthly"}.get(
            signal.forward_horizon, f"{signal.forward_horizon}d"
        )

        title = f"{signal.feature_name} predicts {horizon_name} returns"

        thesis = (
            f"The {signal.feature_name} feature is {direction} correlated with "
            f"{signal.forward_horizon}-day forward returns (IC={signal.ic:.4f}). "
            f"This signal may capture a persistent market inefficiency."
        )

        prediction = (
            f"A long-short strategy based on {signal.feature_name} will achieve "
            f"IC > {self.IC_MODERATE:.2f} out-of-sample with stability score < 1.0."
        )

        falsification = (
            f"The signal fails if: (1) out-of-sample IC < {self.IC_WEAK:.2f}, "
            f"(2) IC is unstable across time periods (stability > 1.5), or "
            f"(3) the signal decays within 6 months of discovery."
        )

        # Determine strategy class based on feature name
        feature = signal.feature_name.lower()
        if "momentum" in feature or "returns" in feature:
            strategy_class = "time_series_momentum"
        else:
            strategy_class = "cross_sectional_factor"

        return self.api.create_hypothesis(
            title=title,
            thesis=thesis,
            prediction=prediction,
            falsification=falsification,
            actor=self.actor,
            strategy_class=strategy_class,
        )

    def _log_to_mlflow(self, results: list[SignalScanResult]) -> str:
        """
        Log scan results to MLflow.

        Args:
            results: List of SignalScanResult objects

        Returns:
            MLflow run ID
        """
        from hrp.research.mlflow_utils import setup_mlflow

        setup_mlflow()

        with mlflow.start_run(run_name=f"signal_scan_{self.as_of_date}") as run:
            # Log parameters
            mlflow.log_params(
                {
                    "scan_date": self.as_of_date.isoformat(),
                    "features_count": len(self.features or self.ALL_FEATURES),
                    "forward_horizons": str(self.forward_horizons),
                    "lookback_days": self.lookback_days,
                    "ic_threshold": self.ic_threshold,
                    "symbols_count": len(self.symbols or []),
                }
            )

            # Log summary metrics
            mlflow.log_metrics(
                {
                    "signals_above_weak": len(
                        [r for r in results if abs(r.ic) >= self.IC_WEAK]
                    ),
                    "signals_above_moderate": len(
                        [r for r in results if abs(r.ic) >= self.IC_MODERATE]
                    ),
                    "signals_above_strong": len(
                        [r for r in results if abs(r.ic) >= self.IC_STRONG]
                    ),
                    "total_signals": len(results),
                }
            )

            # Log individual signal metrics
            for i, result in enumerate(sorted(results, key=lambda x: abs(x.ic), reverse=True)[:20]):
                mlflow.log_metrics(
                    {
                        f"signal_{i}_ic": result.ic,
                        f"signal_{i}_ir": result.ic_ir,
                    }
                )

            return str(run.info.run_id)

    def _send_email_notification(
        self,
        results: list[SignalScanResult],
        hypotheses_created: list[str],
        mlflow_run_id: str,
        duration: float,
    ) -> None:
        """
        Send email notification with scan results.

        Args:
            results: List of SignalScanResult objects
            hypotheses_created: List of created hypothesis IDs
            mlflow_run_id: MLflow run ID
            duration: Scan duration in seconds
        """
        try:
            notifier = EmailNotifier()

            # Sort by absolute IC
            top_signals = sorted(results, key=lambda x: abs(x.ic), reverse=True)[:10]

            summary_data = {
                "scan_date": self.as_of_date.isoformat(),
                "duration_seconds": f"{duration:.1f}",
                "features_scanned": len(self.features or self.ALL_FEATURES),
                "signals_found": len(results),
                "signals_above_threshold": len(
                    [r for r in results if abs(r.ic) >= self.ic_threshold]
                ),
                "hypotheses_created": len(hypotheses_created),
                "mlflow_run_id": mlflow_run_id,
            }

            # Add top signals
            for i, signal in enumerate(top_signals[:5]):
                summary_data[f"top_{i+1}_signal"] = (
                    f"{signal.feature_name} ({signal.forward_horizon}d): "
                    f"IC={signal.ic:.4f}, IR={signal.ic_ir:.2f}"
                )

            subject = f"[HRP] Signal Scan Complete - {len(top_signals)} signals found"

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
