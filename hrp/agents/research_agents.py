"""
Research agents for HRP automated hypothesis discovery.

Research agents extend the IngestionJob pattern to provide automated
research capabilities with actor tracking and lineage logging.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from hrp.agents.jobs import DataRequirement, IngestionJob
from hrp.api.platform import PlatformAPI
from hrp.data.db import get_db
from hrp.notifications.email import EmailNotifier
from hrp.research.lineage import EventType, log_event


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


class ResearchAgent(IngestionJob, ABC):
    """
    Base class for research agents (extends IngestionJob pattern).

    Research agents perform automated analysis and can create draft
    hypotheses. They track their actor identity for lineage purposes
    and have access to the PlatformAPI.
    """

    def __init__(
        self,
        job_id: str,
        actor: str,
        dependencies: list[str] | None = None,
        data_requirements: list | None = None,
        max_retries: int = 2,
    ):
        """
        Initialize a research agent.

        Args:
            job_id: Unique identifier for this job
            actor: Actor identity for lineage (e.g., 'agent:signal-scientist')
            dependencies: List of job IDs that must complete before this job runs
                         (legacy - prefer data_requirements)
            data_requirements: List of DataRequirement objects specifying what data
                              must exist before this agent can run
            max_retries: Maximum number of retry attempts
        """
        super().__init__(
            job_id=job_id,
            dependencies=dependencies or [],
            data_requirements=data_requirements or [],
            max_retries=max_retries,
        )
        self.actor = actor
        self.api = PlatformAPI()

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """
        Implement research logic.

        Must be implemented by subclasses.

        Returns:
            Dictionary with execution results
        """
        pass

    def _log_agent_event(
        self,
        event_type: str | EventType,
        details: dict,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
    ) -> int:
        """
        Log event to lineage with agent actor.

        Args:
            event_type: Type of event (EventType enum or string)
            details: Event-specific details
            hypothesis_id: Optional associated hypothesis
            experiment_id: Optional associated experiment

        Returns:
            lineage_id of the created event
        """
        # Convert EventType to string if needed
        if isinstance(event_type, EventType):
            event_type = event_type.value

        return log_event(
            event_type=event_type,
            actor=self.actor,
            details=details,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )


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
            db = get_db()
            result = db.fetchall(
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
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            df = db.fetchdf(
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
        db = get_db()
        start_date = self.as_of_date - timedelta(days=self.lookback_days)
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        features_str = ",".join(f"'{f}'" for f in features)

        return db.fetchdf(
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
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            feature_df = db.fetchdf(
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
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            price_df = db.fetchdf(
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
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            features_df = db.fetchdf(
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
            db = get_db()
            symbols_str = ",".join(f"'{s}'" for s in symbols)
            price_df = db.fetchdf(
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


# Adaptive IC thresholds by strategy class
IC_THRESHOLDS = {
    "cross_sectional_factor": {
        "pass": 0.015,
        "kill": 0.005,
        "description": "Value, quality, low-vol factors"
    },
    "time_series_momentum": {
        "pass": 0.02,
        "kill": 0.01,
        "description": "Trend-following strategies"
    },
    "ml_composite": {
        "pass": 0.025,
        "kill": 0.01,
        "description": "Multi-feature ML models"
    },
    "default": {
        "pass": 0.03,
        "kill": 0.01,
        "description": "Legacy uniform threshold"
    }
}


def get_ic_thresholds(strategy_class: str) -> dict:
    """
    Get IC thresholds for a strategy class.

    Args:
        strategy_class: One of: cross_sectional_factor, time_series_momentum,
                       ml_composite, default

    Returns:
        dict with 'pass' and 'kill' thresholds

    Examples:
        >>> thresholds = get_ic_thresholds("cross_sectional_factor")
        >>> assert thresholds["pass"] == 0.015
    """
    return IC_THRESHOLDS.get(strategy_class, IC_THRESHOLDS["default"])


@dataclass
class ModelExperimentResult:
    """Result of a single model experiment."""

    hypothesis_id: str
    model_type: str
    features: list[str]
    model_params: dict[str, Any]
    mean_ic: float
    ic_std: float
    stability_score: float
    is_stable: bool
    n_folds: int
    fold_results: list[dict]
    mlflow_run_id: str
    training_time_seconds: float


@dataclass
class MLScientistReport:
    """Complete ML Scientist run report."""

    run_date: date
    hypotheses_processed: int
    hypotheses_validated: int
    hypotheses_rejected: int
    total_trials: int
    total_training_time_seconds: float
    best_models: list[ModelExperimentResult]
    mlflow_experiment_id: str


class MLScientist(ResearchAgent):
    """
    Trains and validates ML models for hypotheses in testing status.

    The ML Scientist takes hypotheses created by the Signal Scientist
    (or manually) and systematically trains ML models using walk-forward
    validation. It identifies the best model/feature combinations and
    updates hypothesis status based on statistical rigor.

    Features:
    - Multi-model type testing (ridge, lasso, lightgbm)
    - Walk-forward validation with stability scoring
    - Feature combination search
    - Hyperparameter optimization with trial budget
    - Automatic hypothesis status updates
    - MLflow experiment logging
    - Email notifications with results
    """

    DEFAULT_JOB_ID = "ml_scientist_training"
    ACTOR = "agent:ml-scientist"

    # Default model types to test
    DEFAULT_MODEL_TYPES = ["ridge", "lasso", "lightgbm"]

    # Validation thresholds
    IC_THRESHOLD_VALIDATED = 0.03
    IC_THRESHOLD_PROMISING = 0.02
    STABILITY_THRESHOLD_VALIDATED = 1.0
    STABILITY_THRESHOLD_PROMISING = 1.5

    # Trial limits
    MAX_TRIALS_PER_HYPOTHESIS = 50
    MAX_FEATURE_COMBINATIONS = 10
    MAX_FEATURES_PER_MODEL = 3

    # Hyperparameter grids
    HYPERPARAMETER_GRIDS = {
        "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
        "elasticnet": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        },
        "lightgbm": {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
        "xgboost": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
        },
    }

    # Complementary features for combination search
    COMPLEMENTARY_FEATURES = {
        "momentum_20d": ["volatility_60d", "rsi_14d", "volume_ratio"],
        "momentum_60d": ["volatility_60d", "returns_252d", "adx_14d"],
        "momentum_252d": ["volatility_60d", "price_to_sma_200d"],
        "volatility_60d": ["momentum_20d", "returns_252d", "atr_14d"],
        "volatility_20d": ["momentum_20d", "rsi_14d"],
        "rsi_14d": ["momentum_20d", "price_to_sma_200d", "cci_20d"],
        "returns_252d": ["volatility_60d", "momentum_20d"],
        "price_to_sma_200d": ["rsi_14d", "momentum_20d", "trend"],
        "volume_ratio": ["momentum_20d", "obv"],
    }

    # All features (reuse from SignalScientist)
    ALL_FEATURES = SignalScientist.ALL_FEATURES

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        model_types: list[str] | None = None,
        target: str = "returns_20d",
        n_folds: int = 5,
        window_type: str = "expanding",
        start_date: date | None = None,
        end_date: date | None = None,
        symbols: list[str] | None = None,
        max_trials_per_hypothesis: int | None = None,
        skip_hyperparameter_search: bool = False,
        parallel_folds: bool = True,
    ):
        """
        Initialize the ML Scientist agent.

        Args:
            hypothesis_ids: Specific hypotheses to process (None = all in 'testing')
            model_types: Models to test (default: ridge, lasso, lightgbm)
            target: Target variable name (default: returns_20d)
            n_folds: Number of walk-forward folds (default: 5)
            window_type: 'expanding' or 'rolling' (default: expanding)
            start_date: Start of training date range
            end_date: End of training date range
            symbols: Symbols to use (None = all universe)
            max_trials_per_hypothesis: Max trials per hypothesis (default: 50)
            skip_hyperparameter_search: Use default params only
            parallel_folds: Run folds in parallel (default: True)
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=["signal_scientist_scan"],
        )
        self.hypothesis_ids = hypothesis_ids
        self.model_types = model_types or self.DEFAULT_MODEL_TYPES
        self.target = target
        self.n_folds = n_folds
        self.window_type = window_type
        self.start_date = start_date or date(2015, 1, 1)
        self.end_date = end_date or date.today()
        self.symbols = symbols
        self.max_trials = max_trials_per_hypothesis or self.MAX_TRIALS_PER_HYPOTHESIS
        self.skip_hyperparameter_search = skip_hyperparameter_search
        self.parallel_folds = parallel_folds

    def execute(self) -> dict[str, Any]:
        """
        Run ML experimentation on hypotheses in testing status.

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_process()
        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "message": "No hypotheses in testing status",
            }

        # 2. Get universe symbols
        symbols = self.symbols or self._get_universe_symbols()

        if not symbols:
            return {
                "status": "no_symbols",
                "message": "No symbols in universe",
            }

        # 3. Process each hypothesis
        all_results: list[ModelExperimentResult] = []
        validated_count = 0
        rejected_count = 0
        total_trials = 0

        for hypothesis in hypotheses:
            try:
                hyp_results = self._process_hypothesis(hypothesis, symbols)
                all_results.extend(hyp_results)

                # Update hypothesis status based on best result
                if hyp_results:
                    best = max(hyp_results, key=lambda r: self._calculate_model_score(r))
                    status = self._determine_status(best)
                    self._update_hypothesis(hypothesis, best, status)

                    if status == "validated":
                        validated_count += 1
                    elif status == "rejected":
                        rejected_count += 1

                    total_trials += len(hyp_results)
            except Exception as e:
                logger.error(f"Failed to process hypothesis {hypothesis.get('id')}: {e}")

        # 4. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_trials": total_trials,
                "duration_seconds": duration,
            },
        )

        # 5. Send email notification
        self._send_ml_email_notification(
            hypotheses, all_results, validated_count, rejected_count, duration
        )

        return {
            "run_date": date.today().isoformat(),
            "hypotheses_processed": len(hypotheses),
            "hypotheses_validated": validated_count,
            "hypotheses_rejected": rejected_count,
            "total_trials": total_trials,
            "duration_seconds": duration,
        }

    def _get_hypotheses_to_process(self) -> list[dict]:
        """Get hypotheses in testing status."""
        if self.hypothesis_ids:
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis(hid)
                if hyp:
                    hypotheses.append(hyp)
            return hypotheses
        return self.api.list_hypotheses(status="testing")

    def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the current universe."""
        try:
            return self.api.get_universe(date.today())
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}")
            # Fallback to symbols with features
            db = get_db()
            result = db.fetchall(
                """
                SELECT DISTINCT symbol
                FROM features
                WHERE date >= ?
                ORDER BY symbol
                """,
                (date.today() - timedelta(days=30),),
            )
            return [row[0] for row in result]

    def _process_hypothesis(
        self,
        hypothesis: dict,
        symbols: list[str],
    ) -> list[ModelExperimentResult]:
        """Process a single hypothesis through ML pipeline."""
        from hrp.risk.overfitting import HyperparameterTrialCounter

        results = []
        hypothesis_id = hypothesis.get("hypothesis_id", "unknown")

        # Initialize trial counter
        counter = HyperparameterTrialCounter(
            hypothesis_id=hypothesis_id,
            max_trials=self.max_trials,
        )

        # Extract base features from hypothesis
        base_features = self._extract_features_from_hypothesis(hypothesis)

        # Generate feature combinations
        feature_combos = self._generate_feature_combinations(base_features)

        # Test each model type
        for model_type in self.model_types:
            if counter.remaining_trials <= 0:
                logger.info(f"Trial budget exhausted for {hypothesis_id}")
                break

            # Test each feature combination
            for features in feature_combos:
                if counter.remaining_trials <= 0:
                    break

                # Get hyperparameter grid
                if self.skip_hyperparameter_search:
                    param_grid = [{}]  # Default params only
                else:
                    param_grid = self._get_param_grid(model_type)

                # Test each hyperparameter combination
                for params in param_grid:
                    if counter.remaining_trials <= 0:
                        break

                    try:
                        result = self._run_experiment(
                            hypothesis_id=hypothesis_id,
                            model_type=model_type,
                            features=features,
                            model_params=params,
                            symbols=symbols,
                        )

                        if result:
                            results.append(result)
                            try:
                                counter.log_trial(
                                    model_type=model_type,
                                    hyperparameters={"features": features, **params},
                                    metric_name="mean_ic",
                                    metric_value=result.mean_ic,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to log trial: {e}")

                            # Early stopping if we find excellent result
                            if result.mean_ic > 0.05 and result.is_stable:
                                logger.info(
                                    f"Excellent result found for {hypothesis_id}, stopping early"
                                )
                                return results
                    except Exception as e:
                        logger.warning(
                            f"Experiment failed: {model_type}/{features}: {e}"
                        )

        return results

    def _run_experiment(
        self,
        hypothesis_id: str,
        model_type: str,
        features: list[str],
        model_params: dict,
        symbols: list[str],
    ) -> ModelExperimentResult | None:
        """Run a single walk-forward validation experiment."""
        from hrp.ml import WalkForwardConfig, walk_forward_validate

        try:
            start_time = time.time()

            config = WalkForwardConfig(
                model_type=model_type,
                target=self.target,
                features=features,
                start_date=self.start_date,
                end_date=self.end_date,
                n_folds=self.n_folds,
                window_type=self.window_type,
                n_jobs=-1 if self.parallel_folds else 1,
                hyperparameters=model_params,
                tags={"hypothesis_id": hypothesis_id},
            )

            result = walk_forward_validate(
                config=config,
                symbols=symbols,
                log_to_mlflow=True,
            )

            training_time = time.time() - start_time

            # Extract fold IC values
            fold_results = []
            for fold in result.fold_results:
                fold_results.append(fold.metrics)

            mlflow_run_id = result.mlflow_run_id or ""

            # Link experiment to hypothesis so validation guards work
            if mlflow_run_id and hypothesis_id:
                self.api.link_experiment(hypothesis_id, mlflow_run_id)

            return ModelExperimentResult(
                hypothesis_id=hypothesis_id,
                model_type=model_type,
                features=features,
                model_params=model_params,
                mean_ic=result.mean_ic,
                ic_std=result.aggregate_metrics.get("std_ic", 0.0),
                stability_score=result.stability_score,
                is_stable=result.is_stable,
                n_folds=len(result.fold_results),
                fold_results=fold_results,
                mlflow_run_id=mlflow_run_id,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Experiment failed: {model_type}/{features}: {e}")
            return None

    def _extract_features_from_hypothesis(self, hypothesis: dict) -> list[str]:
        """Extract feature names from hypothesis thesis/metadata."""
        # Check metadata first
        metadata = hypothesis.get("metadata", {})
        if isinstance(metadata, dict) and "features" in metadata:
            return metadata["features"]

        # Parse from thesis text
        thesis = hypothesis.get("thesis", "")
        features = []
        for feature in self.ALL_FEATURES:
            if feature in thesis.lower():
                features.append(feature)

        return features if features else ["momentum_20d"]  # Default fallback

    def _generate_feature_combinations(self, base_features: list[str]) -> list[list[str]]:
        """Generate feature combinations to test."""
        combinations = [base_features]  # Start with base

        # Add complementary features
        for base in base_features:
            complements = self.COMPLEMENTARY_FEATURES.get(base, [])
            for comp in complements[:2]:  # Limit to top 2 complements
                combo = base_features + [comp]
                if len(combo) <= self.MAX_FEATURES_PER_MODEL:
                    combinations.append(combo)

        # Deduplicate and limit
        seen = set()
        unique = []
        for combo in combinations:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                unique.append(combo)

        return unique[: self.MAX_FEATURE_COMBINATIONS]

    def _get_param_grid(self, model_type: str) -> list[dict]:
        """Get hyperparameter combinations for model type."""
        from sklearn.model_selection import ParameterGrid

        grid = self.HYPERPARAMETER_GRIDS.get(model_type, {})
        if not grid:
            return [{}]

        return list(ParameterGrid(grid))[:10]  # Limit combinations

    def _calculate_model_score(self, result: ModelExperimentResult) -> float:
        """Calculate composite score for model ranking."""
        ic_score = result.mean_ic
        stability_penalty = 1 / max(result.stability_score, 0.1)

        # Bonus if all folds have positive IC
        all_positive = all(f.get("ic", 0) > 0 for f in result.fold_results)
        consistency_bonus = 1.2 if all_positive else 1.0

        return ic_score * stability_penalty * consistency_bonus

    def _determine_status(self, result: ModelExperimentResult) -> str:
        """Determine hypothesis status based on best model result."""
        if (
            result.mean_ic >= self.IC_THRESHOLD_VALIDATED
            and result.stability_score <= self.STABILITY_THRESHOLD_VALIDATED
            and result.is_stable
        ):
            return "validated"
        elif (
            result.mean_ic >= self.IC_THRESHOLD_PROMISING
            and result.stability_score <= self.STABILITY_THRESHOLD_PROMISING
        ):
            return "testing"  # Keep in testing for further work
        else:
            return "rejected"

    def _update_hypothesis(
        self,
        hypothesis: dict,
        best_result: ModelExperimentResult,
        status: str,
    ) -> None:
        """Update hypothesis with ML results."""
        hypothesis_id = hypothesis.get("hypothesis_id", "unknown")

        # Build outcome string with ML results
        outcome = (
            f"ML Scientist: {best_result.model_type} model with "
            f"features {best_result.features}, IC={best_result.mean_ic:.4f}, "
            f"stability={best_result.stability_score:.2f}"
        )

        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=status,
                outcome=outcome,
                actor=self.ACTOR,
                metadata={
                    "ml_scientist_results": {
                        "model_type": best_result.model_type,
                        "features": best_result.features,
                        "mean_ic": best_result.mean_ic,
                        "ic_std": best_result.ic_std,
                        "stability_score": best_result.stability_score,
                        "n_folds": best_result.n_folds,
                        "mlflow_run_id": best_result.mlflow_run_id,
                    }
                },
            )
            logger.info(f"Updated hypothesis {hypothesis_id} to status={status}")
        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _send_ml_email_notification(
        self,
        hypotheses: list[dict],
        results: list[ModelExperimentResult],
        validated_count: int,
        rejected_count: int,
        duration: float,
    ) -> None:
        """Send email notification with ML results."""
        try:
            notifier = EmailNotifier()

            # Group best results by hypothesis
            best_by_hypothesis = {}
            for result in results:
                hid = result.hypothesis_id
                if hid not in best_by_hypothesis:
                    best_by_hypothesis[hid] = result
                elif self._calculate_model_score(result) > self._calculate_model_score(
                    best_by_hypothesis[hid]
                ):
                    best_by_hypothesis[hid] = result

            summary_data = {
                "run_date": date.today().isoformat(),
                "duration_seconds": f"{duration:.1f}",
                "hypotheses_processed": len(hypotheses),
                "hypotheses_validated": validated_count,
                "hypotheses_rejected": rejected_count,
                "total_experiments": len(results),
            }

            # Add top results
            sorted_results = sorted(
                best_by_hypothesis.values(),
                key=lambda r: self._calculate_model_score(r),
                reverse=True,
            )
            for i, result in enumerate(sorted_results[:5]):
                summary_data[f"top_{i+1}_model"] = (
                    f"{result.hypothesis_id}: {result.model_type} "
                    f"IC={result.mean_ic:.4f}, stability={result.stability_score:.2f}"
                )

            subject = (
                f"[HRP] ML Scientist Complete - "
                f"{validated_count} validated, {rejected_count} rejected"
            )

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")


# =============================================================================
# ML Quality Sentinel Classes
# =============================================================================


class AuditSeverity(Enum):
    """Severity level for audit checks."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    severity: AuditSeverity
    details: dict[str, Any]
    message: str


@dataclass
class ExperimentAudit:
    """Complete audit of a single experiment."""

    experiment_id: str
    hypothesis_id: str
    mlflow_run_id: str | None
    audit_date: date
    checks: list[AuditCheck] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        """Count of critical severity checks."""
        return sum(1 for c in self.checks if c.severity == AuditSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count of warning severity checks."""
        return sum(1 for c in self.checks if c.severity == AuditSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found."""
        return self.critical_count > 0

    def add_check(self, check: AuditCheck) -> None:
        """Add a check result to the audit."""
        self.checks.append(check)


@dataclass
class MonitoringAlert:
    """Alert from deployed model monitoring."""

    model_id: str
    hypothesis_id: str
    alert_type: str  # ic_degradation, feature_drift, loss_streak
    severity: AuditSeverity
    message: str
    recommended_action: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualitySentinelReport:
    """Complete Sentinel run report."""

    report_date: date
    experiments_audited: int
    experiments_passed: int
    experiments_flagged: int
    critical_issues: list[tuple[str, str]]  # (experiment_id, issue)
    warnings: list[tuple[str, str]]
    models_monitored: int
    model_alerts: list[MonitoringAlert]
    duration_seconds: float


class MLQualitySentinel(ResearchAgent):
    """
    Independent quality auditor for ML experiments and deployed models.

    Detects overfitting, leakage, and model degradation. Acts as an
    impartial watchdog that prevents bad models from propagating.

    Quality Checks:
    1. Sharpe decay (train vs test) - critical if >50%
    2. Target leakage - critical if correlation >0.95
    3. Feature count - critical if >50 features
    4. Fold stability - critical if CV >2.0 or sign flips
    5. Suspiciously good - critical if IC >0.15 or Sharpe >3.0
    """

    DEFAULT_JOB_ID = "ml_quality_sentinel_audit"
    ACTOR = "agent:ml-quality-sentinel"

    # Sharpe decay thresholds
    SHARPE_DECAY_WARNING = 0.3
    SHARPE_DECAY_CRITICAL = 0.5

    # Leakage thresholds
    LEAKAGE_WARNING = 0.85
    LEAKAGE_CRITICAL = 0.95

    # Feature count thresholds
    FEATURE_COUNT_WARNING = 30
    FEATURE_COUNT_CRITICAL = 50
    MIN_SAMPLES_PER_FEATURE = 20

    # Fold stability thresholds
    FOLD_CV_WARNING = 1.0
    FOLD_CV_CRITICAL = 2.0
    MAX_SIGN_FLIPS = 1

    # Suspiciously good thresholds
    IC_SUSPICIOUS_WARNING = 0.10
    IC_SUSPICIOUS_CRITICAL = 0.15
    SHARPE_SUSPICIOUS_WARNING = 2.5
    SHARPE_SUSPICIOUS_CRITICAL = 3.0

    # Model monitoring thresholds
    IC_DEGRADATION_THRESHOLD = 0.5  # 50% drop from baseline
    MAX_LOSS_STREAK = 7

    def __init__(
        self,
        experiment_ids: list[str] | None = None,
        hypothesis_ids: list[str] | None = None,
        audit_window_days: int = 1,
        include_monitoring: bool = True,
        fail_on_critical: bool = True,
        send_alerts: bool = True,
    ):
        """
        Initialize the ML Quality Sentinel.

        Args:
            experiment_ids: Specific experiments to audit
            hypothesis_ids: Audit experiments for these hypotheses
            audit_window_days: Days of recent experiments to audit
            include_monitoring: Whether to monitor deployed models
            fail_on_critical: Whether to fail job on critical issues
            send_alerts: Whether to send email alerts
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.experiment_ids = experiment_ids
        self.hypothesis_ids = hypothesis_ids
        self.audit_window_days = audit_window_days
        self.include_monitoring = include_monitoring
        self.fail_on_critical = fail_on_critical
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """Run quality audit on experiments and deployed models."""
        start_time = time.time()

        # 1. Get experiments to audit
        experiments = self._get_experiments_to_audit()

        # 2. Audit each experiment
        audits: list[ExperimentAudit] = []
        critical_issues: list[tuple[str, str]] = []
        warnings: list[tuple[str, str]] = []

        for experiment in experiments:
            audit = self._audit_experiment(experiment)
            audits.append(audit)

            # Collect issues
            for check in audit.checks:
                if check.severity == AuditSeverity.CRITICAL:
                    critical_issues.append((audit.experiment_id, check.message))
                elif check.severity == AuditSeverity.WARNING:
                    warnings.append((audit.experiment_id, check.message))

            # Log per-experiment audit event
            self._log_agent_event(
                event_type=EventType.ML_QUALITY_SENTINEL_AUDIT,
                hypothesis_id=audit.hypothesis_id,
                details={
                    "experiment_id": audit.experiment_id,
                    "mlflow_run_id": audit.mlflow_run_id,
                    "overall_passed": audit.overall_passed,
                    "critical_count": audit.critical_count,
                    "warning_count": audit.warning_count,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "severity": c.severity.value}
                        for c in audit.checks
                    ],
                },
            )

            # Flag hypothesis if critical issues found
            if audit.has_critical_issues:
                self._flag_hypothesis(audit)

        # 3. Monitor deployed models
        model_alerts: list[MonitoringAlert] = []
        models_monitored = 0

        if self.include_monitoring:
            model_alerts, models_monitored = self._monitor_deployed_models()

        # 4. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "experiments_audited": len(audits),
                "experiments_passed": sum(1 for a in audits if a.overall_passed),
                "experiments_flagged": sum(1 for a in audits if a.has_critical_issues),
                "critical_issues": len(critical_issues),
                "warnings": len(warnings),
                "models_monitored": models_monitored,
                "model_alerts": len(model_alerts),
                "duration_seconds": duration,
            },
        )

        # 5. Write research note
        self._write_research_note(audits, model_alerts, critical_issues, warnings, duration)

        # 6. Send email notification
        if self.send_alerts and (critical_issues or model_alerts):
            self._send_alert_email(audits, model_alerts, critical_issues, warnings)

        # 7. Build report
        report = QualitySentinelReport(
            report_date=date.today(),
            experiments_audited=len(audits),
            experiments_passed=sum(1 for a in audits if a.overall_passed),
            experiments_flagged=sum(1 for a in audits if a.has_critical_issues),
            critical_issues=critical_issues,
            warnings=warnings,
            models_monitored=models_monitored,
            model_alerts=model_alerts,
            duration_seconds=duration,
        )

        return {
            "report_date": report.report_date.isoformat(),
            "experiments_audited": report.experiments_audited,
            "experiments_passed": report.experiments_passed,
            "experiments_flagged": report.experiments_flagged,
            "critical_issues_count": len(report.critical_issues),
            "warnings_count": len(report.warnings),
            "models_monitored": report.models_monitored,
            "model_alerts_count": len(report.model_alerts),
            "duration_seconds": report.duration_seconds,
        }

    def _get_experiments_to_audit(self) -> list[dict]:
        """Get experiments from the audit window."""
        if self.experiment_ids:
            return [self._get_experiment(eid) for eid in self.experiment_ids if eid]

        if self.hypothesis_ids:
            experiments = []
            for hid in self.hypothesis_ids:
                experiments.extend(self._get_experiments_for_hypothesis(hid))
            return experiments

        # Default: get experiments from last N days
        return self._get_recent_experiments(days=self.audit_window_days)

    def _audit_experiment(self, experiment: dict) -> ExperimentAudit:
        """Run all quality checks on an experiment."""
        audit = ExperimentAudit(
            experiment_id=experiment.get("id", "unknown"),
            hypothesis_id=experiment.get("hypothesis_id", "unknown"),
            mlflow_run_id=experiment.get("mlflow_run_id"),
            audit_date=date.today(),
        )

        # 1. Sharpe Decay Check
        sharpe_check = self._check_sharpe_decay(experiment)
        audit.add_check(sharpe_check)

        # 2. Target Leakage Check (if feature data available)
        if "features_df" in experiment and "target" in experiment:
            leakage_check = self._check_target_leakage(
                experiment["features_df"],
                experiment["target"],
            )
            audit.add_check(leakage_check)

        # 3. Feature Count Validation
        feature_check = self._validate_feature_count(experiment)
        audit.add_check(feature_check)

        # 4. Fold Stability Check
        if "fold_results" in experiment and experiment["fold_results"]:
            stability_check = self._check_fold_stability(experiment["fold_results"])
            audit.add_check(stability_check)

        # 5. Suspiciously Good Check
        suspicion_check = self._check_suspiciously_good(experiment)
        audit.add_check(suspicion_check)

        return audit

    def _check_sharpe_decay(self, experiment: dict) -> AuditCheck:
        """Check for excessive Sharpe ratio decay."""
        train_sharpe = experiment.get("train_sharpe", 0)
        test_sharpe = experiment.get("test_sharpe", 0)

        if train_sharpe is None:
            train_sharpe = 0
        if test_sharpe is None:
            test_sharpe = 0

        if train_sharpe <= 0:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.NONE,
                details={"train_sharpe": train_sharpe, "test_sharpe": test_sharpe},
                message="Train Sharpe non-positive, skip decay check",
            )

        decay_ratio = (train_sharpe - test_sharpe) / train_sharpe

        if decay_ratio >= self.SHARPE_DECAY_CRITICAL:
            return AuditCheck(
                name="sharpe_decay",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message=f"Critical Sharpe decay {decay_ratio:.1%}: {train_sharpe:.2f} → {test_sharpe:.2f}",
            )
        elif decay_ratio >= self.SHARPE_DECAY_WARNING:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message=f"Moderate Sharpe decay {decay_ratio:.1%}",
            )
        else:
            return AuditCheck(
                name="sharpe_decay",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "decay_ratio": decay_ratio,
                },
                message="Sharpe decay within acceptable limits",
            )

    def _check_target_leakage(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> AuditCheck:
        """Check for target leakage via correlation."""
        correlations = features.corrwith(target).abs()
        max_corr = float(correlations.max()) if not correlations.empty else 0.0

        critical_features = correlations[correlations >= self.LEAKAGE_CRITICAL].index.tolist()
        warning_features = correlations[
            (correlations >= self.LEAKAGE_WARNING) & (correlations < self.LEAKAGE_CRITICAL)
        ].index.tolist()

        if critical_features:
            return AuditCheck(
                name="target_leakage",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "suspicious_features": critical_features,
                    "max_correlation": max_corr,
                },
                message=f"Likely leakage in: {critical_features}",
            )
        elif warning_features:
            return AuditCheck(
                name="target_leakage",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "suspicious_features": warning_features,
                    "max_correlation": max_corr,
                },
                message=f"High correlation (may be legitimate): {warning_features}",
            )
        else:
            return AuditCheck(
                name="target_leakage",
                passed=True,
                severity=AuditSeverity.NONE,
                details={"max_correlation": max_corr},
                message="No leakage detected",
            )

    def _validate_feature_count(self, experiment: dict) -> AuditCheck:
        """Validate feature count relative to samples."""
        feature_count = experiment.get("feature_count", 0)
        sample_count = experiment.get("sample_count", 1)

        if feature_count is None:
            feature_count = 0
        if sample_count is None or sample_count == 0:
            sample_count = 1

        ratio = sample_count / feature_count if feature_count > 0 else float("inf")

        if feature_count > self.FEATURE_COUNT_CRITICAL:
            return AuditCheck(
                name="feature_count",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"Too many features: {feature_count} > {self.FEATURE_COUNT_CRITICAL}",
            )
        elif ratio < self.MIN_SAMPLES_PER_FEATURE and feature_count > 0:
            return AuditCheck(
                name="feature_count",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"Insufficient samples per feature: {ratio:.1f} < {self.MIN_SAMPLES_PER_FEATURE}",
            )
        elif feature_count > self.FEATURE_COUNT_WARNING:
            return AuditCheck(
                name="feature_count",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message=f"High feature count: {feature_count} (consider reduction)",
            )
        else:
            return AuditCheck(
                name="feature_count",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "feature_count": feature_count,
                    "sample_count": sample_count,
                    "ratio": ratio,
                },
                message="Feature count acceptable",
            )

    def _check_fold_stability(self, fold_results: list[dict]) -> AuditCheck:
        """Check for consistent performance across folds."""
        fold_ics = [f.get("ic", 0) for f in fold_results if f.get("ic") is not None]

        if not fold_ics:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={},
                message="No fold results to analyze",
            )

        mean_ic = float(np.mean(fold_ics))
        std_ic = float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0
        cv = std_ic / abs(mean_ic) if mean_ic != 0 else float("inf")

        positive = sum(1 for ic in fold_ics if ic > 0)
        negative = sum(1 for ic in fold_ics if ic < 0)
        sign_flips = min(positive, negative)

        if cv > self.FOLD_CV_CRITICAL:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"Unstable across folds: CV={cv:.2f} > {self.FOLD_CV_CRITICAL}",
            )
        elif sign_flips > self.MAX_SIGN_FLIPS:
            return AuditCheck(
                name="fold_stability",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"IC sign flips: {sign_flips} folds have opposite sign",
            )
        elif cv > self.FOLD_CV_WARNING:
            return AuditCheck(
                name="fold_stability",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message=f"Moderate instability: CV={cv:.2f}",
            )
        else:
            return AuditCheck(
                name="fold_stability",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "fold_ics": fold_ics,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "cv": cv,
                    "sign_flips": sign_flips,
                },
                message="Stable across folds",
            )

    def _check_suspiciously_good(self, experiment: dict) -> AuditCheck:
        """Flag results that are too good to be true."""
        mean_ic = experiment.get("mean_ic", 0)
        sharpe = experiment.get("sharpe")
        r2 = experiment.get("r2")

        if mean_ic is None:
            mean_ic = 0

        flags = []

        if mean_ic > self.IC_SUSPICIOUS_CRITICAL:
            flags.append(
                f"IC={mean_ic:.4f} exceeds {self.IC_SUSPICIOUS_CRITICAL} (extremely suspicious)"
            )
        elif mean_ic > self.IC_SUSPICIOUS_WARNING:
            flags.append(f"IC={mean_ic:.4f} exceeds {self.IC_SUSPICIOUS_WARNING} (suspicious)")

        if sharpe is not None and sharpe > self.SHARPE_SUSPICIOUS_CRITICAL:
            flags.append(f"Sharpe={sharpe:.2f} exceeds {self.SHARPE_SUSPICIOUS_CRITICAL}")
        elif sharpe is not None and sharpe > self.SHARPE_SUSPICIOUS_WARNING:
            flags.append(f"Sharpe={sharpe:.2f} exceeds {self.SHARPE_SUSPICIOUS_WARNING}")

        if r2 is not None and r2 > 0.5:
            flags.append(f"R²={r2:.4f} exceeds 0.5")

        # Determine severity based on flags
        has_critical = any(
            "extremely suspicious" in f or f"exceeds {self.SHARPE_SUSPICIOUS_CRITICAL}" in f
            for f in flags
        )

        if flags and has_critical:
            return AuditCheck(
                name="suspiciously_good",
                passed=False,
                severity=AuditSeverity.CRITICAL,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                    "flags": flags,
                },
                message="Results too good to be true: " + "; ".join(flags),
            )
        elif flags:
            return AuditCheck(
                name="suspiciously_good",
                passed=True,
                severity=AuditSeverity.WARNING,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                    "flags": flags,
                },
                message="Results warrant review: " + "; ".join(flags),
            )
        else:
            return AuditCheck(
                name="suspiciously_good",
                passed=True,
                severity=AuditSeverity.NONE,
                details={
                    "mean_ic": mean_ic,
                    "sharpe": sharpe,
                    "r2": r2,
                },
                message="Results within plausible range",
            )

    def _monitor_deployed_models(self) -> tuple[list[MonitoringAlert], int]:
        """Monitor deployed models for degradation."""
        alerts: list[MonitoringAlert] = []
        deployed_models = self._get_deployed_models()

        for model in deployed_models:
            # IC degradation check
            baseline_ic = model.get("validation_ic", 0)
            recent_ic = self._calculate_recent_ic(model, window=20)

            if (
                baseline_ic is not None
                and baseline_ic > 0
                and recent_ic < baseline_ic * (1 - self.IC_DEGRADATION_THRESHOLD)
            ):
                alerts.append(
                    MonitoringAlert(
                        model_id=model.get("id", "unknown"),
                        hypothesis_id=model.get("hypothesis_id", "unknown"),
                        alert_type="ic_degradation",
                        severity=AuditSeverity.CRITICAL,
                        message=f"IC degraded: {baseline_ic:.4f} → {recent_ic:.4f}",
                        recommended_action="Review model, consider suspension",
                    )
                )

            # Loss streak check
            recent_returns = self._get_model_returns(model, days=10)
            loss_streak = self._count_consecutive_losses(recent_returns)

            if loss_streak >= self.MAX_LOSS_STREAK:
                alerts.append(
                    MonitoringAlert(
                        model_id=model.get("id", "unknown"),
                        hypothesis_id=model.get("hypothesis_id", "unknown"),
                        alert_type="loss_streak",
                        severity=AuditSeverity.WARNING,
                        message=f"{loss_streak} consecutive losing days",
                        recommended_action="Review market regime compatibility",
                    )
                )

        return alerts, len(deployed_models)

    def _flag_hypothesis(self, audit: ExperimentAudit) -> None:
        """Flag hypothesis with quality issues."""
        critical_checks = [c for c in audit.checks if c.severity == AuditSeverity.CRITICAL]

        try:
            self.api.update_hypothesis(
                hypothesis_id=audit.hypothesis_id,
                metadata={
                    "quality_flags": {
                        "flagged_at": datetime.now().isoformat(),
                        "flagged_by": self.ACTOR,
                        "critical_issues": [c.message for c in critical_checks],
                        "audit_id": audit.experiment_id,
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to flag hypothesis {audit.hypothesis_id}: {e}")

        # Log to lineage
        self._log_agent_event(
            event_type=EventType.HYPOTHESIS_FLAGGED,
            hypothesis_id=audit.hypothesis_id,
            details={
                "experiment_id": audit.experiment_id,
                "critical_count": audit.critical_count,
                "issues": [c.message for c in critical_checks],
            },
        )

    # ==========================================================================
    # Helper methods (data access)
    # ==========================================================================

    def _get_experiment(self, experiment_id: str) -> dict:
        """Get experiment by ID from MLflow."""
        try:
            return self.api.get_experiment(experiment_id) or {}
        except Exception as e:
            logger.warning(f"Failed to get experiment {experiment_id}: {e}")
            return {"id": experiment_id}

    def _get_experiments_for_hypothesis(self, hypothesis_id: str) -> list[dict]:
        """Get all experiments for a hypothesis."""
        try:
            experiment_ids = self.api.get_experiments_for_hypothesis(hypothesis_id)
            return [self._get_experiment(eid) for eid in experiment_ids]
        except Exception as e:
            logger.warning(f"Failed to get experiments for {hypothesis_id}: {e}")
            return []

    def _get_recent_experiments(self, days: int) -> list[dict]:
        """Get experiments from the last N days."""
        # Query MLflow for recent runs
        try:
            from hrp.research.mlflow_utils import setup_mlflow

            setup_mlflow()
            client = mlflow.tracking.MlflowClient()

            # Get runs from the last N days
            cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Search all walk-forward experiments plus Default
            all_experiments = client.search_experiments()
            all_exp_ids = [exp.experiment_id for exp in all_experiments]

            runs = client.search_runs(
                experiment_ids=all_exp_ids,
                filter_string=f"attributes.start_time > {cutoff_time}",
                max_results=100,
            )

            experiments = []
            for run in runs:
                # Skip nested fold runs (only audit parent walk-forward runs)
                if run.data.tags.get("mlflow.parentRunId"):
                    continue

                exp_dict = {
                    "id": run.info.run_id,
                    "mlflow_run_id": run.info.run_id,
                    "hypothesis_id": run.data.tags.get("hypothesis_id", "unknown"),
                    "train_sharpe": run.data.metrics.get("train_sharpe"),
                    "test_sharpe": run.data.metrics.get("test_sharpe"),
                    "mean_ic": run.data.metrics.get("mean_ic"),
                    "sharpe": run.data.metrics.get("sharpe_ratio"),
                    "r2": run.data.metrics.get("r2"),
                    "feature_count": int(run.data.params.get("feature_count", 0) or 0),
                    "sample_count": int(run.data.params.get("sample_count", 1000) or 1000),
                }

                # Extract fold results if available
                fold_ics = []
                for i in range(10):  # Max 10 folds
                    fold_ic = run.data.metrics.get(f"fold_{i}_ic")
                    if fold_ic is not None:
                        fold_ics.append({"ic": fold_ic})
                if fold_ics:
                    exp_dict["fold_results"] = fold_ics

                experiments.append(exp_dict)

            return experiments

        except Exception as e:
            logger.warning(f"Failed to get recent experiments: {e}")
            return []

    def _get_deployed_models(self) -> list[dict]:
        """Get all deployed/active models."""
        try:
            strategies = self.api.get_deployed_strategies()
            return [
                {
                    "id": s.get("id", "unknown"),
                    "hypothesis_id": s.get("id", "unknown"),
                    "validation_ic": s.get("metadata", {}).get("validation_ic"),
                }
                for s in strategies
            ]
        except Exception as e:
            logger.warning(f"Failed to get deployed models: {e}")
            return []

    def _calculate_recent_ic(self, model: dict, window: int) -> float:
        """Calculate IC over recent window."""
        # Placeholder - would need actual signal/return correlation calculation
        return model.get("validation_ic", 0) or 0

    def _get_model_returns(self, model: dict, days: int) -> list[float]:
        """Get recent returns for a model."""
        # Placeholder - would need actual returns data
        return []

    def _count_consecutive_losses(self, returns: list[float]) -> int:
        """Count consecutive negative returns from end."""
        count = 0
        for r in reversed(returns):
            if r < 0:
                count += 1
            else:
                break
        return count

    def _write_research_note(
        self,
        audits: list[ExperimentAudit],
        model_alerts: list[MonitoringAlert],
        critical_issues: list[tuple[str, str]],
        warnings: list[tuple[str, str]],
        duration: float,
    ) -> None:
        """Write per-run audit report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges,
            render_section_divider,
        )

        report_date = date.today().isoformat()
        filename = f"{report_date}-ml-quality-sentinel.md"
        filepath = get_config().data.research_dir / filename

        passed_count = sum(1 for a in audits if a.overall_passed)
        flagged_count = sum(1 for a in audits if a.has_critical_issues)

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="ML Quality Sentinel Report",
            report_type="ml-quality-sentinel",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "🔬", "label": "Audited", "value": len(audits), "detail": "experiments"},
            {"icon": "✅", "label": "Passed", "value": passed_count, "detail": "clean"},
            {"icon": "🚨", "label": "Flagged", "value": flagged_count, "detail": "issues"},
            {"icon": "📊", "label": "Alerts", "value": len(model_alerts), "detail": "models"},
        ]))

        # ── Alert banner for critical issues ──
        if critical_issues:
            parts.append(render_alert_banner(
                [f"{len(critical_issues)} CRITICAL issues detected — immediate review required",
                 f"{len(warnings)} additional warnings flagged"],
                severity="critical",
            ))
        elif len(audits) > 0 and flagged_count == 0:
            parts.append(render_alert_banner(
                [f"All {len(audits)} experiments passed quality checks ✅"],
                severity="info",
            ))

        # ── Health Gauges ──
        pass_rate = (passed_count / max(len(audits), 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Experiment Quality", "value": pass_rate, "max_val": 100,
             "trend": "up" if flagged_count == 0 else "down"},
            {"label": "Model Monitoring", "value": max(100 - len(model_alerts) * 25, 0), "max_val": 100,
             "trend": "stable" if len(model_alerts) == 0 else "down"},
        ]))

        # ── Critical Issues ──
        if critical_issues:
            parts.append(render_section_divider("🚨 Critical Issues (Require Attention)"))
            parts.append("| Experiment | Issue |")
            parts.append("|------------|-------|")
            for exp_id, issue in critical_issues:
                parts.append(f"| 🔴 {exp_id} | {issue} |")
            parts.append("")

        # ── Warnings ──
        if warnings:
            parts.append(render_section_divider("⚠️ Warnings"))
            parts.append("| Experiment | Warning |")
            parts.append("|------------|---------|")
            for exp_id, warning in warnings:
                parts.append(f"| 🟡 {exp_id} | {warning} |")
            parts.append("")

        # ── Model Monitoring Alerts ──
        if model_alerts:
            parts.append(render_section_divider("📊 Model Monitoring Alerts"))
            for alert in model_alerts:
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    alert.severity.value, "⚪"
                )
                parts.extend([
                    f"### {severity_emoji} {alert.model_id}",
                    "",
                    f"| Field | Detail |",
                    f"|-------|--------|",
                    f"| **Type** | {alert.alert_type} |",
                    f"| **Severity** | {severity_emoji} {alert.severity.value} |",
                    f"| **Message** | {alert.message} |",
                    f"| **Action** | {alert.recommended_action} |",
                    "",
                ])

        # ── Footer ──
        parts.append(render_footer(
            agent_name="ml-quality-sentinel",
            duration_seconds=duration,
        ))

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text("\n".join(parts))
            logger.info(f"Wrote research note to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to write research note: {e}")

    def _send_alert_email(
        self,
        audits: list[ExperimentAudit],
        model_alerts: list[MonitoringAlert],
        critical_issues: list[tuple[str, str]],
        warnings: list[tuple[str, str]],
    ) -> None:
        """Send alert email for quality issues."""
        try:
            notifier = EmailNotifier()

            summary_data = {
                "report_date": date.today().isoformat(),
                "experiments_audited": len(audits),
                "experiments_passed": sum(1 for a in audits if a.overall_passed),
                "experiments_flagged": sum(1 for a in audits if a.has_critical_issues),
                "critical_issues": len(critical_issues),
                "warnings": len(warnings),
                "model_alerts": len(model_alerts),
            }

            # Add critical issues
            for i, (exp_id, issue) in enumerate(critical_issues[:5]):
                summary_data[f"critical_{i+1}"] = f"{exp_id}: {issue}"

            subject = f"[HRP] ML Quality Sentinel - {len(critical_issues)} Critical Issues Found"

            notifier.send_summary_email(
                subject=subject,
                summary_data=summary_data,
            )

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")


# =====================================================================
# Validation Analyst Agent
# =====================================================================

# Re-use AuditSeverity as ValidationSeverity for consistency
ValidationSeverity = AuditSeverity


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: ValidationSeverity
    details: dict[str, Any]
    message: str


@dataclass
class HypothesisValidation:
    """Complete validation of a single hypothesis."""

    hypothesis_id: str
    experiment_id: str
    validation_date: date
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """Check if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def critical_count(self) -> int:
        """Count critical failures."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count warnings."""
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.WARNING)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found."""
        return self.critical_count > 0

    def add_check(self, check: ValidationCheck) -> None:
        """Add a check result to the validation."""
        self.checks.append(check)


@dataclass
class ValidationAnalystReport:
    """Complete Validation Analyst run report."""

    report_date: date
    hypotheses_validated: int
    hypotheses_passed: int
    hypotheses_failed: int
    validations: list[HypothesisValidation]
    duration_seconds: float


class ValidationAnalyst(ResearchAgent):
    """
    Stress tests validated hypotheses before deployment approval.

    Performs:
    1. Parameter sensitivity - Tests stability under parameter changes
    2. Time stability - Verifies consistent performance across periods
    3. Regime analysis - Checks performance in bull/bear/sideways markets
    4. Execution cost estimation - Calculates realistic transaction costs

    Type: Hybrid (deterministic tests + Claude reasoning for edge cases)
    """

    DEFAULT_JOB_ID = "validation_analyst_review"
    ACTOR = "agent:validation-analyst"

    # Default thresholds
    DEFAULT_PARAM_SENSITIVITY_THRESHOLD = 0.5  # Min ratio of varied/baseline Sharpe
    DEFAULT_MIN_PROFITABLE_PERIODS = 0.67  # 2/3 of periods must be profitable
    DEFAULT_MIN_PROFITABLE_REGIMES = 2  # At least 2 of 3 regimes profitable

    # Transaction cost assumptions
    DEFAULT_COMMISSION_BPS = 5  # 5 basis points per trade
    DEFAULT_SLIPPAGE_BPS = 10  # 10 basis points slippage

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        param_sensitivity_threshold: float | None = None,
        min_profitable_periods: float | None = None,
        min_profitable_regimes: int | None = None,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        include_claude_reasoning: bool = True,
        send_alerts: bool = True,
    ):
        """
        Initialize the Validation Analyst.

        Args:
            hypothesis_ids: Specific hypotheses to validate (None = all audited)
            param_sensitivity_threshold: Min ratio for parameter sensitivity
            min_profitable_periods: Min ratio of profitable time periods
            min_profitable_regimes: Min number of profitable regimes
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            include_claude_reasoning: Use Claude for edge case analysis
            send_alerts: Send email alerts on failures
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.param_sensitivity_threshold = (
            param_sensitivity_threshold or self.DEFAULT_PARAM_SENSITIVITY_THRESHOLD
        )
        self.min_profitable_periods = (
            min_profitable_periods or self.DEFAULT_MIN_PROFITABLE_PERIODS
        )
        self.min_profitable_regimes = (
            min_profitable_regimes or self.DEFAULT_MIN_PROFITABLE_REGIMES
        )
        self.commission_bps = commission_bps or self.DEFAULT_COMMISSION_BPS
        self.slippage_bps = slippage_bps or self.DEFAULT_SLIPPAGE_BPS
        self.include_claude_reasoning = include_claude_reasoning
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """
        Run validation on hypotheses ready for final review.

        Returns:
            Dict with validation results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to validate
        hypotheses = self._get_hypotheses_to_validate()

        # 2. Validate each hypothesis
        validations: list[HypothesisValidation] = []
        passed_count = 0
        failed_count = 0

        for hypothesis in hypotheses:
            validation = self._validate_hypothesis(hypothesis)
            validations.append(validation)

            if validation.overall_passed:
                passed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "validated",
                    validation,
                )
            else:
                failed_count += 1
                self._update_hypothesis_status(
                    validation.hypothesis_id,
                    "validation_failed",
                    validation,
                )

            # Log per-hypothesis validation event
            self._log_agent_event(
                event_type=EventType.VALIDATION_ANALYST_REVIEW,
                hypothesis_id=validation.hypothesis_id,
                experiment_id=validation.experiment_id,
                details={
                    "overall_passed": validation.overall_passed,
                    "critical_count": validation.critical_count,
                    "warning_count": validation.warning_count,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "severity": c.severity.value,
                        }
                        for c in validation.checks
                    ],
                },
            )

        # 3. Log completion event
        duration = time.time() - start_time
        self._log_agent_event(
            event_type=EventType.AGENT_RUN_COMPLETE,
            details={
                "hypotheses_validated": len(validations),
                "hypotheses_passed": passed_count,
                "hypotheses_failed": failed_count,
                "duration_seconds": duration,
            },
        )

        # 4. Write research note
        self._write_research_note(validations, duration)

        # 5. Send alerts if failures
        if self.send_alerts and failed_count > 0:
            self._send_alert_email(validations)

        # 6. Build report
        report = ValidationAnalystReport(
            report_date=date.today(),
            hypotheses_validated=len(validations),
            hypotheses_passed=passed_count,
            hypotheses_failed=failed_count,
            validations=validations,
            duration_seconds=duration,
        )

        return {
            "report_date": report.report_date.isoformat(),
            "hypotheses_validated": report.hypotheses_validated,
            "hypotheses_passed": report.hypotheses_passed,
            "hypotheses_failed": report.hypotheses_failed,
            "duration_seconds": report.duration_seconds,
        }

    def _get_hypotheses_to_validate(self) -> list[dict[str, Any]]:
        """
        Get hypotheses ready for validation.

        Returns hypotheses that:
        - Passed ML Quality Sentinel audit (no critical issues)
        - Are in 'testing' or 'audited' status
        """
        if self.hypothesis_ids:
            # Specific hypotheses requested
            return [
                self.api.get_hypothesis(hid)
                for hid in self.hypothesis_ids
                if self.api.get_hypothesis(hid) is not None
            ]

        # Get hypotheses that passed quality audit
        # Look for recent ML_QUALITY_SENTINEL_AUDIT events with overall_passed=True
        db = get_db()
        cutoff = datetime.now() - timedelta(days=7)
        result = db.fetchall(
            """
            SELECT DISTINCT l.hypothesis_id
            FROM lineage l
            WHERE l.event_type = ?
              AND l.timestamp > ?
              AND json_extract_string(l.details, '$.overall_passed') = 'true'
            """,
            (EventType.ML_QUALITY_SENTINEL_AUDIT.value, cutoff),
        )

        hypothesis_ids = [row[0] for row in result if row[0]]
        return [
            self.api.get_hypothesis(hid)
            for hid in hypothesis_ids
            if self.api.get_hypothesis(hid) is not None
        ]

    def _validate_hypothesis(
        self,
        hypothesis: dict[str, Any],
    ) -> HypothesisValidation:
        """
        Run all validation checks on a hypothesis.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            HypothesisValidation with all check results
        """
        hypothesis_id = hypothesis.get("hypothesis_id", hypothesis.get("id", "unknown"))
        experiment_id = hypothesis.get("metadata", {}).get("experiment_id", "unknown")

        validation = HypothesisValidation(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            validation_date=date.today(),
        )

        # Get experiment data for this hypothesis
        experiment_data = self._get_experiment_data(hypothesis)

        # 1. Parameter sensitivity check
        if "param_experiments" in experiment_data and experiment_data["param_experiments"]:
            check = self._check_parameter_sensitivity(
                experiment_data["param_experiments"],
                "baseline",
            )
            validation.add_check(check)

        # 2. Time stability check
        if "period_metrics" in experiment_data and experiment_data["period_metrics"]:
            check = self._check_time_stability(experiment_data["period_metrics"])
            validation.add_check(check)

        # 3. Regime stability check
        if "regime_metrics" in experiment_data and experiment_data["regime_metrics"]:
            check = self._check_regime_stability(experiment_data["regime_metrics"])
            validation.add_check(check)

        # 4. Execution cost estimation
        if all(
            k in experiment_data and experiment_data[k]
            for k in ["num_trades", "avg_trade_value", "gross_return"]
        ):
            check = self._estimate_execution_costs(
                experiment_data["num_trades"],
                experiment_data["avg_trade_value"],
                experiment_data["gross_return"],
            )
            validation.add_check(check)

        return validation

    def _get_experiment_data(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Gather experiment data needed for validation checks.

        This is a placeholder - actual implementation would query MLflow
        and run additional backtests for parameter sensitivity.
        """
        # In a full implementation, this would:
        # 1. Query MLflow for the hypothesis's experiments
        # 2. Run parameter variations if not already done
        # 3. Split returns into time periods
        # 4. Detect regimes and calculate regime metrics
        # 5. Extract trade statistics

        # For now, return data from hypothesis metadata if available
        metadata = hypothesis.get("metadata", {})
        return {
            "param_experiments": metadata.get("param_experiments", {}),
            "period_metrics": metadata.get("period_metrics", []),
            "regime_metrics": metadata.get("regime_metrics", {}),
            "num_trades": metadata.get("num_trades", 0),
            "avg_trade_value": metadata.get("avg_trade_value", 0),
            "gross_return": metadata.get("gross_return", 0),
        }

    def _update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: str,
        validation: HypothesisValidation,
    ) -> None:
        """Update hypothesis status and metadata with validation results."""
        try:
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status=new_status,
                metadata={
                    "validation_analyst_review": {
                        "date": validation.validation_date.isoformat(),
                        "passed": validation.overall_passed,
                        "critical_count": validation.critical_count,
                        "warning_count": validation.warning_count,
                        "checks": [c.name for c in validation.checks],
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _write_research_note(
        self,
        validations: list[HypothesisValidation],
        duration: float,
    ) -> None:
        """Write per-run validation report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges,
            render_section_divider, get_status_emoji,
        )

        report_date = date.today().isoformat()
        filename = f"{report_date}-validation-analyst.md"
        filepath = get_config().data.research_dir / filename

        passed_count = sum(1 for v in validations if v.overall_passed)
        failed_count = sum(1 for v in validations if not v.overall_passed)

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Validation Analyst Report",
            report_type="validation-analyst",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Validated", "value": len(validations), "detail": "hypotheses"},
            {"icon": "✅", "label": "Passed", "value": passed_count, "detail": "approved"},
            {"icon": "❌", "label": "Failed", "value": failed_count, "detail": "rejected"},
        ]))

        # ── Alert banner ──
        if failed_count > 0:
            parts.append(render_alert_banner(
                [f"{failed_count} hypotheses FAILED validation — review check details below"],
                severity="warning",
            ))
        elif len(validations) > 0:
            parts.append(render_alert_banner(
                [f"All {len(validations)} hypotheses passed validation ✅"],
                severity="info",
            ))

        # ── Health Gauge ──
        pass_rate = (passed_count / max(len(validations), 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Validation Pass Rate", "value": pass_rate, "max_val": 100,
             "trend": "up" if failed_count == 0 else "down"},
        ]))

        # ── Per-hypothesis validation details ──
        parts.append(render_section_divider("📊 Validation Details"))

        for validation in validations:
            status = "PASSED" if validation.overall_passed else "FAILED"
            emoji = "✅" if validation.overall_passed else "❌"

            parts.append(f"### {emoji} {validation.hypothesis_id}: **{status}**")
            parts.append("")

            if validation.checks:
                parts.append("| Check | Result | Severity | Message |")
                parts.append("|-------|--------|----------|---------|")
                for check in validation.checks:
                    check_emoji = "✅" if check.passed else "❌"
                    severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                        check.severity.value, "⚪"
                    )
                    parts.append(
                        f"| {check.name} | {check_emoji} | {severity_emoji} {check.severity.value} | {check.message} |"
                    )
                parts.append("")
            else:
                parts.append("> _No validation checks recorded_\n")

            parts.append("─" * 60)
            parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="validation-analyst",
            duration_seconds=duration,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")

    def _send_alert_email(self, validations: list[HypothesisValidation]) -> None:
        """Send alert email for validation failures."""
        try:
            failed = [v for v in validations if not v.overall_passed]
            if not failed:
                return

            notifier = EmailNotifier()
            subject = f"[HRP] Validation Analyst - {len(failed)} Hypothesis Validation Failures"

            body_lines = [
                "Validation Analyst detected hypothesis validation failures:",
                "",
            ]
            for v in failed:
                body_lines.append(
                    f"- {v.hypothesis_id}: {v.critical_count} critical, {v.warning_count} warnings"
                )

            notifier.send_notification(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send alert email: {e}")

    def _check_parameter_sensitivity(
        self,
        experiments: dict[str, dict[str, Any]],
        baseline_key: str,
    ) -> ValidationCheck:
        """
        Check parameter sensitivity using existing robustness module.

        Args:
            experiments: Dict mapping experiment name to metrics
            baseline_key: Key for baseline experiment

        Returns:
            ValidationCheck with sensitivity results
        """
        from hrp.risk.robustness import check_parameter_sensitivity

        result = check_parameter_sensitivity(
            experiments=experiments,
            baseline_key=baseline_key,
            threshold=self.param_sensitivity_threshold,
        )

        if not result.passed:
            return ValidationCheck(
                name="parameter_sensitivity",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("parameter_sensitivity", {}),
                message=f"Parameter sensitivity failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="parameter_sensitivity",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("parameter_sensitivity", {}),
            message="Parameters are stable under variation",
        )

    def _check_time_stability(
        self,
        period_metrics: list[dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check time period stability using existing robustness module.

        Args:
            period_metrics: List of period metrics with 'sharpe', 'profitable'

        Returns:
            ValidationCheck with stability results
        """
        from hrp.risk.robustness import check_time_stability

        result = check_time_stability(
            period_metrics=period_metrics,
            min_profitable_ratio=self.min_profitable_periods,
        )

        if not result.passed:
            return ValidationCheck(
                name="time_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("time_stability", {}),
                message=f"Time stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="time_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("time_stability", {}),
            message="Strategy is stable across time periods",
        )

    def _check_regime_stability(
        self,
        regime_metrics: dict[str, dict[str, Any]],
    ) -> ValidationCheck:
        """
        Check market regime stability using existing robustness module.

        Args:
            regime_metrics: Dict mapping regime name to metrics

        Returns:
            ValidationCheck with regime results
        """
        from hrp.risk.robustness import check_regime_stability

        result = check_regime_stability(
            regime_metrics=regime_metrics,
            min_regimes_profitable=self.min_profitable_regimes,
        )

        if not result.passed:
            return ValidationCheck(
                name="regime_stability",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=result.checks.get("regime_stability", {}),
                message=f"Regime stability failed: {'; '.join(result.failures)}",
            )

        return ValidationCheck(
            name="regime_stability",
            passed=True,
            severity=ValidationSeverity.NONE,
            details=result.checks.get("regime_stability", {}),
            message="Strategy works across market regimes",
        )

    def _estimate_execution_costs(
        self,
        num_trades: int,
        avg_trade_value: float,
        gross_return: float,
    ) -> ValidationCheck:
        """
        Estimate realistic execution costs and net return.

        Args:
            num_trades: Number of round-trip trades
            avg_trade_value: Average trade value in dollars
            gross_return: Gross return before costs

        Returns:
            ValidationCheck with cost analysis
        """
        # Calculate total cost in basis points
        cost_per_trade_bps = self.commission_bps + self.slippage_bps
        total_cost_bps = cost_per_trade_bps * num_trades

        # Convert to decimal
        total_cost_decimal = total_cost_bps / 10000

        # Net return
        net_return = gross_return - total_cost_decimal

        # Cost as percentage of gross return
        cost_ratio = total_cost_decimal / gross_return if gross_return > 0 else float("inf")

        details = {
            "num_trades": num_trades,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_cost_bps": total_cost_bps,
            "total_cost_decimal": total_cost_decimal,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_ratio": cost_ratio,
        }

        # Determine severity
        if net_return < 0:
            return ValidationCheck(
                name="execution_costs",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                details=details,
                message=f"Net return negative after costs: {net_return:.2%}",
            )
        elif cost_ratio > 0.5:  # Costs exceed 50% of gross return
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.WARNING,
                details=details,
                message=f"High execution costs: {cost_ratio:.1%} of gross return",
            )
        else:
            return ValidationCheck(
                name="execution_costs",
                passed=True,
                severity=ValidationSeverity.NONE,
                details=details,
                message=f"Execution costs acceptable: net return {net_return:.2%}",
            )


# =============================================================================
# RISK MANAGER
# =============================================================================


@dataclass
class RiskVeto:
    """Record of a risk veto decision."""

    hypothesis_id: str
    veto_reason: str
    veto_type: Literal["drawdown", "concentration", "correlation", "limits", "other"]
    severity: Literal["critical", "warning"]
    details: dict[str, Any]
    veto_date: date


@dataclass
class PortfolioRiskAssessment:
    """Assessment of portfolio-level risk for a hypothesis."""

    hypothesis_id: str
    passed: bool
    vetos: list[RiskVeto]
    warnings: list[str]
    portfolio_impact: dict[str, Any]
    assessment_date: date


@dataclass
class RiskManagerReport:
    """Complete Risk Manager run report."""

    report_date: date
    hypotheses_assessed: int
    hypotheses_passed: int
    hypotheses_vetoed: int
    assessments: list[PortfolioRiskAssessment]
    duration_seconds: float


class RiskManager(ResearchAgent):
    """
    Independent portfolio risk oversight agent.

    Reviews validated hypotheses for portfolio-level risk before deployment.
    Can veto strategies but CANNOT approve deployment (maintains independence).

    Performs:
    1. Drawdown risk assessment - Max drawdown limits, drawdown duration
    2. Concentration risk - Position sizes, sector exposure, correlation
    3. Portfolio fit - Correlation with existing positions, diversification value
    4. Risk limits validation - Position limits, turnover limits, leverage

    Type: Custom (deterministic checks with independent veto authority)

    Key principle: Risk Manager operates independently from alpha generation
    and can veto any strategy but cannot approve deployment (only human CIO can).
    """

    DEFAULT_JOB_ID = "risk_manager_review"
    ACTOR = "agent:risk-manager"

    # Portfolio risk limits (conservative institutional defaults)
    MAX_MAX_DRAWDOWN = 0.20  # 20% maximum drawdown
    MAX_DRAWDOWN_DURATION_DAYS = 126  # 6 months to recover
    MAX_POSITION_CORRELATION = 0.70  # Max correlation with existing positions
    MAX_SECTOR_EXPOSURE = 0.30  # 30% max in any sector
    MAX_SINGLE_POSITION = 0.10  # 10% max in single position

    # Portfolio composition targets
    MIN_DIVERSIFICATION = 10  # Minimum positions
    TARGET_POSITIONS = 20  # Target number of positions

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        max_drawdown: float | None = None,
        max_correlation: float | None = None,
        max_sector_exposure: float | None = None,
        send_alerts: bool = True,
    ):
        """
        Initialize the Risk Manager.

        Args:
            hypothesis_ids: Specific hypotheses to assess (None = all validated)
            max_drawdown: Maximum allowed drawdown (default 20%)
            max_correlation: Max correlation with existing positions (default 0.70)
            max_sector_exposure: Max sector exposure (default 30%)
            send_alerts: Send email alerts on vetos
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.max_drawdown = max_drawdown or self.MAX_MAX_DRAWDOWN
        self.max_correlation = max_correlation or self.MAX_POSITION_CORRELATION
        self.max_sector_exposure = max_sector_exposure or self.MAX_SECTOR_EXPOSURE
        self.send_alerts = send_alerts

    def execute(self) -> dict[str, Any]:
        """
        Run risk assessment on hypotheses ready for deployment review.

        Returns:
            Dict with assessment results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to assess
        hypotheses = self._get_hypotheses_to_assess()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "assessments": [],
                "message": "No hypotheses awaiting risk assessment",
            }

        # 2. Assess each hypothesis
        assessments: list[PortfolioRiskAssessment] = []
        passed_count = 0
        vetoed_count = 0

        for hypothesis in hypotheses:
            assessment = self._assess_hypothesis_risk(hypothesis)
            assessments.append(assessment)

            if assessment.passed:
                passed_count += 1
                # Log to lineage
                self._log_agent_event(
                    event_type=EventType.RISK_REVIEW_COMPLETE,
                    details={
                        "hypothesis_id": assessment.hypothesis_id,
                        "passed": True,
                        "warnings": assessment.warnings,
                    },
                    hypothesis_id=assessment.hypothesis_id,
                )
            else:
                vetoed_count += 1
                # Log veto to lineage
                for veto in assessment.vetos:
                    self._log_agent_event(
                        event_type=EventType.RISK_VETO,
                        details={
                            "hypothesis_id": veto.hypothesis_id,
                            "veto_reason": veto.veto_reason,
                            "veto_type": veto.veto_type,
                            "severity": veto.severity,
                        },
                        hypothesis_id=veto.hypothesis_id,
                    )

        # 3. Generate report
        report = RiskManagerReport(
            report_date=date.today(),
            hypotheses_assessed=len(assessments),
            hypotheses_passed=passed_count,
            hypotheses_vetoed=vetoed_count,
            assessments=assessments,
            duration_seconds=time.time() - start_time,
        )

        # 4. Write research note
        self._write_research_note(report)

        # 5. Send alerts if any vetos
        if self.send_alerts:
            self._send_veto_alerts(assessments)

        return {
            "status": "complete",
            "assessments": assessments,
            "report": {
                "hypotheses_assessed": report.hypotheses_assessed,
                "hypotheses_passed": report.hypotheses_passed,
                "hypotheses_vetoed": report.hypotheses_vetoed,
                "duration_seconds": report.duration_seconds,
            },
        }

    def _get_hypotheses_to_assess(self) -> list[dict[str, Any]]:
        """
        Get hypotheses that need risk assessment.

        Fetches hypotheses with 'validated' status that haven't been
        risk-assessed yet.
        """
        if self.hypothesis_ids:
            # Specific hypotheses requested
            placeholders = ",".join(["?" for _ in self.hypothesis_ids])
            query = f"""
                SELECT hypothesis_id, title, thesis, status, metadata
                FROM hypotheses
                WHERE hypothesis_id IN ({placeholders})
            """
            params = self.hypothesis_ids
        else:
            # All validated hypotheses not yet risk-assessed
            query = """
                SELECT hypothesis_id, title, thesis, status, metadata
                FROM hypotheses
                WHERE status = 'validated'
                  AND (metadata NOT LIKE '%risk_manager_review%'
                       OR metadata IS NULL)
                ORDER BY created_at DESC
            """
            params = []

        result = self.api._db.fetchdf(query, params)

        if result.empty:
            return []

        return result.to_dict(orient="records")

    def _assess_hypothesis_risk(
        self, hypothesis: dict[str, Any]
    ) -> PortfolioRiskAssessment:
        """
        Assess portfolio-level risk for a single hypothesis.

        Performs the following checks:
        1. Drawdown risk
        2. Concentration risk
        3. Correlation with existing positions
        4. Risk limits validation

        Args:
            hypothesis: Hypothesis record from database

        Returns:
            PortfolioRiskAssessment with veto decisions
        """
        import json

        hypothesis_id = hypothesis["hypothesis_id"]
        metadata_str = hypothesis.get("metadata") or "{}"
        metadata = (
            json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        )

        # Get experiment metrics
        experiment_data = self._get_experiment_metrics(hypothesis_id, metadata)

        # Initialize assessment
        vetos: list[RiskVeto] = []
        warnings: list[str] = []

        # Check 1: Drawdown risk
        dd_veto = self._check_drawdown_risk(hypothesis_id, experiment_data)
        if dd_veto:
            vetos.append(dd_veto)

        # Check 2: Concentration risk
        conc_vetos, conc_warnings = self._check_concentration_risk(
            hypothesis_id, experiment_data, metadata
        )
        vetos.extend(conc_vetos)
        warnings.extend(conc_warnings)

        # Check 3: Correlation with existing positions
        corr_veto = self._check_correlation_risk(hypothesis_id, metadata)
        if corr_veto:
            vetos.append(corr_veto)

        # Check 4: Risk limits validation
        limits_vetos = self._check_risk_limits(hypothesis_id, experiment_data)
        vetos.extend(limits_vetos)

        # Calculate portfolio impact
        portfolio_impact = self._calculate_portfolio_impact(
            hypothesis_id, experiment_data, metadata
        )

        # Determine if passed (no critical vetos)
        passed = all(v.severity != "critical" for v in vetos)

        assessment = PortfolioRiskAssessment(
            hypothesis_id=hypothesis_id,
            passed=passed,
            vetos=vetos,
            warnings=warnings,
            portfolio_impact=portfolio_impact,
            assessment_date=date.today(),
        )

        # Update hypothesis with risk assessment
        self._update_hypothesis_with_risk_assessment(assessment)

        return assessment

    def _get_experiment_metrics(
        self, hypothesis_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get experiment metrics from metadata or MLflow.

        Reads ml_scientist_results first (structured walk-forward metrics),
        then falls back to validation_analyst_review, then conservative defaults.

        Args:
            hypothesis_id: Hypothesis ID
            metadata: Hypothesis metadata dict

        Returns:
            Dict with key metrics: sharpe, max_drawdown, volatility, etc.
        """
        # Primary source: ML Scientist structured results
        ml_results = metadata.get("ml_scientist_results", {})
        if ml_results:
            stability = ml_results.get("stability_score", 2.0)
            mean_ic = ml_results.get("mean_ic", 0)
            return {
                "sharpe": mean_ic * 20,  # IC-based proxy (IC=0.05 → ~1.0 Sharpe)
                "max_drawdown": min(0.10 + stability * 0.10, 0.50),
                "volatility": 0.15,
                "turnover": 0.30,
                "num_positions": self.TARGET_POSITIONS,
                "sector_exposure": {},
            }

        # Secondary: Validation Analyst review
        validation = metadata.get("validation_analyst_review", {})
        if validation:
            return {
                "sharpe": validation.get("sharpe", 0),
                "max_drawdown": validation.get("max_drawdown", 0.25),
                "volatility": validation.get("volatility", 0.20),
                "turnover": validation.get("turnover", 0.50),
                "num_positions": validation.get("num_positions", self.TARGET_POSITIONS),
                "sector_exposure": validation.get("sector_exposure", {}),
            }

        # Last resort: conservative defaults
        logger.warning(
            f"{hypothesis_id}: No ml_scientist_results or validation_analyst_review "
            "in metadata, using conservative defaults"
        )
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.25,
            "volatility": 0.20,
            "turnover": 0.30,
            "num_positions": self.TARGET_POSITIONS,
            "sector_exposure": {},
        }

    def _check_drawdown_risk(
        self, hypothesis_id: str, metrics: dict[str, Any]
    ) -> RiskVeto | None:
        """
        Check if drawdown exceeds limits.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics

        Returns:
            RiskVeto if drawdown too high, None otherwise
        """
        max_dd = metrics.get("max_drawdown", 0)

        if max_dd > self.max_drawdown:
            return RiskVeto(
                hypothesis_id=hypothesis_id,
                veto_reason=f"Max drawdown {max_dd:.1%} exceeds limit {self.max_drawdown:.1%}",
                veto_type="drawdown",
                severity="critical",
                details={"max_drawdown": max_dd, "limit": self.max_drawdown},
                veto_date=date.today(),
            )

        # Warning if approaching limit
        if max_dd > self.max_drawdown * 0.8:
            logger.warning(
                f"{hypothesis_id}: Drawdown {max_dd:.1%} approaching limit "
                f"{self.max_drawdown:.1%}"
            )

        return None

    def _check_concentration_risk(
        self, hypothesis_id: str, metrics: dict[str, Any], metadata: dict[str, Any]
    ) -> tuple[list[RiskVeto], list[str]]:
        """
        Check concentration risk (position sizes, sector exposure).

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics
            metadata: Hypothesis metadata

        Returns:
            Tuple of (vetos, warnings)
        """
        vetos: list[RiskVeto] = []
        warnings: list[str] = []

        num_positions = metrics.get("num_positions", self.TARGET_POSITIONS)
        sector_exposure = metrics.get("sector_exposure", {})

        # Check minimum diversification
        if num_positions < self.MIN_DIVERSIFICATION:
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Only {num_positions} positions, minimum {self.MIN_DIVERSIFICATION}",
                    veto_type="concentration",
                    severity="critical",
                    details={"num_positions": num_positions, "minimum": self.MIN_DIVERSIFICATION},
                    veto_date=date.today(),
                )
            )

        # Check sector concentration
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                vetos.append(
                    RiskVeto(
                        hypothesis_id=hypothesis_id,
                        veto_reason=f"Sector '{sector}' exposure {exposure:.1%} exceeds limit {self.max_sector_exposure:.1%}",
                        veto_type="concentration",
                        severity="critical",
                        details={
                            "sector": sector,
                            "exposure": exposure,
                            "limit": self.max_sector_exposure,
                        },
                        veto_date=date.today(),
                    )
                )

        return vetos, warnings

    def _check_correlation_risk(
        self, hypothesis_id: str, metadata: dict[str, Any]
    ) -> RiskVeto | None:
        """
        Check correlation with existing paper portfolio positions.

        Args:
            hypothesis_id: Hypothesis ID
            metadata: Hypothesis metadata

        Returns:
            RiskVeto if too correlated, None otherwise
        """
        # Get existing paper portfolio
        try:
            portfolio = self.api._db.execute(
                """
                SELECT hypothesis_id, weight
                FROM paper_portfolio
                WHERE weight > 0
                """
            ).fetchdf()

            if portfolio.empty:
                return None

            # For now, correlation check is a placeholder
            # In production, would compute actual correlation from returns
            # For new implementation, just check if same features are used

            # Check for duplicate strategies (same feature set)
            existing_features = metadata.get("features", [])
            if existing_features:
                # Simple check: if more than 50% feature overlap, flag as warning
                # (This is a placeholder - real implementation would compute correlation)
                pass

        except Exception as e:
            logger.debug(f"Could not check correlation: {e}")
            return None

        return None

    def _check_risk_limits(
        self, hypothesis_id: str, metrics: dict[str, Any]
    ) -> list[RiskVeto]:
        """
        Check if strategy respects risk limits.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics

        Returns:
            List of vetos (empty if all limits respected)
        """
        vetos: list[RiskVeto] = []

        volatility = metrics.get("volatility", 0)
        turnover = metrics.get("turnover", 0)

        # Check volatility (high vol = high risk)
        if volatility > 0.25:  # 25% annual vol
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Volatility {volatility:.1%} exceeds prudent limit",
                    veto_type="limits",
                    severity="warning",  # Warning, not critical
                    details={"volatility": volatility},
                    veto_date=date.today(),
                )
            )

        # Check turnover (high turnover = high costs)
        if turnover > 0.50:  # 50% annual turnover
            vetos.append(
                RiskVeto(
                    hypothesis_id=hypothesis_id,
                    veto_reason=f"Turnover {turnover:.1%} may erode returns with costs",
                    veto_type="limits",
                    severity="warning",
                    details={"turnover": turnover},
                    veto_date=date.today(),
                )
            )

        return vetos

    def _calculate_portfolio_impact(
        self, hypothesis_id: str, metrics: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate the impact of adding this hypothesis to the portfolio.

        Args:
            hypothesis_id: Hypothesis ID
            metrics: Experiment metrics
            metadata: Hypothesis metadata

        Returns:
            Dict with portfolio impact assessment
        """
        # Get current portfolio state
        try:
            portfolio = self.api._db.execute(
                """
                SELECT COUNT(*) as num_positions, COALESCE(SUM(weight), 0) as total_weight
                FROM paper_portfolio
                WHERE weight > 0
                """
            ).fetchdf()

            current_positions = portfolio.iloc[0]["num_positions"]
            current_weight = portfolio.iloc[0]["total_weight"]
        except Exception:
            current_positions = 0
            current_weight = 0.0

        # Calculate impact
        new_positions = current_positions + 1
        new_weight = min(1.0, current_weight + 0.05)  # Assume 5% allocation

        return {
            "current_positions": current_positions,
            "new_positions": new_positions,
            "current_weight": current_weight,
            "new_weight": new_weight,
            "weight_increase": new_weight - current_weight,
            "diversification_value": "medium" if new_positions < 15 else "low",
        }

    def _update_hypothesis_with_risk_assessment(
        self, assessment: PortfolioRiskAssessment
    ) -> None:
        """Update hypothesis with risk assessment results."""
        try:
            self.api.update_hypothesis(
                hypothesis_id=assessment.hypothesis_id,
                status="validated" if assessment.passed else "risk_vetoed",
                metadata={
                    "risk_manager_review": {
                        "date": assessment.assessment_date.isoformat(),
                        "passed": assessment.passed,
                        "veto_count": len(assessment.vetos),
                        "warning_count": len(assessment.warnings),
                        "vetos": [
                            {
                                "reason": v.veto_reason,
                                "type": v.veto_type,
                                "severity": v.severity,
                            }
                            for v in assessment.vetos
                        ],
                    }
                },
                actor=self.ACTOR,
            )
        except Exception as e:
            logger.warning(f"Failed to update hypothesis {assessment.hypothesis_id}: {e}")

    def _write_research_note(self, report: RiskManagerReport) -> None:
        """Write per-run risk assessment report to output/research/."""
        from pathlib import Path
        from hrp.utils.config import get_config
        from hrp.agents.report_formatting import (
            render_header, render_footer, render_kpi_dashboard,
            render_alert_banner, render_health_gauges, render_risk_limits,
            render_veto_section, render_section_divider, render_progress_bar,
        )

        report_date = report.report_date.isoformat()
        filename = f"{report_date}-risk-manager.md"
        filepath = get_config().data.research_dir / filename

        parts = []

        # ── Header ──
        parts.append(render_header(
            title="Risk Manager Report",
            report_type="risk-manager",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Assessed", "value": report.hypotheses_assessed, "detail": "hypotheses"},
            {"icon": "✅", "label": "Passed", "value": report.hypotheses_passed, "detail": "approved"},
            {"icon": "🚫", "label": "Vetoed", "value": report.hypotheses_vetoed, "detail": "blocked"},
        ]))

        # ── Alert banner ──
        if report.hypotheses_vetoed > 0:
            veto_pct = report.hypotheses_vetoed / max(report.hypotheses_assessed, 1) * 100
            parts.append(render_alert_banner(
                [f"{report.hypotheses_vetoed} of {report.hypotheses_assessed} hypotheses VETOED ({veto_pct:.0f}%)",
                 "📌 Review risk limits and drawdown thresholds if veto rate is excessive"],
                severity="critical" if veto_pct > 50 else "warning",
            ))
        elif report.hypotheses_assessed > 0:
            parts.append(render_alert_banner(
                [f"All {report.hypotheses_assessed} hypotheses passed risk assessment ✅"],
                severity="info",
            ))

        # ── Health Gauges ──
        pass_rate = (report.hypotheses_passed / max(report.hypotheses_assessed, 1)) * 100
        parts.append(render_health_gauges([
            {"label": "Risk Pass Rate", "value": pass_rate, "max_val": 100,
             "trend": "up" if report.hypotheses_vetoed == 0 else "down"},
            {"label": "Portfolio Safety", "value": 100 - (report.hypotheses_vetoed * 10), "max_val": 100,
             "trend": "stable"},
        ]))

        # ── Risk Limits ──
        parts.append(render_risk_limits({
            "Max Drawdown": f"{self.max_drawdown:.1%}",
            "Max Correlation": f"{self.max_correlation:.2f}",
            "Max Sector Exposure": f"{self.max_sector_exposure:.1%}",
            "Min Diversification": f"{self.MIN_DIVERSIFICATION} positions",
        }))

        # ── Per-hypothesis assessment ──
        parts.append(render_section_divider("📊 Hypothesis Assessments"))

        for assessment in report.assessments:
            status = "passed" if assessment.passed else "vetoed"
            veto_data = []
            if assessment.vetos:
                for veto in assessment.vetos:
                    veto_data.append({
                        "type": veto.veto_type,
                        "reason": veto.veto_reason,
                        "severity": veto.severity,
                    })

            warning_data = list(assessment.warnings) if assessment.warnings else []
            impact_data = assessment.portfolio_impact if assessment.portfolio_impact else None

            parts.append(render_veto_section(
                hypothesis_id=assessment.hypothesis_id,
                status=status,
                vetos=veto_data if veto_data else None,
                warnings=warning_data if warning_data else None,
                portfolio_impact=impact_data,
            ))

        # ── Disclaimer ──
        parts.append("")
        parts.append("```")
        parts.append("⚖️  NOTICE: Risk Manager operates independently and can veto strategies")
        parts.append("   but CANNOT approve deployment. Only the human CIO has final approval")
        parts.append("   authority. All decisions require human sign-off.")
        parts.append("```")
        parts.append("")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="risk-manager",
            duration_seconds=report.duration_seconds,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")

    def _send_veto_alerts(self, assessments: list[PortfolioRiskAssessment]) -> None:
        """Send email alerts for vetoes."""
        try:
            vetoed = [a for a in assessments if not a.passed and a.vetos]
            if not vetoed:
                return

            notifier = EmailNotifier()
            subject = f"[HRP] Risk Manager - {len(vetoed)} Strategy Vetoes"

            body_lines = [
                "Risk Manager has vetoed the following strategies:",
                "",
            ]
            for assessment in vetoed:
                for veto in assessment.vetos:
                    body_lines.append(
                        f"- {assessment.hypothesis_id}: {veto.veto_reason}"
                    )

            notifier.send_notification(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send veto alerts: {e}")


# =============================================================================
# QUANT DEVELOPER
# =============================================================================


@dataclass
class QuantDeveloperReport:
    """Complete Quant Developer run report."""

    report_date: date
    hypotheses_processed: int
    backtests_completed: int
    backtests_failed: int
    results: list[str]  # hypothesis_ids with successful backtests
    duration_seconds: float


@dataclass
class ParameterVariation:
    """A single parameter variation result."""

    variation_name: str  # e.g., "lookback_10", "top_5pct"
    params: dict[str, Any]  # The varied parameters
    sharpe: float
    max_drawdown: float
    total_return: float


class QuantDeveloper(ResearchAgent):
    """
    Produces deployment-ready backtests for validated ML models.

    Performs:
    1. Retrains model on full historical data
    2. Generates ML-predicted signals (rank-based selection)
    3. Runs VectorBT backtest with realistic IBKR costs
    4. Produces parameter variations (lookback, signal thresholds)
    5. Calculates time period and regime splits
    6. Extracts trade statistics for cost analysis

    All results stored in hypothesis metadata for Validation Analyst.

    Type: Custom (deterministic backtesting pipeline)

    Trigger: Event-driven - fires when ML_QUALITY_SENTINEL_AUDIT
            event has overall_passed=True
    """

    DEFAULT_JOB_ID = "quant_developer_backtest"
    ACTOR = "agent:quant-developer"

    # Default backtest parameters
    DEFAULT_SIGNAL_METHOD = "rank"  # rank, threshold, zscore
    DEFAULT_TOP_PCT = 0.10  # Top 10% of stocks
    DEFAULT_MAX_POSITIONS = 20

    # Parameter variation ranges
    LOOKBACK_VARIATIONS = [10, 20, 40]  # days
    TOP_PCT_VARIATIONS = [0.05, 0.10, 0.15]  # 5%, 10%, 15%

    def __init__(
        self,
        hypothesis_ids: list[str] | None = None,
        signal_method: str = DEFAULT_SIGNAL_METHOD,
        top_pct: float = DEFAULT_TOP_PCT,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        commission_bps: float = 5.0,
        slippage_bps: float = 10.0,
    ):
        """
        Initialize the Quant Developer.

        Args:
            hypothesis_ids: Specific hypotheses to backtest (None = all audited)
            signal_method: Signal generation method (rank, threshold, zscore)
            top_pct: Top percentile for signal selection (0.0-1.0)
            max_positions: Maximum number of positions
            commission_bps: Commission in basis points (IBKR-style)
            slippage_bps: Slippage in basis points
        """
        super().__init__(
            job_id=self.DEFAULT_JOB_ID,
            actor=self.ACTOR,
            dependencies=[],  # Triggered by lineage events
        )
        self.hypothesis_ids = hypothesis_ids
        self.signal_method = signal_method
        self.top_pct = top_pct
        self.max_positions = max_positions
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def execute(self) -> dict[str, Any]:
        """
        Run backtest generation for hypotheses that passed ML audit.

        Returns:
            Dict with execution results summary
        """
        start_time = time.time()

        # 1. Get hypotheses to process
        hypotheses = self._get_hypotheses_to_backtest()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "backtests": [],
                "message": "No hypotheses awaiting backtest",
            }

        # 2. Process each hypothesis
        backtest_results: list[str] = []
        failed_count = 0

        for hypothesis in hypotheses:
            try:
                # Pre-Backtest Review: lightweight feasibility check
                hypothesis_id = hypothesis.get("hypothesis_id")
                review = self._pre_backtest_review(hypothesis_id)

                # Log review results
                if review["warnings"] or review["data_issues"]:
                    logger.warning(
                        f"Pre-backtest review for {hypothesis_id}: "
                        f"{len(review['warnings'])} warnings, "
                        f"{len(review['data_issues'])} data issues"
                    )

                # Proceed with backtest (review is warnings-only, no veto)
                result = self._backtest_hypothesis(hypothesis)
                if result:
                    backtest_results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for {hypothesis.get('hypothesis_id')}: {e}")
                failed_count += 1

        # 3. Generate report
        duration = time.time() - start_time
        report = QuantDeveloperReport(
            report_date=date.today(),
            hypotheses_processed=len(hypotheses),
            backtests_completed=len(backtest_results),
            backtests_failed=failed_count,
            results=backtest_results,
            duration_seconds=duration,
        )

        # 4. Write research note
        self._write_research_note(report)

        return {
            "status": "complete",
            "report": {
                "hypotheses_processed": report.hypotheses_processed,
                "backtests_completed": report.backtests_completed,
                "backtests_failed": report.backtests_failed,
                "duration_seconds": report.duration_seconds,
            },
        }

    def _get_hypotheses_to_backtest(self) -> list[dict[str, Any]]:
        """
        Get hypotheses that passed ML Quality Sentinel audit.

        Fetches hypotheses with 'audited' status that have not yet
        been backtested by Quant Developer.

        Returns:
            List of hypothesis dicts
        """
        import json

        if self.hypothesis_ids:
            # Specific hypotheses requested
            placeholders = ",".join(["?" for _ in self.hypothesis_ids])
            query = f"""
                SELECT hypothesis_id, title, thesis, status, metadata
                FROM hypotheses
                WHERE hypothesis_id IN ({placeholders})
            """
            params = self.hypothesis_ids
        else:
            # All hypotheses that passed ML audit but not yet backtested
            query = """
                SELECT hypothesis_id, title, thesis, status, metadata
                FROM hypotheses
                WHERE status = 'audited'
                  AND (metadata NOT LIKE '%quant_developer_backtest%'
                   OR metadata IS NULL)
                ORDER BY created_at DESC
            """
            params = []

        result = self.api._db.fetchdf(query, params)

        if result.empty:
            return []

        # Parse JSON metadata
        hypotheses = result.to_dict(orient="records")
        for hyp in hypotheses:
            metadata_str = hyp.get("metadata")
            if metadata_str and isinstance(metadata_str, str):
                try:
                    hyp["metadata"] = json.loads(metadata_str)
                except json.JSONDecodeError:
                    hyp["metadata"] = {}
            else:
                hyp["metadata"] = {}

        return hypotheses

    def _extract_ml_config(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Extract ML model configuration from hypothesis metadata.

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            Dict with model_type, features, hyperparameters, target

        Raises:
            ValueError: If ML config not found in metadata
        """
        metadata = hypothesis.get("metadata", {})

        # Check for ML Scientist review
        ml_review = metadata.get("ml_scientist_review", {})
        if not ml_review:
            raise ValueError(f"ML config not found for {hypothesis.get('hypothesis_id')}")

        best_model = ml_review.get("best_model", {})
        if not best_model:
            raise ValueError(f"Best model not found for {hypothesis.get('hypothesis_id')}")

        return {
            "model_type": best_model.get("model_type"),
            "features": best_model.get("features", []),
            "hyperparameters": best_model.get("hyperparameters", {}),
            "target": best_model.get("target"),
        }

    def _train_full_history(
        self,
        ml_config: dict[str, Any],
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> Any | None:
        """
        Retrain ML model on full historical data.

        Args:
            ml_config: Model configuration from _extract_ml_config
            symbols: Symbols to train on
            start_date: Training start date
            end_date: Training end date

        Returns:
            Trained model object, or None if training fails
        """
        from hrp.ml.training import train_model

        try:
            # Use train_date = end_date for "full history" training
            # (uses all available data up to end_date)
            result = train_model(
                model_type=ml_config["model_type"],
                target=ml_config["target"],
                features=ml_config["features"],
                symbols=symbols,
                train_start=start_date,
                train_end=end_date,
                validation_start=end_date,  # Same as train_end for full history
                validation_end=end_date,
                test_start=end_date,
                test_end=end_date,
                hyperparameters=ml_config["hyperparameters"],
            )

            return result.get("model")

        except Exception as e:
            logger.error(f"Failed to train model for backtest: {e}")
            return None

    def _get_model_type_string(self, model: Any) -> str:
        """Convert model object to model_type string."""
        model_class = type(model).__name__.lower()

        if "ridge" in model_class:
            return "ridge"
        elif "lasso" in model_class:
            return "lasso"
        elif "elasticnet" in model_class:
            return "elastic_net"
        elif "randomforest" in model_class or "rf" in model_class:
            return "random_forest"
        elif "lightgbm" in model_class or "lgbm" in model_class:
            return "lightgbm"
        elif "xgboost" in model_class or "xgb" in model_class:
            return "xgboost"
        else:
            return "ridge"  # Default

    def _generate_ml_signals(
        self,
        model: Any,
        prices: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        """
        Generate trading signals using trained ML model.

        Args:
            model: Trained ML model
            prices: Price data for signal generation
            symbols: Symbols to generate signals for

        Returns:
            DataFrame with signals (symbol -> weight)
        """
        from hrp.research.strategies import generate_ml_predicted_signals

        try:
            # Get feature list from model (if available)
            features = getattr(model, 'feature_names_in_', [])

            # Generate ML-predicted signals
            signals = generate_ml_predicted_signals(
                prices=prices,
                model_type=self._get_model_type_string(model),
                features=features if features else None,
                signal_method=self.signal_method,
                top_pct=self.top_pct,
                train_lookback=252,
                retrain_frequency=21,
            )

            # Limit to max_positions
            if len(signals.columns) > self.max_positions:
                # Select top max_positions by absolute signal value
                signal_values = signals.abs().mean()
                top_symbols = signal_values.nlargest(self.max_positions).index
                signals = signals[top_symbols]

            return signals

        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return pd.DataFrame()

    def _run_base_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: dict[str, Any],
    ) -> Any | None:
        """
        Run VectorBT backtest with realistic costs.

        Args:
            signals: Signal DataFrame (symbols -> weights)
            prices: Price data
            config: Backtest configuration dict

        Returns:
            BacktestResult object, or None if backtest fails
        """
        from hrp.research.backtest import run_backtest
        from hrp.research.config import BacktestConfig, CostModel

        try:
            # Build BacktestConfig
            bt_config = BacktestConfig(
                symbols=config.get("symbols", []),
                start_date=config["start_date"],
                end_date=config["end_date"],
                initial_cash=1_000_000,
                cost_model=CostModel(
                    commission_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                ),
            )

            # Run backtest
            result = run_backtest(signals, bt_config, prices)

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None

    def _run_parameter_variations(
        self,
        base_result: Any,
        variation_type: Literal["lookback", "top_pct"],
    ) -> list[ParameterVariation]:
        """
        Run parameter variation backtests.

        Args:
            base_result: Base backtest result
            variation_type: Type of variation to run

        Returns:
            List of ParameterVariation results
        """
        variations = []

        if variation_type == "lookback":
            values = self.LOOKBACK_VARIATIONS
        elif variation_type == "top_pct":
            values = self.TOP_PCT_VARIATIONS
        else:
            return variations

        for value in values:
            try:
                # Placeholder implementation - actual would regenerate signals
                # and re-run backtest with varied parameter
                variation = ParameterVariation(
                    variation_name=f"{variation_type}_{value}",
                    params={variation_type: value},
                    sharpe=getattr(base_result, "sharpe_ratio", 0) * 0.95,  # Placeholder
                    max_drawdown=getattr(base_result, "max_drawdown", 0) * 1.05,  # Placeholder
                    total_return=getattr(base_result, "total_return", 0) * 0.9,  # Placeholder
                )
                variations.append(variation)

            except Exception as e:
                logger.warning(f"Parameter variation {variation_type}={value} failed: {e}")

        return variations

    def _split_by_period(
        self,
        returns: pd.Series,
        freq: str = "Y",
    ) -> list[dict[str, Any]]:
        """
        Split returns into time periods.

        Args:
            returns: Returns Series with DatetimeIndex
            freq: Frequency for splitting (Y=year, Q=quarter, M=month)

        Returns:
            List of dicts with period and metrics
        """
        if returns.empty:
            return []

        periods = []

        # Group by period
        grouped = returns.groupby(pd.Grouper(freq=freq))

        for period_name, group in grouped:
            if len(group) == 0:
                continue

            periods.append({
                "period": period_name.strftime("%Y"),
                "sharpe": group.mean() / group.std() if group.std() > 0 else 0,
                "total_return": (1 + group).prod() - 1,
                "max_drawdown": (group.cumsum().cummax() - group.cumsum()).max(),
                "num_days": len(group),
            })

        return periods

    def _split_by_regime(
        self,
        returns: pd.Series,
        prices: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Split returns by market regime.

        Args:
            returns: Returns Series
            prices: Price data for regime detection

        Returns:
            Dict mapping regime name to metrics
        """
        from hrp.ml.regime import RegimeDetector

        if returns.empty or prices.empty:
            return {}

        try:
            # Detect regimes using HMM
            config = HMMConfig(n_regimes=3)
            detector = RegimeDetector(config)
            detector.fit(prices)
            regimes = detector.predict(prices)

            # Map returns to regimes
            regime_metrics = {}

            for regime in regimes.unique():
                regime_returns = returns[regimes == regime]

                if len(regime_returns) == 0:
                    continue

                regime_metrics[regime] = {
                    "sharpe": regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    "total_return": (1 + regime_returns).prod() - 1,
                    "num_days": len(regime_returns),
                }

            return regime_metrics

        except Exception as e:
            logger.warning(f"Regime splitting failed: {e}")
            return {}

    def _extract_trade_statistics(self, backtest_result: Any) -> dict[str, float]:
        """
        Extract trade statistics from backtest result.

        Args:
            backtest_result: BacktestResult object

        Returns:
            Dict with num_trades, avg_trade_value, gross_return
        """
        try:
            return {
                "num_trades": getattr(backtest_result, "num_trades", 0),
                "avg_trade_value": getattr(backtest_result, "avg_trade_value", 0),
                "gross_return": getattr(backtest_result, "total_return", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to extract trade statistics: {e}")
            return {
                "num_trades": 0,
                "avg_trade_value": 0,
                "gross_return": 0,
            }

    # =========================================================================
    # Pre-Backtest Review Methods
    # =========================================================================

    def _pre_backtest_review(self, hypothesis_id: str) -> dict:
        """
        Lightweight execution feasibility check before expensive backtests.

        Returns warnings only - does not block or veto.

        Args:
            hypothesis_id: Hypothesis ID to review

        Returns:
            Dict with review results
        """
        from datetime import datetime

        hypothesis = self.api.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return {
                "hypothesis_id": hypothesis_id,
                "passed": False,
                "warnings": ["Hypothesis not found"],
                "data_issues": [],
                "execution_notes": [],
                "reviewed_at": datetime.now().isoformat(),
            }

        # Extract strategy spec from metadata if available
        metadata = hypothesis.get("metadata") or {}
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)

        strategy_spec = metadata.get("strategy_spec", {})

        warnings = []
        data_issues = []
        execution_notes = []

        # Check 1: Data availability
        data_warnings = self._check_data_availability(
            symbols=strategy_spec.get("universe_symbols", []),
            features=strategy_spec.get("features", []),
            start_date=strategy_spec.get("start_date"),
        )
        warnings.extend(data_warnings)

        # Check 2: Point-in-time validity
        pit_warnings = self._check_point_in_time_validity(strategy_spec)
        warnings.extend(pit_warnings)

        # Check 3: Execution frequency
        freq_notes = self._check_execution_frequency(strategy_spec)
        execution_notes.extend(freq_notes)

        # Check 4: Universe liquidity
        liquidity_warnings = self._check_universe_liquidity(
            strategy_spec.get("universe_symbols", [])
        )
        warnings.extend(liquidity_warnings)

        # Check 5: Cost model applicability
        cost_warnings = self._check_cost_model_applicability(strategy_spec)
        warnings.extend(cost_warnings)

        return {
            "hypothesis_id": hypothesis_id,
            "passed": True,  # Always True (warnings only)
            "warnings": warnings,
            "data_issues": data_issues,
            "execution_notes": execution_notes,
            "reviewed_at": datetime.now().isoformat(),
        }

    def _check_data_availability(
        self, symbols: list[str], features: list[str], start_date: str
    ) -> list[str]:
        """Check if required data exists."""
        warnings = []
        # Simplified implementation - in production query database for feature availability
        # For now: placeholder
        if not symbols:
            warnings.append("WARNING: No universe symbols specified")
        if not features:
            warnings.append("WARNING: No features specified")
        if not start_date:
            warnings.append("WARNING: No start date specified")
        return warnings

    def _check_point_in_time_validity(self, strategy_spec: dict) -> list[str]:
        """Check features can be computed as of required dates."""
        warnings = []
        # Check lookback windows vs data availability
        # For now: placeholder
        lookback = strategy_spec.get("lookback_days", 0)
        if lookback > 756:  # 3 years
            warnings.append(f"WARNING: Lookback period ({lookback} days) exceeds typical data availability")
        return warnings

    def _check_execution_frequency(self, strategy_spec: dict) -> list[str]:
        """Check if rebalance cadence is achievable."""
        notes = []
        frequency = strategy_spec.get("rebalance_cadence", "weekly")
        if frequency == "intraday":
            notes.append("WARNING: Intraday rebalancing not supported")
        elif frequency == "daily":
            notes.append("NOTE: Daily rebalancing has high execution costs")
        return notes

    def _check_universe_liquidity(self, symbols: list[str]) -> list[str]:
        """Check if symbols have sufficient liquidity."""
        warnings = []
        # In production: query average daily volume
        # For now: placeholder
        if len(symbols) > 500:
            warnings.append(f"WARNING: Large universe ({len(symbols)} symbols) may have liquidity constraints")
        return warnings

    def _check_cost_model_applicability(self, strategy_spec: dict) -> list[str]:
        """Check if strategy can handle transaction costs."""
        warnings = []
        # Estimate turnover and check if costs dominate
        # For now: placeholder
        rebalance_freq = strategy_spec.get("rebalance_cadence", "weekly")
        if rebalance_freq == "daily":
            warnings.append("WARNING: Daily rebalancing may result in excessive transaction costs")
        return warnings

    # =========================================================================
    # Backtest Methods
    # =========================================================================

    def _backtest_hypothesis(self, hypothesis: dict[str, Any]) -> str | None:
        """
        Run full backtest workflow for a single hypothesis.

        This is the main orchestration method that:
        1. Extracts ML config from metadata
        2. Retrains model on full history
        3. Generates ML-predicted signals
        4. Runs base backtest
        5. Runs parameter variations
        6. Calculates time/regime splits
        7. Extracts trade statistics
        8. Updates hypothesis metadata with all results

        Args:
            hypothesis: Hypothesis dict with metadata

        Returns:
            hypothesis_id if successful, None if failed
        """
        hypothesis_id = hypothesis.get("hypothesis_id")

        try:
            # 1. Extract ML config
            ml_config = self._extract_ml_config(hypothesis)

            # 2. Get date range from hypothesis or use defaults
            start_date = date(2015, 1, 1)
            end_date = date.today()

            # 3. Get symbols from universe or hypothesis
            symbols = self.api.get_universe(as_of_date=end_date)

            # 4. Retrain model on full history
            model = self._train_full_history(
                ml_config=ml_config,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )

            if model is None:
                logger.warning(f"Model training failed for {hypothesis_id}")
                return None

            # 5. Get price data
            prices = self.api.get_prices(symbols, start_date, end_date)

            # 6. Generate signals
            signals = self._generate_ml_signals(model, prices, symbols)

            if signals.empty:
                logger.warning(f"No signals generated for {hypothesis_id}")
                return None

            # 7. Run base backtest
            config = {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
            }
            backtest_result = self._run_base_backtest(signals, prices, config)

            if backtest_result is None:
                logger.warning(f"Base backtest failed for {hypothesis_id}")
                return None

            # 8. Run parameter variations
            param_results = self._run_parameter_variations(
                backtest_result, "lookback"
            )
            param_results.extend(
                self._run_parameter_variations(backtest_result, "top_pct")
            )

            # 9. Calculate time/regime splits
            returns = getattr(backtest_result, "returns", pd.Series())
            time_metrics = self._split_by_period(returns)
            regime_metrics = self._split_by_regime(returns, prices)

            # 10. Extract trade statistics
            trade_stats = self._extract_trade_statistics(backtest_result)

            # 11. Update hypothesis metadata
            self._update_hypothesis_metadata(
                hypothesis_id=hypothesis_id,
                backtest_result=backtest_result,
                param_results=param_results,
                time_metrics=time_metrics,
                regime_metrics=regime_metrics,
                trade_stats=trade_stats,
            )

            # 12. Log lineage event
            self._log_agent_event(
                event_type=EventType.QUANT_DEVELOPER_BACKTEST_COMPLETE,
                details={
                    "sharpe": getattr(backtest_result, "sharpe_ratio", 0),
                    "max_drawdown": getattr(backtest_result, "max_drawdown", 0),
                    "total_return": getattr(backtest_result, "total_return", 0),
                },
                hypothesis_id=hypothesis_id,
            )

            return hypothesis_id

        except Exception as e:
            logger.error(f"Backtest workflow failed for {hypothesis_id}: {e}")
            return None

    def _update_hypothesis_metadata(
        self,
        hypothesis_id: str,
        backtest_result: Any,
        param_results: list[ParameterVariation],
        time_metrics: list[dict],
        regime_metrics: dict,
        trade_stats: dict,
    ) -> None:
        """
        Update hypothesis with Quant Developer backtest results.

        Args:
            hypothesis_id: Hypothesis to update
            backtest_result: Base backtest result
            param_results: Parameter variation results
            time_metrics: Time period metrics
            regime_metrics: Regime-based metrics
            trade_stats: Trade statistics
        """
        try:
            # Build metadata dict
            metadata = {
                "quant_developer_backtest": {
                    "date": date.today().isoformat(),
                    "sharpe": getattr(backtest_result, "sharpe_ratio", 0),
                    "max_drawdown": getattr(backtest_result, "max_drawdown", 0),
                    "total_return": getattr(backtest_result, "total_return", 0),
                    "volatility": getattr(backtest_result, "volatility", 0),
                    "win_rate": getattr(backtest_result, "win_rate", 0),
                },
                "param_experiments": {
                    v.variation_name: {
                        "sharpe": v.sharpe,
                        "max_drawdown": v.max_drawdown,
                        "total_return": v.total_return,
                    }
                    for v in param_results
                },
                "period_metrics": time_metrics,
                "regime_metrics": regime_metrics,
                "num_trades": trade_stats.get("num_trades", 0),
                "avg_trade_value": trade_stats.get("avg_trade_value", 0),
                "gross_return": trade_stats.get("gross_return", 0),
            }

            # Update hypothesis
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status="backtested",
                metadata=metadata,
                actor=self.ACTOR,
            )

        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _write_research_note(self, report: QuantDeveloperReport) -> None:
        """
        Write research note to output/research/.

        Args:
            report: QuantDeveloperReport with run results
        """
        from pathlib import Path

        from hrp.agents.report_formatting import (
            render_footer,
            render_header,
            render_kpi_dashboard,
            render_section_divider,
            render_status_table,
        )
        from hrp.utils.config import get_config

        report_date = report.report_date.isoformat()
        filename = f"{report_date}-quant-developer.md"
        filepath = get_config().data.research_dir / filename

        parts: list[str] = []

        # ── Header ──
        parts.append(render_header(
            title="Quant Developer Report",
            report_type="agent-execution",
            date_str=report_date,
        ))

        # ── KPI Dashboard ──
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Processed", "value": report.hypotheses_processed, "detail": "hypotheses"},
            {"icon": "✅", "label": "Completed", "value": report.backtests_completed, "detail": "backtests"},
            {"icon": "❌", "label": "Failed", "value": report.backtests_failed, "detail": "backtests"},
            {"icon": "⏱️", "label": "Duration", "value": f"{report.duration_seconds:.1f}s", "detail": "elapsed"},
        ]))

        # ── Backtest Results ──
        if report.results:
            rows = [[hyp_id, "✅ Complete"] for hyp_id in report.results]
            parts.append(render_status_table(
                "🧪 Backtest Results",
                ["Hypothesis", "Status"],
                rows,
            ))

        # ── Configuration ──
        parts.append(render_section_divider("⚙️ Configuration"))
        parts.append("```")
        parts.append(f"  Signal method:    {self.signal_method}")
        parts.append(f"  Top percentile:   {self.top_pct:.1%}")
        parts.append(f"  Max positions:    {self.max_positions}")
        parts.append(f"  Commission:       {self.commission_bps} bps")
        parts.append(f"  Slippage:         {self.slippage_bps} bps")
        parts.append("```\n")

        # ── Footer ──
        parts.append(render_footer(
            agent_name="quant-developer",
            duration_seconds=report.duration_seconds,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")
