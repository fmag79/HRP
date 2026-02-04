"""
Quant Developer agent for deployment-ready backtest generation.

Produces deployment-ready backtests for validated ML models,
including parameter variations, time/regime splits, and trade statistics.
"""

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd
from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.research.lineage import EventType


@dataclass
class HypothesisBacktestResult:
    """Detailed backtest result for a single hypothesis."""

    hypothesis_id: str
    title: str
    # Base backtest metrics
    sharpe: float
    total_return: float
    max_drawdown: float
    volatility: float
    win_rate: float
    # Period analysis (yearly)
    period_metrics: list[dict[str, Any]]
    # Regime analysis
    regime_metrics: dict[str, Any]
    # Parameter variations
    parameter_variations: list[dict[str, Any]]
    # Trade statistics
    num_trades: int
    avg_trade_value: float
    # Features used
    features: list[str]
    model_type: str


@dataclass
class QuantDeveloperReport:
    """Complete Quant Developer run report."""

    report_date: date
    hypotheses_processed: int
    backtests_completed: int
    backtests_failed: int
    results: list[HypothesisBacktestResult]  # Detailed per-hypothesis results
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
        backtest_results: list[HypothesisBacktestResult] = []
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
        if self.hypothesis_ids:
            # Specific hypotheses requested
            hypotheses = []
            for hid in self.hypothesis_ids:
                hyp = self.api.get_hypothesis_with_metadata(hid)
                if hyp:
                    hypotheses.append(hyp)
            return hypotheses
        else:
            # All hypotheses that passed ML audit but not yet backtested
            return self.api.list_hypotheses_with_metadata(
                status='validated',
                metadata_exclude='%quant_developer_backtest%',
                limit=10,
            )

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
        metadata = hypothesis.get("metadata") or {}

        ml_results = metadata.get("ml_scientist_results", {})
        if not ml_results:
            raise ValueError(f"ML config not found for {hypothesis.get('hypothesis_id')}")

        return {
            "model_type": ml_results.get("model_type"),
            "features": ml_results.get("features", []),
            "hyperparameters": ml_results.get("hyperparameters", {}),
            "target": ml_results.get("target"),
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
        from hrp.ml.models import MLConfig
        from hrp.ml.training import train_model

        try:
            config = MLConfig(
                model_type=ml_config["model_type"],
                target=ml_config["target"],
                features=ml_config["features"],
                train_start=start_date,
                train_end=end_date,
                validation_start=end_date,
                validation_end=end_date,
                test_start=end_date,
                test_end=end_date,
                hyperparameters=ml_config.get("hyperparameters", {}),
            )
            result = train_model(config=config, symbols=symbols)

            return result.model

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

            # Pivot flat prices (symbol, date, open, ..., close) into
            # MultiIndex columns (field, symbol) expected by strategies
            if 'symbol' in prices.columns and 'date' in prices.columns:
                pivoted = prices.pivot(index='date', columns='symbol')
                pivoted.index = pd.to_datetime(pivoted.index)
                pivoted = pivoted.sort_index()
            else:
                pivoted = prices

            # Generate ML-predicted signals
            signals = generate_ml_predicted_signals(
                prices=pivoted,
                model_type=self._get_model_type_string(model),
                features=list(features) if len(features) > 0 else None,
                signal_method=self.signal_method,
                top_pct=self.top_pct,
                train_lookback=252,
                retrain_frequency=21,
            )

            # Ensure signals is a DataFrame
            if isinstance(signals, pd.Series):
                signals = signals.to_frame()

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
            prices: Price data (flat or pivoted format)
            config: Backtest configuration dict

        Returns:
            BacktestResult object, or None if backtest fails
        """
        from hrp.research.backtest import run_backtest
        from hrp.research.config import BacktestConfig, CostModel

        try:
            # Ensure prices are in pivoted format for run_backtest
            # Flat format: columns = [symbol, date, open, high, low, close, ...]
            # Pivoted format: MultiIndex columns = (field, symbol), DatetimeIndex rows
            if 'symbol' in prices.columns and 'date' in prices.columns:
                pivoted_prices = prices.pivot(index='date', columns='symbol')
                pivoted_prices.index = pd.to_datetime(pivoted_prices.index)
                pivoted_prices = pivoted_prices.sort_index()
            else:
                pivoted_prices = prices

            # Filter prices to only symbols in signals (signals may be limited
            # to max_positions, but prices has all universe symbols)
            signal_symbols = list(signals.columns)
            if isinstance(pivoted_prices.columns, pd.MultiIndex):
                # MultiIndex: filter at level 1 (symbol)
                available_symbols = pivoted_prices.columns.get_level_values(1).unique()
                common_symbols = [s for s in signal_symbols if s in available_symbols]
                pivoted_prices = pivoted_prices.loc[:, (slice(None), common_symbols)]
            else:
                # Regular columns
                common_symbols = [s for s in signal_symbols if s in pivoted_prices.columns]
                pivoted_prices = pivoted_prices[common_symbols]

            # Build BacktestConfig with filtered symbols
            bt_config = BacktestConfig(
                symbols=common_symbols,
                start_date=config["start_date"],
                end_date=config["end_date"],
                costs=CostModel(
                    spread_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                ),
            )

            # Run backtest
            result = run_backtest(signals, bt_config, pivoted_prices)

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None

    def _run_parameter_variations(
        self,
        variation_type: Literal["lookback", "top_pct"],
        model: Any,
        prices: pd.DataFrame,
        config: dict[str, Any],
    ) -> list[ParameterVariation]:
        """
        Run parameter variation backtests with real signal regeneration.

        Args:
            variation_type: Type of variation to run (lookback or top_pct)
            model: Trained ML model for signal generation
            prices: Price data for backtesting
            config: Backtest configuration dict

        Returns:
            List of ParameterVariation results with real metrics
        """
        from hrp.research.strategies import generate_ml_predicted_signals

        variations = []

        if variation_type == "lookback":
            values = self.LOOKBACK_VARIATIONS
        elif variation_type == "top_pct":
            values = self.TOP_PCT_VARIATIONS
        else:
            return variations

        # Prepare pivoted prices once for all variations
        if 'symbol' in prices.columns and 'date' in prices.columns:
            pivoted = prices.pivot(index='date', columns='symbol')
            pivoted.index = pd.to_datetime(pivoted.index)
            pivoted = pivoted.sort_index()
        else:
            pivoted = prices

        # Get model features
        features = getattr(model, 'feature_names_in_', [])
        model_type = self._get_model_type_string(model)

        for value in values:
            try:
                # Set parameters for this variation
                if variation_type == "lookback":
                    train_lookback = value
                    top_pct = self.top_pct
                else:  # top_pct
                    train_lookback = 252
                    top_pct = value

                # Generate signals with varied parameters
                signals = generate_ml_predicted_signals(
                    prices=pivoted,
                    model_type=model_type,
                    features=list(features) if len(features) > 0 else None,
                    signal_method=self.signal_method,
                    top_pct=top_pct,
                    train_lookback=train_lookback,
                    retrain_frequency=21,
                )

                if signals.empty:
                    logger.warning(f"No signals for {variation_type}={value}")
                    continue

                # Limit to max_positions
                if len(signals.columns) > self.max_positions:
                    signal_values = signals.abs().mean()
                    top_symbols = signal_values.nlargest(self.max_positions).index
                    signals = signals[top_symbols]

                # Run backtest with varied signals
                backtest_result = self._run_base_backtest(signals, prices, config)

                if backtest_result is None:
                    logger.warning(f"Backtest failed for {variation_type}={value}")
                    continue

                # Extract real metrics
                variation = ParameterVariation(
                    variation_name=f"{variation_type}_{value}",
                    params={variation_type: value},
                    sharpe=backtest_result.sharpe,
                    max_drawdown=backtest_result.max_drawdown,
                    total_return=backtest_result.total_return,
                )
                variations.append(variation)

                logger.debug(
                    f"Variation {variation_type}={value}: "
                    f"Sharpe={backtest_result.sharpe:.2f}, "
                    f"Return={backtest_result.total_return*100:.1f}%"
                )

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
        from hrp.ml.regime import HMMConfig, RegimeDetector

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
            # BacktestResult has .trades DataFrame and .total_return property
            trades_df = backtest_result.trades
            num_trades = len(trades_df) if trades_df is not None else 0
            avg_trade_value = 0.0
            if num_trades > 0:
                # VectorBT trades have 'Size' and 'Avg Entry Price' columns
                if "Size" in trades_df.columns and "Avg Entry Price" in trades_df.columns:
                    trade_values = trades_df["Size"].abs() * trades_df["Avg Entry Price"]
                    avg_trade_value = trade_values.mean()
            return {
                "num_trades": num_trades,
                "avg_trade_value": avg_trade_value,
                "gross_return": backtest_result.total_return,
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

    def _backtest_hypothesis(self, hypothesis: dict[str, Any]) -> HypothesisBacktestResult | None:
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
            HypothesisBacktestResult if successful, None if failed
        """
        hypothesis_id = hypothesis.get("hypothesis_id")

        try:
            # 1. Extract ML config
            ml_config = self._extract_ml_config(hypothesis)

            # 2. Get date range from hypothesis or use defaults
            start_date = date(2015, 1, 1)
            end_date = date.today()

            # 3. Get symbols from universe (fall back to latest available if today has no data)
            symbols = self.api.get_universe(as_of_date=end_date)
            if not symbols:
                # Try yesterday or get latest available date from universe
                from datetime import timedelta
                for days_back in range(1, 10):
                    fallback_date = end_date - timedelta(days=days_back)
                    symbols = self.api.get_universe(as_of_date=fallback_date)
                    if symbols:
                        logger.info(f"Using universe from {fallback_date} (today has no data)")
                        end_date = fallback_date
                        break

            if not symbols:
                logger.error(f"No universe symbols found for {hypothesis_id}")
                return None

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

            # 8. Run parameter variations (real backtests with varied parameters)
            param_results = self._run_parameter_variations(
                variation_type="lookback",
                model=model,
                prices=prices,
                config=config,
            )
            param_results.extend(
                self._run_parameter_variations(
                    variation_type="top_pct",
                    model=model,
                    prices=prices,
                    config=config,
                )
            )

            # 9. Calculate time/regime splits
            # BacktestResult has equity_curve, compute returns from it
            equity_curve = backtest_result.equity_curve
            if equity_curve is not None and len(equity_curve) > 1:
                returns = equity_curve.pct_change().dropna()
            else:
                returns = pd.Series()
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
                    "sharpe": backtest_result.sharpe,
                    "max_drawdown": backtest_result.max_drawdown,
                    "total_return": backtest_result.total_return,
                },
                hypothesis_id=hypothesis_id,
            )

            # 13. Build detailed result for report
            return HypothesisBacktestResult(
                hypothesis_id=hypothesis_id,
                title=hypothesis.get("title", "Unknown"),
                sharpe=backtest_result.sharpe,
                total_return=backtest_result.total_return,
                max_drawdown=backtest_result.max_drawdown,
                volatility=backtest_result.volatility,
                win_rate=backtest_result.win_rate,
                period_metrics=time_metrics,
                regime_metrics=regime_metrics,
                parameter_variations=[
                    {
                        "variation_name": v.variation_name,
                        "params": v.params,
                        "sharpe": v.sharpe,
                        "max_drawdown": v.max_drawdown,
                        "total_return": v.total_return,
                    }
                    for v in param_results
                ],
                num_trades=trade_stats.get("num_trades", 0),
                avg_trade_value=trade_stats.get("avg_trade_value", 0),
                features=ml_config.get("features", []),
                model_type=ml_config.get("model_type", "unknown"),
            )

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
            # Build param_experiments dict for Validation Analyst
            # Format: {"baseline": {metrics}, "variation_name": {metrics}, ...}
            param_experiments = {
                "baseline": {
                    "sharpe": backtest_result.sharpe,
                    "max_drawdown": backtest_result.max_drawdown,
                    "total_return": backtest_result.total_return,
                    "params": {"top_pct": self.top_pct, "lookback": 252},
                },
            }
            for v in param_results:
                param_experiments[v.variation_name] = {
                    "sharpe": v.sharpe,
                    "max_drawdown": v.max_drawdown,
                    "total_return": v.total_return,
                    "params": v.params,
                }

            # Build metadata dict
            metadata = {
                "quant_developer_backtest": {
                    "date": date.today().isoformat(),
                    "sharpe": backtest_result.sharpe,
                    "max_drawdown": backtest_result.max_drawdown,
                    "total_return": backtest_result.total_return,
                    "volatility": backtest_result.volatility,
                    "win_rate": backtest_result.win_rate,
                    "parameter_variations": [
                        {
                            "variation_name": v.variation_name,
                            "params": v.params,
                            "sharpe": v.sharpe,
                            "max_drawdown": v.max_drawdown,
                            "total_return": v.total_return,
                        }
                        for v in param_results
                    ],
                },
                "param_experiments": param_experiments,  # For Validation Analyst
                "period_metrics": time_metrics,
                "regime_metrics": regime_metrics,
                "num_trades": trade_stats.get("num_trades", 0),
                "avg_trade_value": trade_stats.get("avg_trade_value", 0),
                "gross_return": trade_stats.get("gross_return", 0),
                "pipeline_stage": "quant_backtest",  # Track stage in metadata (column doesn't exist)
            }

            # Update hypothesis
            self.api.update_hypothesis(
                hypothesis_id=hypothesis_id,
                status="validated",
                metadata=metadata,
                actor=self.ACTOR,
            )

        except Exception as e:
            logger.error(f"Failed to update hypothesis {hypothesis_id}: {e}")

    def _write_research_note(self, report: QuantDeveloperReport) -> None:
        """
        Write comprehensive research note to output/research/.

        Includes per-hypothesis:
        - Base backtest metrics (Sharpe, return, drawdown, volatility, win rate)
        - Yearly performance breakdown
        - Regime analysis (HMM-based)
        - Parameter sensitivity matrix
        - Trade statistics

        Args:
            report: QuantDeveloperReport with detailed results
        """
        from hrp.agents.report_formatting import (
            render_footer,
            render_header,
            render_health_gauges,
            render_kpi_dashboard,
            render_section_divider,
            render_status_table,
        )
        from hrp.agents.output_paths import research_note_path

        report_date = report.report_date.isoformat()
        filepath = research_note_path("05-quant-developer")

        parts: list[str] = []

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_header(
            title="Quant Developer Report",
            report_type="agent-execution",
            date_str=report_date,
        ))

        # ══════════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY KPIs
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_kpi_dashboard([
            {"icon": "📋", "label": "Processed", "value": report.hypotheses_processed, "detail": "hypotheses"},
            {"icon": "✅", "label": "Completed", "value": report.backtests_completed, "detail": "backtests"},
            {"icon": "❌", "label": "Failed", "value": report.backtests_failed, "detail": "backtests"},
            {"icon": "⏱️", "label": "Duration", "value": f"{report.duration_seconds:.1f}s", "detail": "elapsed"},
        ]))

        # ══════════════════════════════════════════════════════════════════════
        # CONFIGURATION
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_section_divider("⚙️ Configuration"))
        parts.append("```")
        parts.append(f"  Signal method:    {self.signal_method}")
        parts.append(f"  Top percentile:   {self.top_pct:.1%}")
        parts.append(f"  Max positions:    {self.max_positions}")
        parts.append(f"  Commission:       {self.commission_bps} bps")
        parts.append(f"  Slippage:         {self.slippage_bps} bps")
        parts.append("```\n")

        # ══════════════════════════════════════════════════════════════════════
        # PER-HYPOTHESIS DETAILED RESULTS
        # ══════════════════════════════════════════════════════════════════════
        if report.results:
            for result in report.results:
                parts.append(render_section_divider(f"📊 {result.hypothesis_id}"))
                parts.append(f"**{result.title}**\n")
                parts.append(f"Model: `{result.model_type}` | Features: `{', '.join(result.features)}`\n")

                # ── Base Backtest Metrics ──
                parts.append("### 📈 Base Backtest Metrics\n")
                parts.append("| Metric | Value |")
                parts.append("|--------|-------|")
                parts.append(f"| Sharpe Ratio | {result.sharpe:.4f} |")
                parts.append(f"| Total Return | {result.total_return:.2%} |")
                parts.append(f"| Max Drawdown | {result.max_drawdown:.2%} |")
                parts.append(f"| Volatility | {result.volatility:.2%} |")
                parts.append(f"| Win Rate | {result.win_rate:.2%} |")
                parts.append(f"| Trades | {result.num_trades} |")
                parts.append(f"| Avg Trade Value | ${result.avg_trade_value:,.0f} |")
                parts.append("")

                # ── Yearly Performance ──
                if result.period_metrics:
                    parts.append("### 📅 Yearly Performance\n")
                    parts.append("| Year | Sharpe | Return | Max DD | Days |")
                    parts.append("|------|--------|--------|--------|------|")
                    for pm in result.period_metrics:
                        year = pm.get("period", "?")
                        sharpe = pm.get("sharpe", 0)
                        ret = pm.get("total_return", 0)
                        dd = pm.get("max_drawdown", 0)
                        days = pm.get("num_days", 0)
                        parts.append(f"| {year} | {sharpe:.2f} | {ret:.1%} | {dd:.1%} | {days} |")
                    parts.append("")

                # ── Regime Analysis ──
                if result.regime_metrics:
                    parts.append("### 🌡️ Regime Analysis (HMM)\n")
                    parts.append("| Regime | Sharpe | Return | Days |")
                    parts.append("|--------|--------|--------|------|")
                    for regime_id, metrics in result.regime_metrics.items():
                        sharpe = metrics.get("sharpe", 0)
                        ret = metrics.get("total_return", 0)
                        days = metrics.get("num_days", 0)
                        parts.append(f"| {regime_id} | {sharpe:.2f} | {ret:.1%} | {days} |")
                    parts.append("")

                # ── Parameter Sensitivity ──
                if result.parameter_variations:
                    parts.append("### 🔧 Parameter Sensitivity\n")
                    parts.append("| Variation | Sharpe | Return | Max DD |")
                    parts.append("|-----------|--------|--------|--------|")
                    for pv in result.parameter_variations:
                        name = pv.get("variation_name", "?")
                        sharpe = pv.get("sharpe", 0)
                        ret = pv.get("total_return", 0)
                        dd = pv.get("max_drawdown", 0)
                        parts.append(f"| {name} | {sharpe:.2f} | {ret:.1%} | {dd:.1%} |")
                    parts.append("")

                parts.append("")

        # ══════════════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════════════
        parts.append(render_footer(
            agent_name="quant-developer",
            duration_seconds=report.duration_seconds,
        ))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("\n".join(parts))
        logger.info(f"Research note written to {filepath}")
