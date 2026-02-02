"""
Platform API for HRP.

The Platform API is the single entry point for all operations.
All consumers (dashboard, MCP, agents) use this API - no direct database access.
"""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.exceptions import NotFoundError, PermissionError, PlatformAPIError
from hrp.research.config import BacktestConfig, BacktestResult
from hrp.api.validators import Validator


class PlatformAPI:
    """
    Single entry point for all HRP operations.

    All consumers (dashboard, MCP servers, agents) use this API.
    Direct database access is prohibited outside this class.

    Usage:
        api = PlatformAPI()
        prices = api.get_prices(['AAPL', 'MSFT'], start_date, end_date)
        hypothesis_id = api.create_hypothesis(
            title="Momentum predicts returns",
            thesis="Stocks with high 12-month returns continue outperforming",
            prediction="Top decile momentum > SPY by 3% annually",
            falsification="Sharpe < SPY or p-value > 0.05",
            actor='user'
        )
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        """
        Initialize the Platform API.

        Args:
            db_path: Optional path to database (uses default if not provided)
            read_only: If True, use read-only database connection (default: False)
        """
        self._db = get_db(db_path, read_only=read_only)
        logger.debug("PlatformAPI initialized")

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_symbols_in_universe(self, symbols: List[str], as_of_date: date = None) -> None:
        """Validate that all symbols are in the universe."""
        # Get valid symbols from universe
        if as_of_date:
            query = """
                SELECT DISTINCT symbol FROM universe
                WHERE in_universe = TRUE AND date = ?
            """
            result = self._db.fetchall(query, (as_of_date,))
        else:
            query = """
                SELECT DISTINCT symbol FROM universe
                WHERE in_universe = TRUE
            """
            result = self._db.fetchall(query)

        valid_symbols = {row[0] for row in result}
        invalid = [s for s in symbols if s not in valid_symbols]

        if invalid:
            invalid_str = ", ".join(sorted(invalid))
            if as_of_date:
                raise ValueError(f"Invalid symbols not in universe as of {as_of_date}: {invalid_str}")
            else:
                raise ValueError(f"Invalid symbols not found in universe: {invalid_str}")

    # =========================================================================
    # Data Operations
    # =========================================================================

    def get_prices(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Get price data for symbols over a date range.

        Args:
            symbols: List of ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
        """
        # Validate inputs
        Validator.not_future(start, "start date")
        Validator.not_future(end, "end date")
        Validator.date_range(start, end)

        if not symbols:
            raise ValueError("symbols list cannot be empty")

        self._validate_symbols_in_universe(symbols)

        symbols_str = ",".join(f"'{s}'" for s in symbols)
        query = f"""
            SELECT symbol, date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY date, symbol
        """

        df = self._db.fetchdf(query, (start, end))

        if df.empty:
            logger.warning(f"No price data found for {symbols} from {start} to {end}")
        else:
            logger.debug(f"Retrieved {len(df)} price records for {len(symbols)} symbols")

        return df

    def get_features(
        self,
        symbols: List[str],
        features: List[str],
        as_of_date: date,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Get feature values for symbols as of a specific date.

        Args:
            symbols: List of ticker symbols
            features: List of feature names to retrieve
            as_of_date: Date to get features for
            version: Feature version (default 'v1')

        Returns:
            DataFrame pivoted with symbols as rows and features as columns
        """
        # Validate inputs
        Validator.not_future(as_of_date, "as_of_date")

        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if not features:
            raise ValueError("features list cannot be empty")

        self._validate_symbols_in_universe(symbols, as_of_date)

        symbols_str = ",".join(f"'{s}'" for s in symbols)
        features_str = ",".join(f"'{f}'" for f in features)

        query = f"""
            SELECT symbol, feature_name, value
            FROM features
            WHERE symbol IN ({symbols_str})
              AND feature_name IN ({features_str})
              AND date = ?
              AND version = ?
        """

        df = self._db.fetchdf(query, (as_of_date, version))

        if df.empty:
            logger.warning(f"No features found for {symbols} on {as_of_date}")
            return df

        # Pivot to get features as columns
        pivoted = df.pivot(index="symbol", columns="feature_name", values="value")
        pivoted = pivoted.reset_index()

        logger.debug(f"Retrieved features for {len(pivoted)} symbols")
        return pivoted

    def get_fundamentals_as_of(
        self,
        symbols: List[str],
        metrics: List[str],
        as_of_date: date,
    ) -> pd.DataFrame:
        """
        Get fundamental metrics for symbols as of a specific date (point-in-time).

        Only returns fundamentals where report_date <= as_of_date to prevent
        look-ahead bias in backtests. For each symbol/metric combination, returns
        the most recent fundamental available by the as_of_date.

        Args:
            symbols: List of ticker symbols
            metrics: List of fundamental metric names (e.g., 'revenue', 'eps')
            as_of_date: Date to get fundamentals for (only data available by this date)

        Returns:
            DataFrame with columns: symbol, metric, value, report_date, period_end
            Returns empty DataFrame if no data is available (not an error).

        Example:
            # Get fundamentals as they would have been known on 2023-01-15
            df = api.get_fundamentals_as_of(
                symbols=['AAPL', 'MSFT'],
                metrics=['revenue', 'eps', 'book_value'],
                as_of_date=date(2023, 1, 15)
            )
        """
        # Validate inputs
        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if not metrics:
            raise ValueError("metrics list cannot be empty")
        Validator.not_future(as_of_date, "as_of_date")

        # Build query with parameterized values for symbols and metrics
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        metrics_str = ",".join(f"'{m}'" for m in metrics)

        # Use window function to get the most recent report for each symbol/metric
        # where report_date <= as_of_date
        query = f"""
            WITH ranked_fundamentals AS (
                SELECT
                    symbol,
                    metric,
                    value,
                    report_date,
                    period_end,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, metric
                        ORDER BY report_date DESC
                    ) as rn
                FROM fundamentals
                WHERE symbol IN ({symbols_str})
                  AND metric IN ({metrics_str})
                  AND report_date <= ?
            )
            SELECT symbol, metric, value, report_date, period_end
            FROM ranked_fundamentals
            WHERE rn = 1
            ORDER BY symbol, metric
        """

        df = self._db.fetchdf(query, (as_of_date,))

        if df.empty:
            logger.debug(
                f"No fundamentals found for {symbols} with metrics {metrics} "
                f"as of {as_of_date}"
            )
        else:
            logger.debug(
                f"Retrieved {len(df)} fundamental records for {len(df['symbol'].unique())} "
                f"symbols as of {as_of_date}"
            )

        return df

    def get_universe(self, as_of_date: date) -> List[str]:
        """
        Get the trading universe as of a specific date.

        Args:
            as_of_date: Date to get universe for

        Returns:
            List of ticker symbols in the universe
        """
        Validator.not_future(as_of_date, "as_of_date")

        query = """
            SELECT symbol
            FROM universe
            WHERE date = ?
              AND in_universe = TRUE
            ORDER BY symbol
        """

        result = self._db.fetchall(query, (as_of_date,))
        symbols = [row[0] for row in result]

        logger.debug(f"Universe contains {len(symbols)} symbols as of {as_of_date}")
        return symbols

    def get_corporate_actions(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Get corporate actions for symbols over a date range.

        Args:
            symbols: List of ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with columns: symbol, date, action_type, factor, source
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        symbols_str = ",".join(f"'{s}'" for s in symbols)
        query = f"""
            SELECT symbol, date, action_type, factor, source
            FROM corporate_actions
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY date, symbol, action_type
        """

        df = self._db.fetchdf(query, (start, end))

        if df.empty:
            logger.warning(f"No corporate actions found for {symbols} from {start} to {end}")
        else:
            logger.debug(f"Retrieved {len(df)} corporate action records for {len(symbols)} symbols")

        return df

    def is_trading_day(self, trading_date: date) -> bool:
        """
        Check if a date is a valid NYSE trading day.

        Args:
            trading_date: Date to check

        Returns:
            True if the date is a weekday and not a NYSE holiday, False otherwise

        Examples:
            >>> api = PlatformAPI()
            >>> api.is_trading_day(date(2022, 7, 4))  # Independence Day
            False
            >>> api.is_trading_day(date(2022, 7, 5))  # Regular Tuesday
            True
        """
        from hrp.utils.calendar import is_trading_day

        return is_trading_day(trading_date)

    def get_trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        """
        Get all NYSE trading days within a date range (inclusive).

        Excludes weekends and NYSE holidays. Returns empty index if range
        contains no trading days.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DatetimeIndex of trading days in chronological order

        Raises:
            ValueError: If start > end

        Examples:
            >>> api = PlatformAPI()
            >>> days = api.get_trading_days(date(2022, 1, 1), date(2022, 1, 31))
            >>> len(days)  # January 2022 has 20 trading days
            20
        """
        from hrp.utils.calendar import get_trading_days

        return get_trading_days(start, end)

    # =========================================================================
    # Hypothesis Operations
    # =========================================================================

    def create_hypothesis(
        self,
        title: str,
        thesis: str,
        prediction: str,
        falsification: str,
        actor: str = "user",
        strategy_class: Optional[str] = None,
    ) -> str:
        """
        Create a new research hypothesis.

        Args:
            title: Short descriptive title
            thesis: The hypothesis being tested
            prediction: Testable prediction
            falsification: Criteria for falsifying the hypothesis
            actor: Who is creating the hypothesis ('user' or 'agent:<name>')
            strategy_class: Optional strategy class (cross_sectional_factor, time_series_momentum, ml_composite)

        Returns:
            hypothesis_id: Unique identifier for the hypothesis
        """
        # Validate inputs
        Validator.not_empty(title, "title")
        Validator.not_empty(thesis, "thesis")
        Validator.not_empty(prediction, "prediction")
        Validator.not_empty(falsification, "falsification")
        Validator.not_empty(actor, "actor")

        hypothesis_id = self._generate_hypothesis_id()

        # Build metadata JSON if strategy_class provided
        metadata_json = None
        if strategy_class:
            import json
            metadata_json = json.dumps({"strategy_class": strategy_class})

        query = """
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, created_by, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?)
        """

        self._db.execute(
            query, (hypothesis_id, title, thesis, prediction, falsification, actor, metadata_json)
        )

        # Log to lineage
        self.log_event(
            event_type="hypothesis_created",
            actor=actor,
            details={"hypothesis_id": hypothesis_id, "title": title},
            hypothesis_id=hypothesis_id,
        )

        logger.info(f"Created hypothesis {hypothesis_id}: {title}")
        return hypothesis_id

    def update_hypothesis(
        self,
        hypothesis_id: str,
        status: Optional[str] = None,
        outcome: Optional[str] = None,
        actor: str = "user",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Update a hypothesis status, outcome, and/or metadata.

        Args:
            hypothesis_id: ID of the hypothesis to update
            status: New status ('draft', 'testing', 'validated', 'rejected', 'deployed').
                    If None, keeps current status (useful for metadata-only updates).
            outcome: Optional outcome description
            actor: Who is making the update
            metadata: Optional metadata dict to store (merged with existing)

        Raises:
            NotFoundError: If hypothesis doesn't exist
            ValueError: If neither status nor metadata is provided
        """
        # Validate inputs
        Validator.not_empty(hypothesis_id, "hypothesis_id")
        Validator.not_empty(actor, "actor")

        if status is not None:
            Validator.not_empty(status, "status")

        if status is None and metadata is None and outcome is None:
            raise ValueError("At least one of status, outcome, or metadata must be provided")

        existing = self.get_hypothesis(hypothesis_id)
        if not existing:
            raise NotFoundError(f"Hypothesis {hypothesis_id} not found")

        # Guard: cannot validate without linked experiments
        if status == "validated" and existing["status"] == "testing":
            experiments = self.get_experiments_for_hypothesis(hypothesis_id)
            if not experiments:
                raise ValueError(
                    f"Cannot validate {hypothesis_id}: no linked experiments. "
                    "Run walk-forward validation or backtest first."
                )

        # Use current status if none provided
        effective_status = status if status is not None else existing["status"]

        # Handle metadata update
        if metadata is not None:
            # Merge with existing metadata
            existing_metadata = existing.get("metadata") or {}
            if isinstance(existing_metadata, str):
                try:
                    existing_metadata = json.loads(existing_metadata)
                except json.JSONDecodeError:
                    existing_metadata = {}
            merged_metadata = {**existing_metadata, **metadata}
            metadata_json = json.dumps(merged_metadata)

            query = """
                UPDATE hypotheses
                SET status = ?, outcome = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE hypothesis_id = ?
            """
            self._db.execute(query, (effective_status, outcome, metadata_json, hypothesis_id))
        else:
            query = """
                UPDATE hypotheses
                SET status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP
                WHERE hypothesis_id = ?
            """
            self._db.execute(query, (effective_status, outcome, hypothesis_id))

        self.log_event(
            event_type="hypothesis_updated",
            actor=actor,
            details={
                "hypothesis_id": hypothesis_id,
                "old_status": existing["status"],
                "new_status": effective_status,
                "outcome": outcome,
            },
            hypothesis_id=hypothesis_id,
        )

        logger.info(f"Updated hypothesis {hypothesis_id}: status={effective_status}")

    def list_hypotheses(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        List hypotheses, optionally filtered by status.

        Args:
            status: Optional status filter
            limit: Maximum number of results (default 100)

        Returns:
            List of hypothesis dictionaries
        """
        Validator.positive(limit, "limit")

        query = """
            SELECT hypothesis_id, title, thesis, testable_prediction,
                   falsification_criteria, status, created_at, created_by,
                   updated_at, outcome, confidence_score
            FROM hypotheses
        """
        params: List[Any] = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        result = self._db.fetchall(query, tuple(params))

        return [
            {
                "hypothesis_id": row[0],
                "title": row[1],
                "thesis": row[2],
                "prediction": row[3],
                "falsification": row[4],
                "status": row[5],
                "created_at": row[6],
                "created_by": row[7],
                "updated_at": row[8],
                "outcome": row[9],
                "confidence_score": row[10],
            }
            for row in result
        ]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict]:
        """
        Get a single hypothesis by ID.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            Hypothesis dictionary or None if not found
        """
        Validator.not_empty(hypothesis_id, "hypothesis_id")

        query = """
            SELECT hypothesis_id, title, thesis, testable_prediction,
                   falsification_criteria, status, created_at, created_by,
                   updated_at, outcome, confidence_score
            FROM hypotheses
            WHERE hypothesis_id = ?
        """

        row = self._db.fetchone(query, (hypothesis_id,))

        if not row:
            return None

        return {
            "hypothesis_id": row[0],
            "title": row[1],
            "thesis": row[2],
            "prediction": row[3],
            "falsification": row[4],
            "status": row[5],
            "created_at": row[6],
            "created_by": row[7],
            "updated_at": row[8],
            "outcome": row[9],
            "confidence_score": row[10],
        }

    # =========================================================================
    # Experiment Operations
    # =========================================================================

    def run_backtest(
        self,
        config: BacktestConfig,
        signals: Optional[pd.DataFrame] = None,
        hypothesis_id: Optional[str] = None,
        actor: str = "user",
        experiment_name: str = "backtests",
    ) -> str:
        """
        Run a backtest and log results to MLflow.

        Args:
            config: BacktestConfig with symbols, dates, and parameters
            signals: Optional signals DataFrame (1=long, 0=no position)
            hypothesis_id: Optional linked hypothesis
            actor: Who is running the backtest
            experiment_name: MLflow experiment name

        Returns:
            experiment_id: The MLflow run ID
        """
        from hrp.research.backtest import generate_momentum_signals, get_price_data
        from hrp.research.backtest import run_backtest as execute_backtest
        from hrp.research.mlflow_utils import log_backtest

        # Load prices
        prices = get_price_data(config.symbols, config.start_date, config.end_date)

        # Generate signals if not provided
        if signals is None:
            logger.info("No signals provided, generating momentum signals")
            signals = generate_momentum_signals(
                prices, lookback=252, top_n=min(10, config.max_positions)
            )

        # Run the backtest
        result = execute_backtest(signals, config, prices)

        # Log to MLflow
        run_name = config.name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_id = log_backtest(
            result=result,
            experiment_name=experiment_name,
            run_name=run_name,
            hypothesis_id=hypothesis_id,
            tags={"actor": actor},
        )

        # Link to hypothesis if provided
        if hypothesis_id:
            self.link_experiment(hypothesis_id, experiment_id)

        # Log to lineage
        self.log_event(
            event_type="backtest_run",
            actor=actor,
            details={
                "experiment_id": experiment_id,
                "config_name": config.name,
                "symbols_count": len(config.symbols),
                "start_date": str(config.start_date),
                "end_date": str(config.end_date),
                "sharpe_ratio": result.metrics.get("sharpe_ratio"),
                "total_return": result.metrics.get("total_return"),
            },
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )

        logger.info(f"Backtest complete: {experiment_id} | Sharpe: {result.sharpe:.2f}")
        return experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        Get experiment details from MLflow.

        Args:
            experiment_id: The MLflow run ID

        Returns:
            Dictionary with experiment details or None if not found
        """
        import mlflow

        from hrp.research.mlflow_utils import setup_mlflow

        setup_mlflow()

        try:
            run = mlflow.get_run(experiment_id)
        except Exception as e:
            logger.warning(f"Experiment {experiment_id} not found: {e}")
            return None

        return {
            "experiment_id": run.info.run_id,
            "experiment_name": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple experiments side by side.

        Args:
            experiment_ids: List of MLflow run IDs to compare
            metrics: Optional list of metrics to include

        Returns:
            DataFrame with experiments as rows and metrics as columns
        """
        if not experiment_ids:
            return pd.DataFrame()

        if metrics is None:
            metrics = [
                "sharpe_ratio",
                "sortino_ratio",
                "total_return",
                "cagr",
                "max_drawdown",
                "volatility",
                "calmar_ratio",
                "win_rate",
            ]

        rows = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                row = {"experiment_id": exp_id}
                row.update({m: exp["metrics"].get(m) for m in metrics})
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("experiment_id")

        return df

    # =========================================================================
    # Deployment Operations
    # =========================================================================

    def approve_deployment(self, hypothesis_id: str, actor: str) -> bool:
        """
        Approve a hypothesis for deployment.

        IMPORTANT: Agents cannot approve deployments. Only users can approve.

        Args:
            hypothesis_id: ID of the hypothesis to deploy
            actor: Who is approving (must not start with 'agent:')

        Returns:
            True if deployment approved

        Raises:
            PermissionError: If actor is an agent
            NotFoundError: If hypothesis doesn't exist
        """
        if actor.startswith("agent:"):
            logger.warning(
                f"Agent {actor} attempted to approve deployment for {hypothesis_id}"
            )
            raise PermissionError(
                f"Agents cannot approve deployments. Actor '{actor}' is not permitted."
            )

        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise NotFoundError(f"Hypothesis {hypothesis_id} not found")

        if hypothesis["status"] not in ("validated", "testing"):
            logger.warning(
                f"Cannot deploy hypothesis {hypothesis_id} with status {hypothesis['status']}"
            )
            return False

        self.update_hypothesis(hypothesis_id, status="deployed", actor=actor)

        self.log_event(
            event_type="deployment_approved",
            actor=actor,
            details={
                "hypothesis_id": hypothesis_id,
                "title": hypothesis["title"],
                "previous_status": hypothesis["status"],
            },
            hypothesis_id=hypothesis_id,
        )

        logger.info(f"Deployment approved for hypothesis {hypothesis_id} by {actor}")
        return True

    def get_deployed_strategies(self) -> List[Dict]:
        """
        Get all currently deployed strategies.

        Returns:
            List of deployed hypothesis dictionaries
        """
        return self.list_hypotheses(status="deployed")

    # =========================================================================
    # Lineage Operations
    # =========================================================================

    def get_lineage(
        self,
        hypothesis_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get lineage/audit trail events.

        Args:
            hypothesis_id: Optional filter by hypothesis
            experiment_id: Optional filter by experiment
            limit: Maximum number of results

        Returns:
            List of lineage event dictionaries
        """
        query = """
            SELECT lineage_id, event_type, timestamp, actor,
                   hypothesis_id, experiment_id, details, parent_lineage_id
            FROM lineage
            WHERE 1=1
        """
        params: List[Any] = []

        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)

        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        result = self._db.fetchall(query, tuple(params))

        events = []
        for row in result:
            details = row[6]
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except json.JSONDecodeError:
                    details = {"raw": details}

            events.append(
                {
                    "lineage_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "actor": row[3],
                    "hypothesis_id": row[4],
                    "experiment_id": row[5],
                    "details": details,
                    "parent_lineage_id": row[7],
                }
            )

        return events

    def log_event(
        self,
        event_type: str,
        actor: str,
        details: Optional[Dict] = None,
        hypothesis_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        parent_lineage_id: Optional[int] = None,
    ) -> int:
        """
        Log an event to the lineage table.

        Args:
            event_type: Type of event (e.g., 'hypothesis_created', 'experiment_run')
            actor: Who triggered the event ('user' or 'agent:<name>')
            details: Optional dictionary of event details
            hypothesis_id: Optional associated hypothesis
            experiment_id: Optional associated experiment
            parent_lineage_id: Optional parent event for chaining

        Returns:
            lineage_id of the created event
        """
        details_json = json.dumps(details) if details else None

        result = self._db.fetchone(
            "SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM lineage"
        )
        lineage_id = result[0]

        query = """
            INSERT INTO lineage (
                lineage_id, event_type, actor, hypothesis_id,
                experiment_id, details, parent_lineage_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        self._db.execute(
            query,
            (
                lineage_id,
                event_type,
                actor,
                hypothesis_id,
                experiment_id,
                details_json,
                parent_lineage_id,
            ),
        )

        logger.debug(f"Logged lineage event: {event_type} by {actor}")
        return lineage_id

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_hypothesis_id(self) -> str:
        """Generate a unique hypothesis ID in format HYP-YYYY-NNN."""
        year = datetime.now().year

        result = self._db.fetchone(
            "SELECT COUNT(*) FROM hypotheses WHERE hypothesis_id LIKE ?",
            (f"HYP-{year}-%",),
        )
        count = result[0] + 1 if result else 1

        return f"HYP-{year}-{count:03d}"

    def link_experiment(
        self,
        hypothesis_id: str,
        experiment_id: str,
        relationship: str = "primary",
    ) -> None:
        """Link an experiment to a hypothesis."""
        query = """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id, relationship)
            VALUES (?, ?, ?)
            ON CONFLICT DO NOTHING
        """

        self._db.execute(query, (hypothesis_id, experiment_id, relationship))
        logger.debug(f"Linked experiment {experiment_id} to hypothesis {hypothesis_id}")

    def get_experiments_for_hypothesis(self, hypothesis_id: str) -> List[str]:
        """
        Get all experiment IDs linked to a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            List of experiment IDs
        """
        query = """
            SELECT experiment_id
            FROM hypothesis_experiments
            WHERE hypothesis_id = ?
            ORDER BY created_at DESC
        """

        result = self._db.fetchall(query, (hypothesis_id,))
        return [row[0] for row in result]

    def health_check(self) -> dict:
        """
        Check API and database health.

        Returns:
            Dictionary with health status including scheduler information
        """
        status: dict[str, Any] = {
            "api": "ok",
            "database": "unknown",
            "tables": {},
            "scheduler": {},
        }

        try:
            self._db.fetchone("SELECT 1")
            status["database"] = "ok"

            from hrp.data.schema import TABLES

            for table_name in TABLES.keys():
                try:
                    result = self._db.fetchone(f"SELECT COUNT(*) FROM {table_name}")
                    status["tables"][table_name] = {"status": "ok", "count": result[0]}
                except Exception:
                    status["tables"][table_name] = {"status": "missing", "count": 0}

        except Exception as e:
            status["database"] = f"error: {str(e)}"

        # Check scheduler status
        try:
            from hrp.utils.scheduler import get_scheduler_status

            scheduler_status = get_scheduler_status()
            status["scheduler"] = {
                "is_installed": scheduler_status.is_installed,
                "is_running": scheduler_status.is_running,
                "pid": scheduler_status.pid,
                "command": scheduler_status.command,
            }
        except Exception as e:
            logger.warning(f"Failed to check scheduler status: {e}")
            status["scheduler"] = {"error": str(e)}

        return status

    def run_quality_checks(
        self,
        as_of_date: date | None = None,
        checks: list[str] | None = None,
        send_alerts: bool = False,
    ) -> dict[str, Any]:
        """
        Run data quality checks and return results.

        Args:
            as_of_date: Date to run checks for (default: today)
            checks: List of check names to run (default: all checks)
            send_alerts: Whether to send email alerts for critical issues

        Returns:
            Dictionary with quality check results

        Raises:
            ValueError: If check names are invalid
        """
        from hrp.data.quality.checks import (
            DEFAULT_CHECKS,
            PriceAnomalyCheck,
            CompletenessCheck,
            GapDetectionCheck,
            StaleDataCheck,
            VolumeAnomalyCheck,
        )
        from hrp.data.quality.report import QualityReportGenerator

        as_of_date = as_of_date or date.today()

        # Map check names to classes
        check_classes = {
            "price_anomaly": PriceAnomalyCheck,
            "completeness": CompletenessCheck,
            "gap_detection": GapDetectionCheck,
            "stale_data": StaleDataCheck,
            "volume_anomaly": VolumeAnomalyCheck,
        }

        # Validate check names
        if checks:
            invalid = [c for c in checks if c not in check_classes]
            if invalid:
                raise ValueError(f"Invalid check names: {invalid}")
            checks_to_run = [check_classes[c] for c in checks]
        else:
            checks_to_run = DEFAULT_CHECKS

        # Run checks
        generator = QualityReportGenerator(checks=checks_to_run)
        report = generator.generate_report(as_of_date)

        # Send alerts if requested
        if send_alerts and report.critical_issues > 0:
            self._send_quality_alerts(report)

        return {
            "health_score": report.health_score,
            "critical_issues": report.critical_issues,
            "warning_issues": report.warning_issues,
            "passed": report.passed,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "critical_count": r.critical_count,
                    "warning_count": r.warning_count,
                    "run_time_ms": r.run_time_ms,
                    "issues": [i.to_dict() for i in r.issues],
                }
                for r in report.results
            ],
            "generated_at": report.generated_at.isoformat(),
        }

    def get_quality_trend(self, days: int = 30) -> dict[str, Any]:
        """
        Get historical quality scores for trend analysis.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with trend data
        """
        from hrp.data.quality.report import QualityReportGenerator

        generator = QualityReportGenerator()
        trend_data = generator.get_health_trend(days=days)

        if not trend_data:
            return {
                "dates": [],
                "health_scores": [],
                "critical_issues": [],
                "warning_issues": [],
            }

        return {
            "dates": [d["date"].isoformat() for d in trend_data],
            "health_scores": [d["health_score"] for d in trend_data],
            "critical_issues": [d.get("critical_issues", 0) for d in trend_data],
            "warning_issues": [d.get("warning_issues", 0) for d in trend_data],
        }

    def get_data_health_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for data health dashboard.

        Returns:
            Dictionary with health metrics
        """
        with self._db.connection() as conn:
            # Symbol count
            symbol_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM prices"
            ).fetchone()[0]

            # Date range
            date_range = conn.execute(
                "SELECT MIN(date), MAX(date) FROM prices"
            ).fetchone()

            # Total records
            prices_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            features_count = conn.execute(
                "SELECT COUNT(*) FROM features"
            ).fetchone()[0]
            fundamentals_count = conn.execute(
                "SELECT COUNT(*) FROM fundamentals"
            ).fetchone()[0]

            # Data freshness
            last_price_date = conn.execute(
                "SELECT MAX(date) FROM prices"
            ).fetchone()[0]
            if last_price_date:
                days_stale = (date.today() - last_price_date).days
                is_fresh = days_stale <= 3
            else:
                days_stale = None
                is_fresh = False

            # Ingestion summary
            ingestion_summary = conn.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN LOWER(status) = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    MAX(completed_at) as last_successful
                FROM ingestion_log
            """).fetchone()

        return {
            "symbol_count": symbol_count,
            "date_range": {
                "start": str(date_range[0]) if date_range[0] else None,
                "end": str(date_range[1]) if date_range[1] else None,
            },
            "total_records": {
                "prices": prices_count,
                "features": features_count,
                "fundamentals": fundamentals_count,
            },
            "data_freshness": {
                "last_date": str(last_price_date) if last_price_date else None,
                "days_stale": days_stale,
                "is_fresh": is_fresh,
            },
            "ingestion_summary": {
                "total_runs": ingestion_summary[0] or 0,
                "success_rate": round(ingestion_summary[1] or 0, 1),
                "last_successful": str(ingestion_summary[2]) if ingestion_summary[2] else None,
            },
        }

    def _send_quality_alerts(self, report) -> None:
        """Send email alerts for critical quality issues."""
        try:
            from hrp.notifications.email import EmailNotifier

            notifier = EmailNotifier()
            notifier.send_summary_email(
                subject=f"HRP Quality Alert: {report.critical_issues} critical issues (score: {report.health_score:.0f})",
                summary_data={
                    "health_score": report.health_score,
                    "critical_issues": report.critical_issues,
                    "warning_issues": report.warning_issues,
                    "timestamp": report.generated_at.isoformat(),
                },
            )
            logger.info(
                f"Sent quality alerts: {report.critical_issues} critical issues"
            )
        except Exception as e:
            logger.error(f"Failed to send quality alerts: {e}")

    # === ML Model Registry Methods ===

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        features: list[str],
        target: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        training_date: date,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
    ) -> str:
        """
        Register a trained model in the Model Registry.

        Args:
            model: Trained model object
            model_name: Name for the registered model
            model_type: Type of model (e.g., "ridge", "lightgbm")
            features: List of feature names used for training
            target: Target variable name
            metrics: Dictionary of performance metrics
            hyperparameters: Model hyperparameters
            training_date: Date when model was trained
            hypothesis_id: Associated hypothesis ID if applicable
            experiment_id: MLflow experiment ID for the run

        Returns:
            model_version: Version string of the registered model
        """
        from hrp.ml.registry import ModelRegistry

        registry = ModelRegistry()
        model_version = registry.register_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            features=features,
            target=target,
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_date=training_date,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )

        # Log to lineage
        self.log_event(
            event_type="experiment_linked" if hypothesis_id else "experiment_run",
            actor="user",
            details={
                "model_name": model_name,
                "model_version": model_version,
                "model_type": model_type,
                "features": features,
                "target": target,
                "metrics": metrics,
            },
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
        )

        return model_version

    def get_production_model(self, model_name: str) -> dict[str, Any] | None:
        """
        Get the current production model version.

        Args:
            model_name: Name of the registered model

        Returns:
            Dict with model details or None if no production model
        """
        from hrp.ml.registry import ModelRegistry

        registry = ModelRegistry()
        model = registry.get_production_model(model_name)

        if model is None:
            return None

        return {
            "model_name": model.model_name,
            "model_version": model.model_version,
            "model_type": model.model_type,
            "stage": model.stage,
            "run_id": model.run_id,
            "registered_at": model.registered_at,
        }

    def promote_model_to_production(
        self,
        model_name: str,
        model_version: str,
        actor: str = "user",
    ) -> None:
        """
        Promote a staging model to production.

        Args:
            model_name: Name of the registered model
            model_version: Version string to promote
            actor: Who is promoting (default: "user")
        """
        from hrp.ml.registry import ModelRegistry

        registry = ModelRegistry()
        registry.promote_to_production(
            model_name=model_name,
            model_version=model_version,
            actor=actor,
        )

        # Log to lineage
        self.log_event(
            event_type="deployment_approved",
            actor=actor,
            details={
                "model_name": model_name,
                "model_version": model_version,
                "stage": "production",
            },
        )

    def rollback_model(
        self,
        model_name: str,
        to_version: str,
        actor: str = "user",
        reason: str = "Manual rollback",
    ) -> None:
        """
        Rollback production model to a previous version.

        Args:
            model_name: Name of the registered model
            to_version: Version string to rollback to
            actor: Who is rolling back
            reason: Reason for rollback
        """
        from hrp.ml.registry import ModelRegistry

        registry = ModelRegistry()
        registry.rollback_production(
            model_name=model_name,
            to_version=to_version,
            actor=actor,
            reason=reason,
        )

        # Log to lineage
        self.log_event(
            event_type="validation_failed",
            actor=actor,
            details={
                "model_name": model_name,
                "to_version": to_version,
                "reason": reason,
            },
        )

    # === ML Drift Monitoring Methods ===

    def check_model_drift(
        self,
        model_name: str,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame | None = None,
        model_version: str | None = None,
        predictions_col: str = "prediction",
        target_col: str | None = None,
        reference_ic: float | None = None,
    ) -> dict[str, Any]:
        """
        Check for model drift and send alerts if detected.

        Args:
            model_name: Name of the model
            current_data: Current feature/prediction DataFrame
            reference_data: Reference DataFrame (None for auto-detect)
            model_version: Optional model version
            predictions_col: Column name for predictions
            target_col: Column name for target (for concept drift)
            reference_ic: Reference IC for concept drift check

        Returns:
            Dict with drift results and alerts
        """
        from hrp.monitoring.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        drift_results = monitor.run_drift_check(
            model_name=model_name,
            current_data=current_data,
            reference_data=reference_data,
            model_version=model_version,
            predictions_col=predictions_col,
            target_col=target_col,
            reference_ic=reference_ic,
        )

        # Convert to dict for JSON response
        results_dict = {
            name: result.to_dict() for name, result in drift_results.items()
        }

        # Add summary
        results_dict["summary"] = {
            "total_checks": len(drift_results),
            "drift_detected": any(r.is_drift_detected for r in drift_results.values()),
            "num_drifts": sum(1 for r in drift_results.values() if r.is_drift_detected),
        }

        # Log to lineage
        self.log_event(
            event_type="validation_passed"
            if not results_dict["summary"]["drift_detected"]
            else "validation_failed",
            actor="system",
            details={
                "model_name": model_name,
                "model_version": model_version,
                "drift_check_type": "full",
                **results_dict["summary"],
            },
        )

        return results_dict

    # === ML Deployment Methods ===

    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        validation_data: pd.DataFrame,
        actor: str = "user",
        environment: str = "staging",
    ) -> dict[str, Any]:
        """
        Deploy a model to staging or production.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            validation_data: Data for validation checks
            actor: Who is deploying
            environment: Target environment ("staging" or "production")

        Returns:
            Dict with deployment result
        """
        from hrp.ml.deployment import DeploymentPipeline

        pipeline = DeploymentPipeline()

        if environment == "staging":
            result = pipeline.deploy_to_staging(
                model_name=model_name,
                model_version=model_version,
                validation_data=validation_data,
                actor=actor,
            )
        elif environment == "production":
            result = pipeline.promote_to_production(
                model_name=model_name,
                actor=actor,
                model_version=model_version,
            )
        else:
            raise ValueError(f"Invalid environment: {environment}")

        # Log to lineage
        self.log_event(
            event_type="deployment_approved" if result.status == "success" else "deployment_rejected",
            actor=actor,
            details={
                "model_name": model_name,
                "model_version": model_version,
                "environment": environment,
                "status": result.status,
                "validation_passed": result.validation_passed,
            },
        )

        return result.to_dict()

    def rollback_deployment(
        self,
        model_name: str,
        to_version: str,
        actor: str = "user",
        reason: str = "Manual rollback",
    ) -> dict[str, Any]:
        """
        Rollback a deployment to a previous version.

        Args:
            model_name: Name of the model
            to_version: Version to rollback to
            actor: Who is rolling back
            reason: Reason for rollback

        Returns:
            Dict with rollback result
        """
        from hrp.ml.deployment import DeploymentPipeline

        pipeline = DeploymentPipeline()
        result = pipeline.rollback_production(
            model_name=model_name,
            to_version=to_version,
            actor=actor,
            reason=reason,
        )

        return result.to_dict()

    # === ML Inference Methods ===

    def predict_model(
        self,
        model_name: str,
        symbols: list[str],
        as_of_date: date,
        model_version: str | None = None,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate predictions using a deployed model.

        Args:
            model_name: Name of the registered model
            symbols: List of symbols to predict
            as_of_date: Date to generate predictions for
            model_version: Specific version (None for production)
            feature_names: Optional list of feature names

        Returns:
            DataFrame with columns: symbol, date, prediction, model_name, model_version
        """
        from hrp.ml.inference import ModelPredictor

        predictor = ModelPredictor(
            model_name=model_name,
            model_version=model_version,
        )

        predictions = predictor.predict_batch(
            symbols=symbols,
            as_of_date=as_of_date,
            feature_names=feature_names,
        )

        # Log to lineage
        self.log_event(
            event_type="experiment_run",
            actor="user",
            details={
                "model_name": model_name,
                "model_version": predictor.model_version,
                "num_predictions": len(predictions),
                "as_of_date": str(as_of_date),
            },
        )

        # Run drift check (non-blocking)
        try:
            drift_result = self.check_model_drift(
                model_name=model_name,
                current_data=predictions,
                reference_data=None,
            )
            if drift_result.get("summary", {}).get("drift_detected"):
                logger.warning(f"Drift detected for model {model_name}")
                self.log_event(
                    event_type="validation_failed",
                    actor="system",
                    details={
                        "model_name": model_name,
                        "drift_detected": True,
                        "num_drifts": drift_result.get("summary", {}).get("num_drifts"),
                    },
                )
        except Exception as e:
            logger.warning(f"Drift check failed for {model_name}: {e}")

        return predictions

    def get_model_predictions(
        self,
        model_name: str,
        start_date: date,
        end_date: date,
        model_version: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical predictions from performance history.

        Args:
            model_name: Name of the model
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            model_version: Optional model version filter

        Returns:
            DataFrame with historical predictions
        """
        query = """
            SELECT
                timestamp,
                metric_name,
                metric_value,
                sample_size
            FROM model_performance_history
            WHERE model_name = ?
              AND timestamp >= ?
              AND timestamp <= ?
        """

        params = [model_name, start_date, end_date]

        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)

        query += " ORDER BY timestamp DESC"

        try:
            return self._db.fetchdf(query, params)
        except Exception:
            return pd.DataFrame()
