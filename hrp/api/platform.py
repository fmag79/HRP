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
from hrp.research.config import BacktestConfig, BacktestResult


class PlatformAPIError(Exception):
    """Base exception for Platform API errors."""

    pass


class PermissionError(PlatformAPIError):
    """Raised when an actor lacks permission for an action."""

    pass


class NotFoundError(PlatformAPIError):
    """Raised when a requested resource is not found."""

    pass


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

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Platform API.

        Args:
            db_path: Optional path to database (uses default if not provided)
        """
        self._db = get_db(db_path)
        logger.debug("PlatformAPI initialized")

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_not_empty(self, value: str, field_name: str) -> None:
        """Validate that a string is not empty or whitespace-only."""
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    def _validate_positive(self, value: int, field_name: str) -> None:
        """Validate that an integer is positive."""
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")

    def _validate_not_future(self, d: date, field_name: str) -> None:
        """Validate that a date is not in the future."""
        if d > date.today():
            raise ValueError(f"{field_name} cannot be in the future")

    def _validate_date_range(self, start: date, end: date) -> None:
        """Validate that start date is not after end date."""
        if start > end:
            raise ValueError("start date must be <= end date")

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
        self._validate_not_future(start, "start date")
        self._validate_not_future(end, "end date")
        self._validate_date_range(start, end)

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
        self._validate_not_future(as_of_date, "as_of_date")

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

    def get_universe(self, as_of_date: date) -> List[str]:
        """
        Get the trading universe as of a specific date.

        Args:
            as_of_date: Date to get universe for

        Returns:
            List of ticker symbols in the universe
        """
        self._validate_not_future(as_of_date, "as_of_date")

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

    def adjust_prices_for_splits(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust prices for stock splits.

        Takes a prices DataFrame and applies split adjustments to create
        split-adjusted close prices. Splits are applied backward in time
        (from most recent to oldest).

        Args:
            prices: DataFrame from get_prices() with columns:
                    symbol, date, open, high, low, close, adj_close, volume

        Returns:
            DataFrame with original columns plus 'split_adjusted_close' column
        """
        if prices.empty:
            logger.warning("Empty prices DataFrame provided to adjust_prices_for_splits")
            return prices.assign(split_adjusted_close=None)

        # Make a copy to avoid modifying the original
        df = prices.copy()

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])

        # Get unique symbols from prices
        symbols = df["symbol"].unique().tolist()

        # Get date range from prices
        start_date = df["date"].min().date()
        end_date = df["date"].max().date()

        # Get all split actions for these symbols
        splits = self.get_corporate_actions(symbols, start_date, end_date)

        # Filter to only splits
        if not splits.empty:
            splits = splits[splits["action_type"] == "split"].copy()
            splits["date"] = pd.to_datetime(splits["date"])

        # Initialize split_adjusted_close with close price
        df["split_adjusted_close"] = df["close"]

        if splits.empty:
            logger.debug("No splits found, returning prices with unadjusted close")
            return df

        # Apply splits symbol by symbol
        for symbol in symbols:
            symbol_splits = splits[splits["symbol"] == symbol].sort_values("date", ascending=False)

            if symbol_splits.empty:
                continue

            # Get mask for this symbol's prices
            symbol_mask = df["symbol"] == symbol

            # Apply each split backward in time
            for _, split in symbol_splits.iterrows():
                split_date = split["date"]
                split_factor = split["factor"]

                # Adjust all prices BEFORE the split date
                date_mask = df["date"] < split_date
                mask = symbol_mask & date_mask

                df.loc[mask, "split_adjusted_close"] *= split_factor

            logger.debug(f"Applied {len(symbol_splits)} splits for {symbol}")

        logger.debug(f"Split adjustment complete for {len(symbols)} symbols")
        return df

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
    ) -> str:
        """
        Create a new research hypothesis.

        Args:
            title: Short descriptive title
            thesis: The hypothesis being tested
            prediction: Testable prediction
            falsification: Criteria for falsifying the hypothesis
            actor: Who is creating the hypothesis ('user' or 'agent:<name>')

        Returns:
            hypothesis_id: Unique identifier for the hypothesis
        """
        # Validate inputs
        self._validate_not_empty(title, "title")
        self._validate_not_empty(thesis, "thesis")
        self._validate_not_empty(prediction, "prediction")
        self._validate_not_empty(falsification, "falsification")
        self._validate_not_empty(actor, "actor")

        hypothesis_id = self._generate_hypothesis_id()

        query = """
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, created_by, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'draft')
        """

        self._db.execute(
            query, (hypothesis_id, title, thesis, prediction, falsification, actor)
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
        status: str,
        outcome: Optional[str] = None,
        actor: str = "user",
    ) -> None:
        """
        Update a hypothesis status and/or outcome.

        Args:
            hypothesis_id: ID of the hypothesis to update
            status: New status ('draft', 'testing', 'validated', 'rejected', 'deployed')
            outcome: Optional outcome description
            actor: Who is making the update

        Raises:
            NotFoundError: If hypothesis doesn't exist
        """
        # Validate inputs
        self._validate_not_empty(hypothesis_id, "hypothesis_id")
        self._validate_not_empty(status, "status")
        self._validate_not_empty(actor, "actor")

        existing = self.get_hypothesis(hypothesis_id)
        if not existing:
            raise NotFoundError(f"Hypothesis {hypothesis_id} not found")

        query = """
            UPDATE hypotheses
            SET status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP
            WHERE hypothesis_id = ?
        """

        self._db.execute(query, (status, outcome, hypothesis_id))

        self.log_event(
            event_type="hypothesis_updated",
            actor=actor,
            details={
                "hypothesis_id": hypothesis_id,
                "old_status": existing["status"],
                "new_status": status,
                "outcome": outcome,
            },
            hypothesis_id=hypothesis_id,
        )

        logger.info(f"Updated hypothesis {hypothesis_id}: status={status}")

    def list_hypotheses(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        List hypotheses, optionally filtered by status.

        Args:
            status: Optional status filter
            limit: Maximum number of results (default 100)

        Returns:
            List of hypothesis dictionaries
        """
        self._validate_positive(limit, "limit")

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
        self._validate_not_empty(hypothesis_id, "hypothesis_id")

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
            self._link_experiment_to_hypothesis(hypothesis_id, experiment_id)

        # Log to lineage
        self.log_event(
            event_type="experiment_run",
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

    def _link_experiment_to_hypothesis(
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
            Dictionary with health status
        """
        status: dict[str, Any] = {
            "api": "ok",
            "database": "unknown",
            "tables": {},
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

        return status
