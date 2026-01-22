"""
Platform API for HRP.

The Platform API is the single entry point for all operations.
All consumers (dashboard, MCP, agents) use this API - no direct database access.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.universe import UniverseManager
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

    def __init__(self, db_path: str | None = None):
        """
        Initialize the Platform API.

        Args:
            db_path: Optional path to database (uses default if not provided)
        """
        self._db = get_db(db_path)
        logger.debug("PlatformAPI initialized")

    # =========================================================================
    # Data Operations
    # =========================================================================

    def get_prices(
        self,
        symbols: list[str],
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
        if not symbols:
            raise ValueError("symbols list cannot be empty")

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
        symbols: list[str],
        features: list[str],
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

        Raises:
            ValueError: If symbols or features list is empty
            NotFoundError: If version doesn't exist
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if not features:
            raise ValueError("features list cannot be empty")

        # Validate that version exists
        version_check = self._db.fetchone(
            "SELECT COUNT(*) FROM feature_definitions WHERE version = ?",
            (version,),
        )
        if not version_check or version_check[0] == 0:
            raise NotFoundError(f"Feature version '{version}' not found")

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

    def get_universe(self, as_of_date: date) -> list[str]:
        """
        Get the trading universe as of a specific date (point-in-time).

        Uses the most recent universe snapshot on or before the given date.
        This prevents look-ahead bias in backtests by only including symbols
        that were known to be in the universe at that point in time.

        Args:
            as_of_date: Date to get universe for

        Returns:
            List of ticker symbols in the universe
        """
        manager = UniverseManager(self._db.db_path)
        return manager.get_universe_at_date(as_of_date)

    def update_universe(
        self,
        as_of_date: date | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        """
        Update the S&P 500 universe with current constituents.

        Fetches current S&P 500 members from Wikipedia, applies exclusion
        rules (financials, REITs, penny stocks), and updates the database.
        All changes are logged to the lineage table.

        Args:
            as_of_date: Date to record for this snapshot. Defaults to today.
            actor: Actor performing the update (for lineage).

        Returns:
            Dictionary with update statistics including:
            - total_constituents: Total S&P 500 members fetched
            - included: Symbols included in trading universe
            - excluded: Symbols excluded (with reasons)
            - added: New symbols added since last update
            - removed: Symbols removed since last update
        """
        manager = UniverseManager(self._db.db_path)
        return manager.update_universe(as_of_date, actor=f"api:{actor}")

    def get_universe_changes(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get universe membership changes between two dates.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with columns: date, symbol, change_type, exclusion_reason
        """
        manager = UniverseManager(self._db.db_path)
        return manager.get_universe_changes(start_date, end_date)

    def get_sector_breakdown(self, as_of_date: date) -> dict[str, int]:
        """
        Get breakdown of universe by sector.

        Args:
            as_of_date: Date to get breakdown for

        Returns:
            Dictionary mapping sector names to symbol counts
        """
        manager = UniverseManager(self._db.db_path)
        return manager.get_sector_breakdown(as_of_date)

    def compute_features(
        self,
        symbols: list[str],
        feature_names: list[str],
        dates: list[date],
        version: str | None = None,
        store: bool = True,
    ) -> pd.DataFrame | dict[str, Any]:
        """
        Compute features for symbols across dates.

        Args:
            symbols: List of ticker symbols
            feature_names: List of feature names to compute
            dates: List of dates to compute features for
            version: Optional feature version (default: latest active version)
            store: If True, store computed features in database (default: True)

        Returns:
            If store=True: Dictionary with computation stats
            If store=False: DataFrame with computed features

        Raises:
            ValueError: If symbols, feature_names, or dates are empty
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if not feature_names:
            raise ValueError("feature_names list cannot be empty")
        if not dates:
            raise ValueError("dates list cannot be empty")

        # Import here to avoid circular dependency
        from hrp.data.features.computation import FeatureComputer

        computer = FeatureComputer(db_path=self._db.db_path if hasattr(self._db, 'db_path') else None)

        if store:
            # Compute and store features
            stats = computer.compute_and_store_features(
                symbols=symbols,
                dates=dates,
                feature_names=feature_names,
                version=version,
            )
            logger.info(
                f"Computed and stored {stats['features_computed']} features "
                f"for {len(symbols)} symbols across {len(dates)} dates"
            )
            return stats
        else:
            # Compute features without storing
            features_df = computer.compute_features(
                symbols=symbols,
                dates=dates,
                feature_names=feature_names,
                version=version,
            )
            logger.info(
                f"Computed {len(feature_names)} features "
                f"for {len(symbols)} symbols across {len(dates)} dates"
            )
            return features_df

    def get_feature_versions(self, feature_name: str) -> list[str]:
        """
        Get all available versions for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            List of version strings, ordered by creation date (newest first)
        """
        if not feature_name:
            raise ValueError("feature_name cannot be empty")

        query = """
            SELECT version
            FROM feature_definitions
            WHERE feature_name = ?
            ORDER BY created_at DESC
        """

        result = self._db.fetchall(query, (feature_name,))
        versions = [row[0] for row in result]

        if not versions:
            logger.warning(f"No versions found for feature '{feature_name}'")
        else:
            logger.debug(f"Found {len(versions)} versions for feature '{feature_name}'")

        return versions

    # =========================================================================
    # Data Ingestion Operations
    # =========================================================================

    def log_ingestion_start(self, source_id: str) -> int:
        """
        Log the start of a data ingestion job.

        Args:
            source_id: Unique identifier for the ingestion source/job

        Returns:
            log_id: The ingestion_log entry ID
        """
        # Generate next log_id
        result = self._db.fetchone(
            "SELECT COALESCE(MAX(log_id), 0) + 1 FROM ingestion_log"
        )
        log_id = result[0]

        query = """
            INSERT INTO ingestion_log (log_id, source_id, started_at, status)
            VALUES (?, ?, CURRENT_TIMESTAMP, 'running')
        """

        self._db.execute(query, (log_id, source_id))

        logger.debug(f"Started ingestion log {log_id} for source {source_id}")
        return log_id

    def log_ingestion_complete(
        self,
        log_id: int,
        status: str,
        records_fetched: int = 0,
        records_inserted: int = 0,
        error_message: str | None = None,
    ) -> None:
        """
        Log the completion of a data ingestion job.

        Args:
            log_id: The ingestion_log entry ID from log_ingestion_start()
            status: Job status ('success' or 'failed')
            records_fetched: Number of records retrieved from source
            records_inserted: Number of records successfully inserted
            error_message: Optional error message if status is 'failed'
        """
        query = """
            UPDATE ingestion_log
            SET completed_at = CURRENT_TIMESTAMP,
                status = ?,
                records_fetched = ?,
                records_inserted = ?,
                error_message = ?
            WHERE log_id = ?
        """

        self._db.execute(
            query,
            (status, records_fetched, records_inserted, error_message, log_id),
        )

        logger.debug(f"Completed ingestion log {log_id} with status {status}")

    def get_ingestion_status(
        self,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get recent ingestion job status.

        Args:
            source_id: Optional filter by specific source/job ID
            limit: Maximum number of results (default 100)

        Returns:
            List of ingestion log dictionaries
        """
        query = """
            SELECT log_id, source_id, started_at, completed_at,
                   records_fetched, records_inserted, status, error_message
            FROM ingestion_log
            WHERE 1=1
        """
        params: list[Any] = []

        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        result = self._db.fetchall(query, tuple(params))

        logs = []
        for row in result:
            logs.append(
                {
                    "log_id": row[0],
                    "source_id": row[1],
                    "started_at": row[2],
                    "completed_at": row[3],
                    "records_fetched": row[4],
                    "records_inserted": row[5],
                    "status": row[6],
                    "error_message": row[7],
                }
            )

        logger.debug(f"Retrieved {len(logs)} ingestion log entries")
        return logs

    def get_last_successful_run(self, source_id: str) -> datetime | None:
        """
        Get the timestamp of the last successful ingestion for a source.

        Args:
            source_id: The source/job identifier

        Returns:
            Datetime of last successful completion, or None if never succeeded
        """
        query = """
            SELECT completed_at
            FROM ingestion_log
            WHERE source_id = ? AND status = 'success'
            ORDER BY completed_at DESC
            LIMIT 1
        """

        result = self._db.fetchone(query, (source_id,))

        if result and result[0]:
            logger.debug(f"Last successful run for {source_id}: {result[0]}")
            return result[0]

        logger.debug(f"No successful runs found for {source_id}")
        return None

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
        outcome: str | None = None,
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

    def list_hypotheses(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """
        List hypotheses, optionally filtered by status.

        Args:
            status: Optional status filter
            limit: Maximum number of results (default 100)

        Returns:
            List of hypothesis dictionaries
        """
        query = """
            SELECT hypothesis_id, title, thesis, testable_prediction,
                   falsification_criteria, status, created_at, created_by,
                   updated_at, outcome, confidence_score
            FROM hypotheses
        """
        params: list[Any] = []

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

    def get_hypothesis(self, hypothesis_id: str) -> dict | None:
        """
        Get a single hypothesis by ID.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            Hypothesis dictionary or None if not found
        """
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
        signals: pd.DataFrame | None = None,
        hypothesis_id: str | None = None,
        actor: str = "user",
        experiment_name: str = "backtests",
        feature_versions: dict[str, str] | None = None,
    ) -> str:
        """
        Run a backtest and log results to MLflow.

        Args:
            config: BacktestConfig with symbols, dates, and parameters
            signals: Optional signals DataFrame (1=long, 0=no position)
            hypothesis_id: Optional linked hypothesis
            actor: Who is running the backtest
            experiment_name: MLflow experiment name
            feature_versions: Optional dict mapping feature names to versions used

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
            feature_versions=feature_versions,
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

    def get_experiment(self, experiment_id: str) -> dict | None:
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
        experiment_ids: list[str],
        metrics: list[str] | None = None,
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

    def get_deployed_strategies(self) -> list[dict]:
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
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
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
        params: list[Any] = []

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
        details: dict | None = None,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
        parent_lineage_id: int | None = None,
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

    def get_experiments_for_hypothesis(self, hypothesis_id: str) -> list[str]:
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
