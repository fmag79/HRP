"""
DuckDB schema definitions for HRP.

Run with: python -m hrp.data.schema --init
"""

import argparse
from typing import Dict, Union

from loguru import logger

from hrp.data.db import get_db


def migrate_agent_token_usage_identity(db_path: Union[str, None] = None) -> None:
    """Recreate agent_token_usage with sequence-based auto-increment on id column.

    The old schema used `INTEGER PRIMARY KEY` which doesn't auto-increment in DuckDB.
    This migration drops and recreates the table with a sequence default.
    Safe because the table has 0 rows in all existing deployments.

    Idempotent - safe to run multiple times.
    """
    db = get_db(db_path)

    # Check if table exists
    tables = db.fetchdf("SHOW TABLES")
    if "agent_token_usage" not in tables["name"].tolist():
        logger.debug("agent_token_usage table does not exist yet, skipping migration")
        return

    # Check if id column already has a default (sequence)
    col_info = db.fetchdf(
        "SELECT column_default FROM information_schema.columns "
        "WHERE table_name = 'agent_token_usage' AND column_name = 'id'"
    )
    if not col_info.empty and col_info.iloc[0, 0] is not None:
        logger.debug("agent_token_usage.id already has a default, skipping migration")
        return

    # Recreate with sequence-based id
    logger.info("Migrating agent_token_usage: adding sequence-based auto-increment to id")
    with db.connection() as conn:
        conn.execute("DROP TABLE agent_token_usage")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_agent_token_usage START 1")
        conn.execute("""
            CREATE TABLE agent_token_usage (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_agent_token_usage'),
                agent_type VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                model VARCHAR,
                estimated_cost DECIMAL(10,6)
            )
        """)
    logger.info("agent_token_usage table recreated with sequence-based id")


def migrate_remove_cio_foreign_keys(db_path: Union[str, None] = None) -> None:
    """Remove FK constraints from CIO tables that reference hypotheses.

    DuckDB FK constraints prevent UPDATE on parent rows. The codebase already
    removed FKs from hypothesis_experiments and lineage but missed these 4 tables:
    - cio_decisions
    - model_cemetery
    - paper_portfolio
    - paper_portfolio_trades

    Idempotent - checks for FK presence before migrating.
    """
    db = get_db(db_path)

    tables = db.fetchdf("SHOW TABLES")
    table_list = tables["name"].tolist()

    # Tables to migrate: (name, has_fk) - we check if FK exists via constraint info
    target_tables = ["cio_decisions", "model_cemetery", "paper_portfolio", "paper_portfolio_trades"]
    present_tables = [t for t in target_tables if t in table_list]

    if not present_tables:
        logger.debug("CIO tables do not exist yet, skipping FK migration")
        return

    # Check if any table still has FK constraints
    has_fk = False
    for tbl in present_tables:
        try:
            constraints = db.fetchdf(
                f"SELECT constraint_type FROM information_schema.table_constraints "
                f"WHERE table_name = '{tbl}' AND constraint_type = 'FOREIGN KEY'"
            )
            if not constraints.empty:
                has_fk = True
                break
        except Exception:
            # If we can't check, assume migration needed
            has_fk = True
            break

    if not has_fk:
        logger.debug("CIO tables already have no FK constraints, skipping migration")
        return

    logger.info("Migrating CIO tables: removing FK constraints on hypotheses")

    with db.connection() as conn:
        # cio_decisions
        if "cio_decisions" in present_tables:
            conn.execute("CREATE TABLE cio_decisions_bak AS SELECT * FROM cio_decisions")
            conn.execute("DROP TABLE cio_decisions")
            conn.execute("""
                CREATE TABLE cio_decisions (
                    id INTEGER PRIMARY KEY,
                    decision_id VARCHAR UNIQUE,
                    report_date DATE,
                    hypothesis_id VARCHAR,
                    decision VARCHAR,
                    score_total DECIMAL(4, 2),
                    score_statistical DECIMAL(4, 2),
                    score_risk DECIMAL(4, 2),
                    score_economic DECIMAL(4, 2),
                    score_cost DECIMAL(4, 2),
                    rationale TEXT,
                    -- Human CIO approval workflow (reserved for future use)
                    -- These fields will be populated when human CIO approves deployment
                    approved BOOLEAN DEFAULT FALSE,
                    approved_by VARCHAR,
                    approved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT INTO cio_decisions SELECT * FROM cio_decisions_bak")
            conn.execute("DROP TABLE cio_decisions_bak")
            logger.info("  cio_decisions: FK removed")

        # model_cemetery
        if "model_cemetery" in present_tables:
            conn.execute("CREATE TABLE model_cemetery_bak AS SELECT * FROM model_cemetery")
            conn.execute("DROP TABLE model_cemetery")
            conn.execute("""
                CREATE TABLE model_cemetery (
                    id INTEGER PRIMARY KEY,
                    hypothesis_id VARCHAR UNIQUE,
                    killed_date DATE,
                    reason TEXT,
                    final_score DECIMAL(4, 2),
                    experiment_count INTEGER,
                    archived_by VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT INTO model_cemetery SELECT * FROM model_cemetery_bak")
            conn.execute("DROP TABLE model_cemetery_bak")
            logger.info("  model_cemetery: FK removed")

        # paper_portfolio
        if "paper_portfolio" in present_tables:
            conn.execute("CREATE TABLE paper_portfolio_bak AS SELECT * FROM paper_portfolio")
            conn.execute("DROP TABLE paper_portfolio")
            conn.execute("""
                CREATE TABLE paper_portfolio (
                    id INTEGER PRIMARY KEY,
                    hypothesis_id VARCHAR NOT NULL UNIQUE,
                    weight DECIMAL(5, 4),
                    entry_price DECIMAL(10, 4),
                    entry_date DATE,
                    current_price DECIMAL(10, 4),
                    unrealized_pnl DECIMAL(10, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT INTO paper_portfolio SELECT * FROM paper_portfolio_bak")
            conn.execute("DROP TABLE paper_portfolio_bak")
            logger.info("  paper_portfolio: FK removed")

        # paper_portfolio_trades
        if "paper_portfolio_trades" in present_tables:
            conn.execute(
                "CREATE TABLE paper_portfolio_trades_bak AS SELECT * FROM paper_portfolio_trades"
            )
            conn.execute("DROP TABLE paper_portfolio_trades")
            conn.execute("""
                CREATE TABLE paper_portfolio_trades (
                    id INTEGER PRIMARY KEY,
                    hypothesis_id VARCHAR,
                    action VARCHAR,
                    weight_before DECIMAL(5, 4),
                    weight_after DECIMAL(5, 4),
                    price DECIMAL(10, 4),
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "INSERT INTO paper_portfolio_trades SELECT * FROM paper_portfolio_trades_bak"
            )
            conn.execute("DROP TABLE paper_portfolio_trades_bak")
            logger.info("  paper_portfolio_trades: FK removed")

    logger.info("CIO tables FK migration complete")


def migrate_add_pipeline_stage_column(db_path: Union[str, None] = None) -> None:
    """Add pipeline_stage column to hypotheses table.

    This migration adds:
        - pipeline_stage VARCHAR: Tracks position in agent pipeline independent of status

    Pipeline stages: created, signal_discovery, alpha_review, ml_training, quality_audit,
    quant_backtest, kill_gate, stress_test, risk_review, cio_review, human_approval,
    deployed, archived.

    The migration is idempotent - safe to run multiple times.
    """
    db = get_db(db_path)

    # Check if column exists
    result = db.fetchdf("DESCRIBE hypotheses")
    columns = result["column_name"].tolist()

    if "pipeline_stage" not in columns:
        db.execute("ALTER TABLE hypotheses ADD COLUMN pipeline_stage VARCHAR DEFAULT 'created'")
        logger.info("Added pipeline_stage column to hypotheses table")

        # Infer pipeline stage from existing status for existing hypotheses
        # deployed → deployed, validated → cio_review, testing → ml_training, draft → created
        db.execute("""
            UPDATE hypotheses SET pipeline_stage = CASE
                WHEN status = 'deployed' THEN 'deployed'
                WHEN status = 'validated' THEN 'cio_review'
                WHEN status = 'testing' THEN 'ml_training'
                WHEN status = 'rejected' THEN 'archived'
                WHEN status = 'deleted' THEN 'archived'
                ELSE 'created'
            END
        """)
        logger.info("Inferred pipeline_stage from status for existing hypotheses")
    else:
        logger.debug("pipeline_stage column already exists in hypotheses table")


def migrate_add_sector_columns(db_path: Union[str, None] = None) -> None:
    """Add sector and industry columns to symbols table.

    This migration adds:
        - sector VARCHAR(50): GICS sector classification
        - industry VARCHAR(100): Industry classification
        - Index on sector column for efficient queries

    The migration is idempotent - safe to run multiple times.
    """
    db = get_db(db_path)

    # Check if columns exist
    result = db.fetchdf("DESCRIBE symbols")
    columns = result["column_name"].tolist()

    # Add sector column if not exists
    if "sector" not in columns:
        db.execute("ALTER TABLE symbols ADD COLUMN sector VARCHAR(50)")
        logger.info("Added sector column to symbols table")
    else:
        logger.debug("Sector column already exists in symbols table")

    # Add industry column if not exists
    if "industry" not in columns:
        db.execute("ALTER TABLE symbols ADD COLUMN industry VARCHAR(100)")
        logger.info("Added industry column to symbols table")
    else:
        logger.debug("Industry column already exists in symbols table")

    # Add index on sector
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_symbols_sector ON symbols(sector)")
        logger.info("Created index on symbols.sector")
    except Exception as e:
        logger.debug(f"Index may already exist: {e}")


# SQL statements for table creation
# IMPORTANT: Tables are ordered by FK dependencies - referenced tables come first
TABLES = {
    # === Base tables (no FK dependencies) ===
    "symbols": """
        CREATE TABLE IF NOT EXISTS symbols (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            exchange VARCHAR,
            asset_type VARCHAR DEFAULT 'equity',
            sector VARCHAR(50),
            industry VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK (asset_type IN ('equity', 'etf', 'index', 'other'))
        )
    """,
    "data_sources": """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id VARCHAR PRIMARY KEY,
            source_type VARCHAR,
            api_name VARCHAR,
            last_fetch TIMESTAMP,
            status VARCHAR DEFAULT 'active',
            CHECK (status IN ('active', 'inactive', 'deprecated'))
        )
    """,
    "hypotheses": """
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            thesis TEXT NOT NULL,
            testable_prediction TEXT NOT NULL,
            falsification_criteria TEXT,
            status VARCHAR DEFAULT 'draft',
            pipeline_stage VARCHAR DEFAULT 'created',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR DEFAULT 'user',
            updated_at TIMESTAMP,
            outcome TEXT,
            confidence_score DECIMAL(3,2),
            metadata JSON,
            CHECK (status IN ('draft', 'testing', 'validated', 'rejected', 'deployed', 'deleted')),
            CHECK (pipeline_stage IN ('created', 'signal_discovery', 'alpha_review', 'ml_training',
                   'quality_audit', 'quant_backtest', 'kill_gate', 'stress_test', 'risk_review',
                   'cio_review', 'human_approval', 'deployed', 'archived')),
            CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1))
        )
    """,
    "feature_definitions": """
        CREATE TABLE IF NOT EXISTS feature_definitions (
            feature_name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            computation_code TEXT NOT NULL,
            description VARCHAR,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (feature_name, version)
        )
    """,
    "test_set_evaluations": """
        CREATE TABLE IF NOT EXISTS test_set_evaluations (
            evaluation_id INTEGER PRIMARY KEY,
            hypothesis_id VARCHAR NOT NULL,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            override BOOLEAN DEFAULT FALSE,
            override_reason VARCHAR,
            metadata JSON
        )
    """,
    "hyperparameter_trials": """
        CREATE TABLE IF NOT EXISTS hyperparameter_trials (
            trial_id INTEGER PRIMARY KEY,
            hypothesis_id VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            hyperparameters JSON NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fold_index INTEGER,
            notes VARCHAR
        )
    """,
    # === Tables that depend on symbols ===
    "universe": """
        CREATE TABLE IF NOT EXISTS universe (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            in_universe BOOLEAN NOT NULL DEFAULT TRUE,
            exclusion_reason VARCHAR,
            sector VARCHAR,
            market_cap DECIMAL(18,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date),
            CHECK (market_cap IS NULL OR market_cap >= 0),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol)
        )
    """,
    "prices": """
        CREATE TABLE IF NOT EXISTS prices (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(12,4),
            high DECIMAL(12,4),
            low DECIMAL(12,4),
            close DECIMAL(12,4) NOT NULL,
            adj_close DECIMAL(12,4),
            volume BIGINT,
            source VARCHAR DEFAULT 'unknown',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date),
            CHECK (close > 0),
            CHECK (volume IS NULL OR volume >= 0),
            CHECK (high IS NULL OR low IS NULL OR high >= low),
            CHECK (open IS NULL OR open > 0),
            CHECK (adj_close IS NULL OR adj_close > 0),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol)
        )
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            feature_name VARCHAR NOT NULL,
            value DECIMAL(24,6),  -- Supports market caps up to ~$999 quadrillion
            version VARCHAR DEFAULT 'v1',
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, feature_name, version),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol)
        )
    """,
    "corporate_actions": """
        CREATE TABLE IF NOT EXISTS corporate_actions (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            action_type VARCHAR NOT NULL,
            factor DECIMAL(12,6),
            source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, action_type),
            CHECK (factor IS NULL OR factor > 0),
            CHECK (action_type IN ('split', 'dividend', 'spinoff', 'merger', 'other')),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol)
        )
    """,
    # === Tables that depend on symbols AND data_sources ===
    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR NOT NULL,
            report_date DATE NOT NULL,
            period_end DATE NOT NULL,
            metric VARCHAR NOT NULL,
            value DECIMAL(18,4),
            source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, report_date, metric),
            CHECK (period_end <= report_date),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol),
            FOREIGN KEY (source) REFERENCES data_sources(source_id)
        )
    """,
    # === Tables that depend on data_sources ===
    "ingestion_log": """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            log_id INTEGER PRIMARY KEY,
            source_id VARCHAR,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            records_fetched INTEGER DEFAULT 0,
            records_inserted INTEGER DEFAULT 0,
            status VARCHAR DEFAULT 'running',
            error_message VARCHAR,
            CHECK (records_fetched >= 0),
            CHECK (records_inserted >= 0),
            CHECK (status IN ('running', 'completed', 'failed')),
            FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
        )
    """,
    # === Tables that depend on hypotheses ===
    # NOTE: FK constraint removed due to DuckDB 1.4.3 limitation - FKs prevent UPDATE on parent table.
    # Referential integrity validated via migrate_constraints.py SQL JOIN checks.
    "hypothesis_experiments": """
        CREATE TABLE IF NOT EXISTS hypothesis_experiments (
            hypothesis_id VARCHAR NOT NULL,
            experiment_id VARCHAR NOT NULL,
            relationship VARCHAR DEFAULT 'primary',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (hypothesis_id, experiment_id)
        )
    """,
    # NOTE: FK constraints removed due to DuckDB 1.4.3 limitation - FKs prevent UPDATE on parent table.
    # Referential integrity validated via migrate_constraints.py SQL JOIN checks.
    "lineage": """
        CREATE TABLE IF NOT EXISTS lineage (
            lineage_id INTEGER PRIMARY KEY,
            event_type VARCHAR NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actor VARCHAR DEFAULT 'system',
            hypothesis_id VARCHAR,
            experiment_id VARCHAR,
            details JSON,
            parent_lineage_id INTEGER,
            CHECK (event_type IN (
                   -- Hypothesis lifecycle events
                   'hypothesis_created', 'hypothesis_updated', 'hypothesis_deleted', 'hypothesis_flagged',
                   -- Experiment events
                   'experiment_run', 'experiment_linked', 'experiment_completed',
                   'experiment_started', 'backtest_run',
                   -- Validation events
                   'validation_passed', 'validation_failed',
                   -- Deployment events
                   'deployment_approved', 'deployment_rejected', 'deployment_requested',
                   -- Generic agent events
                   'agent_run_complete', 'agent_run_start',
                   -- Signal Scientist
                   'signal_scan_complete',
                   -- Alpha Researcher
                   'alpha_researcher_review', 'alpha_researcher_complete',
                   -- ML Scientist
                   'ml_scientist_validation',
                   -- ML Quality Sentinel
                   'ml_quality_sentinel_audit',
                   -- Quant Developer
                   'quant_developer_backtest_complete',
                   -- Kill Gate Enforcer
                   'kill_gate_enforcer_complete', 'kill_gate_triggered',
                   -- Validation Analyst
                   'validation_analyst_review', 'validation_analyst_complete',
                   -- Risk Manager
                   'risk_manager_assessment', 'risk_review_complete', 'risk_veto',
                   -- CIO Agent
                   'cio_agent_decision',
                   -- Data events
                   'data_ingestion', 'data_ingested', 'feature_computed', 'universe_update',
                   -- System events
                   'system_error', 'other'))
        )
    """,
    "agent_checkpoints": """
        CREATE TABLE IF NOT EXISTS agent_checkpoints (
            agent_type VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            state_json TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            completed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (agent_type, run_id)
        )
    """,
    "agent_token_usage": """
        CREATE SEQUENCE IF NOT EXISTS seq_agent_token_usage START 1;
        CREATE TABLE IF NOT EXISTS agent_token_usage (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_agent_token_usage'),
            agent_type VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            model VARCHAR,
            estimated_cost DECIMAL(10,6)
        )
    """,
    # === ML Drift Monitoring tables ===
    "model_drift_checks": """
        CREATE TABLE IF NOT EXISTS model_drift_checks (
            check_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_version VARCHAR,
            check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            drift_type VARCHAR NOT NULL,
            feature_name VARCHAR,
            metric_value DECIMAL(10,4),
            is_drift_detected BOOLEAN,
            threshold_value DECIMAL(10,4),
            details JSON,
            CHECK (drift_type IN ('prediction', 'feature', 'concept'))
        )
    """,
    "model_performance_history": """
        CREATE TABLE IF NOT EXISTS model_performance_history (
            history_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_version VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metric_name VARCHAR NOT NULL,
            metric_value DECIMAL(10,4),
            sample_size INTEGER
        )
    """,
    "model_deployments": """
        CREATE TABLE IF NOT EXISTS model_deployments (
            deployment_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_version VARCHAR NOT NULL,
            environment VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed_by VARCHAR NOT NULL,
            deployment_config JSON,
            validation_results JSON,
            rollback_reason VARCHAR,
            CHECK (environment IN ('staging', 'production', 'shadow')),
            CHECK (status IN ('pending', 'active', 'rolled_back'))
        )
    """,
}

# Indexes for performance
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol, date, feature_name)",
    "CREATE INDEX IF NOT EXISTS idx_universe_date ON universe(date)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_hypothesis ON lineage(hypothesis_id)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_timestamp ON lineage(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_timestamp_hypothesis ON lineage(timestamp, hypothesis_id)",
    "CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)",
    "CREATE INDEX IF NOT EXISTS idx_symbols_exchange ON symbols(exchange)",
    "CREATE INDEX IF NOT EXISTS idx_hp_trials_hypothesis ON hyperparameter_trials(hypothesis_id)",
    # ML Drift Monitoring indexes
    "CREATE INDEX IF NOT EXISTS idx_drift_checks_model_timestamp ON model_drift_checks(model_name, check_timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_drift_checks_drift_type ON model_drift_checks(drift_type, is_drift_detected)",
    "CREATE INDEX IF NOT EXISTS idx_perf_history_model_timestamp ON model_performance_history(model_name, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_deployments_model_env ON model_deployments(model_name, environment, status)",
]


def create_tables(db_path: Union[str, None] = None) -> None:
    """Create all tables in the database."""
    db = get_db(db_path)

    with db.connection() as conn:
        for table_name, create_sql in TABLES.items():
            logger.info(f"Creating table: {table_name}")
            conn.execute(create_sql)

        for index_sql in INDEXES:
            logger.debug(f"Creating index: {index_sql[:50]}...")
            conn.execute(index_sql)

    logger.info(f"Schema initialized with {len(TABLES)} tables and {len(INDEXES)} indexes")

    # Run migrations (idempotent)
    migrate_agent_token_usage_identity(db_path)
    migrate_add_sector_columns(db_path)
    migrate_add_pipeline_stage_column(db_path)
    migrate_remove_cio_foreign_keys(db_path)


def drop_all_tables(db_path: Union[str, None] = None) -> None:
    """Drop all tables (use with caution!).

    Tables are dropped in reverse order to respect FK dependencies.
    """
    db = get_db(db_path)

    # Drop in reverse order due to FK constraints
    table_names = list(TABLES.keys())
    with db.connection() as conn:
        for table_name in reversed(table_names):
            logger.warning(f"Dropping table: {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    logger.warning("All tables dropped")


def get_table_counts(db_path: Union[str, None] = None) -> Dict[str, int]:
    """Get row counts for all tables."""
    db = get_db(db_path)
    counts = {}

    with db.connection() as conn:
        for table_name in TABLES.keys():
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                counts[table_name] = result[0] if result else 0
            except Exception:
                counts[table_name] = -1  # Table doesn't exist

    return counts


def verify_schema(db_path: Union[str, None] = None) -> bool:
    """Verify that all tables exist."""
    db = get_db(db_path)

    with db.connection() as conn:
        for table_name in TABLES.keys():
            try:
                conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            except Exception as e:
                logger.error(f"Table {table_name} missing or invalid: {e}")
                return False

    logger.info("Schema verification passed")
    return True


def main() -> None:
    """CLI entry point for schema management."""
    parser = argparse.ArgumentParser(description="HRP Database Schema Management")
    parser.add_argument("--init", action="store_true", help="Initialize the database schema")
    parser.add_argument("--drop", action="store_true", help="Drop all tables (DANGEROUS)")
    parser.add_argument("--verify", action="store_true", help="Verify schema exists")
    parser.add_argument("--counts", action="store_true", help="Show table row counts")
    parser.add_argument("--db", type=str, help="Database path (optional)")

    args = parser.parse_args()

    if args.init:
        create_tables(args.db)
    elif args.drop:
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            drop_all_tables(args.db)
        else:
            print("Aborted")
    elif args.verify:
        if verify_schema(args.db):
            print("Schema OK")
        else:
            print("Schema verification failed")
            exit(1)
    elif args.counts:
        counts = get_table_counts(args.db)
        print("\nTable Row Counts:")
        print("-" * 30)
        for table, count in counts.items():
            print(f"{table:25} {count:>10}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
