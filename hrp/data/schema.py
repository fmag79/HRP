"""
DuckDB schema definitions for HRP.

This module defines the complete database schema including:
- Table structures with primary keys and foreign keys
- CHECK constraints for data integrity
- Performance indexes for common query patterns

## Usage

Initialize a new database:
    python -m hrp.data.schema --init

Verify existing schema:
    python -m hrp.data.schema --verify

## Migration Instructions

For EXISTING databases (upgrade to add constraints and indexes):

1. **Backup your database first:**
   cp ~/hrp-data/hrp.duckdb ~/hrp-data/hrp.duckdb.backup

2. **Run schema initialization (safe - uses IF NOT EXISTS):**
   python -m hrp.data.schema --init

   This will:
   - Add any missing tables
   - Add new indexes (idempotent)
   - NOT modify existing data

3. **Verify the migration:**
   python -m hrp.data.schema --verify
   python -m hrp.data.schema --counts

## Schema Features

**Integrity Constraints:**
- Primary keys on all tables
- Foreign keys between related tables (fundamentals, ingestion_log)
- CHECK constraints for data validation (prices, volumes, status enums)
- NOT NULL constraints on critical fields

**Performance Indexes:**
- Symbol-date composite indexes for time-series queries
- Status and type indexes for filtering
- Hypothesis and experiment tracking indexes

**Foreign Key Notes:**
Some FK constraints are intentionally omitted due to DuckDB 1.4.3 limitations
where FKs prevent UPDATE operations on parent tables. Application-level
integrity checks handle these cases (see hypothesis_experiments, lineage).
"""

from __future__ import annotations

import argparse

from loguru import logger

from hrp.data.db import get_db


# Sequences for auto-incrementing IDs
SEQUENCES = [
    "CREATE SEQUENCE IF NOT EXISTS ingestion_log_seq START 1",
    "CREATE SEQUENCE IF NOT EXISTS lineage_seq START 1",
    "CREATE SEQUENCE IF NOT EXISTS quality_reports_seq START 1",
]

# SQL statements for table creation
# NOTE: Tables are ordered to satisfy foreign key dependencies.
# Tables without foreign keys come first, then tables with foreign keys.
TABLES = {
    # Tables without foreign key dependencies
    "universe": """
        CREATE TABLE IF NOT EXISTS universe (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            in_universe BOOLEAN NOT NULL DEFAULT TRUE,
            exclusion_reason VARCHAR,
            sector VARCHAR,
            market_cap DECIMAL(18,2),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date),
            CHECK (market_cap IS NULL OR market_cap >= 0)
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
            source VARCHAR NOT NULL DEFAULT 'unknown',
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date),
            CHECK (close > 0),
            CHECK (volume IS NULL OR volume >= 0),
            CHECK (high IS NULL OR low IS NULL OR high >= low)
        )
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            feature_name VARCHAR NOT NULL,
            value DECIMAL(18,6),
            version VARCHAR NOT NULL DEFAULT 'v1',
            computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, feature_name, version)
        )
    """,
    "corporate_actions": """
        CREATE TABLE IF NOT EXISTS corporate_actions (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            action_type VARCHAR NOT NULL,
            factor DECIMAL(12,6),
            source VARCHAR,
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, action_type),
            CHECK (factor IS NULL OR factor > 0)
        )
    """,
    "data_sources": """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id VARCHAR PRIMARY KEY,
            source_type VARCHAR NOT NULL,
            api_name VARCHAR,
            last_fetch TIMESTAMP,
            status VARCHAR NOT NULL DEFAULT 'active',
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
            status VARCHAR NOT NULL DEFAULT 'draft',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR NOT NULL DEFAULT 'user',
            updated_at TIMESTAMP,
            outcome TEXT,
            confidence_score DECIMAL(3,2),
            CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
            CHECK (status IN ('draft', 'testing', 'validated', 'rejected', 'deployed', 'deleted'))
        )
    """,
    # Tables with foreign key dependencies (must be after referenced tables)
    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR NOT NULL,
            report_date DATE NOT NULL,
            period_end DATE NOT NULL,
            metric VARCHAR NOT NULL,
            value DECIMAL(18,4),
            source VARCHAR REFERENCES data_sources(source_id),
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, report_date, metric)
        )
    """,
    "ingestion_log": """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            log_id INTEGER PRIMARY KEY DEFAULT nextval('ingestion_log_seq'),
            source_id VARCHAR NOT NULL REFERENCES data_sources(source_id),
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            records_fetched INTEGER NOT NULL DEFAULT 0,
            records_inserted INTEGER NOT NULL DEFAULT 0,
            status VARCHAR NOT NULL DEFAULT 'running',
            error_message VARCHAR,
            CHECK (records_fetched >= 0),
            CHECK (records_inserted >= 0),
            CHECK (status IN ('running', 'completed', 'failed'))
        )
    """,
    "hypothesis_experiments": """
        CREATE TABLE IF NOT EXISTS hypothesis_experiments (
            hypothesis_id VARCHAR NOT NULL,
            experiment_id VARCHAR NOT NULL,
            relationship VARCHAR NOT NULL DEFAULT 'primary',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (hypothesis_id, experiment_id)
            -- Note: FK constraint on hypothesis_id removed due to DuckDB 1.4.3 limitation
            -- where FKs prevent UPDATE operations on parent table even when not modifying PK
        )
    """,
    "lineage": """
        CREATE TABLE IF NOT EXISTS lineage (
            lineage_id INTEGER PRIMARY KEY DEFAULT nextval('lineage_seq'),
            event_type VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            actor VARCHAR NOT NULL DEFAULT 'system',
            hypothesis_id VARCHAR,
            experiment_id VARCHAR,
            details JSON,
            parent_lineage_id INTEGER
            -- Note: FK constraints removed due to DuckDB 1.4.3 limitation
            -- where FKs prevent UPDATE operations on parent tables even when not modifying PK
            -- Application-level integrity checks must be used instead
        )
    """,
    "feature_definitions": """
        CREATE TABLE IF NOT EXISTS feature_definitions (
            feature_name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            computation_code TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            PRIMARY KEY (feature_name, version)
        )
    """,
    "test_set_evaluations": """
        CREATE TABLE IF NOT EXISTS test_set_evaluations (
            id INTEGER PRIMARY KEY DEFAULT nextval('ingestion_log_seq'),
            hypothesis_id VARCHAR NOT NULL,
            evaluated_at TIMESTAMP NOT NULL,
            override BOOLEAN DEFAULT FALSE,
            override_reason TEXT,
            metadata TEXT
        )
    """,
}

# Indexes for performance
INDEXES = [
    # Existing indexes
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol, date, feature_name)",
    "CREATE INDEX IF NOT EXISTS idx_universe_date ON universe(date)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_hypothesis ON lineage(hypothesis_id)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_timestamp ON lineage(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)",
    "CREATE INDEX IF NOT EXISTS idx_feature_definitions_name_version ON feature_definitions(feature_name, version)",
    # New composite indexes for query optimization
    "CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date ON fundamentals(symbol, report_date)",
    "CREATE INDEX IF NOT EXISTS idx_corporate_actions_symbol_date ON corporate_actions(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_universe_symbol_date ON universe(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_universe_in_universe ON universe(in_universe, date)",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_log_source ON ingestion_log(source_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_log_status ON ingestion_log(status, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_hypothesis_experiments_exp ON hypothesis_experiments(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_experiment ON lineage(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_lineage_event_type ON lineage(event_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_test_evaluations_hypothesis ON test_set_evaluations(hypothesis_id)",
]


def create_tables(db_path: str | None = None) -> None:
    """Create all tables in the database."""
    db = get_db(db_path)

    with db.connection() as conn:
        # Create sequences first (needed for auto-increment PKs)
        for seq_sql in SEQUENCES:
            logger.debug(f"Creating sequence: {seq_sql}")
            conn.execute(seq_sql)

        for table_name, create_sql in TABLES.items():
            logger.info(f"Creating table: {table_name}")
            conn.execute(create_sql)

        for index_sql in INDEXES:
            logger.debug(f"Creating index: {index_sql[:50]}...")
            conn.execute(index_sql)

    logger.info(f"Schema initialized with {len(SEQUENCES)} sequences, {len(TABLES)} tables and {len(INDEXES)} indexes")


def drop_all_tables(db_path: str | None = None) -> None:
    """Drop all tables (use with caution!).

    Tables are dropped in reverse order to respect foreign key dependencies.
    """
    db = get_db(db_path)

    with db.connection() as conn:
        # Drop in reverse order to handle FK constraints
        for table_name in reversed(list(TABLES.keys())):
            logger.warning(f"Dropping table: {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    logger.warning("All tables dropped")


def get_table_counts(db_path: str | None = None) -> dict[str, int]:
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


def verify_schema(db_path: str | None = None) -> bool:
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
