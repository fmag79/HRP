"""
DuckDB schema definitions for HRP.

Run with: python -m hrp.data.schema --init
"""

import argparse
from typing import Dict, Union

from loguru import logger

from hrp.data.db import get_db


# SQL statements for table creation
TABLES = {
    "universe": """
        CREATE TABLE IF NOT EXISTS universe (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            in_universe BOOLEAN DEFAULT TRUE,
            exclusion_reason VARCHAR,
            sector VARCHAR,
            market_cap DECIMAL(18,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
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
            PRIMARY KEY (symbol, date)
        )
    """,
    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR NOT NULL,
            report_date DATE NOT NULL,
            period_end DATE NOT NULL,
            metric VARCHAR NOT NULL,
            value DECIMAL(18,4),
            source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, report_date, metric)
        )
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            feature_name VARCHAR NOT NULL,
            value DECIMAL(18,6),
            version VARCHAR DEFAULT 'v1',
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, action_type)
        )
    """,
    "data_sources": """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id VARCHAR PRIMARY KEY,
            source_type VARCHAR,
            api_name VARCHAR,
            last_fetch TIMESTAMP,
            status VARCHAR DEFAULT 'active'
        )
    """,
    "ingestion_log": """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            log_id INTEGER PRIMARY KEY,
            source_id VARCHAR,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            records_fetched INTEGER DEFAULT 0,
            records_inserted INTEGER DEFAULT 0,
            status VARCHAR DEFAULT 'running',
            error_message VARCHAR
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR DEFAULT 'user',
            updated_at TIMESTAMP,
            outcome TEXT,
            confidence_score DECIMAL(3,2)
        )
    """,
    "hypothesis_experiments": """
        CREATE TABLE IF NOT EXISTS hypothesis_experiments (
            hypothesis_id VARCHAR NOT NULL,
            experiment_id VARCHAR NOT NULL,
            relationship VARCHAR DEFAULT 'primary',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (hypothesis_id, experiment_id)
        )
    """,
    "lineage": """
        CREATE TABLE IF NOT EXISTS lineage (
            lineage_id INTEGER PRIMARY KEY,
            event_type VARCHAR NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actor VARCHAR DEFAULT 'system',
            hypothesis_id VARCHAR,
            experiment_id VARCHAR,
            details JSON,
            parent_lineage_id INTEGER
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
    "CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)",
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


def drop_all_tables(db_path: Union[str, None] = None) -> None:
    """Drop all tables (use with caution!)."""
    db = get_db(db_path)

    with db.connection() as conn:
        for table_name in TABLES.keys():
            logger.warning(f"Dropping table: {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

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
