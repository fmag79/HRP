"""
Database migration script for adding constraints and indexes to existing HRP databases.

This script migrates databases created before constraint enforcement was added.
It adds indexes and validates that existing data meets new constraint requirements.

Usage:
    python -m hrp.data.migrate_constraints --migrate [--dry-run] [--db PATH]
    python -m hrp.data.migrate_constraints --validate [--db PATH]
    python -m hrp.data.migrate_constraints --status [--db PATH]
"""

import argparse
from typing import Any

from loguru import logger

from hrp.data.db import get_db
from hrp.data.schema import INDEXES, TABLES


def get_existing_indexes(db_path: str | None = None) -> set[str]:
    """Get list of existing indexes in the database."""
    db = get_db(db_path)
    existing = set()

    with db.connection() as conn:
        try:
            # Query DuckDB's information schema for indexes
            result = conn.execute("""
                SELECT index_name
                FROM duckdb_indexes()
            """).fetchall()
            existing = {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Could not query existing indexes: {e}")

    return existing


def get_existing_tables(db_path: str | None = None) -> set[str]:
    """Get list of existing tables in the database."""
    db = get_db(db_path)
    existing = set()

    with db.connection() as conn:
        try:
            result = conn.execute("SHOW TABLES").fetchall()
            existing = {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Could not query existing tables: {e}")

    return existing


def validate_data_constraints(db_path: str | None = None) -> dict[str, list[str]]:
    """
    Validate that existing data meets new constraint requirements.

    Returns a dict mapping table names to lists of constraint violations.
    Empty lists indicate no violations.
    """
    db = get_db(db_path)
    violations: dict[str, list[str]] = {}

    with db.connection() as conn:
        # Validate universe constraints
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM universe
                WHERE market_cap IS NOT NULL AND market_cap < 0
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("universe", []).append(
                    f"{result[0]} rows with negative market_cap"
                )
        except Exception:
            pass  # Table might not exist

        # Validate prices constraints
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM prices WHERE close <= 0
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("prices", []).append(
                    f"{result[0]} rows with close <= 0"
                )

            result = conn.execute("""
                SELECT COUNT(*) FROM prices
                WHERE volume IS NOT NULL AND volume < 0
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("prices", []).append(
                    f"{result[0]} rows with negative volume"
                )

            result = conn.execute("""
                SELECT COUNT(*) FROM prices
                WHERE high IS NOT NULL AND low IS NOT NULL AND high < low
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("prices", []).append(
                    f"{result[0]} rows with high < low"
                )
        except Exception:
            pass

        # Validate corporate_actions constraints
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM corporate_actions
                WHERE factor IS NOT NULL AND factor <= 0
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("corporate_actions", []).append(
                    f"{result[0]} rows with non-positive factor"
                )
        except Exception:
            pass

        # Validate data_sources status enum
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM data_sources
                WHERE status NOT IN ('active', 'inactive', 'deprecated')
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("data_sources", []).append(
                    f"{result[0]} rows with invalid status"
                )
        except Exception:
            pass

        # Validate hypotheses constraints
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM hypotheses
                WHERE confidence_score IS NOT NULL
                AND (confidence_score < 0 OR confidence_score > 1)
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("hypotheses", []).append(
                    f"{result[0]} rows with confidence_score out of range [0,1]"
                )

            result = conn.execute("""
                SELECT COUNT(*) FROM hypotheses
                WHERE status NOT IN ('draft', 'testing', 'validated', 'rejected', 'deployed', 'deleted')
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("hypotheses", []).append(
                    f"{result[0]} rows with invalid status"
                )
        except Exception:
            pass

        # Validate ingestion_log constraints
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM ingestion_log
                WHERE records_fetched < 0 OR records_inserted < 0
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("ingestion_log", []).append(
                    f"{result[0]} rows with negative record counts"
                )

            result = conn.execute("""
                SELECT COUNT(*) FROM ingestion_log
                WHERE status NOT IN ('running', 'completed', 'failed')
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("ingestion_log", []).append(
                    f"{result[0]} rows with invalid status"
                )
        except Exception:
            pass

        # Validate foreign key relationships (without enforcing FK constraints)
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM fundamentals f
                LEFT JOIN data_sources ds ON f.source = ds.source_id
                WHERE f.source IS NOT NULL AND ds.source_id IS NULL
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("fundamentals", []).append(
                    f"{result[0]} rows reference non-existent data_sources"
                )
        except Exception:
            pass

        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM ingestion_log il
                LEFT JOIN data_sources ds ON il.source_id = ds.source_id
                WHERE ds.source_id IS NULL
            """).fetchone()
            if result and result[0] > 0:
                violations.setdefault("ingestion_log", []).append(
                    f"{result[0]} rows reference non-existent data_sources"
                )
        except Exception:
            pass

    return violations


def add_indexes(db_path: str | None = None, dry_run: bool = False) -> dict[str, int]:
    """
    Add new indexes to the database.

    Returns a dict with counts of indexes added, skipped, and failed.
    """
    db = get_db(db_path)
    existing_indexes = get_existing_indexes(db_path)
    existing_tables = get_existing_tables(db_path)

    stats = {"added": 0, "skipped": 0, "failed": 0}

    with db.connection() as conn:
        for index_sql in INDEXES:
            # Extract index name from SQL
            index_name = None
            if "IF NOT EXISTS" in index_sql:
                parts = index_sql.split("IF NOT EXISTS")
                if len(parts) > 1:
                    index_name = parts[1].split()[0].strip()

            if index_name and index_name in existing_indexes:
                logger.debug(f"Index already exists: {index_name}")
                stats["skipped"] += 1
                continue

            # Check if the table exists before creating index
            table_name = None
            if " ON " in index_sql:
                table_part = index_sql.split(" ON ")[1]
                table_name = table_part.split("(")[0].strip()

            if table_name and table_name not in existing_tables:
                logger.debug(f"Skipping index for non-existent table: {table_name}")
                stats["skipped"] += 1
                continue

            if dry_run:
                logger.info(f"[DRY RUN] Would create: {index_sql[:80]}...")
                stats["added"] += 1
            else:
                try:
                    conn.execute(index_sql)
                    logger.info(f"Created index: {index_name or 'unnamed'}")
                    stats["added"] += 1
                except Exception as e:
                    logger.error(f"Failed to create index: {e}")
                    logger.debug(f"SQL: {index_sql}")
                    stats["failed"] += 1

    return stats


def migration_status(db_path: str | None = None) -> dict[str, Any]:
    """
    Check the current migration status of the database.

    Returns a dict with information about tables, indexes, and constraint compliance.
    """
    existing_tables = get_existing_tables(db_path)
    existing_indexes = get_existing_indexes(db_path)
    violations = validate_data_constraints(db_path)

    # Count expected indexes per table
    expected_indexes = set()
    for index_sql in INDEXES:
        if "IF NOT EXISTS" in index_sql:
            parts = index_sql.split("IF NOT EXISTS")
            if len(parts) > 1:
                index_name = parts[1].split()[0].strip()
                expected_indexes.add(index_name)

    missing_indexes = expected_indexes - existing_indexes

    status = {
        "tables": {
            "total": len(TABLES),
            "existing": len(existing_tables),
            "missing": [t for t in TABLES.keys() if t not in existing_tables],
        },
        "indexes": {
            "expected": len(expected_indexes),
            "existing": len(existing_indexes & expected_indexes),
            "missing": list(missing_indexes),
        },
        "constraints": {
            "violations": violations,
            "compliant": len(violations) == 0,
        },
    }

    return status


def migrate(db_path: str | None = None, dry_run: bool = False) -> bool:
    """
    Migrate an existing database to add indexes and validate constraints.

    Returns True if migration successful (or would be in dry-run mode).
    """
    logger.info(f"Starting database migration{' (DRY RUN)' if dry_run else ''}")
    logger.info(f"Database: {db_path or 'default'}")

    # Check current status
    status = migration_status(db_path)

    if not status["tables"]["existing"]:
        logger.error("No tables found in database. Run schema initialization first.")
        return False

    # Validate data before migration
    logger.info("Validating existing data against new constraints...")
    violations = status["constraints"]["violations"]

    if violations:
        logger.error("Data validation failed! The following issues must be fixed:")
        for table, issues in violations.items():
            logger.error(f"  {table}:")
            for issue in issues:
                logger.error(f"    - {issue}")
        logger.error("\nMigration cannot proceed with constraint violations.")
        logger.error("Fix the data issues and re-run migration.")
        return False

    logger.info("✓ Data validation passed - all data meets new constraints")

    # Add indexes
    logger.info(f"\nAdding {len(status['indexes']['missing'])} missing indexes...")
    index_stats = add_indexes(db_path, dry_run)

    logger.info(f"\nIndex migration summary:")
    logger.info(f"  Added: {index_stats['added']}")
    logger.info(f"  Skipped: {index_stats['skipped']}")
    logger.info(f"  Failed: {index_stats['failed']}")

    if index_stats["failed"] > 0:
        logger.error("Some indexes failed to create. Check logs above.")
        return False

    # Final status check
    if not dry_run:
        final_status = migration_status(db_path)
        remaining = len(final_status["indexes"]["missing"])

        if remaining > 0:
            logger.warning(f"{remaining} indexes still missing after migration")
            return False

        logger.info("\n✓ Migration completed successfully!")
        logger.info("\nNote: New constraints are enforced for fresh table creation.")
        logger.info("Existing tables have been validated but constraints are not")
        logger.info("retroactively added (DuckDB limitation). Data meets all constraint")
        logger.info("requirements and will be validated on future inserts/updates.")
    else:
        logger.info("\n✓ Dry-run completed - no changes made")
        logger.info(f"Would add {index_stats['added']} indexes")

    return True


def main() -> None:
    """CLI entry point for database migration."""
    parser = argparse.ArgumentParser(
        description="Migrate HRP database to add constraints and indexes"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run migration to add indexes and validate constraints",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate data against new constraints"
    )
    parser.add_argument(
        "--status", action="store_true", help="Check current migration status"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--db", type=str, help="Database path (optional)")

    args = parser.parse_args()

    if args.migrate:
        success = migrate(args.db, args.dry_run)
        exit(0 if success else 1)
    elif args.validate:
        violations = validate_data_constraints(args.db)
        if violations:
            print("\n❌ Constraint Violations Found:\n")
            for table, issues in violations.items():
                print(f"  {table}:")
                for issue in issues:
                    print(f"    - {issue}")
            exit(1)
        else:
            print("\n✓ All data meets constraint requirements")
            exit(0)
    elif args.status:
        status = migration_status(args.db)
        print("\n=== Migration Status ===\n")
        print(f"Tables: {status['tables']['existing']}/{status['tables']['total']}")
        if status["tables"]["missing"]:
            print(f"  Missing: {', '.join(status['tables']['missing'])}")
        print(f"\nIndexes: {status['indexes']['existing']}/{status['indexes']['expected']}")
        if status["indexes"]["missing"]:
            print(f"  Missing: {', '.join(status['indexes']['missing'])}")
        print(f"\nConstraints: {'✓ Compliant' if status['constraints']['compliant'] else '❌ Violations'}")
        if status["constraints"]["violations"]:
            for table, issues in status["constraints"]["violations"].items():
                print(f"  {table}:")
                for issue in issues:
                    print(f"    - {issue}")
        exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
