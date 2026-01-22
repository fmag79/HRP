"""
Tests for database migration script.

Tests that the migration script correctly:
- Adds missing indexes to existing databases
- Validates data constraints before migration
- Reports migration status accurately
- Handles dry-run mode correctly
- Handles databases with no violations
- Handles databases with constraint violations

These tests ensure the migration script can safely upgrade existing HRP databases.
"""

import os

import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.migrate_constraints import (
    add_indexes,
    get_existing_indexes,
    get_existing_tables,
    migrate,
    migration_status,
    validate_data_constraints,
)


@pytest.fixture
def old_db(temp_db: str):
    """Create a database with old schema (no constraints) to simulate pre-migration state."""
    # Reset DatabaseManager to ensure fresh state
    DatabaseManager.reset()

    db = get_db(temp_db)

    # Create tables WITHOUT constraints (simulating old schema)
    with db.connection() as conn:
        # Old prices table without CHECK constraints
        conn.execute("""
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
                PRIMARY KEY (symbol, date)
            )
        """)

        # Old hypotheses table without CHECK constraints
        conn.execute("""
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
                confidence_score DECIMAL(3,2)
            )
        """)

        # Old universe table without CHECK constraints
        conn.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                in_universe BOOLEAN NOT NULL DEFAULT TRUE,
                exclusion_reason VARCHAR,
                sector VARCHAR,
                market_cap DECIMAL(18,2),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Other tables needed by migration script
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                feature_name VARCHAR NOT NULL,
                value DECIMAL(18,6),
                version VARCHAR NOT NULL DEFAULT 'v1',
                computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, feature_name, version)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                action_type VARCHAR NOT NULL,
                factor DECIMAL(12,6),
                source VARCHAR,
                ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, action_type)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                source_id VARCHAR PRIMARY KEY,
                source_type VARCHAR NOT NULL,
                api_name VARCHAR,
                last_fetch TIMESTAMP,
                status VARCHAR NOT NULL DEFAULT 'active'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol VARCHAR NOT NULL,
                report_date DATE NOT NULL,
                period_end DATE NOT NULL,
                metric VARCHAR NOT NULL,
                value DECIMAL(18,4),
                source VARCHAR,
                ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, report_date, metric)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                log_id BIGINT PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                records_fetched INTEGER NOT NULL DEFAULT 0,
                records_inserted INTEGER NOT NULL DEFAULT 0,
                status VARCHAR NOT NULL DEFAULT 'running',
                error_message TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_experiments (
                hypothesis_id VARCHAR NOT NULL,
                experiment_id VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hypothesis_id, experiment_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS lineage (
                lineage_id BIGINT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR NOT NULL,
                actor VARCHAR NOT NULL DEFAULT 'system',
                hypothesis_id VARCHAR,
                experiment_id VARCHAR,
                parent_lineage_id BIGINT,
                metadata JSON
            )
        """)

    yield temp_db

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]


class TestMigrationStatus:
    """Test migration status checking."""

    def test_status_on_fresh_database(self, test_db):
        """Test migration status on a fresh database with all indexes."""
        status = migration_status(test_db)

        # Fresh database should have all tables
        assert status["tables"]["total"] > 0
        assert status["tables"]["existing"] == status["tables"]["total"]
        assert len(status["tables"]["missing"]) == 0

        # Fresh database should have all indexes
        assert status["indexes"]["expected"] > 0
        assert status["indexes"]["existing"] == status["indexes"]["expected"]
        assert len(status["indexes"]["missing"]) == 0

        # Fresh database should have no violations
        assert status["constraints"]["compliant"] is True
        assert len(status["constraints"]["violations"]) == 0

    def test_status_on_database_missing_indexes(self, test_db):
        """Test migration status on database with missing indexes."""
        db = get_db(test_db)

        # Drop some indexes
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")
            conn.execute("DROP INDEX IF EXISTS idx_hypotheses_status")

        status = migration_status(test_db)

        # Should detect missing indexes
        assert len(status["indexes"]["missing"]) >= 2
        assert "idx_prices_symbol_date" in status["indexes"]["missing"]
        assert "idx_hypotheses_status" in status["indexes"]["missing"]

        # Should still have no constraint violations
        assert status["constraints"]["compliant"] is True

    def test_status_on_database_with_violations(self, old_db):
        """Test migration status detects constraint violations."""
        db = get_db(old_db)

        # Insert data that violates constraints (old_db has no constraints enforced)
        with db.connection() as conn:
            # Negative close price
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('BAD', '2020-01-01', -10.0)
            """
            )

            # Invalid hypothesis status
            conn.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-BAD', 'Test', 'Thesis', 'Pred', 'invalid_status')
            """
            )

        status = migration_status(old_db)

        # Should detect violations
        assert status["constraints"]["compliant"] is False
        assert len(status["constraints"]["violations"]) > 0
        assert "prices" in status["constraints"]["violations"]
        assert "hypotheses" in status["constraints"]["violations"]


class TestDataValidation:
    """Test data constraint validation."""

    def test_validation_on_clean_data(self, populated_db):
        """Test validation passes on clean sample data."""
        violations = validate_data_constraints(populated_db)

        # Sample data should have no violations
        assert len(violations) == 0

    def test_validation_detects_negative_prices(self, old_db):
        """Test validation detects negative close prices."""
        db = get_db(old_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('BAD', '2020-01-01', -10.0)
            """
            )

        violations = validate_data_constraints(old_db)

        assert "prices" in violations
        assert any("close <= 0" in v for v in violations["prices"])

    def test_validation_detects_negative_volume(self, old_db):
        """Test validation detects negative volume."""
        db = get_db(old_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, volume)
                VALUES ('BAD', '2020-01-01', 100.0, -1000)
            """
            )

        violations = validate_data_constraints(old_db)

        assert "prices" in violations
        assert any("negative volume" in v for v in violations["prices"])

    def test_validation_detects_high_less_than_low(self, old_db):
        """Test validation detects high < low."""
        db = get_db(old_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close, high, low)
                VALUES ('BAD', '2020-01-01', 100.0, 95.0, 105.0)
            """
            )

        violations = validate_data_constraints(old_db)

        assert "prices" in violations
        assert any("high < low" in v for v in violations["prices"])

    def test_validation_detects_invalid_hypothesis_status(self, old_db):
        """Test validation detects invalid hypothesis status."""
        db = get_db(old_db)

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-BAD', 'Test', 'Thesis', 'Pred', 'invalid')
            """
            )

        violations = validate_data_constraints(old_db)

        assert "hypotheses" in violations
        assert any("invalid status" in v for v in violations["hypotheses"])

    def test_validation_detects_invalid_confidence_score(self, old_db):
        """Test validation detects confidence_score out of range."""
        db = get_db(old_db)

        with db.connection() as conn:
            # confidence_score > 1
            conn.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
                VALUES ('HYP-HIGH', 'Test', 'Thesis', 'Pred', 1.5)
            """
            )

            # confidence_score < 0
            conn.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
                VALUES ('HYP-LOW', 'Test', 'Thesis', 'Pred', -0.5)
            """
            )

        violations = validate_data_constraints(old_db)

        assert "hypotheses" in violations
        assert any("confidence_score out of range" in v for v in violations["hypotheses"])


class TestIndexManagement:
    """Test index creation and management."""

    def test_add_indexes_to_fresh_database(self, test_db):
        """Test adding indexes to a database that already has them."""
        stats = add_indexes(test_db, dry_run=False)

        # Fresh database already has all indexes
        assert stats["added"] == 0
        assert stats["skipped"] > 0
        assert stats["failed"] == 0

    def test_add_indexes_after_dropping_some(self, test_db):
        """Test adding indexes after some have been dropped."""
        db = get_db(test_db)

        # Drop a few indexes
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")
            conn.execute("DROP INDEX IF EXISTS idx_hypotheses_status")

        stats = add_indexes(test_db, dry_run=False)

        # Should re-add the dropped indexes
        assert stats["added"] >= 2
        assert stats["failed"] == 0

        # Verify indexes exist now
        existing = get_existing_indexes(test_db)
        assert "idx_prices_symbol_date" in existing
        assert "idx_hypotheses_status" in existing

    def test_add_indexes_dry_run(self, test_db):
        """Test dry-run mode doesn't actually add indexes."""
        db = get_db(test_db)

        # Drop an index
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")

        before = get_existing_indexes(test_db)
        assert "idx_prices_symbol_date" not in before

        # Run in dry-run mode
        stats = add_indexes(test_db, dry_run=True)

        # Should report it would add indexes
        assert stats["added"] >= 1

        # But index should still be missing
        after = get_existing_indexes(test_db)
        assert "idx_prices_symbol_date" not in after

    def test_get_existing_tables(self, test_db):
        """Test getting list of existing tables."""
        tables = get_existing_tables(test_db)

        # Should include core tables
        assert "prices" in tables
        assert "hypotheses" in tables
        assert "universe" in tables
        assert "features" in tables

    def test_get_existing_indexes(self, test_db):
        """Test getting list of existing indexes."""
        indexes = get_existing_indexes(test_db)

        # Should include expected indexes
        assert "idx_prices_symbol_date" in indexes
        assert "idx_hypotheses_status" in indexes
        assert "idx_lineage_hypothesis" in indexes


class TestFullMigration:
    """Test complete migration workflow."""

    def test_migrate_fresh_database(self, test_db):
        """Test migrating a fresh database (should be no-op)."""
        result = migrate(test_db, dry_run=False)

        # Migration should succeed
        assert result is True

        # Status should show everything is good
        status = migration_status(test_db)
        assert status["constraints"]["compliant"] is True
        assert len(status["indexes"]["missing"]) == 0

    def test_migrate_database_missing_indexes(self, test_db):
        """Test migrating database with missing indexes."""
        db = get_db(test_db)

        # Drop some indexes
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")
            conn.execute("DROP INDEX IF EXISTS idx_hypotheses_status")
            conn.execute("DROP INDEX IF EXISTS idx_universe_date")

        # Verify indexes are missing
        before_status = migration_status(test_db)
        assert len(before_status["indexes"]["missing"]) >= 3

        # Run migration
        result = migrate(test_db, dry_run=False)

        # Should succeed
        assert result is True

        # Verify indexes were added
        after_status = migration_status(test_db)
        assert len(after_status["indexes"]["missing"]) == 0

    def test_migrate_populated_database(self, populated_db):
        """Test migrating a database with valid data."""
        # Drop some indexes first
        db = get_db(populated_db)
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")

        # Run migration
        result = migrate(populated_db, dry_run=False)

        # Should succeed
        assert result is True

        # Data should still be intact
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count > 0

        # Indexes should be restored
        status = migration_status(populated_db)
        assert len(status["indexes"]["missing"]) == 0

    def test_migrate_dry_run(self, test_db):
        """Test migration in dry-run mode."""
        db = get_db(test_db)

        # Drop an index
        with db.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prices_symbol_date")

        # Run dry-run migration
        result = migrate(test_db, dry_run=True)

        # Should succeed
        assert result is True

        # Index should still be missing
        status = migration_status(test_db)
        assert "idx_prices_symbol_date" in status["indexes"]["missing"]

    def test_migrate_fails_on_violations(self, old_db):
        """Test migration fails when data has constraint violations."""
        db = get_db(old_db)

        # Insert invalid data (old_db has no constraints enforced)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('BAD', '2020-01-01', -10.0)
            """
            )

        # Migration should fail
        result = migrate(old_db, dry_run=False)

        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_migration_on_empty_database(self, temp_db):
        """Test migration on database with no tables."""
        # Don't initialize schema, just connect to empty db
        result = migrate(temp_db, dry_run=False)

        # Should fail gracefully
        assert result is False

    def test_validation_handles_missing_tables(self, temp_db):
        """Test validation handles missing tables gracefully."""
        # Validate empty database
        violations = validate_data_constraints(temp_db)

        # Should not crash, may return empty dict
        assert isinstance(violations, dict)

    def test_index_creation_handles_missing_tables(self, temp_db):
        """Test index creation skips indexes for missing tables."""
        # Try to add indexes to empty database
        stats = add_indexes(temp_db, dry_run=False)

        # Should skip all indexes (no tables exist)
        assert stats["added"] == 0
        assert stats["skipped"] > 0
