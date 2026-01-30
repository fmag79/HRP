"""
Tests for data cleanup.

Tests cover:
- DataCleanupJob with dry-run mode
- Safety checks and error handling
- Integration workflow
"""

from datetime import date

from hrp.data.retention.cleanup import DataCleanupJob


class TestDataCleanupJob:
    """Tests for DataCleanupJob."""

    def test_init(self):
        """Should initialize cleanup job."""
        job = DataCleanupJob(dry_run=True)

        assert job._dry_run is True
        assert "prices" in job._data_types

    def test_dry_run_does_not_delete(self, test_db):
        """Dry run should not delete any data."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('DRYTEST', 'Dry Run Test', 'NASDAQ')"
            )

        # Insert some old prices
        old_date = date(2020, 1, 1)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('DRYTEST', ?, 100.0, 1000000, 'test')",
                (old_date,),
            )

        # Count before
        count_before = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        # Run dry run cleanup
        job = DataCleanupJob(test_db, dry_run=True)
        results = job.run()

        # Count after (should be same)
        count_after = db.fetchone("SELECT COUNT(*) FROM prices")[0]

        assert count_before == count_after
        assert results["prices"].dry_run is True

    def test_run_with_no_old_data(self, test_db):
        """Should handle case with no old data gracefully."""
        job = DataCleanupJob(test_db, dry_run=True, data_types=["prices"])

        results = job.run()

        assert "prices" in results
        assert results["prices"].success is True
        assert results["prices"].records_deleted == 0


class TestSafetyChecks:
    """Tests for safety checks in cleanup operations."""

    def test_require_confirmation_flag(self):
        """Should respect require_confirmation flag."""
        job = DataCleanupJob(dry_run=True, require_confirmation=False)

        assert job._require_confirmation is False

    def test_custom_data_types(self):
        """Should allow custom data type selection."""
        job = DataCleanupJob(dry_run=True, data_types=["prices"])

        assert job._data_types == ["prices"]

    def test_error_handling_invalid_data_type(self, test_db):
        """Should handle invalid data types gracefully."""
        job = DataCleanupJob(test_db, dry_run=True, data_types=["invalid_type"])

        results = job.run()

        # Should return error result for invalid type
        assert "invalid_type" in results
        assert len(results["invalid_type"].errors) > 0


class TestIntegration:
    """Integration tests for cleanup system."""

    def test_full_cleanup_workflow(self, test_db):
        """Should complete full cleanup workflow."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbols first
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('HOT', 'Hot Test', 'NASDAQ')"
            )
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('COLD', 'Cold Test', 'NASDAQ')"
            )

        # Insert test data at various ages
        with db.connection() as conn:
            # Hot data (recent)
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('HOT', '2024-01-01', 100.0, 1000000, 'test')"
            )
            # Cold data (old)
            conn.execute(
                "INSERT INTO prices (symbol, date, close, volume, source) "
                "VALUES ('COLD', '2020-01-01', 100.0, 1000000, 'test')"
            )

        # Run dry run
        job = DataCleanupJob(test_db, dry_run=True)
        results = job.run()

        assert results["prices"].dry_run is True
        assert results["prices"].success is True
