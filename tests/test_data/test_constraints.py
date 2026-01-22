"""
Comprehensive tests for database constraints.

Tests cover:
- NOT NULL constraint enforcement
- CHECK constraint validation (value ranges, enums)
- FOREIGN KEY constraint enforcement
- Valid data insertion
- Index existence and naming
"""

from datetime import date, datetime

import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import INDEXES, create_tables


class TestNotNullConstraints:
    """Tests for NOT NULL constraint enforcement."""

    def test_universe_not_null_symbol(self, test_db):
        """Test that universe.symbol NOT NULL is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO universe (symbol, date, in_universe)
                VALUES (NULL, '2020-01-01', TRUE)
            """
            )

    def test_universe_not_null_date(self, test_db):
        """Test that universe.date NOT NULL is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO universe (symbol, date, in_universe)
                VALUES ('AAPL', NULL, TRUE)
            """
            )

    def test_universe_not_null_in_universe(self, test_db):
        """Test that universe.in_universe NOT NULL is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO universe (symbol, date, in_universe)
                VALUES ('AAPL', '2020-01-01', NULL)
            """
            )

    def test_prices_not_null_close(self, test_db):
        """Test that prices.close NOT NULL is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2020-01-01', NULL)
            """
            )

    def test_prices_not_null_source(self, test_db):
        """Test that prices.source NOT NULL is enforced (has default)."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Should succeed because source has a default value
        db.execute(
            """
            INSERT INTO prices (symbol, date, close)
            VALUES ('AAPL', '2020-01-01', 100.00)
        """
        )

        result = db.fetchone("SELECT source FROM prices WHERE symbol = 'AAPL'")
        assert result[0] == "unknown"

    def test_fundamentals_required_fields(self, test_db):
        """Test that fundamentals required fields are enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Missing symbol
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO fundamentals (symbol, report_date, period_end, metric, value)
                VALUES (NULL, '2020-01-01', '2019-12-31', 'revenue', 1000000)
            """
            )

        # Missing metric
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO fundamentals (symbol, report_date, period_end, metric, value)
                VALUES ('AAPL', '2020-01-01', '2019-12-31', NULL, 1000000)
            """
            )

    def test_hypotheses_required_fields(self, test_db):
        """Test that hypotheses required fields are enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Missing title
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
                VALUES ('HYP-001', NULL, 'Test thesis', 'Test prediction')
            """
            )

        # Missing thesis
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
                VALUES ('HYP-001', 'Test', NULL, 'Test prediction')
            """
            )

        # Missing testable_prediction
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
                VALUES ('HYP-001', 'Test', 'Test thesis', NULL)
            """
            )

    def test_lineage_required_fields(self, test_db):
        """Test that lineage required fields are enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Missing event_type
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO lineage (lineage_id, event_type, timestamp, actor)
                VALUES (1, NULL, CURRENT_TIMESTAMP, 'system')
            """
            )

        # timestamp and actor have defaults, so this should succeed
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type)
            VALUES (1, 'test_event')
        """
        )

        result = db.fetchone("SELECT actor FROM lineage WHERE lineage_id = 1")
        assert result[0] == "system"


class TestCheckConstraints:
    """Tests for CHECK constraint validation."""

    def test_universe_market_cap_positive(self, test_db):
        """Test that universe.market_cap >= 0 is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Negative market cap should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO universe (symbol, date, in_universe, market_cap)
                VALUES ('AAPL', '2020-01-01', TRUE, -1000000)
            """
            )

        # Positive market cap should succeed
        db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, market_cap)
            VALUES ('AAPL', '2020-01-01', TRUE, 1000000)
        """
        )

        result = db.fetchone("SELECT market_cap FROM universe WHERE symbol = 'AAPL'")
        assert result[0] == 1000000

        # NULL market cap should be allowed
        db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, market_cap)
            VALUES ('MSFT', '2020-01-01', TRUE, NULL)
        """
        )

    def test_prices_close_positive(self, test_db):
        """Test that prices.close > 0 is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Zero close should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2020-01-01', 0)
            """
            )

        # Negative close should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close)
                VALUES ('AAPL', '2020-01-01', -10.00)
            """
            )

        # Positive close should succeed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close)
            VALUES ('AAPL', '2020-01-01', 100.00)
        """
        )

    def test_prices_volume_non_negative(self, test_db):
        """Test that prices.volume >= 0 is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Negative volume should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close, volume)
                VALUES ('AAPL', '2020-01-01', 100.00, -1000)
            """
            )

        # Zero volume should succeed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, volume)
            VALUES ('AAPL', '2020-01-01', 100.00, 0)
        """
        )

        # NULL volume should be allowed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, volume)
            VALUES ('MSFT', '2020-01-01', 150.00, NULL)
        """
        )

    def test_prices_high_low_relationship(self, test_db):
        """Test that prices.high >= low is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # High < Low should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO prices (symbol, date, close, high, low)
                VALUES ('AAPL', '2020-01-01', 100.00, 95.00, 105.00)
            """
            )

        # High >= Low should succeed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, high, low)
            VALUES ('AAPL', '2020-01-01', 100.00, 105.00, 95.00)
        """
        )

        # High == Low should succeed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, high, low)
            VALUES ('MSFT', '2020-01-01', 100.00, 100.00, 100.00)
        """
        )

        # NULL values should be allowed
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, high, low)
            VALUES ('GOOGL', '2020-01-01', 100.00, NULL, NULL)
        """
        )

    def test_corporate_actions_factor_positive(self, test_db):
        """Test that corporate_actions.factor > 0 is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Zero factor should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO corporate_actions (symbol, date, action_type, factor)
                VALUES ('AAPL', '2020-01-01', 'split', 0)
            """
            )

        # Negative factor should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO corporate_actions (symbol, date, action_type, factor)
                VALUES ('AAPL', '2020-01-01', 'split', -2.0)
            """
            )

        # Positive factor should succeed
        db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor)
            VALUES ('AAPL', '2020-01-01', 'split', 2.0)
        """
        )

        # NULL factor should be allowed
        db.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action_type, factor)
            VALUES ('MSFT', '2020-01-01', 'dividend', NULL)
        """
        )

    def test_data_sources_status_enum(self, test_db):
        """Test that data_sources.status enum is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Invalid status should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO data_sources (source_id, source_type, status)
                VALUES ('test_source', 'api', 'invalid_status')
            """
            )

        # Valid statuses should succeed
        for status in ["active", "inactive", "deprecated"]:
            db.execute(
                f"""
                INSERT INTO data_sources (source_id, source_type, status)
                VALUES ('{status}_source', 'api', '{status}')
            """
            )

        result = db.fetchall("SELECT status FROM data_sources ORDER BY source_id")
        assert len(result) == 3
        assert result[0][0] == "active"
        assert result[1][0] == "deprecated"
        assert result[2][0] == "inactive"

    def test_ingestion_log_records_non_negative(self, test_db):
        """Test that ingestion_log records counts are non-negative."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # First create a data source for FK constraint
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type)
            VALUES ('test_source', 'api')
        """
        )

        # Negative records_fetched should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, records_fetched, records_inserted)
                VALUES (1, 'test_source', -1, 0)
            """
            )

        # Negative records_inserted should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, records_fetched, records_inserted)
                VALUES (1, 'test_source', 0, -1)
            """
            )

        # Non-negative values should succeed
        db.execute(
            """
            INSERT INTO ingestion_log (log_id, source_id, records_fetched, records_inserted)
            VALUES (1, 'test_source', 100, 95)
        """
        )

    def test_ingestion_log_status_enum(self, test_db):
        """Test that ingestion_log.status enum is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # First create a data source for FK constraint
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type)
            VALUES ('test_source', 'api')
        """
        )

        # Invalid status should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id, status)
                VALUES (1, 'test_source', 'invalid_status')
            """
            )

        # Valid statuses should succeed
        for i, status in enumerate(["running", "completed", "failed"], start=1):
            db.execute(
                f"""
                INSERT INTO ingestion_log (log_id, source_id, status)
                VALUES ({i}, 'test_source', '{status}')
            """
            )

        result = db.fetchall("SELECT status FROM ingestion_log ORDER BY log_id")
        assert len(result) == 3
        assert result[0][0] == "running"
        assert result[1][0] == "completed"
        assert result[2][0] == "failed"

    def test_hypotheses_confidence_score_range(self, test_db):
        """Test that hypotheses.confidence_score is between 0 and 1."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Score > 1 should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
                VALUES ('HYP-001', 'Test', 'Test thesis', 'Test prediction', 1.5)
            """
            )

        # Score < 0 should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
                VALUES ('HYP-001', 'Test', 'Test thesis', 'Test prediction', -0.1)
            """
            )

        # Score == 0 should succeed
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-001', 'Test', 'Test thesis', 'Test prediction', 0.0)
        """
        )

        # Score == 1 should succeed
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-002', 'Test', 'Test thesis', 'Test prediction', 1.0)
        """
        )

        # Score between 0 and 1 should succeed
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-003', 'Test', 'Test thesis', 'Test prediction', 0.75)
        """
        )

        # NULL should be allowed
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-004', 'Test', 'Test thesis', 'Test prediction', NULL)
        """
        )

    def test_hypotheses_status_enum(self, test_db):
        """Test that hypotheses.status enum is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Invalid status should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-001', 'Test', 'Test thesis', 'Test prediction', 'invalid_status')
            """
            )

        # Valid statuses should succeed
        valid_statuses = ["draft", "testing", "validated", "rejected", "deployed", "deleted"]
        for i, status in enumerate(valid_statuses, start=1):
            db.execute(
                f"""
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-{i:03d}', 'Test', 'Test thesis', 'Test prediction', '{status}')
            """
            )

        result = db.fetchall("SELECT status FROM hypotheses ORDER BY hypothesis_id")
        assert len(result) == 6
        for i, status in enumerate(valid_statuses):
            assert result[i][0] == status


class TestForeignKeyConstraints:
    """Tests for FOREIGN KEY constraint enforcement."""

    def test_fundamentals_source_fk(self, test_db):
        """Test that fundamentals.source FK is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert with non-existent source should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO fundamentals (symbol, report_date, period_end, metric, value, source)
                VALUES ('AAPL', '2020-01-01', '2019-12-31', 'revenue', 1000000, 'nonexistent_source')
            """
            )

        # Create a valid source
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type)
            VALUES ('valid_source', 'api')
        """
        )

        # Insert with valid source should succeed
        db.execute(
            """
            INSERT INTO fundamentals (symbol, report_date, period_end, metric, value, source)
            VALUES ('AAPL', '2020-01-01', '2019-12-31', 'revenue', 1000000, 'valid_source')
        """
        )

        result = db.fetchone("SELECT source FROM fundamentals WHERE symbol = 'AAPL'")
        assert result[0] == "valid_source"

    def test_ingestion_log_source_id_fk(self, test_db):
        """Test that ingestion_log.source_id FK is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert with non-existent source should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO ingestion_log (log_id, source_id)
                VALUES (1, 'nonexistent_source')
            """
            )

        # Create a valid source
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type)
            VALUES ('test_source', 'api')
        """
        )

        # Insert with valid source should succeed
        db.execute(
            """
            INSERT INTO ingestion_log (log_id, source_id)
            VALUES (1, 'test_source')
        """
        )

        result = db.fetchone("SELECT source_id FROM ingestion_log WHERE log_id = 1")
        assert result[0] == "test_source"

    @pytest.mark.skip(reason="FK constraint removed due to DuckDB 1.4.3 limitation - FKs prevent UPDATE on parent table")
    def test_hypothesis_experiments_hypothesis_id_fk(self, test_db):
        """Test that hypothesis_experiments.hypothesis_id FK is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert with non-existent hypothesis should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
                VALUES ('NONEXISTENT', 'EXP-001')
            """
            )

        # Create a valid hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-001', 'Test Hypothesis', 'Test thesis', 'Test prediction')
        """
        )

        # Insert with valid hypothesis should succeed
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
            VALUES ('HYP-001', 'EXP-001')
        """
        )

        result = db.fetchone(
            "SELECT hypothesis_id FROM hypothesis_experiments WHERE experiment_id = 'EXP-001'"
        )
        assert result[0] == "HYP-001"

    @pytest.mark.skip(reason="FK constraint removed due to DuckDB 1.4.3 limitation - FKs prevent UPDATE on parent table")
    def test_lineage_hypothesis_id_fk(self, test_db):
        """Test that lineage.hypothesis_id FK is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert with non-existent hypothesis should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO lineage (lineage_id, event_type, hypothesis_id)
                VALUES (1, 'test_event', 'NONEXISTENT')
            """
            )

        # Create a valid hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-001', 'Test Hypothesis', 'Test thesis', 'Test prediction')
        """
        )

        # Insert with valid hypothesis should succeed
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, hypothesis_id)
            VALUES (1, 'test_event', 'HYP-001')
        """
        )

        result = db.fetchone("SELECT hypothesis_id FROM lineage WHERE lineage_id = 1")
        assert result[0] == "HYP-001"

        # NULL hypothesis_id should be allowed
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, hypothesis_id)
            VALUES (2, 'test_event', NULL)
        """
        )

    @pytest.mark.skip(reason="FK constraint removed due to DuckDB 1.4.3 limitation - FKs prevent UPDATE on parent table")
    def test_lineage_parent_lineage_id_self_reference_fk(self, test_db):
        """Test that lineage.parent_lineage_id self-reference FK is enforced."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert with non-existent parent should fail
        with pytest.raises(Exception):
            db.execute(
                """
                INSERT INTO lineage (lineage_id, event_type, parent_lineage_id)
                VALUES (1, 'test_event', 999)
            """
            )

        # Create a parent lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type)
            VALUES (1, 'parent_event')
        """
        )

        # Insert with valid parent should succeed
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, parent_lineage_id)
            VALUES (2, 'child_event', 1)
        """
        )

        result = db.fetchone("SELECT parent_lineage_id FROM lineage WHERE lineage_id = 2")
        assert result[0] == 1

        # NULL parent_lineage_id should be allowed
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, parent_lineage_id)
            VALUES (3, 'orphan_event', NULL)
        """
        )


class TestValidDataInsertion:
    """Tests that valid data inserts successfully with all constraints."""

    def test_complete_prices_insert(self, test_db):
        """Test inserting complete price record with all fields."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES ('AAPL', '2020-01-01', 99.00, 102.00, 98.00, 100.00, 100.00, 1000000, 'yahoo')
        """
        )

        result = db.fetchone("SELECT * FROM prices WHERE symbol = 'AAPL'")
        assert result is not None
        assert result[0] == "AAPL"
        assert result[1] == date(2020, 1, 1)
        assert result[8] == "yahoo"

    def test_complete_hypothesis_workflow(self, test_db):
        """Test complete hypothesis workflow with experiments and lineage."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, status, confidence_score
            )
            VALUES (
                'HYP-001', 'Momentum Effect',
                'Stocks with high momentum continue to outperform',
                'Top decile > SPY by 3% annually',
                'Sharpe < 1.0 or p > 0.05',
                'testing', 0.8
            )
        """
        )

        # Create experiment link
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id, relationship)
            VALUES ('HYP-001', 'EXP-2020-001', 'primary')
        """
        )

        # Create lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, hypothesis_id, experiment_id, actor)
            VALUES (1, 'experiment_created', 'HYP-001', 'EXP-2020-001', 'user')
        """
        )

        # Create child lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, hypothesis_id, experiment_id, actor, parent_lineage_id)
            VALUES (2, 'experiment_completed', 'HYP-001', 'EXP-2020-001', 'system', 1)
        """
        )

        # Verify all data
        hyp = db.fetchone("SELECT title, status, confidence_score FROM hypotheses WHERE hypothesis_id = 'HYP-001'")
        assert hyp[0] == "Momentum Effect"
        assert hyp[1] == "testing"
        assert float(hyp[2]) == 0.8

        exp = db.fetchone("SELECT experiment_id FROM hypothesis_experiments WHERE hypothesis_id = 'HYP-001'")
        assert exp[0] == "EXP-2020-001"

        lineages = db.fetchall("SELECT event_type, parent_lineage_id FROM lineage ORDER BY lineage_id")
        assert len(lineages) == 2
        assert lineages[0][0] == "experiment_created"
        assert lineages[0][1] is None
        assert lineages[1][0] == "experiment_completed"
        assert lineages[1][1] == 1

    def test_complete_data_ingestion_workflow(self, test_db):
        """Test complete data ingestion workflow with source and logs."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create data source
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type, api_name, status)
            VALUES ('yahoo_finance', 'market_data', 'yfinance', 'active')
        """
        )

        # Create ingestion log
        db.execute(
            """
            INSERT INTO ingestion_log (
                log_id, source_id, records_fetched, records_inserted, status
            )
            VALUES (1, 'yahoo_finance', 500, 495, 'completed')
        """
        )

        # Insert fundamentals from this source
        db.execute(
            """
            INSERT INTO fundamentals (symbol, report_date, period_end, metric, value, source)
            VALUES ('AAPL', '2020-01-01', '2019-12-31', 'revenue', 91819000000, 'yahoo_finance')
        """
        )

        # Verify all data
        source = db.fetchone("SELECT source_type, status FROM data_sources WHERE source_id = 'yahoo_finance'")
        assert source[0] == "market_data"
        assert source[1] == "active"

        log = db.fetchone("SELECT records_fetched, records_inserted, status FROM ingestion_log WHERE log_id = 1")
        assert log[0] == 500
        assert log[1] == 495
        assert log[2] == "completed"

        fundamental = db.fetchone("SELECT value, source FROM fundamentals WHERE symbol = 'AAPL'")
        assert fundamental[0] == 91819000000
        assert fundamental[1] == "yahoo_finance"


class TestIndexExistence:
    """Tests that all required indexes are created."""

    def test_all_indexes_created(self, test_db):
        """Test that all indexes defined in schema are created."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Get list of indexes from database
        result = db.fetchall(
            """
            SELECT DISTINCT index_name
            FROM duckdb_indexes()
            ORDER BY index_name
        """
        )

        db_indexes = {row[0] for row in result}

        # Extract index names from INDEXES list
        # Index names follow pattern: CREATE INDEX IF NOT EXISTS <name> ON ...
        expected_indexes = set()
        for index_sql in INDEXES:
            # Parse index name from SQL
            parts = index_sql.split()
            idx_name_pos = parts.index("EXISTS") + 1
            expected_indexes.add(parts[idx_name_pos])

        # Verify all expected indexes exist
        missing_indexes = expected_indexes - db_indexes
        assert len(missing_indexes) == 0, f"Missing indexes: {missing_indexes}"

        # Verify we have at least the required count
        assert len(db_indexes) >= 15, f"Expected at least 15 indexes, found {len(db_indexes)}"

    def test_prices_composite_index(self, test_db):
        """Test that prices(symbol, date) composite index exists."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        result = db.fetchall(
            """
            SELECT index_name, expressions
            FROM duckdb_indexes()
            WHERE table_name = 'prices' AND index_name = 'idx_prices_symbol_date'
        """
        )

        assert len(result) > 0, "idx_prices_symbol_date index not found"
        assert 'symbol' in result[0][1] and 'date' in result[0][1], "Index should contain symbol and date columns"

    def test_features_composite_index(self, test_db):
        """Test that features(symbol, date, feature_name) composite index exists."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        result = db.fetchall(
            """
            SELECT index_name, expressions
            FROM duckdb_indexes()
            WHERE table_name = 'features' AND index_name = 'idx_features_symbol_date'
        """
        )

        assert len(result) > 0, "idx_features_symbol_date index not found"
        assert 'symbol' in result[0][1] and 'date' in result[0][1], "Index should contain symbol and date columns"

    def test_lineage_indexes(self, test_db):
        """Test that lineage table has required indexes."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        result = db.fetchall(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = 'lineage'
            ORDER BY index_name
        """
        )

        index_names = {row[0] for row in result}

        # Lineage should have indexes on hypothesis_id, timestamp, experiment_id, and event_type
        expected_lineage_indexes = {
            "idx_lineage_hypothesis",
            "idx_lineage_timestamp",
            "idx_lineage_experiment",
            "idx_lineage_event_type",
        }

        missing = expected_lineage_indexes - index_names
        assert len(missing) == 0, f"Missing lineage indexes: {missing}"


class TestConstraintIntegration:
    """Integration tests for multiple constraints working together."""

    def test_constraints_with_sample_data(self, populated_db):
        """Test that sample data from fixtures works with all constraints."""
        from hrp.data.db import get_db

        db = get_db(populated_db)

        # Verify sample prices were inserted successfully
        result = db.fetchall("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
        symbols = [row[0] for row in result]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

        # Verify all prices meet constraints
        invalid_prices = db.fetchall(
            """
            SELECT symbol, date
            FROM prices
            WHERE close <= 0 OR (volume IS NOT NULL AND volume < 0)
        """
        )
        assert len(invalid_prices) == 0, "Sample data contains invalid prices"

    def test_cascade_constraint_validation(self, test_db):
        """Test that constraints are checked in proper order (FK after data validation)."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-001', 'Test', 'Thesis', 'Prediction')
        """
        )

        # Try to insert hypothesis_experiment with invalid confidence score in hypothesis
        # This should fail on CHECK constraint before FK constraint
        with pytest.raises(Exception):
            db.execute(
                """
                UPDATE hypotheses
                SET confidence_score = 2.0
                WHERE hypothesis_id = 'HYP-001'
            """
            )

        # Verify hypothesis still has NULL confidence score
        result = db.fetchone("SELECT confidence_score FROM hypotheses WHERE hypothesis_id = 'HYP-001'")
        assert result[0] is None

    def test_default_values_with_constraints(self, test_db):
        """Test that default values work correctly with NOT NULL constraints."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert minimal universe record (should use defaults)
        db.execute(
            """
            INSERT INTO universe (symbol, date)
            VALUES ('AAPL', '2020-01-01')
        """
        )

        result = db.fetchone("SELECT in_universe, created_at FROM universe WHERE symbol = 'AAPL'")
        assert result[0] is True  # default value
        assert result[1] is not None  # default CURRENT_TIMESTAMP

        # Insert minimal hypothesis (should use defaults)
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-001', 'Test', 'Thesis', 'Prediction')
        """
        )

        result = db.fetchone(
            "SELECT status, created_by, created_at FROM hypotheses WHERE hypothesis_id = 'HYP-001'"
        )
        assert result[0] == "draft"  # default value
        assert result[1] == "user"  # default value
        assert result[2] is not None  # default CURRENT_TIMESTAMP
