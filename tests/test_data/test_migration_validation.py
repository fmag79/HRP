"""
Migration validation tests for database constraints.

Tests that existing sample data is compatible with new constraints.
Validates that constraints don't break valid use cases.

Tests cover:
- Sample data compatibility with NOT NULL constraints
- Sample data compatibility with CHECK constraints
- Sample data compatibility with FOREIGN KEY constraints
- Realistic workflow scenarios
- No false positives from constraints
"""

from datetime import date

import pandas as pd
import pytest

from hrp.data.db import get_db


class TestSampleDataCompatibility:
    """Test that sample data from fixtures is compatible with new constraints."""

    def test_sample_prices_meet_constraints(self, populated_db):
        """Test that sample_prices fixture data meets all price constraints."""
        db = get_db(populated_db)

        # Verify all sample prices were inserted successfully
        result = db.fetchall("SELECT COUNT(*) as count FROM prices")
        assert result[0][0] > 0, "No sample prices were inserted"

        # Verify all prices meet close > 0 constraint
        invalid_close = db.fetchall(
            """
            SELECT symbol, date, close
            FROM prices
            WHERE close <= 0
        """
        )
        assert len(invalid_close) == 0, f"Found prices with invalid close: {invalid_close}"

        # Verify all prices meet volume >= 0 constraint
        invalid_volume = db.fetchall(
            """
            SELECT symbol, date, volume
            FROM prices
            WHERE volume IS NOT NULL AND volume < 0
        """
        )
        assert len(invalid_volume) == 0, f"Found prices with invalid volume: {invalid_volume}"

        # Verify all prices meet high >= low constraint
        invalid_high_low = db.fetchall(
            """
            SELECT symbol, date, high, low
            FROM prices
            WHERE high IS NOT NULL AND low IS NOT NULL AND high < low
        """
        )
        assert (
            len(invalid_high_low) == 0
        ), f"Found prices with invalid high/low: {invalid_high_low}"

        # Verify source field has value (NOT NULL with default)
        null_source = db.fetchall(
            """
            SELECT symbol, date
            FROM prices
            WHERE source IS NULL
        """
        )
        assert len(null_source) == 0, "Found prices with NULL source"

    def test_sample_prices_all_fields_present(self, populated_db):
        """Test that sample prices have all required fields."""
        db = get_db(populated_db)

        # Check for NULL required fields
        null_required = db.fetchall(
            """
            SELECT symbol, date
            FROM prices
            WHERE symbol IS NULL OR date IS NULL OR close IS NULL
        """
        )
        assert len(null_required) == 0, "Sample prices have NULL required fields"

    def test_sample_prices_symbols(self, populated_db):
        """Test that sample prices contain expected symbols."""
        db = get_db(populated_db)

        symbols = db.fetchall("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
        symbol_list = [row[0] for row in symbols]

        # Sample prices should include AAPL, MSFT, GOOGL
        assert "AAPL" in symbol_list
        assert "MSFT" in symbol_list
        assert "GOOGL" in symbol_list


class TestHypothesisDataCompatibility:
    """Test that hypothesis sample data is compatible with new constraints."""

    def test_sample_hypothesis_structure(self, test_db, sample_hypothesis):
        """Test that sample_hypothesis fixture meets all constraints."""
        db = get_db(test_db)

        # Insert sample hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, status, created_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "HYP-TEST-001",
                sample_hypothesis["title"],
                sample_hypothesis["thesis"],
                sample_hypothesis["prediction"],
                sample_hypothesis["falsification"],
                sample_hypothesis["status"],
                sample_hypothesis["actor"],
            ),
        )

        # Verify hypothesis was inserted
        result = db.fetchone(
            "SELECT title, status FROM hypotheses WHERE hypothesis_id = 'HYP-TEST-001'"
        )
        assert result is not None
        assert result[0] == sample_hypothesis["title"]
        assert result[1] == sample_hypothesis["status"]

    def test_sample_hypothesis_required_fields(self, test_db, sample_hypothesis):
        """Test that sample hypothesis has all required fields."""
        db = get_db(test_db)

        # Verify sample_hypothesis fixture has required keys
        assert "title" in sample_hypothesis
        assert "thesis" in sample_hypothesis
        assert "prediction" in sample_hypothesis
        assert "status" in sample_hypothesis

        # Verify none are None
        assert sample_hypothesis["title"] is not None
        assert sample_hypothesis["thesis"] is not None
        assert sample_hypothesis["prediction"] is not None

    def test_sample_hypothesis_valid_status(self, test_db, sample_hypothesis):
        """Test that sample hypothesis status is valid."""
        db = get_db(test_db)

        valid_statuses = ["draft", "active", "validated", "falsified", "archived"]
        assert (
            sample_hypothesis["status"] in valid_statuses
        ), f"Invalid status: {sample_hypothesis['status']}"


class TestForeignKeyWorkflows:
    """Test that foreign key relationships work with realistic workflows."""

    def test_hypothesis_experiment_lineage_workflow(self, test_db):
        """Test complete workflow: hypothesis -> experiment -> lineage."""
        db = get_db(test_db)

        # Step 1: Create hypothesis
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction)
            VALUES ('HYP-WORKFLOW-001', 'Test Workflow', 'Test thesis', 'Test prediction')
        """
        )

        # Step 2: Link experiment
        db.execute(
            """
            INSERT INTO hypothesis_experiments (hypothesis_id, experiment_id)
            VALUES ('HYP-WORKFLOW-001', 'EXP-WORKFLOW-001')
        """
        )

        # Step 3: Create lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, hypothesis_id, experiment_id)
            VALUES (1, 'experiment_created', 'HYP-WORKFLOW-001', 'EXP-WORKFLOW-001')
        """
        )

        # Verify complete chain
        result = db.fetchone(
            """
            SELECT h.title, he.experiment_id, l.event_type
            FROM hypotheses h
            JOIN hypothesis_experiments he ON h.hypothesis_id = he.hypothesis_id
            JOIN lineage l ON h.hypothesis_id = l.hypothesis_id
            WHERE h.hypothesis_id = 'HYP-WORKFLOW-001'
        """
        )

        assert result is not None
        assert result[0] == "Test Workflow"
        assert result[1] == "EXP-WORKFLOW-001"
        assert result[2] == "experiment_created"

    def test_data_source_ingestion_fundamentals_workflow(self, test_db):
        """Test complete workflow: data_source -> ingestion_log -> fundamentals."""
        db = get_db(test_db)

        # Step 1: Create data source
        db.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES ('test_api', 'market_data', 'active')
        """
        )

        # Step 2: Create ingestion log
        db.execute(
            """
            INSERT INTO ingestion_log (log_id, source_id, records_fetched, records_inserted, status)
            VALUES (1, 'test_api', 100, 98, 'completed')
        """
        )

        # Step 3: Insert fundamentals from this source
        db.execute(
            """
            INSERT INTO fundamentals (symbol, report_date, period_end, metric, value, source)
            VALUES ('AAPL', '2020-01-01', '2019-12-31', 'revenue', 1000000, 'test_api')
        """
        )

        # Verify complete chain
        result = db.fetchone(
            """
            SELECT ds.source_type, il.status, f.metric, f.value
            FROM data_sources ds
            JOIN ingestion_log il ON ds.source_id = il.source_id
            JOIN fundamentals f ON ds.source_id = f.source
            WHERE ds.source_id = 'test_api'
        """
        )

        assert result is not None
        assert result[0] == "market_data"
        assert result[1] == "completed"
        assert result[2] == "revenue"
        assert result[3] == 1000000

    def test_lineage_parent_child_hierarchy(self, test_db):
        """Test lineage parent-child hierarchy with self-referencing FK."""
        db = get_db(test_db)

        # Create parent lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, actor)
            VALUES (1, 'parent_event', 'user')
        """
        )

        # Create child lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, actor, parent_lineage_id)
            VALUES (2, 'child_event', 'system', 1)
        """
        )

        # Create grandchild lineage entry
        db.execute(
            """
            INSERT INTO lineage (lineage_id, event_type, actor, parent_lineage_id)
            VALUES (3, 'grandchild_event', 'system', 2)
        """
        )

        # Verify hierarchy with self-join
        result = db.fetchall(
            """
            SELECT l1.event_type as parent, l2.event_type as child, l3.event_type as grandchild
            FROM lineage l1
            LEFT JOIN lineage l2 ON l1.lineage_id = l2.parent_lineage_id
            LEFT JOIN lineage l3 ON l2.lineage_id = l3.parent_lineage_id
            WHERE l1.lineage_id = 1
        """
        )

        assert len(result) > 0
        assert result[0][0] == "parent_event"
        assert result[0][1] == "child_event"
        assert result[0][2] == "grandchild_event"


class TestNoFalsePositives:
    """Test that constraints don't reject valid data (no false positives)."""

    def test_valid_edge_case_prices(self, test_db):
        """Test that edge case but valid prices are accepted."""
        db = get_db(test_db)

        # Very small close price (penny stock)
        db.execute(
            """
            INSERT INTO prices (symbol, date, close)
            VALUES ('PENNY', '2020-01-01', 0.01)
        """
        )

        # Very large close price
        db.execute(
            """
            INSERT INTO prices (symbol, date, close)
            VALUES ('EXPENSIVE', '2020-01-01', 999999.99)
        """
        )

        # Zero volume (some stocks have no trading on some days)
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, volume)
            VALUES ('ILLIQUID', '2020-01-01', 100.00, 0)
        """
        )

        # High == Low (opening and closing at same price)
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, high, low)
            VALUES ('FLAT', '2020-01-01', 100.00, 100.00, 100.00)
        """
        )

        # Verify all were inserted
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count == 4

    def test_valid_edge_case_hypotheses(self, test_db):
        """Test that edge case but valid hypotheses are accepted."""
        db = get_db(test_db)

        # Confidence score at boundaries
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-ZERO', 'Test', 'Thesis', 'Prediction', 0.0)
        """
        )

        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, confidence_score)
            VALUES ('HYP-ONE', 'Test', 'Thesis', 'Prediction', 1.0)
        """
        )

        # All valid statuses
        valid_statuses = ["draft", "testing", "validated", "rejected", "deployed", "deleted"]
        for i, status in enumerate(valid_statuses):
            db.execute(
                f"""
                INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
                VALUES ('HYP-{status.upper()}', 'Test', 'Thesis', 'Prediction', '{status}')
            """
            )

        # Verify all were inserted
        count = db.fetchone("SELECT COUNT(*) FROM hypotheses")[0]
        assert count == 8  # 2 confidence + 6 status

    def test_valid_null_optional_fields(self, test_db):
        """Test that NULL values are accepted for optional fields."""
        db = get_db(test_db)

        # Prices with NULL optional fields
        db.execute(
            """
            INSERT INTO prices (symbol, date, close, open, high, low, adj_close, volume)
            VALUES ('PARTIAL', '2020-01-01', 100.00, NULL, NULL, NULL, NULL, NULL)
        """
        )

        # Hypothesis with NULL optional fields
        db.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, title, thesis, testable_prediction,
                falsification_criteria, confidence_score, outcome
            )
            VALUES ('HYP-PARTIAL', 'Test', 'Thesis', 'Prediction', NULL, NULL, NULL)
        """
        )

        # Universe with NULL market_cap and exclusion_reason
        db.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, market_cap, exclusion_reason, sector)
            VALUES ('UNKNOWN', '2020-01-01', TRUE, NULL, NULL, NULL)
        """
        )

        # Verify all were inserted
        prices_count = db.fetchone("SELECT COUNT(*) FROM prices WHERE symbol = 'PARTIAL'")[0]
        assert prices_count == 1

        hyp_count = db.fetchone(
            "SELECT COUNT(*) FROM hypotheses WHERE hypothesis_id = 'HYP-PARTIAL'"
        )[0]
        assert hyp_count == 1

        universe_count = db.fetchone("SELECT COUNT(*) FROM universe WHERE symbol = 'UNKNOWN'")[0]
        assert universe_count == 1


class TestRealisticDataScenarios:
    """Test realistic data scenarios that should work with constraints."""

    def test_real_world_price_data_patterns(self, test_db):
        """Test realistic price data patterns."""
        db = get_db(test_db)

        # Stock split scenario (high volume, price change)
        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES ('SPLIT', '2020-01-01', 200.00, 210.00, 198.00, 205.00, 50000000)
        """
        )

        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES ('SPLIT', '2020-01-02', 102.50, 105.00, 101.00, 103.00, 75000000)
        """
        )

        # Gap up scenario (open > previous close)
        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES ('GAPUP', '2020-01-01', 100.00, 105.00, 99.00, 104.00, 1000000)
        """
        )

        db.execute(
            """
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES ('GAPUP', '2020-01-02', 110.00, 115.00, 109.00, 114.00, 2000000)
        """
        )

        # Verify all inserts succeeded
        count = db.fetchone("SELECT COUNT(*) FROM prices WHERE symbol IN ('SPLIT', 'GAPUP')")[0]
        assert count == 4

    def test_real_world_hypothesis_lifecycle(self, test_db):
        """Test realistic hypothesis lifecycle transitions."""
        db = get_db(test_db)

        # Create hypothesis in draft
        db.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, title, thesis, testable_prediction, status)
            VALUES ('HYP-LIFE-001', 'Test Lifecycle', 'Thesis', 'Prediction', 'draft')
        """
        )

        # Update to testing
        db.execute(
            """
            UPDATE hypotheses
            SET status = 'testing', updated_at = CURRENT_TIMESTAMP
            WHERE hypothesis_id = 'HYP-LIFE-001'
        """
        )

        # Add confidence score after testing
        db.execute(
            """
            UPDATE hypotheses
            SET confidence_score = 0.85, updated_at = CURRENT_TIMESTAMP
            WHERE hypothesis_id = 'HYP-LIFE-001'
        """
        )

        # Mark as validated
        db.execute(
            """
            UPDATE hypotheses
            SET status = 'validated', outcome = 'Hypothesis confirmed with p < 0.01'
            WHERE hypothesis_id = 'HYP-LIFE-001'
        """
        )

        # Verify final state
        result = db.fetchone(
            """
            SELECT status, confidence_score, outcome
            FROM hypotheses
            WHERE hypothesis_id = 'HYP-LIFE-001'
        """
        )

        assert result[0] == "validated"
        assert float(result[1]) == 0.85
        assert "confirmed" in result[2].lower()

    def test_bulk_data_insertion(self, test_db, sample_prices):
        """Test bulk insertion of sample data meets all constraints."""
        db = get_db(test_db)

        # Insert all sample prices
        for _, row in sample_prices.iterrows():
            db.execute(
                """
                INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row["symbol"],
                    row["date"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["adj_close"],
                    row["volume"],
                    "test",
                ),
            )

        # Verify all were inserted
        count = db.fetchone("SELECT COUNT(*) FROM prices")[0]
        assert count == len(sample_prices)

        # Verify no constraint violations
        violations = db.fetchall(
            """
            SELECT symbol, date
            FROM prices
            WHERE close <= 0
               OR (volume IS NOT NULL AND volume < 0)
               OR (high IS NOT NULL AND low IS NOT NULL AND high < low)
        """
        )
        assert len(violations) == 0, f"Found constraint violations: {violations}"
