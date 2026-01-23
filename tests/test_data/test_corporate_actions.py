"""
Comprehensive integration tests for corporate actions ingestion.

Tests cover:
- ingest_corporate_actions() function with various scenarios
- _upsert_corporate_actions() database operations
- get_corporate_action_stats() statistics reporting
- Error handling and edge cases
- Multi-symbol ingestion
- Duplicate handling (upsert behavior)
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.ingestion.corporate_actions import (
    ingest_corporate_actions,
    get_corporate_action_stats,
    _upsert_corporate_actions,
)


class TestIngestCorporateActions:
    """Tests for the ingest_corporate_actions function."""

    def test_ingest_single_symbol_with_actions(self, test_db):
        """Test ingesting corporate actions for a single symbol."""
        # Create mock data
        mock_actions = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': [date(2023, 3, 15), date(2023, 6, 15)],
            'action_type': ['dividend', 'dividend'],
            'value': [0.23, 0.24],
            'source': ['yfinance', 'yfinance'],
        })

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.return_value = mock_actions
            mock_source_class.return_value = mock_source

            stats = ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats['symbols_requested'] == 1
        assert stats['symbols_success'] == 1
        assert stats['symbols_failed'] == 0
        assert stats['rows_fetched'] == 2
        assert stats['rows_inserted'] == 2
        assert stats['failed_symbols'] == []

    def test_ingest_multiple_symbols(self, test_db):
        """Test ingesting corporate actions for multiple symbols."""
        # Create mock data for different symbols
        mock_actions_aapl = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': [date(2023, 3, 15), date(2023, 6, 15)],
            'action_type': ['dividend', 'dividend'],
            'value': [0.23, 0.24],
            'source': ['yfinance', 'yfinance'],
        })

        mock_actions_msft = pd.DataFrame({
            'symbol': ['MSFT'],
            'date': [date(2023, 5, 20)],
            'action_type': ['dividend'],
            'value': [0.68],
            'source': ['yfinance'],
        })

        def get_actions_side_effect(symbol, start, end):
            if symbol == 'AAPL':
                return mock_actions_aapl
            elif symbol == 'MSFT':
                return mock_actions_msft
            else:
                return pd.DataFrame()

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.side_effect = get_actions_side_effect
            mock_source_class.return_value = mock_source

            stats = ingest_corporate_actions(
                symbols=['AAPL', 'MSFT'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats['symbols_requested'] == 2
        assert stats['symbols_success'] == 2
        assert stats['symbols_failed'] == 0
        assert stats['rows_fetched'] == 3
        assert stats['rows_inserted'] == 3

    def test_ingest_symbol_with_no_actions(self, test_db):
        """Test ingesting a symbol with no corporate actions."""
        # Return empty DataFrame
        mock_actions = pd.DataFrame()

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.return_value = mock_actions
            mock_source_class.return_value = mock_source

            stats = ingest_corporate_actions(
                symbols=['GOOGL'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats['symbols_requested'] == 1
        assert stats['symbols_success'] == 1  # Not an error, just no actions
        assert stats['symbols_failed'] == 0
        assert stats['rows_fetched'] == 0
        assert stats['rows_inserted'] == 0

    def test_ingest_with_failed_symbol(self, test_db):
        """Test ingestion when one symbol fails."""
        mock_actions = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.23],
            'source': ['yfinance'],
        })

        def get_actions_side_effect(symbol, start, end):
            if symbol == 'AAPL':
                return mock_actions
            elif symbol == 'INVALID':
                raise Exception("Invalid symbol")
            else:
                return pd.DataFrame()

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.side_effect = get_actions_side_effect
            mock_source_class.return_value = mock_source

            stats = ingest_corporate_actions(
                symbols=['AAPL', 'INVALID'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats['symbols_requested'] == 2
        assert stats['symbols_success'] == 1
        assert stats['symbols_failed'] == 1
        assert stats['failed_symbols'] == ['INVALID']

    def test_ingest_with_both_dividends_and_splits(self, test_db):
        """Test ingesting both dividend and split actions."""
        mock_actions = pd.DataFrame({
            'symbol': ['TSLA', 'TSLA', 'TSLA'],
            'date': [date(2023, 3, 15), date(2023, 6, 20), date(2023, 9, 15)],
            'action_type': ['dividend', 'split', 'dividend'],
            'value': [0.10, 3.0, 0.12],
            'source': ['yfinance', 'yfinance', 'yfinance'],
        })

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.return_value = mock_actions
            mock_source_class.return_value = mock_source

            stats = ingest_corporate_actions(
                symbols=['TSLA'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats['rows_fetched'] == 3
        assert stats['rows_inserted'] == 3

    def test_ingest_upsert_behavior(self, test_db):
        """Test that re-ingesting same data uses upsert (no duplicates)."""
        mock_actions = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.23],
            'source': ['yfinance'],
        })

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.return_value = mock_actions
            mock_source_class.return_value = mock_source

            # First ingestion
            stats1 = ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

            # Second ingestion of same data
            stats2 = ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        assert stats1['rows_inserted'] == 1
        assert stats2['rows_inserted'] == 1

        # Verify only one row in database (upsert, not duplicate)
        from hrp.data.db import get_db
        db = get_db()
        count = db.fetchone("SELECT COUNT(*) FROM corporate_actions")
        assert count[0] == 1

    def test_ingest_invalid_source(self, test_db):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='invalid_source',
            )

        assert "Unknown source" in str(exc_info.value)


class TestUpsertCorporateActions:
    """Tests for the _upsert_corporate_actions helper function."""

    def test_upsert_single_action(self, test_db):
        """Test upserting a single corporate action."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.23],
            'source': ['yfinance'],
        })

        rows = _upsert_corporate_actions(db, df)

        assert rows == 1

        # Verify data in database
        result = db.fetchone(
            "SELECT symbol, date, action_type, factor, source FROM corporate_actions WHERE symbol = 'AAPL'"
        )
        assert result[0] == 'AAPL'
        assert result[1] == date(2023, 3, 15)
        assert result[2] == 'dividend'
        assert float(result[3]) == 0.23
        assert result[4] == 'yfinance'

    def test_upsert_multiple_actions(self, test_db):
        """Test upserting multiple corporate actions."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT'],
            'date': [date(2023, 3, 15), date(2023, 6, 15), date(2023, 5, 20)],
            'action_type': ['dividend', 'dividend', 'split'],
            'value': [0.23, 0.24, 2.0],
            'source': ['yfinance', 'yfinance', 'yfinance'],
        })

        rows = _upsert_corporate_actions(db, df)

        assert rows == 3

        # Verify count in database
        count = db.fetchone("SELECT COUNT(*) FROM corporate_actions")
        assert count[0] == 3

    def test_upsert_empty_dataframe(self, test_db):
        """Test upserting empty DataFrame returns 0."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame()

        rows = _upsert_corporate_actions(db, df)

        assert rows == 0

    def test_upsert_replaces_existing(self, test_db):
        """Test that upsert replaces existing records with same key."""
        from hrp.data.db import get_db

        db = get_db()

        # First insert
        df1 = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.23],
            'source': ['yfinance'],
        })
        _upsert_corporate_actions(db, df1)

        # Second insert with same key but different value
        df2 = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.25],  # Updated value
            'source': ['yfinance_v2'],
        })
        _upsert_corporate_actions(db, df2)

        # Verify only one row exists with updated value
        count = db.fetchone("SELECT COUNT(*) FROM corporate_actions")
        assert count[0] == 1

        result = db.fetchone(
            "SELECT factor, source FROM corporate_actions WHERE symbol = 'AAPL'"
        )
        assert float(result[0]) == 0.25
        assert result[1] == 'yfinance_v2'


class TestGetCorporateActionStats:
    """Tests for the get_corporate_action_stats function."""

    def test_stats_empty_table(self, test_db):
        """Test statistics with empty corporate actions table."""
        stats = get_corporate_action_stats()

        assert stats['total_rows'] == 0
        assert stats['unique_symbols'] == 0
        assert stats['date_range']['start'] is None
        assert stats['date_range']['end'] is None
        assert stats['by_type'] == []
        assert stats['per_symbol'] == []

    def test_stats_with_single_symbol(self, test_db):
        """Test statistics with single symbol."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': [date(2023, 3, 15), date(2023, 6, 15)],
            'action_type': ['dividend', 'dividend'],
            'value': [0.23, 0.24],
            'source': ['yfinance', 'yfinance'],
        })
        _upsert_corporate_actions(db, df)

        stats = get_corporate_action_stats()

        assert stats['total_rows'] == 2
        assert stats['unique_symbols'] == 1
        assert stats['date_range']['start'] == date(2023, 3, 15)
        assert stats['date_range']['end'] == date(2023, 6, 15)
        assert len(stats['by_type']) == 1
        assert stats['by_type'][0]['action_type'] == 'dividend'
        assert stats['by_type'][0]['count'] == 2
        assert len(stats['per_symbol']) == 1
        assert stats['per_symbol'][0]['symbol'] == 'AAPL'
        assert stats['per_symbol'][0]['rows'] == 2

    def test_stats_with_multiple_symbols(self, test_db):
        """Test statistics with multiple symbols."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'TSLA'],
            'date': [date(2023, 3, 15), date(2023, 6, 15), date(2023, 5, 20), date(2023, 8, 10)],
            'action_type': ['dividend', 'dividend', 'dividend', 'split'],
            'value': [0.23, 0.24, 0.68, 3.0],
            'source': ['yfinance', 'yfinance', 'yfinance', 'yfinance'],
        })
        _upsert_corporate_actions(db, df)

        stats = get_corporate_action_stats()

        assert stats['total_rows'] == 4
        assert stats['unique_symbols'] == 3
        assert stats['date_range']['start'] == date(2023, 3, 15)
        assert stats['date_range']['end'] == date(2023, 8, 10)
        assert len(stats['by_type']) == 2
        assert len(stats['per_symbol']) == 3

    def test_stats_by_action_type(self, test_db):
        """Test statistics grouping by action type."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'TSLA', 'NVDA'],
            'date': [date(2023, 3, 15), date(2023, 6, 15), date(2023, 5, 20), date(2023, 8, 10), date(2023, 9, 1)],
            'action_type': ['dividend', 'dividend', 'dividend', 'split', 'split'],
            'value': [0.23, 0.24, 0.68, 3.0, 4.0],
            'source': ['yfinance', 'yfinance', 'yfinance', 'yfinance', 'yfinance'],
        })
        _upsert_corporate_actions(db, df)

        stats = get_corporate_action_stats()

        # Find dividend and split counts
        by_type = {item['action_type']: item['count'] for item in stats['by_type']}
        assert by_type['dividend'] == 3
        assert by_type['split'] == 2

    def test_stats_per_symbol_date_ranges(self, test_db):
        """Test that per-symbol stats include correct date ranges."""
        from hrp.data.db import get_db

        db = get_db()
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT'],
            'date': [date(2023, 1, 15), date(2023, 12, 15), date(2023, 6, 20)],
            'action_type': ['dividend', 'dividend', 'dividend'],
            'value': [0.23, 0.24, 0.68],
            'source': ['yfinance', 'yfinance', 'yfinance'],
        })
        _upsert_corporate_actions(db, df)

        stats = get_corporate_action_stats()

        # Find AAPL stats
        aapl_stats = [s for s in stats['per_symbol'] if s['symbol'] == 'AAPL'][0]
        assert aapl_stats['rows'] == 2
        assert aapl_stats['start'] == date(2023, 1, 15)
        assert aapl_stats['end'] == date(2023, 12, 15)

        # Find MSFT stats
        msft_stats = [s for s in stats['per_symbol'] if s['symbol'] == 'MSFT'][0]
        assert msft_stats['rows'] == 1
        assert msft_stats['start'] == date(2023, 6, 20)
        assert msft_stats['end'] == date(2023, 6, 20)


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_full_ingestion_workflow(self, test_db):
        """Test complete workflow from ingestion to stats."""
        # Mock corporate actions data
        mock_actions = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT'],
            'date': [date(2023, 3, 15), date(2023, 6, 15), date(2023, 5, 20)],
            'action_type': ['dividend', 'dividend', 'split'],
            'value': [0.23, 0.24, 2.0],
            'source': ['yfinance', 'yfinance', 'yfinance'],
        })

        def get_actions_side_effect(symbol, start, end):
            return mock_actions[mock_actions['symbol'] == symbol]

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            mock_source = MagicMock()
            mock_source.get_corporate_actions.side_effect = get_actions_side_effect
            mock_source_class.return_value = mock_source

            # Run ingestion
            ingest_stats = ingest_corporate_actions(
                symbols=['AAPL', 'MSFT'],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        # Verify ingestion stats
        assert ingest_stats['symbols_success'] == 2
        assert ingest_stats['rows_inserted'] == 3

        # Get storage stats
        storage_stats = get_corporate_action_stats()

        # Verify storage stats
        assert storage_stats['total_rows'] == 3
        assert storage_stats['unique_symbols'] == 2
        assert len(storage_stats['by_type']) == 2

        # Verify database contents directly
        from hrp.data.db import get_db
        db = get_db()

        aapl_count = db.fetchone("SELECT COUNT(*) FROM corporate_actions WHERE symbol = 'AAPL'")
        assert aapl_count[0] == 2

        msft_count = db.fetchone("SELECT COUNT(*) FROM corporate_actions WHERE symbol = 'MSFT'")
        assert msft_count[0] == 1

    def test_incremental_ingestion(self, test_db):
        """Test incremental ingestion over time."""
        # First batch
        mock_actions_batch1 = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 3, 15)],
            'action_type': ['dividend'],
            'value': [0.23],
            'source': ['yfinance'],
        })

        # Second batch (different date)
        mock_actions_batch2 = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [date(2023, 6, 15)],
            'action_type': ['dividend'],
            'value': [0.24],
            'source': ['yfinance'],
        })

        with patch('hrp.data.ingestion.corporate_actions.YFinanceSource') as mock_source_class:
            # First ingestion
            mock_source = MagicMock()
            mock_source.get_corporate_actions.return_value = mock_actions_batch1
            mock_source_class.return_value = mock_source

            ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 1, 1),
                end=date(2023, 3, 31),
                source='yfinance',
            )

            # Second ingestion
            mock_source.get_corporate_actions.return_value = mock_actions_batch2

            ingest_corporate_actions(
                symbols=['AAPL'],
                start=date(2023, 4, 1),
                end=date(2023, 12, 31),
                source='yfinance',
            )

        # Verify both actions are stored
        stats = get_corporate_action_stats()
        assert stats['total_rows'] == 2
        assert stats['unique_symbols'] == 1

        from hrp.data.db import get_db
        db = get_db()
        actions = db.fetchall("SELECT date, factor FROM corporate_actions WHERE symbol = 'AAPL' ORDER BY date")
        assert len(actions) == 2
        assert actions[0][0] == date(2023, 3, 15)
        assert actions[1][0] == date(2023, 6, 15)
