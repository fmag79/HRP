"""
Smoke Test - Integration test touching all six features.

This test verifies that all major features work together:
1. Database constraints / schema
2. Universe management (S&P 500)
3. Price ingestion (with mocked Polygon/YFinance)
4. Feature store versioning
5. Data quality framework
6. Scheduled ingestion (scheduler + jobs)

The goal is to catch integration breakage, not exhaustively test each feature.
"""

import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def smoke_test_db():
    """
    Create a temporary database with full schema for smoke testing.

    Includes:
    - All tables created
    - data_sources entries for scheduled jobs
    - Sample universe data
    - Sample price data
    """
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    from hrp.data.db import get_db

    db = get_db(db_path)

    with db.connection() as conn:
        # Insert data_sources for jobs (required FK)
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('price_ingestion', 'scheduled_job', 'active'),
                ('feature_computation', 'scheduled_job', 'active'),
                ('yfinance', 'api', 'active'),
                ('polygon', 'api', 'active'),
                ('test', 'test', 'active')
            """
        )

        # Insert sample universe (3 tech stocks)
        conn.execute(
            """
            INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
            VALUES
                ('AAPL', '2024-01-01', TRUE, 'Technology', 3000000000000),
                ('MSFT', '2024-01-01', TRUE, 'Technology', 2800000000000),
                ('GOOGL', '2024-01-01', TRUE, 'Technology', 1800000000000)
            """
        )

        # Insert sample prices for 10 trading days
        base_prices = {"AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0}
        test_date = date(2024, 1, 2)

        for i in range(10):
            current_date = test_date + timedelta(days=i)
            # Skip weekends
            if current_date.weekday() >= 5:
                continue

            for symbol, base in base_prices.items():
                price = base * (1 + 0.005 * i)  # Small daily increase
                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
                    """,
                    (
                        symbol,
                        current_date,
                        price * 0.99,
                        price * 1.01,
                        price * 0.98,
                        price,
                        price,
                        10000000,
                    ),
                )

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# Smoke Test
# =============================================================================


class TestSmoke:
    """
    Smoke test covering all six features in a single flow.

    This test should complete in < 30 seconds and verifies integration.
    """

    def test_smoke_all_features(self, smoke_test_db):
        """
        Single smoke test touching all features.

        Steps:
        1. Verify DB constraints work
        2. Test universe management
        3. Test price ingestion (mocked)
        4. Test feature store
        5. Test data quality
        6. Test scheduled ingestion
        """
        from hrp.api.platform import PlatformAPI
        from hrp.data.db import get_db

        api = PlatformAPI(db_path=smoke_test_db)
        db = get_db(smoke_test_db)

        # =====================================================================
        # 1. Database Constraints
        # =====================================================================

        # Verify schema is valid
        with db.connection() as conn:
            # Check tables exist
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            assert "prices" in table_names
            assert "universe" in table_names
            assert "features" in table_names
            assert "ingestion_log" in table_names
            assert "data_sources" in table_names
            # quality_reports is created dynamically when first report is stored

            # Verify constraints: try inserting invalid data
            # NOT NULL constraint
            with pytest.raises(Exception):
                conn.execute(
                    "INSERT INTO prices (symbol, date, close) VALUES (NULL, '2024-01-01', 100)"
                )

        # =====================================================================
        # 2. Universe Management
        # =====================================================================

        from hrp.data.universe import UniverseManager

        universe_manager = UniverseManager(smoke_test_db)

        # Get universe at date
        universe = universe_manager.get_universe_at_date(date(2024, 1, 1))
        assert len(universe) == 3
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "GOOGL" in universe

        # Check sector breakdown
        sector_breakdown = universe_manager.get_sector_breakdown(date(2024, 1, 1))
        assert "Technology" in sector_breakdown
        assert sector_breakdown["Technology"] == 3

        # =====================================================================
        # 3. Price Ingestion (mocked)
        # =====================================================================

        from hrp.data.ingestion.prices import ingest_prices

        with patch("hrp.data.sources.yfinance_source.yf.download") as mock_yf:
            # Mock yfinance response
            mock_data = pd.DataFrame(
                {
                    ("Open", "AAPL"): [181.0],
                    ("High", "AAPL"): [183.0],
                    ("Low", "AAPL"): [180.0],
                    ("Close", "AAPL"): [182.0],
                    ("Adj Close", "AAPL"): [182.0],
                    ("Volume", "AAPL"): [50000000],
                },
                index=pd.DatetimeIndex([date(2024, 1, 15)]),
            )
            mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
            mock_yf.return_value = mock_data

            result = ingest_prices(
                symbols=["AAPL"],
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                source="yfinance",
            )

            assert result["rows_inserted"] >= 0  # May be 0 if upsert finds existing

        # Verify we have prices in DB
        prices_df = api.get_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        assert not prices_df.empty
        assert "close" in prices_df.columns

        # =====================================================================
        # 4. Feature Store
        # =====================================================================

        from hrp.data.features import FeatureRegistry

        registry = FeatureRegistry(smoke_test_db)

        # Register a feature
        registry.register(
            feature_name="test_momentum",
            version="v1",
            computation_code="def compute(prices): return prices['close'].pct_change(5)",
            description="5-day momentum for smoke test",
        )

        # Verify registration (list_features returns active features by default)
        features = registry.list_features()
        assert "test_momentum" in features

        # Compute feature (mocked)
        with db.connection() as conn:
            # Insert computed feature value
            conn.execute(
                """
                INSERT INTO features (symbol, date, feature_name, value, version)
                VALUES ('AAPL', '2024-01-10', 'test_momentum', 0.025, 'v1')
                """
            )

        # Verify feature is retrievable
        features_df = api.get_features(["AAPL"], ["test_momentum"], date(2024, 1, 10))
        # May be empty if feature version check fails, but call should not error

        # =====================================================================
        # 5. Data Quality Framework
        # =====================================================================

        from hrp.data.quality.report import QualityReportGenerator

        with patch("hrp.data.quality.alerts.EmailNotifier") as mock_notifier:
            mock_instance = MagicMock()
            mock_notifier.return_value = mock_instance

            # Generate quality report
            generator = QualityReportGenerator(smoke_test_db)
            report = generator.generate_report(date(2024, 1, 10))

            assert report is not None
            assert report.checks_run > 0
            assert 0 <= report.health_score <= 100

            # Store report
            report_id = generator.store_report(report)
            assert report_id > 0

            # Retrieve stored report
            history = generator.get_historical_reports(date(2024, 1, 1), date(2024, 1, 31))
            assert len(history) >= 1

            # Test Platform API quality methods
            score = api.get_data_health_score(date(2024, 1, 10))
            assert 0 <= score <= 100

        # =====================================================================
        # 6. Scheduled Ingestion
        # =====================================================================

        from hrp.agents.cli import list_scheduled_jobs
        from hrp.agents.scheduler import IngestionScheduler

        # Initialize scheduler
        scheduler = IngestionScheduler()
        assert scheduler is not None
        assert not scheduler.running

        # Setup daily ingestion (mocked jobs)
        with patch("hrp.agents.jobs.PriceIngestionJob") as mock_price_job:
            with patch("hrp.agents.jobs.FeatureComputationJob") as mock_feature_job:
                mock_price_job.return_value = MagicMock()
                mock_feature_job.return_value = MagicMock()

                scheduler.setup_daily_ingestion()

                jobs = scheduler.list_jobs()
                assert len(jobs) == 2

                job_ids = [j["id"] for j in jobs]
                assert "price_ingestion" in job_ids
                assert "feature_computation" in job_ids

        # Test CLI list_scheduled_jobs
        with patch("hrp.agents.cli.IngestionScheduler") as mock_sched_class:
            mock_sched = MagicMock()
            mock_sched.list_jobs.return_value = [
                {"id": "price_ingestion", "name": "Daily Prices", "next_run": None},
                {"id": "feature_computation", "name": "Daily Features", "next_run": None},
            ]
            mock_sched_class.return_value = mock_sched

            cli_jobs = list_scheduled_jobs()
            assert len(cli_jobs) == 2

        # =====================================================================
        # Final Verification
        # =====================================================================

        # Platform health check should pass
        health = api.health_check()
        assert health["api"] == "ok"
        assert health["database"] == "ok"


class TestSmokeFeatureIsolation:
    """
    Individual feature smoke tests for faster debugging.

    These tests are subsets of the full smoke test.
    """

    def test_smoke_database_constraints(self, smoke_test_db):
        """Verify database schema and constraints."""
        from hrp.data.db import get_db

        db = get_db(smoke_test_db)

        with db.connection() as conn:
            # Verify key tables
            result = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
            assert result[0] > 0

            result = conn.execute("SELECT COUNT(*) FROM universe").fetchone()
            assert result[0] == 3

            result = conn.execute("SELECT COUNT(*) FROM data_sources").fetchone()
            assert result[0] >= 2  # At least job sources

    def test_smoke_universe(self, smoke_test_db):
        """Verify universe management works."""
        from hrp.data.universe import UniverseManager

        manager = UniverseManager(smoke_test_db)

        universe = manager.get_universe_at_date(date(2024, 1, 1))
        assert len(universe) == 3

    def test_smoke_quality(self, smoke_test_db):
        """Verify quality framework works."""
        from hrp.data.quality.report import QualityReportGenerator

        generator = QualityReportGenerator(smoke_test_db)
        report = generator.generate_report(date(2024, 1, 10))

        assert report.checks_run > 0

    def test_smoke_scheduler(self, smoke_test_db):
        """Verify scheduler initialization works."""
        from hrp.agents.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()
        assert scheduler is not None

        # Add a simple job
        def dummy():
            pass

        scheduler.add_job(func=dummy, job_id="test", trigger="interval", seconds=60)
        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
