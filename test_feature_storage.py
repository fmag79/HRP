#!/usr/bin/env python3
"""
Manual verification script for feature storage functionality.

To run this test:
1. Ensure you're in the HRP project root directory
2. Activate the virtual environment (if using one)
3. Install dependencies: pip install -r requirements.txt
4. Run: python3 test_feature_storage.py

This will:
- Compute features for test symbols
- Store them in the database with version information
- Retrieve and verify the stored data
"""

from datetime import date, timedelta
from loguru import logger

from hrp.data.features.computation import FeatureComputer, register_default_features
from hrp.data.schema import create_tables

def test_feature_storage():
    """Test computing and storing features."""
    logger.info("=" * 60)
    logger.info("FEATURE STORAGE VERIFICATION TEST")
    logger.info("=" * 60)

    # Initialize schema (safe - creates tables if not exist)
    logger.info("\n1. Ensuring database schema exists...")
    create_tables()

    # Register default features
    logger.info("2. Registering default features (momentum_20d, volatility_60d)...")
    register_default_features()

    # Initialize computer
    logger.info("3. Initializing feature computer...")
    computer = FeatureComputer()

    # Define test parameters
    symbols = ["AAPL", "MSFT"]
    end_date = date(2024, 1, 31)
    start_date = end_date - timedelta(days=5)
    dates = [start_date + timedelta(days=i) for i in range(6)]
    feature_names = ["momentum_20d", "volatility_60d"]

    logger.info(f"\n4. Computing and storing features:")
    logger.info(f"   Symbols: {symbols}")
    logger.info(f"   Dates: {start_date} to {end_date} ({len(dates)} days)")
    logger.info(f"   Features: {feature_names}")

    try:
        # Compute and store features
        stats = computer.compute_and_store_features(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
        )

        logger.info(f"\n5. Storage completed:")
        logger.info(f"   Features computed: {stats['features_computed']}")
        logger.info(f"   Rows stored: {stats['rows_stored']}")
        logger.info(f"   Versions: {stats['versions']}")

        # Verify data was stored by retrieving it
        logger.info("\n6. Verifying stored features...")
        stored = computer.get_stored_features(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
        )

        logger.info(f"   Retrieved {len(stored)} rows")
        logger.info(f"   Shape: {stored.shape}")
        logger.info(f"\n   Sample data (first 5 rows):")
        logger.info(f"\n{stored.head()}")

        # Check that we got non-empty data
        if stored.empty:
            logger.error("\n✗ FAILED: No data retrieved from database!")
            return False

        # Verify versions by querying DB directly
        logger.info("\n7. Verifying version information in database...")
        from hrp.data.db import get_db
        db = get_db()
        version_check = db.fetchdf("""
            SELECT DISTINCT feature_name, version, COUNT(*) as count
            FROM features
            WHERE symbol IN ('AAPL', 'MSFT')
              AND feature_name IN ('momentum_20d', 'volatility_60d')
            GROUP BY feature_name, version
            ORDER BY feature_name, version
        """)
        logger.info(f"\n   Versions in database:")
        logger.info(f"\n{version_check}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ FEATURE STORAGE TEST PASSED")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"\n✗ FEATURE STORAGE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_storage()
    exit(0 if success else 1)
