"""
Pytest Configuration and Shared Fixtures for E2E Tests

This module provides shared fixtures and configuration for end-to-end
testing of HRP agent workflows.
"""

import pytest
import os
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Ensure tests can import hrp modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        'markers',
        'e2e: Marks tests as end-to-end tests (slow, require full stack)'
    )
    config.addinivalue_line(
        'markers',
        'benchmark: Marks tests as performance benchmarks'
    )
    config.addinivalue_line(
        'markers',
        'integration: Marks tests as integration tests (require external services)'
    )


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(scope='session')
def test_environment():
    """
    Set up test environment for E2E tests.

    Creates temporary directories and configures test database.
    """
    temp_dir = tempfile.mkdtemp(prefix='hrp_e2e_')

    # Configure test environment variables
    original_env = {
        'HRP_DB_PATH': os.environ.get('HRP_DB_PATH'),
        'HRP_LOG_PATH': os.environ.get('HRP_LOG_PATH'),
    }

    os.environ['HRP_DB_PATH'] = str(Path(temp_dir) / 'test.duckdb')
    os.environ['HRP_LOG_PATH'] = str(Path(temp_dir) / 'logs')

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Restore environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def test_db_path(test_environment):
    """Provide path to test database."""
    return Path(test_environment) / 'test.duckdb'


@pytest.fixture
def clean_test_db(test_db_path):
    """
    Provide clean test database for each test.

    Database is created fresh before each test and cleaned up after.
    """
    # Import here to avoid import errors if modules not available
    try:
        from hrp.data.db import get_connection

        # Create database schema
        conn = get_connection(str(test_db_path))
        cursor = conn.cursor()

        # Create tables (simplified for E2E tests)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id VARCHAR PRIMARY KEY,
                status VARCHAR,
                title VARCHAR,
                thesis TEXT,
                prediction TEXT,
                falsification TEXT,
                created_at TIMESTAMP,
                actor VARCHAR
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id VARCHAR PRIMARY KEY,
                hypothesis_id VARCHAR,
                experiment_type VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineage_events (
                id VARCHAR PRIMARY KEY,
                event_type VARCHAR,
                hypothesis_id VARCHAR,
                experiment_id VARCHAR,
                actor VARCHAR,
                timestamp TIMESTAMP,
                metadata JSON
            )
        """)

        conn.close()

        yield test_db_path

    except ImportError:
        # If db module not available, just yield the path
        yield test_db_path

    finally:
        # Cleanup database file
        if test_db_path.exists():
            test_db_path.unlink()


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def sample_hypothesis_data():
    """Sample hypothesis data for testing."""
    return {
        'id': 'HYP-2026-TEST-001',
        'status': 'draft',
        'title': 'Test Hypothesis: Momentum Predicts Returns',
        'thesis': 'Stocks with high 12-month momentum continue to outperform',
        'prediction': 'Top decile momentum stocks beat SPY by 3% annually',
        'falsification': 'Sharpe ratio < 0.5 or IC < 0.02',
        'created_at': datetime.now(),
        'actor': 'test:e2e',
    }


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        'id': 'EXP-2026-TEST-001',
        'hypothesis_id': 'HYP-2026-TEST-001',
        'experiment_type': 'walk_forward_validation',
        'status': 'completed',
        'metrics': {
            'sharpe': 1.5,
            'ic': 0.05,
            'max_drawdown': 0.12,
            'stability_score': 0.8,
        },
        'params': {
            'model_type': 'ridge',
            'n_folds': 5,
            'window_type': 'expanding',
        },
        'created_at': datetime.now(),
    }


@pytest.fixture
def sample_symbols():
    """Standard symbol sets for testing."""
    return {
        'small': ['AAPL', 'MSFT', 'GOOGL'],
        'medium': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'],
        'large': [f'STOCK{i:03d}' for i in range(1, 101)],
    }


@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start='2023-01-01', periods=252)
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for symbol in symbols:
        for date in dates:
            base_price = 100.0 if symbol == 'AAPL' else (150.0 if symbol == 'MSFT' else 120.0)
            data.append({
                'symbol': symbol,
                'date': date,
                'open': base_price + np.random.randn() * 2,
                'high': base_price + np.random.randn() * 2 + 1,
                'low': base_price + np.random.randn() * 2 - 1,
                'close': base_price + np.random.randn() * 2,
                'volume': np.random.randint(1000000, 10000000),
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_data():
    """Sample feature data for testing."""
    import pandas as pd
    import numpy as np

    symbols = ['AAPL', 'MSFT', 'GOOGL']
    features = ['momentum_20d', 'volatility_60d', 'rsi_14d']
    as_of_date = date.today()

    data = {
        'symbol': symbols,
        'date': [as_of_date] * len(symbols),
        'momentum_20d': np.random.randn(len(symbols)) * 0.1,
        'volatility_60d': np.random.rand(len(symbols)) * 0.2 + 0.1,
        'rsi_14d': np.random.rand(len(symbols)) * 100,
    }

    return pd.DataFrame(data)


# =============================================================================
# Mock Platform API Fixture
# =============================================================================

@pytest.fixture
def mock_platform_api_factory():
    """
    Factory for creating mock PlatformAPI instances.

    Returns a function that creates configured mock instances.
    """
    def _create_mock(
        hypothesis_data=None,
        experiment_data=None,
        price_data=None,
        feature_data=None,
    ):
        """Create a configured mock PlatformAPI."""
        mock = Mock()

        # Configure hypothesis methods
        mock.create_hypothesis.return_value = 'HYP-2026-TEST-001'
        mock.get_hypothesis.return_value = hypothesis_data or {
            'id': 'HYP-2026-TEST-001',
            'status': 'draft',
            'title': 'Test Hypothesis',
        }
        mock.update_hypothesis.return_value = True
        mock.list_hypotheses.return_value = []

        # Configure experiment methods
        mock.run_backtest.return_value = 'EXP-2026-TEST-001'
        mock.get_experiment.return_value = experiment_data or {
            'id': 'EXP-2026-TEST-001',
            'metrics': {'sharpe': 1.5},
        }
        mock.train_ml_model.return_value = 'EXP-2026-TEST-002'
        mock.run_walk_forward_validation.return_value = 'EXP-2026-TEST-003'

        # Configure data methods
        mock.get_prices.return_value = price_data
        mock.get_features.return_value = feature_data
        mock.get_available_features.return_value = [
            'momentum_20d',
            'volatility_60d',
            'rsi_14d',
        ]
        mock.get_universe.return_value = ['AAPL', 'MSFT', 'GOOGL']

        # Configure lineage methods
        mock.log_lineage_event.return_value = 'EVT-001'
        mock.get_lineage.return_value = []

        # Configure quality methods
        mock.run_quality_checks.return_value = {
            'health_score': 95,
            'critical_issues': 0,
            'warning_issues': 1,
            'passed': True,
        }

        return mock

    return _create_mock


# =============================================================================
# Test Context Manager
# =============================================================================

@pytest.fixture
def test_context():
    """
    Context manager for test execution tracking.

    Helps track test execution flow for debugging.
    """
    from contextlib import contextmanager

    @contextmanager
    def _context(test_name):
        print(f"\n{'='*60}")
        print(f"Starting test: {test_name}")
        print(f"{'='*60}")
        yield
        print(f"{'='*60}")
        print(f"Completed test: {test_name}")
        print(f"{'='*60}\n")

    return _context


# =============================================================================
# Performance Monitoring
# =============================================================================

@pytest.fixture
def performance_monitor():
    """
    Monitor and report test performance metrics.

    Tracks execution time for performance assertions.
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def _monitor(operation_name):
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f"\n[Performance] {operation_name}: {elapsed:.3f}s")

    return _monitor


# =============================================================================
# Test Data Generators
# =============================================================================

@pytest.fixture
def hypothesis_generator():
    """Generate sequential hypothesis IDs for testing."""
    counter = [0]

    def _generate(title=None, status='draft'):
        counter[0] += 1
        return {
            'id': f'HYP-2026-GEN-{counter[0]:03d}',
            'title': title or f'Generated Hypothesis {counter[0]}',
            'status': status,
            'created_at': datetime.now(),
        }

    return _generate


@pytest.fixture
def experiment_generator():
    """Generate sequential experiment IDs for testing."""
    counter = [0]

    def _generate(hypothesis_id, experiment_type='backtest'):
        counter[0] += 1
        return {
            'id': f'EXP-2026-GEN-{counter[0]:03d}',
            'hypothesis_id': hypothesis_id,
            'experiment_type': experiment_type,
            'status': 'completed',
            'created_at': datetime.now(),
        }

    return _generate


# =============================================================================
# Assertion Helpers
# =============================================================================

@pytest.fixture
def assert_valid_hypothesis():
    """Assert hypothesis data structure is valid."""
    def _assert(data):
        assert 'id' in data
        assert data['id'].startswith('HYP-')
        assert 'status' in data
        assert data['status'] in ['draft', 'testing', 'validated', 'rejected', 'deployed']
        assert 'title' in data
        assert 'thesis' in data

    return _assert


@pytest.fixture
def assert_valid_experiment():
    """Assert experiment data structure is valid."""
    def _assert(data):
        assert 'id' in data
        assert data['id'].startswith('EXP-')
        assert 'hypothesis_id' in data
        assert 'experiment_type' in data
        assert 'status' in data

    return _assert


@pytest.fixture
def assert_valid_lineage_event():
    """Assert lineage event structure is valid."""
    def _assert(data):
        assert 'event_type' in data
        assert 'timestamp' in data
        assert 'actor' in data
        assert data['actor'].startswith('agent:') or data['actor'] == 'user'

    return _assert
