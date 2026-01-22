"""
Comprehensive tests for input validation functions.

Tests cover all validators:
- Symbol validation (against universe whitelist)
- Date validation (not future, valid format)
- Date range validation
- Numeric parameter validation (positive integers)
- String parameter validation (non-empty)
"""

import os
import tempfile
from datetime import date, datetime, timedelta

import pytest

from hrp.api.validators import (
    validate_date,
    validate_date_range,
    validate_non_empty_string,
    validate_positive_int,
    validate_symbols,
)
from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db():
    """Create a temporary DuckDB database with schema for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Delete the empty file so DuckDB can create a fresh database
    os.remove(db_path)

    # Reset the singleton to ensure fresh state
    DatabaseManager.reset()

    # Initialize schema
    create_tables(db_path)

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if os.path.exists(db_path):
        os.remove(db_path)
    # Also remove any wal/tmp files
    for ext in [".wal", ".tmp", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


@pytest.fixture
def test_db_manager(test_db):
    """Create a DatabaseManager instance with a test database."""
    return DatabaseManager(db_path=test_db)


@pytest.fixture
def populated_db(test_db_manager):
    """
    Populate the test database with sample universe data.

    Returns the DatabaseManager instance for convenience.
    """
    # Insert sample universe data
    test_db_manager.execute(
        """
        INSERT INTO universe (symbol, date, in_universe, sector, market_cap)
        VALUES
            ('AAPL', '2023-01-01', TRUE, 'Technology', 2500000000000),
            ('MSFT', '2023-01-01', TRUE, 'Technology', 2400000000000),
            ('GOOGL', '2023-01-01', TRUE, 'Technology', 1500000000000),
            ('JPM', '2023-01-01', FALSE, 'Financials', 400000000000),
            ('AAPL', '2023-06-01', TRUE, 'Technology', 2800000000000),
            ('MSFT', '2023-06-01', TRUE, 'Technology', 2600000000000),
            ('TSLA', '2023-06-01', TRUE, 'Consumer Discretionary', 800000000000)
        """
    )

    return test_db_manager


# =============================================================================
# Symbol Validation Tests
# =============================================================================


class TestValidateSymbols:
    """Tests for validate_symbols() function."""

    def test_valid_symbols_pass(self, populated_db):
        """Valid symbols from universe should pass validation."""
        # Should not raise any exception
        validate_symbols(['AAPL', 'MSFT', 'GOOGL'], db=populated_db)

    def test_valid_single_symbol(self, populated_db):
        """Single valid symbol should pass validation."""
        validate_symbols(['AAPL'], db=populated_db)

    def test_valid_symbols_with_date(self, populated_db):
        """Valid symbols at specific date should pass validation."""
        # AAPL and MSFT are in universe on 2023-01-01
        validate_symbols(['AAPL', 'MSFT'], as_of_date=date(2023, 1, 1), db=populated_db)

    def test_empty_symbols_list_raises_error(self, populated_db):
        """Empty symbols list should raise ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            validate_symbols([], db=populated_db)

    def test_invalid_symbol_raises_error(self, populated_db):
        """Invalid symbol not in universe should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid symbols not found in universe: INVALID"):
            validate_symbols(['INVALID'], db=populated_db)

    def test_mixed_valid_invalid_symbols_raises_error(self, populated_db):
        """Mix of valid and invalid symbols should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid symbols not found in universe"):
            validate_symbols(['AAPL', 'INVALID', 'FAKE'], db=populated_db)

    def test_symbol_not_in_universe_at_date(self, populated_db):
        """Symbol not in universe at specific date should raise ValueError."""
        # TSLA is only in universe on 2023-06-01, not 2023-01-01
        with pytest.raises(ValueError, match="Invalid symbols not in universe as of 2023-01-01: TSLA"):
            validate_symbols(['TSLA'], as_of_date=date(2023, 1, 1), db=populated_db)

    def test_excluded_symbol_raises_error(self, populated_db):
        """Symbol explicitly excluded from universe should raise ValueError."""
        # JPM has in_universe = FALSE
        with pytest.raises(ValueError, match="Invalid symbols not found in universe: JPM"):
            validate_symbols(['JPM'], db=populated_db)

    def test_multiple_invalid_symbols_in_error_message(self, populated_db):
        """Error message should list all invalid symbols."""
        with pytest.raises(ValueError, match="INVALID1.*INVALID2"):
            validate_symbols(['INVALID1', 'INVALID2'], db=populated_db)


# =============================================================================
# Date Validation Tests
# =============================================================================


class TestValidateDate:
    """Tests for validate_date() function."""

    def test_past_date_passes(self):
        """Past dates should pass validation."""
        past_date = date(2020, 1, 1)
        validate_date(past_date)

    def test_today_passes(self):
        """Today's date should pass validation."""
        today = date.today()
        validate_date(today)

    def test_future_date_raises_error(self):
        """Future dates should raise ValueError."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="date cannot be in the future"):
            validate_date(future_date)

    def test_far_future_date_raises_error(self):
        """Far future dates should raise ValueError."""
        future_date = date(2099, 12, 31)
        with pytest.raises(ValueError, match="date cannot be in the future"):
            validate_date(future_date)

    def test_custom_parameter_name_in_error(self):
        """Custom parameter name should appear in error message."""
        future_date = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="start_date cannot be in the future"):
            validate_date(future_date, parameter_name="start_date")

    def test_invalid_type_raises_error(self):
        """Non-date type should raise ValueError."""
        with pytest.raises(ValueError, match="date must be a date object, got str"):
            validate_date("2023-01-01")

    def test_datetime_raises_error(self):
        """datetime object should raise TypeError when comparing (datetime is subclass of date)."""
        dt = datetime.now()
        # datetime is a subclass of date, so isinstance check passes
        # but comparison with date.today() raises TypeError
        with pytest.raises(TypeError, match="can't compare datetime.datetime to datetime.date"):
            validate_date(dt)

    def test_none_raises_error(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="date must be a date object, got NoneType"):
            validate_date(None)


# =============================================================================
# Date Range Validation Tests
# =============================================================================


class TestValidateDateRange:
    """Tests for validate_date_range() function."""

    def test_valid_date_range_passes(self):
        """Valid date range should pass validation."""
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        validate_date_range(start, end)

    def test_same_start_and_end_passes(self):
        """Start and end on same date should pass validation."""
        same_date = date(2023, 6, 1)
        validate_date_range(same_date, same_date)

    def test_one_day_range_passes(self):
        """Single day range should pass validation."""
        start = date(2023, 6, 1)
        end = date(2023, 6, 2)
        validate_date_range(start, end)

    def test_start_after_end_raises_error(self):
        """Start date after end date should raise ValueError."""
        start = date(2023, 12, 31)
        end = date(2023, 1, 1)
        with pytest.raises(ValueError, match="start date must be <= end date"):
            validate_date_range(start, end)

    def test_future_start_date_raises_error(self):
        """Future start date should raise ValueError."""
        start = date.today() + timedelta(days=1)
        end = date.today() + timedelta(days=10)
        with pytest.raises(ValueError, match="start date cannot be in the future"):
            validate_date_range(start, end)

    def test_future_end_date_raises_error(self):
        """Future end date should raise ValueError."""
        start = date.today()
        end = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="end date cannot be in the future"):
            validate_date_range(start, end)

    def test_both_future_dates_raises_error(self):
        """Both future dates should raise ValueError."""
        start = date.today() + timedelta(days=1)
        end = date.today() + timedelta(days=10)
        with pytest.raises(ValueError, match="start date cannot be in the future"):
            validate_date_range(start, end)


# =============================================================================
# Positive Integer Validation Tests
# =============================================================================


class TestValidatePositiveInt:
    """Tests for validate_positive_int() function."""

    def test_positive_integer_passes(self):
        """Positive integers should pass validation."""
        validate_positive_int(1)
        validate_positive_int(10)
        validate_positive_int(1000)

    def test_zero_without_allow_zero_raises_error(self):
        """Zero should raise ValueError when allow_zero is False."""
        with pytest.raises(ValueError, match="value must be positive, got 0"):
            validate_positive_int(0)

    def test_zero_with_allow_zero_passes(self):
        """Zero should pass validation when allow_zero is True."""
        validate_positive_int(0, allow_zero=True)

    def test_negative_integer_raises_error(self):
        """Negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="value must be positive, got -1"):
            validate_positive_int(-1)

    def test_negative_with_allow_zero_raises_error(self):
        """Negative integers should raise ValueError even with allow_zero."""
        with pytest.raises(ValueError, match="value must be non-negative, got -5"):
            validate_positive_int(-5, allow_zero=True)

    def test_float_raises_error(self):
        """Float values should raise ValueError."""
        with pytest.raises(ValueError, match="value must be an integer, got float"):
            validate_positive_int(3.14)

    def test_string_raises_error(self):
        """String values should raise ValueError."""
        with pytest.raises(ValueError, match="value must be an integer, got str"):
            validate_positive_int("10")

    def test_none_raises_error(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="value must be an integer, got NoneType"):
            validate_positive_int(None)

    def test_bool_passes_as_int(self):
        """Boolean True (1) should pass as positive integer."""
        # In Python, bool is a subclass of int, so True == 1
        validate_positive_int(True)

    def test_custom_parameter_name_in_error(self):
        """Custom parameter name should appear in error message."""
        with pytest.raises(ValueError, match="limit must be positive, got -5"):
            validate_positive_int(-5, parameter_name="limit")


# =============================================================================
# Non-Empty String Validation Tests
# =============================================================================


class TestValidateNonEmptyString:
    """Tests for validate_non_empty_string() function."""

    def test_non_empty_string_passes(self):
        """Non-empty strings should pass validation."""
        validate_non_empty_string("AAPL")
        validate_non_empty_string("This is a valid string")
        validate_non_empty_string("123")

    def test_string_with_content_after_strip_passes(self):
        """String with whitespace but content should pass when strip=True."""
        validate_non_empty_string("  AAPL  ")
        validate_non_empty_string("\tMSFT\n")
        validate_non_empty_string("  content  ")

    def test_empty_string_raises_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="value cannot be empty"):
            validate_non_empty_string("")

    def test_whitespace_only_string_raises_error_with_strip(self):
        """Whitespace-only string should raise ValueError when strip=True."""
        with pytest.raises(ValueError, match="value cannot be empty or whitespace-only"):
            validate_non_empty_string("   ")

    def test_whitespace_only_string_raises_error_tabs(self):
        """Tab/newline-only string should raise ValueError when strip=True."""
        with pytest.raises(ValueError, match="value cannot be empty or whitespace-only"):
            validate_non_empty_string("\t\n\r")

    def test_whitespace_string_passes_without_strip(self):
        """Whitespace-only string should pass when strip=False."""
        validate_non_empty_string("   ", strip=False)

    def test_empty_string_raises_error_without_strip(self):
        """Empty string should raise ValueError even when strip=False."""
        with pytest.raises(ValueError, match="value cannot be empty"):
            validate_non_empty_string("", strip=False)

    def test_integer_raises_error(self):
        """Integer values should raise ValueError."""
        with pytest.raises(ValueError, match="value must be a string, got int"):
            validate_non_empty_string(123)

    def test_none_raises_error(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="value must be a string, got NoneType"):
            validate_non_empty_string(None)

    def test_list_raises_error(self):
        """List values should raise ValueError."""
        with pytest.raises(ValueError, match="value must be a string, got list"):
            validate_non_empty_string(["AAPL"])

    def test_custom_parameter_name_in_error(self):
        """Custom parameter name should appear in error message."""
        with pytest.raises(ValueError, match="title cannot be empty or whitespace-only"):
            validate_non_empty_string("   ", parameter_name="title")

    def test_special_characters_pass(self):
        """Strings with special characters should pass."""
        validate_non_empty_string("test@example.com")
        validate_non_empty_string("file-name_123.txt")
        validate_non_empty_string("!@#$%^&*()")


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidatorsIntegration:
    """Integration tests for validators working together."""

    def test_validators_with_realistic_api_inputs(self, populated_db):
        """Test validators with realistic API call inputs."""
        # Simulate typical get_prices() call validation
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)

        validate_symbols(symbols, db=populated_db)
        validate_date_range(start, end)

    def test_validators_catch_multiple_errors_separately(self, populated_db):
        """Test that validators catch different types of errors."""
        # Invalid symbols
        with pytest.raises(ValueError, match="Invalid symbols"):
            validate_symbols(['INVALID'], db=populated_db)

        # Future date
        with pytest.raises(ValueError, match="cannot be in the future"):
            validate_date(date.today() + timedelta(days=1))

        # Negative integer
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_int(-1, parameter_name="limit")

        # Empty string
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_non_empty_string("", parameter_name="title")

    def test_validators_with_hypothesis_creation_inputs(self):
        """Test validators with hypothesis creation parameters."""
        # Validate hypothesis fields
        validate_non_empty_string("Momentum Factor Test", parameter_name="title")
        validate_non_empty_string("High momentum stocks outperform", parameter_name="thesis")
        validate_non_empty_string("Top decile > SPY by 3%", parameter_name="prediction")
        validate_non_empty_string("Sharpe < SPY", parameter_name="falsification")

    def test_validators_with_list_hypotheses_inputs(self):
        """Test validators with list_hypotheses() limit parameter."""
        validate_positive_int(10, parameter_name="limit")
        validate_positive_int(100, parameter_name="limit")

        # Should fail for negative limit
        with pytest.raises(ValueError, match="limit must be positive"):
            validate_positive_int(-1, parameter_name="limit")

    def test_edge_case_date_boundaries(self):
        """Test date validators with boundary conditions."""
        # Today should pass
        validate_date(date.today())

        # Yesterday should pass
        validate_date(date.today() - timedelta(days=1))

        # Tomorrow should fail
        with pytest.raises(ValueError, match="cannot be in the future"):
            validate_date(date.today() + timedelta(days=1))

    def test_edge_case_integer_boundaries(self):
        """Test integer validators with boundary conditions."""
        # 1 should pass
        validate_positive_int(1)

        # 0 should pass with allow_zero
        validate_positive_int(0, allow_zero=True)

        # 0 should fail without allow_zero
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_int(0, allow_zero=False)

        # Very large integers should pass
        validate_positive_int(999999999)
