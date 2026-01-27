"""Tests for JobResult dataclasses."""

import pytest

from hrp.agents.job_results import (
    JobResult,
    PriceIngestionResult,
    UniverseUpdateResult,
    FundamentalsIngestionResult,
)


class TestJobResult:
    """Tests for base JobResult dataclass."""

    def test_job_result_creation(self):
        """Test creating JobResult with all fields."""
        result = JobResult(
            status="success",
            records_fetched=100,
            records_inserted=95,
            symbols_success=10,
            symbols_failed=0,
            failed_symbols=[],
            error=None,
        )

        assert result.status == "success"
        assert result.records_fetched == 100
        assert result.records_inserted == 95
        assert result.symbols_success == 10
        assert result.symbols_failed == 0

    def test_job_result_defaults(self):
        """Test JobResult default values."""
        result = JobResult(status="success")

        assert result.records_fetched == 0
        assert result.records_inserted == 0
        assert result.symbols_success == 0
        assert result.symbols_failed == 0
        assert result.failed_symbols == []
        assert result.error is None

    def test_job_result_failed(self):
        """Test JobResult with failure status."""
        result = JobResult(
            status="failed",
            error="Connection timeout",
        )

        assert result.status == "failed"
        assert result.error == "Connection timeout"

    def test_job_result_dictionary_style_access(self):
        """Test JobResult supports dictionary-style access for backward compatibility."""
        result = JobResult(
            status="success",
            records_fetched=100,
            records_inserted=95,
        )

        # Dictionary-style access should work
        assert result["status"] == "success"
        assert result["records_fetched"] == 100
        assert result["records_inserted"] == 95

    def test_job_result_get_method(self):
        """Test JobResult supports get() method for backward compatibility."""
        result = JobResult(
            status="success",
            records_fetched=100,
        )

        # get() method should work
        assert result.get("status") == "success"
        assert result.get("records_fetched") == 100
        assert result.get("nonexistent") is None
        assert result.get("nonexistent", "default") == "default"


class TestPriceIngestionResult:
    """Tests for PriceIngestionResult dataclass."""

    def test_price_ingestion_result_defaults(self):
        """Test PriceIngestionResult inherits from JobResult."""
        result = PriceIngestionResult(status="success")

        assert result.status == "success"
        assert result.records_fetched == 0
        assert result.fallback_used == 0  # Specific to PriceIngestionResult

    def test_price_ingestion_result_with_fallback(self):
        """Test PriceIngestionResult with fallback tracking."""
        result = PriceIngestionResult(
            status="success",
            records_fetched=100,
            records_inserted=95,
            symbols_success=9,
            symbols_failed=1,
            failed_symbols=["TICKER"],
            fallback_used=1,
        )

        assert result.fallback_used == 1
        assert result.symbols_failed == 1


class TestUniverseUpdateResult:
    """Tests for UniverseUpdateResult dataclass."""

    def test_universe_update_result(self):
        """Test UniverseUpdateResult with universe-specific fields."""
        result = UniverseUpdateResult(
            status="success",
            records_fetched=500,
            records_inserted=450,
            symbols_added=5,
            symbols_removed=3,
            symbols_excluded=42,
            exclusion_breakdown={"financials": 10, "reits": 5, "penny_stocks": 27},
        )

        assert result.symbols_added == 5
        assert result.symbols_removed == 3
        assert result.symbols_excluded == 42
        assert result.exclusion_breakdown["financials"] == 10

    def test_universe_update_result_defaults(self):
        """Test UniverseUpdateResult defaults."""
        result = UniverseUpdateResult(status="success")

        assert result.symbols_added == 0
        assert result.symbols_removed == 0
        assert result.symbols_excluded == 0
        assert result.exclusion_breakdown == {}


class TestFundamentalsIngestionResult:
    """Tests for FundamentalsIngestionResult dataclass."""

    def test_fundamentals_ingestion_result(self):
        """Test FundamentalsIngestionResult with fundamentals-specific fields."""
        result = FundamentalsIngestionResult(
            status="success",
            records_fetched=200,
            records_inserted=180,
            fallback_used=10,
            pit_violations_filtered=5,
        )

        assert result.fallback_used == 10
        assert result.pit_violations_filtered == 5

    def test_fundamentals_ingestion_result_defaults(self):
        """Test FundamentalsIngestionResult defaults."""
        result = FundamentalsIngestionResult(status="success")

        assert result.fallback_used == 0
        assert result.pit_violations_filtered == 0
