"""Tests for sector data ingestion."""

import pytest
from unittest.mock import Mock, patch

from hrp.data.ingestion.sectors import (
    SectorIngestionJob,
    SIC_TO_GICS_MAPPING,
    fetch_sector_from_polygon,
    fetch_sector_from_yahoo,
)


class TestSicToGicsMapping:
    """Tests for SIC to GICS sector mapping."""

    def test_technology_mapping(self):
        """Technology SIC codes map to Technology sector."""
        # Software
        assert SIC_TO_GICS_MAPPING.get("7372") == "Technology"
        # Computer equipment
        assert SIC_TO_GICS_MAPPING.get("3571") == "Technology"

    def test_healthcare_mapping(self):
        """Healthcare SIC codes map to Healthcare sector."""
        # Pharmaceutical
        assert SIC_TO_GICS_MAPPING.get("2834") == "Healthcare"

    def test_unknown_sic_returns_none(self):
        """Unknown SIC code returns None."""
        assert SIC_TO_GICS_MAPPING.get("9999") is None


class TestFetchSectorFromPolygon:
    """Tests for Polygon sector fetching."""

    @patch("hrp.data.ingestion.sectors.requests.get")
    def test_successful_fetch(self, mock_get):
        """Successful Polygon API call returns sector data."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "results": {
                    "sic_code": "7372",
                    "sic_description": "Prepackaged Software",
                }
            },
        )

        result = fetch_sector_from_polygon("AAPL")

        assert result is not None
        assert result["sic_code"] == "7372"
        assert result["sector"] == "Technology"

    @patch("hrp.data.ingestion.sectors.requests.get")
    def test_api_failure_returns_none(self, mock_get):
        """API failure returns None."""
        mock_get.return_value = Mock(status_code=500)

        result = fetch_sector_from_polygon("AAPL")
        assert result is None


class TestFetchSectorFromYahoo:
    """Tests for Yahoo Finance fallback."""

    @patch("hrp.data.ingestion.sectors.yf.Ticker")
    def test_successful_fetch(self, mock_ticker):
        """Successful Yahoo fetch returns sector data."""
        mock_ticker.return_value.info = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        result = fetch_sector_from_yahoo("AAPL")

        assert result is not None
        assert result["sector"] == "Technology"
        assert result["industry"] == "Consumer Electronics"

    @patch("hrp.data.ingestion.sectors.yf.Ticker")
    def test_missing_sector_returns_unknown(self, mock_ticker):
        """Missing sector info returns Unknown."""
        mock_ticker.return_value.info = {}

        result = fetch_sector_from_yahoo("AAPL")
        assert result["sector"] == "Unknown"


class TestSectorIngestionJob:
    """Tests for SectorIngestionJob."""

    @patch("hrp.data.ingestion.sectors.fetch_sector_from_polygon")
    @patch("hrp.data.ingestion.sectors.get_db")
    def test_execute_updates_symbols(self, mock_db, mock_polygon):
        """Job execute updates sector in symbols table."""
        mock_polygon.return_value = {
            "sic_code": "7372",
            "sector": "Technology",
            "industry": "Software",
        }
        mock_db.return_value.fetchall.return_value = [("AAPL",)]

        job = SectorIngestionJob(symbols=["AAPL"])
        result = job.execute()

        assert result["symbols_updated"] == 1

    @patch("hrp.data.ingestion.sectors.fetch_sector_from_polygon")
    @patch("hrp.data.ingestion.sectors.fetch_sector_from_yahoo")
    @patch("hrp.data.ingestion.sectors.get_db")
    def test_execute_falls_back_to_yahoo(self, mock_db, mock_yahoo, mock_polygon):
        """Job execute falls back to Yahoo when Polygon fails."""
        mock_polygon.return_value = None
        mock_yahoo.return_value = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_db.return_value.fetchall.return_value = [("AAPL",)]

        job = SectorIngestionJob(symbols=["AAPL"])
        result = job.execute()

        assert result["symbols_updated"] == 1
        mock_yahoo.assert_called_once()
