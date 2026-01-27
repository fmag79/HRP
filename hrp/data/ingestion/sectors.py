"""
Sector data ingestion from Polygon.io and Yahoo Finance.

Provides GICS sector classification for universe symbols.
"""

import os
from typing import Any

import requests
import yfinance as yf
from loguru import logger

from hrp.data.db import get_db
from hrp.agents.jobs import IngestionJob


# SIC code to GICS sector mapping (partial - common codes)
SIC_TO_GICS_MAPPING = {
    # Technology
    "7372": "Technology",  # Prepackaged Software
    "7371": "Technology",  # Computer Programming
    "3571": "Technology",  # Electronic Computers
    "3674": "Technology",  # Semiconductors
    "7370": "Technology",  # Computer Services
    "3661": "Technology",  # Telephone Equipment
    "3663": "Technology",  # Radio/TV Equipment
    # Healthcare
    "2834": "Healthcare",  # Pharmaceutical
    "3841": "Healthcare",  # Medical Instruments
    "8071": "Healthcare",  # Medical Labs
    "8011": "Healthcare",  # Offices of Physicians
    # Consumer Discretionary
    "5961": "Consumer Discretionary",  # Catalog/Mail-Order
    "5731": "Consumer Discretionary",  # Radio/TV Stores
    "5411": "Consumer Staples",  # Grocery Stores
    "5912": "Consumer Staples",  # Drug Stores
    # Financials
    "6022": "Financials",  # State Commercial Banks
    "6211": "Financials",  # Security Brokers
    # Industrials
    "3721": "Industrials",  # Aircraft
    "3711": "Industrials",  # Motor Vehicles
    # Energy
    "1311": "Energy",  # Crude Petroleum
    "2911": "Energy",  # Petroleum Refining
    # Materials
    "2800": "Materials",  # Chemicals
    # Utilities
    "4911": "Utilities",  # Electric Services
    # Communication Services
    "4813": "Communication Services",  # Telephone
    "7812": "Communication Services",  # Motion Pictures
    # Real Estate
    "6798": "Real Estate",  # REITs
}


def fetch_sector_from_polygon(symbol: str) -> dict[str, Any] | None:
    """
    Fetch sector data from Polygon.io Ticker Details API.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with sic_code, sector, industry or None on failure
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set")
        return None

    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": api_key}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logger.debug(f"Polygon API returned {response.status_code} for {symbol}")
            return None

        data = response.json()
        results = data.get("results", {})

        sic_code = results.get("sic_code", "")
        sic_description = results.get("sic_description", "")

        # Map SIC to GICS sector
        sector = SIC_TO_GICS_MAPPING.get(sic_code, "Unknown")

        return {
            "sic_code": sic_code,
            "sic_description": sic_description,
            "sector": sector,
            "industry": sic_description,  # Use SIC description as industry
        }

    except Exception as e:
        logger.warning(f"Polygon API error for {symbol}: {e}")
        return None


def fetch_sector_from_yahoo(symbol: str) -> dict[str, Any]:
    """
    Fetch sector data from Yahoo Finance (fallback).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with sector and industry
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

    except Exception as e:
        logger.warning(f"Yahoo Finance error for {symbol}: {e}")
        return {
            "sector": "Unknown",
            "industry": "Unknown",
        }


class SectorIngestionJob(IngestionJob):
    """
    Job to ingest sector data for universe symbols.

    Uses Polygon.io as primary source with Yahoo Finance fallback.
    """

    def __init__(self, symbols: list[str] | None = None):
        """
        Initialize sector ingestion job.

        Args:
            symbols: List of symbols to update (None = all universe)
        """
        super().__init__(job_id="sector_ingestion")
        self.symbols = symbols

    def execute(self) -> dict[str, Any]:
        """Execute sector ingestion (called by IngestionJob.run())."""
        db = get_db()

        # Get symbols to update
        if self.symbols:
            symbols = self.symbols
        else:
            result = db.fetchall("SELECT symbol FROM symbols")
            symbols = [r[0] for r in result]

        logger.info(f"Starting sector ingestion for {len(symbols)} symbols")

        updated = 0
        failed = 0

        for symbol in symbols:
            # Try Polygon first
            sector_data = fetch_sector_from_polygon(symbol)

            # Fallback to Yahoo
            if sector_data is None:
                sector_data = fetch_sector_from_yahoo(symbol)

            # Update database
            try:
                db.execute(
                    """
                    UPDATE symbols
                    SET sector = ?, industry = ?
                    WHERE symbol = ?
                    """,
                    (sector_data["sector"], sector_data.get("industry", "Unknown"), symbol),
                )
                updated += 1
                logger.debug(f"Updated {symbol}: {sector_data['sector']}")

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                failed += 1

        logger.info(f"Sector ingestion complete: {updated} updated, {failed} failed")

        return {
            "symbols_updated": updated,
            "symbols_failed": failed,
        }
