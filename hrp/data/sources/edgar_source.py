"""
EDGAR (SEC) filing data source.

Fetches SEC filings (10-K, 10-Q, 8-K) for sentiment analysis.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import pandas as pd
from loguru import logger

from hrp.data.sources.base import DataSource


class EDGARSource(DataSource):
    """Fetches SEC filings from EDGAR database.

    Supports:
        - 10-K: Annual reports
        - 10-Q: Quarterly reports
        - 8-K: Current reports

    Rate limited to 10 requests/second per SEC requirements.
    """

    BASE_URL = "https://www.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/cgi-bin/browse-edgar"
    FILINGS_URL = f"{BASE_URL}/Archives/edgar/data"

    def __init__(self, user_agent: str = None):
        """Initialize EDGAR source.

        Args:
            user_agent: User agent string (required by SEC). Format: "Name Email"
        """
        self.user_agent = user_agent or "HRP Platform contact@hrpplatform.com"
        self.client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
        )
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 10 requests/second max

    def _rate_limit(self) -> None:
        """Enforce SEC rate limiting (10 requests/second)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_company_ticker(self, ticker: str) -> Optional[dict[str, Any]]:
        """Get SEC CIK and company info for ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dict with CIK and company info, or None if not found
        """
        self._rate_limit()

        # SEC CIK lookup
        params = {
            "action": "getcompany",
            "CIK": ticker,
            "type": "",
            "dateb": "",
            "owner": "exclude",
            "count": "",
            "output": "atom",
        }

        try:
            response = self.client.get(self.SUBMISSIONS_URL, params=params)
            response.raise_for_status()
            # Parse for CIK from response
            content = response.text
            if "CIK" in content:
                # Extract CIK from atom feed
                import re
                match = re.search(r"CIK=([0-9]+)", content)
                if match:
                    cik = match.group(1).zfill(10)  # Pad to 10 digits
                    return {"ticker": ticker.upper(), "cik": cik}
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")

        return None

    def get_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types (default: ["10-K", "10-Q", "8-K"])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max number of filings to return

        Returns:
            DataFrame with columns: ticker, cik, filing_type, filing_date,
                accession_number, document_url, filing_url
        """
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        company_info = self.get_company_ticker(ticker)
        if not company_info:
            logger.warning(f"Company not found for ticker: {ticker}")
            return pd.DataFrame()

        cik = company_info["cik"]

        self._rate_limit()

        # Get company submissions
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": ",".join(filing_types),
            "dateb": end_date or "",
            "owner": "exclude",
            "count": str(limit),
            "output": "atom",
        }

        try:
            response = self.client.get(self.SUBMISSIONS_URL, params=params)
            response.raise_for_status()

            # Parse atom feed
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.text)
            filings = []

            for entry in root.findall(
                "{http://www.w3.org/2005/Atom}entry"
            ):
                # Extract filing info
                filing_type_elem = entry.find(
                    "{http://www.w3.org/2005/Atom}category"
                )
                if filing_type_elem is not None:
                    filing_type = filing_type_elem.get("term")

                    filing_date_elem = entry.find(
                        "{http://www.w3.org/2005/Atom}updated"
                    )
                    filing_date = (
                        filing_date_elem.text[:10]
                        if filing_date_elem is not None
                        else None
                    )

                    # Filter by start_date if provided
                    if start_date and filing_date and filing_date < start_date:
                        continue

                    accession_number_elem = entry.find(
                        "{http://www.w3.org/2005/Atom}id"
                    )
                    if accession_number_elem is not None:
                        accession_url = accession_number_elem.text
                        # Extract accession number from URL
                        accession_number = accession_url.split("accession-number=")[
                            -1
                        ]

                        # Build URLs
                        doc_num = accession_number.replace("-", "")
                        document_url = (
                            f"{self.FILINGS_URL}/{cik}/{doc_num}/{doc_num}.txt"
                        )
                        filing_url = accession_url

                        filings.append(
                            {
                                "ticker": ticker.upper(),
                                "cik": cik,
                                "filing_type": filing_type,
                                "filing_date": filing_date,
                                "accession_number": accession_number,
                                "document_url": document_url,
                                "filing_url": filing_url,
                            }
                        )

            return pd.DataFrame(filings)

        except Exception as e:
            logger.error(f"Failed to get filings for {ticker}: {e}")
            return pd.DataFrame()

    def get_filing_text(self, document_url: str) -> Optional[str]:
        """Get full text of a filing.

        Args:
            document_url: URL to the filing document

        Returns:
            Filing text, or None if fetch fails
        """
        self._rate_limit()

        try:
            response = self.client.get(document_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get filing text from {document_url}: {e}")
            return None

    def fetch_filings_data(
        self,
        tickers: list[str],
        filing_types: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit_per_ticker: int = 10,
    ) -> pd.DataFrame:
        """Fetch filings for multiple tickers.

        Args:
            tickers: List of ticker symbols
            filing_types: List of filing types
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit_per_ticker: Max filings per ticker

        Returns:
            DataFrame with filings for all tickers
        """
        all_filings = []

        for ticker in tickers:
            filings = self.get_filings(
                ticker=ticker,
                filing_types=filing_types,
                start_date=start_date,
                end_date=end_date,
                limit=limit_per_ticker,
            )
            all_filings.append(filings)

        if all_filings:
            return pd.concat(all_filings, ignore_index=True)
        return pd.DataFrame()

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
