"""Robinhood API client for order execution.

This module provides authentication, order placement, and account management
for Robinhood trading. It handles rate limiting, error recovery, and
session management.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from decimal import Decimal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class RobinhoodConfig:
    """Robinhood API configuration."""

    username: str
    password: str
    mfa_code: Optional[str] = None
    paper_trading: bool = True
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.2  # Seconds between requests

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.username:
            raise ValueError("username is required")
        if not self.password:
            raise ValueError("password is required")


class RobinhoodError(Exception):
    """Base exception for Robinhood API errors."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RateLimitError(RobinhoodError):
    """Raised when rate limit is exceeded."""


class AuthenticationError(RobinhoodError):
    """Raised when authentication fails."""


class RobinhoodClient:
    """Robinhood API client with session management and rate limiting."""

    BASE_URL = "https://api.robinhood.com"
    LOGIN_URL = "https://api.robinhood.com/oauth2/token/"
    ORDERS_URL = "https://api.robinhood.com/orders/"
    POSITIONS_URL = "https://api.robinhood.com/positions/"
    ACCOUNT_URL = "https://api.robinhood.com/accounts/"

    def __init__(self, config: RobinhoodConfig) -> None:
        """Initialize Robinhood client.

        Args:
            config: Robinhood API configuration
        """
        self.config = config
        self._session: Optional[requests.Session] = None
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires: Optional[float] = None
        self._last_request_time: float = 0

        logger.info(
            f"Robinhood client initialized (paper_trading={config.paper_trading})"
        )

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with retry logic.

        Returns:
            Configured requests.Session
        """
        if self._session is None:
            session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            self._session = session

        return self._session

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        time_since_last = now - self._last_request_time

        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            data: Request body data
            params: Query parameters
            headers: Additional headers

        Returns:
            Response JSON data

        Raises:
            RobinhoodError: If request fails
        """
        self._rate_limit()

        session = self._get_session()

        # Add authorization header
        request_headers = headers or {}
        if self._access_token:
            request_headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            response = session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=self.config.timeout,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(method, url, data, params, headers)

            # Check for errors
            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            self._handle_error(e.response)

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {url}")
            raise RobinhoodError("Request timed out")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise RobinhoodError(f"Request failed: {e}")

    def _handle_error(self, response: requests.Response) -> None:
        """Handle API error response.

        Args:
            response: Failed HTTP response

        Raises:
            RobinhoodError: Appropriate error type
        """
        try:
            error_data = response.json()
            detail = error_data.get("detail", "Unknown error")
        except:
            detail = response.text

        if response.status_code == 401:
            logger.error("Authentication error, may need to refresh token")
            raise AuthenticationError(detail, code="401")

        elif response.status_code == 429:
            raise RateLimitError(detail, code="429")

        elif response.status_code >= 500:
            raise RobinhoodError(f"Server error: {detail}", code=str(response.status_code))

        else:
            raise RobinhoodError(f"API error ({response.status_code}): {detail}")

    def login(self) -> None:
        """Authenticate with Robinhood and store tokens.

        Raises:
            AuthenticationError: If login fails
        """
        logger.info("Logging into Robinhood...")

        # Get credentials from 1password if available
        from hrp.utils.credentials import get_credential

        username = get_credential("robinhood", "username") or self.config.username
        password = get_credential("robinhood", "password") or self.config.password

        payload = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "c82SH0WZOsabOXAS2MwK0j0z3kF",
            "expires_in": 86400,
            "scope": "internal",
        }

        # Add MFA if provided
        if self.config.mfa_code:
            payload["mfa_code"] = self.config.mfa_code

        try:
            response = self._get_session().post(
                self.LOGIN_URL, json=payload, timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()
            self._access_token = data.get("access_token")
            self._refresh_token = data.get("refresh_token")
            # Set expiry to slightly before actual expiry for safety
            self._token_expires = time.time() + data.get("expires_in", 86400) - 300

            logger.info("Successfully logged into Robinhood")

        except requests.exceptions.HTTPError as e:
            logger.error(f"Login failed: {e}")
            raise AuthenticationError(f"Login failed: {e}") from e

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token, refresh if needed."""
        now = time.time()

        # Check if token is expired or will expire soon (within 5 minutes)
        if self._access_token is None or (
            self._token_expires and now >= self._token_expires
        ):
            logger.info("Access token expired, logging in...")
            self.login()

    def get_account(self) -> Dict[str, Any]:
        """Get account information.

        Returns:
            Account information dictionary

        Raises:
            RobinhoodError: If request fails
        """
        self._ensure_authenticated()

        url = self.ACCOUNT_URL
        data = self._make_request("GET", url)
        return data["results"][0]

    def get_positions(self) -> list[Dict[str, Any]]:
        """Get current positions.

        Returns:
            List of position dictionaries

        Raises:
            RobinhoodError: If request fails
        """
        self._ensure_authenticated()

        url = self.POSITIONS_URL
        data = self._make_request("GET", url)
        return data.get("results", [])

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "gfd",  # Good for day
    ) -> Dict[str, Any]:
        """Place a trading order.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            quantity: Number of shares
            side: "buy" or "sell"
            order_type: "market", "limit", or "stop"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force ("gfd", "gtc", "ioc", "fok")

        Returns:
            Order response dictionary

        Raises:
            RobinhoodError: If order fails
        """
        self._ensure_authenticated()

        logger.info(
            f"Placing order: {side.upper()} {quantity} {symbol} "
            f"({order_type}, limit={limit_price}, stop={stop_price})"
        )

        # Build order payload
        order_data = {
            "account": self.get_account()["id"],
            "symbol": symbol.upper(),
            "type": order_type,
            "side": side,
            "quantity": str(quantity),
            "time_in_force": time_in_force,
        }

        # Add price parameters based on order type
        if order_type == "limit" and limit_price:
            order_data["price"] = str(limit_price)
        elif order_type == "stop" and stop_price:
            order_data["stop_price"] = str(stop_price)
        elif order_type == "stop_limit" and limit_price and stop_price:
            order_data["stop_price"] = str(stop_price)
            order_data["price"] = str(limit_price)

        # Place order
        response = self._make_request("POST", self.ORDERS_URL, data=order_data)

        logger.info(f"Order placed successfully: {response.get('id')}")
        return response

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details by ID.

        Args:
            order_id: Robinhood order ID

        Returns:
            Order details dictionary

        Raises:
            RobinhoodError: If request fails
        """
        self._ensure_authenticated()

        url = f"{self.ORDERS_URL}{order_id}/"
        return self._make_request("GET", url)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Robinhood order ID

        Returns:
            Cancellation response

        Raises:
            RobinhoodError: If cancellation fails
        """
        self._ensure_authenticated()

        logger.info(f"Cancelling order: {order_id}")

        url = f"{self.ORDERS_URL}{order_id}/cancel/"
        response = self._make_request("POST", url)

        logger.info(f"Order cancelled: {order_id}")
        return response

    def get_orders(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> list[Dict[str, Any]]:
        """Get order history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries

        Raises:
            RobinhoodError: If request fails
        """
        self._ensure_authenticated()

        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol.upper()

        data = self._make_request("GET", self.ORDERS_URL, params=params)
        return data.get("results", [])

    def logout(self) -> None:
        """Logout and invalidate tokens."""
        if self._access_token:
            logger.info("Logging out of Robinhood")
            # Invalidate refresh token
            try:
                self._make_request("POST", f"{self.LOGIN_URL}revoke_token/")
            except:
                pass

            self._access_token = None
            self._refresh_token = None
            self._token_expires = None

        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> "RobinhoodClient":
        """Context manager entry."""
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.logout()
