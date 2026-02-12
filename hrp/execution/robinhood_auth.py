"""Robinhood authentication and session management."""
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RobinhoodAuthConfig:
    """Robinhood authentication configuration.

    Security notes:
        - password and totp_secret should NEVER be in code or config files
        - Load from environment variables or secrets manager only
        - pickle_name determines session file: ~/.tokens/{pickle_name}.pickle
    """

    username: str  # Email
    password: str  # Password (from env var)
    totp_secret: str | None = None  # Base32 TOTP secret for automated MFA
    session_expiry: int = 86400  # 24 hours default
    pickle_name: str = ""  # Session file suffix
    device_token: str | None = None  # Persistent device token


class RobinhoodSession:
    """Manages Robinhood authentication lifecycle.

    Handles login, MFA (via pyotp if configured), session persistence,
    and automatic re-authentication on expiry.

    Example:
        >>> config = RobinhoodAuthConfig(
        ...     username=os.getenv("ROBINHOOD_USERNAME"),
        ...     password=os.getenv("ROBINHOOD_PASSWORD"),
        ...     totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
        ... )
        >>> session = RobinhoodSession(config)
        >>> session.login()  # Authenticate
        >>> session.ensure_authenticated()  # Re-auth if expired
        >>> session.logout()  # Clean up
    """

    def __init__(self, config: RobinhoodAuthConfig) -> None:
        """Initialize session manager.

        Args:
            config: Authentication configuration.

        Raises:
            ValueError: If username or password is empty.
        """
        if not config.username or not config.password:
            raise ValueError("username and password are required")

        self.config = config
        self._authenticated = False
        self._login_info: dict | None = None

        # Import here to avoid global dependency
        try:
            import robin_stocks.robinhood as rh

            self._rh = rh
        except ImportError:
            raise ImportError(
                "robin_stocks not installed. Install with: pip install robin-stocks"
            )

        if config.totp_secret:
            try:
                import pyotp

                self._pyotp = pyotp
            except ImportError:
                raise ImportError(
                    "pyotp not installed but totp_secret provided. "
                    "Install with: pip install pyotp"
                )
        else:
            self._pyotp = None

        logger.info(
            "RobinhoodSession initialized for user: %s (MFA: %s)",
            config.username,
            "enabled" if config.totp_secret else "disabled",
        )

    def login(self) -> bool:
        """Authenticate with Robinhood.

        Uses pyotp for MFA if totp_secret is configured, otherwise
        falls back to interactive MFA (not recommended for production).

        Returns:
            True if login successful, False otherwise.

        Raises:
            RuntimeError: If authentication fails after retries.
        """
        try:
            # Generate MFA code if configured
            mfa_code = None
            if self.config.totp_secret and self._pyotp:
                totp = self._pyotp.TOTP(self.config.totp_secret)
                mfa_code = totp.now()
                logger.debug("Generated MFA code from TOTP secret")

            # Attempt login
            logger.info("Attempting Robinhood login for %s", self.config.username)
            self._login_info = self._rh.login(
                username=self.config.username,
                password=self.config.password,
                mfa_code=mfa_code,
                pickle_name=self.config.pickle_name,
                expiresIn=self.config.session_expiry,
            )

            # Check for success
            if self._login_info is None:
                logger.error("Robinhood login failed: login() returned None")
                self._authenticated = False
                return False

            # robin_stocks returns dict with 'detail' key on error
            if isinstance(self._login_info, dict) and "detail" in self._login_info:
                error_detail = self._login_info["detail"]
                logger.error("Robinhood login failed: %s", error_detail)
                self._authenticated = False
                return False

            self._authenticated = True
            logger.info("Robinhood login successful for %s", self.config.username)
            return True

        except Exception as e:
            logger.exception("Robinhood login exception: %s", e)
            self._authenticated = False
            raise RuntimeError(f"Robinhood authentication failed: {e}") from e

    def logout(self) -> None:
        """End session and clean up.

        Calls robin_stocks.logout() to invalidate the session token.
        """
        if self._authenticated:
            try:
                self._rh.logout()
                logger.info("Robinhood session logged out")
            except Exception as e:
                logger.warning("Logout exception (ignoring): %s", e)
            finally:
                self._authenticated = False
                self._login_info = None

    def is_authenticated(self) -> bool:
        """Check if session is valid.

        Note: This only checks local state, not actual session validity
        with Robinhood. Use ensure_authenticated() for robust checking.

        Returns:
            True if authenticated locally, False otherwise.
        """
        return self._authenticated

    def ensure_authenticated(self) -> None:
        """Re-authenticate if session expired.

        Should be called before every API operation to ensure the
        session is valid. Re-logins automatically if needed.

        Raises:
            RuntimeError: If re-authentication fails.
        """
        # Try a lightweight API call to verify session
        try:
            # get_account() is a fast operation to check session validity
            account_info = self._rh.load_account_profile()

            # None response means session expired
            if account_info is None:
                logger.warning("Session expired (got None), re-authenticating")
                self._authenticated = False

        except Exception as e:
            logger.warning("Session validation failed: %s, re-authenticating", e)
            self._authenticated = False

        # Re-login if needed
        if not self._authenticated:
            logger.info("Re-authenticating Robinhood session")
            if not self.login():
                raise RuntimeError("Failed to re-authenticate Robinhood session")


def load_config_from_env() -> RobinhoodAuthConfig:
    """Load Robinhood auth config from environment variables.

    Expected environment variables:
        ROBINHOOD_USERNAME: Email address
        ROBINHOOD_PASSWORD: Password
        ROBINHOOD_TOTP_SECRET: (Optional) Base32 TOTP secret for MFA
        ROBINHOOD_PICKLE_NAME: (Optional) Session file suffix

    Returns:
        RobinhoodAuthConfig populated from environment.

    Raises:
        ValueError: If required variables are missing.
    """
    username = os.getenv("ROBINHOOD_USERNAME")
    password = os.getenv("ROBINHOOD_PASSWORD")

    if not username or not password:
        raise ValueError(
            "ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD environment variables required"
        )

    return RobinhoodAuthConfig(
        username=username,
        password=password,
        totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
        pickle_name=os.getenv("ROBINHOOD_PICKLE_NAME", ""),
    )
