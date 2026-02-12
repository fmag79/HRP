"""Tests for Robinhood authentication."""
from unittest.mock import MagicMock, patch

import pytest

from hrp.execution.robinhood_auth import (
    RobinhoodAuthConfig,
    RobinhoodSession,
    load_config_from_env,
)


class TestRobinhoodAuthConfig:
    """Tests for RobinhoodAuthConfig."""

    def test_default_config(self):
        """Test default auth configuration."""
        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        assert config.username == "test@example.com"
        assert config.password == "test_password"
        assert config.totp_secret is None
        assert config.session_expiry == 86400
        assert config.pickle_name == ""
        assert config.device_token is None

    def test_custom_config(self):
        """Test custom auth configuration."""
        config = RobinhoodAuthConfig(
            username="test@example.com",
            password="test_password",
            totp_secret="ABC123",
            session_expiry=3600,
            pickle_name="test",
        )

        assert config.totp_secret == "ABC123"
        assert config.session_expiry == 3600
        assert config.pickle_name == "test"


class TestRobinhoodSession:
    """Tests for RobinhoodSession."""

    @patch("hrp.execution.robinhood_auth.rh")
    def test_initialization(self, mock_rh):
        """Test session initialization."""
        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        session = RobinhoodSession(config)

        assert session.config == config
        assert not session._authenticated
        assert session._login_info is None

    def test_initialization_requires_credentials(self):
        """Test initialization fails without credentials."""
        with pytest.raises(ValueError, match="username and password are required"):
            RobinhoodSession(RobinhoodAuthConfig(username="", password="test"))

        with pytest.raises(ValueError, match="username and password are required"):
            RobinhoodSession(RobinhoodAuthConfig(username="test", password=""))

    @patch("hrp.execution.robinhood_auth.rh")
    def test_initialization_without_robin_stocks(self, mock_rh):
        """Test initialization fails without robin_stocks."""
        mock_rh.side_effect = ImportError("robin_stocks not found")

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with pytest.raises(ImportError, match="robin_stocks not installed"):
            RobinhoodSession(config)

    @patch("hrp.execution.robinhood_auth.pyotp")
    @patch("hrp.execution.robinhood_auth.rh")
    def test_login_success(self, mock_rh_module, mock_pyotp):
        """Test successful login."""
        # Mock robin_stocks login
        mock_rh = MagicMock()
        mock_rh.login.return_value = {"access_token": "test_token"}

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._pyotp = None
            session._authenticated = False
            session._login_info = None

            result = session.login()

            assert result is True
            assert session._authenticated is True
            assert session._login_info == {"access_token": "test_token"}
            mock_rh.login.assert_called_once()

    @patch("hrp.execution.robinhood_auth.pyotp")
    @patch("hrp.execution.robinhood_auth.rh")
    def test_login_with_mfa(self, mock_rh_module, mock_pyotp):
        """Test login with MFA."""
        # Mock TOTP
        mock_totp = MagicMock()
        mock_totp.now.return_value = "123456"
        mock_pyotp.TOTP.return_value = mock_totp

        # Mock robin_stocks
        mock_rh = MagicMock()
        mock_rh.login.return_value = {"access_token": "test_token"}

        config = RobinhoodAuthConfig(
            username="test@example.com",
            password="test_password",
            totp_secret="ABCDEF123456",
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._pyotp = mock_pyotp
            session._authenticated = False
            session._login_info = None

            result = session.login()

            assert result is True
            mock_pyotp.TOTP.assert_called_once_with("ABCDEF123456")
            mock_totp.now.assert_called_once()

    @patch("hrp.execution.robinhood_auth.rh")
    def test_login_failure_none_response(self, mock_rh_module):
        """Test login failure with None response."""
        mock_rh = MagicMock()
        mock_rh.login.return_value = None

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._pyotp = None
            session._authenticated = False
            session._login_info = None

            result = session.login()

            assert result is False
            assert session._authenticated is False

    @patch("hrp.execution.robinhood_auth.rh")
    def test_login_failure_error_response(self, mock_rh_module):
        """Test login failure with error response."""
        mock_rh = MagicMock()
        mock_rh.login.return_value = {"detail": "Invalid credentials"}

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._pyotp = None
            session._authenticated = False
            session._login_info = None

            result = session.login()

            assert result is False
            assert session._authenticated is False

    @patch("hrp.execution.robinhood_auth.rh")
    def test_logout(self, mock_rh_module):
        """Test logout."""
        mock_rh = MagicMock()

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._authenticated = True
            session._login_info = {"token": "test"}

            session.logout()

            assert session._authenticated is False
            assert session._login_info is None
            mock_rh.logout.assert_called_once()

    @patch("hrp.execution.robinhood_auth.rh")
    def test_is_authenticated(self, mock_rh_module):
        """Test authentication check."""
        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._authenticated = False

            assert session.is_authenticated() is False

            session._authenticated = True
            assert session.is_authenticated() is True

    @patch("hrp.execution.robinhood_auth.rh")
    def test_ensure_authenticated_when_valid(self, mock_rh_module):
        """Test ensure_authenticated with valid session."""
        mock_rh = MagicMock()
        mock_rh.load_account_profile.return_value = {"account_number": "12345"}

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._authenticated = True

            # Should not raise
            session.ensure_authenticated()

            assert session._authenticated is True
            mock_rh.load_account_profile.assert_called_once()

    @patch("hrp.execution.robinhood_auth.rh")
    def test_ensure_authenticated_relogins_on_expiry(self, mock_rh_module):
        """Test ensure_authenticated re-logins on expiry."""
        mock_rh = MagicMock()
        mock_rh.load_account_profile.return_value = None  # Expired session
        mock_rh.login.return_value = {"access_token": "new_token"}

        config = RobinhoodAuthConfig(
            username="test@example.com", password="test_password"
        )

        with patch.object(
            RobinhoodSession, "_RobinhoodSession__init__", return_value=None
        ):
            session = RobinhoodSession.__new__(RobinhoodSession)
            session.config = config
            session._rh = mock_rh
            session._pyotp = None
            session._authenticated = True
            session._login_info = None

            session.ensure_authenticated()

            # Should have re-authenticated
            mock_rh.login.assert_called_once()


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env."""

    @patch.dict(
        "os.environ",
        {
            "ROBINHOOD_USERNAME": "test@example.com",
            "ROBINHOOD_PASSWORD": "test_password",
            "ROBINHOOD_TOTP_SECRET": "ABC123",
            "ROBINHOOD_PICKLE_NAME": "test",
        },
    )
    def test_load_config_from_env(self):
        """Test loading config from environment variables."""
        config = load_config_from_env()

        assert config.username == "test@example.com"
        assert config.password == "test_password"
        assert config.totp_secret == "ABC123"
        assert config.pickle_name == "test"

    @patch.dict(
        "os.environ",
        {"ROBINHOOD_USERNAME": "test@example.com", "ROBINHOOD_PASSWORD": "test_password"},
    )
    def test_load_config_optional_vars(self):
        """Test loading config with optional vars."""
        config = load_config_from_env()

        assert config.username == "test@example.com"
        assert config.password == "test_password"
        assert config.totp_secret is None
        assert config.pickle_name == ""

    @patch.dict("os.environ", {"ROBINHOOD_USERNAME": "test@example.com"}, clear=True)
    def test_load_config_missing_password(self):
        """Test loading config with missing password."""
        with pytest.raises(ValueError, match="environment variables required"):
            load_config_from_env()

    @patch.dict("os.environ", {}, clear=True)
    def test_load_config_missing_all(self):
        """Test loading config with missing variables."""
        with pytest.raises(ValueError, match="environment variables required"):
            load_config_from_env()
