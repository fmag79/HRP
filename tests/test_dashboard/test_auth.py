"""Tests for dashboard authentication module."""

import pytest
from pathlib import Path
import yaml


class TestAuthConfig:
    """Tests for AuthConfig dataclass."""

    def test_auth_config_defaults(self):
        """AuthConfig should have sensible defaults."""
        from hrp.dashboard.auth import AuthConfig

        config = AuthConfig()
        assert config.enabled is True
        assert config.cookie_expiry_days == 30
        assert config.cookie_name == "hrp_auth"

    def test_auth_config_from_env(self, monkeypatch, tmp_path):
        """AuthConfig should load from environment variables."""
        users_file = tmp_path / "users.yaml"
        monkeypatch.setenv("HRP_AUTH_USERS_FILE", str(users_file))
        monkeypatch.setenv("HRP_AUTH_COOKIE_KEY", "test_secret_key_32chars_long!!!")
        monkeypatch.setenv("HRP_AUTH_COOKIE_NAME", "custom_cookie")
        monkeypatch.setenv("HRP_AUTH_COOKIE_EXPIRY_DAYS", "7")

        from hrp.dashboard.auth import AuthConfig

        config = AuthConfig.from_env()
        assert config.users_file == users_file
        assert config.cookie_key == "test_secret_key_32chars_long!!!"
        assert config.cookie_name == "custom_cookie"
        assert config.cookie_expiry_days == 7

    def test_auth_config_disabled(self, monkeypatch):
        """AuthConfig should support disabling auth."""
        monkeypatch.setenv("HRP_AUTH_ENABLED", "false")

        from hrp.dashboard.auth import AuthConfig

        config = AuthConfig.from_env()
        assert config.enabled is False


class TestUserManagement:
    """Tests for user management functions."""

    def test_load_users_from_yaml(self, tmp_path):
        """load_users should load users from YAML file."""
        users_file = tmp_path / "users.yaml"
        users_file.write_text("""
credentials:
  usernames:
    admin:
      email: admin@example.com
      name: Admin User
      password: $2b$12$hashedpassword
""")
        from hrp.dashboard.auth import load_users

        users = load_users(users_file)
        assert "admin" in users["credentials"]["usernames"]
        assert users["credentials"]["usernames"]["admin"]["email"] == "admin@example.com"

    def test_load_users_creates_empty_if_missing(self, tmp_path):
        """load_users should create empty structure if file doesn't exist."""
        users_file = tmp_path / "nonexistent" / "users.yaml"

        from hrp.dashboard.auth import load_users

        users = load_users(users_file)
        assert users["credentials"]["usernames"] == {}
        assert users_file.exists()

    def test_save_users_creates_file(self, tmp_path):
        """save_users should create file with user data."""
        users_file = tmp_path / "users.yaml"
        users = {
            "credentials": {
                "usernames": {
                    "test": {"email": "test@example.com", "name": "Test", "password": "hashed"}
                }
            }
        }

        from hrp.dashboard.auth import save_users

        save_users(users_file, users)

        assert users_file.exists()
        loaded = yaml.safe_load(users_file.read_text())
        assert "test" in loaded["credentials"]["usernames"]

    def test_hash_password_creates_bcrypt_hash(self):
        """hash_password should create a valid bcrypt hash."""
        from hrp.dashboard.auth import hash_password

        hashed = hash_password("test_password")
        assert hashed.startswith("$2b$")
        assert len(hashed) == 60  # bcrypt hash length

    def test_hash_password_different_each_time(self):
        """hash_password should create different hashes for same password."""
        from hrp.dashboard.auth import hash_password

        hash1 = hash_password("same_password")
        hash2 = hash_password("same_password")
        assert hash1 != hash2  # Different salts

    def test_add_user_creates_entry(self, tmp_path):
        """add_user should create a new user entry."""
        users_file = tmp_path / "users.yaml"

        from hrp.dashboard.auth import add_user, load_users

        add_user(users_file, "testuser", "test@example.com", "Test User", "password123")

        users = load_users(users_file)
        assert "testuser" in users["credentials"]["usernames"]
        assert users["credentials"]["usernames"]["testuser"]["email"] == "test@example.com"
        assert users["credentials"]["usernames"]["testuser"]["name"] == "Test User"
        # Password should be hashed
        assert users["credentials"]["usernames"]["testuser"]["password"].startswith("$2b$")

    def test_add_user_raises_if_exists(self, tmp_path):
        """add_user should raise if user already exists."""
        users_file = tmp_path / "users.yaml"

        from hrp.dashboard.auth import add_user

        add_user(users_file, "testuser", "test@example.com", "Test User", "password123")

        with pytest.raises(ValueError, match="already exists"):
            add_user(users_file, "testuser", "other@example.com", "Other", "pass")

    def test_remove_user_deletes_entry(self, tmp_path):
        """remove_user should delete user from file."""
        users_file = tmp_path / "users.yaml"

        from hrp.dashboard.auth import add_user, load_users, remove_user

        add_user(users_file, "testuser", "test@example.com", "Test User", "password123")
        remove_user(users_file, "testuser")

        users = load_users(users_file)
        assert "testuser" not in users["credentials"]["usernames"]

    def test_remove_user_raises_if_not_found(self, tmp_path):
        """remove_user should raise if user not found."""
        users_file = tmp_path / "users.yaml"

        from hrp.dashboard.auth import remove_user

        with pytest.raises(ValueError, match="not found"):
            remove_user(users_file, "nonexistent")


class TestGetAuthenticator:
    """Tests for get_authenticator function."""

    def test_get_authenticator_returns_none_if_disabled(self):
        """get_authenticator should return None if auth disabled."""
        from hrp.dashboard.auth import AuthConfig, get_authenticator

        config = AuthConfig(enabled=False)
        result = get_authenticator(config)
        assert result is None

    def test_get_authenticator_returns_none_if_no_cookie_key(self):
        """get_authenticator should return None if cookie key not set."""
        from hrp.dashboard.auth import AuthConfig, get_authenticator

        config = AuthConfig(enabled=True, cookie_key="")
        result = get_authenticator(config)
        assert result is None

    def test_get_authenticator_returns_none_if_no_users(self, tmp_path):
        """get_authenticator should return None if no users configured."""
        users_file = tmp_path / "users.yaml"
        users_file.parent.mkdir(parents=True, exist_ok=True)
        users_file.write_text("credentials:\n  usernames: {}\n")

        from hrp.dashboard.auth import AuthConfig, get_authenticator

        config = AuthConfig(
            enabled=True,
            cookie_key="test_key_32_chars_long_enough!!",
            users_file=users_file,
        )
        result = get_authenticator(config)
        assert result is None
