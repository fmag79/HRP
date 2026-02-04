"""
Authentication utilities for HRP Dashboard.

Provides user authentication using streamlit-authenticator with bcrypt hashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml
from loguru import logger

from hrp.utils.config import get_config

if TYPE_CHECKING:
    import bcrypt


@dataclass
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = True
    users_file: Path = field(
        default_factory=lambda: get_config().data.data_dir / "auth" / "users.yaml"
    )
    cookie_name: str = "hrp_auth"
    cookie_key: str = ""  # Must be set via env or generated
    cookie_expiry_days: int = 30

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load auth config from environment variables."""
        data_config = get_config().data

        users_file_str = os.getenv("HRP_AUTH_USERS_FILE")
        users_file = (
            Path(users_file_str)
            if users_file_str
            else data_config.data_dir / "auth" / "users.yaml"
        )

        return cls(
            enabled=os.getenv("HRP_AUTH_ENABLED", "true").lower() == "true",
            users_file=users_file,
            cookie_name=os.getenv("HRP_AUTH_COOKIE_NAME", "hrp_auth"),
            cookie_key=os.getenv("HRP_AUTH_COOKIE_KEY", ""),
            cookie_expiry_days=int(os.getenv("HRP_AUTH_COOKIE_EXPIRY_DAYS", "30")),
        )


def load_users(users_file: Path) -> dict[str, Any]:
    """
    Load users from YAML file.

    Creates empty structure if file doesn't exist.

    Args:
        users_file: Path to users YAML file

    Returns:
        Dictionary with credentials structure
    """
    if not users_file.exists():
        users_file.parent.mkdir(parents=True, exist_ok=True)
        empty = {"credentials": {"usernames": {}}}
        users_file.write_text(yaml.dump(empty))
        return empty

    with open(users_file) as f:
        return yaml.safe_load(f) or {"credentials": {"usernames": {}}}


def save_users(users_file: Path, users: dict[str, Any]) -> None:
    """
    Save users to YAML file.

    Args:
        users_file: Path to users YAML file
        users: Dictionary with credentials structure
    """
    users_file.parent.mkdir(parents=True, exist_ok=True)
    with open(users_file, "w") as f:
        yaml.dump(users, f, default_flow_style=False)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        bcrypt hash string
    """
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def add_user(
    users_file: Path, username: str, email: str, name: str, password: str
) -> None:
    """
    Add a new user to the users file.

    Args:
        users_file: Path to users YAML file
        username: Unique username
        email: User email
        name: Display name
        password: Plain text password (will be hashed)

    Raises:
        ValueError: If user already exists
    """
    users = load_users(users_file)

    if username in users["credentials"]["usernames"]:
        raise ValueError(f"User {username} already exists")

    users["credentials"]["usernames"][username] = {
        "email": email,
        "name": name,
        "password": hash_password(password),
    }

    save_users(users_file, users)
    logger.info(f"Added user: {username}")


def remove_user(users_file: Path, username: str) -> None:
    """
    Remove a user from the users file.

    Args:
        users_file: Path to users YAML file
        username: Username to remove

    Raises:
        ValueError: If user not found
    """
    users = load_users(users_file)

    if username not in users["credentials"]["usernames"]:
        raise ValueError(f"User {username} not found")

    del users["credentials"]["usernames"][username]
    save_users(users_file, users)
    logger.info(f"Removed user: {username}")


def get_authenticator(config: AuthConfig) -> Any | None:
    """
    Get configured streamlit-authenticator instance.

    Returns None if auth is disabled or not configured.

    Args:
        config: AuthConfig instance

    Returns:
        stauth.Authenticate instance or None
    """
    if not config.enabled:
        logger.debug("Authentication disabled")
        return None

    if not config.cookie_key:
        logger.warning("HRP_AUTH_COOKIE_KEY not set - authentication disabled")
        return None

    try:
        import streamlit_authenticator as stauth
    except ImportError:
        logger.error("streamlit-authenticator not installed")
        return None

    users = load_users(config.users_file)

    if not users["credentials"]["usernames"]:
        logger.warning("No users configured - authentication disabled")
        return None

    return stauth.Authenticate(
        users["credentials"],
        config.cookie_name,
        config.cookie_key,
        config.cookie_expiry_days,
    )
