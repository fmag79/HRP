"""
Feature registry for managing feature definitions and versions.

The registry tracks feature computation logic, versions, and metadata
in the feature_definitions table.
"""

import inspect
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from hrp.data.db import get_db


class FeatureRegistry:
    """
    Manages feature definitions and versions.

    Features are stored in the feature_definitions table with:
    - feature_name: unique identifier
    - version: version string (e.g., 'v1', 'v2')
    - computation_code: Python code or formula for computing the feature
    - description: human-readable description
    - is_active: whether this version is active
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the feature registry.

        Args:
            db_path: Optional path to database (defaults to standard location)
        """
        self.db = get_db(db_path)
        logger.debug("Feature registry initialized")

    def register_feature(
        self,
        feature_name: str,
        computation_fn: Callable,
        version: str,
        description: str | None = None,
        is_active: bool = True,
    ) -> None:
        """
        Register a new feature with a callable computation function.

        Args:
            feature_name: Unique identifier for the feature
            computation_fn: Callable function to compute the feature
            version: Version string (e.g., 'v1', 'v2')
            description: Optional human-readable description
            is_active: Whether this version is active (default: True)

        Raises:
            Exception: If feature+version already exists
        """
        # Serialize the function to string
        try:
            # Try to get source code for regular functions
            computation_code = inspect.getsource(computation_fn).strip()
        except (OSError, TypeError):
            # For lambdas and built-in functions, use repr
            computation_code = repr(computation_fn)

        # Use the existing register method
        self.register(
            feature_name=feature_name,
            version=version,
            computation_code=computation_code,
            description=description,
            is_active=is_active,
        )

    def register(
        self,
        feature_name: str,
        version: str,
        computation_code: str,
        description: str | None = None,
        is_active: bool = True,
    ) -> None:
        """
        Register a new feature definition.

        Args:
            feature_name: Unique identifier for the feature
            version: Version string (e.g., 'v1', 'v2')
            computation_code: Code or formula for computing the feature
            description: Optional human-readable description
            is_active: Whether this version is active (default: True)

        Raises:
            Exception: If feature+version already exists
        """
        with self.db.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO feature_definitions
                    (feature_name, version, computation_code, description, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feature_name,
                        version,
                        computation_code,
                        description,
                        is_active,
                        datetime.now(),
                    ),
                )
                logger.info(f"Registered feature: {feature_name} ({version})")
            except Exception as e:
                logger.error(f"Failed to register feature {feature_name} ({version}): {e}")
                raise

    def get(self, feature_name: str, version: str | None = None) -> dict[str, Any] | None:
        """
        Get a feature definition.

        Args:
            feature_name: Feature identifier
            version: Optional version (defaults to latest active version)

        Returns:
            Dictionary with feature definition or None if not found
        """
        with self.db.connection() as conn:
            if version:
                result = conn.execute(
                    """
                    SELECT feature_name, version, computation_code, description,
                           created_at, is_active
                    FROM feature_definitions
                    WHERE feature_name = ? AND version = ?
                    """,
                    (feature_name, version),
                ).fetchone()
            else:
                # Get latest active version
                result = conn.execute(
                    """
                    SELECT feature_name, version, computation_code, description,
                           created_at, is_active
                    FROM feature_definitions
                    WHERE feature_name = ? AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (feature_name,),
                ).fetchone()

            if result:
                return {
                    "feature_name": result[0],
                    "version": result[1],
                    "computation_code": result[2],
                    "description": result[3],
                    "created_at": result[4],
                    "is_active": result[5],
                }
            return None

    def list_features(self, active_only: bool = True) -> list[str]:
        """
        List all registered features.

        Args:
            active_only: If True, only return active features

        Returns:
            List of feature names (distinct)
        """
        with self.db.connection() as conn:
            if active_only:
                results = conn.execute(
                    """
                    SELECT DISTINCT feature_name
                    FROM feature_definitions
                    WHERE is_active = TRUE
                    ORDER BY feature_name
                    """
                ).fetchall()
            else:
                results = conn.execute(
                    """
                    SELECT DISTINCT feature_name
                    FROM feature_definitions
                    ORDER BY feature_name
                    """
                ).fetchall()

            return [row[0] for row in results]

    def list_all_features(self, active_only: bool = True) -> list[dict[str, Any]]:
        """
        List all registered features with full details.

        Args:
            active_only: If True, only return active features

        Returns:
            List of feature definition dictionaries
        """
        with self.db.connection() as conn:
            if active_only:
                results = conn.execute(
                    """
                    SELECT feature_name, version, computation_code, description,
                           created_at, is_active
                    FROM feature_definitions
                    WHERE is_active = TRUE
                    ORDER BY feature_name, created_at DESC
                    """
                ).fetchall()
            else:
                results = conn.execute(
                    """
                    SELECT feature_name, version, computation_code, description,
                           created_at, is_active
                    FROM feature_definitions
                    ORDER BY feature_name, created_at DESC
                    """
                ).fetchall()

            return [
                {
                    "feature_name": row[0],
                    "version": row[1],
                    "computation_code": row[2],
                    "description": row[3],
                    "created_at": row[4],
                    "is_active": row[5],
                }
                for row in results
            ]

    def list_versions(self, feature_name: str) -> list[dict[str, Any]]:
        """
        List all versions of a feature.

        Args:
            feature_name: Feature identifier

        Returns:
            List of version dictionaries for the feature
        """
        with self.db.connection() as conn:
            results = conn.execute(
                """
                SELECT feature_name, version, computation_code, description,
                       created_at, is_active
                FROM feature_definitions
                WHERE feature_name = ?
                ORDER BY created_at DESC
                """,
                (feature_name,),
            ).fetchall()

            return [
                {
                    "feature_name": row[0],
                    "version": row[1],
                    "computation_code": row[2],
                    "description": row[3],
                    "created_at": row[4],
                    "is_active": row[5],
                }
                for row in results
            ]

    def deactivate(self, feature_name: str, version: str) -> None:
        """
        Deactivate a feature version.

        Args:
            feature_name: Feature identifier
            version: Version to deactivate
        """
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE feature_definitions
                SET is_active = FALSE
                WHERE feature_name = ? AND version = ?
                """,
                (feature_name, version),
            )
            logger.info(f"Deactivated feature: {feature_name} ({version})")

    def activate(self, feature_name: str, version: str) -> None:
        """
        Activate a feature version.

        Args:
            feature_name: Feature identifier
            version: Version to activate
        """
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE feature_definitions
                SET is_active = TRUE
                WHERE feature_name = ? AND version = ?
                """,
                (feature_name, version),
            )
            logger.info(f"Activated feature: {feature_name} ({version})")
