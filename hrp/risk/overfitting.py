"""
Overfitting prevention mechanisms.

Implements test set discipline and overfitting detection.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any

from loguru import logger

from hrp.data.db import get_db


class OverfittingError(Exception):
    """Raised when test set evaluation limit is exceeded."""
    pass


def _load_evaluation_count(hypothesis_id: str) -> int:
    """Load existing evaluation count from database."""
    db = get_db()
    
    with db.connection() as conn:
        result = conn.execute(
            """
            SELECT COUNT(*) 
            FROM test_set_evaluations 
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
    
    return result[0] if result else 0


def _log_evaluation(
    hypothesis_id: str,
    override: bool,
    override_reason: str | None,
    metadata: dict[str, Any] | None,
):
    """Log test set evaluation to database."""
    db = get_db()
    
    # Convert metadata dict to JSON string if present
    import json
    metadata_json = json.dumps(metadata) if metadata else None
    
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO test_set_evaluations 
            (hypothesis_id, evaluated_at, override, override_reason, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                hypothesis_id,
                datetime.utcnow(),
                override,
                override_reason,
                metadata_json,
            ),
        )


class TestSetGuard:
    """
    Guard against excessive test set evaluation.
    
    Enforces limit on number of times test set can be evaluated per hypothesis
    to prevent data snooping and overfitting.
    
    Usage:
        guard = TestSetGuard(hypothesis_id='HYP-2025-001')
        
        with guard.evaluate():
            metrics = model.evaluate(test_data)
    
    Raises:
        OverfittingError: If evaluation limit exceeded without explicit override
    """

    def __init__(self, hypothesis_id: str, max_evaluations: int = 3):
        """
        Initialize test set guard.
        
        Args:
            hypothesis_id: Hypothesis ID
            max_evaluations: Maximum allowed evaluations (default 3)
        """
        self.hypothesis_id = hypothesis_id
        self.max_evaluations = max_evaluations
        self._count = _load_evaluation_count(hypothesis_id)
        
        logger.debug(
            f"TestSetGuard for {hypothesis_id}: "
            f"{self._count}/{max_evaluations} evaluations used"
        )

    @property
    def evaluation_count(self) -> int:
        """Current evaluation count."""
        return self._count

    @property
    def remaining_evaluations(self) -> int:
        """Remaining evaluations allowed."""
        return max(0, self.max_evaluations - self._count)

    @contextmanager
    def evaluate(
        self,
        override: bool = False,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for test set evaluation.
        
        Args:
            override: Explicitly override limit (requires reason)
            reason: Reason for override (required if override=True)
            metadata: Optional metadata to log with evaluation
            
        Raises:
            OverfittingError: If limit exceeded without override
            ValueError: If override=True but reason not provided
        """
        if override and not reason:
            raise ValueError("Override requires a reason")

        if not override and self._count >= self.max_evaluations:
            raise OverfittingError(
                f"Test set evaluation limit exceeded for {self.hypothesis_id}. "
                f"Already evaluated {self._count} times (limit: {self.max_evaluations}). "
                f"Use override=True with justification if needed."
            )

        if override:
            logger.warning(
                f"Test set evaluation override for {self.hypothesis_id}: {reason}"
            )

        # Log the evaluation
        _log_evaluation(self.hypothesis_id, override, reason, metadata)
        self._count += 1

        try:
            yield
        except Exception:
            # Evaluation failed, but still counts toward limit
            logger.error(f"Test set evaluation failed for {self.hypothesis_id}")
            raise
