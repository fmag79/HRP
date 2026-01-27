"""Base exception hierarchy for HRP.

All HRP exceptions inherit from HRPError, enabling consistent error handling
across the platform.
"""


class HRPError(Exception):
    """Base exception for all HRP errors."""

    pass


class APIError(HRPError):
    """Base for API-related errors."""

    pass


class ValidationError(HRPError):
    """Base for validation errors."""

    pass


class NotificationError(HRPError):
    """Base for notification/service errors."""

    pass


# =============================================================================
# API Exceptions
# =============================================================================


class PlatformAPIError(APIError):
    """Platform-level API errors."""

    pass


class PermissionError(APIError):
    """Raised when an actor lacks permission for an action."""

    pass


class NotFoundError(APIError):
    """Raised when a requested resource is not found."""

    pass


# =============================================================================
# Validation Exceptions
# =============================================================================


class OverfittingError(ValidationError):
    """Overfitting guard violations."""

    pass


# =============================================================================
# Notification Exceptions
# =============================================================================


class EmailNotificationError(NotificationError):
    """Email notification errors."""

    pass


class EmailConfigurationError(EmailNotificationError):
    """Raised when email configuration is missing or invalid."""

    pass
