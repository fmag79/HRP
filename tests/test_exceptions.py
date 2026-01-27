"""Tests for HRP exception hierarchy."""

import pytest

from hrp.exceptions import (
    HRPError,
    APIError,
    ValidationError,
    NotificationError,
    PlatformAPIError,
    PermissionError,
    NotFoundError,
    EmailNotificationError,
    EmailConfigurationError,
    OverfittingError,
)


class TestExceptionHierarchy:
    """Tests for exception base classes."""

    def test_hrp_error_is_base_exception(self):
        """HRPError should inherit from Exception."""
        assert issubclass(HRPError, Exception)

    def test_api_error_inherits_from_hrp_error(self):
        """APIError should inherit from HRPError."""
        assert issubclass(APIError, HRPError)

    def test_validation_error_inherits_from_hrp_error(self):
        """ValidationError should inherit from HRPError."""
        assert issubclass(ValidationError, HRPError)

    def test_notification_error_inherits_from_hrp_error(self):
        """NotificationError should inherit from HRPError."""
        assert issubclass(NotificationError, HRPError)


class TestAPIExceptions:
    """Tests for API-related exceptions."""

    def test_platform_api_error_inherits_from_api_error(self):
        """PlatformAPIError should inherit from APIError."""
        assert issubclass(PlatformAPIError, APIError)

    def test_permission_error_inherits_from_api_error(self):
        """PermissionError should inherit from APIError."""
        assert issubclass(PermissionError, APIError)

    def test_not_found_error_inherits_from_api_error(self):
        """NotFoundError should inherit from APIError."""
        assert issubclass(NotFoundError, APIError)

    def test_platform_api_error_catch_as_base(self):
        """PlatformAPIError should be catchable as APIError."""
        exc = PlatformAPIError("test")
        assert isinstance(exc, APIError)
        assert isinstance(exc, HRPError)


class TestValidationExceptions:
    """Tests for validation-related exceptions."""

    def test_overfitting_error_inherits_from_validation_error(self):
        """OverfittingError should inherit from ValidationError."""
        assert issubclass(OverfittingError, ValidationError)

    def test_overfitting_error_catch_as_base(self):
        """OverfittingError should be catchable as ValidationError."""
        exc = OverfittingError("test")
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, HRPError)


class TestNotificationExceptions:
    """Tests for notification-related exceptions."""

    def test_email_notification_error_inherits_from_notification_error(self):
        """EmailNotificationError should inherit from NotificationError."""
        assert issubclass(EmailNotificationError, NotificationError)

    def test_email_configuration_error_inherits_from_email_notification_error(self):
        """EmailConfigurationError should inherit from EmailNotificationError."""
        assert issubclass(EmailConfigurationError, EmailNotificationError)

    def test_email_notification_error_catch_as_base(self):
        """EmailNotificationError should be catchable as NotificationError."""
        exc = EmailNotificationError("test")
        assert isinstance(exc, NotificationError)
        assert isinstance(exc, HRPError)
