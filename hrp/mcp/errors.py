"""
Error handling utilities for MCP server.

Provides decorators and utilities for consistent error handling across all tools.
"""

from functools import wraps
from typing import Any, Callable

from loguru import logger

from hrp.exceptions import NotFoundError, PermissionError, PlatformAPIError
from hrp.mcp.formatters import format_response


def handle_api_error(func: Callable) -> Callable:
    """
    Decorator that catches API errors and returns structured error responses.

    Handles:
    - PlatformAPIError: Platform-level errors
    - PermissionError: Authorization errors
    - NotFoundError: Resource not found
    - ValueError: Invalid input parameters
    - Exception: Unexpected errors (logged, message sanitized)

    Args:
        func: The MCP tool function to wrap

    Returns:
        Wrapped function that returns error responses instead of raising
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except PermissionError as e:
            logger.warning(f"Permission denied in {func.__name__}: {e}")
            return format_response(
                success=False,
                message="Permission denied",
                error=str(e),
            )
        except NotFoundError as e:
            logger.info(f"Resource not found in {func.__name__}: {e}")
            return format_response(
                success=False,
                message="Resource not found",
                error=str(e),
            )
        except ValueError as e:
            logger.info(f"Invalid input in {func.__name__}: {e}")
            return format_response(
                success=False,
                message="Invalid input",
                error=str(e),
            )
        except PlatformAPIError as e:
            logger.error(f"Platform API error in {func.__name__}: {e}")
            return format_response(
                success=False,
                message="Platform error",
                error=str(e),
            )
        except Exception as e:
            # Log full exception but return sanitized message
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            return format_response(
                success=False,
                message="An unexpected error occurred",
                error=f"Internal error: {type(e).__name__}",
            )

    return wrapper


class MCPError(Exception):
    """Base exception for MCP-specific errors."""

    def __init__(self, message: str, code: str = "MCP_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class ToolNotFoundError(MCPError):
    """Raised when a requested tool does not exist."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' not found",
            code="TOOL_NOT_FOUND",
        )
        self.tool_name = tool_name


class InvalidParameterError(MCPError):
    """Raised when a tool parameter is invalid."""

    def __init__(self, param_name: str, message: str):
        super().__init__(
            f"Invalid parameter '{param_name}': {message}",
            code="INVALID_PARAMETER",
        )
        self.param_name = param_name
