"""Secret filtering for log messages.

Prevents accidental logging of API keys, passwords, and tokens.
"""

from __future__ import annotations

import re

# Patterns to mask in log messages
SECRET_PATTERNS = [
    # API keys with values
    (r"(ANTHROPIC_API_KEY|POLYGON_API_KEY|SIMFIN_API_KEY|RESEND_API_KEY)\s*[=:]\s*\S+", r"\1=***"),
    # Generic API key patterns
    (r"(api[_-]?key|apikey)\s*[=:]\s*\S+", r"\1=***", re.IGNORECASE),
    # Password patterns
    (r"(password|passwd|pwd)\s*[=:]\s*\S+", r"\1=***", re.IGNORECASE),
    # Token patterns
    (r"(token|secret|credential)\s*[=:]\s*\S+", r"\1=***", re.IGNORECASE),
    # Bearer tokens
    (r"Bearer\s+\S+", "Bearer ***"),
    # sk-ant-* pattern (Anthropic keys)
    (r"sk-ant-[a-zA-Z0-9-]+", "***"),
]


def filter_secrets(text: str) -> str:
    """
    Mask secrets in text.

    Args:
        text: Input text that may contain secrets

    Returns:
        Text with secrets masked as ***
    """
    result = text
    for pattern in SECRET_PATTERNS:
        if len(pattern) == 3:
            regex, replacement, flags = pattern
            result = re.sub(regex, replacement, result, flags=flags)
        else:
            regex, replacement = pattern
            result = re.sub(regex, replacement, result)
    return result
