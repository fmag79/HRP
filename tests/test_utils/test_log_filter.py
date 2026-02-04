"""Test secret filtering in logs."""


def test_filter_secrets_masks_api_keys():
    """API keys should be masked."""
    from hrp.utils.log_filter import filter_secrets

    text = "Using ANTHROPIC_API_KEY=sk-ant-abc123 for requests"
    result = filter_secrets(text)
    assert "sk-ant-abc123" not in result
    assert "***" in result


def test_filter_secrets_masks_passwords():
    """Passwords should be masked."""
    from hrp.utils.log_filter import filter_secrets

    text = "password=super_secret_123"
    result = filter_secrets(text)
    assert "super_secret_123" not in result


def test_filter_secrets_preserves_safe_text():
    """Non-secret text should be preserved."""
    from hrp.utils.log_filter import filter_secrets

    text = "Processing 100 records for AAPL"
    result = filter_secrets(text)
    assert result == text
