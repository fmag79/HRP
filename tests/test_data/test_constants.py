"""Tests for data layer constants."""

import pytest

from hrp.data.constants import TEST_SYMBOLS, DEFAULT_SYMBOLS


class TestConstants:
    """Tests for module constants."""

    def test_test_symbols_not_empty(self):
        """TEST_SYMBOLS should contain symbols."""
        assert len(TEST_SYMBOLS) > 0

    def test_test_symbols_are_strings(self):
        """TEST_SYMBOLS should be list of strings."""
        assert all(isinstance(s, str) for s in TEST_SYMBOLS)

    def test_test_symbols_contains_common_tickers(self):
        """TEST_SYMBOLS should contain common large-cap tickers."""
        common_tickers = ["AAPL", "MSFT", "GOOGL"]
        for ticker in common_tickers:
            assert ticker in TEST_SYMBOLS

    def test_default_symbols_equals_test_symbols(self):
        """DEFAULT_SYMBOLS should be an alias for TEST_SYMBOLS."""
        assert DEFAULT_SYMBOLS is TEST_SYMBOLS
