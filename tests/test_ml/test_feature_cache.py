"""Tests for simplified feature selection caching."""

import pandas as pd
import pytest

from hrp.ml.validation import _feature_selection_cache


class TestFeatureSelectionCache:
    """Tests for module-level feature selection cache."""

    def test_cache_is_dict(self):
        """Test that cache is a dictionary."""
        assert isinstance(_feature_selection_cache, dict)

    def test_cache_can_store_and_retrieve(self):
        """Test that cache can store and retrieve feature lists."""
        # Clear cache first
        _feature_selection_cache.clear()

        # Store a result
        cache_key = "test_fold_0_expanding"
        features = ["feature1", "feature2"]
        _feature_selection_cache[cache_key] = features

        # Retrieve it
        retrieved = _feature_selection_cache.get(cache_key)
        assert retrieved == features

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        # Clear cache first
        _feature_selection_cache.clear()

        result = _feature_selection_cache.get("nonexistent_key")
        assert result is None

    def test_cache_clear(self):
        """Test that clear empties the cache."""
        # Add some data
        _feature_selection_cache["key1"] = ["feature1"]
        _feature_selection_cache["key2"] = ["feature2"]

        # Clear
        _feature_selection_cache.clear()

        # Verify empty
        assert len(_feature_selection_cache) == 0

    def test_cache_key_pattern(self):
        """Test that cache key pattern matches expected format."""
        # This tests the actual key pattern used in the code
        fold_idx = 0
        window_type = "expanding"
        cache_key = f"fold_{fold_idx}_{window_type}"

        # Should be able to use this as a dict key
        _feature_selection_cache[cache_key] = ["feature1", "feature2"]
        assert cache_key in _feature_selection_cache
