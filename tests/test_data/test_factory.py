"""Tests for DataSourceFactory."""

import pytest

from hrp.data.sources.factory import DataSourceFactory


class TestDataSourceFactory:
    """Tests for DataSourceFactory class."""

    def test_create_polygon_source_with_fallback(self):
        """Test creating Polygon source with YFinance fallback."""
        primary, fallback = DataSourceFactory.create("polygon", with_fallback=True)

        assert primary is not None
        assert primary.source_name == "polygon"
        assert fallback is not None
        assert fallback.source_name == "yfinance"

    def test_create_yfinance_source_no_fallback(self):
        """Test creating YFinance source without fallback."""
        primary, fallback = DataSourceFactory.create("yfinance", with_fallback=False)

        assert primary is not None
        assert primary.source_name == "yfinance"
        assert fallback is None

    def test_create_polygon_without_fallback(self):
        """Test creating Polygon source without explicit fallback."""
        primary, fallback = DataSourceFactory.create("polygon", with_fallback=False)

        assert primary is not None
        assert primary.source_name == "polygon"
        # YFinance shouldn't be created as fallback when with_fallback=False
        assert fallback is None

    def test_create_unknown_source_raises_error(self):
        """Test that unknown source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            DataSourceFactory.create("unknown_source")

    def test_polygon_fallback_on_init_failure(self, monkeypatch):
        """Test that Polygon falls back to YFinance when init fails."""

        def mock_polygon_init_error(self):
            raise ValueError("Mock init error")

        # Monkey-patch PolygonSource.__init__ to raise ValueError
        import hrp.data.sources.polygon_source as ps
        monkeypatch.setattr(ps.PolygonSource, "__init__", mock_polygon_init_error)

        primary, fallback = DataSourceFactory.create("polygon", with_fallback=True)

        # Should fall back to YFinance
        assert primary.source_name == "yfinance"
        assert fallback is None
