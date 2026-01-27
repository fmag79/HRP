"""Factory for creating data sources with automatic fallback."""

from loguru import logger

from hrp.data.sources.base import DataSourceBase
from hrp.data.sources.polygon_source import PolygonSource
from hrp.data.sources.yfinance_source import YFinanceSource


class DataSourceFactory:
    """
    Factory for creating data sources with optional fallback.

    Encapsulates the complex logic of source initialization and fallback handling.
    Supports multiple data sources with automatic fallback when primary is unavailable.

    Example:
        # Create Polygon source with YFinance fallback
        primary, fallback = DataSourceFactory.create("polygon")

        # Create YFinance source only
        primary, fallback = DataSourceFactory.create("yfinance", with_fallback=False)
    """

    # Registry of available sources with their fallback configurations
    _sources = {
        "polygon": (PolygonSource, YFinanceSource),
        "yfinance": (YFinanceSource, None),
    }

    @staticmethod
    def create(source: str, with_fallback: bool = True) -> tuple[DataSourceBase, DataSourceBase | None]:
        """
        Create data source with optional fallback.

        Args:
            source: Source name ("polygon" or "yfinance")
            with_fallback: Whether to create fallback source (default True)

        Returns:
            Tuple of (primary_source, fallback_source)
            fallback_source is None if not applicable or with_fallback=False

        Raises:
            ValueError: If source name is unknown

        Example:
            >>> primary, fallback = DataSourceFactory.create("polygon")
            >>> assert primary.source_name == "Polygon.io"
            >>> assert fallback.source_name == "Yahoo Finance"
        """
        if source not in DataSourceFactory._sources:
            raise ValueError(
                f"Unknown source: {source}. Available sources: {', '.join(DataSourceFactory._sources.keys())}"
            )

        primary_cls, fallback_cls = DataSourceFactory._sources[source]

        try:
            primary = primary_cls()
            fallback = None

            # Create fallback if requested and available
            if with_fallback and fallback_cls is not None:
                try:
                    fallback = fallback_cls()
                    logger.info(f"Using {primary.source_name} as primary with {fallback.source_name} fallback")
                except ValueError:
                    # Fallback initialization failed - this is unusual but shouldn't prevent primary use
                    logger.warning(f"{fallback_cls.__name__} initialization failed, using {primary.source_name} only")
                    fallback = None
            else:
                logger.info(f"Using {primary.source_name} as primary source")

        except ValueError as e:
            # Primary source initialization failed
            if with_fallback and fallback_cls is not None:
                logger.warning(f"{primary_cls.__name__} unavailable ({e}), falling back to {fallback_cls.__name__}")
                primary = fallback_cls()
                fallback = None
            else:
                # No fallback available or requested
                raise

        return primary, fallback
