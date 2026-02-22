"""
HRP - Hedgefund Research Platform

Personal quantitative research platform for systematic trading strategy development.
"""

try:
    from importlib.metadata import version

    __version__ = version("hrp")
except Exception:
    # Fallback: read version from pyproject.toml
    try:
        import tomllib
        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        __version__ = data["project"]["version"]
    except Exception:
        __version__ = "0.0.0"  # Final fallback

__author__ = "Fernando"
