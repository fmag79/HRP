"""Test configurable thresholds."""

import pytest


class TestOpsThresholds:
    """Tests for OpsThresholds dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        from hrp.ops.thresholds import OpsThresholds

        t = OpsThresholds()
        assert t.health_score_warning == 90.0
        assert t.health_score_critical == 70.0
        assert t.freshness_warning_days == 3
        assert t.freshness_critical_days == 5

    def test_custom_values(self):
        """Should accept custom values."""
        from hrp.ops.thresholds import OpsThresholds

        t = OpsThresholds(health_score_warning=85.0)
        assert t.health_score_warning == 85.0


class TestLoadThresholds:
    """Tests for load_thresholds function."""

    def test_loads_defaults(self):
        """Should load defaults when no config exists."""
        from hrp.ops.thresholds import load_thresholds

        t = load_thresholds(config_path="/nonexistent/path.yaml")
        assert t.health_score_warning == 90.0

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("HRP_THRESHOLD_HEALTH_SCORE_WARNING", "85")

        from hrp.ops.thresholds import load_thresholds

        t = load_thresholds(config_path="/nonexistent/path.yaml")
        assert t.health_score_warning == 85.0

    def test_yaml_config(self, tmp_path):
        """YAML config should override defaults."""
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text("health_score_warning: 80.0\n")

        from hrp.ops.thresholds import load_thresholds

        t = load_thresholds(config_path=str(config_file))
        assert t.health_score_warning == 80.0
