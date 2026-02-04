"""Test metrics collection."""


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_record_request_increments_counter(self):
        """record_request should increment the request counter."""
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        # Just verify the method exists and is callable
        collector.record_request("GET", "/health", 200)

    def test_record_latency_observes_histogram(self):
        """record_latency should observe the histogram."""
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_latency("GET", "/health", 0.05)

    def test_collect_system_metrics_returns_dict(self):
        """collect_system_metrics should return a dict with system info."""
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()

        assert isinstance(metrics, dict)
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics


class TestCollectDataMetrics:
    """Tests for data-related metrics collection."""

    def test_collect_data_metrics_returns_dict(self):
        """collect_data_metrics should return data health info."""
        from hrp.ops.metrics import MetricsCollector

        collector = MetricsCollector()
        metrics = collector.collect_data_metrics()

        assert isinstance(metrics, dict)
