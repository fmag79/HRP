"""Metrics collection for HRP ops.

Collects system and application metrics for monitoring.
"""

from __future__ import annotations

import psutil
from loguru import logger
from prometheus_client import Gauge

# Import existing metrics from server module to avoid duplication
from hrp.ops.server import REQUEST_COUNT, REQUEST_LATENCY

# System metrics gauges
SYSTEM_CPU = Gauge("hrp_system_cpu_percent", "System CPU usage percentage")
SYSTEM_MEMORY = Gauge("hrp_system_memory_percent", "System memory usage percentage")
SYSTEM_DISK = Gauge("hrp_system_disk_percent", "System disk usage percentage")


class MetricsCollector:
    """Collects and records metrics for HRP."""

    def record_request(self, method: str, endpoint: str, status: int) -> None:
        """
        Record an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Request path
            status: HTTP status code
        """
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()

    def record_latency(self, method: str, endpoint: str, duration: float) -> None:
        """
        Record request latency.

        Args:
            method: HTTP method
            endpoint: Request path
            duration: Request duration in seconds
        """
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    def collect_system_metrics(self) -> dict:
        """
        Collect system metrics.

        Returns:
            Dict with cpu_percent, memory_percent, disk_percent
        """
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        # Update Prometheus gauges
        SYSTEM_CPU.set(cpu)
        SYSTEM_MEMORY.set(memory)
        SYSTEM_DISK.set(disk)

        return {
            "cpu_percent": cpu,
            "memory_percent": memory,
            "disk_percent": disk,
        }

    def collect_data_metrics(self) -> dict:
        """
        Collect data pipeline metrics.

        Returns:
            Dict with data health information
        """
        metrics: dict = {
            "status": "unknown",
        }

        try:
            from hrp.api.platform import PlatformAPI

            api = PlatformAPI(read_only=True)
            health = api.run_quality_checks()
            api.close()

            metrics["status"] = "ok" if health.get("overall_health", 0) >= 70 else "degraded"
            metrics["health_score"] = health.get("overall_health", 0)
        except Exception as e:
            logger.warning(f"Failed to collect data metrics: {e}")
            metrics["status"] = "error"
            metrics["error"] = str(e)

        return metrics
