"""FastAPI ops server with health endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, Response


def check_system_ready() -> tuple[bool, dict]:
    """
    Check if system is ready to serve requests.

    Returns:
        Tuple of (is_ready, details_dict)
    """
    details = {}
    is_ready = True

    try:
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(read_only=True)
        health = api.health_check()
        api.close()

        details["database"] = health.get("database", "unknown")
        details["api"] = health.get("api", "unknown")

        if health.get("database") != "ok":
            is_ready = False
    except Exception as e:
        details["error"] = str(e)
        is_ready = False

    return is_ready, details


def create_app() -> FastAPI:
    """Create FastAPI application for ops endpoints."""
    app = FastAPI(
        title="HRP Ops",
        description="Health and metrics endpoints for HRP",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness probe - returns 200 if API is responsive."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/ready")
    def ready(response: Response):
        """Readiness probe - returns 200 if system is ready, 503 otherwise."""
        is_ready, details = check_system_ready()

        result = {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now().isoformat(),
            "checks": details,
        }

        if not is_ready:
            response.status_code = 503

        return result

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the ops server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
