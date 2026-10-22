"""FastAPI application factory for the local GPUOps console."""

from __future__ import annotations

import ipaddress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gpuops.config import MONITOR_TIMEOUT_SECONDS
from gpuops.monitor import MonitorService
from gpuops.web.provider import (
    DashboardProvider,
    DemoProvider,
    LiveProvider,
    ProviderConflict,
    ProviderUnavailable,
)

WEB_ROOT = Path(__file__).resolve().parent


def create_app(provider: DashboardProvider | None = None) -> FastAPI:
    """Create a local console app backed by a dashboard provider."""

    dashboard = provider or DemoProvider()
    app = FastAPI(
        title="GPUOps Console",
        description="Local console for GPU status and managed job scheduling.",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.dashboard_provider = dashboard
    templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")

    @app.middleware("http")
    async def local_console_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"poll_interval_ms": 5_000},
        )

    @app.get("/api/v1/snapshot")
    async def snapshot():  # type: ignore[no-untyped-def]
        return await dashboard.get_snapshot()

    @app.get("/api/v1/process-etas")
    async def process_etas():  # type: ignore[no-untyped-def]
        return await dashboard.get_process_etas()

    @app.get("/api/v1/queue")
    async def queue():  # type: ignore[no-untyped-def]
        return await dashboard.get_queue()

    @app.get("/api/v1/history")
    async def history(
        limit: int = Query(50, ge=1, le=100),
        project: str | None = Query(None, min_length=1),
        cursor: str | None = Query(None, max_length=1024),
    ):  # type: ignore[no-untyped-def]
        try:
            return await dashboard.get_history(limit=limit, project=project, cursor=cursor)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}")
    async def job(job_id: str):  # type: ignore[no-untyped-def]
        try:
            result = await dashboard.get_job(job_id)
        except ProviderConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ProviderUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):  # type: ignore[no-untyped-def]
        try:
            result = await dashboard.cancel_job(job_id)
        except ProviderConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ProviderUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result

    @app.get("/healthz")
    async def health():  # type: ignore[no-untyped-def]
        return await dashboard.health()

    return app


def validate_loopback_host(host: str) -> str:
    if host == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("web host must be a loopback IP address or localhost") from exc
    if not address.is_loopback:
        raise ValueError("web host must be a loopback address")
    return host


def run_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    demo: bool = False,
    timeout_seconds: float = MONITOR_TIMEOUT_SECONDS,
) -> None:
    import uvicorn

    validate_loopback_host(host)
    if not 1 <= port <= 65535:
        raise ValueError("web port must be between 1 and 65535")
    if timeout_seconds <= 0:
        raise ValueError("server timeout must be positive")
    if demo:
        provider: DashboardProvider = DemoProvider()
    else:
        from gpuops.transport import SshWorkerTransport
        from gpuops.web.control_provider import ControlDashboardProvider
        from gpuops.web.log_eta import LogEtaCollector

        live = LiveProvider(
            MonitorService(timeout_seconds=timeout_seconds),
            log_eta=LogEtaCollector(timeout_seconds=min(timeout_seconds, 5.0)),
        )
        provider = ControlDashboardProvider(
            live,
            SshWorkerTransport(timeout_seconds=timeout_seconds),
        )
    uvicorn.run(create_app(provider), host=host, port=port)
