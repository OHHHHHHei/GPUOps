"""Local web console for GPUOps."""

from gpuops.web.app import create_app, run_web, validate_loopback_host
from gpuops.web.provider import DashboardProvider, DemoProvider, LiveProvider

__all__ = [
    "DashboardProvider",
    "DemoProvider",
    "LiveProvider",
    "create_app",
    "run_web",
    "validate_loopback_host",
]
