from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Launcher", "Worker", "WorkerTickResult"]

if TYPE_CHECKING:
    from gpuops.worker.service import Launcher, Worker, WorkerTickResult


def __getattr__(name: str) -> Any:
    # Keep `python -m gpuops.worker.<helper>` free of the scheduler's imports.
    if name in __all__:
        from gpuops.worker import service

        return getattr(service, name)
    raise AttributeError(f"module 'gpuops.worker' has no attribute {name!r}")
