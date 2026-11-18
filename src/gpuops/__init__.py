"""Core contracts for the GPUOps compute orchestration service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["JobRecord", "JobSpec", "JobStatus"]
__version__ = "0.5.0"

if TYPE_CHECKING:
    from gpuops.models import JobRecord, JobSpec, JobStatus


def __getattr__(name: str) -> Any:
    # Import the models on first use so short-lived helper processes, such as
    # the GPU sampler, can import submodules without loading pydantic.
    if name in __all__:
        from gpuops import models

        return getattr(models, name)
    raise AttributeError(f"module 'gpuops' has no attribute {name!r}")
