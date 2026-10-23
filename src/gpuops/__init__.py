"""Core contracts for the GPUOps compute orchestration service."""

from gpuops.models import JobRecord, JobSpec, JobStatus

__all__ = ["JobRecord", "JobSpec", "JobStatus"]
__version__ = "0.2.1"
