"""Check CUDA on a GPU just before the scheduler leases it.

Periodic sampling no longer starts CUDA on every idle GPU. Instead the
scheduler asks this checker about the GPUs it is about to lease, while a job is
waiting. A result is reused for a few minutes, and a failed GPU is retried with
backoff so a broken card is skipped instead of failing the user's job.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 10.0
READY_CACHE_SECONDS = 300.0
FAILURE_BACKOFF_INITIAL_SECONDS = 60.0
FAILURE_BACKOFF_MAX_SECONDS = 300.0


class GpuIdentity(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def uuid(self) -> str: ...


class CudaReadiness:
    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        timeout_seconds: float = CHECK_TIMEOUT_SECONDS,
    ) -> None:
        self.monotonic_clock = monotonic_clock
        self.popen = popen
        self.timeout_seconds = timeout_seconds
        self._ready_until: dict[str, float] = {}
        self._retry_at: dict[str, float] = {}
        self._backoff: dict[str, float] = {}
        # A check stuck in the driver cannot be reaped; keep it so the same GPU
        # never accumulates more than one such process.
        self._stuck: dict[str, subprocess.Popen[str]] = {}

    def __call__(self, gpu: GpuIdentity) -> bool:
        now = self.monotonic_clock()
        uuid = gpu.uuid
        if self._ready_until.get(uuid, 0.0) > now:
            return True
        stuck = self._stuck.get(uuid)
        if stuck is not None:
            if stuck.poll() is None:
                return False
            del self._stuck[uuid]
        if self._retry_at.get(uuid, 0.0) > now:
            return False
        error = self._check(uuid)
        if error is None:
            self._ready_until[uuid] = self.monotonic_clock() + READY_CACHE_SECONDS
            self._retry_at.pop(uuid, None)
            self._backoff.pop(uuid, None)
            return True
        backoff = min(
            self._backoff.get(uuid, FAILURE_BACKOFF_INITIAL_SECONDS / 2) * 2,
            FAILURE_BACKOFF_MAX_SECONDS,
        )
        self._backoff[uuid] = backoff
        self._retry_at[uuid] = self.monotonic_clock() + backoff
        self._ready_until.pop(uuid, None)
        logger.warning("GPU %s (%s) is not usable through CUDA: %s", gpu.index, uuid, error)
        return False

    def _check(self, uuid: str) -> str | None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = uuid
        try:
            process = self.popen(
                [sys.executable, "-m", "gpuops.worker.cuda_check", uuid],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            return str(error) or error.__class__.__name__
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            self._stuck[uuid] = process
            return "CUDA check timed out"
        if process.returncode != 0:
            return (stderr or stdout or "CUDA check failed").strip()[:300]
        if stdout.strip() != uuid:
            return "CUDA returned a different device identity"
        return None
