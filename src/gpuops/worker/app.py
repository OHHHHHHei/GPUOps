"""Composition root and lifecycle commands for a host-local GPUOps worker."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from gpuops.config import SchedulerSettings
from gpuops.core.queue import QueueStore
from gpuops.core.scheduler import Scheduler
from gpuops.core.storage import StorageBudget
from gpuops.models import utc_now
from gpuops.worker.runtime import (
    RuntimePaths,
    WorkerHeartbeat,
    WorkerProcessState,
    WorkerProcessStatus,
    WorkerRuntime,
    inspect_worker,
    request_worker_stop,
    spawn_detached_worker,
)
from gpuops.worker.sampling import AsyncGpuSampler, CachedGpuProbe
from gpuops.worker.service import Worker
from gpuops.worker.supervisor import DurableJobSupervisor, DurableSupervisorLauncher

DEFAULT_STATE_DIRECTORY = Path("~/.local/state/gpuops").expanduser()
REMOTE_TOTAL_BUDGET_BYTES = 1024 * 1024 * 1024
REMOTE_INSTALL_BUDGET_BYTES = 256 * 1024 * 1024
STATE_BUDGET_BYTES = REMOTE_TOTAL_BUDGET_BYTES - REMOTE_INSTALL_BUDGET_BYTES
LOG_BUDGET_BYTES = 640 * 1024 * 1024
WORKER_LOG_MAX_BYTES = 10 * 1024 * 1024
WORKER_LOG_BACKUPS = 2
WORKER_HEARTBEAT_MAX_AGE_SECONDS = 75.0
WORKER_HEARTBEAT_FUTURE_SKEW_SECONDS = 5.0


def worker_readiness(
    status: WorkerProcessStatus,
    *,
    server_id: str | None = None,
    now: datetime | None = None,
    max_age_seconds: float = WORKER_HEARTBEAT_MAX_AGE_SECONDS,
) -> tuple[bool, str]:
    """Validate that the PID identity and heartbeat describe one live worker."""

    if status.state != WorkerProcessState.RUNNING or status.identity is None:
        return False, f"worker process is {status.state.value}"
    if status.heartbeat is None:
        return False, "worker heartbeat is missing"
    try:
        heartbeat = WorkerHeartbeat.model_validate(status.heartbeat)
    except Exception:
        return False, "worker heartbeat is invalid"
    if heartbeat.pid != status.identity.pid:
        return False, "worker heartbeat PID does not match the verified process"
    if server_id is not None and heartbeat.server_id != server_id:
        return False, "worker heartbeat belongs to a different server id"
    if heartbeat.status != "running":
        return False, f"worker heartbeat status is {heartbeat.status!r}"
    current = now or utc_now()
    if heartbeat.heartbeat_at.tzinfo is None or current.tzinfo is None:
        return False, "worker heartbeat timestamp is not timezone-aware"
    age = (current - heartbeat.heartbeat_at).total_seconds()
    if age < -WORKER_HEARTBEAT_FUTURE_SKEW_SECONDS:
        return False, "worker heartbeat is implausibly in the future"
    if age > max_age_seconds:
        return False, f"worker heartbeat is stale ({age:.1f}s old)"
    return True, "ready"


def build_worker_runtime(
    *,
    server_id: str,
    state_directory: Path = DEFAULT_STATE_DIRECTORY,
    interval_seconds: float = 5.0,
) -> WorkerRuntime:
    state = state_directory.expanduser()
    worker_id = f"{server_id}:{socket.gethostname()}"
    storage = StorageBudget(
        state,
        total_budget_bytes=STATE_BUDGET_BYTES,
        log_budget_bytes=LOG_BUDGET_BYTES,
    )
    store = QueueStore(state / "queue", storage=storage)
    cached_probe = CachedGpuProbe(state, server_id)
    scheduler = Scheduler(
        store,
        cached_probe,
        server_id=server_id,
        worker_id=worker_id,
        settings=SchedulerSettings(),
        confirmation_provider=cached_probe,
    )
    supervisor = DurableJobSupervisor(state / "runs", storage=storage)
    worker = Worker(store, scheduler, DurableSupervisorLauncher(supervisor))
    return WorkerRuntime(
        server_id=server_id,
        worker_id=worker_id,
        paths=RuntimePaths.from_state_directory(state),
        tick=worker.tick,
        interval_seconds=interval_seconds,
        sampler_factory=lambda: AsyncGpuSampler(state, server_id),
    )


def run_worker(
    *,
    server_id: str,
    state_directory: Path = DEFAULT_STATE_DIRECTORY,
    interval_seconds: float = 5.0,
    once: bool = False,
) -> int:
    runtime = build_worker_runtime(
        server_id=server_id,
        state_directory=state_directory,
        interval_seconds=interval_seconds,
    )
    runtime.run_foreground(max_iterations=1 if once else None)
    return 0


def start_worker(
    *,
    server_id: str,
    state_directory: Path = DEFAULT_STATE_DIRECTORY,
    interval_seconds: float = 5.0,
    wait_seconds: float = 3.0,
) -> dict[str, Any]:
    state = state_directory.expanduser()
    paths = RuntimePaths.from_state_directory(state)
    status = inspect_worker(paths)
    if status.state == WorkerProcessState.RUNNING:
        ready, reason = worker_readiness(status, server_id=server_id)
        if ready:
            return {"started": False, "status": status.state.value, "pid": status.identity.pid}
        request_worker_stop(paths)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while time.monotonic() < deadline:
            if inspect_worker(paths).state != WorkerProcessState.RUNNING:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"cannot replace unhealthy worker: {reason}")

    storage = StorageBudget(
        state,
        total_budget_bytes=STATE_BUDGET_BYTES,
        log_budget_bytes=LOG_BUDGET_BYTES,
    )
    log_path = storage.prepare_log_file("worker/worker.log")
    _rotate_worker_log(log_path)
    argv = [
        sys.executable,
        "-m",
        "gpuops",
        "worker",
        "run",
        "--server-id",
        server_id,
        "--state-dir",
        str(state),
        "--interval",
        str(interval_seconds),
    ]
    spawned_pid = spawn_detached_worker(argv, stdout_path=log_path)
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        current = inspect_worker(paths)
        ready, _reason = worker_readiness(current, server_id=server_id)
        if ready:
            assert current.identity is not None
            return {"started": True, "status": "running", "pid": current.identity.pid}
        time.sleep(0.05)
    return {"started": True, "status": "starting", "pid": spawned_pid}


def stop_worker(*, state_directory: Path = DEFAULT_STATE_DIRECTORY) -> dict[str, Any]:
    paths = RuntimePaths.from_state_directory(state_directory.expanduser())
    stopped = request_worker_stop(paths)
    return {"stop_requested": stopped, "status": "stopping" if stopped else "not-running"}


def worker_status(*, state_directory: Path = DEFAULT_STATE_DIRECTORY) -> dict[str, Any]:
    status = inspect_worker(RuntimePaths.from_state_directory(state_directory.expanduser()))
    heartbeat_server_id = (
        str(status.heartbeat.get("server_id"))
        if status.heartbeat is not None and status.heartbeat.get("server_id")
        else None
    )
    ready, reason = worker_readiness(status, server_id=heartbeat_server_id)
    return {
        "status": status.state.value,
        "ready": ready,
        "reason": reason,
        "identity": (
            {
                "pid": status.identity.pid,
                "start_time_ticks": status.identity.start_time_ticks,
                "boot_id": status.identity.boot_id,
            }
            if status.identity is not None
            else None
        ),
        "heartbeat": status.heartbeat,
    }


def render_worker_result(value: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    print(value.get("status", "unknown"))


def _rotate_worker_log(path: Path) -> None:
    try:
        if path.stat().st_size < WORKER_LOG_MAX_BYTES:
            return
    except FileNotFoundError:
        return
    oldest = path.with_name(f"{path.name}.{WORKER_LOG_BACKUPS}")
    oldest.unlink(missing_ok=True)
    for index in range(WORKER_LOG_BACKUPS - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            os.replace(source, path.with_name(f"{path.name}.{index + 1}"))
    os.replace(path, path.with_name(f"{path.name}.1"))
