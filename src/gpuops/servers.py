"""Connect a prepared Linux GPU host through its existing SSH alias."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpuops.config import (
    ServerTarget,
    load_targets,
    register_target,
    select_targets,
    unregister_target,
    validate_server_id,
)
from gpuops.control import RemoteTarget
from gpuops.deployment.client import RemoteDeploymentManager
from gpuops.models import ACTIVE_JOB_STATUSES
from gpuops.ssh import SSHClient
from gpuops.transport import SshWorkerTransport

WORKER_START_TIMEOUT_SECONDS = 75

_PREREQUISITES = """
import pathlib, shutil, sys
if sys.platform != 'linux':
    raise SystemExit('GPUOps Worker requires Linux')
if sys.version_info < (3, 10):
    raise SystemExit('GPUOps Worker requires Python 3.10 or newer')
import venv
if not shutil.which('nvidia-smi') or not pathlib.Path('/proc/driver/nvidia/gpus').is_dir():
    raise SystemExit('NVIDIA driver and nvidia-smi must already be installed')
"""

async def add_server(ssh_alias: str, *, project_root: Path | None = None) -> dict[str, Any]:
    validate_server_id(ssh_alias)
    ssh = SSHClient()
    prerequisites = await ssh.run_python(ssh_alias, _PREREQUISITES, timeout_seconds=15)
    if prerequisites.returncode:
        raise RuntimeError(prerequisites.stderr.strip() or prerequisites.stdout.strip())
    manager = RemoteDeploymentManager(ssh=ssh)
    status = await manager.status(ssh_alias)
    existing = next((item for item in load_targets() if item.ssh_host == ssh_alias), None)
    server_id = status.server_id or (existing.id if existing else ssh_alias)
    validate_server_id(server_id)
    if existing is not None and existing.id != server_id:
        raise ValueError(
            f"{ssh_alias} identifies as {server_id}, but is registered as {existing.id}"
        )
    conflict = next((item for item in load_targets() if item.id == server_id), None)
    if conflict is not None and conflict.ssh_host != ssh_alias:
        raise ValueError(f"server {server_id!r} is already registered at {conflict.ssh_host}")
    target = existing or ServerTarget(server_id, server_id, ssh_alias)
    remote = RemoteTarget(server_id, ssh_alias)
    transport = SshWorkerTransport(ssh=ssh)
    installed = False
    if not status.installed:
        await manager.deploy(ssh_alias, project_root, server_id=server_id)
        installed = True
    elif not status.worker_ready:
        await transport.worker_command(remote, "start")
    await _wait_worker_ready(transport, remote)
    register_target(target)
    return {
        "server": asdict(target),
        "installed": installed,
        "ready": True,
    }


async def _wait_worker_ready(transport: SshWorkerTransport, remote: RemoteTarget) -> None:
    deadline = asyncio.get_running_loop().time() + WORKER_START_TIMEOUT_SECONDS
    while True:
        worker = await transport.worker_command(remote, "status")
        if worker.get("ready") is True:
            return
        if asyncio.get_running_loop().time() >= deadline:
            reason = worker.get("reason") or "startup timed out"
            raise RuntimeError(f"Worker is not ready on {remote.ssh_host}: {reason}")
        await asyncio.sleep(1)


async def remove_server(server_id: str) -> dict[str, Any]:
    target = select_targets([server_id])[0]
    remote = RemoteTarget(target.id, target.ssh_host)
    transport = SshWorkerTransport()
    if await transport.list_summaries(remote, statuses=ACTIVE_JOB_STATUSES):
        raise ValueError(
            "server has active jobs; wait for completion or cancel them before removing"
        )
    before = await transport.worker_command(remote, "status")
    try:
        await transport.worker_command(remote, "stop")
        deadline = asyncio.get_running_loop().time() + 75
        while True:
            state = await transport.worker_command(remote, "status")
            if state.get("status") in {"stopped", "stale"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("Worker is still stopping; server registration has been kept")
            await asyncio.sleep(1)
        # A submission may have arrived while the Worker was shutting down.
        if await transport.list_summaries(remote, statuses=ACTIVE_JOB_STATUSES):
            raise ValueError("new active jobs appeared during shutdown; server registration kept")
        unregister_target(server_id)
    except Exception:
        if before.get("status") == "running":
            with suppress(Exception):
                await transport.worker_command(remote, "start")
        raise
    return {"removed": server_id, "remote_data_preserved": True}
