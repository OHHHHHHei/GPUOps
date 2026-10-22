"""Read worker readiness independently of direct GPU monitoring."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from gpuops.cli_errors import error_payload
from gpuops.config import ServerTarget
from gpuops.control import RemoteTarget
from gpuops.transport import SshWorkerTransport


async def collect_worker_health(
    targets: Sequence[ServerTarget], transport: SshWorkerTransport
) -> dict[str, dict[str, Any]]:
    async def collect(target: ServerTarget) -> tuple[str, dict[str, Any]]:
        try:
            # Unlike RPC construction, worker status never initializes queue storage.
            status = await transport.worker_command(
                RemoteTarget(target.id, target.ssh_host), "status"
            )
            ready = status.get("ready") is True
            reason = status.get("reason") or ("ready" if ready else "worker is not ready")
            heartbeat = status.get("heartbeat")
            if ready and (
                not isinstance(heartbeat, dict) or heartbeat.get("server_id") != target.id
            ):
                ready = False
                reason = "worker heartbeat belongs to a different server id"
            return target.id, {
                "reachable": True,
                "ready": ready,
                "reason": reason,
                "error": None,
                "error_code": None,
            }
        except Exception as error:
            failure = error_payload(error)
            return target.id, {
                "reachable": False,
                "ready": False,
                "reason": "Worker readiness could not be verified.",
                "error": failure["error"],
                "error_code": failure["error_code"],
            }

    return dict(await asyncio.gather(*(collect(target) for target in targets)))


def format_worker_health(workers: dict[str, dict[str, Any]]) -> str:
    lines = []
    for server_id, worker in workers.items():
        state = "ready" if worker["ready"] else "unavailable"
        detail = worker["error"] or worker["reason"]
        lines.append(f"{server_id} worker: {state} ({detail})")
    return "\n".join(lines)
