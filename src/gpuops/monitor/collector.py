"""Remote GPU snapshot collection."""

from __future__ import annotations

import json
from typing import Any

from gpuops.config import MONITOR_TIMEOUT_SECONDS
from gpuops.execution import HostClient
from gpuops.ssh import SSHClient

from .remote import COLLECTOR_SCRIPT, DOCTOR_SCRIPT


class CollectionError(RuntimeError):
    """GPU collection failed, with a machine-readable failure category."""

    def __init__(self, message: str, *, code: str = "collection_failed") -> None:
        super().__init__(message)
        self.code = code


class RemoteCollector:
    def __init__(self, ssh: SSHClient | None = None) -> None:
        self.ssh = ssh or HostClient()

    async def collect(
        self, host: str, *, timeout_seconds: float = MONITOR_TIMEOUT_SECONDS
    ) -> tuple[dict[str, Any], float]:
        result = await self.ssh.run_python(host, COLLECTOR_SCRIPT, timeout_seconds=timeout_seconds)
        payload = _decode_single_json_line(host, result.stdout, result.stderr, result.returncode)
        if isinstance(payload, dict) and isinstance(payload.get("collection_error"), dict):
            error = payload["collection_error"]
            code = error.get("code")
            raise CollectionError(
                f"{host}: {error.get('message') or 'GPU telemetry is unavailable'}",
                code=code
                if code
                in {
                    "gpu_query_timeout",
                    "gpu_query_failed",
                    "telemetry_unavailable",
                    "telemetry_stale",
                }
                else "collection_failed",
            )
        if not isinstance(payload, dict) or not isinstance(payload.get("gpus"), list):
            raise CollectionError(f"{host}: collector returned an invalid snapshot")
        return payload, result.duration_seconds

    async def doctor(
        self, host: str, *, timeout_seconds: float = MONITOR_TIMEOUT_SECONDS
    ) -> tuple[dict[str, Any], float]:
        result = await self.ssh.run_python(host, DOCTOR_SCRIPT, timeout_seconds=timeout_seconds)
        payload = _decode_single_json_line(host, result.stdout, result.stderr, result.returncode)
        if not isinstance(payload, dict):
            raise CollectionError(f"{host}: doctor returned an invalid response")
        return payload, result.duration_seconds


def _decode_single_json_line(host: str, stdout: str, stderr: str, returncode: int) -> Any:
    if returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit status {returncode}"
        raise CollectionError(
            f"{host}: {detail}",
            code="connection_failed" if returncode == 255 else "collection_failed",
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CollectionError(
            f"{host}: expected one JSON line from collector, received {len(lines)}"
        )
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CollectionError(f"{host}: collector returned malformed JSON") from exc
