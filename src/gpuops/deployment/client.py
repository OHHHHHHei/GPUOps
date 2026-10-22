"""SSH orchestration and response mapping for remote deployments."""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from gpuops.config import validate_server_id
from gpuops.deployment.artifacts import (
    _default_command_runner,
    build_artifact,
    copy_artifact,
    validate_host,
)
from gpuops.deployment.bootstrap import (
    REMOTE_ABORT_SCRIPT,
    REMOTE_DEPLOY_SCRIPT,
    REMOTE_PREPARE_SCRIPT,
    REMOTE_ROLLBACK_SCRIPT,
    REMOTE_STATUS_SCRIPT,
    REMOTE_UNINSTALL_SCRIPT,
)
from gpuops.deployment.contracts import (
    REMOTE_SHARE_BUDGET_BYTES,
    REMOTE_TOTAL_BUDGET_BYTES,
    CommandRunner,
    DeployError,
    DeploymentResult,
    DeploymentStatus,
    ReleaseInfo,
)
from gpuops.ssh import SSHClient, SSHResult


class RemoteDeploymentManager:
    """Package the current client and deploy it on one remote host."""

    def __init__(
        self,
        *,
        ssh: SSHClient | None = None,
        command_runner: CommandRunner | None = None,
        remote_timeout_seconds: float = 900.0,
    ) -> None:
        if remote_timeout_seconds <= 0:
            raise ValueError("remote_timeout_seconds must be positive")
        self.ssh = ssh or SSHClient()
        self.command_runner = command_runner or _default_command_runner
        self.remote_timeout_seconds = remote_timeout_seconds

    async def deploy(
        self, host: str, project_root: str | Path | None = None, *, server_id: str | None = None
    ) -> DeploymentResult:
        validate_host(host)
        if server_id is not None:
            validate_server_id(server_id)
        with tempfile.TemporaryDirectory(prefix="gpuops-deploy-") as temporary_directory:
            artifact = build_artifact(
                temporary_directory,
                project_root=project_root,
                command_runner=self.command_runner,
            )
            deployment_id = uuid.uuid4().hex
            prepared = await self._run(
                host,
                REMOTE_PREPARE_SCRIPT,
                arguments=(
                    artifact.release_id,
                    str(artifact.size_bytes),
                    deployment_id,
                    artifact.filename,
                ),
                timeout_seconds=30.0,
            )
            available = int(prepared.get("available_upload_bytes", -1))
            if available < artifact.size_bytes:
                raise DeployError(
                    "remote GPUOps budget cannot fit the deployment artifact "
                    f"({artifact.size_bytes} bytes required, {max(0, available)} available)"
                )
            try:
                copy_artifact(
                    host,
                    artifact,
                    deployment_id=deployment_id,
                    command_runner=self.command_runner,
                )
            except Exception:
                with suppress(Exception):
                    await self._run(
                        host,
                        REMOTE_ABORT_SCRIPT,
                        arguments=(artifact.release_id, deployment_id),
                        timeout_seconds=30.0,
                    )
                raise
            payload = await self._run(
                host,
                REMOTE_DEPLOY_SCRIPT,
                arguments=(artifact.release_id, deployment_id, server_id or "", host),
            )
        result = _deployment_result(host, payload)
        if result.status.current_release != artifact.release_id:
            raise DeployError("remote current release does not match the requested release ID")
        return result

    async def status(self, host: str) -> DeploymentStatus:
        validate_host(host)
        payload = await self._run(host, REMOTE_STATUS_SCRIPT, timeout_seconds=30.0)
        return _deployment_status(host, payload["status"])

    async def rollback(self, host: str) -> DeploymentResult:
        validate_host(host)
        return _deployment_result(host, await self._run(host, REMOTE_ROLLBACK_SCRIPT))

    async def uninstall(self, host: str) -> DeploymentResult:
        validate_host(host)
        return _deployment_result(host, await self._run(host, REMOTE_UNINSTALL_SCRIPT))

    async def _run(
        self,
        host: str,
        script: str,
        *,
        arguments: Sequence[str] = (),
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        effective_timeout = timeout_seconds or self.remote_timeout_seconds
        if arguments:
            result = await self.ssh.run_command(
                host,
                ["python3", "-u", "-", *arguments],
                stdin=script.encode("utf-8"),
                timeout_seconds=effective_timeout,
            )
        else:
            result = await self.ssh.run_python(
                host,
                script,
                timeout_seconds=effective_timeout,
            )
        return _parse_remote_result(result)


def _parse_remote_result(result: SSHResult) -> dict[str, Any]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] | None = None
    if len(lines) == 1:
        try:
            decoded = json.loads(lines[0])
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            payload = decoded
    if result.returncode != 0 or payload is None or not payload.get("ok"):
        detail = ""
        if payload is not None:
            detail = str(payload.get("error") or payload.get("message") or "")
        detail = detail or result.stderr.strip() or result.stdout.strip()
        raise DeployError(f"remote deployment failed on {result.host}: {detail or 'unknown error'}")
    return payload


def _deployment_result(host: str, payload: Mapping[str, Any]) -> DeploymentResult:
    return DeploymentResult(
        action=str(payload["action"]),
        host=host,
        changed=bool(payload.get("changed")),
        message=str(payload.get("message") or ""),
        status=_deployment_status(host, payload["status"]),
    )


def _deployment_status(host: str, value: Mapping[str, Any]) -> DeploymentStatus:
    releases = tuple(
        ReleaseInfo(
            release_id=str(item["release_id"]),
            installed_at=(str(item["installed_at"]) if item.get("installed_at") else None),
            size_bytes=int(item["size_bytes"]),
        )
        for item in value.get("releases", [])
    )
    return DeploymentStatus(
        host=host,
        installed=bool(value.get("installed")),
        current_release=(str(value["current_release"]) if value.get("current_release") else None),
        previous_release=(
            str(value["previous_release"]) if value.get("previous_release") else None
        ),
        releases=releases,
        share_bytes=int(value.get("share_bytes", 0)),
        budget_bytes=int(value.get("budget_bytes", REMOTE_SHARE_BUDGET_BYTES)),
        cache_bytes=int(value.get("cache_bytes", 0)),
        state_bytes=int(value.get("state_bytes", 0)),
        total_bytes=int(value.get("total_bytes", 0)),
        total_budget_bytes=int(value.get("total_budget_bytes", REMOTE_TOTAL_BUDGET_BYTES)),
        worker_ready=bool(value.get("worker_ready")),
        worker_release=(str(value["worker_release"]) if value.get("worker_release") else None),
        executable=str(value["executable"]) if value.get("executable") else None,
        server_id=str(value["server_id"]) if value.get("server_id") else None,
    )
