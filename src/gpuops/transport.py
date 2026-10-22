"""Fixed Worker JSON RPC over SSH, or directly on the installed server."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpuops.config import MONITOR_TIMEOUT_SECONDS
from gpuops.control import DefinitiveRemoteSubmissionError, RemoteHostState, RemoteTarget
from gpuops.core.resources import (
    DEFAULT_RESOURCE_ELIGIBILITY_POLICY,
    GpuResource,
    ServerResourceSnapshot,
    normalize_nonnegative_number,
    telemetry_from_mapping,
)
from gpuops.execution import HostClient
from gpuops.models import ACTIVE_JOB_STATUSES, JobRecord, JobSpec, JobStatus, JobSummary, utc_now
from gpuops.monitor.collector import RemoteCollector
from gpuops.ssh import SSHClient
from gpuops.worker.rpc import RPC_API_VERSION


class RemoteWorkerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RemoteSubmissionRejected(RemoteWorkerError, DefinitiveRemoteSubmissionError):
    """A structured remote response proving the requested job was not accepted."""

    def __init__(self, code: str, message: str) -> None:
        RuntimeError.__init__(self, message)
        self.code = code


@dataclass(frozen=True)
class JobDetail:
    record: JobRecord
    explanation: dict[str, Any]


@dataclass(frozen=True)
class JobHistory:
    jobs: list[JobSummary]
    projects: list[str]


_DEFINITIVE_SUBMISSION_REJECTION_CODES = frozenset(
    {
        "host_mismatch",
        "invalid_params",
        "method_not_found",
        "run_id_conflict",
        "worker_unavailable",
    }
)


class SshWorkerTransport:
    def __init__(
        self,
        *,
        ssh: SSHClient | None = None,
        collector: RemoteCollector | None = None,
        timeout_seconds: float = MONITOR_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.ssh = ssh or HostClient()
        self.collector = collector or RemoteCollector(self.ssh)
        self.timeout_seconds = timeout_seconds

    async def inspect(self, target: RemoteTarget) -> RemoteHostState:
        snapshot_result, queue_result, health = await asyncio.gather(
            self.collector.collect(target.ssh_host, timeout_seconds=self.timeout_seconds),
            self.list_jobs(target, statuses=ACTIVE_JOB_STATUSES),
            self.health(target),
        )
        self._require_ready_worker(health)
        payload, _latency = snapshot_result
        return RemoteHostState(
            snapshot=_resource_snapshot(target.server_id, payload),
            queue=tuple(queue_result),
        )

    async def submit(self, target: RemoteTarget, run_id: str, spec: JobSpec) -> JobRecord:
        try:
            self._require_ready_worker(await self.health(target))
            result = await self._rpc(
                target,
                "submit",
                {"run_id": run_id, "job_spec": spec.model_dump(mode="json")},
            )
            if result.get("durable") is not True:
                raise RemoteWorkerError("not_durable", "remote worker did not confirm durability")
            return _job_record(result, run_id)
        except RemoteWorkerError as error:
            if error.code in _DEFINITIVE_SUBMISSION_REJECTION_CODES:
                raise RemoteSubmissionRejected(error.code, str(error)) from error
            raise

    async def get_job(self, target: RemoteTarget, run_id: str) -> JobRecord | None:
        try:
            result = await self._rpc(target, "show", {"run_id": run_id})
        except RemoteWorkerError as error:
            if error.code == "job_not_found":
                return None
            raise
        return _job_record(result, run_id)

    async def get_job_detail(self, target: RemoteTarget, run_id: str) -> JobDetail | None:
        try:
            result = await self._rpc(target, "show", {"run_id": run_id, "explain": True})
        except RemoteWorkerError as error:
            if error.code == "job_not_found":
                return None
            if error.code != "invalid_params":
                raise
            record = await self.get_job(target, run_id)
            if record is None:
                return None
            return JobDetail(record, unknown_job_explanation(record))
        record = _job_record(result, run_id)
        return JobDetail(record, _explanation(result.get("explanation"), record))

    async def cancel(
        self,
        target: RemoteTarget,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> JobRecord:
        params: dict[str, Any] = {"run_id": run_id}
        if reason is not None:
            params["reason"] = reason
        result = await self._rpc(target, "cancel", params)
        if result.get("durable") is not True:
            raise RemoteWorkerError("not_durable", "remote worker did not confirm durability")
        return _job_record(result, run_id)

    async def list_jobs(
        self, target: RemoteTarget, *, statuses: set[JobStatus] | None = None
    ) -> list[JobRecord]:
        params = (
            {} if statuses is None else {"statuses": sorted(status.value for status in statuses)}
        )
        result = await self._rpc(target, "list", params)
        return [JobRecord.model_validate(value) for value in result.get("jobs", [])]

    async def list_summaries(
        self, target: RemoteTarget, *, statuses: set[JobStatus]
    ) -> list[JobSummary]:
        try:
            result = await self._rpc(
                target,
                "list",
                {"statuses": sorted(status.value for status in statuses), "summary": True},
            )
        except RemoteWorkerError as error:
            if error.code != "invalid_params":
                raise
            # Older workers can still filter active jobs during a rolling upgrade.
            return [
                JobSummary.from_record(job)
                for job in await self.list_jobs(target, statuses=statuses)
            ]
        return [JobSummary.model_validate(value) for value in result["jobs"]]

    async def history(
        self,
        target: RemoteTarget,
        *,
        limit: int = 50,
        project: str | None = None,
        cursor: str | None = None,
    ) -> JobHistory:
        params: dict[str, Any] = {"limit": limit}
        if project is not None:
            params["project"] = project
        if cursor is not None:
            params["cursor"] = cursor
        result = await self._rpc(target, "history", params)
        return JobHistory(
            [JobSummary.model_validate(value) for value in result["jobs"]],
            result["projects"],
        )

    async def get_logs(
        self,
        target: RemoteTarget,
        run_id: str,
        *,
        tail_bytes: int = 64 * 1024,
    ) -> dict[str, Any]:
        return await self._rpc(
            target,
            "logs",
            {"run_id": run_id, "tail_bytes": tail_bytes},
        )

    async def health(self, target: RemoteTarget) -> dict[str, Any]:
        return await self._rpc(target, "health", {})

    @staticmethod
    def _require_ready_worker(health: dict[str, Any]) -> None:
        worker = health.get("worker") if isinstance(health.get("worker"), dict) else {}
        if (
            health.get("status") != "ok"
            or health.get("durable") is not True
            or worker.get("ready") is not True
        ):
            reason = worker.get("reason") or "background worker is not ready"
            raise RemoteWorkerError("worker_unavailable", str(reason))

    async def worker_command(
        self,
        target: RemoteTarget,
        action: str,
    ) -> dict[str, Any]:
        if action not in {"start", "stop", "status"}:
            raise ValueError("unsupported worker action")
        command = ["~/.local/bin/gpuops", "worker", action]
        if action == "start":
            command.extend(["--server-id", target.server_id])
        command.append("--json")
        result = await self.ssh.run_command(
            target.ssh_host,
            command,
            timeout_seconds=self.timeout_seconds,
        )
        payload = _json_document(result.stdout)
        if result.returncode != 0 or payload.get("ok") is not True:
            detail = payload.get("error") or result.stderr.strip() or "worker command failed"
            raise RemoteWorkerError("worker_command_failed", str(detail))
        return payload

    async def _rpc(
        self,
        target: RemoteTarget,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request = {
            "api_version": RPC_API_VERSION,
            "request_id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        result = await self.ssh.run_command(
            target.ssh_host,
            ["~/.local/bin/gpuops", "remote", "--server-id", target.server_id],
            stdin=json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            timeout_seconds=self.timeout_seconds,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "remote command failed"
            raise RemoteWorkerError("transport_error", detail)
        response = _single_json(result.stdout)
        if response.get("api_version") != RPC_API_VERSION:
            raise RemoteWorkerError("protocol_error", "unexpected worker RPC version")
        if response.get("request_id") != request["request_id"]:
            raise RemoteWorkerError("protocol_error", "worker response request_id mismatch")
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise RemoteWorkerError(
                str(error.get("code") or "remote_error"),
                str(error.get("message") or "remote worker rejected the request"),
            )
        result_payload = response.get("result")
        if not isinstance(result_payload, dict):
            raise RemoteWorkerError("protocol_error", "worker result must be a JSON object")
        return result_payload


def unknown_job_explanation(
    record: JobRecord,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    observed_at = observed_at or utc_now()
    return {
        "waiting": record.status.value in {"queued", "leased", "starting"},
        "code": "unknown",
        "message": "A scheduling explanation is unavailable from this worker.",
        "observed_at": observed_at.isoformat(),
        "details": {},
    }


def _job_record(result: dict[str, Any], run_id: str) -> JobRecord:
    payload = result["job"]
    execution_result = result.get("execution_result")
    if isinstance(payload, dict) and isinstance(execution_result, dict):
        payload = {
            **payload,
            **{
                key: execution_result[key]
                for key in ("exit_code", "launch_error")
                if key in execution_result
            },
        }
    record = JobRecord.model_validate(payload)
    if record.run_id != run_id:
        raise RemoteWorkerError("protocol_error", "worker returned a different run_id")
    return record


def _explanation(value: object, record: JobRecord) -> dict[str, Any]:
    if not isinstance(value, dict):
        return unknown_job_explanation(record)
    required = {
        "waiting": bool,
        "code": str,
        "message": str,
        "observed_at": str,
        "details": dict,
    }
    if any(not isinstance(value.get(key), expected) for key, expected in required.items()):
        return unknown_job_explanation(record)
    if any(not str(value[key]).strip() for key in ("code", "message", "observed_at")):
        return unknown_job_explanation(record)
    next_action = value.get("next_action")
    if next_action is not None and not isinstance(next_action, str):
        return unknown_job_explanation(record)
    explanation = {key: value[key] for key in required}
    if next_action:
        explanation["next_action"] = next_action
    return explanation


def _single_json(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RemoteWorkerError("protocol_error", "expected one JSON response line")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RemoteWorkerError("protocol_error", "worker returned malformed JSON") from error
    if not isinstance(value, dict):
        raise RemoteWorkerError("protocol_error", "worker response must be a JSON object")
    return value


def _json_document(output: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise RemoteWorkerError("protocol_error", "worker returned malformed JSON") from error
    if not isinstance(value, dict):
        raise RemoteWorkerError("protocol_error", "worker response must be a JSON object")
    return value


def _resource_snapshot(server_id: str, payload: dict[str, Any]) -> ServerResourceSnapshot:
    sampled_at = datetime.fromisoformat(str(payload["collected_at"]))
    gpus: list[GpuResource] = []
    for value in payload.get("gpus", []):
        telemetry = telemetry_from_mapping(value)
        eligibility = DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(telemetry)
        index = int(value["index"])
        gpus.append(
            GpuResource(
                index=index,
                uuid=str(value.get("uuid") or f"{server_id}-gpu-{index}"),
                memory_total_mib=normalize_nonnegative_number(
                    value.get("memory.total"),
                    integral=True,
                ),
                memory_used_mib=telemetry.memory_used_mib,
                utilization_percent=telemetry.utilization_percent,
                process_count=telemetry.process_count,
                idle=eligibility.value == "idle",
            )
        )
    return ServerResourceSnapshot(
        server_id=server_id,
        sampled_at=sampled_at,
        healthy=bool(gpus),
        error=None if gpus else "collector reported no GPUs",
        gpus=gpus,
    )
