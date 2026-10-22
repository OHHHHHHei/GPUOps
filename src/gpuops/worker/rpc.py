"""Fixed JSON RPC boundary for host-local GPUOps worker operations."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gpuops.config import SchedulerSettings
from gpuops.core.queue import (
    InvalidRunId,
    JobNotFound,
    QueueStore,
    SubmissionConflict,
    re_safe_run_id,
)
from gpuops.core.resources import ResourceProbe
from gpuops.core.storage import StorageBudget, StorageQuotaExceeded
from gpuops.history import decode_cursor
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now, validate_new_submission
from gpuops.worker.app import LOG_BUDGET_BYTES, STATE_BUDGET_BYTES, worker_readiness
from gpuops.worker.explanations import explain_job
from gpuops.worker.runtime import RuntimePaths, WorkerProcessStatus, inspect_worker
from gpuops.worker.sampling import CachedGpuProbe

RPC_API_VERSION = "gpuops.worker.rpc/v1"
DEFAULT_TOTAL_BUDGET_BYTES = STATE_BUDGET_BYTES
DEFAULT_LOG_BUDGET_BYTES = LOG_BUDGET_BYTES
DEFAULT_LOG_TAIL_BYTES = 64 * 1024
DEFAULT_MAX_LOG_TAIL_BYTES = 1024 * 1024


class WorkerRpcError(RuntimeError):
    code = "rpc_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


class InvalidEnvelope(WorkerRpcError):
    code = "invalid_envelope"


class InvalidParams(WorkerRpcError):
    code = "invalid_params"


class UnsupportedMethod(WorkerRpcError):
    code = "method_not_found"


class HostMismatch(WorkerRpcError):
    code = "host_mismatch"


class RunIdConflict(WorkerRpcError):
    code = "run_id_conflict"


class RpcJobNotFound(WorkerRpcError):
    code = "job_not_found"


class LogNotFound(WorkerRpcError):
    code = "log_not_found"


class InternalRpcError(WorkerRpcError):
    code = "internal_error"


class WorkerUnavailable(WorkerRpcError):
    code = "worker_unavailable"


class WorkerRpc:
    """Parse and dispatch a deliberately small, versioned JSON request surface."""

    _METHODS = frozenset({"submit", "list", "show", "logs", "health", "cancel", "history"})

    def __init__(
        self,
        state_root: str | Path,
        *,
        server_id: str,
        queue: QueueStore | None = None,
        storage: StorageBudget | None = None,
        total_budget_bytes: int = DEFAULT_TOTAL_BUDGET_BYTES,
        log_budget_bytes: int = DEFAULT_LOG_BUDGET_BYTES,
        max_log_tail_bytes: int = DEFAULT_MAX_LOG_TAIL_BYTES,
        worker_inspector: Callable[[RuntimePaths], WorkerProcessStatus] = inspect_worker,
        clock: Callable[[], datetime] = utc_now,
        readiness_probe: Callable[[], tuple[bool, str]] | None = None,
        resource_probe: ResourceProbe | None = None,
        max_snapshot_age_seconds: float | None = None,
    ) -> None:
        if not server_id.strip():
            raise ValueError("server_id cannot be empty")
        if max_log_tail_bytes <= 0:
            raise ValueError("max_log_tail_bytes must be positive")
        snapshot_age = (
            SchedulerSettings().max_snapshot_age_seconds
            if max_snapshot_age_seconds is None
            else max_snapshot_age_seconds
        )
        if snapshot_age <= 0:
            raise ValueError("max_snapshot_age_seconds must be positive")

        self.state_root = Path(state_root).expanduser()
        self.state_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.state_root, 0o700)
        self.server_id = server_id
        self.worker_inspector = worker_inspector
        self.clock = clock
        self.readiness_probe = readiness_probe
        self.resource_probe = (
            resource_probe
            if resource_probe is not None
            else CachedGpuProbe(self.state_root, server_id, clock=clock)
        )
        self.max_snapshot_age_seconds = snapshot_age
        self.storage = storage or StorageBudget(
            self.state_root,
            total_budget_bytes=total_budget_bytes,
            log_budget_bytes=log_budget_bytes,
        )
        if self.storage.root.resolve() != self.state_root.resolve():
            raise ValueError("storage budget must use the worker state root")

        self.queue_directory = self.state_root / "queue"
        self.runs_directory = self.state_root / "runs"
        self.logs_directory = self.storage.logs_directory
        self.runs_directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.runs_directory, 0o700)
        self.queue = queue or QueueStore(self.queue_directory, storage=self.storage)
        if self.queue.root.resolve() != self.queue_directory.resolve():
            raise ValueError("queue must use state_root/queue")
        self.max_log_tail_bytes = max_log_tail_bytes

    def handle_json(self, payload: str | bytes) -> dict[str, Any]:
        try:
            request = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            message = getattr(error, "msg", str(error))
            return self._error_response(InvalidEnvelope(f"invalid JSON: {message}"), None)
        return self.handle(request)

    def handle(self, request: Any) -> dict[str, Any]:
        request_id: str | None = None
        try:
            envelope = self._parse_envelope(request)
            request_id = envelope.get("request_id")
            method = envelope["method"]
            params = envelope["params"]
            result = getattr(self, method)(params)
            return self._success_response(result, request_id)
        except WorkerRpcError as error:
            return self._error_response(error, request_id)
        except ValidationError as error:
            details = {
                "errors": [
                    {
                        "location": [str(part) for part in item["loc"]],
                        "message": item["msg"],
                        "type": item["type"],
                    }
                    for item in error.errors()
                ]
            }
            return self._error_response(
                InvalidParams("job_spec validation failed", details=details),
                request_id,
            )
        except InvalidRunId as error:
            return self._error_response(InvalidParams(str(error)), request_id)
        except StorageQuotaExceeded as error:
            quota_error = WorkerRpcError(str(error))
            quota_error.code = "storage_quota_exceeded"
            return self._error_response(quota_error, request_id)
        except Exception:
            return self._error_response(
                InternalRpcError("worker RPC failed unexpectedly"),
                request_id,
            )

    def submit(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required={"run_id", "job_spec"})
        run_id = self._require_run_id(params["run_id"])
        job_spec = JobSpec.model_validate(params["job_spec"])
        if job_spec.resources.host != self.server_id:
            raise HostMismatch(
                f"job host {job_spec.resources.host!r} does not match {self.server_id!r}"
            )
        try:
            existing = self.queue.get(run_id)
        except JobNotFound:
            pass
        else:
            if existing.spec != job_spec:
                raise RunIdConflict(f"run_id {run_id!r} already exists with a different job spec")
            return {"durable": True, **self._record_result(existing)}
        try:
            validate_new_submission(job_spec)
        except ValueError as error:
            raise InvalidParams(str(error)) from error
        ready, reason = self._worker_readiness()
        if not ready:
            raise WorkerUnavailable(f"background worker is unavailable: {reason}")
        try:
            record = self.queue.enqueue(job_spec, run_id=run_id)
        except SubmissionConflict as error:
            raise RunIdConflict(str(error)) from error
        return {"durable": True, **self._record_result(record)}

    def list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required=set(), optional={"statuses", "summary"})
        statuses_value = params.get("statuses")
        statuses: set[JobStatus] | None = None
        if statuses_value is not None:
            if not isinstance(statuses_value, list) or not statuses_value:
                raise InvalidParams("statuses must be a non-empty JSON array")
            try:
                statuses = {JobStatus(value) for value in statuses_value}
            except (TypeError, ValueError) as error:
                raise InvalidParams("statuses contains an unknown job status") from error
        summary = params.get("summary", False)
        if not isinstance(summary, bool):
            raise InvalidParams("summary must be a boolean")
        if summary:
            jobs = self.queue.summaries(statuses)
            return {"jobs": [job.model_dump(mode="json") for job in jobs], "count": len(jobs)}
        records = self.queue.list(statuses)
        return {"jobs": [self._record_json(record) for record in records], "count": len(records)}

    def history(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required=set(), optional={"limit", "project", "cursor"})
        limit = params.get("limit", 50)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 101:
            raise InvalidParams("limit must be an integer between 1 and 101")
        project = params.get("project")
        if project is not None and (not isinstance(project, str) or not project):
            raise InvalidParams("project must be a non-empty string")
        try:
            before = decode_cursor(params.get("cursor"))
        except ValueError as error:
            raise InvalidParams(str(error)) from error
        jobs = self.queue.history(
            server_id=self.server_id, limit=limit, project=project, before=before
        )
        return {
            "jobs": [job.model_dump(mode="json") for job in jobs],
            "projects": self.queue.history_projects(),
        }

    def show(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required={"run_id"}, optional={"explain"})
        explain = params.get("explain", False)
        if not isinstance(explain, bool):
            raise InvalidParams("explain must be a boolean")
        record = self._get_job(params["run_id"])
        result = self._record_result(record)
        if explain:
            result["explanation"] = explain_job(
                record,
                server_id=self.server_id,
                queue=self.queue,
                resource_probe=self.resource_probe,
                readiness_probe=self._worker_readiness,
                clock=self.clock,
                max_snapshot_age_seconds=self.max_snapshot_age_seconds,
            )
        return result

    def cancel(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required={"run_id"}, optional={"reason"})
        run_id = self._require_run_id(params["run_id"])
        reason = params.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidParams("reason must be a non-empty string")
        if isinstance(reason, str) and len(reason) > 500:
            raise InvalidParams("reason cannot exceed 500 characters")
        try:
            record = self.queue.request_cancel(run_id, reason=reason)
        except JobNotFound as error:
            raise RpcJobNotFound(f"job {run_id!r} does not exist") from error
        return {"durable": True, **self._record_result(record)}

    def logs(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required={"run_id"}, optional={"tail_bytes"})
        run_id = self._require_run_id(params["run_id"])
        self._get_job(run_id)
        tail_bytes = params.get(
            "tail_bytes",
            min(DEFAULT_LOG_TAIL_BYTES, self.max_log_tail_bytes),
        )
        if isinstance(tail_bytes, bool) or not isinstance(tail_bytes, int):
            raise InvalidParams("tail_bytes must be an integer")
        if not 1 <= tail_bytes <= self.max_log_tail_bytes:
            raise InvalidParams(f"tail_bytes must be between 1 and {self.max_log_tail_bytes}")

        run_directory = self.logs_directory / run_id
        expected_directory = run_directory.resolve(strict=False)
        if self.logs_directory.resolve() not in expected_directory.parents:
            raise InvalidParams("run_id escapes the log directory")
        if run_directory.is_symlink() or not run_directory.is_dir():
            raise LogNotFound(f"no console log exists for {run_id!r}")

        candidates = []
        for path in run_directory.glob("console.log*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.resolve().parent != expected_directory:
                    continue
                metadata = path.stat()
            except FileNotFoundError:
                continue
            candidates.append((metadata.st_mtime_ns, path.name, path, metadata.st_size))
        candidates.sort()
        if not candidates:
            raise LogNotFound(f"no console log exists for {run_id!r}")

        remaining = tail_bytes
        chunks: list[bytes] = []
        selected_names: list[str] = []
        modified_at: float | None = None
        total_bytes = sum(item[3] for item in candidates)
        for _, name, path, size in reversed(candidates):
            if remaining == 0:
                break
            try:
                with path.open("rb") as handle:
                    modified = os.fstat(handle.fileno()).st_mtime
                    read_bytes = min(size, remaining)
                    handle.seek(-read_bytes, os.SEEK_END)
                    chunk = handle.read(read_bytes)
                    if chunk:
                        modified_at = (
                            max(modified_at, modified) if modified_at is not None else modified
                        )
                        chunks.append(chunk)
                        selected_names.append(name)
                        remaining -= len(chunk)
            except FileNotFoundError:
                continue
        content = b"".join(reversed(chunks))
        if not content and total_bytes:
            raise LogNotFound(f"console logs for {run_id!r} changed while being read")
        return {
            "run_id": run_id,
            "content": content.decode("utf-8", errors="replace"),
            "returned_bytes": len(content),
            "truncated": total_bytes > len(content),
            "files": list(reversed(selected_names)),
            "source": str(run_directory / selected_names[0]) if selected_names else None,
            "modified_at": modified_at,
            "sampled_at": time.time(),
            "utc_offset_seconds": datetime.now().astimezone().utcoffset().total_seconds(),
        }

    def health(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_param_keys(params, required=set())
        job_count = self.queue.count()
        usage = self.storage.usage()
        ready, reason = self._worker_readiness()
        return {
            "status": "ok" if ready else "unavailable",
            "server_id": self.server_id,
            "durable": ready,
            "worker": {"ready": ready, "reason": reason},
            "queue": {"jobs": job_count},
            "storage": {
                "total_bytes": usage.total_bytes,
                "log_bytes": usage.log_bytes,
                "total_budget_bytes": self.storage.total_budget_bytes,
                "log_budget_bytes": self.storage.log_budget_bytes,
            },
            "paths": {"queue": "queue", "runs": "runs", "logs": "logs"},
        }

    def _worker_readiness(self) -> tuple[bool, str]:
        if self.readiness_probe is not None:
            return self.readiness_probe()
        paths = RuntimePaths.from_state_directory(self.state_root)
        return worker_readiness(
            self.worker_inspector(paths),
            server_id=self.server_id,
            now=self.clock(),
        )

    def _get_job(self, run_id_value: Any) -> JobRecord:
        run_id = self._require_run_id(run_id_value)
        try:
            return self.queue.get(run_id)
        except JobNotFound as error:
            raise RpcJobNotFound(f"job {run_id!r} does not exist") from error

    @staticmethod
    def _record_json(record: JobRecord) -> dict[str, Any]:
        # v1 clients use extra="forbid" for JobRecord. Extend the single-job response
        # beside the legacy record instead of changing its wire schema.
        return record.model_dump(mode="json", exclude={"exit_code", "launch_error"})

    @classmethod
    def _record_result(cls, record: JobRecord) -> dict[str, Any]:
        return {
            "job": cls._record_json(record),
            "execution_result": {
                "exit_code": record.exit_code,
                "launch_error": record.launch_error,
            },
        }

    def _parse_envelope(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise InvalidEnvelope("request must be a JSON object")
        allowed = {"api_version", "method", "params", "request_id"}
        unexpected = set(request).difference(allowed)
        if unexpected:
            raise InvalidEnvelope(f"unexpected envelope fields: {', '.join(sorted(unexpected))}")
        if request.get("api_version") != RPC_API_VERSION:
            raise InvalidEnvelope(f"api_version must be {RPC_API_VERSION!r}")
        method = request.get("method")
        if not isinstance(method, str) or method not in self._METHODS:
            raise UnsupportedMethod(f"unsupported worker RPC method: {method!r}")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise InvalidEnvelope("params must be a JSON object")
        request_id = request.get("request_id")
        if request_id is not None and (
            not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
        ):
            raise InvalidEnvelope("request_id must be a string of 1 to 128 characters")
        return {
            "api_version": RPC_API_VERSION,
            "method": method,
            "params": params,
            "request_id": request_id,
        }

    @staticmethod
    def _validate_param_keys(
        params: Mapping[str, Any],
        *,
        required: set[str],
        optional: set[str] | None = None,
    ) -> None:
        optional = optional or set()
        missing = required.difference(params)
        unexpected = set(params).difference(required | optional)
        if missing:
            raise InvalidParams(f"missing parameters: {', '.join(sorted(missing))}")
        if unexpected:
            raise InvalidParams(f"unexpected parameters: {', '.join(sorted(unexpected))}")

    @staticmethod
    def _require_run_id(value: Any) -> str:
        if not isinstance(value, str) or not re_safe_run_id(value):
            raise InvalidParams("run_id is invalid")
        return value

    @staticmethod
    def _success_response(result: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        response: dict[str, Any] = {
            "api_version": RPC_API_VERSION,
            "ok": True,
            "result": result,
        }
        if request_id is not None:
            response["request_id"] = request_id
        return response

    @staticmethod
    def _error_response(error: WorkerRpcError, request_id: str | None) -> dict[str, Any]:
        error_payload: dict[str, Any] = {"code": error.code, "message": str(error)}
        if error.details is not None:
            error_payload["details"] = error.details
        response: dict[str, Any] = {
            "api_version": RPC_API_VERSION,
            "ok": False,
            "error": error_payload,
        }
        if request_id is not None:
            response["request_id"] = request_id
        return response
