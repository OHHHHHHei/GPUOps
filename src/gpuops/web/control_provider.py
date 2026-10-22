"""Live remote queue and managed-job adapter for the local dashboard."""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import gpuops.config as config
from gpuops.control import RemoteTarget
from gpuops.models import (
    ACTIVE_JOB_STATUSES,
    JobRecord,
    JobStatus,
    JobSummary,
    utc_now,
)
from gpuops.queries import collect_history
from gpuops.transport import JobDetail, JobHistory, unknown_job_explanation
from gpuops.web.job_eta import job_estimate, managed_log_estimate
from gpuops.web.provider import (
    DashboardProvider,
    JsonObject,
    ProviderConflict,
    ProviderUnavailable,
)


class QueueTransport(Protocol):
    async def list_summaries(
        self, target: RemoteTarget, *, statuses: set[JobStatus]
    ) -> list[JobSummary]: ...

    async def history(
        self, target: RemoteTarget, *, limit: int, project: str | None, cursor: str | None
    ) -> JobHistory: ...

    async def get_job(self, target: RemoteTarget, run_id: str) -> JobRecord | None: ...

    async def get_job_detail(self, target: RemoteTarget, run_id: str) -> JobDetail | None: ...

    async def get_logs(
        self,
        target: RemoteTarget,
        run_id: str,
        *,
        tail_bytes: int,
    ) -> JsonObject: ...

    async def cancel(self, target: RemoteTarget, run_id: str) -> JobRecord: ...


@dataclass(frozen=True)
class _HostJobs:
    target: RemoteTarget
    jobs: tuple[JobSummary, ...]
    error: str | None = None

    @property
    def online(self) -> bool:
        return self.error is None


class ControlDashboardProvider:
    """Combine live monitoring with fault-isolated remote worker state."""

    def __init__(
        self,
        snapshot_provider: DashboardProvider,
        transport: QueueTransport,
        targets: Sequence[RemoteTarget] | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        log_tail_bytes: int = 64 * 1024,
        max_log_tail_bytes: int = 1024 * 1024,
    ) -> None:
        if targets is not None and not targets:
            raise ValueError("at least one remote target is required")
        if targets is not None and len({target.server_id for target in targets}) != len(targets):
            raise ValueError("remote target server ids must be unique")
        if not 1 <= log_tail_bytes <= max_log_tail_bytes:
            raise ValueError("log_tail_bytes must be within the configured bound")
        self.snapshot_provider = snapshot_provider
        self.transport = transport
        self._targets = tuple(sorted(targets)) if targets is not None else None
        self.clock = clock
        self.log_tail_bytes = log_tail_bytes
        self._log_samples: dict[tuple[str, str], tuple[JobRecord, dict]] = {}

    @property
    def targets(self) -> tuple[RemoteTarget, ...]:
        """Resolve the same fleet as monitoring for each remote operation."""
        if self._targets is not None:
            return self._targets
        return tuple(
            RemoteTarget(target.id, target.ssh_host)
            for target in config.select_targets()
        )

    async def get_snapshot(self) -> JsonObject:
        return await self.snapshot_provider.get_snapshot()

    async def get_process_etas(self) -> JsonObject:
        return await self.snapshot_provider.get_process_etas()

    async def health(self) -> JsonObject:
        return await self.snapshot_provider.health()

    async def get_queue(self) -> JsonObject:
        hosts = await self._collect_hosts()
        now = self.clock()
        items: list[JsonObject] = []
        for host in hosts:
            if not host.online:
                continue
            active_ids = {record.run_id for record in host.jobs}
            for key in list(self._log_samples):
                if key[0] == host.target.server_id and key[1] not in active_ids:
                    self._log_samples.pop(key)
            items.extend(
                _queue_item(record, host.target.server_id, self._estimate(record, host.target, now))
                for record in host.jobs
            )
        items.sort(key=_queue_item_sort_key)
        return {
            "updated_at": _iso(now),
            "mode": "live-control",
            "items": items,
            "servers": [
                {
                    "id": host.target.server_id,
                    "online": host.online,
                    "job_count": len(host.jobs),
                    "error": host.error,
                }
                for host in hosts
            ],
        }

    async def get_history(
        self, *, limit: int = 50, project: str | None = None, cursor: str | None = None
    ) -> JsonObject:
        page = await collect_history(
            self.transport, self.targets, limit=limit, project=project, cursor=cursor
        )
        return {
            "updated_at": _iso(self.clock()),
            "mode": "live-control",
            "items": [
                _queue_item(record, target.server_id, job_estimate(record, self.clock()))
                for target, record in page.items
            ],
            "projects": page.projects,
            "next_cursor": page.next_cursor,
            "servers": [
                {
                    "id": host.target.server_id,
                    "online": host.error is None,
                    "error": _error_text(host.error) if host.error is not None else None,
                }
                for host in page.hosts
            ],
        }

    async def get_job(self, job_id: str) -> JsonObject | None:
        located = await self._locate(job_id)
        if located is None:
            return None
        target, record = located

        async def load_detail() -> tuple[JobDetail | None, str | None]:
            try:
                return await self.transport.get_job_detail(target, job_id), None
            except Exception as error:
                return None, _error_text(error)

        async def load_logs() -> tuple[dict, str | None]:
            try:
                response = await self.transport.get_logs(
                    target,
                    job_id,
                    tail_bytes=self.log_tail_bytes,
                )
                response["content"] = (
                    str(response.get("content") or "")
                    .encode("utf-8")[-self.log_tail_bytes :]
                    .decode("utf-8", errors="replace")
                )
                return response, None
            except Exception as error:
                return {}, _error_text(error)

        (detail_result, detail_error), (log_response, log_error) = await asyncio.gather(
            load_detail(), load_logs()
        )
        now = self.clock()
        explanation = unknown_job_explanation(record, observed_at=now)
        if detail_result is not None:
            record = detail_result.record
            explanation = detail_result.explanation

        summary = JobSummary.from_record(record)
        key = (target.server_id, record.run_id)
        if record.status == JobStatus.RUNNING and not log_error:
            log_response.setdefault("sampled_at", now.timestamp())
            self._log_samples[key] = (record, log_response)
            # Only visited running jobs are retained; queue reads prune completed jobs.
            if len(self._log_samples) > 32:
                self._log_samples.pop(next(iter(self._log_samples)))
        else:
            self._log_samples.pop(key, None)
        return _job_detail(
            record,
            target.server_id,
            self._estimate(summary, target, now),
            logs=_bounded_log_lines(str(log_response.get("content") or ""), self.log_tail_bytes),
            log_error=log_error,
            explanation=explanation,
            detail_error=detail_error,
        )

    def _estimate(self, record: JobSummary, target: RemoteTarget, now: datetime) -> JsonObject:
        cached = self._log_samples.get((target.server_id, record.run_id))
        log_eta = None
        if (
            cached
            and cached[0].launch
            and cached[0].launch.started_at == record.started_at
            and now.timestamp() - cached[1]["sampled_at"] <= 60
        ):
            log_eta = managed_log_estimate(*cached, now)
        return job_estimate(record, now, log_eta)

    async def cancel_job(self, job_id: str) -> JsonObject | None:
        located = await self._locate(job_id)
        if located is None:
            return None
        target, _record = located
        cancel = getattr(self.transport, "cancel", None)
        if cancel is None:
            raise ProviderUnavailable("remote transport does not support cancellation")
        try:
            cancelled = await cancel(target, job_id)
        except Exception as error:
            raise ProviderUnavailable(
                f"cannot cancel {job_id!r} on {target.server_id}: {_error_text(error)}"
            ) from error
        if cancelled.run_id != job_id:
            raise ProviderConflict("remote cancellation returned a different run id")
        return {
            "id": cancelled.run_id,
            "server": target.server_id,
            "state": _ui_state(cancelled.status),
            "raw_state": cancelled.status.value,
            "accepted": cancelled.status in {JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED},
        }

    async def _collect_hosts(self) -> tuple[_HostJobs, ...]:
        async def collect(target: RemoteTarget) -> _HostJobs:
            try:
                jobs = await self.transport.list_summaries(target, statuses=ACTIVE_JOB_STATUSES)
                return _HostJobs(target, tuple(jobs))
            except Exception as error:
                return _HostJobs(target, (), _error_text(error))

        return tuple(await asyncio.gather(*(collect(target) for target in self.targets)))

    async def _locate(self, job_id: str) -> tuple[RemoteTarget, JobRecord] | None:
        async def locate(target: RemoteTarget) -> tuple[RemoteTarget, JobRecord | None, str | None]:
            try:
                return target, await self.transport.get_job(target, job_id), None
            except Exception as error:
                return target, None, _error_text(error)

        hosts = await asyncio.gather(*(locate(target) for target in self.targets))
        matches = [(target, record) for target, record, _error in hosts if record is not None]
        if len(matches) > 1:
            servers = ", ".join(match[0].server_id for match in matches)
            raise ProviderConflict(f"job {job_id!r} exists on multiple servers: {servers}")
        if matches:
            return matches[0]
        failed = [target.server_id for target, _record, error in hosts if error is not None]
        if failed:
            raise ProviderUnavailable(
                f"cannot determine whether job {job_id!r} exists; unavailable: {', '.join(failed)}"
            )
        return None


def _queue_item(record: JobSummary, server_id: str, eta: JsonObject) -> JsonObject:
    return {
        "id": record.run_id,
        "name": record.name,
        "project": record.project,
        "state": _ui_state(record.status),
        "raw_state": record.status.value,
        "server": server_id,
        "gpus": record.gpus,
        "priority": record.priority.value,
        "submitted_at": _iso(record.created_at),
        "estimated_start": _iso(record.started_at),
        "estimated_end": eta.get("end_at"),
        "eta_confidence": eta["confidence"],
        "eta": eta,
        "finished_at": _iso(record.finished_at),
    }


def _job_detail(
    record: JobRecord,
    server_id: str,
    eta: JsonObject,
    *,
    logs: list[str],
    log_error: str | None,
    explanation: JsonObject,
    detail_error: str | None,
) -> JsonObject:
    detail = _queue_item(JobSummary.from_record(record), server_id, eta)
    detail.update(
        {
            "command": shlex.join(record.spec.command),
            "working_directory": record.spec.working_directory,
            "created_by": record.spec.metadata.get("created_by") or "unknown",
            "progress": None,
            "logs": logs,
            "log_error": log_error,
            "detail_error": detail_error,
            "explanation": explanation,
            "failure_reason": record.failure_reason,
            "events": _record_events(record),
        }
    )
    return detail


def _record_events(record: JobRecord) -> list[JsonObject]:
    events: list[JsonObject] = [
        {
            "at": _iso(record.created_at),
            "type": "submitted",
            "message": "作业已进入调度队列",
        }
    ]
    if record.launch is not None:
        events.append(
            {
                "at": _iso(record.launch.started_at),
                "type": "started",
                "message": "作业已启动",
            }
        )
    if record.finished_at is not None:
        events.append(
            {
                "at": _iso(record.finished_at),
                "type": record.status.value,
                "message": "作业已结束",
            }
        )
    return events


def _ui_state(status: JobStatus) -> str:
    if status in {JobStatus.QUEUED, JobStatus.LEASED}:
        return "queued"
    if status in {JobStatus.STARTING, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
        return "running"
    if status == JobStatus.SUCCEEDED:
        return "finished"
    return status.value


def _queue_item_sort_key(item: JsonObject) -> tuple[int, int, str, str]:
    state_order = {
        "running": 0,
        "queued": 1,
        "failed": 2,
        "cancelled": 3,
        "finished": 4,
    }
    return (
        state_order.get(str(item["state"]), 5),
        {"urgent": 0, "high": 1, "normal": 2, "low": 3}.get(str(item["priority"]), 4),
        str(item["submitted_at"]),
        str(item["id"]),
    )


def _bounded_log_lines(content: str, limit: int) -> list[str]:
    encoded = content.encode("utf-8", errors="replace")
    bounded = encoded[-limit:]
    return bounded.decode("utf-8", errors="ignore").splitlines()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _error_text(error: BaseException) -> str:
    message = str(error).strip() or error.__class__.__name__
    return f"{error.__class__.__name__}: {message}".replace("\n", " ")[:400]
