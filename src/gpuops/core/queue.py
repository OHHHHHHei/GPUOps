from __future__ import annotations

import builtins
import fcntl
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gpuops.contracts import is_valid_run_id
from gpuops.core.queue_index import QueueIndex
from gpuops.core.state_machine import ensure_transition
from gpuops.core.storage import StorageBudget
from gpuops.file_io import atomic_write_bytes
from gpuops.models import (
    PRIORITY_ORDER,
    Allocation,
    JobRecord,
    JobSpec,
    JobStatus,
    JobSummary,
    LaunchInfo,
    utc_now,
)


class QueueError(RuntimeError):
    pass


class JobNotFound(QueueError):
    pass


class ConcurrentUpdate(QueueError):
    pass


class ResourceConflict(QueueError):
    pass


class SubmissionConflict(QueueError):
    pass


class InvalidRunId(QueueError):
    pass


_UNSET = object()


class QueueStore:
    """Filesystem queue with one atomic JSON file per job and a derived SQLite index."""

    def __init__(self, root: str | Path, *, storage: StorageBudget | None = None):
        self.root = Path(root)
        self.storage = storage or StorageBudget(self.root)
        resolved_root = self.root.resolve()
        resolved_budget_root = self.storage.root.resolve()
        if (
            resolved_root != resolved_budget_root
            and resolved_budget_root not in resolved_root.parents
        ):
            raise ValueError("queue root must be inside the storage budget root")
        self.jobs_directory = self.root / "jobs"
        self.lock_path = self.root / "queue.lock"
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.jobs_directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(self.jobs_directory, 0o700)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.lock_path, 0o600)
        self.index = QueueIndex(self.root, self.jobs_directory, self.storage)
        # Earlier releases kept an event log that nothing read; reclaim its space.
        (self.root / "events.jsonl").unlink(missing_ok=True)
        (self.root / "event-sequence").unlink(missing_ok=True)
        shutil.rmtree(self.storage.root / "event-archive", ignore_errors=True)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, run_id: str) -> Path:
        if not is_valid_run_id(run_id):
            raise InvalidRunId(f"invalid run_id: {run_id!r}")
        return self.jobs_directory / f"{run_id}.json"

    def _read_unlocked(self, run_id: str) -> JobRecord:
        try:
            return JobRecord.model_validate_json(self._path(run_id).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise JobNotFound(run_id) from error

    def _all_unlocked(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in sorted(self.jobs_directory.glob("*.json")):
            records.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
        return records

    def _records_for_statuses_unlocked(self, statuses: set[JobStatus]) -> list[JobRecord]:
        summaries = self.index.summaries(self._all_unlocked, statuses)
        return [self._read_unlocked(summary.run_id) for summary in summaries]

    def _write_unlocked(self, record: JobRecord) -> None:
        target = self._path(record.run_id)
        payload = (record.model_dump_json(indent=2) + "\n").encode("utf-8")
        self.storage.ensure_capacity(additional_bytes=len(payload))
        previous_signature = self.index.signature()
        atomic_write_bytes(target, payload, prefix=f".{record.run_id}.")
        self.index.record_written(
            record,
            previous_signature=previous_signature,
            current_signature=self.index.signature(),
        )

    def enqueue(
        self,
        spec: JobSpec,
        now: datetime | None = None,
        *,
        run_id: str | None = None,
    ) -> JobRecord:
        timestamp = now or utc_now()
        stable_run_id = run_id or (
            f"{spec.id}-{timestamp.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        path = self._path(stable_run_id)
        with self._lock():
            if path.exists():
                existing = self._read_unlocked(stable_run_id)
                if existing.spec == spec:
                    return existing
                raise SubmissionConflict(
                    f"run_id {stable_run_id!r} already exists with a different job spec"
                )
            record = JobRecord(
                run_id=stable_run_id,
                spec=spec,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._write_unlocked(record)
        return record

    def get(self, run_id: str) -> JobRecord:
        with self._lock():
            return self._read_unlocked(run_id)

    def list(self, statuses: set[JobStatus] | None = None) -> list[JobRecord]:
        with self._lock():
            records = (
                self._all_unlocked()
                if statuses is None
                else self._records_for_statuses_unlocked(statuses)
            )
        return sorted(
            records,
            key=lambda record: (
                PRIORITY_ORDER[record.spec.schedule.priority],
                record.created_at,
                record.run_id,
            ),
        )

    def count(self) -> int:
        with self._lock():
            return self.index.count(self._all_unlocked)

    def summaries(self, statuses: set[JobStatus] | None = None) -> builtins.list[JobSummary]:
        with self._lock():
            return self.index.summaries(self._all_unlocked, statuses)

    def history(
        self,
        *,
        server_id: str,
        limit: int = 50,
        project: str | None = None,
        before: tuple[datetime, str, str] | None = None,
    ) -> builtins.list[JobSummary]:
        with self._lock():
            return self.index.history(
                self._all_unlocked,
                server_id=server_id,
                limit=limit,
                project=project,
                before=before,
            )

    def history_projects(self) -> builtins.list[str]:
        with self._lock():
            return self.index.history_projects(self._all_unlocked)

    def terminal_after(
        self,
        after: tuple[datetime, str],
        *,
        limit: int = 50,
    ) -> builtins.list[JobSummary]:
        with self._lock():
            return self.index.terminal_after(self._all_unlocked, after, limit=limit)

    def lease(
        self,
        run_id: str,
        *,
        server_id: str,
        gpu_indices: builtins.list[int],
        worker_id: str,
        ttl: timedelta,
        expected_revision: int,
        now: datetime | None = None,
        gpu_uuids: dict[int, str] | None = None,
    ) -> JobRecord:
        timestamp = now or utc_now()
        with self._lock():
            record = self._read_unlocked(run_id)
            if record.revision != expected_revision or record.status != JobStatus.QUEUED:
                raise ConcurrentUpdate(f"job {run_id} changed while it was being scheduled")
            request = record.spec.resources
            if request.host not in {"auto", server_id}:
                raise ResourceConflict(f"job {run_id} is pinned to {request.host}")
            if len(gpu_indices) != request.gpu_count:
                raise ResourceConflict("leased GPU count does not match the job request")
            if request.gpus is not None and sorted(gpu_indices) != request.gpus:
                raise ResourceConflict("leased GPUs do not match the job's exact request")
            requested = set(gpu_indices)
            if len(requested) != len(gpu_indices):
                raise ResourceConflict("leased GPU indices must be unique")
            for other in self._records_for_statuses_unlocked(
                {
                    JobStatus.LEASED,
                    JobStatus.STARTING,
                    JobStatus.RUNNING,
                    JobStatus.CANCEL_REQUESTED,
                }
            ):
                if other.run_id == run_id or other.allocation is None:
                    continue
                if other.allocation.server_id == server_id and (
                    requested.intersection(other.allocation.gpu_indices)
                    or set((gpu_uuids or {}).values()).intersection(
                        other.allocation.gpu_uuids.values()
                    )
                ):
                    raise ResourceConflict(f"GPU allocation conflicts with {other.run_id}")
            allocation = Allocation(
                server_id=server_id,
                gpu_indices=gpu_indices,
                gpu_uuids=gpu_uuids or {},
                worker_id=worker_id,
                lease_token=uuid.uuid4().hex,
                leased_at=timestamp,
                lease_expires_at=timestamp + ttl,
            )
            return self._transition_unlocked(
                record,
                JobStatus.LEASED,
                timestamp,
                allocation=allocation,
            )

    def transition(
        self,
        run_id: str,
        target: JobStatus,
        *,
        expected_status: JobStatus | None = None,
        expected_revision: int | None = None,
        allocation: Allocation | None | object = _UNSET,
        launch: LaunchInfo | None | object = _UNSET,
        failure_reason: str | None | object = _UNSET,
        exit_code: int | None | object = _UNSET,
        launch_error: str | None | object = _UNSET,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = now or utc_now()
        with self._lock():
            record = self._read_unlocked(run_id)
            if expected_status is not None and record.status != expected_status:
                raise ConcurrentUpdate(f"expected {expected_status}, found {record.status}")
            if expected_revision is not None and record.revision != expected_revision:
                raise ConcurrentUpdate(
                    f"expected revision {expected_revision}, found {record.revision}"
                )
            return self._transition_unlocked(
                record,
                target,
                timestamp,
                allocation=allocation,
                launch=launch,
                failure_reason=failure_reason,
                exit_code=exit_code,
                launch_error=launch_error,
            )

    def request_cancel(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        """Persist an idempotent cancellation request under the queue lock."""

        if reason is not None and (not reason.strip() or len(reason) > 500):
            raise ValueError("cancel reason must contain 1 to 500 characters")
        timestamp = now or utc_now()
        with self._lock():
            record = self._read_unlocked(run_id)
            if record.status.terminal or record.status == JobStatus.CANCEL_REQUESTED:
                return record
            target = (
                JobStatus.CANCELLED
                if record.status in {JobStatus.QUEUED, JobStatus.LEASED}
                else JobStatus.CANCEL_REQUESTED
            )
            return self._transition_unlocked(record, target, timestamp)

    def _transition_unlocked(
        self,
        record: JobRecord,
        target: JobStatus,
        timestamp: datetime,
        *,
        allocation: Allocation | None | object = _UNSET,
        launch: LaunchInfo | None | object = _UNSET,
        failure_reason: str | None | object = _UNSET,
        exit_code: int | None | object = _UNSET,
        launch_error: str | None | object = _UNSET,
    ) -> JobRecord:
        ensure_transition(record.status, target)
        updates: dict[str, Any] = {
            "status": target,
            "revision": record.revision + 1,
            "updated_at": timestamp,
        }
        if allocation is not _UNSET:
            updates["allocation"] = allocation
        if launch is not _UNSET:
            updates["launch"] = launch
        if failure_reason is not _UNSET:
            updates["failure_reason"] = failure_reason
        if exit_code is not _UNSET:
            updates["exit_code"] = exit_code
        if launch_error is not _UNSET:
            updates["launch_error"] = launch_error
        if target.terminal:
            updates["finished_at"] = timestamp
        updated = JobRecord.model_validate(record.model_copy(update=updates).model_dump())
        self._write_unlocked(updated)
        return updated

    def release_expired_leases(self, now: datetime | None = None) -> builtins.list[JobRecord]:
        timestamp = now or utc_now()
        released: list[JobRecord] = []
        with self._lock():
            for record in self._records_for_statuses_unlocked({JobStatus.LEASED}):
                if (
                    record.status == JobStatus.LEASED
                    and record.allocation is not None
                    and record.allocation.lease_expires_at <= timestamp
                ):
                    released.append(
                        self._transition_unlocked(
                            record,
                            JobStatus.QUEUED,
                            timestamp,
                            allocation=None,
                        )
                    )
        return released
