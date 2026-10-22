from __future__ import annotations

import fcntl
import gzip
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gpuops.contracts import is_valid_run_id
from gpuops.core.queue_index import QueueIndex
from gpuops.core.state_machine import ensure_transition
from gpuops.core.storage import StorageBudget
from gpuops.models import (
    PRIORITY_ORDER,
    Allocation,
    JobRecord,
    JobSpec,
    JobStatus,
    JobSummary,
    LaunchInfo,
    QueueEvent,
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
    """Filesystem queue with one atomic JSON file per job and an append-only event log."""

    def __init__(
        self,
        root: str | Path,
        *,
        storage: StorageBudget | None = None,
        event_log_max_bytes: int = 8 * 1024 * 1024,
        event_archive_count: int = 4,
    ):
        if event_log_max_bytes < 512:
            raise ValueError("event_log_max_bytes must be at least 512 bytes")
        if event_archive_count < 0:
            raise ValueError("event_archive_count cannot be negative")
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
        self.events_path = self.root / "events.jsonl"
        self.event_sequence_path = self.root / "event-sequence"
        self.event_archive_directory = self.storage.event_archive_directory
        self.event_log_max_bytes = event_log_max_bytes
        self.event_archive_count = event_archive_count
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.jobs_directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(self.jobs_directory, 0o700)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.lock_path, 0o600)
        self.index = QueueIndex(self.root, self.jobs_directory, self.storage)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, run_id: str) -> Path:
        if not re_safe_run_id(run_id):
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
        # Absent results keep the old storage schema. Observed terminal results
        # require a worker that understands them; deployment checks this before rollback.
        absent_results = {
            field for field in ("exit_code", "launch_error") if getattr(record, field) is None
        }
        payload = (record.model_dump_json(indent=2, exclude=absent_results) + "\n").encode("utf-8")
        self.storage.ensure_capacity(additional_bytes=len(payload))
        previous_signature = self.index.signature()
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.jobs_directory, prefix=f".{record.run_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, target)
            directory_fd = os.open(self.jobs_directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.index.record_written(
                record,
                previous_signature=previous_signature,
                current_signature=self.index.signature(),
            )
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _reserve_event_sequence_unlocked(self) -> int:
        try:
            current = int(self.event_sequence_path.read_text(encoding="ascii").strip())
        except FileNotFoundError:
            current = self._discover_last_event_sequence_unlocked()
        except ValueError as error:
            raise QueueError("event sequence file is corrupt") from error
        sequence = current + 1
        self._atomic_text_write_unlocked(self.event_sequence_path, f"{sequence}\n")
        return sequence

    def _atomic_text_write_unlocked(self, target: Path, contents: str) -> None:
        encoded = contents.encode("utf-8")
        self.storage.ensure_capacity(additional_bytes=len(encoded))
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, target)
            self._fsync_directory(target.parent)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _append_event_unlocked(
        self,
        record: JobRecord,
        event_type: str,
        previous_status: JobStatus | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = QueueEvent(
            sequence=self._reserve_event_sequence_unlocked(),
            run_id=record.run_id,
            event_type=event_type,
            previous_status=previous_status,
            status=record.status,
            revision=record.revision,
            details=details or {},
        )
        line = event.model_dump_json() + "\n"
        encoded = line.encode("utf-8")
        if len(encoded) > self.event_log_max_bytes:
            event = event.model_copy(update={"details": {"truncated": True}})
            encoded = (event.model_dump_json() + "\n").encode("utf-8")
        if len(encoded) > self.event_log_max_bytes:
            raise QueueError("event record exceeds the configured event log limit")
        self._rotate_event_log_unlocked(len(encoded))
        self.storage.ensure_capacity(additional_bytes=len(encoded))
        file_descriptor = os.open(
            self.events_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.fchmod(file_descriptor, 0o600)
            os.write(file_descriptor, encoded)
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

    def _rotate_event_log_unlocked(self, incoming_bytes: int) -> None:
        try:
            current_bytes = self.events_path.stat().st_size
        except FileNotFoundError:
            return
        if current_bytes == 0 or current_bytes + incoming_bytes <= self.event_log_max_bytes:
            return

        if self.event_archive_count == 0:
            self.events_path.unlink(missing_ok=True)
            self._fsync_directory(self.root)
            return

        sequence = self._current_event_sequence_unlocked()
        archive = self.event_archive_directory / f"events-{sequence:020d}.jsonl.gz"
        self.storage.ensure_capacity(additional_bytes=current_bytes)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.event_archive_directory,
            prefix=f".{archive.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "wb") as raw_handle:
                with gzip.GzipFile(  # noqa: SIM117
                    fileobj=raw_handle, mode="wb", mtime=0
                ) as compressed:
                    with self.events_path.open("rb") as source:
                        while chunk := source.read(64 * 1024):
                            compressed.write(chunk)
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, archive)
            self.events_path.unlink()
            self._prune_event_archives_unlocked()
            self._fsync_directory(self.event_archive_directory)
            self._fsync_directory(self.root)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _current_event_sequence_unlocked(self) -> int:
        try:
            return int(self.event_sequence_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            return self._discover_last_event_sequence_unlocked()

    def _discover_last_event_sequence_unlocked(self) -> int:
        last_sequence = 0
        for archive in sorted(self.event_archive_directory.glob("events-*.jsonl.gz")):
            try:
                with gzip.open(archive, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            last_sequence = max(
                                last_sequence,
                                QueueEvent.model_validate_json(line).sequence,
                            )
            except FileNotFoundError:
                continue
        with suppress(FileNotFoundError), self.events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_sequence = max(
                        last_sequence,
                        QueueEvent.model_validate_json(line).sequence,
                    )
        return last_sequence

    def _prune_event_archives_unlocked(self) -> None:
        archives = sorted(self.event_archive_directory.glob("events-*.jsonl.gz"))
        for archive in archives[: -self.event_archive_count]:
            archive.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

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
            self._append_event_unlocked(record, "enqueued", None)
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

    def summaries(self, statuses: set[JobStatus] | None = None) -> list[JobSummary]:
        with self._lock():
            return self.index.summaries(self._all_unlocked, statuses)

    def history(
        self,
        *,
        server_id: str,
        limit: int = 50,
        project: str | None = None,
        before: tuple[datetime, str, str] | None = None,
    ) -> list[JobSummary]:
        with self._lock():
            return self.index.history(
                self._all_unlocked,
                server_id=server_id,
                limit=limit,
                project=project,
                before=before,
            )

    def history_projects(self) -> list[str]:
        with self._lock():
            return self.index.history_projects(self._all_unlocked)

    def events(self) -> list[QueueEvent]:
        with self._lock():
            lines: list[str] = []
            for archive in sorted(self.event_archive_directory.glob("events-*.jsonl.gz")):
                try:
                    with gzip.open(archive, "rt", encoding="utf-8") as handle:
                        lines.extend(handle.read().splitlines())
                except FileNotFoundError:
                    continue
            with suppress(FileNotFoundError):
                lines.extend(self.events_path.read_text(encoding="utf-8").splitlines())
        events_by_sequence = {
            event.sequence: event
            for line in lines
            if line.strip()
            for event in [QueueEvent.model_validate_json(line)]
        }
        return [events_by_sequence[sequence] for sequence in sorted(events_by_sequence)]

    def lease(
        self,
        run_id: str,
        *,
        server_id: str,
        gpu_indices: list[int],
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
            updated = self._transition_unlocked(
                record,
                JobStatus.LEASED,
                timestamp,
                allocation=allocation,
            )
            self._append_event_unlocked(
                updated,
                "leased",
                JobStatus.QUEUED,
                {"server_id": server_id, "gpu_indices": gpu_indices, "worker_id": worker_id},
            )
            return updated

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
        details: dict[str, Any] | None = None,
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
            previous = record.status
            updated = self._transition_unlocked(
                record,
                target,
                timestamp,
                allocation=allocation,
                launch=launch,
                failure_reason=failure_reason,
                exit_code=exit_code,
                launch_error=launch_error,
            )
            self._append_event_unlocked(updated, "status_changed", previous, details)
            return updated

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
            previous = record.status
            updated = self._transition_unlocked(record, target, timestamp)
            details = {"reason": reason} if reason is not None else {}
            self._append_event_unlocked(updated, "cancel_requested", previous, details)
            return updated

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

    def release_expired_leases(self, now: datetime | None = None) -> list[JobRecord]:
        timestamp = now or utc_now()
        released: list[JobRecord] = []
        with self._lock():
            for record in self._records_for_statuses_unlocked({JobStatus.LEASED}):
                if (
                    record.status == JobStatus.LEASED
                    and record.allocation is not None
                    and record.allocation.lease_expires_at <= timestamp
                ):
                    updated = self._transition_unlocked(
                        record,
                        JobStatus.QUEUED,
                        timestamp,
                        allocation=None,
                    )
                    self._append_event_unlocked(
                        updated,
                        "lease_expired",
                        JobStatus.LEASED,
                    )
                    released.append(updated)
        return released


def re_safe_run_id(value: str) -> bool:
    """Compatibility alias for the shared run ID contract."""

    return is_valid_run_id(value)
