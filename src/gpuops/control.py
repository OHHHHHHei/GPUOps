"""Mac-side host selection and idempotent remote submission core."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from gpuops.contracts import RunId, is_valid_run_id
from gpuops.core.resources import ServerResourceSnapshot
from gpuops.models import (
    JobRecord,
    JobSpec,
    JobStatus,
    StrictModel,
    utc_now,
    validate_new_submission,
)


class ControlError(RuntimeError):
    """Base error for client-side control operations."""


class HostSelectionError(ControlError):
    """No live target can safely accept an automatically placed job."""


class SubmissionConflict(ControlError):
    """A run id is already associated with different submission data."""


class SubmissionNotFound(ControlError):
    """A targeted reconciliation has no durable local submission identity."""

    code = "submission_not_found"

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run_id {run_id!r} is not in the submission journal")
        self.run_id = run_id


class SubmissionState(str, Enum):
    ACCEPTED = "accepted"
    NOT_ACCEPTED = "not_accepted"
    UNKNOWN = "unknown"


class SubmissionAttemptError(ControlError):
    """A journal-backed submission attempt that did not complete successfully."""

    submission_state: SubmissionState

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        server_id: str,
        code: str,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.server_id = server_id
        self.code = code


class SubmissionUncertain(SubmissionAttemptError):
    """Submission state is unknown and remains pending in the local journal."""

    submission_state = SubmissionState.UNKNOWN


class SubmissionNotAccepted(SubmissionAttemptError):
    """The remote endpoint definitively rejected this submission attempt."""

    submission_state = SubmissionState.NOT_ACCEPTED


class DefinitiveRemoteSubmissionError(RuntimeError):
    """Transport contract for a response proving no submitted job was accepted."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RemoteProtocolError(ControlError):
    """A remote response does not match the requested run id or manifest."""


@dataclass(frozen=True, order=True)
class RemoteTarget:
    server_id: str
    ssh_host: str

    def __post_init__(self) -> None:
        if not self.server_id.strip() or not self.ssh_host.strip():
            raise ValueError("remote target fields cannot be empty")


@dataclass(frozen=True)
class RemoteHostState:
    snapshot: ServerResourceSnapshot
    queue: tuple[JobRecord, ...] = ()


class RemoteTransport(Protocol):
    """SSH-backed operations required by the control plane."""

    async def inspect(self, target: RemoteTarget) -> RemoteHostState: ...

    async def submit(self, target: RemoteTarget, run_id: str, spec: JobSpec) -> JobRecord: ...

    async def get_job(self, target: RemoteTarget, run_id: str) -> JobRecord | None: ...


class JournalState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"


class JournalEntry(StrictModel):
    schema_version: str = "gpuops-control/v1"
    run_id: RunId
    server_id: str
    ssh_host: str
    state: JournalState
    request_spec: JobSpec
    frozen_spec: JobSpec
    created_at: datetime
    updated_at: datetime
    remote_record: JobRecord | None = None

    @property
    def target(self) -> RemoteTarget:
        return RemoteTarget(server_id=self.server_id, ssh_host=self.ssh_host)


@dataclass(frozen=True)
class SubmissionResult:
    run_id: str
    target: RemoteTarget
    spec: JobSpec
    record: JobRecord
    reconciled: bool
    journal_warning: str | None = None


@dataclass(frozen=True)
class ReconcileReport:
    accepted: tuple[str, ...]
    pending: tuple[str, ...]
    errors: dict[str, str]
    outcomes: dict[str, SubmissionState] = field(default_factory=dict)
    error_codes: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HostEvaluation:
    target: RemoteTarget
    state: RemoteHostState
    available_gpu_count: int
    queued_gpu_demand: int
    active_job_count: int
    can_fit_now: bool

    @property
    def score(self) -> tuple[int, int, int, int, str]:
        return (
            0 if self.can_fit_now else 1,
            self.queued_gpu_demand,
            self.active_job_count,
            -self.available_gpu_count,
            self.target.server_id,
        )


class SubmissionJournal:
    """Atomic local record of host decisions and remote acknowledgements."""

    def __init__(self, directory: Path):
        self.directory = directory.expanduser()
        self.lock_path = self.directory / "journal.lock"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.lock_path, 0o600)

    def get(self, run_id: str) -> JournalEntry | None:
        with self._lock():
            return self._read_unlocked(run_id)

    def list_pending(self) -> list[JournalEntry]:
        with self._lock():
            entries = [self._read_path(path) for path in sorted(self.directory.glob("*.json"))]
        return [entry for entry in entries if entry.state == JournalState.PENDING]

    def create_pending(
        self,
        *,
        run_id: str,
        target: RemoteTarget,
        request_spec: JobSpec,
        frozen_spec: JobSpec,
        now: datetime,
    ) -> JournalEntry:
        candidate = JournalEntry(
            run_id=run_id,
            server_id=target.server_id,
            ssh_host=target.ssh_host,
            state=JournalState.PENDING,
            request_spec=request_spec,
            frozen_spec=frozen_spec,
            created_at=now,
            updated_at=now,
        )
        with self._lock():
            existing = self._read_unlocked(run_id)
            if existing is not None:
                self._ensure_same_submission(existing, target, request_spec)
                return existing
            self._write_unlocked(candidate)
        return candidate

    def mark_accepted(
        self,
        run_id: str,
        record: JobRecord,
        *,
        now: datetime,
    ) -> JournalEntry:
        with self._lock():
            existing = self._read_unlocked(run_id)
            if existing is None:
                raise SubmissionConflict(f"run_id {run_id!r} is not in the journal")
            _validate_remote_record(existing, record)
            accepted = existing.model_copy(
                update={
                    "state": JournalState.ACCEPTED,
                    "updated_at": now,
                    "remote_record": record,
                }
            )
            self._write_unlocked(accepted)
            return accepted

    def _path(self, run_id: str) -> Path:
        if not _safe_run_id(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.directory / f"{run_id}.json"

    def _read_unlocked(self, run_id: str) -> JournalEntry | None:
        path = self._path(run_id)
        try:
            return self._read_path(path)
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_path(path: Path) -> JournalEntry:
        return JournalEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_unlocked(self, entry: JournalEntry) -> None:
        target = self._path(entry.run_id)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory,
            prefix=f".{entry.run_id}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                absent_results = (
                    {
                        field
                        for field in ("exit_code", "launch_error")
                        if getattr(entry.remote_record, field) is None
                    }
                    if entry.remote_record is not None
                    else set()
                )
                handle.write(
                    entry.model_dump_json(indent=2, exclude={"remote_record": absent_results})
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, target)
            directory_descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _ensure_same_submission(
        existing: JournalEntry,
        target: RemoteTarget,
        request_spec: JobSpec,
    ) -> None:
        same_spec = request_spec == existing.request_spec or request_spec == existing.frozen_spec
        if existing.target != target or not same_spec:
            raise SubmissionConflict(
                f"run_id {existing.run_id!r} is already bound to another submission"
            )


class ControlClient:
    """Select one host, journal the decision, and submit without duplication."""

    def __init__(
        self,
        targets: Sequence[RemoteTarget],
        transport: RemoteTransport,
        journal: SubmissionJournal,
        *,
        clock: Callable[[], datetime] = utc_now,
        max_snapshot_age_seconds: float = 15.0,
    ):
        if not targets:
            raise ValueError("at least one remote target is required")
        if max_snapshot_age_seconds <= 0:
            raise ValueError("max_snapshot_age_seconds must be positive")
        by_id = {target.server_id: target for target in targets}
        if len(by_id) != len(targets):
            raise ValueError("remote target server ids must be unique")
        self.targets = tuple(sorted(targets))
        self._targets_by_id = by_id
        self.transport = transport
        self.journal = journal
        self.clock = clock
        self.max_snapshot_age_seconds = max_snapshot_age_seconds

    async def submit(
        self,
        spec: JobSpec,
        *,
        run_id: str | None = None,
    ) -> SubmissionResult:
        now = self.clock()
        stable_run_id = run_id or generate_run_id(spec, now)
        existing = self.journal.get(stable_run_id)
        if existing is not None:
            if spec != existing.request_spec and spec != existing.frozen_spec:
                raise SubmissionConflict(
                    f"run_id {stable_run_id!r} is already bound to another manifest"
                )
            return await self._reconcile_entry(existing)

        validate_new_submission(spec)
        target = await self._select_target(spec)
        frozen_spec = _freeze_host(spec, target.server_id)
        entry = self.journal.create_pending(
            run_id=stable_run_id,
            target=target,
            request_spec=spec,
            frozen_spec=frozen_spec,
            now=now,
        )
        return await self._finish_pending(entry, reconciled=False)

    async def reconcile(self, run_id: str) -> SubmissionResult:
        """Reconcile one existing run using its immutable journal identity."""

        entry = self.journal.get(run_id)
        if entry is None:
            raise SubmissionNotFound(run_id)
        return await self._reconcile_entry(entry)

    async def reconcile_pending(self) -> ReconcileReport:
        accepted: list[str] = []
        pending: list[str] = []
        errors: dict[str, str] = {}
        outcomes: dict[str, SubmissionState] = {}
        error_codes: dict[str, str] = {}
        warnings: dict[str, str] = {}
        for entry in self.journal.list_pending():
            try:
                result = await self._finish_pending(entry, reconciled=True)
                accepted.append(entry.run_id)
                outcomes[entry.run_id] = SubmissionState.ACCEPTED
                if result.journal_warning is not None:
                    warnings[entry.run_id] = result.journal_warning
            except SubmissionAttemptError as error:
                pending.append(entry.run_id)
                errors[entry.run_id] = _error_text(error)
                outcomes[entry.run_id] = error.submission_state
                error_codes[entry.run_id] = error.code
            except Exception as error:
                pending.append(entry.run_id)
                errors[entry.run_id] = _error_text(error)
                outcomes[entry.run_id] = SubmissionState.UNKNOWN
                error_codes[entry.run_id] = "reconcile_failed"
        return ReconcileReport(
            tuple(accepted),
            tuple(pending),
            errors,
            outcomes,
            error_codes,
            warnings,
        )

    async def _reconcile_entry(self, entry: JournalEntry) -> SubmissionResult:
        if entry.state == JournalState.ACCEPTED:
            if entry.remote_record is None:
                raise RemoteProtocolError("accepted journal entry has no remote record")
            return self._result(entry, entry.remote_record, reconciled=True)
        return await self._finish_pending(entry, reconciled=True)

    async def _select_target(self, spec: JobSpec) -> RemoteTarget:
        requested_host = spec.resources.host
        if requested_host != "auto":
            try:
                return self._targets_by_id[requested_host]
            except KeyError as error:
                raise HostSelectionError(f"unknown requested host: {requested_host}") from error

        # A server-local client has no placement choice. Accept into that host's
        # queue even while telemetry is unavailable; submit still checks Worker
        # readiness, and the scheduler alone decides when GPUs can be leased.
        if len(self.targets) == 1:
            return self.targets[0]

        inspections = await asyncio.gather(
            *(self.transport.inspect(target) for target in self.targets),
            return_exceptions=True,
        )
        now = self.clock()
        evaluations: list[HostEvaluation] = []
        errors: list[str] = []
        for target, inspected in zip(self.targets, inspections, strict=True):
            if isinstance(inspected, BaseException):
                errors.append(f"{target.server_id}: {_error_text(inspected)}")
                continue
            if not inspected.snapshot.is_usable(
                target.server_id, self.max_snapshot_age_seconds, now
            ):
                errors.append(f"{target.server_id}: snapshot is unhealthy or stale")
                continue
            evaluation = _evaluate_host(target, inspected, spec)
            if evaluation is None:
                errors.append(f"{target.server_id}: requested GPU shape is unsupported")
                continue
            evaluations.append(evaluation)
        if not evaluations:
            details = "; ".join(errors) or "no targets returned a usable state"
            raise HostSelectionError(f"no live host can accept the job: {details}")
        return min(evaluations, key=lambda evaluation: evaluation.score).target

    async def _finish_pending(
        self,
        entry: JournalEntry,
        *,
        reconciled: bool,
    ) -> SubmissionResult:
        target = entry.target
        try:
            existing = await self.transport.get_job(target, entry.run_id)
        except Exception as error:
            raise self._uncertain(
                entry,
                f"cannot reconcile {entry.run_id!r} on {target.server_id}: "
                f"{_error_text(error)}",
                code=_exception_code(error, "lookup_failed"),
            ) from error
        if existing is not None:
            return self._accept_existing(entry, existing)

        try:
            record = await self.transport.submit(target, entry.run_id, entry.frozen_spec)
        except DefinitiveRemoteSubmissionError as error:
            raise self._not_accepted(
                entry,
                f"submission of {entry.run_id!r} to {target.server_id} was rejected: "
                f"{_error_text(error)}",
                code=error.code,
            ) from error
        except Exception as submit_error:
            return await self._reconcile_after_uncertain_submit(entry, submit_error)
        try:
            return self._accept(entry, record, reconciled=reconciled)
        except (RemoteProtocolError, SubmissionConflict) as response_error:
            return await self._reconcile_after_uncertain_submit(entry, response_error)

    async def _reconcile_after_uncertain_submit(
        self,
        entry: JournalEntry,
        submit_error: BaseException,
    ) -> SubmissionResult:
        try:
            existing = await self.transport.get_job(entry.target, entry.run_id)
        except Exception as reconcile_error:
            raise self._uncertain(
                entry,
                f"submission response for {entry.run_id!r} was unclear on "
                f"{entry.server_id}; reconciliation failed: {_error_text(reconcile_error)}",
                code=_exception_code(submit_error, "submission_uncertain"),
            ) from submit_error
        if existing is not None:
            return self._accept_existing(entry, existing)
        raise self._uncertain(
            entry,
            f"submission of {entry.run_id!r} to {entry.server_id} was not acknowledged; "
            "the pending journal entry must be retried on the same host",
            code=_exception_code(submit_error, "submission_uncertain"),
        ) from submit_error

    def _accept_existing(self, entry: JournalEntry, record: JobRecord) -> SubmissionResult:
        try:
            _validate_remote_record(entry, record)
        except SubmissionConflict as error:
            raise self._not_accepted(
                entry,
                f"remote run {entry.run_id!r} on {entry.server_id} conflicts with the "
                f"saved submission: {_error_text(error)}",
                code="run_id_conflict",
            ) from error
        except RemoteProtocolError as error:
            raise self._uncertain(
                entry,
                f"remote lookup for {entry.run_id!r} on {entry.server_id} returned an "
                f"unrelated record: {_error_text(error)}",
                code="remote_protocol_error",
            ) from error
        return self._accept(entry, record, reconciled=True)

    def _accept(
        self,
        entry: JournalEntry,
        record: JobRecord,
        *,
        reconciled: bool,
    ) -> SubmissionResult:
        _validate_remote_record(entry, record)
        try:
            accepted = self.journal.mark_accepted(entry.run_id, record, now=self.clock())
        except OSError as error:
            warning = (
                f"remote accepted {entry.run_id!r}, but the accepted journal update failed: "
                f"{_error_text(error)}"
            )
            return self._result(
                entry,
                record,
                reconciled=reconciled,
                journal_warning=warning,
            )
        return self._result(accepted, record, reconciled=reconciled)

    @staticmethod
    def _result(
        entry: JournalEntry,
        record: JobRecord,
        *,
        reconciled: bool,
        journal_warning: str | None = None,
    ) -> SubmissionResult:
        return SubmissionResult(
            run_id=entry.run_id,
            target=entry.target,
            spec=entry.frozen_spec,
            record=record,
            reconciled=reconciled,
            journal_warning=journal_warning,
        )

    @staticmethod
    def _uncertain(entry: JournalEntry, message: str, *, code: str) -> SubmissionUncertain:
        return SubmissionUncertain(
            message,
            run_id=entry.run_id,
            server_id=entry.server_id,
            code=code,
        )

    @staticmethod
    def _not_accepted(
        entry: JournalEntry,
        message: str,
        *,
        code: str,
    ) -> SubmissionNotAccepted:
        return SubmissionNotAccepted(
            message,
            run_id=entry.run_id,
            server_id=entry.server_id,
            code=code,
        )


def generate_run_id(spec: JobSpec, created_at: datetime) -> str:
    """Generate a deterministic client-side id for one timestamped submission intent."""

    if created_at.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    canonical = json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:12]
    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{spec.id}-{timestamp}-{digest}"


def _freeze_host(spec: JobSpec, server_id: str) -> JobSpec:
    resources = spec.resources.model_copy(update={"host": server_id})
    return spec.model_copy(update={"resources": resources})


def _evaluate_host(
    target: RemoteTarget,
    state: RemoteHostState,
    spec: JobSpec,
) -> HostEvaluation | None:
    request = spec.resources
    gpu_by_index = state.snapshot.by_index()
    if request.gpus is not None:
        capable = [gpu_by_index.get(index) for index in request.gpus]
        if any(
            gpu is None
            or gpu.memory_total_mib is None
            or gpu.memory_total_mib < request.memory_mib
            for gpu in capable
        ):
            return None
        candidate_gpus = [gpu for gpu in capable if gpu is not None]
    else:
        candidate_gpus = [
            gpu
            for gpu in state.snapshot.gpus
            if gpu.memory_total_mib is not None and gpu.memory_total_mib >= request.memory_mib
        ]
        if len(candidate_gpus) < request.gpu_count:
            return None

    reserved = {
        gpu_index
        for record in state.queue
        if record.status
        in {
            JobStatus.LEASED,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
        }
        and record.allocation is not None
        and record.allocation.server_id == target.server_id
        for gpu_index in record.allocation.gpu_indices
    }
    available = [
        gpu
        for gpu in candidate_gpus
        if gpu.index not in reserved and gpu.can_allocate(request.memory_mib)
    ]
    queued_gpu_demand = sum(
        record.spec.resources.gpu_count
        for record in state.queue
        if record.status in {JobStatus.QUEUED, JobStatus.LEASED, JobStatus.STARTING}
    )
    active_job_count = sum(1 for record in state.queue if not record.status.terminal)
    required = len(candidate_gpus) if request.gpus is not None else request.gpu_count
    return HostEvaluation(
        target=target,
        state=state,
        available_gpu_count=len(available),
        queued_gpu_demand=queued_gpu_demand,
        active_job_count=active_job_count,
        can_fit_now=len(available) >= required,
    )


def _validate_remote_record(entry: JournalEntry, record: JobRecord) -> None:
    if record.run_id != entry.run_id:
        raise RemoteProtocolError(
            f"remote returned run_id {record.run_id!r}, expected {entry.run_id!r}"
        )
    if record.spec != entry.frozen_spec:
        raise SubmissionConflict(f"remote run_id {entry.run_id!r} exists with a different manifest")


def _safe_run_id(value: str) -> bool:
    return is_valid_run_id(value)


def _error_text(error: BaseException) -> str:
    message = str(error).strip() or error.__class__.__name__
    return message.replace("\n", " ")[:400]


def _exception_code(error: BaseException, fallback: str) -> str:
    code = getattr(error, "code", None)
    return code if isinstance(code, str) and code else fallback
