"""Durable aggregation of follow-ups spanning jobs and servers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from gpuops.contracts import RunId
from gpuops.control import RemoteTarget
from gpuops.file_io import atomic_write_bytes
from gpuops.followups import (
    Followup,
    FollowupStore,
    JobKey,
    JobObservations,
    completion_message,
    job_key,
    job_outcome,
)
from gpuops.models import JobRecord, StrictModel, utc_now

GroupMode = Literal["per_job", "all_terminal"]
NotifyMode = Literal["none", "terminal", "analysis_complete"]


class RemoteJob(StrictModel):
    """A group member and its latest durable observation."""

    run_id: RunId
    server_id: str
    ssh_host: str
    job: JobRecord | None = None
    error: str | None = None
    event_id: str | None = None

    @model_validator(mode="after")
    def member_invariants(self) -> RemoteJob:
        RemoteTarget(self.server_id, self.ssh_host)
        if self.job is not None and str(self.job.run_id) != self.run_id:
            raise ValueError("observed job does not match the member run_id")
        if self.event_id is not None:
            UUID(self.event_id)
        return self


class GroupRecord(StrictModel):
    """Persistent definition and observations for one follow-up group."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str | None = None
    members: list[RemoteJob]
    event_id: str | None = None
    harness: str = "codex-desktop"
    thread_id: str
    workspace: str
    action: str
    mode: GroupMode = "all_terminal"
    notify: NotifyMode = "none"
    resume: bool = True
    context: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    stopped: bool = False

    @model_validator(mode="after")
    def group_invariants(self) -> GroupRecord:
        UUID(self.id)
        if not self.members:
            raise ValueError("a follow-up group requires at least one member")
        identities = [(member.server_id, member.ssh_host, member.run_id) for member in self.members]
        if len(set(identities)) != len(identities):
            raise ValueError("a follow-up group cannot contain duplicate members")
        required = (self.harness, self.workspace)
        if any(not value.strip() for value in required):
            raise ValueError("harness and workspace cannot be empty")
        if self.resume and (not self.thread_id.strip() or not self.action.strip()):
            raise ValueError("resuming a session requires thread_id and action")
        if self.name is not None and not self.name.strip():
            raise ValueError("group name cannot be empty")
        if self.mode == "all_terminal":
            if self.event_id is None:
                raise ValueError("all_terminal groups require one event_id")
            UUID(self.event_id)
            if any(member.event_id is not None for member in self.members):
                raise ValueError("all_terminal members cannot have individual event IDs")
        else:
            if self.event_id is not None:
                raise ValueError("per_job groups cannot have a shared event_id")
            event_ids = [member.event_id for member in self.members]
            if any(event_id is None for event_id in event_ids):
                raise ValueError("per_job members require individual event IDs")
            if len(set(event_ids)) != len(event_ids):
                raise ValueError("per_job event IDs must be unique")
        return self


class GroupStore:
    """JSON records stored separately from individual follow-up events."""

    def __init__(self, directory: Path):
        self.directory = directory.expanduser()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, group_id: str) -> Path:
        return self.directory / f"{UUID(group_id)}.json"

    def get(self, group_id: str) -> GroupRecord:
        return GroupRecord.model_validate_json(self._path(group_id).read_bytes())

    def scan(self) -> tuple[list[GroupRecord], list[dict[str, str]]]:
        """Read valid records while isolating malformed records."""
        groups: list[GroupRecord] = []
        invalid: list[dict[str, str]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                groups.append(GroupRecord.model_validate_json(path.read_bytes()))
            except (OSError, ValueError) as error:
                invalid.append({"path": str(path), "error": str(error)})
        return groups, invalid

    def list(self) -> list[GroupRecord]:
        return self.scan()[0]

    def save(self, group: GroupRecord) -> None:
        atomic_write_bytes(self._path(group.id), group.model_dump_json(indent=2).encode())


def _member_key(member: RemoteJob) -> tuple[str, str, str]:
    return member.server_id, member.ssh_host, member.run_id


def _member_set(group: GroupRecord) -> frozenset[tuple[str, str, str]]:
    return frozenset(_member_key(member) for member in group.members)


def _same_registration_target(
    existing: GroupRecord,
    candidate: GroupRecord,
) -> bool:
    target = (existing.harness, existing.thread_id, existing.workspace)
    if target != (candidate.harness, candidate.thread_id, candidate.workspace):
        return False
    if candidate.name is not None:
        return existing.name == candidate.name
    return existing.name is None and _member_set(existing) == _member_set(candidate)


def _registration_differences(
    existing: GroupRecord,
    candidate: GroupRecord,
) -> list[str]:
    fields = ("name", "action", "mode", "notify", "resume", "context")
    differences = [
        field for field in fields if getattr(existing, field) != getattr(candidate, field)
    ]
    if _member_set(existing) != _member_set(candidate):
        differences.append("members")
    return differences


def create_group(
    store: GroupStore,
    event_store: FollowupStore,
    members: list[RemoteJob],
    *,
    thread_id: str,
    workspace: str,
    action: str,
    mode: GroupMode = "all_terminal",
    harness: str = "codex-desktop",
    notify: NotifyMode = "none",
    resume: bool = True,
    context: dict[str, str] | None = None,
    name: str | None = None,
    on_registered: Callable[[], None] | None = None,
) -> GroupRecord:
    """Register a durable group and preallocate stable event IDs."""
    clean_members = [
        RemoteJob(
            run_id=member.run_id,
            server_id=member.server_id,
            ssh_host=member.ssh_host,
            event_id=str(uuid4()) if mode == "per_job" else None,
        )
        for member in members
    ]
    candidate = GroupRecord(
        name=name,
        members=clean_members,
        event_id=str(uuid4()) if mode == "all_terminal" else None,
        harness=harness,
        thread_id=thread_id,
        workspace=workspace,
        action=action,
        mode=mode,
        notify=notify,
        resume=resume,
        context=dict(context or {}),
    )
    # Group and event mutations share one lock, avoiding a second lock order.
    with event_store.lock():
        for existing in store.list():
            if not _same_registration_target(existing, candidate):
                continue
            if existing.stopped:
                raise ValueError(
                    "this follow-up group is stopped; use a new group name to register another"
                )
            differences = _registration_differences(existing, candidate)
            if differences:
                raise ValueError(
                    "this follow-up group already exists with different settings: "
                    + ", ".join(differences)
                )
            candidate = existing
            break
        else:
            store.save(candidate)
        if on_registered is not None:
            on_registered()
    return candidate


def _event_path(store: FollowupStore, event_id: str) -> Path:
    return store.directory / f"{UUID(event_id)}.json"


def _event_exists(store: FollowupStore, event_id: str) -> bool:
    return _event_path(store, event_id).exists()


def _all_terminal(group: GroupRecord) -> bool:
    return all(member.job is not None and member.job.status.terminal for member in group.members)


def _create_event(
    group: GroupRecord,
    events: FollowupStore,
    member: RemoteJob,
    event_id: str,
) -> None:
    if _event_exists(events, event_id):
        return
    assert member.job is not None and member.job.status.terminal
    event = Followup(
        id=event_id,
        run_id=member.run_id,
        server_id=member.server_id,
        ssh_host=member.ssh_host,
        harness=group.harness,
        thread_id=group.thread_id,
        workspace=group.workspace,
        action=group.action,
        status="ready" if group.resume else "observed",
        job=member.job,
        group_id=group.id,
        mode=group.mode,
        context=dict(group.context),
        outcomes=[job_outcome(item.server_id, item.run_id, item.job) for item in group.members],
        resume=group.resume,
        notify=group.notify,
    )
    if group.resume:
        event.message = completion_message(event)
    events.save(event)


def _pending_members(group: GroupRecord, events: FollowupStore) -> list[RemoteJob]:
    if group.stopped or (
        group.mode == "all_terminal"
        and group.event_id is not None
        and _event_exists(events, group.event_id)
    ):
        return []
    return [
        member for member in group.members if member.job is None or not member.job.status.terminal
    ]


def pending_group_jobs(records: list[GroupRecord], events: FollowupStore) -> Iterator[JobKey]:
    for group in records:
        for member in _pending_members(group, events):
            yield job_key(member.server_id, member.ssh_host, member.run_id)


def pending_group_events(records: list[GroupRecord], events: FollowupStore) -> int:
    """Account for a round interrupted between saving a group and creating its events."""
    pending = 0
    for group in records:
        if group.stopped:
            continue
        if group.mode == "all_terminal":
            assert group.event_id is not None
            pending += int(_all_terminal(group) and not _event_exists(events, group.event_id))
        else:
            for member in group.members:
                assert member.event_id is not None
                pending += int(
                    member.job is not None
                    and member.job.status.terminal
                    and not _event_exists(events, member.event_id)
                )
    return pending


def apply_group_observations(
    groups: GroupStore, events: FollowupStore, observations: JobObservations
) -> None:
    """Apply observations to current group records and create eligible events under the lock."""
    for group in groups.list():
        if group.stopped:
            continue
        changed = False
        newly_terminal = []
        for member in _pending_members(group, events):
            result = observations.get(job_key(member.server_id, member.ssh_host, member.run_id))
            if result is None:
                continue
            changed = True
            if isinstance(result, str):
                member.error = result
            else:
                member.job = result
                member.error = None
                if result.status.terminal:
                    newly_terminal.append(member)
        if changed:
            groups.save(group)
        if group.mode == "per_job":
            for member in group.members:
                if member.job is None or not member.job.status.terminal:
                    continue
                assert member.event_id is not None
                _create_event(group, events, member, member.event_id)
        elif _all_terminal(group):
            assert group.event_id is not None
            trigger = newly_terminal[-1] if newly_terminal else group.members[-1]
            _create_event(group, events, trigger, group.event_id)


def _planned_event_ids(group: GroupRecord) -> list[str]:
    if group.mode == "all_terminal":
        assert group.event_id is not None
        return [group.event_id]
    return [member.event_id for member in group.members if member.event_id is not None]


def summarize_group(record: GroupRecord, events: FollowupStore) -> dict[str, Any]:
    """Return public group state together with every preallocated event's real state."""
    event_rows: list[dict[str, Any]] = []
    event_models: list[Followup | None] = []
    for event_id in _planned_event_ids(record):
        path = _event_path(events, event_id)
        if not path.exists():
            event_rows.append({"id": event_id, "status": "pending", "exists": False})
            event_models.append(None)
            continue
        try:
            event = events.get(event_id)
        except (OSError, ValueError) as error:
            event_rows.append(
                {"id": event_id, "status": "invalid", "exists": True, "error": str(error)}
            )
            event_models.append(None)
            continue
        event_rows.append(
            {
                **event.model_dump(mode="json", exclude={"job", "message"}),
                "exists": True,
            }
        )
        event_models.append(event)

    completed = bool(event_models) and all(
        event is not None
        and (event.status == "completed" or (not event.resume and event.status == "observed"))
        for event in event_models
    )
    all_terminal = _all_terminal(record)
    all_delivery_stopped = bool(event_models) and all(
        event is not None and event.status == "stopped" for event in event_models
    )
    any_delivery_stopped = any(
        event is not None and event.status == "stopped" for event in event_models
    )
    if record.stopped:
        status = "stopped"
    elif all_delivery_stopped:
        status = "delivery_stopped"
    elif any_delivery_stopped:
        status = "partially_stopped"
    elif completed:
        status = "analysis_complete" if record.resume else "observed"
    elif all_terminal:
        status = "analysis_pending" if record.resume else "observation_pending"
    else:
        status = "waiting"

    return {
        "id": record.id,
        "name": record.name,
        "status": status,
        "mode": record.mode,
        "harness": record.harness,
        "thread_id": record.thread_id,
        "workspace": record.workspace,
        "action": record.action,
        "notify": record.notify,
        "resume": record.resume,
        "context": dict(record.context),
        "created_at": record.created_at.isoformat(),
        "stopped": record.stopped,
        "all_terminal": all_terminal,
        "completed": completed,
        "completion_kind": "analysis" if record.resume else "observation",
        "members": [
            {
                **job_outcome(member.server_id, member.run_id, member.job),
                "ssh_host": member.ssh_host,
                "error": member.error,
                "event_id": member.event_id,
            }
            for member in record.members
        ],
        "events": event_rows,
    }


def stop_group(
    groups: GroupStore,
    events: FollowupStore,
    group_id: str,
) -> dict[str, Any]:
    """Stop future events and any existing events that have not started delivery."""
    with events.lock():
        record = groups.get(group_id)
        record.stopped = True
        groups.save(record)
        for event_id in _planned_event_ids(record):
            if not _event_exists(events, event_id):
                continue
            event = events.get(event_id)
            if event.status in {"waiting", "ready"} or (
                event.status == "observed" and event.email_sent_at is None
            ):
                event.status = "stopped"
                events.save(event)
        return summarize_group(record, events)
