from __future__ import annotations

import stat
from datetime import datetime, timedelta, timezone

import pytest

from gpuops.core.queue import (
    ConcurrentUpdate,
    InvalidRunId,
    QueueStore,
    ResourceConflict,
    SubmissionConflict,
)
from gpuops.core.state_machine import InvalidTransition
from gpuops.core.storage import StorageBudget
from gpuops.models import JobSpec, JobStatus, LaunchInfo

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def spec(identifier: str, priority: str = "normal") -> JobSpec:
    return JobSpec(
        id=identifier,
        project="tests",
        command=["python", "train.py"],
        working_directory="/work",
        schedule={"priority": priority},
    )


def test_enqueue_is_atomic_and_records_event(tmp_path) -> None:
    store = QueueStore(tmp_path)
    job = store.enqueue(spec("one"), now=NOW)

    persisted = store.get(job.run_id)
    assert persisted == job
    assert list((tmp_path / "jobs").glob("*.tmp")) == []
    events = store.events()
    assert [(event.sequence, event.event_type) for event in events] == [(1, "enqueued")]


def test_gpu_uuid_conflict_is_rejected_after_device_index_changes(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(spec("before-renumber"), now=NOW)
    second = store.enqueue(spec("after-renumber"), now=NOW)
    leased = store.lease(
        first.run_id, server_id="gpu01", gpu_indices=[4], gpu_uuids={4: "GPU-physical"},
        worker_id="worker", ttl=timedelta(seconds=30), expected_revision=0, now=NOW,
    )
    assert store.get(first.run_id).allocation.gpu_uuids == leased.allocation.gpu_uuids
    with pytest.raises(ResourceConflict):
        store.lease(
            second.run_id, server_id="gpu01", gpu_indices=[3], gpu_uuids={3: "GPU-physical"},
            worker_id="worker", ttl=timedelta(seconds=30), expected_revision=0, now=NOW,
        )


def test_enqueue_with_stable_run_id_is_idempotent(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(spec("retry"), now=NOW, run_id="agent-request-123")
    second = store.enqueue(
        spec("retry"),
        now=NOW + timedelta(hours=1),
        run_id="agent-request-123",
    )

    assert second == first
    assert len(store.list()) == 1
    assert [event.event_type for event in store.events()] == ["enqueued"]


def test_stable_run_id_conflict_does_not_overwrite_original(tmp_path) -> None:
    store = QueueStore(tmp_path)
    original = store.enqueue(spec("original"), now=NOW, run_id="stable-id")

    with pytest.raises(SubmissionConflict, match="different job spec"):
        store.enqueue(spec("different"), now=NOW, run_id="stable-id")

    assert store.get("stable-id") == original
    assert len(store.events()) == 1


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "two words", ".", "中文", "x" * 193],
)
def test_stable_run_id_is_validated(tmp_path, run_id) -> None:
    store = QueueStore(tmp_path)

    with pytest.raises(InvalidRunId):
        store.enqueue(spec("invalid"), run_id=run_id)


@pytest.mark.parametrize("run_id", ["_review", ".review", "x" * 192])
def test_stable_run_id_keeps_legacy_safe_names(tmp_path, run_id) -> None:
    store = QueueStore(tmp_path)

    queued = store.enqueue(spec("legacy-compatible"), run_id=run_id)

    assert queued.run_id == run_id
    assert store.get(run_id) == queued


def test_queue_state_permissions_are_user_only(tmp_path) -> None:
    root = tmp_path / "state"
    store = QueueStore(root)
    store.enqueue(spec("private"), now=NOW)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "jobs").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "queue.lock").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "events.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE(next((root / "jobs").glob("*.json")).stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "event-sequence").stat().st_mode) == 0o600


def test_queue_can_share_a_parent_state_budget(tmp_path) -> None:
    state = tmp_path / "state"
    storage = StorageBudget(state)
    store = QueueStore(state / "queue", storage=storage)

    queued = store.enqueue(spec("shared-budget"), now=NOW)

    assert store.get(queued.run_id) == queued
    assert storage.usage().total_bytes > 0


def test_queue_rejects_an_unrelated_storage_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside"):
        QueueStore(tmp_path / "queue", storage=StorageBudget(tmp_path / "other"))


def test_event_log_rotates_to_bounded_compressed_archives(tmp_path) -> None:
    store = QueueStore(tmp_path, event_log_max_bytes=512, event_archive_count=2)
    for index in range(12):
        store.enqueue(spec(f"event-{index}"), now=NOW, run_id=f"event-{index}")

    archives = sorted((tmp_path / "event-archive").glob("*.jsonl.gz"))
    events = store.events()

    assert (tmp_path / "events.jsonl").stat().st_size <= 512
    assert len(archives) <= 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in archives)
    assert [event.sequence for event in events] == sorted(
        event.sequence for event in events
    )
    assert len({event.sequence for event in events}) == len(events)
    assert events[-1].sequence == 12


def test_queue_orders_priority_then_fifo(tmp_path) -> None:
    store = QueueStore(tmp_path)
    low = store.enqueue(spec("low", "low"), now=NOW)
    newer_high = store.enqueue(spec("newer-high", "high"), now=NOW + timedelta(seconds=2))
    older_high = store.enqueue(spec("older-high", "high"), now=NOW + timedelta(seconds=1))

    assert [job.run_id for job in store.list()] == [
        older_high.run_id,
        newer_high.run_id,
        low.run_id,
    ]


def test_state_machine_rejects_unsafe_transition(tmp_path) -> None:
    store = QueueStore(tmp_path)
    job = store.enqueue(spec("unsafe"), now=NOW)

    with pytest.raises(InvalidTransition):
        store.transition(job.run_id, JobStatus.RUNNING)


def test_revision_prevents_stale_update(tmp_path) -> None:
    store = QueueStore(tmp_path)
    job = store.enqueue(spec("revision"), now=NOW)
    cancelled = store.transition(
        job.run_id,
        JobStatus.CANCELLED,
        expected_revision=job.revision,
        now=NOW + timedelta(seconds=1),
    )

    assert cancelled.revision == 1
    with pytest.raises(ConcurrentUpdate):
        store.transition(
            job.run_id,
            JobStatus.CANCELLED,
            expected_revision=job.revision,
        )


def test_atomic_lease_rejects_gpu_conflict(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(spec("first"), now=NOW)
    second = store.enqueue(spec("second"), now=NOW + timedelta(seconds=1))
    store.lease(
        first.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(seconds=30),
        expected_revision=0,
        now=NOW,
    )

    with pytest.raises(ResourceConflict):
        store.lease(
            second.run_id,
            server_id="gpu01",
            gpu_indices=[0],
            worker_id="worker",
            ttl=timedelta(seconds=30),
            expected_revision=0,
            now=NOW,
        )


def test_lease_must_match_declared_resources(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(
        JobSpec(
            id="exact",
            project="tests",
            command=["python", "train.py"],
            working_directory="/work",
            resources={"host": "gpu01", "gpu_count": 2, "gpus": [2, 3]},
        ),
        now=NOW,
    )

    with pytest.raises(ResourceConflict, match="exact request"):
        store.lease(
            queued.run_id,
            server_id="gpu01",
            gpu_indices=[0, 1],
            worker_id="worker",
            ttl=timedelta(seconds=30),
            expected_revision=0,
            now=NOW,
        )


def test_expired_lease_returns_to_queue(tmp_path) -> None:
    store = QueueStore(tmp_path)
    job = store.enqueue(spec("expiry"), now=NOW)
    store.lease(
        job.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(seconds=10),
        expected_revision=0,
        now=NOW,
    )

    released = store.release_expired_leases(NOW + timedelta(seconds=11))

    assert len(released) == 1
    assert released[0].status == JobStatus.QUEUED
    assert released[0].allocation is None
    assert store.events()[-1].event_type == "lease_expired"


def test_valid_execution_lifecycle(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("lifecycle"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[2],
        worker_id="worker",
        ttl=timedelta(seconds=30),
        expected_revision=0,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING)
    running = store.transition(
        starting.run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=1234, started_at=NOW + timedelta(seconds=2)),
    )
    finished = store.transition(running.run_id, JobStatus.SUCCEEDED)

    assert finished.status == JobStatus.SUCCEEDED
    assert finished.finished_at is not None


def test_cancel_queued_job_is_atomic_and_idempotent(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("cancel-queued"), now=NOW, run_id="cancel-queued")

    cancelled = store.request_cancel(queued.run_id, reason="no longer needed", now=NOW)
    repeated = store.request_cancel(queued.run_id, reason="ignored retry", now=NOW)

    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.finished_at == NOW
    assert repeated == cancelled
    assert repeated.revision == 1
    assert [event.event_type for event in store.events()] == [
        "enqueued",
        "cancel_requested",
    ]


def test_cancel_lease_wins_before_stale_launcher_can_start(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("cancel-leased"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(minutes=1),
        expected_revision=queued.revision,
        now=NOW,
    )

    cancelled = store.request_cancel(leased.run_id, now=NOW)

    assert cancelled.status == JobStatus.CANCELLED
    with pytest.raises(ConcurrentUpdate):
        store.transition(
            leased.run_id,
            JobStatus.STARTING,
            expected_status=JobStatus.LEASED,
            expected_revision=leased.revision,
        )


def test_cancel_starting_and_running_jobs_becomes_pending(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("cancel-starting"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(minutes=1),
        expected_revision=queued.revision,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING, now=NOW)
    pending_without_receipt = store.request_cancel(starting.run_id, now=NOW)

    assert pending_without_receipt.status == JobStatus.CANCEL_REQUESTED
    assert pending_without_receipt.launch is None

    other = store.enqueue(spec("cancel-running"), now=NOW + timedelta(seconds=1))
    other_leased = store.lease(
        other.run_id,
        server_id="gpu01",
        gpu_indices=[1],
        worker_id="worker",
        ttl=timedelta(minutes=1),
        expected_revision=other.revision,
        now=NOW,
    )
    other_starting = store.transition(other_leased.run_id, JobStatus.STARTING, now=NOW)
    running = store.transition(
        other_starting.run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=1234, started_at=NOW, external_id="token"),
        now=NOW,
    )

    pending = store.request_cancel(running.run_id, now=NOW)

    assert pending.status == JobStatus.CANCEL_REQUESTED
    assert pending.launch == running.launch


@pytest.mark.parametrize("terminal", [JobStatus.SUCCEEDED, JobStatus.FAILED])
def test_cancel_terminal_job_is_noop(tmp_path, terminal) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("terminal"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(minutes=1),
        expected_revision=queued.revision,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING, now=NOW)
    running = store.transition(
        starting.run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=1234, started_at=NOW),
        now=NOW,
    )
    finished = store.transition(
        running.run_id,
        terminal,
        failure_reason="failed" if terminal == JobStatus.FAILED else None,
        now=NOW,
    )

    assert store.request_cancel(finished.run_id, now=NOW) == finished
