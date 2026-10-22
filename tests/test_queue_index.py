from __future__ import annotations

import json
import os
import sqlite3
import stat
from datetime import datetime, timedelta, timezone

import pytest

from gpuops.core.queue import QueueStore
from gpuops.core.queue_index import INDEX_FILENAME
from gpuops.core.storage import StorageBudget
from gpuops.models import JobRecord, JobSpec, JobStatus, JobSummary, LaunchInfo, Priority

NOW = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def spec(
    identifier: str,
    *,
    project: str = "tests",
    priority: str = "normal",
    name: str | None = None,
    expected_duration: str | None = None,
) -> JobSpec:
    schedule: dict[str, str] = {"priority": priority}
    if expected_duration is not None:
        schedule["expected_duration"] = expected_duration
    return JobSpec(
        id=identifier,
        project=project,
        command=["python", "train.py", "--secret", "not-in-summary"],
        working_directory="/private/work",
        resources={"host": "gpu01", "gpus": [3]},
        schedule=schedule,
        runtime={"environment": {"TOKEN": "not-in-summary"}},
        metadata={"name": name} if name is not None else {},
    )


def cancel(
    store: QueueStore,
    identifier: str,
    *,
    project: str = "tests",
    created_at: datetime = NOW,
    finished_at: datetime | None = None,
) -> JobRecord:
    queued = store.enqueue(
        spec(identifier, project=project),
        now=created_at,
        run_id=identifier,
    )
    return store.request_cancel(
        queued.run_id,
        now=finished_at or created_at + timedelta(seconds=1),
    )


def test_job_summary_projects_only_safe_queue_fields() -> None:
    record = JobRecord(
        run_id="summary-run",
        spec=spec(
            "summary",
            priority="high",
            name="Readable name",
            expected_duration="5m",
        ),
        status=JobStatus.RUNNING,
        allocation={
            "server_id": "gpu01",
            "gpu_indices": [2],
            "worker_id": "worker",
            "lease_token": "lease",
            "leased_at": NOW,
            "lease_expires_at": NOW + timedelta(minutes=1),
        },
        launch=LaunchInfo(pid=1234, started_at=NOW + timedelta(seconds=2)),
        created_at=NOW,
        updated_at=NOW,
    )

    summary = JobSummary.from_record(record)

    assert summary == JobSummary(
        run_id="summary-run",
        name="Readable name",
        project="tests",
        status=JobStatus.RUNNING,
        priority=Priority.HIGH,
        gpus=[2],
        created_at=NOW,
        started_at=NOW + timedelta(seconds=2),
        finished_at=None,
        expected_duration=timedelta(minutes=5),
    )
    assert "command" not in summary.model_dump()
    assert "working_directory" not in summary.model_dump()
    assert "environment" not in summary.model_dump()


def test_literal_legacy_record_builds_index_on_first_summary_read(tmp_path) -> None:
    store = QueueStore(tmp_path)
    legacy = {
        "run_id": "legacy-finished",
        "spec": {
            "api_version": "gpuops/v1",
            "id": "legacy-job",
            "project": "legacy",
            "command": ["python", "legacy.py"],
            "working_directory": "/legacy",
            "resources": {"host": "auto", "gpu_count": 1, "gpus": None},
            "runtime": {"environment": {}, "conda_environment": None},
            "schedule": {
                "priority": "low",
                "expected_duration": 90,
                "not_before": None,
                "deadline": None,
            },
            "tracking": {
                "log_path": None,
                "output_directory": None,
                "progress_adapter": "none",
            },
            "metadata": {"name": "Legacy display"},
        },
        "status": "succeeded",
        "revision": 4,
        "created_at": "2026-09-19T10:00:00Z",
        "updated_at": "2026-09-19T10:02:00Z",
        "allocation": None,
        "launch": {"pid": 4321, "started_at": "2026-09-19T10:00:10Z"},
        "finished_at": "2026-09-19T10:02:00Z",
        "failure_reason": None,
    }
    (tmp_path / "jobs" / "legacy-finished.json").write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )

    assert not (tmp_path / INDEX_FILENAME).exists()
    summaries = store.summaries()

    assert summaries == [
        JobSummary(
            run_id="legacy-finished",
            name="Legacy display",
            project="legacy",
            status=JobStatus.SUCCEEDED,
            priority=Priority.LOW,
            gpus=[],
            created_at=datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc),
            started_at=datetime(2026, 9, 19, 10, 0, 10, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 19, 10, 2, tzinfo=timezone.utc),
            expected_duration=timedelta(seconds=90),
        )
    ]
    assert stat.S_IMODE((tmp_path / INDEX_FILENAME).stat().st_mode) == 0o600


def test_warm_summary_history_and_projects_do_not_read_job_json(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    cancel(store, "old-run", project="alpha")
    store.enqueue(spec("active-run", project="beta"), now=NOW, run_id="active-run")
    store.summaries()

    def fail_full_scan() -> list[JobRecord]:
        raise AssertionError("warm query scanned authoritative job JSON")

    monkeypatch.setattr(store, "_all_unlocked", fail_full_scan)

    assert {summary.run_id for summary in store.summaries()} == {"old-run", "active-run"}
    assert [summary.run_id for summary in store.history(server_id="gpu01")] == ["old-run"]
    assert store.history_projects() == ["alpha"]


def test_filtered_list_reads_only_index_matches(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    cancel(store, "historical")
    store.enqueue(spec("queued"), now=NOW, run_id="queued")
    store.summaries()
    read_run_ids: list[str] = []
    original_read = store._read_unlocked

    def track_read(run_id: str) -> JobRecord:
        read_run_ids.append(run_id)
        return original_read(run_id)

    monkeypatch.setattr(store, "_read_unlocked", track_read)
    monkeypatch.setattr(
        store,
        "_all_unlocked",
        lambda: (_ for _ in ()).throw(AssertionError("filtered list performed a full scan")),
    )

    records = store.list({JobStatus.QUEUED})

    assert [record.run_id for record in records] == ["queued"]
    assert read_run_ids == ["queued"]


def test_lease_and_expiry_scan_only_active_record_json(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    cancel(store, "historical")
    queued = store.enqueue(spec("lease-target"), now=NOW, run_id="lease-target")
    store.summaries()
    read_run_ids: list[str] = []
    original_read = store._read_unlocked

    def track_read(run_id: str) -> JobRecord:
        read_run_ids.append(run_id)
        if run_id == "historical":
            raise AssertionError("active operation read historical JSON")
        return original_read(run_id)

    monkeypatch.setattr(store, "_read_unlocked", track_read)
    monkeypatch.setattr(
        store,
        "_all_unlocked",
        lambda: (_ for _ in ()).throw(AssertionError("active operation performed a full scan")),
    )

    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[3],
        worker_id="worker",
        ttl=timedelta(seconds=1),
        expected_revision=queued.revision,
        now=NOW,
    )
    assert read_run_ids == [queued.run_id]

    read_run_ids.clear()
    released = store.release_expired_leases(now=NOW + timedelta(seconds=2))
    assert [record.run_id for record in released] == [leased.run_id]
    assert read_run_ids == [leased.run_id]


def test_incremental_transition_enters_history_without_rebuild(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("incremental"), now=NOW, run_id="incremental")
    store.summaries()
    cancelled = store.request_cancel(queued.run_id, now=NOW + timedelta(seconds=3))

    monkeypatch.setattr(
        store,
        "_all_unlocked",
        lambda: (_ for _ in ()).throw(AssertionError("incremental update rebuilt the index")),
    )

    assert store.history(server_id="gpu01") == [JobSummary.from_record(cancelled)]


def test_idempotent_enqueue_does_not_duplicate_index_row(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(spec("retry"), now=NOW, run_id="same-run")
    store.summaries()

    repeated = store.enqueue(
        spec("retry"),
        now=NOW + timedelta(hours=1),
        run_id="same-run",
    )

    assert repeated == first
    assert [summary.run_id for summary in store.summaries()] == ["same-run"]


def test_missing_and_old_index_rebuild_from_authoritative_records(tmp_path) -> None:
    store = QueueStore(tmp_path)
    finished = cancel(store, "recover-index")
    assert store.history(server_id="gpu01") == [JobSummary.from_record(finished)]
    index_path = tmp_path / INDEX_FILENAME

    index_path.unlink()
    assert QueueStore(tmp_path).history(server_id="gpu01") == [JobSummary.from_record(finished)]

    with sqlite3.connect(index_path) as connection:
        connection.execute("UPDATE metadata SET value = '0' WHERE key = 'schema_version'")
    assert QueueStore(tmp_path).history(server_id="gpu01") == [JobSummary.from_record(finished)]
    with sqlite3.connect(index_path) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("1",)


def test_legacy_atomic_record_replace_invalidates_warm_index(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(spec("rollback-write"), now=NOW, run_id="rollback-write")
    assert store.history(server_id="gpu01") == []
    jobs_directory = tmp_path / "jobs"
    old_signature = jobs_directory.stat()
    cancelled = JobRecord.model_validate(
        queued.model_copy(
            update={
                "status": JobStatus.CANCELLED,
                "revision": queued.revision + 1,
                "updated_at": NOW + timedelta(seconds=2),
                "finished_at": NOW + timedelta(seconds=2),
            }
        ).model_dump()
    )
    replacement = jobs_directory / ".rollback-write.legacy.tmp"
    replacement.write_text(cancelled.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(replacement, jobs_directory / "rollback-write.json")
    if jobs_directory.stat().st_mtime_ns == old_signature.st_mtime_ns:
        os.utime(
            jobs_directory,
            ns=(old_signature.st_atime_ns, old_signature.st_mtime_ns + 1_000_000),
        )

    assert store.history(server_id="gpu01") == [JobSummary.from_record(cancelled)]


def test_history_filters_projects_and_paginates_strictly(tmp_path) -> None:
    store = QueueStore(tmp_path)
    oldest = cancel(
        store,
        "oldest",
        project="alpha",
        created_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )
    middle = cancel(
        store,
        "middle",
        project="beta",
        created_at=NOW + timedelta(seconds=1),
        finished_at=NOW + timedelta(seconds=2),
    )
    newest = cancel(
        store,
        "newest",
        project="alpha",
        created_at=NOW + timedelta(seconds=2),
        finished_at=NOW + timedelta(seconds=3),
    )

    first_page = store.history(server_id="gpu01", limit=2)
    assert [summary.run_id for summary in first_page] == [newest.run_id, middle.run_id]
    assert middle.finished_at is not None
    second_page = store.history(
        server_id="gpu01",
        limit=2,
        before=(middle.finished_at, middle.run_id, "gpu01"),
    )
    assert [summary.run_id for summary in second_page] == [oldest.run_id]
    assert [summary.run_id for summary in store.history(server_id="gpu01", project="alpha")] == [
        newest.run_id,
        oldest.run_id,
    ]
    assert store.history_projects() == ["alpha", "beta"]


def test_history_cursor_uses_server_id_as_final_global_tiebreaker(tmp_path) -> None:
    gpu01 = QueueStore(tmp_path / "gpu01")
    gpu02 = QueueStore(tmp_path / "gpu02")
    finished_at = NOW + timedelta(seconds=1)
    on_gpu01 = cancel(gpu01, "same-run", finished_at=finished_at)
    cancel(gpu02, "same-run", finished_at=finished_at)
    cursor = (finished_at, "same-run", "gpu02")

    assert gpu02.history(server_id="gpu02", before=cursor) == []
    assert gpu01.history(server_id="gpu01", before=cursor) == [JobSummary.from_record(on_gpu01)]


def test_index_rebuild_obeys_shared_storage_budget_without_blocking_reads(tmp_path) -> None:
    state = tmp_path / "state"
    queue_root = state / "queue"
    storage = StorageBudget(state)
    store = QueueStore(queue_root, storage=storage)
    for index in range(3):
        cancel(store, f"budget-{index}", finished_at=NOW + timedelta(seconds=index + 1))
    store.summaries()
    index_path = queue_root / INDEX_FILENAME
    index_bytes = index_path.stat().st_size
    index_path.unlink()
    without_index = storage.usage().total_bytes
    constrained_total = without_index + index_bytes - 1
    constrained = StorageBudget(
        state,
        total_budget_bytes=constrained_total,
        log_budget_bytes=constrained_total,
    )

    constrained_store = QueueStore(queue_root, storage=constrained)
    assert [summary.run_id for summary in constrained_store.summaries()] == [
        "budget-0",
        "budget-1",
        "budget-2",
    ]
    assert [summary.run_id for summary in constrained_store.history(server_id="gpu01")] == [
        "budget-2",
        "budget-1",
        "budget-0",
    ]
    assert constrained_store.history_projects() == ["tests"]
    assert constrained_store.count() == 3

    assert constrained.usage().total_bytes <= constrained_total
    assert not index_path.exists()


def test_history_queries_seek_ordered_indexes_without_temporary_sort(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    finished = cancel(store, "indexed-history", finished_at=NOW + timedelta(seconds=1))
    assert finished.finished_at is not None
    captured: list[tuple[str, list[object]]] = []
    original_query = store.index._query

    def capture_query(load_records, query, parameters):
        captured.append((query, parameters))
        return original_query(load_records, query, parameters)

    monkeypatch.setattr(store.index, "_query", capture_query)
    store.history(server_id="gpu01")
    store.history(server_id="gpu01", project="alpha")
    store.history(
        server_id="gpu01",
        before=(finished.finished_at, finished.run_id, "gpu01"),
    )

    with sqlite3.connect(tmp_path / INDEX_FILENAME) as connection:
        plans = [
            " ".join(
                str(row[3]) for row in connection.execute(f"EXPLAIN QUERY PLAN {query}", parameters)
            )
            for query, parameters in captured
        ]

    assert "USING INDEX jobs_history" in plans[0]
    assert "USING INDEX jobs_project_history" in plans[1]
    assert "USING INDEX jobs_history" in plans[2]
    assert "(finished_us,run_id)<(?,?)" in plans[2]
    assert "TEMP B-TREE" not in " ".join(plans)


def test_two_store_instances_keep_incremental_index_consistent(tmp_path, monkeypatch) -> None:
    first = QueueStore(tmp_path)
    queued = first.enqueue(spec("shared-lock"), now=NOW, run_id="shared-lock")
    first.summaries()
    second = QueueStore(tmp_path)

    cancelled = second.request_cancel(queued.run_id, now=NOW + timedelta(seconds=2))
    monkeypatch.setattr(
        first,
        "_all_unlocked",
        lambda: (_ for _ in ()).throw(AssertionError("peer update forced a rebuild")),
    )

    assert first.history(server_id="gpu01") == [JobSummary.from_record(cancelled)]


def test_history_rejects_nonpositive_limit_and_naive_cursor(tmp_path) -> None:
    store = QueueStore(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        store.history(server_id="gpu01", limit=0)
    with pytest.raises(ValueError, match="timezone"):
        store.history(
            server_id="gpu01",
            before=(datetime(2026, 9, 20), "run", "gpu01"),
        )


def test_count_tracks_all_jobs_without_loading_records_on_warm_reads(tmp_path, monkeypatch):
    store = QueueStore(tmp_path)
    assert store.count() == 0
    store.enqueue(spec("count-active"), run_id="count-active", now=NOW)
    cancel(store, "count-terminal")
    assert store.count() == 2
    store.request_cancel("count-active", now=NOW)

    def no_records(*args, **kwargs):
        raise AssertionError("warm counts must not load job records or summaries")

    monkeypatch.setattr(QueueStore, "_all_unlocked", no_records)
    monkeypatch.setattr(QueueStore, "list", no_records)
    monkeypatch.setattr(QueueStore, "summaries", no_records)
    assert QueueStore(tmp_path).count() == 2
