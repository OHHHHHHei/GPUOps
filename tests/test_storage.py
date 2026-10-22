from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from gpuops.core.storage import (
    GIB,
    StorageBudget,
    StorageQuotaExceeded,
)


def write_bytes(path, size: int, *, mtime: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime, mtime))


def test_default_budget_is_one_gibibyte(tmp_path) -> None:
    storage = StorageBudget(tmp_path)

    assert storage.total_budget_bytes == GIB
    assert storage.log_budget_bytes == GIB


@pytest.mark.parametrize("finish_write", ["replace", "unlink"])
def test_usage_ignores_temporary_file_removed_after_enumeration(
    tmp_path, monkeypatch, finish_write
) -> None:
    storage = StorageBudget(tmp_path)
    heartbeat = tmp_path / "worker-health.json"
    heartbeat.write_bytes(b"old heartbeat")
    temporary = tmp_path / ".worker-health.json.tmp"
    updated = b"updated heartbeat"
    temporary.write_bytes(updated)
    regular_files = StorageBudget._regular_files

    def enumerate_then_finish_write(directory):
        paths = regular_files(directory)
        if temporary in paths:
            if finish_write == "replace":
                temporary.replace(heartbeat)
            else:
                temporary.unlink()
        return paths

    monkeypatch.setattr(
        StorageBudget, "_regular_files", staticmethod(enumerate_then_finish_write)
    )

    usage = storage.usage()

    assert not temporary.exists()
    assert heartbeat.read_bytes() == (updated if finish_write == "replace" else b"old heartbeat")
    assert usage.total_bytes == heartbeat.stat().st_size
    assert usage.log_bytes == 0
    assert usage.event_archive_bytes == 0


@pytest.mark.parametrize("error", [PermissionError("permission denied"), OSError("I/O error")])
def test_usage_propagates_other_file_stat_errors(tmp_path, monkeypatch, error) -> None:
    storage = StorageBudget(tmp_path)
    heartbeat = tmp_path / "worker-health.json"
    heartbeat.write_bytes(b"heartbeat")
    original_stat = Path.stat

    def fail_heartbeat_stat(path, *args, **kwargs):
        if path == heartbeat:
            raise error
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        StorageBudget,
        "_regular_files",
        staticmethod(lambda directory: [heartbeat] if directory == tmp_path else []),
    )
    monkeypatch.setattr(Path, "stat", fail_heartbeat_stat)

    with pytest.raises(type(error)) as caught:
        storage.usage()

    assert caught.value is error


def test_storage_paths_and_prepared_logs_are_private(tmp_path) -> None:
    storage = StorageBudget(tmp_path)
    log = storage.prepare_log_file("project/run-1/worker.log")

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(storage.logs_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(storage.event_archive_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(log.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


@pytest.mark.parametrize("path", ["../outside.log", "/tmp/outside.log", "."])
def test_log_path_cannot_escape_managed_directory(tmp_path, path) -> None:
    storage = StorageBudget(tmp_path)

    with pytest.raises(ValueError):
        storage.prepare_log_file(path)


def test_log_budget_cleanup_removes_oldest_first(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=1000,
        log_budget_bytes=100,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    oldest = storage.logs_directory / "old.log"
    newest = storage.logs_directory / "new.log"
    write_bytes(oldest, 60, mtime=1)
    write_bytes(newest, 30, mtime=2)

    report = storage.cleanup_high_watermark()

    assert report.removed_paths == ("logs/old.log",)
    assert report.after.log_bytes == 30
    assert not oldest.exists()
    assert newest.exists()


def test_global_high_water_cleanup_includes_event_archives(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=200,
        log_budget_bytes=200,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    state = tmp_path / "jobs" / "must-stay.json"
    archive = storage.event_archive_directory / "events-1.jsonl.gz"
    log = storage.logs_directory / "worker.log"
    write_bytes(state, 70, mtime=0)
    write_bytes(archive, 50, mtime=1)
    write_bytes(log, 50, mtime=2)

    report = storage.cleanup_high_watermark()

    assert state.exists()
    assert report.after.total_bytes <= 100
    assert "event-archive/events-1.jsonl.gz" in report.removed_paths


def test_cleanup_ignores_symlinks_and_external_experiment_outputs(tmp_path) -> None:
    external = tmp_path.parent / "experiment-output.bin"
    external.write_bytes(b"important-result")
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=100,
        log_budget_bytes=100,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    (storage.logs_directory / "external").symlink_to(external)

    usage = storage.usage()
    report = storage.cleanup_high_watermark()

    assert usage.log_bytes == 0
    assert report.removed_paths == ()
    assert external.read_bytes() == b"important-result"


def test_protected_active_log_is_not_removed(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=100,
        log_budget_bytes=100,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    active = storage.logs_directory / "active.log"
    write_bytes(active, 90, mtime=1)

    report = storage.cleanup_high_watermark(protected_paths=[active])

    assert report.removed_paths == ()
    assert active.exists()
    with pytest.raises(StorageQuotaExceeded):
        storage.ensure_capacity(additional_bytes=11, protected_paths=[active])


def test_unmanaged_state_causes_hard_limit_error_instead_of_deletion(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=100,
        log_budget_bytes=100,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    state = tmp_path / "jobs" / "job.json"
    write_bytes(state, 95, mtime=1)

    with pytest.raises(StorageQuotaExceeded):
        storage.ensure_capacity(additional_bytes=10)

    assert state.exists()


def test_log_reservation_is_atomic_counted_and_releasable(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=200,
        log_budget_bytes=150,
        high_watermark=0.9,
        target_watermark=0.7,
    )

    reservation = storage.reserve_log_capacity("active-run", 80)
    repeated = storage.reserve_log_capacity("active-run", 80)

    assert repeated == reservation
    assert reservation.stat().st_size == 80
    assert storage.usage().log_bytes == 80
    with pytest.raises(StorageQuotaExceeded):
        storage.reserve_log_capacity("second-run", 80)

    storage.release_log_capacity("active-run")
    assert storage.usage().log_bytes == 0


@pytest.mark.parametrize("run_id", ["_review", ".review", "x" * 192])
def test_log_reservations_share_the_run_id_contract(tmp_path, run_id) -> None:
    storage = StorageBudget(tmp_path)

    reservation = storage.reserve_log_capacity(run_id, 1)

    assert reservation.name == f"{run_id}.reserve"
    storage.release_log_capacity(run_id)
    assert not reservation.exists()


def test_active_run_logs_are_protected_by_reservation(tmp_path) -> None:
    storage = StorageBudget(
        tmp_path,
        total_budget_bytes=300,
        log_budget_bytes=200,
        high_watermark=0.8,
        target_watermark=0.5,
    )
    storage.reserve_log_capacity("active", 40)
    active = storage.logs_directory / "active" / "console.log"
    old = storage.logs_directory / "finished" / "console.log"
    write_bytes(active, 50, mtime=1)
    write_bytes(old, 80, mtime=0)

    report = storage.cleanup_high_watermark(additional_bytes=1, additional_log_bytes=1)

    assert active.exists()
    assert not old.exists()
    assert "logs/finished/console.log" in report.removed_paths
