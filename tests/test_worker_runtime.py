from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpuops import __version__
from gpuops.worker.app import WORKER_HEARTBEAT_MAX_AGE_SECONDS, worker_readiness
from gpuops.worker.probe import LocalGpuProbe
from gpuops.worker.runtime import (
    ProcessIdentity,
    RuntimePaths,
    WorkerAlreadyRunning,
    WorkerFileLock,
    WorkerProcessState,
    WorkerProcessStatus,
    WorkerRuntime,
    inspect_worker,
    read_process_identity,
    request_worker_stop,
    spawn_detached_worker,
    write_process_identity,
)
from gpuops.worker.service import Worker, WorkerTickResult

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "options",
    [
        {"idle_memory_baseline_mib": -1},
        {"idle_memory_tolerance_mib": -1},
        {"idle_memory_baseline_mib": 1.5},
        {"idle_memory_tolerance_mib": False},
    ],
)
def test_local_probe_rejects_invalid_memory_calibration(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        LocalGpuProbe("gpu01", **options)


def test_worker_file_lock_allows_only_one_runtime(tmp_path: Path) -> None:
    path = tmp_path / "worker.lock"
    first = WorkerFileLock(path)
    second = WorkerFileLock(path)
    first.acquire()
    try:
        with pytest.raises(WorkerAlreadyRunning):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


@pytest.mark.parametrize("heartbeat_status", ["starting", "degraded", "stopped"])
def test_worker_readiness_requires_a_fresh_running_heartbeat(heartbeat_status: str) -> None:
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    status = WorkerProcessStatus(
        state=WorkerProcessState.RUNNING,
        identity=identity,
        heartbeat={
            "server_id": "gpu01",
            "worker_id": "gpu01:host",
            "release": __version__,
            "pid": identity.pid,
            "status": heartbeat_status,
            "started_at": NOW.isoformat(),
            "heartbeat_at": NOW.isoformat(),
            "iteration": 1,
            "tick_failures": 0,
        },
    )

    readiness = worker_readiness(status, server_id="gpu01", now=NOW)

    assert readiness.ready is False
    assert heartbeat_status in readiness.reason


def test_worker_readiness_matches_identity_server_and_age() -> None:
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")

    def status(*, pid=4321, server_id="gpu01", heartbeat_at=NOW):
        return WorkerProcessStatus(
            state=WorkerProcessState.RUNNING,
            identity=identity,
            heartbeat={
                "server_id": server_id,
                "worker_id": f"{server_id}:host",
                "release": __version__,
                "pid": pid,
                "status": "running",
                "started_at": NOW.isoformat(),
                "heartbeat_at": heartbeat_at.isoformat(),
                "iteration": 1,
                "tick_failures": 0,
            },
        )

    assert worker_readiness(status(), server_id="gpu01", now=NOW).ready is True
    assert worker_readiness(status(pid=999), server_id="gpu01", now=NOW).ready is False
    assert worker_readiness(status(server_id="gpu02"), server_id="gpu01", now=NOW).ready is False
    assert (
        worker_readiness(
            status(heartbeat_at=NOW - timedelta(seconds=WORKER_HEARTBEAT_MAX_AGE_SECONDS + 1)),
            server_id="gpu01",
            now=NOW,
        ).ready
        is False
    )
    assert (
        worker_readiness(
            status(heartbeat_at=NOW - timedelta(seconds=WORKER_HEARTBEAT_MAX_AGE_SECONDS)),
            server_id="gpu01",
            now=NOW,
        ).ready
        is True
    )


def test_foreground_loop_injects_tick_clock_sleeper_and_writes_heartbeat(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    tick_calls = 0
    sleeps: list[float] = []
    current_time = NOW

    def clock() -> datetime:
        nonlocal current_time
        result = current_time
        current_time += timedelta(seconds=1)
        return result

    def tick() -> None:
        nonlocal tick_calls
        tick_calls += 1
        if tick_calls == 2:
            raise RuntimeError("temporary probe failure")

    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=tick,
        interval_seconds=2.5,
        clock=clock,
        sleeper=sleeps.append,
        getpid=lambda: identity.pid,
        identity_reader=lambda pid: identity if pid == identity.pid else None,
    )

    iterations = runtime.run_foreground(max_iterations=3, install_signal_handlers=False)

    assert iterations == 3
    assert tick_calls == 3
    assert sleeps == [2.5, 2.5]
    assert not paths.pid_path.exists()
    health = json.loads(paths.heartbeat_path.read_text(encoding="utf-8"))
    assert health["status"] == "stopped"
    assert health["iteration"] == 3
    assert health["tick_failures"] == 1
    assert health["last_tick_error"] is None
    assert paths.heartbeat_path.stat().st_mode & 0o777 == 0o600


def test_foreground_loop_ticks_and_closes_sampler_inside_worker_lock(tmp_path: Path) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    events: list[str] = []

    class Sampler:
        def tick(self) -> None:
            events.append("sample")

        def close(self) -> None:
            events.append("close")

    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=lambda: events.append("worker"),
        interval_seconds=0,
        clock=lambda: NOW,
        sleeper=lambda _: None,
        getpid=lambda: identity.pid,
        identity_reader=lambda pid: identity if pid == identity.pid else None,
        sampler_factory=lambda: events.append("create") or Sampler(),
    )

    assert runtime.run_foreground(max_iterations=2, install_signal_handlers=False) == 2
    assert events == ["create", "sample", "worker", "sample", "worker", "close"]


def test_sampler_factory_is_not_called_without_worker_lock(tmp_path: Path) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    factory_calls: list[bool] = []

    def unexpected_factory():
        factory_calls.append(True)
        raise AssertionError("sampler factory ran before the worker lock was acquired")

    lock = WorkerFileLock(paths.lock_path)
    lock.acquire()
    try:
        runtime = WorkerRuntime(
            server_id="gpu01",
            worker_id="worker-1",
            paths=paths,
            tick=lambda: None,
            sampler_factory=unexpected_factory,
        )
        with pytest.raises(WorkerAlreadyRunning):
            runtime.run_foreground(max_iterations=1, install_signal_handlers=False)
    finally:
        lock.release()

    assert factory_calls == []


def test_sampler_tick_failure_does_not_skip_worker_tick(tmp_path: Path) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    worker_calls: list[bool] = []

    class BrokenSampler:
        def tick(self) -> None:
            raise RuntimeError("sampler unavailable")

        def close(self) -> None:
            pass

    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=lambda: worker_calls.append(True),
        clock=lambda: NOW,
        getpid=lambda: identity.pid,
        identity_reader=lambda pid: identity if pid == identity.pid else None,
        sampler_factory=BrokenSampler,
    )

    runtime.run_foreground(max_iterations=1, install_signal_handlers=False)

    health = json.loads(paths.heartbeat_path.read_text(encoding="utf-8"))
    assert worker_calls == [True]
    assert health["tick_failures"] == 1
    assert health["last_tick_error"] == "RuntimeError: sampler unavailable"


@pytest.mark.parametrize("fail_at", [None, "create", "start", "close"])
def test_notifications_share_worker_lifetime_without_affecting_health(
    tmp_path: Path,
    fail_at: str | None,
) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    events: list[str] = []

    def event(action: str) -> None:
        with pytest.raises(WorkerAlreadyRunning), WorkerFileLock(paths.lock_path):
            pass
        events.append(action)
        if action == fail_at:
            raise RuntimeError("notification service unavailable")

    class Notifier:
        def start(self) -> None:
            event("start")

        def close(self) -> None:
            event("close")

    def factory() -> Notifier:
        event("create")
        return Notifier()

    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=lambda: events.append("tick"),
        interval_seconds=0,
        clock=lambda: NOW,
        sleeper=lambda _: None,
        getpid=lambda: identity.pid,
        identity_reader=lambda _: identity,
        notifier_factory=factory,
    )
    assert runtime.run_foreground(max_iterations=2, install_signal_handlers=False) == 2
    assert events == (
        ["create", "tick", "tick"]
        if fail_at == "create"
        else ["create", "start", "tick", "tick", "close"]
    )
    health = json.loads(paths.heartbeat_path.read_text())
    assert health["tick_failures"] == 0
    assert health["last_tick_error"] is None
    assert not paths.pid_path.exists()
    with WorkerFileLock(paths.lock_path):
        pass


def test_foreground_loop_surfaces_task_errors_in_degraded_heartbeat(
    monkeypatch, tmp_path: Path
) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    heartbeat_statuses: list[str] = []

    from gpuops.worker import runtime as runtime_module

    write_heartbeat = runtime_module.write_heartbeat

    def capture_heartbeat(path, heartbeat):  # type: ignore[no-untyped-def]
        heartbeat_statuses.append(heartbeat.status)
        write_heartbeat(path, heartbeat)

    monkeypatch.setattr(runtime_module, "write_heartbeat", capture_heartbeat)
    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=lambda: WorkerTickResult(
            errors={
                "run-b": "poll failed: remote\nresponse was invalid",
                "run-a": "launch recovery pending",
            }
        ),
        clock=lambda: NOW,
        getpid=lambda: identity.pid,
        identity_reader=lambda pid: identity if pid == identity.pid else None,
    )

    assert runtime.run_foreground(max_iterations=1, install_signal_handlers=False) == 1

    health = json.loads(paths.heartbeat_path.read_text(encoding="utf-8"))
    assert health["status"] == "stopped"
    assert health["tick_failures"] == 1
    assert health["last_tick_error"] == (
        "job errors: run-a: launch recovery pending; "
        "run-b: poll failed: remote response was invalid"
    )
    assert heartbeat_statuses == ["starting", "degraded", "stopped"]


def test_foreground_loop_accepts_none_tick_result(tmp_path: Path) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path / "state")
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")
    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="worker-1",
        paths=paths,
        tick=lambda: None,
        clock=lambda: NOW,
        getpid=lambda: identity.pid,
        identity_reader=lambda pid: identity if pid == identity.pid else None,
    )

    runtime.run_foreground(max_iterations=1, install_signal_handlers=False)

    health = json.loads(paths.heartbeat_path.read_text(encoding="utf-8"))
    assert health["tick_failures"] == 0
    assert health["last_tick_error"] is None


def test_worker_tick_preserves_launch_and_failure_recording_errors() -> None:
    class BrokenStore:
        get_calls = 0

        def list(self, statuses):  # type: ignore[no-untyped-def]
            del statuses
            return []

        def get(self, run_id):  # type: ignore[no-untyped-def]
            del run_id
            self.get_calls += 1
            if self.get_calls == 1:
                raise OSError("queue read failed")
            raise RuntimeError("state store offline")

    class OneDecisionScheduler:
        server_id = "gpu01"
        worker_id = "worker-1"

        def tick(self):  # type: ignore[no-untyped-def]
            return [SimpleNamespace(run_id="run-a", lease_token="lease-a")]

    result = Worker(BrokenStore(), OneDecisionScheduler(), object()).tick()  # type: ignore[arg-type]

    assert result.errors["run-a"] == (
        "queue read failed; failed to record launch failure: RuntimeError: state store offline"
    )


def test_pid_identity_helpers_reject_reused_pid_and_stop_only_worker(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths.from_state_directory(tmp_path)
    worker = ProcessIdentity(pid=111, start_time_ticks=20, boot_id="boot-a")
    experiment_pid = 222
    write_process_identity(paths.pid_path, worker)

    running = inspect_worker(paths, identity_reader=lambda pid: worker if pid == 111 else None)
    assert running.state == WorkerProcessState.RUNNING

    signals: list[tuple[int, int]] = []
    assert request_worker_stop(
        paths,
        identity_reader=lambda pid: worker if pid == 111 else None,
        signal_sender=lambda pid, signum: signals.append((pid, signum)),
    )
    assert signals == [(worker.pid, signal.SIGTERM)]
    assert all(pid != experiment_pid for pid, _ in signals)

    reused = ProcessIdentity(pid=111, start_time_ticks=21, boot_id="boot-a")
    assert not request_worker_stop(
        paths,
        identity_reader=lambda pid: reused,
        signal_sender=lambda pid, signum: signals.append((pid, signum)),
    )
    assert signals == [(worker.pid, signal.SIGTERM)]
    stale = inspect_worker(paths, identity_reader=lambda pid: reused)
    assert stale.state == WorkerProcessState.STALE


def test_read_process_identity_handles_spaces_in_process_name(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    process_directory = proc_root / "77"
    process_directory.mkdir(parents=True)
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    fields = ["S"] + ["0"] * 18 + ["4567"] + ["0"] * 3
    (process_directory / "stat").write_text(
        f"77 (worker with spaces) {' '.join(fields)}\n", encoding="utf-8"
    )
    (proc_root / "sys/kernel/random/boot_id").write_text("boot-z\n", encoding="utf-8")

    identity = read_process_identity(77, proc_root)

    assert identity == ProcessIdentity(pid=77, start_time_ticks=4567, boot_id="boot-z")


def test_read_process_identity_does_not_require_a_valid_process_group(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    process_directory = proc_root / "77"
    process_directory.mkdir(parents=True)
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    fields = ["S", "0", "not-a-pgid"] + ["0"] * 16 + ["4567"] + ["0"] * 3
    (process_directory / "stat").write_text(
        f"77 (worker) {' '.join(fields)}\n",
        encoding="utf-8",
    )
    (proc_root / "sys/kernel/random/boot_id").write_text("boot-z\n", encoding="utf-8")

    identity = read_process_identity(77, proc_root)

    assert identity == ProcessIdentity(pid=77, start_time_ticks=4567, boot_id="boot-z")


def test_read_process_identity_propagates_unexpected_procfs_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    process_directory = proc_root / "77"
    process_directory.mkdir(parents=True)

    def fail_read_text(_path: Path, *args, **kwargs) -> str:
        del args, kwargs
        raise OSError("unexpected procfs failure")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(OSError, match="unexpected procfs failure"):
        read_process_identity(77, proc_root)


def test_spawn_detached_worker_uses_no_shell_and_starts_only_worker_pid(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 7654

    def process_factory(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return FakeProcess()

    pid = spawn_detached_worker(
        ["python", "-m", "gpuops", "worker", "run"],
        stdout_path=tmp_path / "worker.log",
        cwd=tmp_path,
        env={"PATH": "/usr/bin"},
        process_factory=process_factory,
    )

    assert pid == 7654
    assert captured["args"] == ["python", "-m", "gpuops", "worker", "run"]
    assert captured["shell"] is False
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert captured["stderr"] == subprocess.STDOUT
    assert (tmp_path / "worker.log").stat().st_mode & 0o777 == 0o600

    with pytest.raises(ValueError, match="sudo"):
        spawn_detached_worker(
            ["sudo", "gpuops", "worker"],
            stdout_path=tmp_path / "rejected.log",
            process_factory=process_factory,
        )
