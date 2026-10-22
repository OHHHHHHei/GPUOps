from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpuops.config import GPU_QUERY_TIMEOUT_SECONDS
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


def completed(command: list[str], stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def test_local_probe_prefers_gpustat_and_maps_resources() -> None:
    calls: list[list[str]] = []
    payload = {
        "gpus": [
            {
                "index": 1,
                "uuid": "GPU-B",
                "memory.total": 24576,
                "memory.used": 1234,
                "utilization.gpu": 87,
                "processes": [{"pid": 99}],
            },
            {
                "index": 0,
                "uuid": "GPU-A",
                "memory.total": 24576,
                "memory.used": 318,
                "utilization.gpu": 0,
                "processes": [],
            },
        ]
    }

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == pytest.approx(GPU_QUERY_TIMEOUT_SECONDS, abs=0.01)
        return completed(command, json.dumps(payload))

    snapshot = LocalGpuProbe("gpu01", runner=runner, clock=lambda: NOW).snapshot()

    assert calls == [["gpustat", "--json"]]
    assert snapshot.healthy is True
    assert [gpu.index for gpu in snapshot.gpus] == [0, 1]
    assert snapshot.gpus[0].idle is True
    assert snapshot.gpus[1].process_count == 1
    assert snapshot.gpus[1].idle is False


def test_local_probe_falls_back_to_nvidia_smi_and_fails_closed() -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        if command[0] == "gpustat":
            raise FileNotFoundError("gpustat missing")
        if "--query-gpu" in command[1]:
            return completed(
                command,
                ("0, GPU-A, 24576, 318, 0\n1, GPU-B, 24576, 12300, 90\n2, GPU-C, 24576, 4096, 0\n"),
            )
        return completed(command, "GPU-B, 8123\n")

    snapshot = LocalGpuProbe("gpu02", runner=runner, clock=lambda: NOW).snapshot()

    assert len(calls) == 3
    assert snapshot.healthy is True
    assert snapshot.gpus[0].idle is True
    assert snapshot.gpus[1].process_count == 1
    assert snapshot.gpus[1].idle is False
    assert snapshot.gpus[2].process_count == 0
    assert snapshot.gpus[2].idle is False

    def failing_runner(command, **kwargs):
        del kwargs
        return completed(command, stderr=f"{command[0]} failed", code=1)

    failed = LocalGpuProbe("gpu02", runner=failing_runner, clock=lambda: NOW).snapshot()
    assert failed.healthy is False
    assert failed.gpus == []
    assert "gpustat" in failed.error
    assert "nvidia-smi" in failed.error


def test_local_probe_shares_one_deadline_across_fallback_commands() -> None:
    current = [0.0]
    timeouts: list[float] = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if command[0] == "gpustat":
            current[0] = 10.0
            return completed(command, stderr="gpustat failed", code=1)
        if "--query-gpu" in command[1]:
            current[0] = 22.0
            return completed(command, "0, GPU-A, 24576, 318, 0\n")
        current[0] = 24.0
        return completed(command, "")

    snapshot = LocalGpuProbe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        monotonic_clock=lambda: current[0],
        sample_timeout_seconds=25.0,
    ).snapshot()

    assert snapshot.healthy is True
    assert timeouts == pytest.approx([25.0, 15.0, 3.0])


def test_local_probe_rejects_over_budget_result_without_starting_fallback() -> None:
    current = [0.0]
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        current[0] = 26.0
        return completed(
            command,
            json.dumps(
                {
                    "gpus": [
                        {
                            "index": 0,
                            "uuid": "GPU-A",
                            "memory.total": 24576,
                            "memory.used": 318,
                            "utilization.gpu": 0,
                            "processes": [],
                        }
                    ]
                }
            ),
        )

    snapshot = LocalGpuProbe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        monotonic_clock=lambda: current[0],
        sample_timeout_seconds=25.0,
    ).snapshot()

    assert calls == [["gpustat", "--json"]]
    assert snapshot.healthy is False
    assert snapshot.gpus == []
    assert "sampling deadline exhausted" in snapshot.error


def test_local_probe_memory_idle_baseline_and_tolerance_are_configurable() -> None:
    payload = {
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-A",
                "memory.total": 24576,
                "memory.used": 350,
                "utilization.gpu": 0,
                "processes": [],
            },
            {
                "index": 1,
                "uuid": "GPU-B",
                "memory.total": 24576,
                "memory.used": 351,
                "utilization.gpu": 0,
                "processes": [],
            },
        ]
    }

    def runner(command, **kwargs):
        del kwargs
        return completed(command, json.dumps(payload))

    snapshot = LocalGpuProbe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        idle_memory_baseline_mib=300,
        idle_memory_tolerance_mib=50,
    ).snapshot()

    assert snapshot.gpus[0].idle is True
    assert snapshot.gpus[1].process_count == 0
    assert snapshot.gpus[1].idle is False


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
            "pid": identity.pid,
            "status": heartbeat_status,
            "started_at": NOW.isoformat(),
            "heartbeat_at": NOW.isoformat(),
            "iteration": 1,
            "tick_failures": 0,
        },
    )

    ready, reason = worker_readiness(status, server_id="gpu01", now=NOW)

    assert ready is False
    assert heartbeat_status in reason


def test_worker_readiness_matches_identity_server_and_age() -> None:
    identity = ProcessIdentity(pid=4321, start_time_ticks=9876, boot_id="boot-a")

    def status(*, pid=4321, server_id="gpu01", heartbeat_at=NOW):
        return WorkerProcessStatus(
            state=WorkerProcessState.RUNNING,
            identity=identity,
            heartbeat={
                "server_id": server_id,
                "worker_id": f"{server_id}:host",
                "pid": pid,
                "status": "running",
                "started_at": NOW.isoformat(),
                "heartbeat_at": heartbeat_at.isoformat(),
                "iteration": 1,
                "tick_failures": 0,
            },
        )

    assert worker_readiness(status(), server_id="gpu01", now=NOW) == (True, "ready")
    assert worker_readiness(status(pid=999), server_id="gpu01", now=NOW)[0] is False
    assert worker_readiness(status(server_id="gpu02"), server_id="gpu01", now=NOW)[0] is False
    assert (
        worker_readiness(
            status(heartbeat_at=NOW - timedelta(seconds=WORKER_HEARTBEAT_MAX_AGE_SECONDS + 1)),
            server_id="gpu01",
            now=NOW,
        )[0]
        is False
    )
    assert (
        worker_readiness(
            status(heartbeat_at=NOW - timedelta(seconds=WORKER_HEARTBEAT_MAX_AGE_SECONDS)),
            server_id="gpu01",
            now=NOW,
        )[0]
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
        sampler_factory=lambda: (events.append("create") or Sampler()),
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
