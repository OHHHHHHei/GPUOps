from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuops.config import SchedulerSettings
from gpuops.core.queue import QueueStore
from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.core.scheduler import Scheduler
from gpuops.core.storage import StorageBudget, StorageQuotaExceeded
from gpuops.models import JobSpec, JobStatus, utc_now
from gpuops.worker import supervisor as supervisor_module
from gpuops.worker import supervisor_records as supervisor_records_module
from gpuops.worker.service import Worker
from gpuops.worker.supervisor import (
    DefinitiveLaunchFailure,
    DurableJobSupervisor,
    DurableSupervisorLauncher,
    IntentConflict,
    InvalidReceipt,
    ProcessIdentity,
    SupervisorPaths,
    SupervisorReceipt,
    UnsafeSignal,
    process_identity_matches,
    signal_process,
)


def write_fake_proc(
    proc_root: Path,
    identity: ProcessIdentity,
    *,
    state: str = "S",
) -> None:
    process_directory = proc_root / str(identity.pid)
    process_directory.mkdir(parents=True)
    boot_path = proc_root / "sys/kernel/random"
    boot_path.mkdir(parents=True, exist_ok=True)
    boot_path.joinpath("boot_id").write_text(f"{identity.boot_id}\n", encoding="utf-8")
    tail = [state, "1", str(identity.pgid), "0"] + ["0"] * 15 + [
        str(identity.start_ticks)
    ] + ["0"] * 4
    process_directory.joinpath("stat").write_text(
        f"{identity.pid} (gpuops supervisor) " + " ".join(tail),
        encoding="utf-8",
    )


def write_supervisor_artifacts(
    paths: SupervisorPaths,
    *,
    supervisor: ProcessIdentity,
    child: ProcessIdentity | None = None,
    token: str = "launch-token",
    status: str = "running",
) -> None:
    supervisor_records_module.atomic_write_json(
        paths.intent,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": paths.directory.name,
            "launch_token": token,
        },
    )
    supervisor_records_module.atomic_write_json(
        paths.receipt,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": paths.directory.name,
            "launch_token": token,
            "status": status,
            "accepted_at": "2026-09-19T00:00:00+00:00",
            "supervisor": {
                "pid": supervisor.pid,
                "pgid": supervisor.pgid,
                "start_ticks": supervisor.start_ticks,
                "boot_id": supervisor.boot_id,
            },
            "child": (
                {
                    "pid": child.pid,
                    "pgid": child.pgid,
                    "start_ticks": child.start_ticks,
                    "boot_id": child.boot_id,
                }
                if child is not None
                else None
            ),
            "child_started_at": (
                "2026-09-19T00:00:01+00:00" if child is not None else None
            ),
        },
    )


def test_launch_writes_receipt_result_and_merged_console(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path)
    receipt = supervisor.launch(
        "basic-run",
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
        ],
        cwd=tmp_path,
    )
    assert receipt.run_id == "basic-run"
    assert receipt.status in {"running", "finished"}
    assert receipt.supervisor.pid > 1
    assert receipt.supervisor.pgid == receipt.supervisor.pid

    result = supervisor.wait("basic-run", timeout_seconds=5)
    assert result.exit_code == 0
    assert result.launch_token == receipt.launch_token
    paths = SupervisorPaths.create(tmp_path, "basic-run")
    assert "stdout-line" in paths.console.read_text(encoding="utf-8")
    assert "stderr-line" in paths.console.read_text(encoding="utf-8")
    assert json.loads(paths.intent.read_text())["launch_token"] == receipt.launch_token
    assert json.loads(paths.result.read_text())["exit_code"] == 0


@pytest.mark.parametrize("identities,expected", [
    ({4: "GPU-four", 5: "GPU-five"}, "GPU-four,GPU-five"), ({}, "4,5"),
])
def test_launcher_binds_confirmed_physical_gpus_and_reads_legacy_leases(
    tmp_path: Path, identities: dict[int, str], expected: str,
) -> None:
    store = QueueStore(tmp_path / "queue")
    record = store.enqueue(JobSpec(
        id="cuda-binding", project="tests", working_directory=str(tmp_path),
        command=[sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        resources={"gpu_count": 2}, runtime={"environment": {"CUDA_VISIBLE_DEVICES": "0,1"}},
    ))
    record = store.lease(
        record.run_id, server_id="gpu01", gpu_indices=[5, 4], gpu_uuids=identities,
        worker_id="worker", ttl=timedelta(seconds=30), expected_revision=0,
    )
    supervisor = DurableJobSupervisor(tmp_path / "supervisor")
    DurableSupervisorLauncher(supervisor).start(record)
    result = supervisor.wait(record.run_id, timeout_seconds=5)
    assert result.exit_code == 0
    paths = SupervisorPaths.create(tmp_path / "supervisor", record.run_id)
    assert paths.console.read_text().strip() == expected


def test_flushed_console_output_is_visible_before_child_exit(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    code = (
        "from pathlib import Path; import sys, time; "
        "sys.stdout.write('streamed-before-exit\\n'); sys.stdout.flush(); "
        f"Path({str(ready)!r}).write_text('ready'); "
        f"release=Path({str(release)!r}); "
        "deadline=time.monotonic()+5; "
        "exec(\"while not release.exists() and time.monotonic() < deadline:\\n"
        "    time.sleep(0.01)\")"
    )
    supervisor = DurableJobSupervisor(tmp_path)
    supervisor.launch("streaming-run", [sys.executable, "-c", code], cwd=tmp_path)
    paths = SupervisorPaths.create(tmp_path, "streaming-run")

    try:
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()

        deadline = time.monotonic() + 1
        output = ""
        while time.monotonic() < deadline:
            if paths.console.exists():
                output = paths.console.read_text(encoding="utf-8")
            if "streamed-before-exit" in output:
                break
            time.sleep(0.01)
        assert "streamed-before-exit" in output
        assert supervisor.poll("streaming-run") is None
    finally:
        release.write_text("release", encoding="utf-8")
        supervisor.wait("streaming-run", timeout_seconds=5)


@pytest.mark.skipif(
    not Path("/proc/sys/kernel/random/boot_id").exists(),
    reason="real zombie-reaping lifecycle requires Linux procfs",
)
def test_worker_completes_final_job_without_a_follow_up_launch(tmp_path: Path) -> None:
    class StableProbe:
        def snapshot(self) -> ServerResourceSnapshot:
            return ServerResourceSnapshot(
                server_id="test-server",
                sampled_at=utc_now(),
                healthy=True,
                gpus=[
                    GpuResource(
                        index=0,
                        uuid="GPU-test",
                        memory_total_mib=1024,
                        memory_used_mib=0,
                        utilization_percent=0,
                        process_count=0,
                        idle=True,
                    )
                ],
            )

    store = QueueStore(tmp_path / "queue")
    queued = store.enqueue(
        JobSpec(
            id="only-job",
            project="tests",
            command=[sys.executable, "-c", "pass"],
            working_directory=str(tmp_path),
            resources={"host": "test-server", "gpu_count": 1, "memory_mib": 0},
        ),
        run_id="only-job",
    )
    scheduler = Scheduler(
        store,
        StableProbe(),
        server_id="test-server",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
    )
    supervisor = DurableJobSupervisor(tmp_path / "runs")
    worker = Worker(store, scheduler, DurableSupervisorLauncher(supervisor))

    assert worker.tick().started == [queued.run_id]
    receipt = supervisor.reconcile(queued.run_id)
    assert receipt is not None
    deadline = time.monotonic() + 5
    while store.get(queued.run_id).status == JobStatus.RUNNING:
        worker.tick()
        if time.monotonic() >= deadline:
            pytest.fail("the final supervisor was not reaped without another launch")
        time.sleep(0.01)

    assert store.get(queued.run_id).status == JobStatus.SUCCEEDED
    assert not Path("/proc", str(receipt.supervisor.pid)).exists()


def test_duplicate_launch_reconciles_receipt_without_rerunning(tmp_path: Path) -> None:
    marker = tmp_path / "launch-count.txt"
    code = (
        "from pathlib import Path; import time; "
        f"p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        "time.sleep(0.2)"
    )
    supervisor = DurableJobSupervisor(tmp_path)
    first = supervisor.launch("one-shot", [sys.executable, "-c", code], cwd=tmp_path)
    second = supervisor.launch("one-shot", [sys.executable, "-c", code], cwd=tmp_path)
    assert first.launch_token == second.launch_token
    assert supervisor.reconcile("one-shot").launch_token == first.launch_token
    supervisor.wait("one-shot", timeout_seconds=5)
    assert marker.read_text(encoding="utf-8") == "x"


def test_lock_waiter_learns_launch_token_from_persisted_intent(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path, receipt_timeout_seconds=1)
    paths = SupervisorPaths.create(tmp_path, "concurrent-run")
    lock_fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = "token-written-by-lock-owner"

    def publish_receipt() -> None:
        time.sleep(0.05)
        requested = supervisor_records_module.requested_intent(
            "concurrent-run",
            [sys.executable, "-c", "pass"],
            tmp_path,
            None,
            paths.console,
            4 * 1024 * 1024,
            2,
        )
        supervisor_records_module.atomic_write_json(
            paths.intent,
            {
                **requested,
                "schema": "gpuops-supervisor/v1",
                "launch_token": token,
                "created_at": "2026-09-19T00:00:00+00:00",
            },
        )
        identity = ProcessIdentity(pid=999, pgid=999, start_ticks=1, boot_id="boot")
        supervisor_records_module.atomic_write_json(
            paths.receipt,
            {
                "schema": "gpuops-supervisor/v1",
                "run_id": "concurrent-run",
                "launch_token": token,
                "status": "running",
                "accepted_at": "2026-09-19T00:00:00+00:00",
                "supervisor": {
                    "pid": identity.pid,
                    "pgid": identity.pgid,
                    "start_ticks": identity.start_ticks,
                    "boot_id": identity.boot_id,
                },
                "child": None,
                "child_started_at": None,
            },
        )

    publisher = threading.Thread(target=publish_receipt)
    publisher.start()
    try:
        receipt = supervisor.launch("concurrent-run", [sys.executable, "-c", "pass"], cwd=tmp_path)
    finally:
        publisher.join()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert receipt.launch_token == token


def test_reusing_run_id_with_different_command_is_rejected(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path)
    supervisor.launch("stable-intent", [sys.executable, "-c", "pass"], cwd=tmp_path)
    supervisor.wait("stable-intent", timeout_seconds=5)
    with pytest.raises(IntentConflict):
        supervisor.launch(
            "stable-intent", [sys.executable, "-c", "print('different')"], cwd=tmp_path
        )


def test_nonzero_exit_code_is_durable(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path)
    supervisor.launch("failed-run", [sys.executable, "-c", "raise SystemExit(23)"], cwd=tmp_path)
    result = supervisor.wait("failed-run", timeout_seconds=5)
    assert result.exit_code == 23
    assert supervisor.poll("failed-run") == result


def test_pre_refactor_persisted_records_reconcile_without_schema_drift(
    tmp_path: Path,
) -> None:
    paths = SupervisorPaths.create(tmp_path, "persisted-fixture")
    paths.intent.write_text(
        json.dumps(
            {
                "schema": "gpuops-supervisor/v1",
                "run_id": "persisted-fixture",
                "argv": ["python", "experiment.py"],
                "cwd": "/srv/experiment",
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "console_path": str(paths.console),
                "console_max_bytes": 4194304,
                "console_backups": 2,
                "reservation_path": None,
                "launch_token": "persisted-token",
                "created_at": "2026-09-19T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths.receipt.write_text(
        json.dumps(
            {
                "schema": "gpuops-supervisor/v1",
                "run_id": "persisted-fixture",
                "launch_token": "persisted-token",
                "status": "finished",
                "accepted_at": "2026-09-19T00:00:00+00:00",
                "supervisor": {
                    "pid": 4100,
                    "pgid": 4100,
                    "start_ticks": None,
                    "boot_id": None,
                },
                "child": {
                    "pid": 4200,
                    "pgid": 4200,
                    "start_ticks": 123456,
                    "boot_id": "boot-a",
                },
                "child_started_at": "2026-09-19T00:00:01+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths.result.write_text(
        json.dumps(
            {
                "schema": "gpuops-supervisor/v1",
                "run_id": "persisted-fixture",
                "launch_token": "persisted-token",
                "exit_code": 7,
                "started_at": "2026-09-19T00:00:01+00:00",
                "finished_at": "2026-09-19T00:00:02+00:00",
                "launch_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    supervisor = DurableJobSupervisor(tmp_path)
    receipt = supervisor.reconcile("persisted-fixture")
    result = supervisor.poll("persisted-fixture")

    assert receipt is not None
    assert receipt.supervisor == ProcessIdentity(
        pid=4100,
        pgid=4100,
        start_ticks=None,
        boot_id=None,
    )
    assert receipt.child == ProcessIdentity(
        pid=4200,
        pgid=4200,
        start_ticks=123456,
        boot_id="boot-a",
    )
    assert result is not None and result.exit_code == 7
    assert set(receipt.as_json_dict()) == {
        "schema",
        "run_id",
        "launch_token",
        "status",
        "accepted_at",
        "supervisor",
        "child",
        "child_started_at",
    }
    assert set(receipt.as_json_dict()["supervisor"]) == {
        "pid",
        "pgid",
        "start_ticks",
        "boot_id",
    }
    assert set(result.as_json_dict()) == {
        "schema",
        "run_id",
        "launch_token",
        "exit_code",
        "started_at",
        "finished_at",
        "launch_error",
    }


def test_wait_reaps_owned_supervisor_after_result_and_lock_release(tmp_path: Path) -> None:
    class PollSequence:
        def __init__(self, *values: int | None) -> None:
            self.values = iter(values)
            self.calls = 0

        def poll(self) -> int | None:
            self.calls += 1
            return next(self.values)

    supervisor = DurableJobSupervisor(tmp_path)
    paths = SupervisorPaths.create(tmp_path, "wait-reap")
    identity = ProcessIdentity(pid=39001, pgid=39001, start_ticks=1, boot_id="boot-A")
    write_supervisor_artifacts(
        paths,
        supervisor=identity,
        token="wait-token",
        status="finished",
    )
    supervisor_records_module.atomic_write_json(
        paths.result,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": "wait-reap",
            "launch_token": "wait-token",
            "exit_code": 0,
            "started_at": "2026-09-19T00:00:01+00:00",
            "finished_at": "2026-09-19T00:00:02+00:00",
            "launch_error": None,
        },
    )
    process = PollSequence(None, 0)
    supervisor._owned_supervisor_processes["wait-reap"] = process  # type: ignore[assignment]

    assert supervisor.wait("wait-reap", timeout_seconds=1).exit_code == 0
    assert process.calls == 2
    assert "wait-reap" not in supervisor._owned_supervisor_processes


def test_console_log_is_rolled_with_a_hard_size_bound(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path)
    supervisor.launch(
        "large-console",
        [sys.executable, "-c", "import sys; sys.stdout.write('A'*300+'B'*300)"],
        cwd=tmp_path,
        console_max_bytes=128,
        console_backups=1,
    )
    supervisor.wait("large-console", timeout_seconds=5)
    paths = SupervisorPaths.create(tmp_path, "large-console")
    logs = sorted(paths.directory.glob("console.log*"))
    assert {path.name for path in logs} == {"console.log", "console.log.1"}
    assert all(path.stat().st_size <= 128 for path in logs)
    assert sum(path.stat().st_size for path in logs) <= 256
    assert "B" in paths.console.read_text(encoding="utf-8")


def test_supervisor_reserves_global_log_budget_and_releases_it(tmp_path: Path) -> None:
    state = tmp_path / "state"
    storage = StorageBudget(
        state,
        total_budget_bytes=2048,
        log_budget_bytes=1024,
        high_watermark=0.9,
        target_watermark=0.7,
    )
    supervisor = DurableJobSupervisor(state / "runs", storage=storage)
    receipt = supervisor.launch(
        "budgeted-run",
        [sys.executable, "-c", "print('bounded')"],
        cwd=tmp_path,
        console_max_bytes=128,
        console_backups=1,
    )
    reservation = storage.reservation_directory / "budgeted-run.reserve"
    console = storage.logs_directory / "budgeted-run" / "console.log"

    assert receipt.launch_token
    result = supervisor.wait("budgeted-run", timeout_seconds=5)
    assert result.exit_code == 0
    assert console.read_text(encoding="utf-8").strip() == "bounded"
    assert not reservation.exists()
    assert storage.usage().total_bytes <= storage.total_budget_bytes


def test_supervisor_refuses_launch_when_log_reservation_exceeds_budget(tmp_path: Path) -> None:
    state = tmp_path / "state"
    storage = StorageBudget(
        state,
        total_budget_bytes=512,
        log_budget_bytes=256,
        high_watermark=0.9,
        target_watermark=0.7,
    )
    supervisor = DurableJobSupervisor(state / "runs", storage=storage)

    with pytest.raises(StorageQuotaExceeded):
        supervisor.launch(
            "too-large",
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            console_max_bytes=200,
            console_backups=1,
        )


def test_definitive_supervisor_spawn_failure_releases_intent_and_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    storage = StorageBudget(
        state,
        total_budget_bytes=4096,
        log_budget_bytes=2048,
    )
    supervisor = DurableJobSupervisor(state / "runs", storage=storage)

    def fail_spawn(*args, **kwargs):
        del args, kwargs
        raise OSError("cannot spawn")

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fail_spawn)

    with pytest.raises(DefinitiveLaunchFailure, match="cannot spawn"):
        supervisor.launch(
            "spawn-failure",
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            console_max_bytes=128,
            console_backups=1,
        )

    paths = SupervisorPaths.create(state / "runs", "spawn-failure")
    assert not paths.intent.exists()
    assert not paths.receipt.exists()
    assert not (storage.reservation_directory / "spawn-failure.reserve").exists()


def test_intent_commit_error_cleans_committed_intent_and_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    storage = StorageBudget(
        state,
        total_budget_bytes=4096,
        log_budget_bytes=2048,
    )
    supervisor = DurableJobSupervisor(state / "runs", storage=storage)
    original_write = supervisor_records_module.atomic_write_json

    def fail_after_intent_commit(path, value):
        original_write(path, value)
        if path.name == "intent.json":
            raise OSError("directory sync failed")

    monkeypatch.setattr(
        supervisor_records_module,
        "atomic_write_json",
        fail_after_intent_commit,
    )

    with pytest.raises(OSError, match="directory sync failed"):
        supervisor.launch(
            "intent-failure",
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            console_max_bytes=128,
            console_backups=1,
        )

    paths = SupervisorPaths.create(state / "runs", "intent-failure")
    assert not paths.intent.exists()
    assert not (storage.reservation_directory / "intent-failure.reserve").exists()


def test_process_identity_and_signal_require_all_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    pid = 43210
    pgid = pid
    process_directory = proc / str(pid)
    process_directory.mkdir(parents=True)
    boot_path = proc / "sys/kernel/random"
    boot_path.mkdir(parents=True)
    (boot_path / "boot_id").write_text("boot-A\n", encoding="utf-8")
    # Fields after comm begin at field 3. starttime is field 22 (index 19 here).
    tail = ["S", "1", str(pgid), "0"] + ["0"] * 15 + ["987654"] + ["0"] * 4
    (process_directory / "stat").write_text(
        f"{pid} (test worker) " + " ".join(tail), encoding="utf-8"
    )
    identity = ProcessIdentity(pid=pid, pgid=pgid, start_ticks=987654, boot_id="boot-A")
    assert process_identity_matches(identity, proc_root=proc)

    calls = []
    monkeypatch.setattr(os, "killpg", lambda group, sig: calls.append((group, sig)))
    signal_process(identity, signal.SIGTERM, proc_root=proc)
    assert calls == [(pgid, signal.SIGTERM)]

    stale = ProcessIdentity(pid=pid, pgid=pgid, start_ticks=987655, boot_id="boot-A")
    with pytest.raises(UnsafeSignal):
        signal_process(stale, signal.SIGTERM, proc_root=proc)


def test_zombie_leader_waits_for_live_process_group_members(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    leader = ProcessIdentity(pid=40101, pgid=40101, start_ticks=10, boot_id="boot-A")
    member = ProcessIdentity(pid=40102, pgid=40101, start_ticks=11, boot_id="boot-A")
    write_fake_proc(proc, leader, state="Z")
    write_fake_proc(proc, member)
    receipt = SupervisorReceipt(
        run_id="zombie-group",
        launch_token="zombie-token",
        status="finished",
        accepted_at="2026-09-19T00:00:00+00:00",
        supervisor=leader,
    )
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.process_groups_gone(receipt) is False

    member_directory = proc / str(member.pid)
    member_directory.joinpath("stat").unlink()
    member_directory.rmdir()
    assert supervisor.process_groups_gone(receipt) is True


def test_process_groups_gone_waits_until_owned_supervisor_is_reaped(tmp_path: Path) -> None:
    class PollSequence:
        def __init__(self, *values: int | None) -> None:
            self.values = iter(values)

        def poll(self) -> int | None:
            return next(self.values)

    proc = tmp_path / "proc"
    identity = ProcessIdentity(pid=40201, pgid=40201, start_ticks=20, boot_id="boot-A")
    write_fake_proc(proc, identity, state="Z")
    receipt = SupervisorReceipt(
        run_id="reap-race",
        launch_token="reap-token",
        status="finished",
        accepted_at="2026-09-19T00:00:00+00:00",
        supervisor=identity,
    )
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)
    supervisor._owned_supervisor_processes["reap-race"] = PollSequence(  # type: ignore[assignment]
        None,
        None,
        0,
    )

    assert supervisor.process_groups_gone(receipt) is False
    assert "reap-race" in supervisor._owned_supervisor_processes
    assert supervisor.process_groups_gone(receipt) is True
    assert "reap-race" not in supervisor._owned_supervisor_processes


def test_running_process_can_be_signalled_only_when_procfs_is_available(tmp_path: Path) -> None:
    if not Path("/proc/sys/kernel/random/boot_id").exists():
        pytest.skip("safe signalling intentionally requires Linux procfs credentials")
    supervisor = DurableJobSupervisor(tmp_path)
    supervisor.launch(
        "signal-run",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
    )
    supervisor.signal("signal-run", signal.SIGTERM)
    result = supervisor.wait("signal-run", timeout_seconds=5)
    assert result.exit_code < 0


def test_cancel_without_intent_or_receipt_is_already_complete(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "proc")

    assert supervisor.request_cancel("never-launched") is True


def test_cancel_without_artifacts_stays_pending_while_launch_lock_is_owned(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    supervisor = DurableJobSupervisor(root, proc_root=tmp_path / "proc")
    paths = SupervisorPaths.create(root, "launch-race")
    lock_fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert supervisor.request_cancel("launch-race") is False
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_cancel_with_intent_but_no_receipt_stays_pending(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "proc")
    paths = SupervisorPaths.create(tmp_path / "runs", "launching")
    supervisor_records_module.atomic_write_json(
        paths.intent,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": "launching",
            "launch_token": "pending-token",
        },
    )

    assert supervisor.request_cancel(
        "launching", expected_launch_token="pending-token"
    ) is False


def test_cancel_is_complete_once_matching_durable_result_exists(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "proc")
    paths = SupervisorPaths.create(tmp_path / "runs", "finished")
    identity = ProcessIdentity(
        pid=40501,
        pgid=40501,
        start_ticks=10,
        boot_id="boot-A",
    )
    write_supervisor_artifacts(
        paths,
        supervisor=identity,
        token="finished-token",
        status="finished",
    )
    supervisor_records_module.atomic_write_json(
        paths.result,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": "finished",
            "launch_token": "finished-token",
            "exit_code": -signal.SIGTERM,
            "started_at": "2026-09-19T00:00:01+00:00",
            "finished_at": "2026-09-19T00:00:02+00:00",
            "launch_error": None,
        },
    )

    assert supervisor.request_cancel(
        "finished", expected_launch_token="finished-token"
    ) is True


def test_result_with_surviving_child_group_stays_active_and_is_escalated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    missing_supervisor = ProcessIdentity(
        pid=40801,
        pgid=40801,
        start_ticks=11,
        boot_id="boot-A",
    )
    missing_child = ProcessIdentity(
        pid=40802,
        pgid=40802,
        start_ticks=12,
        boot_id="boot-A",
    )
    descendant = ProcessIdentity(
        pid=40803,
        pgid=40802,
        start_ticks=13,
        boot_id="boot-A",
    )
    write_fake_proc(proc, descendant)
    paths = SupervisorPaths.create(tmp_path / "runs", "result-with-descendant")
    write_supervisor_artifacts(
        paths,
        supervisor=missing_supervisor,
        child=missing_child,
        token="result-token",
        status="finished",
    )
    supervisor_records_module.atomic_write_json(
        paths.result,
        {
            "schema": "gpuops-supervisor/v1",
            "run_id": "result-with-descendant",
            "launch_token": "result-token",
            "exit_code": 0,
            "started_at": "2026-09-19T00:00:01+00:00",
            "finished_at": "2026-09-19T00:00:02+00:00",
            "launch_error": None,
        },
    )
    current = [datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)]
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(
        tmp_path / "runs",
        proc_root=proc,
        cancel_grace_seconds=5,
        clock=lambda: current[0],
    )
    launcher = DurableSupervisorLauncher(supervisor)
    job = SimpleNamespace(
        run_id="result-with-descendant",
        launch=SimpleNamespace(external_id="result-token"),
    )

    assert launcher.poll(job) is None
    assert supervisor.request_cancel("result-with-descendant") is False
    assert calls == [(missing_child.pgid, signal.SIGTERM)]

    current[0] += timedelta(seconds=6)
    assert supervisor.request_cancel("result-with-descendant") is False
    assert calls[-1] == (missing_child.pgid, signal.SIGKILL)
    assert launcher.poll(job) is None

    descendant_directory = proc / str(descendant.pid)
    descendant_directory.joinpath("stat").unlink()
    descendant_directory.rmdir()
    assert supervisor.request_cancel("result-with-descendant") is True
    assert launcher.poll(job) == 0


def test_cancel_signals_only_validated_supervisor_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    identity = ProcessIdentity(pid=41001, pgid=41001, start_ticks=77, boot_id="boot-A")
    write_fake_proc(proc, identity)
    paths = SupervisorPaths.create(tmp_path / "runs", "managed")
    write_supervisor_artifacts(paths, supervisor=identity)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.request_cancel(
        "managed", expected_launch_token="launch-token"
    ) is False
    assert calls == [(identity.pgid, signal.SIGTERM)]


def test_cancel_terms_descendants_after_child_leader_becomes_zombie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    supervisor_identity = ProcessIdentity(
        pid=41501,
        pgid=41501,
        start_ticks=80,
        boot_id="boot-A",
    )
    child_identity = ProcessIdentity(
        pid=41502,
        pgid=41502,
        start_ticks=81,
        boot_id="boot-A",
    )
    descendant = ProcessIdentity(
        pid=41503,
        pgid=41502,
        start_ticks=82,
        boot_id="boot-A",
    )
    write_fake_proc(proc, supervisor_identity)
    write_fake_proc(proc, child_identity, state="Z")
    write_fake_proc(proc, descendant)
    paths = SupervisorPaths.create(tmp_path / "runs", "zombie-child")
    write_supervisor_artifacts(
        paths,
        supervisor=supervisor_identity,
        child=child_identity,
    )
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.request_cancel("zombie-child") is False
    assert calls == [
        (supervisor_identity.pgid, signal.SIGTERM),
        (child_identity.pgid, signal.SIGTERM),
    ]


def test_cancel_escalates_after_persisted_grace_and_waits_for_group_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    proc = tmp_path / "proc"
    storage = StorageBudget(
        state,
        total_budget_bytes=4096,
        log_budget_bytes=2048,
    )
    storage.reserve_log_capacity("stubborn", 128)
    supervisor_identity = ProcessIdentity(
        pid=46001,
        pgid=46001,
        start_ticks=21,
        boot_id="boot-A",
    )
    child_identity = ProcessIdentity(
        pid=46002,
        pgid=46002,
        start_ticks=22,
        boot_id="boot-A",
    )
    write_fake_proc(proc, supervisor_identity)
    write_fake_proc(proc, child_identity)
    paths = SupervisorPaths.create(state / "runs", "stubborn")
    write_supervisor_artifacts(
        paths,
        supervisor=supervisor_identity,
        child=child_identity,
    )
    current = [datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)]
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    supervisor = DurableJobSupervisor(
        state / "runs",
        storage=storage,
        proc_root=proc,
        cancel_grace_seconds=5,
        clock=lambda: current[0],
    )
    assert supervisor.request_cancel("stubborn") is False
    assert calls == [(supervisor_identity.pgid, signal.SIGTERM)]
    assert (storage.reservation_directory / "stubborn.reserve").exists()

    current[0] += timedelta(seconds=6)
    restarted = DurableJobSupervisor(
        state / "runs",
        storage=storage,
        proc_root=proc,
        cancel_grace_seconds=999,
        clock=lambda: current[0],
    )
    assert restarted.request_cancel("stubborn") is False
    assert calls == [
        (supervisor_identity.pgid, signal.SIGTERM),
        (child_identity.pgid, signal.SIGKILL),
        (supervisor_identity.pgid, signal.SIGKILL),
    ]
    assert (storage.reservation_directory / "stubborn.reserve").exists()

    for identity in (supervisor_identity, child_identity):
        process_directory = proc / str(identity.pid)
        process_directory.joinpath("stat").unlink()
        process_directory.rmdir()
    assert restarted.request_cancel("stubborn") is True
    assert not (storage.reservation_directory / "stubborn.reserve").exists()


def test_cancel_never_sigkills_supervisor_before_child_identity_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    identity = ProcessIdentity(pid=46501, pgid=46501, start_ticks=25, boot_id="boot-A")
    write_fake_proc(proc, identity)
    paths = SupervisorPaths.create(tmp_path / "runs", "accepted")
    write_supervisor_artifacts(paths, supervisor=identity, status="accepted")
    current = [datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)]
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(
        tmp_path / "runs",
        proc_root=proc,
        cancel_grace_seconds=5,
        clock=lambda: current[0],
    )

    assert supervisor.request_cancel("accepted") is False
    current[0] += timedelta(seconds=6)
    assert supervisor.request_cancel("accepted") is False
    assert calls == [(identity.pgid, signal.SIGTERM)]


def test_cancel_does_not_release_while_original_process_group_has_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    proc = tmp_path / "proc"
    storage = StorageBudget(
        state,
        total_budget_bytes=4096,
        log_budget_bytes=2048,
    )
    storage.reserve_log_capacity("orphan-group", 128)
    missing_leader = ProcessIdentity(
        pid=47001,
        pgid=47001,
        start_ticks=30,
        boot_id="boot-A",
    )
    group_member = ProcessIdentity(
        pid=47002,
        pgid=47001,
        start_ticks=31,
        boot_id="boot-A",
    )
    write_fake_proc(proc, group_member)
    paths = SupervisorPaths.create(state / "runs", "orphan-group")
    write_supervisor_artifacts(paths, supervisor=missing_leader)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(
        state / "runs",
        storage=storage,
        proc_root=proc,
        cancel_grace_seconds=0,
    )

    assert supervisor.request_cancel("orphan-group") is False
    assert calls == []
    assert (storage.reservation_directory / "orphan-group.reserve").exists()


def test_cancel_falls_back_to_validated_child_when_supervisor_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    missing_supervisor = ProcessIdentity(
        pid=42001,
        pgid=42001,
        start_ticks=12,
        boot_id="boot-A",
    )
    child = ProcessIdentity(pid=42002, pgid=42002, start_ticks=13, boot_id="boot-A")
    write_fake_proc(proc, child)
    paths = SupervisorPaths.create(tmp_path / "runs", "orphan-child")
    write_supervisor_artifacts(paths, supervisor=missing_supervisor, child=child)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.request_cancel("orphan-child") is False
    assert calls == [(child.pgid, signal.SIGTERM)]


def test_cancel_completes_when_managed_supervisor_and_child_are_both_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_supervisor = ProcessIdentity(
        pid=43001,
        pgid=43001,
        start_ticks=12,
        boot_id="boot-A",
    )
    missing_child = ProcessIdentity(
        pid=43002,
        pgid=43002,
        start_ticks=13,
        boot_id="boot-A",
    )
    paths = SupervisorPaths.create(tmp_path / "runs", "all-gone")
    write_supervisor_artifacts(
        paths,
        supervisor=missing_supervisor,
        child=missing_child,
    )
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "proc")

    assert supervisor.request_cancel("all-gone") is True
    assert calls == []


def test_cancel_never_signals_incomplete_receipt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    (proc / "44001").mkdir(parents=True)
    incomplete = ProcessIdentity(pid=44001, pgid=44001, start_ticks=None, boot_id=None)
    paths = SupervisorPaths.create(tmp_path / "runs", "incomplete")
    write_supervisor_artifacts(paths, supervisor=incomplete)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.request_cancel("incomplete") is False
    assert calls == []


def test_cancel_rejects_launch_token_mismatch_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = ProcessIdentity(pid=45001, pgid=45001, start_ticks=1, boot_id="boot-A")
    paths = SupervisorPaths.create(tmp_path / "runs", "wrong-token")
    write_supervisor_artifacts(paths, supervisor=identity)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "proc")

    with pytest.raises(InvalidReceipt, match="launch token"):
        supervisor.request_cancel("wrong-token", expected_launch_token="other-token")
    assert calls == []


def test_cancel_never_signals_a_reused_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    stored = ProcessIdentity(pid=45501, pgid=45501, start_ticks=1, boot_id="boot-A")
    replacement = ProcessIdentity(
        pid=45501,
        pgid=45501,
        start_ticks=2,
        boot_id="boot-A",
    )
    write_fake_proc(proc, replacement)
    paths = SupervisorPaths.create(tmp_path / "runs", "reused-pgid")
    write_supervisor_artifacts(paths, supervisor=stored)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    supervisor = DurableJobSupervisor(tmp_path / "runs", proc_root=proc)

    assert supervisor.request_cancel("reused-pgid") is True
    assert calls == []


def test_short_command_does_not_use_shell_expansion(tmp_path: Path) -> None:
    supervisor = DurableJobSupervisor(tmp_path)
    literal = tmp_path / "literal.txt"
    supervisor.launch(
        "argv-only",
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'w').write(sys.argv[2])",
            str(literal),
            "$HOME;*",
        ],
        cwd=tmp_path,
    )
    supervisor.wait("argv-only", timeout_seconds=5)
    assert literal.read_text(encoding="utf-8") == "$HOME;*"
