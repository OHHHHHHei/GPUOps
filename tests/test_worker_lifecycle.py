from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call

import pytest

from gpuops.worker import app
from gpuops.worker.runtime import (
    ProcessIdentity,
    WorkerHeartbeat,
    WorkerProcessState,
    WorkerProcessStatus,
)

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
SERVER_ID = "gpu01"
OLD_PID = 4321
NEW_PID = 4322
STOPPED = WorkerProcessStatus(WorkerProcessState.STOPPED, None, None)


def running_worker(pid: int, *, heartbeat_status: str = "running") -> WorkerProcessStatus:
    heartbeat = WorkerHeartbeat(
        server_id=SERVER_ID,
        worker_id=f"{SERVER_ID}:host",
        pid=pid,
        status=heartbeat_status,
        started_at=NOW,
        heartbeat_at=NOW,
        iteration=1,
        tick_failures=0,
    )
    return WorkerProcessStatus(
        WorkerProcessState.RUNNING,
        ProcessIdentity(pid=pid, start_time_ticks=9876, boot_id="boot-a"),
        heartbeat.model_dump(mode="json"),
    )


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> Mock:
    elapsed = 0.0

    def monotonic() -> float:
        nonlocal elapsed
        elapsed += 0.01
        return elapsed

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(app, "time", SimpleNamespace(monotonic=monotonic, sleep=sleep))
    monkeypatch.setattr(app, "utc_now", lambda: NOW)
    processes = Mock()
    processes.stop.return_value = True
    processes.spawn.return_value = NEW_PID
    monkeypatch.setattr(app, "inspect_worker", processes.inspect)
    monkeypatch.setattr(app, "request_worker_stop", processes.stop)
    monkeypatch.setattr(app, "spawn_detached_worker", processes.spawn)
    return processes


def test_start_worker_reuses_ready_worker(tmp_path: Path, processes: Mock) -> None:
    processes.inspect.return_value = running_worker(OLD_PID)

    result = app.start_worker(server_id=SERVER_ID, state_directory=tmp_path)

    assert result == {"started": False, "status": "running", "pid": OLD_PID}
    processes.stop.assert_not_called()
    processes.spawn.assert_not_called()


def test_start_worker_waits_for_unhealthy_worker_to_stop_before_spawning(
    tmp_path: Path, processes: Mock
) -> None:
    unhealthy = running_worker(OLD_PID, heartbeat_status="degraded")
    processes.inspect.side_effect = [unhealthy, unhealthy, STOPPED, running_worker(NEW_PID)]

    result = app.start_worker(server_id=SERVER_ID, state_directory=tmp_path)

    assert result == {"started": True, "status": "running", "pid": NEW_PID}
    processes.stop.assert_called_once()
    processes.spawn.assert_called_once()
    processes.assert_has_calls(
        [call.stop(ANY), call.inspect(ANY), call.inspect(ANY), call.spawn(ANY, stdout_path=ANY)]
    )


def test_start_worker_does_not_spawn_if_unhealthy_worker_never_stops(
    tmp_path: Path, processes: Mock
) -> None:
    processes.inspect.return_value = running_worker(OLD_PID, heartbeat_status="degraded")

    with pytest.raises(RuntimeError, match="cannot replace unhealthy worker: .*degraded"):
        app.start_worker(server_id=SERVER_ID, state_directory=tmp_path, wait_seconds=0.2)

    processes.stop.assert_called_once()
    processes.spawn.assert_not_called()


def test_start_worker_reports_starting_while_new_worker_is_not_ready(
    tmp_path: Path, processes: Mock
) -> None:
    starting = running_worker(NEW_PID, heartbeat_status="starting")
    processes.inspect.side_effect = lambda _paths: starting if processes.spawn.called else STOPPED

    result = app.start_worker(server_id=SERVER_ID, state_directory=tmp_path, wait_seconds=0.2)

    assert result == {"started": True, "status": "starting", "pid": NEW_PID}
    processes.stop.assert_not_called()
    processes.spawn.assert_called_once()
