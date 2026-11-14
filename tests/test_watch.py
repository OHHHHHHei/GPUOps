from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from gpuops import __version__
from gpuops.control import RemoteTarget
from gpuops.models import JobSpec, JobStatus
from gpuops.ssh import SSHResult
from gpuops.transport import SshWorkerTransport
from gpuops.worker.rpc import MAX_WATCH_JOBS, RPC_API_VERSION, WorkerRpc
from gpuops.worker.runtime import ProcessIdentity, WorkerProcessState, WorkerProcessStatus

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def status(state="running", *, age=0, release=__version__):
    return WorkerProcessStatus(
        WorkerProcessState(state),
        ProcessIdentity(123, 456, "boot") if state == "running" else None,
        {
            "server_id": "gpu01",
            "worker_id": "gpu01:host",
            "release": release,
            "pid": 123,
            "status": "running",
            "started_at": NOW.isoformat(),
            "heartbeat_at": (NOW - timedelta(seconds=age)).isoformat(),
            "iteration": 1,
            "tick_failures": 1,
            "last_tick_error": "previous sampling failed",
        },
    )


def request(run_ids):
    return {
        "api_version": RPC_API_VERSION,
        "release": __version__,
        "method": "watch",
        "params": {"run_ids": run_ids},
        "request_id": "watch-test",
    }


def forbidden(*args, **kwargs):
    pytest.fail("watch must not scan storage, queues or GPU state")


def test_watch_batches_complete_records_and_isolates_item_errors(tmp_path, monkeypatch):
    rpc = WorkerRpc(
        tmp_path, server_id="gpu01", worker_inspector=lambda _: status(), clock=lambda: NOW
    )
    records = {}
    for run_id, job_status in (("queued", JobStatus.QUEUED), ("done", JobStatus.FAILED)):
        record = rpc.queue.enqueue(
            JobSpec(id=run_id, project="watch", command=["true"], working_directory="/work"),
            run_id=run_id,
        )
        if job_status == JobStatus.FAILED:
            record = record.model_copy(
                update={"status": job_status, "exit_code": 7, "finished_at": NOW}
            )
            rpc.queue.jobs_directory.joinpath(f"{run_id}.json").write_text(record.model_dump_json())
        records[run_id] = record.model_dump(mode="json")
    rpc.queue.jobs_directory.joinpath("broken.json").write_text("{bad json")
    monkeypatch.setattr(rpc.storage, "usage", forbidden)
    monkeypatch.setattr(rpc.queue, "list", forbidden)
    monkeypatch.setattr(rpc.queue, "count", forbidden)
    monkeypatch.setattr(rpc.resource_probe, "snapshot", forbidden)

    class Ssh:
        calls = []

        async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
            self.calls.append(json.loads(stdin))
            return SSHResult(host, 0, json.dumps(rpc.handle_json(stdin)), "", 0.01)

    ssh = Ssh()
    payload = asyncio.run(
        SshWorkerTransport(ssh=ssh).watch(
            RemoteTarget("gpu01", "alias01"), ["queued", "done", "missing", "broken", "queued"]
        )
    )
    assert len(ssh.calls) == 1
    assert payload["jobs"]["queued"] == records["queued"]
    assert payload["jobs"]["done"] == records["done"]
    assert payload["jobs"]["missing"] is None
    assert payload["jobs"]["broken"]["error"]["code"] == "job_read_failed"
    assert len(payload["jobs"]) == 4
    assert payload["worker"]["ready"] is True
    assert payload["worker"]["release"] == __version__
    assert payload["worker"]["last_tick_error"] == "previous sampling failed"
    assert payload["worker"]["heartbeat_at"] == NOW.isoformat()


@pytest.mark.parametrize(
    "process,age,release,ready,error_code",
    [
        ("running", 0, __version__, True, None),
        ("running", 120, __version__, False, None),
        ("running", 0, "0.3.0", False, "version_mismatch"),
        ("stopped", 0, __version__, False, None),
        ("stale", 0, __version__, False, None),
        ("invalid", 0, __version__, False, None),
    ],
)
def test_watch_reports_worker_state_without_requiring_readiness(
    tmp_path, process, age, release, ready, error_code
):
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        worker_inspector=lambda _: status(process, age=age, release=release),
        clock=lambda: NOW,
    )
    response = rpc.handle(request(["missing"]))
    assert response["ok"] is True
    assert response["result"]["jobs"] == {"missing": None}
    worker = response["result"]["worker"]
    assert worker["process"] == process
    assert worker["ready"] is ready
    assert worker["error_code"] == error_code
    assert worker["reason"]


@pytest.mark.parametrize("run_ids", [None, "job", ["../escape"], [3], ["a"] * (MAX_WATCH_JOBS + 1)])
def test_watch_rejects_invalid_batches_before_observing(tmp_path, run_ids):
    rpc = WorkerRpc(tmp_path, server_id="gpu01", worker_inspector=forbidden)
    response = rpc.handle(request(run_ids))
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_params"


def test_empty_watch_only_observes_worker(tmp_path, monkeypatch):
    rpc = WorkerRpc(
        tmp_path, server_id="gpu01", worker_inspector=lambda _: status(), clock=lambda: NOW
    )
    monkeypatch.setattr(rpc.queue, "get", forbidden)
    response = rpc.handle(request([]))
    assert response["ok"] is True
    assert response["result"]["jobs"] == {}
    assert response["result"]["worker"]["ready"] is True
