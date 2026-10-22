from __future__ import annotations

import asyncio
import json
import sys

import pytest
from conftest import TEST_TARGETS

from gpuops.cli import main
from gpuops.models import JobRecord, JobSpec
from gpuops.monitor import ServerResult
from gpuops.ssh import SSHResult
from gpuops.status import collect_worker_health
from gpuops.transport import SshWorkerTransport


@pytest.mark.parametrize(
    ("gpu_offline", "worker_offline", "worker_ready", "exit_code"),
    [
        (False, False, True, 0),
        (False, True, False, 0),
        (False, False, False, 0),
        (True, False, True, 1),
    ],
)
def test_status_keeps_gpu_and_worker_failures_independent(
    monkeypatch, capsys, gpu_offline, worker_offline, worker_ready, exit_code
):
    results = [
        ServerResult(
            target,
            online=(target.id != "gpu02" or not gpu_offline),
            data={"gpus": []},
            error="monitor offline" if gpu_offline else None,
        )
        for target in TEST_TARGETS
    ]
    calls = []

    class Monitor:
        def __init__(self, targets, *, timeout_seconds):
            assert targets == TEST_TARGETS

        async def collect_all(self):
            return results

    class Transport:
        def __init__(self, *, timeout_seconds):
            assert timeout_seconds == 35.0

        async def worker_command(self, target, action):
            assert action == "status"
            calls.append(target.server_id)
            if target.server_id == "gpu02" and worker_offline:
                raise OSError("worker status offline")
            ready = target.server_id != "gpu02" or worker_ready
            return {
                "ready": ready,
                "reason": "running" if ready else "stopped",
                "heartbeat": {"server_id": target.server_id},
            }

    monkeypatch.setattr("gpuops.cli.MonitorService", Monitor)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["status", "--json"]) == exit_code
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert calls == ["gpu01", "gpu02"]
    for row, original in zip(payload, results, strict=True):
        assert {key: row[key] for key in original.as_dict()} == original.as_dict()
    assert payload[0]["worker"]["ready"] is True
    assert payload[1]["online"] is not gpu_offline
    assert payload[1]["worker"]["reachable"] is not worker_offline
    assert payload[1]["worker"]["ready"] is worker_ready
    if worker_offline:
        assert payload[1]["worker"]["error"] == "worker status offline"
    elif not worker_ready:
        assert payload[1]["worker"]["reason"] == "stopped"


def test_status_collects_monitor_and_both_workers_concurrently(monkeypatch, capsys):
    monitor_started = asyncio.Event()
    both_workers_started = asyncio.Event()
    worker_calls = []

    class Monitor:
        def __init__(self, targets, *, timeout_seconds):
            self.targets = targets

        async def collect_all(self):
            monitor_started.set()
            await asyncio.wait_for(both_workers_started.wait(), timeout=1)
            return [ServerResult(target, online=True, data={"gpus": []}) for target in self.targets]

    class Transport:
        def __init__(self, *, timeout_seconds):
            pass

        async def worker_command(self, target, action):
            assert action == "status"
            await asyncio.wait_for(monitor_started.wait(), timeout=1)
            worker_calls.append(target.server_id)
            if len(worker_calls) == 2:
                both_workers_started.set()
            await asyncio.wait_for(both_workers_started.wait(), timeout=1)
            return {"ready": True, "heartbeat": {"server_id": target.server_id}}

    monkeypatch.setattr("gpuops.cli.MonitorService", Monitor)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["status", "--json"]) == 0
    assert all(row["worker"]["ready"] for row in json.loads(capsys.readouterr().out))


def test_status_text_shows_worker_reason_on_selected_server(monkeypatch, capsys):
    class Monitor:
        def __init__(self, targets, *, timeout_seconds):
            self.targets = targets

        async def collect_all(self):
            return [ServerResult(target, online=True, data={"gpus": []}) for target in self.targets]

    class Transport:
        def __init__(self, *, timeout_seconds):
            pass

        async def worker_command(self, target, action):
            assert action == "status"
            assert target.server_id == "gpu01"
            return {
                "ready": False,
                "reason": "heartbeat expired",
            }

    monkeypatch.setattr("gpuops.cli.MonitorService", Monitor)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["status", "--server", "gpu01"]) == 0
    text = capsys.readouterr().out
    assert "gpu01 worker: unavailable (heartbeat expired)" in text
    assert "gpu02" not in text


@pytest.mark.parametrize("heartbeat", [{"server_id": "gpu02"}, None])
def test_status_does_not_mark_a_different_or_missing_server_heartbeat_ready(heartbeat):
    class Transport:
        async def worker_command(self, target, action):
            return {"ready": True, "reason": "ready", "heartbeat": heartbeat}

    result = asyncio.run(collect_worker_health(TEST_TARGETS[:1], Transport()))
    assert result["gpu01"]["reachable"] is True
    assert result["gpu01"]["ready"] is False
    assert "different server id" in result["gpu01"]["reason"]


def test_status_on_an_uninitialized_worker_does_not_create_state(tmp_path):
    state_directory = tmp_path / "new-worker"

    class LocalWorkerSsh:
        async def run_command(self, host, command, *, timeout_seconds):
            assert command == ["~/.local/bin/gpuops", "worker", "status", "--json"]
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "gpuops",
                *command[1:],
                "--state-dir",
                str(state_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
            return SSHResult(host, process.returncode, stdout.decode(), stderr.decode(), 0.0)

    result = asyncio.run(
        collect_worker_health(TEST_TARGETS[:1], SshWorkerTransport(ssh=LocalWorkerSsh()))
    )
    assert result["gpu01"]["reachable"] is True
    assert result["gpu01"]["ready"] is False
    assert result["gpu01"]["reason"] == "worker process is stopped"
    assert not state_directory.exists()


def test_job_query_returns_explanation_without_submission_or_reconciliation(monkeypatch, capsys):
    from gpuops.transport import JobDetail

    record = JobRecord(
        run_id="waiting",
        spec=JobSpec(
            id="train",
            project="research",
            command=["python", "train.py"],
            working_directory="/work",
            resources={"host": "gpu01", "gpus": [0]},
        ),
    )
    explanation = {
        "waiting": True,
        "code": "gpu_occupied",
        "message": "GPU 0 is occupied.",
        "next_action": "wait",
        "observed_at": record.created_at.isoformat(),
        "details": {"occupied_gpu_indices": [0]},
    }

    class Transport:
        async def get_job_detail(self, target, run_id):
            assert (target.server_id, run_id) == ("gpu01", "waiting")
            return JobDetail(record=record, explanation=explanation)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: object())
    assert main(["job", "waiting", "--server", "gpu01", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["job"] == record.model_dump(mode="json")
    assert payload["explanation"] == explanation
