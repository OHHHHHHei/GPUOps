"""Exercise Agent query commands against real RPC and queue implementations."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import TEST_TARGETS
from fastapi.testclient import TestClient

from gpuops.cli import main
from gpuops.control import RemoteTarget
from gpuops.models import ACTIVE_JOB_STATUSES, JobSpec
from gpuops.ssh import SSHResult
from gpuops.transport import SshWorkerTransport
from gpuops.web.app import create_app
from gpuops.web.control_provider import ControlDashboardProvider
from gpuops.web.provider import DemoProvider
from gpuops.worker.rpc import WorkerRpc

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


@pytest.fixture
def query_fleet(tmp_path, monkeypatch):
    workers = {
        target.id: WorkerRpc(
            tmp_path / target.id, server_id=target.id, readiness_probe=lambda: (True, "ready")
        )
        for target in TEST_TARGETS
    }
    for server, rpc in workers.items():
        for i in range(32):
            record = rpc.queue.enqueue(
                JobSpec(
                    id=f"job-{i:02}",
                    project="even" if i % 2 == 0 else "odd",
                    command=["echo", "private-command"],
                    working_directory="/work",
                    runtime={"environment": {"PRIVATE_ENV": "secret-value"}},
                    resources={"host": server, "gpus": [0]},
                ),
                run_id=f"run-{i:02}",
                now=NOW,
            )
            if i < 31:
                rpc.queue.request_cancel(record.run_id, now=NOW + timedelta(seconds=i // 2))

    class LocalRpcSsh:
        calls = []
        offline = set()

        async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
            server = next(target.id for target in TEST_TARGETS if target.ssh_host == host)
            request = json.loads(stdin)
            self.calls.append((server, request))
            if server in self.offline:
                raise OSError("worker offline")
            return SSHResult(host, 0, json.dumps(workers[server].handle(request)), "", 0.01)

    ssh = LocalRpcSsh()
    transport = SshWorkerTransport(ssh=ssh)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: transport)

    def no_submission_journal(*args, **kwargs):
        raise AssertionError("queries must not create a submission journal")

    monkeypatch.setattr("gpuops.control.SubmissionJournal", no_submission_journal)
    return transport, ssh, workers


def test_queue_defaults_to_active_summaries_with_explicit_legacy_escape(query_fleet, capsys):
    _transport, ssh, workers = query_fleet
    assert main(["queue", "--json"]) == 0
    output = capsys.readouterr().out
    active = json.loads(output)
    assert active["view"] == "active"
    assert [job["run_id"] for host in active["servers"] for job in host["jobs"]] == [
        "run-31",
        "run-31",
    ]
    assert "private-command" not in output and "secret-value" not in output
    assert all(
        request["params"]
        == {"summary": True, "statuses": sorted(status.value for status in ACTIVE_JOB_STATUSES)}
        for _server, request in ssh.calls
    )

    assert main(["queue", "--all", "--json"]) == 0
    legacy = json.loads(capsys.readouterr().out)
    assert legacy["view"] == "all"
    for host in legacy["servers"]:
        assert host["jobs"] == [
            job.model_dump(mode="json") for job in workers[host["server_id"]].queue.list()
        ]


def test_cli_and_web_share_history_pages_and_cursors(query_fleet, capsys):
    transport, ssh, workers = query_fleet
    assert main(["history", "--json"]) == 0
    output = capsys.readouterr().out
    first = json.loads(output)
    assert len(first["jobs"]) == 50
    assert first["projects"] == ["even", "odd"]
    assert "private-command" not in output and "secret-value" not in output
    targets = [RemoteTarget(target.id, target.ssh_host) for target in TEST_TARGETS]
    with TestClient(
        create_app(ControlDashboardProvider(DemoProvider(), transport, targets))
    ) as web:
        page = web.get("/api/v1/history").json()
        assert first["next_cursor"] == page["next_cursor"]
        assert [(job["server_id"], job["run_id"]) for job in first["jobs"]] == [
            (job["server"], job["id"]) for job in page["items"]
        ]
        # New completion does not shift the next page; a Web cursor works in the CLI.
        workers["gpu01"].queue.request_cancel("run-31", now=NOW + timedelta(hours=1))
        assert main(["history", "--cursor", page["next_cursor"], "--json"]) == 0
        second = json.loads(capsys.readouterr().out)
    assert second["next_cursor"] is None
    seen = [(job["server_id"], job["run_id"]) for job in first["jobs"] + second["jobs"]]
    assert seen == [
        (server, f"run-{i:02}") for i in range(30, -1, -1) for server in ("gpu02", "gpu01")
    ]
    assert all(request["method"] == "history" for _server, request in ssh.calls)


def test_history_filters_project_and_server_remotely(query_fleet, capsys):
    _transport, ssh, _workers = query_fleet
    assert (
        main(["history", "--server", "gpu02", "--project", "even", "--limit", "3", "--json"]) == 0
    )
    page = json.loads(capsys.readouterr().out)
    assert len(page["jobs"]) == 3
    assert all(job["server_id"] == "gpu02" and job["project"] == "even" for job in page["jobs"])
    assert page["next_cursor"]
    assert [(server, request["params"]) for server, request in ssh.calls] == [
        ("gpu02", {"project": "even", "limit": 4})
    ]


@pytest.mark.parametrize(
    "offline,outcome", [({"gpu02"}, "partial"), ({"gpu01", "gpu02"}, "failed")]
)
def test_history_preserves_errors_and_disables_paging_for_partial_results(
    query_fleet, capsys, offline, outcome
):
    _transport, ssh, _workers = query_fleet
    ssh.offline = offline
    assert main(["history", "--limit", "3", "--json"]) == 1
    page = json.loads(capsys.readouterr().out)
    assert page["outcome"] == outcome and page["ok"] is False
    assert page["next_cursor"] is None
    assert len(page["jobs"]) == (3 if outcome == "partial" else 0)
    for host in page["servers"]:
        if host["server_id"] in offline:
            assert host["online"] is False and host["error_code"] == "io_error"


@pytest.mark.parametrize(
    "args", [["--limit", "0"], ["--limit", "101"], ["--project", ""], ["--cursor", "invalid"]]
)
def test_invalid_history_input_never_sends_an_rpc(query_fleet, capsys, args):
    _transport, ssh, _workers = query_fleet
    assert main(["history", *args, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_argument"
    assert not ssh.calls


def test_historical_detail_remains_a_direct_full_record_query(query_fleet, capsys):
    _transport, ssh, workers = query_fleet
    assert main(["job", "run-00", "--server", "gpu01", "--json"]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["job"] == workers["gpu01"].queue.get("run-00").model_dump(mode="json")
    assert [request["method"] for _server, request in ssh.calls] == ["show"]
