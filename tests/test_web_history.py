"""Exercise the web -> transport -> RPC -> queue history boundary together."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from gpuops.control import RemoteTarget
from gpuops.models import JobSpec, JobStatus
from gpuops.ssh import SSHResult
from gpuops.transport import SshWorkerTransport
from gpuops.web.app import create_app
from gpuops.web.control_provider import ControlDashboardProvider
from gpuops.web.provider import DemoProvider
from gpuops.worker.rpc import WorkerRpc

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


class LocalRpcSsh:
    def __init__(self, workers):
        self.workers = workers
        self.calls = []
        self.offline = set()

    async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
        request = json.loads(stdin)
        self.calls.append((host, request))
        if host in self.offline:
            raise ConnectionError("worker offline")
        response = self.workers[host].handle(request)
        return SSHResult(host, 0, json.dumps(response), "", 0.01)


@pytest.fixture
def fleet(tmp_path):
    workers = {
        server: WorkerRpc(tmp_path / server, server_id=server, clock=lambda: NOW)
        for server in ("gpu01", "gpu02")
    }
    for server, worker in workers.items():
        for i in range(63):
            spec = JobSpec(
                id=f"job-{i:03}",
                project="even" if i % 2 == 0 else "odd",
                command=["python", "train.py", "--private-arg"],
                working_directory="/work",
                runtime={"environment": {"PRIVATE_ENV": "hidden"}},
                resources={"host": server},
            )
            record = worker.queue.enqueue(spec, run_id=f"run-{i:03}", now=NOW)
            if i < 62:
                worker.queue.request_cancel(record.run_id, now=NOW + timedelta(seconds=i // 2))
        # Any accidentally unfiltered web list would now fail this integration test.
        original_list = worker.queue.list

        def filtered_only(statuses=None, original_list=original_list):
            assert statuses is not None, "web must never enumerate all historical records"
            return original_list(statuses)

        worker.queue.list = filtered_only
    ssh = LocalRpcSsh(workers)
    provider = ControlDashboardProvider(
        DemoProvider(),
        SshWorkerTransport(ssh=ssh),
        [RemoteTarget(server, server) for server in workers],
        clock=lambda: NOW,
    )
    with TestClient(create_app(provider)) as client:
        yield client, ssh, workers


def test_active_summary_excludes_terminal_and_execution_payload(fleet):
    client, ssh, _workers = fleet
    payload = client.get("/api/v1/queue").json()
    assert len(payload["items"]) == 2
    assert {item["id"] for item in payload["items"]} == {"run-062"}
    assert all("command" not in item and "env" not in item for item in payload["items"])
    assert all(request["params"]["summary"] for _host, request in ssh.calls)
    assert all(
        set(request["params"]["statuses"])
        == {status.value for status in JobStatus if not status.terminal}
        for _host, request in ssh.calls
    )


def test_global_history_pages_handle_ties_and_new_completions(fleet):
    client, ssh, workers = fleet
    first = client.get("/api/v1/history").json()
    assert len(first["items"]) == 50
    assert first["projects"] == ["even", "odd"]
    seen = [(item["server"], item["id"]) for item in first["items"]]
    # A job completes between requests; an offset page would repeat an old entry.
    workers["gpu01"].queue.request_cancel("run-062", now=NOW + timedelta(hours=1))
    cursor = first["next_cursor"]
    while cursor:
        page = client.get("/api/v1/history", params={"cursor": cursor}).json()
        seen.extend((item["server"], item["id"]) for item in page["items"])
        cursor = page["next_cursor"]
    expected = [(server, f"run-{i:03}") for i in range(61, -1, -1) for server in ("gpu02", "gpu01")]
    assert seen == expected
    assert all(request["method"] == "history" for _host, request in ssh.calls)
    assert all(request["params"]["limit"] == 51 for _host, request in ssh.calls)


def test_project_filter_is_remote_and_keeps_project_options(fleet):
    client, ssh, _workers = fleet
    payload = client.get("/api/v1/history", params={"project": "odd"}).json()
    assert len(payload["items"]) == 50
    assert {item["project"] for item in payload["items"]} == {"odd"}
    assert payload["projects"] == ["even", "odd"]
    assert all(request["params"]["project"] == "odd" for _host, request in ssh.calls)
    assert all("command" not in item for item in payload["items"])


def test_offline_history_is_visible_and_does_not_issue_a_skipping_cursor(fleet):
    client, ssh, _workers = fleet
    ssh.offline.add("gpu02")
    payload = client.get("/api/v1/history").json()
    assert len(payload["items"]) == 50
    assert payload["next_cursor"] is None
    assert not payload["servers"][1]["online"]
    assert "worker offline" in payload["servers"][1]["error"]


def test_terminal_detail_uses_direct_lookup_and_keeps_command_and_logs(fleet):
    client, ssh, workers = fleet
    # Give this ID a unique host, as the public detail API requires.
    record = workers["gpu01"].queue.enqueue(
        JobSpec(
            id="detail",
            project="detail",
            command=["echo", "done"],
            working_directory="/work",
            resources={"host": "gpu01"},
        ),
        run_id="unique-done",
        now=NOW,
    )
    workers["gpu01"].queue.request_cancel(record.run_id, now=NOW)
    response = client.get("/api/v1/jobs/unique-done")
    assert response.status_code == 200
    assert response.json()["command"] == "echo done"
    assert response.json()["state"] == "cancelled"
    assert all(request["method"] in {"show", "logs"} for _host, request in ssh.calls)


@pytest.mark.parametrize(
    "params,status",
    [
        ({"limit": 0}, 422),
        ({"limit": 101}, 422),
        ({"project": ""}, 422),
        ({"cursor": "invalid"}, 400),
        ({"cursor": "bnVsbA"}, 400),
    ],
)
def test_invalid_history_query_is_rejected_before_remote_reads(fleet, params, status):
    client, ssh, _workers = fleet
    assert client.get("/api/v1/history", params=params).status_code == status
    assert not ssh.calls


def test_demo_separates_active_and_history_and_supports_pagination():
    with TestClient(create_app(DemoProvider())) as client:
        active = client.get("/api/v1/queue").json()["items"]
        history = client.get("/api/v1/history", params={"limit": 2}).json()
        next_page = client.get("/api/v1/history", params={"cursor": history["next_cursor"]}).json()
    assert all(item["state"] in {"running", "queued"} for item in active)
    assert len(history["items"]) == 2
    assert len(next_page["items"]) == 1
    assert next_page["next_cursor"] is None
