"""Live Web targets and card slots follow the current fleet and Worker data."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import gpuops.config as config
from gpuops.config import ServerTarget
from gpuops.control import RemoteTarget
from gpuops.monitor import MonitorService
from gpuops.monitor.collector import CollectionError
from gpuops.monitor.service import ServerResult
from gpuops.transport import JobHistory
from gpuops.web.app import create_app
from gpuops.web.control_provider import ControlDashboardProvider
from gpuops.web.provider import LiveProvider


def _sample(*indices: int) -> dict:
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gpus": [
            {
                "index": index,
                "name": f"GPU {index}",
                "utilization.gpu": 0,
                "memory.used": 0,
                "memory.total": 24576,
                "processes": [],
            }
            for index in indices
        ],
    }


def test_running_web_uses_current_targets_for_monitor_queue_history_and_jobs(monkeypatch):
    targets: list[ServerTarget] = []
    monkeypatch.setattr(config, "select_targets", lambda: tuple(targets))

    class Collector:
        offline: set[str] = set()
        calls: list[str] = []

        async def collect(self, host, *, timeout_seconds):
            self.calls.append(host)
            if host in self.offline:
                raise CollectionError("SSH is down", code="connection_failed")
            return _sample(2, 5), 0.01

    class Transport:
        calls: list[tuple[str, str, str]] = []

        async def list_summaries(self, target, *, statuses):
            self.calls.append(("queue", target.server_id, target.ssh_host))
            return []

        async def history(self, target, *, limit, project, cursor):
            self.calls.append(("history", target.server_id, target.ssh_host))
            return JobHistory([], [])

        async def get_job(self, target, run_id):
            self.calls.append(("job", target.server_id, target.ssh_host))
            return None

    collector = Collector()
    transport = Transport()
    provider = ControlDashboardProvider(
        LiveProvider(MonitorService(collector=collector), cache_seconds=60), transport
    )
    with TestClient(create_app(provider)) as client:
        assert client.get("/api/v1/snapshot").json()["servers"] == []
        assert client.get("/api/v1/queue").json()["servers"] == []
        assert client.get("/api/v1/history").json()["servers"] == []
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["reason"] == "no_servers_registered"

        targets.append(ServerTarget("gpu01", "Server six", "ssh-six"))
        snapshot = client.get("/api/v1/snapshot").json()
        assert [server["id"] for server in snapshot["servers"]] == ["gpu01"]
        assert [gpu["index"] for gpu in snapshot["servers"][0]["gpus"]] == [2, 5]
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/api/v1/queue").json()["servers"][0]["id"] == "gpu01"
        assert client.get("/api/v1/history").json()["servers"][0]["id"] == "gpu01"
        assert client.get("/api/v1/jobs/absent").status_code == 404
        assert {call[2] for call in transport.calls} == {"ssh-six"}

        targets.append(ServerTarget("new", "New server", "ssh-new"))
        collector.offline.add("ssh-new")
        snapshot = client.get("/api/v1/snapshot").json()
        by_id = {server["id"]: server for server in snapshot["servers"]}
        assert set(by_id) == {"gpu01", "new"}
        assert by_id["new"]["status"] == "offline"
        assert by_id["new"]["gpus"] == []
        assert snapshot["summary"]["total"] == 2
        assert {server["id"] for server in client.get("/api/v1/queue").json()["servers"]} == {
            "gpu01", "new"
        }
        assert {server["id"] for server in client.get("/api/v1/history").json()["servers"]} == {
            "gpu01", "new"
        }
        transport.calls.clear()
        assert client.get("/api/v1/jobs/absent").status_code == 404
        assert {call[2] for call in transport.calls} == {"ssh-six", "ssh-new"}

        targets[:] = [targets[1]]
        snapshot = client.get("/api/v1/snapshot").json()
        assert [server["id"] for server in snapshot["servers"]] == ["new"]
        assert [server["id"] for server in client.get("/api/v1/queue").json()["servers"]] == [
            "new"
        ]

    assert collector.calls.count("ssh-six") == 2
    assert collector.calls.count("ssh-new") == 2


def test_live_cards_follow_reported_indices_and_keep_disappeared_cards_visible():
    target = ServerTarget("host", "Host", "ssh-host")
    results = [
        ServerResult(server=target, online=True, data=_sample(2, 7)),
        ServerResult(server=target, online=True, data=_sample(7, 9)),
        ServerResult(
            server=target,
            online=False,
            error="SSH is down",
            error_code="connection_failed",
        ),
    ]

    class Monitor:
        async def collect_all(self):
            return [results.pop(0)]

    provider = LiveProvider(Monitor(), cache_seconds=0)
    first = asyncio.run(provider.get_snapshot())
    assert [gpu["index"] for gpu in first["servers"][0]["gpus"]] == [2, 7]
    assert first["summary"] == {
        "total": 2, "idle": 2, "busy": 0, "residual": 0, "unknown": 0, "offline": 0
    }

    second = asyncio.run(provider.get_snapshot())
    assert [(gpu["index"], gpu["status"]) for gpu in second["servers"][0]["gpus"]] == [
        (2, "unknown"), (7, "idle"), (9, "idle")
    ]
    assert second["servers"][0]["status"] == "degraded"

    offline = asyncio.run(provider.get_snapshot())
    assert [(gpu["index"], gpu["status"]) for gpu in offline["servers"][0]["gpus"]] == [
        (2, "offline"), (7, "offline"), (9, "offline")
    ]
    assert offline["servers"][0]["stale"] is True


def test_explicit_monitor_and_control_targets_stay_fixed(monkeypatch):
    def unexpected_registry_read():
        raise AssertionError("explicit targets must not read the registry")

    monkeypatch.setattr(config, "select_targets", unexpected_registry_read)
    target = ServerTarget("fixed", "Fixed", "ssh-fixed")
    monitor = MonitorService((target,))
    control = ControlDashboardProvider(
        LiveProvider(monitor), object(), (RemoteTarget("fixed", "ssh-fixed"),)
    )
    assert monitor.targets == (target,)
    assert control.targets == (RemoteTarget("fixed", "ssh-fixed"),)
