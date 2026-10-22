from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpuops.control import RemoteTarget
from gpuops.models import Allocation, JobRecord, JobSpec, JobStatus, JobSummary, LaunchInfo
from gpuops.monitor import ServerTarget
from gpuops.monitor.service import ServerResult
from gpuops.transport import JobDetail, SshWorkerTransport
from gpuops.web import DemoProvider, LiveProvider, create_app, validate_loopback_host
from gpuops.web.control_provider import ControlDashboardProvider


def test_homepage_loads_console_assets() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "GPUOps 控制台" in response.text
    assert "GPU 总览" in response.text
    assert 'id="queue-errors"' in response.text
    assert "/static/app.js" in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"


def test_web_assets_include_managed_job_cancel_control() -> None:
    static_root = Path(__file__).parents[1] / "src" / "gpuops" / "web" / "static"
    script = (static_root / "app.js").read_text(encoding="utf-8")
    styles = (static_root / "styles.css").read_text(encoding="utf-8")

    assert 'method: "POST"' in script
    assert '"取消作业"' in script
    assert ".danger-button" in styles


def test_demo_snapshot_contains_two_servers_and_sixteen_gpus() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        response = client.get("/api/v1/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert [server["id"] for server in payload["servers"]] == ["gpu01", "gpu02"]
    assert sum(len(server["gpus"]) for server in payload["servers"]) == 16
    assert payload["summary"] == {
        "total": 16,
        "idle": 7,
        "busy": 7,
        "residual": 1,
        "unknown": 0,
        "offline": 1,
    }
    statuses = {gpu["status"] for server in payload["servers"] for gpu in server["gpus"]}
    assert {"idle", "busy", "residual", "offline"} <= statuses
    offline = next(
        gpu for server in payload["servers"] for gpu in server["gpus"] if gpu["status"] == "offline"
    )
    assert all(
        offline[field] is None
        for field in (
            "utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "temperature_c",
            "power_watts",
        )
    )


def test_queue_and_job_detail_include_agent_facing_fields() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        queue_response = client.get("/api/v1/queue")
        job_response = client.get("/api/v1/jobs/run-ranking-042")

    assert queue_response.status_code == 200
    queue_item = queue_response.json()["items"][0]
    expected_fields = {"id", "state", "server", "gpus", "estimated_start", "estimated_end"}
    assert expected_fields <= queue_item.keys()

    assert job_response.status_code == 200
    job = job_response.json()
    assert job["id"] == "run-ranking-042"
    assert job["working_directory"].startswith("/data/")
    assert job["logs"]


def test_unknown_job_returns_404() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        response = client.get("/api/v1/jobs/not-a-job")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_custom_provider_is_used_through_protocol() -> None:
    class StubProvider:
        async def get_snapshot(self) -> dict[str, Any]:
            return {"generated_at": "now", "summary": {}, "servers": []}

        async def get_queue(self) -> dict[str, Any]:
            return {"updated_at": "now", "items": []}

        async def get_job(self, job_id: str) -> dict[str, Any] | None:
            return {"id": job_id, "logs": []}

        async def cancel_job(self, job_id: str) -> dict[str, Any] | None:
            return {"id": job_id, "accepted": True}

        async def health(self) -> dict[str, Any]:
            return {"status": "ok", "mode": "stub"}

    with TestClient(create_app(StubProvider())) as client:
        snapshot = client.get("/api/v1/snapshot").json()
        job = client.get("/api/v1/jobs/custom-job").json()
        health = client.get("/healthz").json()

    assert snapshot["servers"] == []
    assert job["id"] == "custom-job"
    assert health == {"status": "ok", "mode": "stub"}


def test_health_reports_demo_mode_and_server_health() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["mode"] == "demo"
    assert response.json()["servers"] == [
        {"id": "gpu01", "status": "online"},
        {"id": "gpu02", "status": "degraded"},
    ]


@pytest.mark.parametrize("host_recovered", [True, False])
def test_web_refreshes_expired_sample_before_reporting_stale(monkeypatch, host_recovered) -> None:
    now = datetime.now(timezone.utc)

    class FakeMonitor:
        calls = 0

        async def collect_all(self):
            self.calls += 1
            sampled_at = now
            if host_recovered and self.calls > 1:
                sampled_at += timedelta(seconds=10)
            return [ServerResult(
                server=ServerTarget("gpu01", "gpu01", "gpu01"), online=True,
                data={"collected_at": sampled_at.isoformat(), "gpus": [{
                    "index": 0, "utilization.gpu": 0, "memory.used": 0,
                    "memory.total": 24576, "processes": [],
                }]},
            )]

    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now + timedelta(seconds=21)

    monitor = FakeMonitor()
    provider = LiveProvider(monitor, cache_seconds=30, expected_gpu_count=1)
    assert asyncio.run(provider.get_snapshot())["summary"]["idle"] == 1
    monkeypatch.setattr("gpuops.web.provider.datetime", LaterDatetime)
    snapshot = asyncio.run(provider.get_snapshot())
    assert monitor.calls == 2
    if host_recovered:
        assert snapshot["summary"]["idle"] == 1
        assert snapshot["servers"][0]["stale"] is False
        assert snapshot["servers"][0]["error_code"] is None
    else:
        assert snapshot["summary"]["idle"] == 0
        assert snapshot["servers"][0]["stale"] is True
        assert snapshot["servers"][0]["error_code"] == "telemetry_stale"
        # Even a result that expires in transit gets the short failure cache.
        asyncio.run(provider.get_snapshot())
        assert monitor.calls == 2


@pytest.fixture
def web_clock(monkeypatch):
    clock = SimpleNamespace(seconds=0.0)
    start = datetime(2026, 9, 27, tzinfo=timezone.utc)

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return start + timedelta(seconds=clock.seconds)

    monkeypatch.setattr("gpuops.web.provider.datetime", ClockDatetime)
    # Replace only the provider's clock, leaving asyncio's event-loop clock intact.
    monkeypatch.setattr(
        "gpuops.web.provider.time", SimpleNamespace(monotonic=lambda: clock.seconds)
    )
    clock.now = ClockDatetime.now
    return clock


def test_five_second_polls_keep_healthy_ten_second_samples_online(web_clock) -> None:
    class Monitor:
        calls = 0

        async def collect_all(self):
            self.calls += 1
            # Reproduce the live phase: the host's sample is already 9s old on first read.
            age = (9 + web_clock.seconds) % 10
            sampled_at = web_clock.now() - timedelta(seconds=age)
            return [ServerResult(
                server=ServerTarget("gpu02", "gpu02", "gpu02"), online=True,
                data={"collected_at": sampled_at.isoformat(), "gpus": [{
                    "index": 0, "utilization.gpu": 0, "memory.used": 0,
                    "memory.total": 24576, "processes": [],
                }]},
            )]

    monitor = Monitor()
    provider = LiveProvider(monitor, expected_gpu_count=1)

    async def poll():
        calls_by_poll = []
        for seconds in range(0, 31, 5):
            web_clock.seconds = seconds
            snapshot, health, _ = await asyncio.gather(
                provider.get_snapshot(), provider.health(), provider.get_process_etas()
            )
            assert snapshot["summary"]["idle"] == 1
            assert snapshot["servers"][0]["stale"] is False
            assert health["status"] == "ok"
            calls_by_poll.append(monitor.calls)
        return calls_by_poll

    # The 15s local cache expires at t=15; the host's 10s sample stays usable.
    assert asyncio.run(poll()) == [1, 1, 1, 2, 2, 2, 3]


@pytest.mark.parametrize("clock_offset_seconds", [-4.2, 4.2])
def test_five_second_polls_tolerate_small_host_clock_offset(
    web_clock, clock_offset_seconds
) -> None:
    class Monitor:
        calls = 0

        async def collect_all(self):
            self.calls += 1
            # The host samples every 10s using its own wall clock.
            sampled_at = web_clock.now() + timedelta(
                seconds=clock_offset_seconds - web_clock.seconds % 10
            )
            return [ServerResult(
                server=ServerTarget("gpu02", "gpu02", "gpu02"), online=True,
                data={"collected_at": sampled_at.isoformat(), "gpus": [{
                    "index": 0, "sampled_at": sampled_at.isoformat(),
                    "utilization.gpu": 0, "memory.used": 0,
                    "memory.total": 24576, "processes": [],
                }]},
            )]

    monitor = Monitor()
    provider = LiveProvider(monitor, expected_gpu_count=1)

    async def poll():
        calls_by_poll = []
        for seconds in range(0, 31, 5):
            web_clock.seconds = seconds
            snapshot, health = await asyncio.gather(provider.get_snapshot(), provider.health())
            assert snapshot["summary"]["idle"] == 1
            assert snapshot["servers"][0]["status"] == "online"
            assert snapshot["servers"][0]["stale"] is False
            assert snapshot["servers"][0]["gpus"][0]["status"] == "idle"
            assert health["status"] == "ok"
            calls_by_poll.append(monitor.calls)
        return calls_by_poll

    assert asyncio.run(poll()) == [1, 1, 1, 2, 2, 2, 3]


@pytest.mark.parametrize(
    "error_code", ["telemetry_stale", "gpu_query_timeout", "connection_failed"]
)
def test_failed_cache_coalesces_requests_and_recovers_by_next_poll(web_clock, error_code) -> None:
    class Monitor:
        calls = 0

        async def collect_all(self):
            self.calls += 1
            # Let concurrent endpoints enter the lock while collection is in flight.
            await asyncio.sleep(0)
            healthy = ServerResult(
                server=ServerTarget("gpu02", "gpu02", "gpu02"), online=True,
                data={"collected_at": web_clock.now().isoformat(), "gpus": [{
                    "index": 0, "utilization.gpu": 0, "memory.used": 0,
                    "memory.total": 24576, "processes": [],
                }]},
            )
            if self.calls == 1:
                return [ServerResult(
                    server=ServerTarget("gpu01", "gpu01", "gpu01"), online=False,
                    error="sample unavailable", error_code=error_code,
                ), healthy]
            return [ServerResult(
                server=ServerTarget("gpu01", "gpu01", "gpu01"), online=True, data=healthy.data,
            ), healthy]

    monitor = Monitor()
    provider = LiveProvider(monitor, expected_gpu_count=1)

    async def poll():
        for seconds in (0, 0.5, 5):
            web_clock.seconds = seconds
            snapshot, health, _ = await asyncio.gather(
                provider.get_snapshot(), provider.health(), provider.get_process_etas()
            )
            assert snapshot["servers"][1]["status"] == "online"
            if seconds < 5:
                assert monitor.calls == 1
                assert snapshot["servers"][0]["error_code"] == error_code
                assert snapshot["summary"]["idle"] == 1
                assert health["status"] == "degraded"
            else:
                assert monitor.calls == 2
                assert snapshot["servers"][0]["status"] == "online"
                assert snapshot["summary"]["idle"] == 2
                assert health["status"] == "ok"

    asyncio.run(poll())


def test_live_provider_maps_monitor_data_and_reuses_short_cache() -> None:
    class FakeMonitor:
        def __init__(self) -> None:
            self.calls = 0

        async def collect_all(self) -> list[ServerResult]:
            self.calls += 1
            return [
                ServerResult(
                    server=ServerTarget("gpu01", "GPU 01", "lab-gpu01"),
                    online=True,
                    data={
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "gpus": [
                            {
                                "index": 0,
                                "name": "RTX 3090",
                                "utilization.gpu": 91,
                                "memory.used": 18000,
                                "memory.total": 24576,
                                "temperature.gpu": 70,
                                "power.draw": 250,
                                "processes": [
                                    {
                                        "pid": 123,
                                        "username": "researcher",
                                        "cmdline": "python train.py",
                                        "elapsed_seconds": 600,
                                        "gpu_memory_usage": 17500,
                                    }
                                ],
                            }
                        ],
                    },
                )
            ]

    monitor = FakeMonitor()
    provider = LiveProvider(monitor, cache_seconds=30, expected_gpu_count=2)  # type: ignore[arg-type]
    with TestClient(create_app(provider)) as client:
        snapshot = client.get("/api/v1/snapshot").json()
        health = client.get("/healthz").json()

    assert monitor.calls == 1
    assert snapshot["summary"] == {
        "total": 2,
        "idle": 0,
        "busy": 1,
        "residual": 0,
        "unknown": 0,
        "offline": 1,
    }
    assert snapshot["servers"][0]["gpus"][0]["processes"][0]["eta"]["label"] == "未知"
    missing_gpu = snapshot["servers"][0]["gpus"][1]
    assert missing_gpu["status"] == "offline"
    assert missing_gpu["utilization_percent"] is None
    assert missing_gpu["memory_used_mib"] is None
    assert missing_gpu["memory_total_mib"] is None
    assert missing_gpu["temperature_c"] is None
    assert missing_gpu["power_watts"] is None
    assert health["mode"] == "live"
    assert health["status"] == "degraded"


def test_live_provider_normalizes_nonfinite_metrics_at_json_boundary() -> None:
    class NonfiniteMonitor:
        async def collect_all(self) -> list[ServerResult]:
            return [
                ServerResult(
                    server=ServerTarget("gpu01", "GPU 01", "lab-gpu01"),
                    online=True,
                    data={
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "gpus": [
                            {
                                "index": 0,
                                "name": "RTX 3090",
                                "utilization.gpu": float("nan"),
                                "memory.used": float("inf"),
                                "memory.total": float("-inf"),
                                "temperature.gpu": float("nan"),
                                "power.draw": float("inf"),
                                "processes": [
                                    {
                                        "pid": float("nan"),
                                        "elapsed_seconds": float("inf"),
                                        "gpu_memory_usage": float("-inf"),
                                    }
                                ],
                            }
                        ],
                    },
                )
            ]

    provider = LiveProvider(NonfiniteMonitor(), expected_gpu_count=1)  # type: ignore[arg-type]
    with TestClient(create_app(provider)) as client:
        response = client.get("/api/v1/snapshot")
        health = client.get("/healthz").json()

    assert response.status_code == 200
    snapshot = response.json()
    gpu = snapshot["servers"][0]["gpus"][0]
    assert snapshot["summary"]["unknown"] == 1
    assert snapshot["servers"][0]["status"] == "degraded"
    assert gpu["status"] == "unknown"
    assert gpu["utilization_percent"] is None
    assert gpu["memory_used_mib"] is None
    assert gpu["memory_total_mib"] is None
    assert gpu["temperature_c"] is None
    assert gpu["power_watts"] is None
    assert gpu["processes"][0]["pid"] is None
    assert gpu["processes"][0]["elapsed_seconds"] is None
    assert gpu["processes"][0]["gpu_memory_mib"] is None
    assert health["status"] == "degraded"
    assert health["servers"] == [{"id": "gpu01", "status": "degraded"}]


def test_live_provider_preserves_last_good_metrics_without_advertising_stale_capacity() -> None:
    target = ServerTarget("gpu01", "GPU 01", "lab-gpu01")
    success_data = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gpus": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization.gpu": 91,
                "memory.used": 18000,
                "memory.total": 24576,
                "temperature.gpu": 70,
                "power.draw": 250,
                "processes": [
                    {
                        "pid": 123,
                        "username": "researcher",
                        "cmdline": "python train.py",
                        "elapsed_seconds": 600,
                        "gpu_memory_usage": 17500,
                        "start_time_ticks": 456,
                    }
                ],
            }
        ],
    }
    recovered_data = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gpus": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization.gpu": 0,
                "memory.used": 0,
                "memory.total": 24576,
                "temperature.gpu": 40,
                "power.draw": 30,
                "processes": [],
            }
        ],
    }

    class SequentialMonitor:
        def __init__(self) -> None:
            self.results = [
                ServerResult(
                    server=target, online=True, data=success_data, error=None, error_code=None
                ),
                ServerResult(
                    server=target,
                    online=False,
                    data=None,
                    error="collection exceeded 15 seconds",
                    error_code="collection_timeout",
                ),
                ServerResult(
                    server=target, online=True, data=recovered_data, error=None, error_code=None
                ),
            ]

        async def collect_all(self) -> list[Any]:
            return [self.results.pop(0)]

    provider = LiveProvider(SequentialMonitor(), cache_seconds=0, expected_gpu_count=1)  # type: ignore[arg-type]
    first = asyncio.run(provider.get_snapshot())
    stale = asyncio.run(provider.get_snapshot())
    recovered = asyncio.run(provider.get_snapshot())

    assert first["summary"]["busy"] == 1
    stale_server = stale["servers"][0]
    stale_gpu = stale_server["gpus"][0]
    assert stale_server["status"] == "degraded"
    assert stale_server["stale"] is True
    assert stale_server["last_seen"] == success_data["collected_at"]
    assert stale_server["error_code"] == "collection_timeout"
    assert stale["summary"]["idle"] == 0
    assert stale["summary"]["busy"] == 0
    assert stale["summary"]["unknown"] == 1
    assert stale_gpu["status"] == "unknown"
    assert stale_gpu["utilization_percent"] == 91
    assert stale_gpu["memory_used_mib"] == 18000
    assert stale_gpu["processes"][0]["pid"] == 123
    assert stale_gpu["processes"][0]["eta"]["end_at"] is None
    assert recovered["servers"][0]["stale"] is False
    assert recovered["servers"][0]["error_code"] is None
    assert recovered["summary"]["idle"] == 1


def test_live_provider_isolates_card_errors_and_recovers_quickly(web_clock) -> None:
    target = ServerTarget("gpu01", "GPU 01", "lab-gpu01")

    def gpu(index: int, **extra: Any) -> dict[str, Any]:
        return {
            "index": index,
            "name": "RTX 3090",
            "sampled_at": web_clock.now().isoformat(),
            "utilization.gpu": 0,
            "memory.used": 0,
            "memory.total": 24576,
            "processes": [],
            **extra,
        }

    healthy = {"collected_at": web_clock.now().isoformat(), "gpus": [gpu(i) for i in range(8)]}
    partial = {
        "collected_at": web_clock.now().isoformat(),
        "gpus": [
            gpu(i)
            if i not in {3, 6}
            else gpu(
                i,
                **(
                    {"utilization.gpu": None, "memory.used": None, "processes": None,
                     "error": "NVML query failed", "error_code": "gpu_query_failed"}
                    if i == 3
                    else {"error": "GPU sample is stale", "error_code": "telemetry_stale"}
                ),
            )
            for i in range(8)
        ],
    }

    class Monitor:
        calls = 0

        async def collect_all(self):
            self.calls += 1
            data = healthy if self.calls in {1, 3} else partial
            return [ServerResult(server=target, online=True, data=data)]

    monitor = Monitor()
    provider = LiveProvider(monitor, cache_seconds=15, expected_gpu_count=8)
    asyncio.run(provider.get_snapshot())
    provider._cached_at = -20
    degraded = asyncio.run(provider.get_snapshot())
    server = degraded["servers"][0]
    by_index = {gpu["index"]: gpu for gpu in server["gpus"]}

    assert server["status"] == "degraded"
    assert server["stale"] is False
    assert server["error"] is None
    assert degraded["summary"] == {
        "total": 8, "idle": 6, "busy": 0, "residual": 0, "unknown": 2, "offline": 0,
    }
    assert by_index[3]["status"] == "unknown"
    assert by_index[3]["memory_total_mib"] == 24576
    assert by_index[6]["status"] == "unknown"
    assert by_index[4]["status"] == by_index[5]["status"] == "idle"

    web_clock.seconds = 2
    recovered = asyncio.run(provider.get_snapshot())
    assert monitor.calls == 3
    assert recovered["servers"][0]["status"] == "online"
    assert all(gpu["error"] is None for gpu in recovered["servers"][0]["gpus"])


def test_stale_card_timestamp_cannot_be_reported_idle(web_clock) -> None:
    old = web_clock.now() - timedelta(minutes=1)

    class Monitor:
        async def collect_all(self):
            return [ServerResult(
                server=ServerTarget("gpu01", "gpu01", "gpu01"),
                online=True,
                data={"collected_at": web_clock.now().isoformat(), "gpus": [{
                    "index": 0, "sampled_at": old.isoformat(),
                    "utilization.gpu": 0, "memory.used": 0,
                    "memory.total": 24576, "processes": [],
                }]},
            )]

    snapshot = asyncio.run(LiveProvider(Monitor(), expected_gpu_count=1).get_snapshot())
    assert snapshot["servers"][0]["stale"] is False
    assert snapshot["servers"][0]["gpus"][0]["status"] == "unknown"
    assert snapshot["servers"][0]["gpus"][0]["error_code"] == "telemetry_stale"
    assert snapshot["summary"]["idle"] == 0


@pytest.mark.parametrize("future_field", ["collected_at", "sampled_at"])
def test_timestamp_more_than_five_seconds_ahead_remains_unusable(web_clock, future_field) -> None:
    now = web_clock.now().isoformat()
    future = (web_clock.now() + timedelta(seconds=6)).isoformat()
    data = {"collected_at": now, "gpus": [{
        "index": 0, "sampled_at": now,
        "utilization.gpu": 0, "memory.used": 0,
        "memory.total": 24576, "processes": [],
    }]}
    if future_field == "collected_at":
        data["collected_at"] = future
    else:
        data["gpus"][0]["sampled_at"] = future

    class Monitor:
        async def collect_all(self):
            return [ServerResult(
                server=ServerTarget("gpu01", "gpu01", "gpu01"), online=True, data=data,
            )]

    snapshot = asyncio.run(LiveProvider(Monitor(), expected_gpu_count=1).get_snapshot())
    server = snapshot["servers"][0]
    assert snapshot["summary"]["idle"] == 0
    if future_field == "collected_at":
        assert server["status"] == "degraded"
        assert server["error_code"] == "telemetry_stale"
    else:
        assert server["stale"] is False
        assert server["gpus"][0]["status"] == "unknown"
        assert server["gpus"][0]["error_code"] == "telemetry_stale"


def test_live_provider_distinguishes_connection_and_collection_failures() -> None:
    target = ServerTarget("gpu01", "GPU 01", "lab-gpu01")
    good = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gpus": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization.gpu": 0,
                "memory.used": 0,
                "memory.total": 24576,
                "processes": [],
            }
        ],
    }

    class Monitor:
        def __init__(self, results: list[Any]) -> None:
            self.results = results

        async def collect_all(self) -> list[Any]:
            return [self.results.pop(0)]

    connection_monitor = Monitor(
        [
            ServerResult(server=target, online=True, data=good, error=None, error_code=None),
            ServerResult(
                server=target,
                online=False,
                data=None,
                error="SSH exited 255",
                error_code="connection_failed",
            ),
        ]
    )
    connection_provider = LiveProvider(
        connection_monitor,
        cache_seconds=0,
        expected_gpu_count=1,  # type: ignore[arg-type]
    )
    asyncio.run(connection_provider.get_snapshot())
    connection = asyncio.run(connection_provider.get_snapshot())
    assert connection["servers"][0]["status"] == "offline"
    assert connection["servers"][0]["stale"] is True
    assert connection["servers"][0]["gpus"][0]["status"] == "offline"
    assert connection["servers"][0]["gpus"][0]["memory_total_mib"] == 24576

    collection_provider = LiveProvider(
        Monitor(
            [
                ServerResult(
                    server=target,
                    online=False,
                    data=None,
                    error="malformed JSON",
                    error_code="collection_failed",
                )
            ]
        ),
        cache_seconds=0,
        expected_gpu_count=1,
    )  # type: ignore[arg-type]
    collection = asyncio.run(collection_provider.get_snapshot())
    assert collection["servers"][0]["status"] == "degraded"
    assert collection["servers"][0]["stale"] is False
    assert collection["servers"][0]["gpus"][0]["status"] == "unknown"
    assert collection["summary"]["unknown"] == 1

    legacy_provider = LiveProvider(
        Monitor([ServerResult(server=target, online=False, data=None, error="offline")]),
        cache_seconds=0,
        expected_gpu_count=1,
    )  # type: ignore[arg-type]
    legacy = asyncio.run(legacy_provider.get_snapshot())
    assert legacy["servers"][0]["status"] == "offline"


def test_web_bind_is_restricted_to_loopback() -> None:
    assert validate_loopback_host("127.0.0.1") == "127.0.0.1"
    assert validate_loopback_host("::1") == "::1"
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host("0.0.0.0")


CONTROL_NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
CONTROL_TARGET_01 = RemoteTarget("gpu01", "lab-gpu01")
CONTROL_TARGET_02 = RemoteTarget("gpu02", "lab-gpu02")


def managed_spec(identifier: str, *, duration: str) -> JobSpec:
    return JobSpec(
        id=identifier,
        project="ranking",
        command=["python", "train.py", "--run", identifier],
        working_directory="/data/researcher/ranking",
        resources={"host": "gpu01", "gpu_count": 1},
        schedule={"expected_duration": duration},
        metadata={"name": f"Managed {identifier}", "created_by": "researcher"},
    )


def managed_running_job() -> JobRecord:
    return JobRecord(
        run_id="running-job",
        spec=managed_spec("running", duration="4h"),
        status=JobStatus.RUNNING,
        created_at=CONTROL_NOW - timedelta(hours=2),
        updated_at=CONTROL_NOW - timedelta(hours=1),
        allocation=Allocation(
            server_id="gpu01",
            gpu_indices=[0],
            worker_id="worker-01",
            lease_token="lease-running",
            leased_at=CONTROL_NOW - timedelta(hours=1, minutes=1),
            lease_expires_at=CONTROL_NOW + timedelta(minutes=1),
        ),
        launch=LaunchInfo(
            pid=42001,
            started_at=CONTROL_NOW - timedelta(hours=1),
        ),
    )


def managed_queued_job() -> JobRecord:
    return JobRecord(
        run_id="queued-job",
        spec=managed_spec("queued", duration="2h"),
        created_at=CONTROL_NOW - timedelta(minutes=10),
        updated_at=CONTROL_NOW - timedelta(minutes=10),
    )


class FakeSnapshotProvider:
    async def get_snapshot(self) -> dict[str, Any]:
        return {"generated_at": CONTROL_NOW.isoformat(), "summary": {}, "servers": []}

    async def get_queue(self) -> dict[str, Any]:
        raise AssertionError("control provider must own queue reads")

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        raise AssertionError(f"control provider must own job reads: {job_id}")

    async def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        raise AssertionError(f"control provider must own cancellation: {job_id}")

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "mode": "live", "servers": []}


class FakeQueueTransport:
    def __init__(self) -> None:
        self.jobs: dict[str, list[JobRecord] | Exception] = {
            "gpu01": [managed_running_job(), managed_queued_job()],
            "gpu02": ConnectionError("worker offline"),
        }
        self.log_content = "old-line\n" + ("x" * 80) + "\nlast-line\n"
        self.log_calls: list[tuple[str, str, int]] = []
        self.detail_calls: list[tuple[str, str]] = []
        self.detail_record: JobRecord | None = None
        self.detail_error: Exception | None = None
        self.explanation: dict[str, Any] = {
            "waiting": False,
            "code": "running",
            "message": "The job is running",
            "observed_at": CONTROL_NOW.isoformat(),
            "details": {},
        }
        self.cancel_calls: list[tuple[str, str]] = []

    async def list_jobs(self, target: RemoteTarget) -> list[JobRecord]:
        result = self.jobs[target.server_id]
        if isinstance(result, Exception):
            raise result
        return result

    async def list_summaries(self, target: RemoteTarget, *, statuses: set[JobStatus]):
        return [
            JobSummary.from_record(record)
            for record in await self.list_jobs(target)
            if record.status in statuses
        ]

    async def get_job(self, target: RemoteTarget, run_id: str):
        return next(
            (record for record in await self.list_jobs(target) if record.run_id == run_id), None
        )

    async def get_logs(
        self,
        target: RemoteTarget,
        run_id: str,
        *,
        tail_bytes: int,
    ) -> dict[str, Any]:
        self.log_calls.append((target.server_id, run_id, tail_bytes))
        return {"run_id": run_id, "content": self.log_content}

    async def get_job_detail(self, target: RemoteTarget, run_id: str) -> JobDetail | None:
        self.detail_calls.append((target.server_id, run_id))
        if self.detail_error is not None:
            raise self.detail_error
        record = self.detail_record or next(
            record for record in self.jobs[target.server_id] if record.run_id == run_id
        )
        return JobDetail(record, self.explanation)

    async def cancel(self, target: RemoteTarget, run_id: str) -> JobRecord:
        self.cancel_calls.append((target.server_id, run_id))
        original = next(record for record in self.jobs[target.server_id] if record.run_id == run_id)
        return JobRecord(
            run_id=original.run_id,
            spec=original.spec,
            status=JobStatus.CANCELLED,
            revision=original.revision + 1,
            created_at=original.created_at,
            updated_at=CONTROL_NOW,
            finished_at=CONTROL_NOW,
        )


def control_provider(
    transport: FakeQueueTransport,
    *,
    log_tail_bytes: int = 64,
) -> ControlDashboardProvider:
    return ControlDashboardProvider(
        FakeSnapshotProvider(),  # type: ignore[arg-type]
        transport,
        [CONTROL_TARGET_02, CONTROL_TARGET_01],
        clock=lambda: CONTROL_NOW,
        log_tail_bytes=log_tail_bytes,
    )


def test_control_provider_isolates_queue_failure_and_estimates_managed_eta() -> None:
    transport = FakeQueueTransport()
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.get("/api/v1/queue")

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["items"]] == ["running-job", "queued-job"]
    running, queued = payload["items"]
    assert running["estimated_start"] == "2026-09-19T09:00:00+00:00"
    assert running["estimated_end"] == "2026-09-19T13:00:00+00:00"
    assert running["eta_confidence"] == "low"
    assert running["eta"]["source_kind"] == "submitted"
    assert queued["estimated_start"] is None
    assert queued["estimated_end"] is None
    assert queued["eta_confidence"] == "unknown"
    assert payload["servers"] == [
        {"id": "gpu01", "online": True, "job_count": 2, "error": None},
        {
            "id": "gpu02",
            "online": False,
            "job_count": 0,
            "error": "ConnectionError: worker offline",
        },
    ]
    assert transport.detail_calls == []


def test_queued_eta_is_unknown_when_a_running_duration_is_unknown() -> None:
    transport = FakeQueueTransport()
    running = managed_running_job()
    unknown_spec = running.spec.model_copy(
        update={"schedule": running.spec.schedule.model_copy(update={"expected_duration": None})}
    )
    transport.jobs["gpu01"] = [
        running.model_copy(update={"spec": unknown_spec}),
        managed_queued_job(),
    ]
    with TestClient(create_app(control_provider(transport))) as client:
        payload = client.get("/api/v1/queue").json()

    queued = next(item for item in payload["items"] if item["id"] == "queued-job")
    assert queued["estimated_start"] is None
    assert queued["estimated_end"] is None
    assert queued["eta_confidence"] == "unknown"


def test_control_provider_job_detail_has_bounded_logs_and_located_host() -> None:
    transport = FakeQueueTransport()
    with TestClient(create_app(control_provider(transport, log_tail_bytes=32))) as client:
        response = client.get("/api/v1/jobs/running-job")

    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == "running-job"
    assert detail["server"] == "gpu01"
    assert detail["working_directory"] == "/data/researcher/ranking"
    assert detail["estimated_end"] == "2026-09-19T13:00:00+00:00"
    assert detail["logs"][-1] == "last-line"
    assert len("\n".join(detail["logs"]).encode("utf-8")) <= 32
    assert transport.log_calls == [("gpu01", "running-job", 32)]
    assert transport.detail_calls == [("gpu01", "running-job")]


def test_control_provider_uses_newest_record_and_exposes_explanation() -> None:
    transport = FakeQueueTransport()
    transport.detail_record = managed_running_job().model_copy(
        update={
            "status": JobStatus.FAILED,
            "updated_at": CONTROL_NOW,
            "finished_at": CONTROL_NOW,
            "failure_reason": "training process exited with status 2",
        }
    )
    transport.explanation = {
        "waiting": True,
        "code": "resource_probe_failed",
        "message": "GPU telemetry is unavailable",
        "observed_at": CONTROL_NOW.isoformat(),
        "details": {"error": "probe timed out"},
        "next_action": "Retry after telemetry recovers",
    }
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.get("/api/v1/jobs/running-job")

    assert response.status_code == 200
    detail = response.json()
    assert detail["raw_state"] == "failed"
    assert detail["failure_reason"] == "training process exited with status 2"
    assert detail["explanation"] == transport.explanation
    assert detail["logs"][-1] == "last-line"


def test_control_provider_preserves_known_job_and_logs_when_detail_fails() -> None:
    transport = FakeQueueTransport()
    transport.detail_error = ConnectionError("explanation RPC unavailable")
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.get("/api/v1/jobs/running-job")

    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == "running-job"
    assert detail["raw_state"] == "running"
    assert detail["logs"][-1] == "last-line"
    assert detail["explanation"]["code"] == "unknown"
    assert detail["explanation"]["observed_at"] == CONTROL_NOW.isoformat()
    assert "explanation RPC unavailable" in detail["detail_error"]


def test_job_lookup_is_unavailable_when_an_unsearched_host_is_offline() -> None:
    transport = FakeQueueTransport()
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.get("/api/v1/jobs/not-present")

    assert response.status_code == 503
    assert "gpu02" in response.json()["detail"]


def test_cancel_endpoint_uses_located_host_and_provider_protocol() -> None:
    transport = FakeQueueTransport()
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.post("/api/v1/jobs/queued-job/cancel")

    assert response.status_code == 200
    assert response.json() == {
        "id": "queued-job",
        "server": "gpu01",
        "state": "cancelled",
        "raw_state": "cancelled",
        "accepted": True,
    }
    assert transport.cancel_calls == [("gpu01", "queued-job")]


def test_cancel_endpoint_reports_unavailable_modes() -> None:
    with TestClient(create_app(DemoProvider())) as client:
        missing = client.post("/api/v1/jobs/not-present/cancel")
        unavailable = client.post("/api/v1/jobs/run-ranking-042/cancel")

    assert missing.status_code == 404
    assert unavailable.status_code == 503


def test_real_ssh_transport_exposes_provider_cancel_contract() -> None:
    assert callable(getattr(SshWorkerTransport, "cancel", None))


def test_detail_does_not_read_queue_or_invent_waiting_times() -> None:
    transport = FakeQueueTransport()

    async def unavailable_summaries(*args, **kwargs):
        raise AssertionError("details must not fetch the queue")

    transport.list_summaries = unavailable_summaries
    with TestClient(create_app(control_provider(transport))) as client:
        response = client.get("/api/v1/jobs/queued-job")

    assert response.status_code == 200
    detail = response.json()
    assert detail["estimated_start"] is None
    assert detail["estimated_end"] is None
    assert detail["detail_error"] is None
    assert detail["logs"][-1] == "last-line"


def test_process_estimates_have_a_separate_web_endpoint():
    with TestClient(create_app(DemoProvider())) as client:
        response = client.get("/api/v1/process-etas")
    assert response.status_code == 200
    assert response.json()["servers"] == []


def test_detail_log_eta_is_reused_by_queue_without_reading_more_logs():
    transport = FakeQueueTransport()
    provider = control_provider(transport, log_tail_bytes=4096)
    content = "epoch=1/5 seconds=120\nepoch=2/5 seconds=100\n"

    async def logs(target, run_id, *, tail_bytes):
        transport.log_calls.append((target.server_id, run_id, tail_bytes))
        return {
            "content": content,
            "modified_at": CONTROL_NOW.timestamp() - 30,
            "sampled_at": CONTROL_NOW.timestamp(),
            "utc_offset_seconds": 28800,
            "source": "/logs/run/console.log",
        }

    transport.get_logs = logs
    with TestClient(create_app(provider)) as client:
        detail = client.get("/api/v1/jobs/running-job").json()
        queue = client.get("/api/v1/queue").json()
        assert detail["eta"]["source_kind"] == "logs"
        assert queue["items"][0]["eta"] == detail["eta"]
        assert len(transport.log_calls) == 1
        provider.clock = lambda: CONTROL_NOW + timedelta(seconds=61)
        queue = client.get("/api/v1/queue").json()
        assert queue["items"][0]["eta"]["source_kind"] == "submitted"
        assert len(transport.log_calls) == 1
        provider.clock = lambda: CONTROL_NOW
        terminal = managed_running_job().model_copy(
            update={"status": JobStatus.SUCCEEDED, "finished_at": CONTROL_NOW}
        )
        transport.detail_record = terminal
        detail = client.get("/api/v1/jobs/running-job").json()
        assert detail["eta"]["source_kind"] == "actual"
        assert detail["estimated_end"] == CONTROL_NOW.isoformat()
        assert not provider._log_samples
