from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pytest

from gpuops.config import GPU_QUERY_TIMEOUT_SECONDS, MONITOR_TIMEOUT_SECONDS
from gpuops.monitor import MonitorService, RemoteCollector, ServerTarget
from gpuops.monitor.formatting import format_status_table
from gpuops.monitor.remote import COLLECTOR_SCRIPT, DOCTOR_SCRIPT
from gpuops.ssh import SSHResult, SSHTimeoutError


class FakeSSH:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def run_python(self, host, script, *, timeout_seconds):
        self.calls.append((host, script, timeout_seconds))
        response = self.responses[host]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return await response()
        return response


def ssh_result(host, payload, *, returncode=0, stderr="", duration=0.2):
    return SSHResult(
        host=host,
        returncode=returncode,
        stdout=json.dumps(payload) + "\n",
        stderr=stderr,
        duration_seconds=duration,
    )


def test_collector_accepts_one_json_line():
    payload = {"hostname": "gpu01", "gpus": []}
    ssh = FakeSSH({"lab-gpu01": ssh_result("lab-gpu01", payload)})
    collected, latency = asyncio.run(RemoteCollector(ssh).collect("lab-gpu01"))
    assert collected == payload
    assert latency == 0.2
    assert ssh.calls[0][0] == "lab-gpu01"
    assert "gpu-sample.json" in ssh.calls[0][1]
    assert "subprocess" not in ssh.calls[0][1]


def test_servers_are_collected_concurrently_and_fail_independently():
    gate = asyncio.Event()
    started = 0

    async def successful():
        nonlocal started
        started += 1
        if started == 2:
            gate.set()
        await asyncio.wait_for(gate.wait(), timeout=0.1)
        return ssh_result("gpu01", {"gpus": []})

    async def failed():
        nonlocal started
        started += 1
        if started == 2:
            gate.set()
        await asyncio.wait_for(gate.wait(), timeout=0.1)
        raise RuntimeError("connection refused")

    targets = (
        ServerTarget("gpu01", "GPU 01", "gpu01"),
        ServerTarget("gpu02", "GPU 02", "gpu02"),
    )
    service = MonitorService(
        targets,
        collector=RemoteCollector(FakeSSH({"gpu01": successful, "gpu02": failed})),
    )
    results = asyncio.run(service.collect_all())
    assert results[0].online is True
    assert results[1].online is False
    assert "connection refused" in results[1].error


def test_status_table_includes_process_details_and_offline_host():
    payload = {
        "gpus": [
            {
                "index": 0,
                "utilization.gpu": 91,
                "memory.used": 18000,
                "memory.total": 24576,
                "processes": [
                    {
                        "username": "researcher",
                        "pid": 123,
                        "cmdline": "python train.py --seed 42",
                        "elapsed_seconds": 3723,
                    }
                ],
            }
        ]
    }
    ssh = FakeSSH(
        {
            "gpu01": ssh_result("gpu01", payload),
            "gpu02": ssh_result("gpu02", {}, returncode=255, stderr="Connection refused"),
        }
    )
    service = MonitorService(
        (
            ServerTarget("gpu01", "GPU 01", "gpu01"),
            ServerTarget("gpu02", "GPU 02", "gpu02"),
        ),
        collector=RemoteCollector(ssh),
    )
    table = format_status_table(asyncio.run(service.collect_all()))
    assert "occupied" in table
    assert "researcher" in table
    assert "1h02m" in table
    assert "GPU 02" in table
    assert "offline" in table


def test_doctor_reports_worker_cache_health():
    payload = {
        "python": "3.10.12",
        "gpustat_path": "/usr/local/bin/gpustat",
        "gpustat_ok": True,
        "error": None,
        "error_code": None,
        "source": "worker_cache",
        "sampled_at": "2026-09-27T08:00:00+00:00",
    }
    ssh = FakeSSH({"gpu01": ssh_result("gpu01", payload)})
    service = MonitorService(
        (ServerTarget("gpu01", "GPU 01", "gpu01"),),
        collector=RemoteCollector(ssh),
    )
    result = asyncio.run(service.doctor_all())[0]
    assert result.online
    assert result.data["gpustat_ok"] is True
    assert "shutil.which" in ssh.calls[0][1]
    assert "subprocess" not in ssh.calls[0][1]
    assert result.data["source"] == "worker_cache"


def test_doctor_keeps_ssh_online_when_worker_cache_is_missing():
    payload = {
        "python": "3.10.12",
        "gpustat_path": None,
        "gpustat_ok": False,
        "error": "Worker telemetry cache is absent; install and start the GPUOps Worker",
        "error_code": "telemetry_unavailable",
        "source": "worker_cache",
        "sampled_at": None,
    }
    ssh = FakeSSH({"gpu01": ssh_result("gpu01", payload)})
    service = MonitorService(
        (ServerTarget("gpu01", "GPU 01", "gpu01"),),
        collector=RemoteCollector(ssh),
    )
    result = asyncio.run(service.doctor_all())[0]
    assert result.online
    assert result.data["gpustat_ok"] is False


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (ssh_result("gpu01", {}, returncode=255, stderr="Connection refused"), "connection_failed"),
        (ssh_result("gpu01", {}, returncode=1, stderr="python failed"), "collection_failed"),
        (SSHTimeoutError("remote operation timed out"), "collection_timeout"),
        (
            ssh_result(
                "gpu01",
                {
                    "collection_error": {
                        "code": "gpu_query_timeout",
                        "message": "gpustat exceeded 25s",
                    }
                },
            ),
            "gpu_query_timeout",
        ),
        (
            ssh_result(
                "gpu01",
                {
                    "collection_error": {
                        "code": "gpu_query_failed",
                        "message": "gpustat is unavailable",
                    }
                },
            ),
            "gpu_query_failed",
        ),
        (
            ssh_result(
                "gpu01",
                {
                    "collection_error": {
                        "code": "telemetry_unavailable",
                        "message": "start the GPUOps Worker",
                    }
                },
            ),
            "telemetry_unavailable",
        ),
        (
            ssh_result(
                "gpu01",
                {
                    "collection_error": {
                        "code": "telemetry_stale",
                        "message": "Worker telemetry sample is stale",
                    }
                },
            ),
            "telemetry_stale",
        ),
    ],
)
def test_collection_failure_categories_do_not_claim_valid_gpu_data(response, error_code):
    service = MonitorService(
        (ServerTarget("gpu01", "GPU 01", "gpu01"),),
        collector=RemoteCollector(FakeSSH({"gpu01": response})),
    )
    result = asyncio.run(service.collect_all())[0]
    assert result.online is False
    assert result.data is None
    assert result.error_code == result.as_dict()["error_code"] == error_code
    table = format_status_table([result])
    assert ("offline" if error_code == "connection_failed" else "unknown") in table


def _cache_document(
    sampled_at: datetime,
    *,
    healthy: bool = True,
    error: str | None = None,
    error_code: str | None = None,
) -> dict:
    timestamp = sampled_at.isoformat()
    return {
        "snapshot": {
            "server_id": "gpu01",
            "sampled_at": timestamp,
            "healthy": healthy,
            "error": error,
            "gpus": [],
        },
        "previous": None,
        "data": {"hostname": "gpu01", "gpus": [], "collected_at": timestamp}
        if healthy
        else None,
        "error_code": error_code,
    }


def _run_remote_script(script, tmp_path, monkeypatch, capsys, document=None):
    home = tmp_path / "home"
    cache = home / ".local/state/gpuops/gpu-sample.json"
    cache.parent.mkdir(parents=True)
    if document is not None:
        cache.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    exec(script, {})
    return json.loads(capsys.readouterr().out)


def test_remote_collector_reads_fresh_worker_cache_without_gpu_subprocess(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("remote monitor must not query a GPU subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    document = _cache_document(datetime.now(timezone.utc))
    payload = _run_remote_script(COLLECTOR_SCRIPT, tmp_path, monkeypatch, capsys, document)

    assert payload == document["data"]


def test_remote_collector_marks_only_cards_with_bad_sample_times_stale(
    tmp_path, monkeypatch, capsys
):
    now = datetime.now(timezone.utc)
    document = _cache_document(now)
    document["data"]["gpus"] = [
        {"index": 3, "sampled_at": now.isoformat(), "utilization.gpu": 0},
        {
            "index": 4,
            "sampled_at": (now - timedelta(minutes=1)).isoformat(),
            "utilization.gpu": 72,
        },
        {"index": 5, "sampled_at": "not-a-time", "memory.used": 2048},
        {"index": 6, "utilization.gpu": 0},
    ]

    payload = _run_remote_script(COLLECTOR_SCRIPT, tmp_path, monkeypatch, capsys, document)

    by_index = {gpu["index"]: gpu for gpu in payload["gpus"]}
    assert "error_code" not in by_index[3]
    assert by_index[4]["error_code"] == "telemetry_stale"
    assert by_index[4]["utilization.gpu"] == 72
    assert by_index[5]["error_code"] == "telemetry_stale"
    assert by_index[5]["memory.used"] == 2048
    assert "error_code" not in by_index[6]


@pytest.mark.parametrize(
    ("document", "error_code"),
    [
        (None, "telemetry_unavailable"),
        (_cache_document(datetime.now(timezone.utc) - timedelta(minutes=1)), "telemetry_stale"),
        (_cache_document(datetime.now(timezone.utc) + timedelta(minutes=1)), "telemetry_stale"),
        ({"snapshot": "wrong", "data": []}, "telemetry_unavailable"),
    ],
)
def test_remote_collector_rejects_absent_stale_future_and_wrong_cache(
    tmp_path, monkeypatch, capsys, document, error_code
):
    payload = _run_remote_script(COLLECTOR_SCRIPT, tmp_path, monkeypatch, capsys, document)
    assert payload["collection_error"]["code"] == error_code
    assert "gpus" not in payload
    if document is None:
        assert "start the GPUOps Worker" in payload["collection_error"]["message"]


def test_remote_collector_rejects_corrupt_cache(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    cache = home / ".local/state/gpuops/gpu-sample.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    exec(COLLECTOR_SCRIPT, {})
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_error"]["code"] == "telemetry_unavailable"


def test_remote_collector_rejects_fifo_without_blocking(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    cache = home / ".local/state/gpuops/gpu-sample.json"
    cache.parent.mkdir(parents=True)
    os.mkfifo(cache)
    monkeypatch.setenv("HOME", str(home))

    started = time.monotonic()
    exec(COLLECTOR_SCRIPT, {})
    elapsed = time.monotonic() - started

    payload = json.loads(capsys.readouterr().out)
    assert elapsed < 1
    assert payload["collection_error"]["code"] == "telemetry_unavailable"
    assert "regular file" in payload["collection_error"]["message"]


def test_remote_doctor_reflects_cached_probe_error(tmp_path, monkeypatch, capsys):
    sampled_at = datetime.now(timezone.utc)
    document = _cache_document(
        sampled_at,
        healthy=False,
        error="gpustat exceeded its deadline",
        error_code="gpu_query_timeout",
    )
    payload = _run_remote_script(DOCTOR_SCRIPT, tmp_path, monkeypatch, capsys, document)

    assert payload["source"] == "worker_cache"
    assert payload["sampled_at"] == sampled_at.isoformat()
    assert payload["cache_fresh"] is False
    assert payload["gpustat_ok"] is False
    assert payload["error_code"] == "gpu_query_timeout"
    assert payload["error"] == "gpustat exceeded its deadline"


def test_remote_doctor_reports_fresh_worker_cache(tmp_path, monkeypatch, capsys):
    sampled_at = datetime.now(timezone.utc)
    payload = _run_remote_script(
        DOCTOR_SCRIPT,
        tmp_path,
        monkeypatch,
        capsys,
        _cache_document(sampled_at),
    )

    assert payload["source"] == "worker_cache"
    assert payload["sampled_at"] == sampled_at.isoformat()
    assert payload["cache_fresh"] is True
    assert payload["gpustat_ok"] is True
    assert payload["error"] is None
    assert payload["error_code"] is None


def test_slow_success_uses_longer_outer_deadline_and_preserves_sample():
    payload = {"gpus": [], "collected_at": "2026-09-27T08:00:00+00:00"}
    ssh = FakeSSH({"gpu01": ssh_result("gpu01", payload, duration=17.0)})
    result = asyncio.run(
        MonitorService(
            (ServerTarget("gpu01", "GPU 01", "gpu01"),),
            collector=RemoteCollector(ssh),
        ).collect_all()
    )[0]
    assert result.online and result.error_code is None
    assert result.data == payload
    assert result.latency_seconds == 17.0
    assert ssh.calls[0][2] == MONITOR_TIMEOUT_SECONDS > GPU_QUERY_TIMEOUT_SECONDS
