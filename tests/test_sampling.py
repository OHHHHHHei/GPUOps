from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpuops.core.queue import QueueStore
from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.core.scheduler import Scheduler
from gpuops.models import JobSpec, JobStatus
from gpuops.worker import sampling
from gpuops.worker.probe import LocalGpuProbe
from gpuops.worker.sampling import AsyncGpuSampler, CachedGpuProbe

NOW = datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)


def _snapshot(sampled_at: datetime, *, healthy: bool = True) -> ServerResourceSnapshot:
    return ServerResourceSnapshot(
        server_id="gpu01",
        sampled_at=sampled_at,
        healthy=healthy,
        error=None if healthy else "failed",
        gpus=(
            [
                GpuResource(
                    index=0,
                    uuid="GPU-A",
                    memory_total_mib=24_576,
                    memory_used_mib=300,
                    utilization_percent=0,
                    process_count=0,
                    idle=True,
                    sampled_at=sampled_at,
                )
            ]
            if healthy
            else []
        ),
    )


def _write_cache(
    path: Path,
    latest: ServerResourceSnapshot,
    previous: ServerResourceSnapshot | None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "snapshot": latest.model_dump(mode="json"),
                "previous": previous.model_dump(mode="json") if previous else None,
                "data": (
                    {"gpus": [], "collected_at": latest.sampled_at.isoformat()}
                    if latest.healthy
                    else None
                ),
                "error_code": None if latest.healthy else "gpu_query_timeout",
            }
        ),
        encoding="utf-8",
    )


def _install_fake_gpustat(
    tmp_path: Path, monkeypatch, *, gated: bool = False
) -> tuple[Path, Path, Path]:
    executable = tmp_path / "gpustat"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, time\n"
        "with open(os.environ['GPUOPS_TEST_STARTED'], 'a', encoding='utf-8') as f:\n"
        " f.write('start\\n'); f.flush(); os.fsync(f.fileno())\n"
        "while not os.path.exists(os.environ['GPUOPS_TEST_RELEASE']): time.sleep(0.01)\n"
        "with open(os.environ['GPUOPS_TEST_CALLS'], 'a', encoding='utf-8') as f:\n"
        " f.write('call\\n'); f.flush(); os.fsync(f.fileno())\n"
        "print(json.dumps({'hostname':'test','gpus':[{'index':0,'uuid':'GPU-A',"
        "'name':'Test GPU','memory.total':24576,'memory.used':300,"
        "'utilization.gpu':0,'processes':[]}]}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    started = tmp_path / "started"
    release = tmp_path / "release"
    calls = tmp_path / "calls"
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GPUOPS_TEST_STARTED", str(started))
    monkeypatch.setenv("GPUOPS_TEST_RELEASE", str(release))
    monkeypatch.setenv("GPUOPS_TEST_CALLS", str(calls))
    if not gated:
        release.touch()
    return started, release, calls


def _wait_for_lines(path: Path, count: int, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text(encoding="utf-8").splitlines()) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"{path.name} did not reach {count} lines")


def _wait_for_process_exit(sampler: AsyncGpuSampler, timeout: float = 5) -> None:
    assert sampler._process is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sampler._process.poll() is not None:
            return
        time.sleep(0.01)
    raise AssertionError("sampler child did not exit")


def _tick_until_cached(sampler: AsyncGpuSampler, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sampler.tick()
        if sampler.cache_path.exists():
            return
        time.sleep(0.01)
    raise AssertionError("sampler did not publish its cache")


def test_local_probe_preserves_raw_gpustat_and_process_details() -> None:
    pid = os.getpid()
    payload = {
        "hostname": "test",
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-A",
                "name": "Test GPU",
                "memory.total": 24_576,
                "memory.used": 1024,
                "utilization.gpu": 10,
                "processes": [{"pid": pid, "command": "python", "custom": "kept"}],
                "custom.gpu": "kept",
            }
        ],
    }

    def runner(command, **kwargs):
        del kwargs
        from subprocess import CompletedProcess

        return CompletedProcess(command, 0, json.dumps(payload), "")

    sample = LocalGpuProbe("gpu01", runner=runner, clock=lambda: NOW).sample()

    assert sample.snapshot.healthy
    assert sample.data is not None
    assert sample.data["collected_at"] == sample.snapshot.sampled_at.isoformat()
    raw_gpu = sample.data["gpus"][0]
    assert raw_gpu["custom.gpu"] == "kept"
    process = raw_gpu["processes"][0]
    assert process["custom"] == "kept"
    assert "start_time_ticks" in process
    assert "elapsed_seconds" in process
    # Linux enriches the current process from /proc; other platforms retain
    # gpustat's fallback command. Either way the raw custom fields survive.
    assert process["cmdline"]


def test_process_details_are_read_from_procfs(tmp_path: Path) -> None:
    pid = 123
    process_root = tmp_path / str(pid)
    process_root.mkdir()
    (process_root / "cmdline").write_bytes(b"python\0train.py\0")
    fields = ["S", "1", "123"] + ["0"] * 16 + ["250"] + ["0"] * 5
    (process_root / "stat").write_text(
        f"{pid} (python worker) {' '.join(fields)}\n", encoding="utf-8"
    )
    (tmp_path / "uptime").write_text("10.0 2.0\n", encoding="utf-8")
    process = {"pid": pid, "command": "python"}

    LocalGpuProbe._enrich_process(process, tmp_path)

    assert process["cmdline"] == "python train.py"
    assert process["ppid"] == 1
    assert process["pgrp"] == 123
    assert process["start_time_ticks"] == 250
    assert process["elapsed_seconds"] is not None


def test_nvidia_fallback_raw_payload_contains_every_observed_pid() -> None:
    pid = os.getpid()

    def runner(command, **kwargs):
        del kwargs
        from subprocess import CompletedProcess

        if command[0] == "gpustat":
            raise FileNotFoundError("missing")
        if "--query-gpu" in command[1]:
            return CompletedProcess(command, 0, "0, GPU-A, 24576, 1200, 20\n", "")
        return CompletedProcess(command, 0, f"GPU-A, {pid}\n", "")

    sample = LocalGpuProbe("gpu01", runner=runner, clock=lambda: NOW).sample()

    assert sample.snapshot.gpus[0].process_count == 1
    assert sample.data is not None
    assert [item["pid"] for item in sample.data["gpus"][0]["processes"]] == [pid]


def test_cached_probe_rejects_bad_cache_and_accepts_historical_confirmation(
    tmp_path: Path,
) -> None:
    probe = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW)
    assert probe.snapshot().healthy is False

    first = _snapshot(NOW - timedelta(seconds=30))
    latest = _snapshot(NOW - timedelta(seconds=5))
    _write_cache(tmp_path / "gpu-sample.json", latest, first)

    assert probe.snapshot() == latest
    assert probe.confirmation_samples() == (first, latest)

    wrong_host = latest.model_copy(update={"server_id": "gpu02"})
    _write_cache(tmp_path / "gpu-sample.json", wrong_host, first)
    assert probe.snapshot().healthy is False
    assert probe.confirmation_samples() is None

    future = latest.model_copy(update={"sampled_at": NOW + timedelta(milliseconds=1)})
    _write_cache(tmp_path / "gpu-sample.json", future, first)
    assert probe.snapshot().healthy is False

    timeout = _snapshot(NOW - timedelta(seconds=1), healthy=False)
    _write_cache(tmp_path / "gpu-sample.json", timeout, None)
    assert probe.snapshot() == timeout


def test_confirmation_accepts_fast_publication_with_distinct_device_samples(
    tmp_path: Path,
) -> None:
    first = _snapshot(NOW - timedelta(seconds=6))
    latest = _snapshot(NOW - timedelta(seconds=5))
    first.gpus[0].sampled_at = NOW - timedelta(seconds=10)
    latest.gpus[0].sampled_at = NOW - timedelta(seconds=5)
    _write_cache(tmp_path / "gpu-sample.json", latest, first)

    pair = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW).confirmation_samples()

    assert pair == (first, latest)


def test_staggered_device_samples_keep_independent_confirmation_history(
    tmp_path: Path,
) -> None:
    devices = [_device(4), _device(6)]
    now = [NOW]
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        clock=lambda: now[0],
        device_discovery=lambda: devices,
    )
    for device in devices:
        sampler._devices[sampler._device_key(device)] = sampling._DeviceState(
            device=device,
            latest=_successful_result(device, NOW - timedelta(seconds=10), "first"),
        )
    sampler._publish()

    now[0] += timedelta(seconds=1)
    device = devices[0]
    sampler._devices[sampler._device_key(device)].latest = _successful_result(
        device, NOW - timedelta(seconds=5), "second"
    )
    sampler._publish()

    now[0] += timedelta(seconds=1)
    device = devices[1]
    sampler._devices[sampler._device_key(device)].latest = _successful_result(
        device, NOW - timedelta(seconds=5), "second"
    )
    sampler._publish()

    first, latest = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: now[0]).confirmation_samples()
    assert [gpu.sampled_at for gpu in first.gpus] == [NOW - timedelta(seconds=10)] * 2
    assert [gpu.sampled_at for gpu in latest.gpus] == [NOW - timedelta(seconds=5)] * 2


def test_cached_probe_rejects_fifo_and_oversized_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "gpu-sample.json"
    os.mkfifo(cache_path)
    probe = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW)

    assert probe.snapshot().healthy is False
    assert probe.confirmation_samples() is None

    cache_path.unlink()
    cache_path.write_bytes(b"{" + b" " * (2 * 1024 * 1024))
    assert probe.snapshot().healthy is False
    assert probe.confirmation_samples() is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data", None),
        ("data", {"gpus": {}, "collected_at": NOW.isoformat()}),
        (
            "data",
            {"gpus": [], "collected_at": (NOW - timedelta(seconds=1)).isoformat()},
        ),
        ("error_code", "gpu_query_failed"),
    ],
)
def test_cached_probe_rejects_inconsistent_healthy_envelope(
    tmp_path: Path, field: str, value: object
) -> None:
    latest = _snapshot(NOW)
    previous = _snapshot(NOW - timedelta(seconds=5))
    envelope = {
        "snapshot": latest.model_dump(mode="json"),
        "previous": previous.model_dump(mode="json"),
        "data": {"gpus": [], "collected_at": latest.sampled_at.isoformat()},
        "error_code": None,
    }
    envelope[field] = value
    (tmp_path / "gpu-sample.json").write_text(json.dumps(envelope), encoding="utf-8")
    probe = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW)

    assert probe.snapshot().healthy is False
    assert probe.confirmation_samples() is None


class _FakeProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


def _device(index: int) -> sampling.GpuDevice:
    return sampling.GpuDevice(index=index, uuid=f"GPU-{index}", pci_bus=f"0000:0{index}:00.0")


def _successful_result(device: sampling.GpuDevice, sampled_at: datetime, attempt: str) -> dict:
    snapshot = ServerResourceSnapshot(
        server_id="gpu01",
        sampled_at=sampled_at,
        gpus=[
            GpuResource(
                index=device.index,
                uuid=device.uuid,
                memory_total_mib=24_576,
                memory_used_mib=300,
                utilization_percent=0,
                process_count=0,
                idle=True,
                sampled_at=sampled_at,
            )
        ],
    )
    return {
        "attempt_id": attempt,
        "device": {
            "index": device.index,
            "uuid": device.uuid,
            "pci_bus": device.pci_bus,
        },
        "snapshot": snapshot.model_dump(mode="json"),
        "data": {
            "gpus": [
                {
                    "index": device.index,
                    "uuid": device.uuid,
                    "pci.bus_id": device.pci_bus,
                    "memory.total": 24_576,
                    "memory.used": 300,
                    "utilization.gpu": 0,
                    "processes": [],
                }
            ],
            "collected_at": sampled_at.isoformat(),
        },
        "error_code": None,
    }


def _failed_result(
    device: sampling.GpuDevice, sampled_at: datetime, attempt: str, code: str
) -> dict:
    snapshot = ServerResourceSnapshot(
        server_id="gpu01", sampled_at=sampled_at, healthy=False,
        error="device is unavailable", gpus=[],
    )
    return {
        "attempt_id": attempt,
        "device": {
            "index": device.index, "uuid": device.uuid, "pci_bus": device.pci_bus,
        },
        "snapshot": snapshot.model_dump(mode="json"),
        "data": None,
        "error": snapshot.error,
        "error_code": code,
    }


def test_hung_card_does_not_block_other_cards_or_publication(tmp_path: Path, monkeypatch) -> None:
    devices = [_device(4), _device(5), _device(6)]
    monotonic = [0.0]
    spawned: dict[str, _FakeProcess] = {}

    def spawn(argv, **kwargs):
        del kwargs
        pci = argv[argv.index("--pci-bus") + 1]
        attempt = argv[argv.index("--attempt-id") + 1]
        device = next(item for item in devices if item.pci_bus == pci)
        process = _FakeProcess(None if device.index == 6 else 0)
        spawned[pci] = process
        if device.index != 6:
            result = _successful_result(device, NOW, attempt)
            path = Path(argv[argv.index("--result") + 1])
            path.write_text(json.dumps(result), encoding="utf-8")
        return process

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    monkeypatch.setattr(sampling, "PARENT_SAMPLE_TIMEOUT_SECONDS", 0.5)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW,
        device_discovery=lambda: devices,
    )

    sampler.tick()
    assert len(sampler.processes) == 3
    monotonic[0] = 0.1
    sampler.tick()
    cache = json.loads(sampler.cache_path.read_text())
    assert [gpu["error"] for gpu in cache["snapshot"]["gpus"]] == [
        None,
        None,
        "GPU sample has not completed",
    ]
    monotonic[0] = 1.0
    sampler.tick()
    cache = json.loads(sampler.cache_path.read_text())
    assert cache["snapshot"]["healthy"] is True
    assert cache["snapshot"]["gpus"][2]["error"] == "GPU sampling deadline exhausted"
    assert cache["data"]["gpus"][0]["sampled_at"] == NOW.isoformat()
    assert spawned[devices[6 - 4].pci_bus].poll() is None


def test_empty_inventory_fails_closed_but_known_pending_card_is_host_healthy(
    tmp_path: Path, monkeypatch
) -> None:
    empty = AsyncGpuSampler(tmp_path / "empty", "gpu01", device_discovery=lambda: [])
    empty.tick()
    empty_cache = json.loads(empty.cache_path.read_text())
    assert empty_cache["snapshot"]["healthy"] is False
    assert empty_cache["snapshot"]["error"] == (
        "GPU discovery failed: no NVIDIA devices discovered"
    )
    assert empty_cache["error_code"] == "gpu_discovery_failed"
    assert empty_cache["data"] is None

    monkeypatch.setattr(
        sampling.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )
    known = AsyncGpuSampler(
        tmp_path / "known",
        "gpu01",
        clock=lambda: NOW,
        device_discovery=lambda: [_device(4)],
    )
    known.tick()
    known_cache = json.loads(known.cache_path.read_text())
    assert known_cache["snapshot"]["healthy"] is True
    assert known_cache["snapshot"]["error"] is None
    assert known_cache["snapshot"]["gpus"][0]["idle"] is False
    assert known_cache["data"]["gpus"][0]["error_code"] == "gpu_sample_pending"


@pytest.mark.parametrize("contents", ['{"GPU-6": "old maintenance note"}', "not-json"])
def test_legacy_quarantine_file_does_not_block_sampling(
    tmp_path: Path, monkeypatch, contents: str
) -> None:
    devices = [_device(4), _device(6)]
    (tmp_path / "gpu-quarantine.json").write_text(contents, encoding="utf-8")
    calls = []

    def spawn(argv, **kwargs):
        del kwargs
        calls.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        clock=lambda: NOW,
        device_discovery=lambda: devices,
    )
    sampler.tick()

    assert {argv[argv.index("--uuid") + 1] for argv in calls} == {
        device.uuid for device in devices
    }
    cache = json.loads(sampler.cache_path.read_text())
    assert cache["snapshot"]["healthy"] is True
    assert [gpu["error_code"] for gpu in cache["data"]["gpus"]] == [
        "gpu_sample_pending", "gpu_sample_pending"
    ]


@pytest.mark.parametrize("error_code", ["gpu_device_unavailable", "gpu_cuda_unavailable"])
def test_device_error_recovers_without_blocking_sibling_or_early_lease(
    tmp_path: Path, monkeypatch, error_code: str
) -> None:
    devices = [_device(4), _device(6)]
    monotonic = [0.0]
    starts: dict[int, int] = {4: 0, 6: 0}

    def spawn(argv, **kwargs):
        del kwargs
        index = int(argv[argv.index("--index") + 1])
        device = next(item for item in devices if item.index == index)
        starts[index] += 1
        attempt = argv[argv.index("--attempt-id") + 1]
        path = Path(argv[argv.index("--result") + 1])
        sampled_at = NOW + timedelta(seconds=monotonic[0])
        result = (
            _failed_result(device, sampled_at, attempt, error_code)
            if index == 4 and starts[index] == 1
            else _successful_result(device, sampled_at, attempt)
        )
        path.write_text(json.dumps(result), encoding="utf-8")
        return _FakeProcess(0)

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW + timedelta(seconds=monotonic[0]),
        device_discovery=lambda: devices,
    )
    sampler.tick()
    monotonic[0] = 0.1
    sampler.tick()
    cache = json.loads(sampler.cache_path.read_text())
    assert cache["data"]["gpus"][0]["error_code"] == error_code
    assert cache["snapshot"]["gpus"][0]["idle"] is False
    assert cache["data"]["gpus"][1]["error_code"] is None
    assert cache["snapshot"]["gpus"][1]["idle"] is True

    store = QueueStore(tmp_path / "queue")
    queued = store.enqueue(JobSpec(
        id="after-recovery", project="tests", command=["python", "train.py"],
        working_directory=str(tmp_path),
        resources={"host": "gpu01", "gpu_count": 1, "gpus": [4]},
    ), now=NOW)
    probe = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW + timedelta(seconds=monotonic[0]))
    scheduler = Scheduler(
        store, probe, server_id="gpu01", worker_id="worker-1",
        clock=lambda: NOW + timedelta(seconds=monotonic[0]),
        confirmation_provider=probe,
    )

    monotonic[0] = 5.2
    sampler.tick()
    monotonic[0] = 5.3
    sampler.tick()
    recovered = json.loads(sampler.cache_path.read_text())
    assert recovered["data"]["gpus"][0]["error_code"] is None
    assert recovered["snapshot"]["gpus"][0]["idle"] is True
    assert scheduler.tick() == []
    assert store.get(queued.run_id).status == JobStatus.QUEUED

    monotonic[0] = 10.4
    sampler.tick()
    monotonic[0] = 10.5
    sampler.tick()
    assert [decision.run_id for decision in scheduler.tick()] == [queued.run_id]
    assert starts[4] == 3
    assert starts[6] >= 2


def test_timed_out_device_never_starts_overlapping_probe(
    tmp_path: Path, monkeypatch
) -> None:
    devices = [_device(4), _device(6)]
    monotonic = [0.0]
    bad_processes: list[_FakeProcess] = []
    healthy_starts = [0]

    def spawn(argv, **kwargs):
        del kwargs
        index = int(argv[argv.index("--index") + 1])
        if index == 4:
            process = _FakeProcess()
            bad_processes.append(process)
            return process
        healthy_starts[0] += 1
        attempt = argv[argv.index("--attempt-id") + 1]
        path = Path(argv[argv.index("--result") + 1])
        path.write_text(json.dumps(_successful_result(
            devices[1], NOW + timedelta(seconds=monotonic[0]), attempt,
        )), encoding="utf-8")
        return _FakeProcess(0)

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    monkeypatch.setattr(sampling, "PARENT_SAMPLE_TIMEOUT_SECONDS", 0.5)
    sampler = AsyncGpuSampler(
        tmp_path, "gpu01", monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW + timedelta(seconds=monotonic[0]),
        device_discovery=lambda: devices,
    )
    sampler.tick()
    monotonic[0] = 0.6
    sampler.tick()
    assert len(bad_processes) == 1
    assert json.loads(sampler.cache_path.read_text())["data"]["gpus"][0][
        "error_code"
    ] == "gpu_query_timeout"

    monotonic[0] = 20
    sampler.tick()
    assert len(bad_processes) == 1
    assert healthy_starts[0] >= 2
    monotonic[0] = 20.1
    sampler.tick()
    cache = json.loads(sampler.cache_path.read_text())
    assert cache["data"]["gpus"][1]["error_code"] is None
    assert cache["snapshot"]["gpus"][1]["idle"] is True

    bad_processes[0].return_code = 0
    sampler.tick()
    state = sampler._devices[sampler._device_key(devices[0])]
    assert len(bad_processes) == 1
    assert state.next_attempt_at == pytest.approx(80.1)
    assert state.timeout_backoff == 120
    assert json.loads(sampler.cache_path.read_text())["data"]["gpus"][0][
        "error_code"
    ] == "gpu_query_timeout"
    monotonic[0] = 80
    sampler.tick()
    assert len(bad_processes) == 1
    monotonic[0] = 80.1
    sampler.tick()
    assert len(bad_processes) == 2
    monotonic[0] = 81
    sampler.tick()
    assert state.timed_out is True
    bad_processes[1].return_code = 0
    monotonic[0] = 120
    sampler.tick()
    assert len(bad_processes) == 2
    assert state.next_attempt_at == pytest.approx(240)
    assert state.timeout_backoff == 240
    monotonic[0] = 239.9
    sampler.tick()
    assert len(bad_processes) == 2
    monotonic[0] = 240
    sampler.tick()
    assert len(bad_processes) == 3


def test_inherited_per_device_lock_blocks_duplicate_after_restart(tmp_path: Path) -> None:
    device = _device(4)
    sampler = AsyncGpuSampler(tmp_path, "gpu01", device_discovery=lambda: [device])
    tmp_path.mkdir(exist_ok=True)
    lock_fd = os.open(sampler._device_lock_path(device), os.O_RDWR | os.O_CREAT, 0o600)
    import fcntl

    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        pass_fds=(lock_fd,),
    )
    os.close(lock_fd)
    try:
        sampler.tick()
        assert sampler.processes == ()
    finally:
        orphan.terminate()
        orphan.wait(timeout=5)


def test_legacy_host_query_barrier_blocks_new_children(tmp_path: Path, monkeypatch) -> None:
    import fcntl

    device = _device(4)
    tmp_path.mkdir(exist_ok=True)
    barrier_fd = os.open(tmp_path / sampling.LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(barrier_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls = []
    monkeypatch.setattr(sampling.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    sampler = AsyncGpuSampler(tmp_path, "gpu01", device_discovery=lambda: [device])
    try:
        sampler.tick()
        assert calls == []
        assert sampler.processes == ()
    finally:
        os.close(barrier_fd)


def test_equivalent_pci_formats_share_one_lock_key(tmp_path: Path) -> None:
    sampler = AsyncGpuSampler(tmp_path, "gpu01", device_discovery=lambda: [])
    short = sampling.GpuDevice(index=4, uuid="GPU-4", pci_bus="0000:81:00.0")
    long = sampling.GpuDevice(index=4, uuid="GPU-4", pci_bus="00000000:81:00.0")

    assert sampler._device_lock_path(short) == sampler._device_lock_path(long)
    assert sampler._result_path(short) == sampler._result_path(long)


def test_removed_device_child_is_retained_until_reaped(tmp_path: Path, monkeypatch) -> None:
    device = _device(4)
    inventory = [[device]]
    monotonic = [0.0]
    child = _FakeProcess()
    monkeypatch.setattr(sampling.subprocess, "Popen", lambda *args, **kwargs: child)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        monotonic_clock=lambda: monotonic[0],
        device_discovery=lambda: inventory[-1],
    )
    sampler.tick()
    inventory.append([])
    monotonic[0] = 31
    sampler.tick()

    assert sampler.processes == ()
    assert sampler._retired_processes == [child]
    child.return_code = 0
    sampler.tick()
    assert sampler._retired_processes == []


def test_result_without_real_per_device_timestamp_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    device = _device(4)
    process = _FakeProcess()
    monkeypatch.setattr(sampling.subprocess, "Popen", lambda *args, **kwargs: process)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        clock=lambda: NOW,
        device_discovery=lambda: [device],
    )
    sampler.tick()
    state = next(iter(sampler._devices.values()))
    result = _successful_result(device, NOW, state.attempt_id or "")
    result["snapshot"]["gpus"][0]["sampled_at"] = None
    sampler._result_path(device).write_text(json.dumps(result), encoding="utf-8")
    process.return_code = 0

    sampler.tick()

    assert state.latest["error_code"] == "gpu_query_failed"
    assert state.latest["error"] == "GPU sampler child exited 0"


def test_stale_card_metrics_are_masked_without_retimestamping(tmp_path: Path) -> None:
    sampled_at = NOW - timedelta(seconds=20)
    latest = _snapshot(NOW - timedelta(seconds=1))
    latest.gpus[0].sampled_at = sampled_at
    _write_cache(tmp_path / "gpu-sample.json", latest, None)

    snapshot = CachedGpuProbe(tmp_path, "gpu01", clock=lambda: NOW).snapshot()

    gpu = snapshot.gpus[0]
    assert gpu.sampled_at == sampled_at
    assert gpu.idle is False
    assert gpu.memory_total_mib is None
    assert gpu.memory_used_mib is None
    assert gpu.utilization_percent is None
    assert gpu.process_count is None
    assert gpu.error == "GPU sample is stale; waiting for a new sample"


def test_failed_card_recovers_and_resets_backoff(tmp_path: Path, monkeypatch) -> None:
    device = _device(4)
    monotonic = [0.0]
    outcomes = [1, 0]

    def spawn(argv, **kwargs):
        del kwargs
        return_code = outcomes.pop(0)
        if return_code == 0:
            attempt = argv[argv.index("--attempt-id") + 1]
            path = Path(argv[argv.index("--result") + 1])
            path.write_text(json.dumps(_successful_result(device, NOW, attempt)), encoding="utf-8")
        return _FakeProcess(return_code)

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW,
        device_discovery=lambda: [device],
    )
    sampler.tick()
    monotonic[0] = 0.1
    sampler.tick()
    state = next(iter(sampler._devices.values()))
    assert state.latest["error_code"] == "gpu_query_failed"
    assert state.next_attempt_at == pytest.approx(5.1)
    assert state.failure_backoff == 10
    monotonic[0] = 5.2
    sampler.tick()
    monotonic[0] = 5.3
    sampler.tick()
    assert state.latest["error_code"] is None
    assert state.failure_backoff == 5
    assert state.next_attempt_at == pytest.approx(10.3)


def test_child_reported_timeout_cools_down_then_success_resets_backoff(
    tmp_path: Path, monkeypatch
) -> None:
    device = _device(6)
    monotonic = [0.0]
    outcomes = ["gpu_query_timeout", None, "gpu_query_timeout"]
    starts = []

    def spawn(argv, **kwargs):
        del kwargs
        starts.append(monotonic[0])
        attempt = argv[argv.index("--attempt-id") + 1]
        path = Path(argv[argv.index("--result") + 1])
        sampled_at = NOW + timedelta(seconds=monotonic[0])
        code = outcomes.pop(0)
        result = (
            _failed_result(device, sampled_at, attempt, code)
            if code is not None
            else _successful_result(device, sampled_at, attempt)
        )
        path.write_text(json.dumps(result), encoding="utf-8")
        return _FakeProcess(0)

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    sampler = AsyncGpuSampler(
        tmp_path, "gpu01", monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW + timedelta(seconds=monotonic[0]),
        device_discovery=lambda: [device],
    )
    sampler.tick()
    monotonic[0] = 0.1
    sampler.tick()
    state = next(iter(sampler._devices.values()))
    assert state.latest["error_code"] == "gpu_query_timeout"
    assert state.next_attempt_at == pytest.approx(60.1)
    assert state.timeout_backoff == 120

    monotonic[0] = 60
    sampler.tick()
    assert starts == [0.0]
    monotonic[0] = 60.1
    sampler.tick()
    monotonic[0] = 60.2
    sampler.tick()
    assert state.latest["error_code"] is None
    assert state.timeout_backoff == 60
    assert state.next_attempt_at == pytest.approx(65.2)

    monotonic[0] = 65.2
    sampler.tick()
    monotonic[0] = 65.3
    sampler.tick()
    assert starts == [0.0, 60.1, 65.2]
    assert state.latest["error_code"] == "gpu_query_timeout"
    assert state.next_attempt_at == pytest.approx(125.3)


def test_correlated_timeout_retries_are_spread_across_ticks(
    tmp_path: Path, monkeypatch
) -> None:
    devices = [_device(4), _device(5), _device(6)]
    monotonic = [0.0]
    starts: list[tuple[int, float]] = []

    def spawn(argv, **kwargs):
        del kwargs
        index = int(argv[argv.index("--index") + 1])
        device = next(item for item in devices if item.index == index)
        first_attempt = all(started != index for started, _ in starts)
        starts.append((index, monotonic[0]))
        attempt = argv[argv.index("--attempt-id") + 1]
        sampled_at = NOW + timedelta(seconds=monotonic[0])
        result = (
            _failed_result(device, sampled_at, attempt, "gpu_query_timeout")
            if index in {4, 5} and first_attempt
            else _successful_result(device, sampled_at, attempt)
        )
        Path(argv[argv.index("--result") + 1]).write_text(
            json.dumps(result), encoding="utf-8"
        )
        return _FakeProcess(0)

    monkeypatch.setattr(sampling.subprocess, "Popen", spawn)
    sampler = AsyncGpuSampler(
        tmp_path, "gpu01", monotonic_clock=lambda: monotonic[0],
        clock=lambda: NOW + timedelta(seconds=monotonic[0]),
        device_discovery=lambda: devices,
    )
    sampler.tick()
    assert starts == [(4, 0.0), (5, 0.0), (6, 0.0)]
    monotonic[0] = 0.1
    sampler.tick()
    assert all(
        state.next_attempt_at == pytest.approx(60.1)
        for state in sampler._devices.values()
        if state.device.index in {4, 5}
    )

    monotonic[0] = 60.1
    sampler.tick()
    assert starts[-2:] == [(6, 60.1), (4, 60.1)]
    assert len(sampler.processes) == 2
    monotonic[0] = 60.2
    sampler.tick()
    assert starts[-1] == (5, 60.2)
    assert len(sampler.processes) == 1
    monotonic[0] = 60.3
    sampler.tick()
    cache = json.loads(sampler.cache_path.read_text())
    assert all(gpu["idle"] for gpu in cache["snapshot"]["gpus"])


def test_failure_backoff_is_bounded_at_sixty_seconds(tmp_path: Path, monkeypatch) -> None:
    device = _device(4)
    monotonic = [0.0]
    monkeypatch.setattr(sampling.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess(1))
    sampler = AsyncGpuSampler(
        tmp_path,
        "gpu01",
        monotonic_clock=lambda: monotonic[0],
        device_discovery=lambda: [device],
    )
    sampler.tick()
    state = next(iter(sampler._devices.values()))
    for _ in range(8):
        monotonic[0] += 0.1
        sampler.tick()
        monotonic[0] = state.next_attempt_at
        sampler.tick()
    assert state.failure_backoff == 60
    assert state.next_attempt_at - monotonic[0] <= 60
