"""Exercise queue control while an actual sampling child is still blocked."""

import json
import os
import subprocess
import sys
import time
from datetime import timedelta

from gpuops.core.queue import QueueStore
from gpuops.core.scheduler import Scheduler
from gpuops.models import JobSpec, JobStatus, LaunchInfo, utc_now
from gpuops.worker import sampling
from gpuops.worker.app import worker_readiness
from gpuops.worker.rpc import WorkerRpc
from gpuops.worker.runtime import ProcessIdentity, RuntimePaths, WorkerRuntime, inspect_worker
from gpuops.worker.sampling import AsyncGpuSampler, CachedGpuProbe
from gpuops.worker.service import Worker


def test_slow_sample_does_not_block_control_or_grant_resources(tmp_path, monkeypatch):
    pci_root = tmp_path / "pci"
    pci_device = pci_root / "0000:01:00.0"
    pci_device.mkdir(parents=True)
    (pci_device / "config").write_bytes(bytes.fromhex("de 10 30 22"))
    (tmp_path / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "from gpuops.worker import probe\n"
        "original_check = probe.check_pci_config\n"
        "probe.check_pci_config = lambda device: "
        f"original_check(device, Path({str(pci_root)!r}))\n",
        encoding="utf-8",
    )
    nvidia_smi = tmp_path / "nvidia-smi"
    release = tmp_path / "release-query"
    nvidia_smi.write_text(
        "#!/usr/bin/env python3\nimport sys, time\nfrom pathlib import Path\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "if any('--query-gpu=' in arg for arg in sys.argv):\n"
        " print('0, GPU-test, 0000:01:00.0, 24576, 0, 0, Test GPU, 40, 50, 250')\n"
    )
    nvidia_smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setattr(sampling, "PARENT_SAMPLE_TIMEOUT_SECONDS", 0.05)
    state = tmp_path / "state"
    queue = QueueStore(state / "queue")
    probe = CachedGpuProbe(state, "gpu01")
    device = sampling.GpuDevice(index=0, uuid="GPU-test", pci_bus="0000:01:00.0")
    sampler = AsyncGpuSampler(state, "gpu01", device_discovery=lambda: [device])
    identity = ProcessIdentity(os.getpid(), 1, "test-boot")
    paths = RuntimePaths.from_state_directory(state)

    def inspect(paths):
        return inspect_worker(paths, identity_reader=lambda pid: identity)

    rpc = WorkerRpc(state, server_id="gpu01", queue=queue, worker_inspector=inspect)
    spec = JobSpec(
        id="control-canary",
        project="test",
        command=["must-not-run"],
        working_directory=str(tmp_path),
        resources={"host": "gpu01", "gpu_count": 1},
    )
    for run_id, gpu in (("completed", 0), ("cancel-running", 1)):
        record = queue.enqueue(spec, run_id=run_id)
        queue.lease(
            run_id,
            server_id="gpu01",
            gpu_indices=[gpu],
            worker_id="test-worker",
            ttl=timedelta(minutes=1),
            expected_revision=record.revision,
        )
        queue.transition(run_id, JobStatus.STARTING)
        queue.transition(
            run_id, JobStatus.RUNNING, launch=LaunchInfo(pid=123, started_at=utc_now())
        )
    queue.request_cancel("cancel-running")

    class Launcher:
        def start(self, record):
            raise AssertionError("unavailable GPU evidence must not start a job")

        def poll(self, record):
            return 0

        def cancel(self, record):
            return True

    worker = Worker(
        queue,
        Scheduler(
            queue, probe, server_id="gpu01", worker_id="test-worker", confirmation_provider=probe
        ),
        Launcher(),
    )
    heartbeats = []
    children = []

    def between_ticks(_seconds):
        assert worker_readiness(inspect(paths), server_id="gpu01").ready
        assert len(sampler.processes) == 1 and sampler.processes[0].poll() is None
        children.append(sampler.processes[0])
        heartbeats.append(json.loads(paths.heartbeat_path.read_text())["iteration"])
        if len(heartbeats) == 1:
            for run_id in ("keep-queued", "cancel-queued"):
                assert rpc.submit({"run_id": run_id, "job_spec": spec.model_dump(mode="json")})[
                    "durable"
                ]
            rpc.cancel({"run_id": "cancel-queued"})
        assert rpc.health({})["worker"]["ready"] is True
        detail = rpc.show({"run_id": "keep-queued", "explain": True})
        assert detail["explanation"]["code"] == "unknown"
        time.sleep(0.03)

    runtime = WorkerRuntime(
        server_id="gpu01",
        worker_id="test-worker",
        paths=paths,
        tick=worker.tick,
        identity_reader=lambda pid: identity,
        sampler_factory=lambda: sampler,
        sleeper=between_ticks,
    )
    try:
        runtime.run_foreground(max_iterations=5, install_signal_handlers=False)
        assert heartbeats == [1, 2, 3, 4]
        assert queue.get("keep-queued").status == JobStatus.QUEUED
        assert queue.get("cancel-queued").status == JobStatus.CANCELLED
        assert queue.get("cancel-running").status == JobStatus.CANCELLED
        assert queue.get("completed").status == JobStatus.SUCCEEDED
        cache = json.loads(sampler.cache_path.read_text())
        assert cache["error_code"] is None
        assert cache["data"]["gpus"][0]["error_code"] == "gpu_query_timeout"
    finally:
        release.touch()
        for child in set(children):
            child.wait(timeout=5)


def test_sampling_child_does_not_import_pydantic():
    # The child starts every few seconds per GPU; pydantic would cost ~0.1 s each.
    check = (
        "import sys, gpuops.worker.sampling_child; "
        "sys.exit(any(name.startswith('pydantic') for name in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", check], check=False).returncode == 0


def test_sampling_child_writes_a_result_the_worker_accepts(tmp_path, monkeypatch):
    from gpuops.worker import probe, sampling_child
    from gpuops.worker.devices import GpuDevice

    def run(command, **kwargs):
        if "--query-gpu" in command[3]:
            stdout = "0, GPU-0, 00000000:01:00.0, 24576, 300, 0, GPU, 40, 50, 250\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(sampling_child.subprocess, "run", run)
    monkeypatch.setattr(probe, "check_pci_config", lambda _device: None)
    result = tmp_path / "result.json"
    with (tmp_path / "lock").open("w") as lock:
        sampling_child.run_sample(
            server_id="gpu01",
            result_path=result,
            lock_fd=lock.fileno(),
            attempt_id="attempt",
            device=GpuDevice(index=0, uuid="GPU-0", pci_bus="0000:01:00.0"),
        )

    envelope = json.loads(result.read_text())
    snapshot = sampling._validated_current_snapshot(envelope, "gpu01")
    assert snapshot is not None and snapshot.healthy
    assert snapshot.gpus[0].idle and snapshot.gpus[0].memory_used_mib == 300
