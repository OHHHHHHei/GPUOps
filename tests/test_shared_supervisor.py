import os
import signal
import sys
import time
from pathlib import Path

import pytest

from gpuops.config import SchedulerSettings
from gpuops.core.queue import QueueStore
from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.core.scheduler import Scheduler
from gpuops.models import JobSpec, JobStatus, utc_now
from gpuops.worker.service import Worker
from gpuops.worker.supervisor import (
    DurableJobSupervisor,
    DurableSupervisorLauncher,
    SupervisorPaths,
)


class OneCardProbe:
    def snapshot(self) -> ServerResourceSnapshot:
        return ServerResourceSnapshot(
            server_id="local",
            sampled_at=utc_now(),
            gpus=[
                GpuResource(
                    index=0,
                    uuid="GPU-one",
                    memory_total_mib=24576,
                    memory_used_mib=256,
                    utilization_percent=0,
                    process_count=0,
                    idle=True,
                ),
            ],
        )


# Each process reports the device it actually received, then waits for its
# own gate. This lets the test release B only after A has exited.
CHILD = """
import os, sys, time
from pathlib import Path

name, code = sys.argv[1:]
print(name, os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
gate = Path(name + ".finish")
deadline = time.monotonic() + 20
while not gate.exists():
    if time.monotonic() >= deadline:
        sys.exit(99)
    time.sleep(0.01)
print(name, "finished", flush=True)
sys.exit(int(code))
"""


@pytest.mark.parametrize(
    "cancel_a",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not Path("/proc/sys/kernel/random/boot_id").exists(),
                reason="supervisor cancellation verifies Linux process groups",
            ),
        ),
    ],
    ids=["failure", "cancellation"],
)
def test_two_runs_on_one_gpu_have_independent_lifecycles(tmp_path: Path, cancel_a: bool) -> None:
    store = QueueStore(tmp_path / "queue")
    for name, code in [("a", 7), ("b", 0)]:
        store.enqueue(
            JobSpec(
                id=name,
                project="tests",
                working_directory=str(tmp_path),
                command=[sys.executable, "-c", CHILD, name, str(code)],
                resources={"host": "local", "exclusive": False, "memory_mib": 11500},
            ),
            run_id=name,
        )
    scheduler = Scheduler(
        store,
        OneCardProbe(),
        server_id="local",
        worker_id="worker",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
    )
    runs = tmp_path / "runs"
    supervisor = DurableJobSupervisor(runs)
    worker = Worker(store, scheduler, DurableSupervisorLauncher(supervisor))
    a_log, b_log = [SupervisorPaths.create(runs, name).console for name in ("a", "b")]

    def tick_until_terminal(run_id: str) -> None:
        deadline = time.monotonic() + 8
        while not store.get(run_id).status.terminal:
            assert not worker.tick().errors
            assert time.monotonic() < deadline, f"{run_id} did not reach a terminal state"
            time.sleep(0.01)

    try:
        assert worker.tick().started == ["a", "b"]
        a, b = store.get("a"), store.get("b")
        assert a.status == b.status == JobStatus.RUNNING
        assert a.allocation.gpu_uuids == b.allocation.gpu_uuids == {0: "GPU-one"}
        assert a.launch.pid != b.launch.pid
        for name, log in [("a", a_log), ("b", b_log)]:
            deadline = time.monotonic() + 5
            while not log.exists() or f"{name} GPU-one" not in log.read_text():
                assert time.monotonic() < deadline, f"{name} did not start on the simulated GPU"
                time.sleep(0.01)
        if cancel_a:
            store.request_cancel("a")
            tick_until_terminal("a")
        else:
            (tmp_path / "a.finish").touch()
        result_a = supervisor.wait("a", timeout_seconds=5)
        assert result_a.exit_code == (-signal.SIGTERM if cancel_a else 7)
        assert supervisor.poll("b") is None
        os.kill(b.launch.pid, 0)
        assert b_log.read_text().strip() == "b GPU-one"

        (tmp_path / "b.finish").touch()
        assert supervisor.wait("b", timeout_seconds=5).exit_code == 0
        tick_until_terminal("b")
        a, b = store.get("a"), store.get("b")
        assert a.status == (JobStatus.CANCELLED if cancel_a else JobStatus.FAILED)
        assert a.exit_code == result_a.exit_code
        assert b.status == JobStatus.SUCCEEDED and b.exit_code == 0
        assert b_log.read_text().splitlines() == ["b GPU-one", "b finished"]
        assert "b finished" not in a_log.read_text()
    finally:
        # Release either gate even when an assertion fails.
        for name in ("a", "b"):
            (tmp_path / f"{name}.finish").touch()
