from datetime import timedelta

import pytest

from gpuops.core.capacity import gpu_capacity
from gpuops.core.resources import GpuGroupMemory, GpuResource
from gpuops.models import Allocation, JobRecord, JobSpec, JobStatus, LaunchInfo, utc_now

NOW = utc_now()


def running(run_id, pid, request=11500):
    return JobRecord(
        run_id=run_id,
        spec=JobSpec(
            id=run_id,
            project="tests",
            command=["true"],
            working_directory="/tmp",
            resources={"memory_mib": request},
        ),
        status=JobStatus.RUNNING,
        allocation=Allocation(
            server_id="lab",
            gpu_indices=[0],
            gpu_uuids={0: "GPU-a"},
            worker_id="worker",
            lease_token=run_id,
            leased_at=NOW,
            lease_expires_at=NOW + timedelta(seconds=30),
        ),
        launch=LaunchInfo(
            pid=pid, started_at=NOW, external_id=run_id, start_ticks=42, boot_id="boot"
        ),
    )


def card(used, observed):
    return GpuResource(
        index=0,
        uuid="GPU-a",
        idle=False,
        memory_total_mib=24576,
        memory_used_mib=used,
        utilization_percent=45,
        process_count=len(observed),
        group_memory=[
            GpuGroupMemory(pgid=pid, leader_start_ticks=ticks, boot_id=boot, memory_mib=amount)
            for pid, ticks, boot, amount in observed
        ],
    )


@pytest.mark.parametrize(
    ("board_used", "observed", "expected"),
    [
        (256, [], 12820),  # Reserved before the model starts using memory.
        (10496, [(10, 42, "boot", 10240)], 12820),
        (12544, [(10, 42, "boot", 10240), (20, 42, "boot", 2048)], 10772),
        (13256, [(10, 42, "boot", 13000)], 11320),
        (10496, [(10, 43, "boot", 10240)], 2580),  # Reused PID.
        (10496, [(10, 42, "old-boot", 10240)], 2580),
        (9000, [(10, 42, "boot", 10000)], None),  # Process released memory between queries.
    ],
)
def test_only_credit_observed_usage_of_this_allocation(board_used, observed, expected):
    capacity = gpu_capacity(
        card(board_used, observed), [running("a", 10)], server_id="lab", now=NOW, max_jobs_per_gpu=2
    )
    assert capacity.available_memory_mib == expected


def test_one_jobs_excess_does_not_spend_another_jobs_reservation():
    capacity = gpu_capacity(
        card(13256, [(10, 42, "boot", 13000)]),
        [running("a", 10), running("b", 20, 8000)],
        server_id="lab",
        now=NOW,
        max_jobs_per_gpu=3,
    )
    assert capacity.available_memory_mib == 3320
