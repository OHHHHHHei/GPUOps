from datetime import timedelta

from gpuops.core.capacity import gpu_capacity
from gpuops.core.resources import GpuResource
from gpuops.models import Allocation, JobRecord, JobSpec, JobStatus, utc_now


def _record(run_id, status, indices, uuids, *, expires_in=60.0, server_id="gpu01"):
    now = utc_now()
    spec = JobSpec(id=run_id, project="tests", command=["true"], working_directory="/work")
    allocation = Allocation(
        server_id=server_id,
        worker_id="worker",
        lease_token=run_id,
        gpu_indices=indices,
        gpu_uuids=dict(zip(indices, uuids, strict=True)),
        leased_at=now - timedelta(seconds=120),
        lease_expires_at=now + timedelta(seconds=expires_in),
    )
    # The helper reads only status and allocation; skip unrelated lifecycle fields.
    return JobRecord.model_construct(run_id=run_id, spec=spec, status=status, allocation=allocation)


def _gpu(index, uuid):
    return GpuResource(index=index, uuid=uuid, idle=True)


def test_reservation_follows_uuid_and_ignores_expired_or_foreign_leases():
    gpus = [_gpu(0, "GPU-b"), _gpu(1, "GPU-a"), _gpu(2, "GPU-c"), _gpu(3, "GPU-d")]
    records = [
        # Leased as index 0 before the cards were renumbered; GPU-a is now index 1.
        _record("running", JobStatus.RUNNING, [0], ["GPU-a"]),
        _record("expired", JobStatus.LEASED, [2], ["GPU-c"], expires_in=-1),
        _record("other-host", JobStatus.RUNNING, [3], ["GPU-d"], server_id="gpu02"),
        _record("done", JobStatus.SUCCEEDED, [3], ["GPU-d"]),
    ]

    by_gpu = {
        gpu.index: [
            job.run_id
            for job in gpu_capacity(
                gpu, records, server_id="gpu01", now=utc_now(), max_jobs_per_gpu=2
            ).jobs
        ]
        for gpu in gpus
    }

    assert by_gpu == {0: [], 1: ["running"], 2: [], 3: []}
