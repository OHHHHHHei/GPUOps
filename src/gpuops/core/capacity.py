"""Per-card admission from a GPU sample and the current durable allocations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime

from pydantic import Field

from gpuops.core.resources import GpuResource
from gpuops.models import JobRecord, JobStatus, ResourceRequest, StrictModel

ALLOCATED_STATUSES = frozenset(
    {JobStatus.LEASED, JobStatus.STARTING, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
)


def records_on_gpu(
    index: int,
    uuid: str | None,
    active_records: Iterable[JobRecord],
    *,
    server_id: str,
    now: datetime | None,
) -> Iterator[JobRecord]:
    """Match physical allocations by UUID, or by index when either identity is missing.

    The lease writer passes no expiry cutoff: it holds the reservation until the
    expired lease is actually returned to the queue under the same lock.
    """
    for record in active_records:
        allocation = record.allocation
        if allocation is None or allocation.server_id != server_id:
            continue
        if record.status not in ALLOCATED_STATUSES:
            continue
        if (
            now is not None
            and record.status == JobStatus.LEASED
            and allocation.lease_expires_at <= now
        ):
            continue
        if uuid and allocation.gpu_uuids:
            if uuid in allocation.gpu_uuids.values():
                yield record
        elif index in allocation.gpu_indices:
            yield record


class GpuJob(StrictModel):
    run_id: str
    exclusive: bool
    memory_mib: int

    @classmethod
    def from_record(cls, record: JobRecord) -> GpuJob:
        request = record.spec.resources
        return cls(run_id=record.run_id, exclusive=request.exclusive, memory_mib=request.memory_mib)


class GpuCapacity(StrictModel):
    jobs: list[GpuJob]
    max_jobs_per_gpu: int
    available_memory_mib: int | None
    idle: bool = Field(exclude=True)

    def blocker(self, request: ResourceRequest) -> str | None:
        if self.jobs and (request.exclusive or any(job.exclusive for job in self.jobs)):
            return "gpu_occupied"
        if self.available_memory_mib is None:
            return "unknown"
        if request.exclusive and not self.idle:
            return "gpu_occupied"
        if not request.exclusive and len(self.jobs) >= self.max_jobs_per_gpu:
            return "gpu_job_limit_reached"
        if self.available_memory_mib < request.memory_mib:
            return "insufficient_free_memory"
        return None


def gpu_capacity(
    gpu: GpuResource,
    active_records: Iterable[JobRecord],
    *,
    server_id: str,
    now: datetime | None,
    max_jobs_per_gpu: int,
) -> GpuCapacity:
    """Reserve each job's shortfall on top of the board's measured use."""
    groups = {
        (group.pgid, group.leader_start_ticks, group.boot_id): group.memory_mib
        for group in gpu.group_memory
    }
    jobs = []
    attributed = 0
    outstanding = 0
    for record in records_on_gpu(gpu.index, gpu.uuid, active_records, server_id=server_id, now=now):
        job = GpuJob.from_record(record)
        jobs.append(job)
        used = 0
        launch = record.launch
        if launch and launch.external_id and launch.start_ticks is not None and launch.boot_id:
            used = groups.pop((launch.pid, launch.start_ticks, launch.boot_id), 0)
        attributed += used
        outstanding += max(0, job.memory_mib - used)
    free = gpu.memory_free_mib
    available = None
    if (
        free is not None
        and not gpu.error
        and gpu.process_count is not None
        and gpu.utilization_percent is not None
        and gpu.memory_used_mib is not None
        and attributed <= gpu.memory_used_mib
    ):
        available = max(0, free - outstanding)
    return GpuCapacity(
        jobs=jobs,
        max_jobs_per_gpu=max_jobs_per_gpu,
        available_memory_mib=available,
        idle=gpu.idle and gpu.process_count == 0,
    )
