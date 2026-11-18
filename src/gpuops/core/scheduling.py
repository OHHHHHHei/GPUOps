from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from gpuops.core.capacity import gpu_capacity
from gpuops.core.resources import ServerResourceSnapshot
from gpuops.models import JobRecord, ResourceRequest


class ScheduleBlocker(str, Enum):
    HOST_MISMATCH = "host_mismatch"
    NOT_BEFORE = "not_before"
    DEADLINE = "deadline"


@dataclass(frozen=True, slots=True)
class ScheduleWindowAssessment:
    blocker: ScheduleBlocker | None = None
    available_at: datetime | None = None
    deadline: datetime | None = None

    @property
    def eligible(self) -> bool:
        return self.blocker is None


class GpuWaitBlocker(str, Enum):
    OCCUPIED = "gpu_occupied"
    INSUFFICIENT_MEMORY = "insufficient_free_memory"
    JOB_LIMIT = "gpu_job_limit_reached"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GpuWaitAssessment:
    blocker: GpuWaitBlocker | None
    details: dict[str, object]


def assess_schedule_window(
    record: JobRecord,
    *,
    server_id: str,
    now: datetime,
) -> ScheduleWindowAssessment:
    resources = record.spec.resources
    if resources.host not in {"auto", server_id}:
        return ScheduleWindowAssessment(blocker=ScheduleBlocker.HOST_MISMATCH)

    normalized_now = now.astimezone(timezone.utc)
    not_before = record.spec.schedule.not_before
    if not_before is not None and not_before.astimezone(timezone.utc) > normalized_now:
        return ScheduleWindowAssessment(
            blocker=ScheduleBlocker.NOT_BEFORE,
            available_at=not_before,
        )

    deadline = record.spec.schedule.deadline
    expected = record.spec.schedule.expected_duration or timedelta(0)
    if deadline is not None and normalized_now + expected > deadline.astimezone(timezone.utc):
        return ScheduleWindowAssessment(
            blocker=ScheduleBlocker.DEADLINE,
            deadline=deadline,
        )
    return ScheduleWindowAssessment()


def assess_gpu_wait(
    request: ResourceRequest,
    snapshot: ServerResourceSnapshot,
    active: Sequence[JobRecord],
    *,
    now: datetime,
) -> GpuWaitAssessment:
    by_index = snapshot.by_index()
    requested = sorted(by_index) if request.gpus is None else request.gpus
    available: list[int] = []
    missing: list[int] = []
    blocked: dict[GpuWaitBlocker, list[int]] = {}
    cards: list[dict[str, object]] = []
    for index in requested:
        gpu = by_index.get(index)
        if gpu is None:
            missing.append(index)
            continue
        capacity = gpu_capacity(
            gpu,
            active,
            server_id=snapshot.server_id,
            now=now,
            max_jobs_per_gpu=snapshot.max_jobs_per_gpu,
        )
        reason = capacity.blocker(request)
        if reason is None:
            available.append(index)
            continue
        blocked.setdefault(GpuWaitBlocker(reason), []).append(index)
        cards.append({"gpu_index": index, "reason": reason, **capacity.model_dump(mode="json")})

    blocker: GpuWaitBlocker | None = GpuWaitBlocker.UNKNOWN
    if not missing and len(available) >= request.gpu_count:
        blocker = None
    elif not missing and len(requested) >= request.gpu_count and len(blocked) == 1:
        blocker = next(iter(blocked))

    details: dict[str, object] = {
        "required_gpu_count": request.gpu_count,
        "requested_memory_mib": request.memory_mib,
        "requested_exclusive": request.exclusive,
        "allocatable_gpu_count": len(available),
        "allocatable_gpu_indices": available,
    }
    if request.gpus is not None:
        details["requested_gpu_indices"] = requested
    if missing:
        details["missing_gpu_indices"] = missing
    keys = {
        GpuWaitBlocker.OCCUPIED: "occupied_gpu_indices",
        GpuWaitBlocker.INSUFFICIENT_MEMORY: "insufficient_memory_gpu_indices",
        GpuWaitBlocker.JOB_LIMIT: "job_limit_gpu_indices",
        GpuWaitBlocker.UNKNOWN: "unknown_gpu_indices",
    }
    for reason_code, key in keys.items():
        if reason_code in blocked:
            details[key] = blocked[reason_code]
    observed = [code.value for code in keys if code in blocked and code != GpuWaitBlocker.UNKNOWN]
    if observed:
        details["observed_blockers"] = observed
    if cards:
        details["blocked_gpus"] = cards
    return GpuWaitAssessment(blocker, details)
