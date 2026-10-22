from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from gpuops.core.resources import GpuResource, ServerResourceSnapshot
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
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GpuWaitAssessment:
    blocker: GpuWaitBlocker | None
    allocatable_indices: tuple[int, ...]
    occupied_indices: tuple[int, ...]
    insufficient_memory_indices: tuple[int, ...]
    unknown_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]

    def details(self, request: ResourceRequest) -> dict[str, object]:
        details: dict[str, object] = {
            "required_gpu_count": request.gpu_count,
            "requested_memory_mib": request.memory_mib,
            "allocatable_gpu_count": len(self.allocatable_indices),
            "allocatable_gpu_indices": list(self.allocatable_indices),
        }
        if request.gpus is not None:
            details["requested_gpu_indices"] = request.gpus
        if self.occupied_indices:
            details["occupied_gpu_indices"] = list(self.occupied_indices)
        if self.insufficient_memory_indices:
            details["insufficient_memory_gpu_indices"] = list(self.insufficient_memory_indices)
        if self.unknown_indices:
            details["unknown_gpu_indices"] = list(self.unknown_indices)
        if self.missing_indices:
            details["missing_gpu_indices"] = list(self.missing_indices)
        blockers: list[str] = []
        if self.occupied_indices:
            blockers.append(GpuWaitBlocker.OCCUPIED.value)
        if self.insufficient_memory_indices:
            blockers.append(GpuWaitBlocker.INSUFFICIENT_MEMORY.value)
        if blockers:
            details["observed_blockers"] = blockers
        return details


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
    *,
    reserved_indices: set[int],
) -> GpuWaitAssessment:
    by_index = snapshot.by_index()
    requested_indices = request.gpus
    missing = (
        sorted(set(requested_indices).difference(by_index)) if requested_indices is not None else []
    )
    relevant = (
        [by_index[index] for index in requested_indices if index in by_index]
        if requested_indices is not None
        else list(snapshot.gpus)
    )

    allocatable: list[int] = []
    occupied: list[int] = []
    insufficient_memory: list[int] = []
    unknown: list[int] = []
    for gpu in relevant:
        if gpu.index in reserved_indices:
            occupied.append(gpu.index)
        elif not _has_complete_scheduling_evidence(gpu):
            unknown.append(gpu.index)
        elif gpu.can_allocate(request.memory_mib):
            allocatable.append(gpu.index)
        elif not gpu.idle or gpu.process_count != 0:
            occupied.append(gpu.index)
        elif gpu.memory_free_mib is not None and gpu.memory_free_mib < request.memory_mib:
            insufficient_memory.append(gpu.index)
        else:
            unknown.append(gpu.index)

    required = request.gpu_count
    if not missing and len(allocatable) >= required:
        blocker = None
    elif missing or unknown or len(relevant) < required:
        blocker = GpuWaitBlocker.UNKNOWN
    else:
        if occupied and insufficient_memory:
            blocker = GpuWaitBlocker.UNKNOWN
        elif occupied:
            blocker = GpuWaitBlocker.OCCUPIED
        elif insufficient_memory:
            blocker = GpuWaitBlocker.INSUFFICIENT_MEMORY
        else:
            blocker = GpuWaitBlocker.UNKNOWN

    return GpuWaitAssessment(
        blocker=blocker,
        allocatable_indices=tuple(sorted(allocatable)),
        occupied_indices=tuple(sorted(occupied)),
        insufficient_memory_indices=tuple(sorted(insufficient_memory)),
        unknown_indices=tuple(sorted(unknown)),
        missing_indices=tuple(missing),
    )


def _has_complete_scheduling_evidence(gpu: GpuResource) -> bool:
    return not gpu.error and all(
        value is not None
        for value in (
            gpu.memory_total_mib,
            gpu.memory_used_mib,
            gpu.utilization_percent,
            gpu.process_count,
        )
    )
