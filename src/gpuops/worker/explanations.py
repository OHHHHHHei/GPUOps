"""Read-only explanations for a job's current scheduling state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from gpuops.core.capacity import ALLOCATED_STATUSES, GpuJob, records_on_gpu
from gpuops.core.queue import QueueStore
from gpuops.core.resources import ResourceProbe
from gpuops.core.scheduling import (
    GpuWaitBlocker,
    ScheduleBlocker,
    assess_gpu_wait,
    assess_schedule_window,
)
from gpuops.models import JobRecord, JobStatus


def explain_job(
    record: JobRecord,
    *,
    server_id: str,
    queue: QueueStore,
    resource_probe: ResourceProbe,
    readiness_probe: Callable[[], tuple[bool, str]],
    clock: Callable[[], datetime],
    max_snapshot_age_seconds: float,
) -> dict[str, Any]:
    """Describe known blockers without changing queue or worker state."""

    now = clock()
    if record.status != JobStatus.QUEUED:
        return _explain_nonqueued(record, now, queue)

    window = assess_schedule_window(record, server_id=server_id, now=now)
    if window.blocker == ScheduleBlocker.NOT_BEFORE:
        assert window.available_at is not None
        return _explanation(
            waiting=True,
            code=window.blocker.value,
            message="Job is waiting until its configured not-before time.",
            observed_at=now,
            details={"available_at": window.available_at.isoformat()},
            next_action="wait",
        )
    if window.blocker == ScheduleBlocker.DEADLINE:
        assert window.deadline is not None
        expected = record.spec.schedule.expected_duration
        return _explanation(
            waiting=True,
            code=window.blocker.value,
            message="Job cannot finish within its configured deadline.",
            observed_at=now,
            details={
                "deadline": window.deadline.isoformat(),
                "expected_duration_seconds": (
                    expected.total_seconds() if expected is not None else 0
                ),
            },
            next_action="adjust_schedule",
        )
    if window.blocker == ScheduleBlocker.HOST_MISMATCH:
        return _explanation(
            waiting=True,
            code=window.blocker.value,
            message="Job scheduling target does not match this worker.",
            observed_at=now,
            details={
                "job_host": record.spec.resources.host,
                "worker_server_id": server_id,
            },
            next_action="resubmit",
        )

    try:
        ready, reason = readiness_probe()
    except Exception as error:
        return _unknown_explanation(now, error)
    if not ready:
        bounded_reason = _error_text(reason)
        return _explanation(
            waiting=True,
            code="worker_unavailable",
            message=f"Background worker is unavailable: {bounded_reason}",
            observed_at=now,
            details={"worker_reason": bounded_reason},
            next_action="start_worker",
        )

    try:
        snapshot = resource_probe.snapshot()
    except Exception as error:
        return _unknown_explanation(now, error)
    snapshot_now = clock()
    if not snapshot.is_usable(server_id, max_snapshot_age_seconds, snapshot_now):
        snapshot_error = snapshot.error or "GPU snapshot is stale or does not match this worker"
        return _unknown_explanation(snapshot_now, snapshot_error)

    try:
        active = queue.list(set(ALLOCATED_STATUSES))
    except Exception as error:
        return _unknown_explanation(snapshot_now, error)
    assessment = assess_gpu_wait(
        record.spec.resources,
        snapshot.with_fresh_gpus(snapshot_now, max_snapshot_age_seconds),
        active,
        now=snapshot_now,
    )
    details = assessment.details
    if assessment.blocker is None:
        details["confirmation_required"] = True
        return _explanation(
            waiting=True,
            code="unknown",
            message=(
                "No blocker was proven by one GPU sample; scheduler confirmation "
                "or queue order may still delay the job."
            ),
            observed_at=snapshot_now,
            details=details,
            next_action="wait",
        )
    if assessment.blocker == GpuWaitBlocker.OCCUPIED:
        message = (
            "An exclusive job requires an idle GPU with no managed allocations."
            if record.spec.resources.exclusive
            else "An exclusive allocation holds the requested GPU capacity."
        )
        next_action = "wait"
    elif assessment.blocker == GpuWaitBlocker.INSUFFICIENT_MEMORY:
        message = "The available per-GPU memory budget is below the requested amount."
        next_action = "adjust_resources"
    elif assessment.blocker == GpuWaitBlocker.JOB_LIMIT:
        message = "The shared GPUs have reached their managed job limit."
        next_action = "wait"
    else:
        message = "The current GPU evidence does not identify one definite wait reason."
        next_action = "inspect_gpu_status"
    return _explanation(
        waiting=True,
        code=assessment.blocker.value,
        message=message,
        observed_at=snapshot_now,
        details=details,
        next_action=next_action,
    )


def _explain_nonqueued(record: JobRecord, now: datetime, queue: QueueStore) -> dict[str, Any]:
    status = record.status
    waiting = status in {JobStatus.LEASED, JobStatus.STARTING}
    details: dict[str, Any] = {"status": status.value}
    if status == JobStatus.LEASED:
        message = "GPU resources are leased and the job is waiting to start."
        next_action = "wait"
    elif status == JobStatus.STARTING:
        message = "The worker is starting the job process."
        next_action = "wait"
    elif status == JobStatus.RUNNING:
        message = "The job process is running."
        next_action = "inspect_logs"
    elif status == JobStatus.CANCEL_REQUESTED:
        message = "Cancellation was requested and the worker is stopping the job."
        next_action = "wait"
    elif status == JobStatus.SUCCEEDED:
        message = "The job completed successfully."
        next_action = "inspect_logs"
    elif status == JobStatus.FAILED:
        details["failure_reason"] = record.failure_reason
        message = (
            f"The job failed: {record.failure_reason}"
            if record.failure_reason
            else "The job failed without a recorded reason."
        )
        next_action = "inspect_logs"
    else:
        message = "The job was cancelled."
        next_action = "inspect_logs"
    allocation = record.allocation
    if status in ALLOCATED_STATUSES and allocation is not None:
        try:
            active = queue.list(set(ALLOCATED_STATUSES))
        except Exception as error:
            details["gpu_peers_error"] = _error_text(error)
        else:
            cards = []
            for index in allocation.gpu_indices:
                uuid = allocation.gpu_uuids.get(index)
                peers = records_on_gpu(index, uuid, active, server_id=allocation.server_id, now=now)
                cards.append(
                    {
                        "allocated_index": index,
                        "uuid": uuid,
                        "jobs": [
                            GpuJob.from_record(peer).model_dump(mode="json") for peer in peers
                        ],
                    }
                )
            details["gpus"] = cards
    return _explanation(
        waiting=waiting,
        code=status.value,
        message=message,
        observed_at=now,
        details=details,
        next_action=next_action,
    )


def _unknown_explanation(
    observed_at: datetime,
    error: BaseException | str,
) -> dict[str, Any]:
    return _explanation(
        waiting=True,
        code="unknown",
        message="GPU availability could not be determined.",
        observed_at=observed_at,
        details={"error": _error_text(error)},
        next_action="retry",
    )


def _explanation(
    *,
    waiting: bool,
    code: str,
    message: str,
    observed_at: datetime,
    details: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    return {
        "waiting": waiting,
        "code": code,
        "message": message,
        "observed_at": observed_at.isoformat(),
        "details": details,
        "next_action": next_action,
    }


def _error_text(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        message = str(error).strip() or error.__class__.__name__
        text = f"{error.__class__.__name__}: {message}"
    else:
        text = error.strip() or "GPU snapshot is unavailable"
    return text.replace("\n", " ")[:300]
