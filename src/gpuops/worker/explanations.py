"""Read-only explanations for a job's current scheduling state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

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
        return _explain_nonqueued(record, now)

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
        error = snapshot.error or "GPU snapshot is stale or does not match this worker"
        return _unknown_explanation(snapshot_now, error)

    try:
        active = queue.list(
            {
                JobStatus.LEASED,
                JobStatus.STARTING,
                JobStatus.RUNNING,
                JobStatus.CANCEL_REQUESTED,
            }
        )
    except Exception as error:
        return _unknown_explanation(snapshot_now, error)
    reserved: set[int] = set()
    for active_record in active:
        if not _holds_reservation(active_record, server_id=server_id, now=snapshot_now):
            continue
        assert active_record.allocation is not None
        reserved.update(active_record.allocation.gpu_indices)
    assessment = assess_gpu_wait(
        record.spec.resources,
        snapshot,
        reserved_indices=reserved,
    )
    details = assessment.details(record.spec.resources)
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
        message = "The requested GPU capacity is currently occupied."
        next_action = "wait"
    elif assessment.blocker == GpuWaitBlocker.INSUFFICIENT_MEMORY:
        message = "Idle GPUs do not have enough free memory for this job."
        next_action = "adjust_resources"
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


def _explain_nonqueued(record: JobRecord, now: datetime) -> dict[str, Any]:
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
    return _explanation(
        waiting=waiting,
        code=status.value,
        message=message,
        observed_at=now,
        details=details,
        next_action=next_action,
    )


def _holds_reservation(record: JobRecord, *, server_id: str, now: datetime) -> bool:
    allocation = record.allocation
    if allocation is None or allocation.server_id != server_id:
        return False
    return record.status != JobStatus.LEASED or allocation.lease_expires_at > now


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
