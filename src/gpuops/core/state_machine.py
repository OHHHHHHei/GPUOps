from __future__ import annotations

from gpuops.models import JobStatus


class InvalidTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.LEASED, JobStatus.CANCELLED}),
    JobStatus.LEASED: frozenset(
        {JobStatus.QUEUED, JobStatus.STARTING, JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.STARTING: frozenset(
        {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, JobStatus.FAILED}
    ),
    JobStatus.RUNNING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCEL_REQUESTED}
    ),
    JobStatus.CANCEL_REQUESTED: frozenset({JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def ensure_transition(source: JobStatus, target: JobStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[source]:
        raise InvalidTransition(f"cannot transition job from {source} to {target}")
