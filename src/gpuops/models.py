from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from gpuops.contracts import RunId


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


_DURATION_RE = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>s|m|h|d)$")


def parse_duration(value: Any) -> timedelta:
    if isinstance(value, timedelta):
        duration = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        duration = timedelta(seconds=float(value))
    elif isinstance(value, str):
        match = _DURATION_RE.fullmatch(value.strip().lower())
        if not match:
            raise ValueError("duration must use a suffix such as 30m, 4h, or 2d")
        number = float(match.group("value"))
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group("unit")]
        duration = timedelta(seconds=number * multiplier)
    else:
        raise ValueError("duration must be seconds, a timedelta, or a suffixed string")
    if duration.total_seconds() <= 0:
        raise ValueError("duration must be positive")
    return duration


Duration = Annotated[
    timedelta,
    BeforeValidator(parse_duration),
    PlainSerializer(lambda value: int(value.total_seconds()), return_type=int),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


PRIORITY_ORDER: dict[Priority, int] = {
    Priority.URGENT: 0,
    Priority.HIGH: 1,
    Priority.NORMAL: 2,
    Priority.LOW: 3,
}

DEFAULT_GPU_MEMORY_HEADROOM_MIB = 1024


class JobStatus(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    STARTING = "starting"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


ACTIVE_JOB_STATUSES = {status for status in JobStatus if not status.terminal}


class ResourceRequest(StrictModel):
    host: str = "auto"
    gpu_count: int = Field(default=1, ge=1)
    gpus: list[int] | None = None
    memory_mib: int = Field(default=DEFAULT_GPU_MEMORY_HEADROOM_MIB, ge=0)
    exclusive: bool = True

    @model_validator(mode="before")
    @classmethod
    def infer_exact_gpu_count(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "gpu_count" in value:
            return value
        gpus = value.get("gpus")
        if isinstance(gpus, list) and gpus:
            return {**value, "gpu_count": len(gpus)}
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip()
        if not value or any(character.isspace() for character in value):
            raise ValueError("host must be 'auto' or a non-empty server identifier")
        return value

    @field_validator("gpus")
    @classmethod
    def validate_gpus(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("gpus cannot be empty; omit it for automatic allocation")
        if any(index < 0 for index in value):
            raise ValueError("GPU indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("GPU indices must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def exact_count_matches(self) -> ResourceRequest:
        if not self.exclusive:
            raise ValueError("shared GPU scheduling is not supported; exclusive must be true")
        if self.gpus is not None and len(self.gpus) != self.gpu_count:
            raise ValueError("gpu_count must equal the number of exact gpus")
        return self


class RuntimeSpec(StrictModel):
    environment: dict[str, str] = Field(default_factory=dict)
    conda_environment: str | None = None

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key in value if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {', '.join(invalid)}")
        return value


class ScheduleSpec(StrictModel):
    priority: Priority = Priority.NORMAL
    expected_duration: Duration | None = None
    not_before: datetime | None = None
    deadline: datetime | None = None

    @field_validator("not_before", "deadline")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("schedule timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> ScheduleSpec:
        if self.not_before and self.deadline and self.not_before >= self.deadline:
            raise ValueError("not_before must be earlier than deadline")
        return self


class TrackingSpec(StrictModel):
    log_path: str | None = None
    output_directory: str | None = None
    progress_adapter: str = "none"


class JobSpec(StrictModel):
    api_version: Literal["gpuops/v1"] = "gpuops/v1"
    id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    project: str = Field(min_length=1, max_length=96)
    command: list[str] = Field(min_length=1)
    working_directory: str
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if any(not isinstance(argument, str) or "\x00" in argument for argument in value):
            raise ValueError("command arguments must be strings without NUL bytes")
        if not value[0].strip():
            raise ValueError("command executable cannot be empty")
        return value

    @field_validator("working_directory")
    @classmethod
    def absolute_working_directory(cls, value: str) -> str:
        if "\x00" in value or not PurePosixPath(value).is_absolute():
            raise ValueError("working_directory must be an absolute POSIX path")
        return value


class UnsupportedSubmissionOptions(ValueError):
    code = "unsupported_option"


def validate_new_submission(spec: JobSpec) -> None:
    """Reject parsed options that the current submission path cannot honor."""

    unsupported: list[str] = []
    if spec.tracking.log_path is not None:
        unsupported.append("tracking.log_path")
    if spec.tracking.output_directory is not None:
        unsupported.append("tracking.output_directory")
    if spec.tracking.progress_adapter != "none":
        unsupported.append("tracking.progress_adapter")
    if unsupported:
        raise UnsupportedSubmissionOptions(
            "new submissions do not support these tracking options: " + ", ".join(unsupported)
        )


class Allocation(StrictModel):
    server_id: str
    gpu_indices: list[int] = Field(min_length=1)
    gpu_uuids: dict[int, str] = Field(default_factory=dict)
    worker_id: str
    lease_token: str
    leased_at: datetime
    lease_expires_at: datetime

    @field_validator("gpu_indices")
    @classmethod
    def unique_gpu_indices(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)) or any(index < 0 for index in value):
            raise ValueError("allocated GPU indices must be unique and non-negative")
        return sorted(value)

    @model_validator(mode="after")
    def expiry_after_lease(self) -> Allocation:
        if self.gpu_uuids and (
            set(self.gpu_uuids) != set(self.gpu_indices)
            or len(set(self.gpu_uuids.values())) != len(self.gpu_indices)
            or any(not value.strip() for value in self.gpu_uuids.values())
        ):
            raise ValueError("allocated GPU UUIDs must uniquely identify each allocated index")
        if self.lease_expires_at <= self.leased_at:
            raise ValueError("lease expiry must be after lease creation")
        return self


class LaunchInfo(StrictModel):
    pid: int = Field(gt=0)
    started_at: datetime
    external_id: str | None = None


class JobRecord(StrictModel):
    run_id: RunId
    spec: JobSpec
    status: JobStatus = JobStatus.QUEUED
    revision: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    allocation: Allocation | None = None
    launch: LaunchInfo | None = None
    finished_at: datetime | None = None
    failure_reason: str | None = None
    # Only observed terminal outcomes; historical records retain unknown values.
    exit_code: int | None = Field(default=None, strict=True)
    launch_error: str | None = None

    @model_validator(mode="after")
    def state_invariants(self) -> JobRecord:
        allocated_states = {
            JobStatus.LEASED,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
        }
        if self.status in allocated_states and self.allocation is None:
            raise ValueError(f"{self.status} jobs require an allocation")
        if self.status == JobStatus.RUNNING and self.launch is None:
            raise ValueError(f"{self.status} jobs require launch information")
        if self.status.terminal and self.finished_at is None:
            raise ValueError("terminal jobs require finished_at")
        if not self.status.terminal and (
            self.exit_code is not None or self.launch_error is not None
        ):
            raise ValueError("execution results require a terminal job")
        return self


class JobSummary(StrictModel):
    """Small queue projection suitable for frequent list and history reads."""

    run_id: RunId
    name: str
    project: str
    status: JobStatus
    priority: Priority
    gpus: list[int]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    expected_duration: Duration | None

    @classmethod
    def from_record(cls, record: JobRecord) -> JobSummary:
        allocation = record.allocation
        return cls(
            run_id=record.run_id,
            name=record.spec.metadata.get("name") or record.spec.id,
            project=record.spec.project,
            status=record.status,
            priority=record.spec.schedule.priority,
            gpus=(
                list(allocation.gpu_indices)
                if allocation is not None
                else list(record.spec.resources.gpus or [])
            ),
            created_at=record.created_at,
            started_at=record.launch.started_at if record.launch is not None else None,
            finished_at=record.finished_at,
            expected_duration=record.spec.schedule.expected_duration,
        )


class QueueEvent(StrictModel):
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)
    run_id: RunId
    event_type: str
    previous_status: JobStatus | None = None
    status: JobStatus
    revision: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
