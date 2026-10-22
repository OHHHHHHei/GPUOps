from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Protocol

from pydantic import Field, field_validator, model_validator

from gpuops.models import StrictModel

DEFAULT_IDLE_UTILIZATION_THRESHOLD = 0.0
DEFAULT_IDLE_MEMORY_BASELINE_MIB = 384
DEFAULT_IDLE_MEMORY_TOLERANCE_MIB = 256


class GpuEligibility(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    RESIDUAL = "residual"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    process_count: int | None
    utilization_percent: float | None
    memory_used_mib: int | None


@dataclass(frozen=True, slots=True)
class ResourceEligibilityPolicy:
    idle_utilization_threshold: float = DEFAULT_IDLE_UTILIZATION_THRESHOLD
    idle_memory_baseline_mib: int = DEFAULT_IDLE_MEMORY_BASELINE_MIB
    idle_memory_tolerance_mib: int = DEFAULT_IDLE_MEMORY_TOLERANCE_MIB

    def __post_init__(self) -> None:
        if not 0 <= self.idle_utilization_threshold <= 100:
            raise ValueError("idle_utilization_threshold must be between 0 and 100")
        for field, value in (
            ("idle_memory_baseline_mib", self.idle_memory_baseline_mib),
            ("idle_memory_tolerance_mib", self.idle_memory_tolerance_mib),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")

    @property
    def idle_memory_limit_mib(self) -> int:
        return self.idle_memory_baseline_mib + self.idle_memory_tolerance_mib

    def classify(self, telemetry: GpuTelemetry) -> GpuEligibility:
        if not _valid_telemetry(telemetry):
            return GpuEligibility.UNKNOWN
        if telemetry.process_count > 0:
            return GpuEligibility.BUSY
        if telemetry.utilization_percent > self.idle_utilization_threshold:
            return GpuEligibility.BUSY
        if telemetry.memory_used_mib > self.idle_memory_limit_mib:
            return GpuEligibility.RESIDUAL
        return GpuEligibility.IDLE

    def is_allocatable(self, telemetry: GpuTelemetry) -> bool:
        return self.classify(telemetry) is GpuEligibility.IDLE


DEFAULT_RESOURCE_ELIGIBILITY_POLICY = ResourceEligibilityPolicy()


def telemetry_from_mapping(value: Mapping[str, Any]) -> GpuTelemetry:
    """Extract scheduling evidence without changing the source monitoring payload."""

    if value.get("error") or value.get("error_code"):
        return GpuTelemetry(None, None, None)
    processes = value.get("processes")
    process_count = len(processes) if isinstance(processes, list) else None
    utilization = normalize_nonnegative_number(
        value.get("utilization.gpu"),
        maximum=100,
    )
    return GpuTelemetry(
        process_count=process_count,
        utilization_percent=float(utilization) if utilization is not None else None,
        memory_used_mib=normalize_nonnegative_number(value.get("memory.used"), integral=True),
    )


def normalize_nonnegative_number(
    value: Any,
    *,
    integral: bool = False,
    maximum: float | None = None,
) -> int | float | None:
    """Normalize an external numeric metric, returning None for unsafe values."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        return None
    if integral:
        return int(number) if number.is_integer() else None
    return number


def _valid_telemetry(telemetry: GpuTelemetry) -> bool:
    process_count = telemetry.process_count
    utilization = telemetry.utilization_percent
    memory_used = telemetry.memory_used_mib
    return (
        isinstance(process_count, int)
        and not isinstance(process_count, bool)
        and process_count >= 0
        and isinstance(utilization, (int, float))
        and not isinstance(utilization, bool)
        and isfinite(utilization)
        and 0 <= utilization <= 100
        and isinstance(memory_used, int)
        and not isinstance(memory_used, bool)
        and memory_used >= 0
    )


class GpuResource(StrictModel):
    index: int = Field(ge=0)
    uuid: str
    memory_total_mib: int | None = Field(default=None, ge=0)
    memory_used_mib: int | None = Field(default=None, ge=0)
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    process_count: int | None = Field(default=None, ge=0)
    idle: bool
    sampled_at: datetime | None = None
    error: str | None = None

    @field_validator("sampled_at")
    @classmethod
    def require_sample_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("GPU sampled_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_memory(self) -> GpuResource:
        if (
            self.memory_used_mib is not None
            and self.memory_total_mib is not None
            and self.memory_used_mib > self.memory_total_mib
        ):
            raise ValueError("used GPU memory cannot exceed total memory")
        return self

    @property
    def memory_free_mib(self) -> int | None:
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        return max(0, self.memory_total_mib - self.memory_used_mib)

    def can_allocate(self, memory_mib: int) -> bool:
        free_mib = self.memory_free_mib
        return (
            not self.error
            and self.idle
            and self.process_count == 0
            and self.utilization_percent is not None
            and self.memory_used_mib is not None
            and free_mib is not None
            and free_mib >= memory_mib
        )


class ServerResourceSnapshot(StrictModel):
    server_id: str
    sampled_at: datetime
    healthy: bool = True
    error: str | None = None
    gpus: list[GpuResource] = Field(default_factory=list)

    @field_validator("sampled_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("sampled_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> ServerResourceSnapshot:
        indices = [gpu.index for gpu in self.gpus]
        if len(indices) != len(set(indices)):
            raise ValueError("snapshot contains duplicate GPU indices")
        if not self.healthy and not self.error:
            raise ValueError("unhealthy snapshots must include an error")
        return self

    def is_usable(self, expected_server: str, max_age_seconds: float, now: datetime) -> bool:
        if not self.healthy or self.error or self.server_id != expected_server:
            return False
        age = (
            now.astimezone(timezone.utc) - self.sampled_at.astimezone(timezone.utc)
        ).total_seconds()
        return 0 <= age <= max_age_seconds

    def by_index(self) -> dict[int, GpuResource]:
        return {gpu.index: gpu for gpu in self.gpus}

    def with_fresh_gpus(self, now: datetime, max_age_seconds: float) -> ServerResourceSnapshot:
        """Publication of another card must not renew this card's scheduling evidence."""
        gpus = []
        for gpu in self.gpus:
            sampled_at = gpu.sampled_at or self.sampled_at
            age = (now - sampled_at).total_seconds()
            if gpu.error or not 0 <= age <= max_age_seconds:
                gpu = gpu.model_copy(update={
                    "idle": False,
                    "memory_total_mib": None,
                    "memory_used_mib": None,
                    "utilization_percent": None,
                    "process_count": None,
                    "error": gpu.error or "GPU sample is stale; waiting for a new sample",
                })
            gpus.append(gpu)
        return self.model_copy(update={"gpus": gpus})


class ResourceProbe(Protocol):
    def snapshot(self) -> ServerResourceSnapshot: ...
