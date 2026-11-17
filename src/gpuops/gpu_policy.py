"""GPU sampling limits shared by the scheduler and the lightweight sampler.

This module uses only the standard library so per-device sampler processes can
import it without loading the pydantic models.
"""

from __future__ import annotations

GPU_QUERY_TIMEOUT_SECONDS = 25.0
DEFAULT_IDLE_UTILIZATION_THRESHOLD = 0.0
DEFAULT_IDLE_MEMORY_BASELINE_MIB = 384
DEFAULT_IDLE_MEMORY_TOLERANCE_MIB = 256


def validate_idle_limits(
    utilization_threshold: float, memory_baseline_mib: int, memory_tolerance_mib: int
) -> None:
    if not 0 <= utilization_threshold <= 100:
        raise ValueError("idle_utilization_threshold must be between 0 and 100")
    for field, value in (
        ("idle_memory_baseline_mib", memory_baseline_mib),
        ("idle_memory_tolerance_mib", memory_tolerance_mib),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
