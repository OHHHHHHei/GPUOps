from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gpuops.contracts import is_valid_run_id
from gpuops.control import RemoteTarget, SubmissionJournal
from gpuops.core.resources import (
    DEFAULT_RESOURCE_ELIGIBILITY_POLICY,
    GpuEligibility,
    GpuResource,
    GpuTelemetry,
    telemetry_from_mapping,
)
from gpuops.models import JobRecord, JobSpec
from gpuops.monitor.formatting import _idle_state
from gpuops.transport import _resource_snapshot
from gpuops.web.provider import LiveProvider
from gpuops.worker.probe import LocalGpuProbe

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def _spec() -> JobSpec:
    return JobSpec(
        id="contract-test",
        project="tests",
        command=["python", "train.py"],
        working_directory="/work",
    )


@pytest.mark.parametrize("run_id", ["_review", ".review", "-review", "x" * 192])
def test_run_id_contract_accepts_historically_safe_names(tmp_path, run_id: str) -> None:
    assert is_valid_run_id(run_id)
    assert JobRecord(run_id=run_id, spec=_spec(), created_at=NOW, updated_at=NOW).run_id == run_id

    journal = SubmissionJournal(tmp_path / run_id.replace(".", "dot"))
    entry = journal.create_pending(
        run_id=run_id,
        target=RemoteTarget("gpu01", "lab-gpu01"),
        request_spec=_spec(),
        frozen_spec=_spec(),
        now=NOW,
    )
    assert entry.run_id == run_id


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "with space", "x" * 193])
def test_run_id_contract_rejects_unsafe_names(run_id: str) -> None:
    assert not is_valid_run_id(run_id)
    with pytest.raises(ValidationError, match="run_id"):
        JobRecord(run_id=run_id, spec=_spec(), created_at=NOW, updated_at=NOW)


@pytest.mark.parametrize(
    ("memory_used", "utilization", "expected"),
    [
        (600, 0, GpuEligibility.IDLE),
        (2_000, 0, GpuEligibility.RESIDUAL),
        (200, 1, GpuEligibility.BUSY),
    ],
)
def test_default_eligibility_is_shared_across_consumers(
    memory_used: int,
    utilization: int,
    expected: GpuEligibility,
) -> None:
    raw = {
        "index": 0,
        "uuid": "GPU-A",
        "memory.total": 24_576,
        "memory.used": memory_used,
        "utilization.gpu": utilization,
        "processes": [],
    }
    telemetry = telemetry_from_mapping(raw)
    snapshot = _resource_snapshot(
        "gpu01",
        {"collected_at": NOW.isoformat(), "gpus": [raw]},
    )
    dashboard_gpu = LiveProvider._gpu_snapshot(0, raw, None)
    probe = LocalGpuProbe("gpu01")

    assert DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(telemetry) is expected
    assert snapshot.gpus[0].idle is (expected is GpuEligibility.IDLE)
    assert dashboard_gpu["status"] == expected.value
    assert _idle_state(raw) == ("occupied" if expected is GpuEligibility.BUSY else expected.value)
    assert probe._is_idle(0, utilization, memory_used) is (expected is GpuEligibility.IDLE)


def test_unknown_telemetry_stays_visible_and_fails_closed() -> None:
    raw = {
        "index": 0,
        "uuid": "GPU-A",
        "memory.total": 24_576,
        "memory.used": 200,
        "processes": [],
        "custom.monitor.field": "preserved upstream",
    }
    snapshot = _resource_snapshot(
        "gpu01",
        {"collected_at": NOW.isoformat(), "gpus": [raw]},
    )
    gpu = snapshot.gpus[0]
    dashboard_gpu = LiveProvider._gpu_snapshot(0, raw, None)

    assert DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(
        telemetry_from_mapping(raw)
    ) is GpuEligibility.UNKNOWN
    assert gpu.utilization_percent is None
    assert gpu.idle is False
    assert gpu.can_allocate(0) is False
    assert dashboard_gpu["status"] == "unknown"
    assert dashboard_gpu["utilization_percent"] is None
    assert _idle_state(raw) == "unknown"
    assert raw["custom.monitor.field"] == "preserved upstream"


@pytest.mark.parametrize(
    "telemetry",
    [
        GpuTelemetry(process_count=-1, utilization_percent=0, memory_used_mib=200),
        GpuTelemetry(process_count=float("nan"), utilization_percent=0, memory_used_mib=200),
        GpuTelemetry(process_count=float("inf"), utilization_percent=0, memory_used_mib=200),
        GpuTelemetry(process_count=0, utilization_percent=-1, memory_used_mib=200),
        GpuTelemetry(process_count=0, utilization_percent=float("nan"), memory_used_mib=200),
        GpuTelemetry(process_count=0, utilization_percent=float("inf"), memory_used_mib=200),
        GpuTelemetry(process_count=0, utilization_percent=101, memory_used_mib=200),
        GpuTelemetry(process_count=0, utilization_percent=0, memory_used_mib=-1),
        GpuTelemetry(process_count=0, utilization_percent=0, memory_used_mib=float("nan")),
        GpuTelemetry(process_count=0, utilization_percent=0, memory_used_mib=float("inf")),
    ],
)
def test_direct_invalid_telemetry_is_unknown_and_nonallocatable(
    telemetry: GpuTelemetry,
) -> None:
    assert DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(telemetry) is GpuEligibility.UNKNOWN
    assert DEFAULT_RESOURCE_ELIGIBILITY_POLICY.is_allocatable(telemetry) is False


def test_local_probe_rejects_snapshot_with_unknown_eligibility_evidence() -> None:
    payload = {
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-A",
                "memory.total": 24_576,
                "memory.used": 200,
                "processes": [],
            }
        ]
    }

    def runner(command, **kwargs):
        del kwargs
        if command[0] == "gpustat":
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 1, "", "unavailable")

    snapshot = LocalGpuProbe("gpu01", runner=runner, clock=lambda: NOW).snapshot()

    assert snapshot.healthy is False
    assert snapshot.gpus == []


def test_gpu_resource_cannot_allocate_with_unknown_utilization() -> None:
    gpu = GpuResource(
        index=0,
        uuid="GPU-A",
        memory_total_mib=24_576,
        memory_used_mib=200,
        utilization_percent=None,
        process_count=0,
        idle=True,
    )

    assert gpu.can_allocate(1) is False
