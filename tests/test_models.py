from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from gpuops.models import (
    DEFAULT_GPU_MEMORY_HEADROOM_MIB,
    JobRecord,
    JobSpec,
    ResourceRequest,
    validate_new_submission,
)


def make_spec(**overrides: object) -> JobSpec:
    values: dict[str, object] = {
        "id": "baseline-seed-42",
        "project": "ranking",
        "command": ["python", "train.py", "--seed", "42"],
        "working_directory": "/data/researcher/ranking",
    }
    values.update(overrides)
    return JobSpec.model_validate(values)


def test_job_spec_accepts_structured_command_and_duration() -> None:
    spec = make_spec(
        resources={"host": "auto", "gpu_count": 2, "gpus": [3, 1], "memory_mib": 12000},
        schedule={"priority": "high", "expected_duration": "4h"},
    )

    assert spec.command == ["python", "train.py", "--seed", "42"]
    assert spec.resources.gpus == [1, 3]
    assert spec.schedule.expected_duration == timedelta(hours=4)
    assert '"expected_duration":14400' in spec.model_dump_json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_version", "gpuops/v2"),
        ("id", "unsafe id"),
        ("command", []),
        ("working_directory", "relative/path"),
    ],
)
def test_job_spec_rejects_invalid_contract(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_spec(**{field: value})


def test_exact_gpu_request_must_match_count() -> None:
    with pytest.raises(ValidationError, match="gpu_count"):
        ResourceRequest(gpu_count=2, gpus=[0])


def test_exact_gpu_request_infers_omitted_count_and_serializes_normalized_schema() -> None:
    spec = make_spec(resources={"host": "auto", "gpus": [3, 1]})

    assert spec.resources.gpus == [1, 3]
    assert spec.resources.gpu_count == 2
    serialized = spec.model_dump(mode="json")
    assert serialized["resources"]["gpu_count"] == 2
    assert isinstance(serialized["resources"]["gpu_count"], int)
    assert JobSpec.model_validate(serialized) == spec


def test_count_only_gpu_request_keeps_the_concrete_default() -> None:
    request = ResourceRequest()

    assert request.gpus is None
    assert request.gpu_count == 1
    assert request.model_dump()["gpu_count"] == 1


def test_shared_gpu_request_is_rejected_until_supported() -> None:
    with pytest.raises(ValidationError, match="shared GPU scheduling is not supported"):
        ResourceRequest(exclusive=False)


def test_resource_request_defaults_to_minimum_free_memory_headroom() -> None:
    assert ResourceRequest().memory_mib == DEFAULT_GPU_MEMORY_HEADROOM_MIB


def test_explicit_zero_memory_request_remains_backward_compatible() -> None:
    spec = make_spec(resources={"memory_mib": 0})

    assert spec.resources.memory_mib == 0


def test_environment_variable_names_are_validated() -> None:
    with pytest.raises(ValidationError, match="environment variable"):
        make_spec(runtime={"environment": {"BAD-NAME": "value"}})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_spec(unexpected=True)


def test_legacy_tracking_fields_remain_readable_in_persisted_records() -> None:
    record = JobRecord.model_validate(
        {
            "run_id": "legacy-tracking",
            "spec": make_spec(
                tracking={
                    "log_path": "/legacy/train.log",
                    "output_directory": "/legacy/output",
                    "progress_adapter": "legacy-jsonl",
                }
            ).model_dump(mode="json"),
        }
    )

    assert record.spec.tracking.log_path == "/legacy/train.log"
    assert record.spec.tracking.output_directory == "/legacy/output"
    assert record.spec.tracking.progress_adapter == "legacy-jsonl"
    assert JobRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("tracking", "unsupported_field"),
    [
        ({"log_path": "/tmp/train.log"}, "tracking.log_path"),
        ({"output_directory": "/tmp/output"}, "tracking.output_directory"),
        ({"progress_adapter": "jsonl"}, "tracking.progress_adapter"),
    ],
)
def test_new_submission_rejects_tracking_options_the_worker_cannot_honor(
    tracking: dict[str, str], unsupported_field: str
) -> None:
    spec = make_spec(tracking=tracking)

    with pytest.raises(ValueError, match=unsupported_field):
        validate_new_submission(spec)


def test_new_submission_accepts_default_tracking_contract() -> None:
    assert validate_new_submission(make_spec()) is None
