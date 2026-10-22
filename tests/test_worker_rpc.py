from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.models import JobSpec, JobStatus, LaunchInfo
from gpuops.worker.app import LOG_BUDGET_BYTES, STATE_BUDGET_BYTES
from gpuops.worker.rpc import (
    DEFAULT_LOG_BUDGET_BYTES,
    DEFAULT_TOTAL_BUDGET_BYTES,
    RPC_API_VERSION,
    HostMismatch,
    InvalidParams,
    WorkerRpc,
)
from gpuops.worker.sampling import CachedGpuProbe

NOW = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def job_spec(*, job_id: str = "train", host: str = "gpu01") -> dict:
    return {
        "api_version": "gpuops/v1",
        "id": job_id,
        "project": "rpc-tests",
        "command": ["python", "train.py", "--seed", "42"],
        "working_directory": "/data/researcher/project",
        "resources": {"host": host, "gpu_count": 1},
    }


def envelope(method: str, params: dict | None = None, *, request_id: str = "request-1") -> dict:
    return {
        "api_version": RPC_API_VERSION,
        "method": method,
        "params": params or {},
        "request_id": request_id,
    }


def submit(rpc: WorkerRpc, run_id: str = "run-001", *, spec: dict | None = None) -> dict:
    rpc.readiness_probe = lambda: (True, "ready")
    return rpc.handle(envelope("submit", {"run_id": run_id, "job_spec": spec or job_spec()}))


def gpu(index: int, *, idle: bool = True, free_mib: int = 24000) -> GpuResource:
    total_mib = 24576
    return GpuResource(
        index=index,
        uuid=f"GPU-{index}",
        memory_total_mib=total_mib,
        memory_used_mib=total_mib - free_mib,
        utilization_percent=0 if idle else 80,
        process_count=0 if idle else 1,
        idle=idle,
    )


def gpu_snapshot(*gpus: GpuResource, healthy: bool = True) -> ServerResourceSnapshot:
    return ServerResourceSnapshot(
        server_id="gpu01",
        sampled_at=NOW,
        healthy=healthy,
        error=None if healthy else "GPU probe failed",
        gpus=list(gpus),
    )


class ExplanationProbe:
    def __init__(
        self,
        value: ServerResourceSnapshot | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.value = value
        self.error = error
        self.calls = 0

    def snapshot(self) -> ServerResourceSnapshot:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.value is not None
        return self.value


def test_initializes_private_state_layout_and_default_budgets(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    assert rpc.queue.root == tmp_path / "queue"
    assert rpc.runs_directory == tmp_path / "runs"
    assert rpc.logs_directory == tmp_path / "logs"
    assert rpc.storage.total_budget_bytes == DEFAULT_TOTAL_BUDGET_BYTES
    assert rpc.storage.log_budget_bytes == DEFAULT_LOG_BUDGET_BYTES
    assert DEFAULT_TOTAL_BUDGET_BYTES == STATE_BUDGET_BYTES
    assert DEFAULT_LOG_BUDGET_BYTES == LOG_BUDGET_BYTES
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(rpc.queue.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(rpc.runs_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(rpc.logs_directory.stat().st_mode) == 0o700


def test_default_explanation_probe_reads_shared_worker_cache(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01", clock=lambda: NOW)

    assert isinstance(rpc.resource_probe, CachedGpuProbe)
    assert rpc.resource_probe.cache_path == tmp_path / "gpu-sample.json"
    assert rpc.resource_probe.server_id == "gpu01"


def test_submit_is_durable_and_idempotent(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    first = submit(rpc)
    second = submit(rpc)

    assert first["ok"] is True
    assert first["request_id"] == "request-1"
    assert first["result"]["durable"] is True
    assert first["result"]["job"]["run_id"] == "run-001"
    assert second["result"]["job"] == first["result"]["job"]
    assert len(rpc.queue.list()) == 1
    assert len(rpc.queue.events()) == 1
    json.dumps(first)


def test_submit_normalizes_omitted_gpu_count_for_exact_gpu_requests(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    spec = job_spec()
    spec["resources"] = {"host": "gpu01", "gpus": [3, 1]}

    response = submit(rpc, spec=spec)

    assert response["ok"] is True
    resources = response["result"]["job"]["spec"]["resources"]
    assert resources["gpus"] == [1, 3]
    assert resources["gpu_count"] == 2


def test_submit_rejects_unsupported_tracking_for_a_fresh_run(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    spec = {**job_spec(), "tracking": {"output_directory": "/legacy/output"}}

    response = submit(rpc, spec=spec)

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_params"
    assert "tracking.output_directory" in response["error"]["message"]
    assert rpc.queue.list() == []


def test_submit_retry_returns_matching_legacy_record_before_new_submission_checks(
    tmp_path,
) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    legacy_spec = JobSpec.model_validate(
        {
            **job_spec(),
            "resources": {"host": "gpu01", "gpu_count": 2, "gpus": [3, 1]},
            "tracking": {"progress_adapter": "legacy-jsonl"},
        }
    )
    existing = rpc.queue.enqueue(legacy_spec, run_id="legacy-run")
    cross_version_payload = legacy_spec.model_dump(mode="json")
    del cross_version_payload["resources"]["gpu_count"]

    response = rpc.handle(
        envelope(
            "submit",
            {"run_id": "legacy-run", "job_spec": cross_version_payload},
        )
    )

    assert response["ok"] is True
    assert response["result"]["job"] == existing.model_dump(
        mode="json", exclude={"exit_code", "launch_error"}
    )
    assert len(rpc.queue.events()) == 1


def test_submit_requires_job_host_to_match_worker(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    response = submit(rpc, spec=job_spec(host="auto"))

    assert response["ok"] is False
    assert response["error"]["code"] == "host_mismatch"
    assert rpc.queue.list() == []
    with pytest.raises(HostMismatch):
        rpc.submit({"run_id": "direct", "job_spec": job_spec(host="gpu02")})


def test_submit_rejects_run_id_reuse_with_different_spec(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    assert submit(rpc)["ok"] is True

    response = submit(rpc, spec=job_spec(job_id="different"))

    assert response["ok"] is False
    assert response["error"]["code"] == "run_id_conflict"
    assert rpc.queue.get("run-001").spec.id == "train"


def test_list_and_show_return_json_job_records(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    submit(rpc, "run-normal")
    submit(rpc, "run-high", spec={**job_spec(job_id="high"), "schedule": {"priority": "high"}})

    listed = rpc.handle(envelope("list", {"statuses": ["queued"]}))
    shown = rpc.handle(envelope("show", {"run_id": "run-high"}))
    shown_without_explanation = rpc.handle(
        envelope("show", {"run_id": "run-high", "explain": False})
    )

    assert listed["ok"] is True
    assert listed["result"]["count"] == 2
    assert [job["run_id"] for job in listed["result"]["jobs"]] == [
        "run-high",
        "run-normal",
    ]
    assert shown["result"]["job"]["spec"]["id"] == "high"
    assert set(shown["result"]) == {"job", "execution_result"}
    assert shown["result"]["execution_result"] == {"exit_code": None, "launch_error": None}
    assert shown_without_explanation["result"] == shown["result"]
    json.dumps(listed)
    json.dumps(shown)


def test_show_unknown_job_returns_defined_error(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    response = rpc.handle(envelope("show", {"run_id": "missing"}))

    assert response["ok"] is False
    assert response["error"]["code"] == "job_not_found"


def test_show_explanation_rejects_non_boolean_flag(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    submit(rpc)

    response = rpc.handle(envelope("show", {"run_id": "run-001", "explain": 1}))

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_params"


def test_show_explains_schedule_windows_without_readiness_or_gpu_probe(tmp_path) -> None:
    probe = ExplanationProbe(error=AssertionError("GPU probe should not run"))

    def unexpected_readiness() -> tuple[bool, str]:
        raise AssertionError("readiness should not be checked")

    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=unexpected_readiness,
        resource_probe=probe,
    )
    future = JobSpec.model_validate(
        {
            **job_spec(job_id="future"),
            "schedule": {"not_before": NOW + timedelta(hours=1)},
        }
    )
    impossible = JobSpec.model_validate(
        {
            **job_spec(job_id="deadline"),
            "schedule": {
                "expected_duration": "2h",
                "deadline": NOW + timedelta(hours=1),
            },
        }
    )
    rpc.queue.enqueue(future, run_id="future-run", now=NOW)
    rpc.queue.enqueue(impossible, run_id="deadline-run", now=NOW)
    other_host = rpc.queue.enqueue(
        JobSpec.model_validate(job_spec(job_id="other-host", host="gpu02")),
        run_id="other-host-run",
        now=NOW,
    )

    future_response = rpc.handle(envelope("show", {"run_id": "future-run", "explain": True}))
    deadline_response = rpc.handle(envelope("show", {"run_id": "deadline-run", "explain": True}))
    host_response = rpc.handle(envelope("show", {"run_id": other_host.run_id, "explain": True}))

    assert future_response["result"]["explanation"] == {
        "waiting": True,
        "code": "not_before",
        "message": "Job is waiting until its configured not-before time.",
        "observed_at": NOW.isoformat(),
        "details": {"available_at": (NOW + timedelta(hours=1)).isoformat()},
        "next_action": "wait",
    }
    assert deadline_response["result"]["explanation"]["code"] == "deadline"
    assert deadline_response["result"]["explanation"]["details"] == {
        "deadline": (NOW + timedelta(hours=1)).isoformat(),
        "expected_duration_seconds": 7200.0,
    }
    assert host_response["result"]["explanation"]["code"] == "host_mismatch"
    assert host_response["result"]["explanation"]["next_action"] == "resubmit"
    assert probe.calls == 0


def test_show_explains_worker_unavailability_without_gpu_probe(tmp_path) -> None:
    probe = ExplanationProbe(error=AssertionError("GPU probe should not run"))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (False, "heartbeat is stale"),
        resource_probe=probe,
    )
    rpc.queue.enqueue(JobSpec.model_validate(job_spec()), run_id="waiting-run", now=NOW)

    response = rpc.handle(envelope("show", {"run_id": "waiting-run", "explain": True}))

    explanation = response["result"]["explanation"]
    assert explanation["code"] == "worker_unavailable"
    assert explanation["details"] == {"worker_reason": "heartbeat is stale"}
    assert probe.calls == 0


def test_show_explains_active_gpu_reservation_without_mutating_queue(tmp_path) -> None:
    probe = ExplanationProbe(gpu_snapshot(gpu(0)))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    exact_spec = JobSpec.model_validate(
        {
            **job_spec(),
            "resources": {"host": "gpu01", "gpus": [0], "memory_mib": 12000},
        }
    )
    active = rpc.queue.enqueue(exact_spec, run_id="active-run", now=NOW)
    rpc.queue.lease(
        active.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker-1",
        ttl=timedelta(seconds=30),
        expected_revision=active.revision,
        now=NOW,
    )
    waiting = rpc.queue.enqueue(exact_spec, run_id="waiting-run", now=NOW)
    events_before = rpc.queue.events()

    response = rpc.handle(envelope("show", {"run_id": waiting.run_id, "explain": True}))

    explanation = response["result"]["explanation"]
    assert explanation["code"] == "gpu_occupied"
    assert explanation["details"]["occupied_gpu_indices"] == [0]
    assert rpc.queue.get(waiting.run_id) == waiting
    assert rpc.queue.events() == events_before


def test_show_ignores_expired_lease_without_mutating_queue(tmp_path) -> None:
    probe = ExplanationProbe(gpu_snapshot(gpu(0)))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    exact_spec = JobSpec.model_validate(
        {
            **job_spec(),
            "resources": {"host": "gpu01", "gpus": [0], "memory_mib": 12000},
        }
    )
    active = rpc.queue.enqueue(exact_spec, run_id="expired-run", now=NOW - timedelta(minutes=1))
    expired = rpc.queue.lease(
        active.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker-1",
        ttl=timedelta(seconds=30),
        expected_revision=active.revision,
        now=NOW - timedelta(minutes=1),
    )
    waiting = rpc.queue.enqueue(exact_spec, run_id="waiting-run", now=NOW)
    events_before = rpc.queue.events()

    response = rpc.handle(envelope("show", {"run_id": waiting.run_id, "explain": True}))

    explanation = response["result"]["explanation"]
    assert explanation["code"] == "unknown"
    assert explanation["details"]["confirmation_required"] is True
    assert "occupied_gpu_indices" not in explanation["details"]
    assert rpc.queue.get(expired.run_id) == expired
    assert rpc.queue.events() == events_before


@pytest.mark.parametrize(
    ("snapshot", "resources", "expected_code"),
    [
        (
            gpu_snapshot(gpu(0, free_mib=8000)),
            {"host": "gpu01", "gpu_count": 1, "memory_mib": 12000},
            "insufficient_free_memory",
        ),
        (
            gpu_snapshot(gpu(0)),
            {"host": "gpu01", "gpu_count": 1, "memory_mib": 12000},
            "unknown",
        ),
        (
            gpu_snapshot(gpu(0, idle=False), gpu(1, free_mib=8000)),
            {"host": "gpu01", "gpu_count": 2, "memory_mib": 12000},
            "unknown",
        ),
        (
            gpu_snapshot(GpuResource(index=0, uuid="GPU-0", idle=False)),
            {"host": "gpu01", "gpu_count": 1, "memory_mib": 12000},
            "unknown",
        ),
    ],
)
def test_show_explains_gpu_evidence_without_overstating_certainty(
    tmp_path,
    snapshot,
    resources,
    expected_code,
) -> None:
    probe = ExplanationProbe(snapshot)
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    spec = JobSpec.model_validate({**job_spec(), "resources": resources})
    rpc.queue.enqueue(spec, run_id="waiting-run", now=NOW)

    response = rpc.handle(envelope("show", {"run_id": "waiting-run", "explain": True}))

    explanation = response["result"]["explanation"]
    assert explanation["code"] == expected_code
    if snapshot.gpus == [gpu(0)]:
        assert explanation["details"]["confirmation_required"] is True
    if len(snapshot.gpus) == 2:
        assert explanation["details"]["observed_blockers"] == [
            "gpu_occupied",
            "insufficient_free_memory",
        ]
    assert probe.calls == 1


def test_show_validates_snapshot_freshness_after_probe_finishes(tmp_path) -> None:
    sampled_at = NOW + timedelta(milliseconds=500)
    probe = ExplanationProbe(gpu_snapshot(gpu(0)).model_copy(update={"sampled_at": sampled_at}))
    times = iter((NOW, sampled_at))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: next(times),
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    rpc.queue.enqueue(JobSpec.model_validate(job_spec()), run_id="waiting-run", now=NOW)

    response = rpc.handle(envelope("show", {"run_id": "waiting-run", "explain": True}))

    explanation = response["result"]["explanation"]
    assert explanation["code"] == "unknown"
    assert explanation["observed_at"] == sampled_at.isoformat()
    assert explanation["details"]["confirmation_required"] is True


def test_show_preserves_job_and_queue_when_explanation_probe_fails(tmp_path) -> None:
    probe = ExplanationProbe(error=OSError("NVML unavailable\nwith details"))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    record = rpc.queue.enqueue(
        JobSpec.model_validate(job_spec()),
        run_id="waiting-run",
        now=NOW,
    )
    events_before = rpc.queue.events()

    response = rpc.handle(envelope("show", {"run_id": record.run_id, "explain": True}))

    assert response["ok"] is True
    assert response["result"]["job"] == record.model_dump(
        mode="json", exclude={"exit_code", "launch_error"}
    )
    explanation = response["result"]["explanation"]
    assert explanation["code"] == "unknown"
    assert explanation["details"]["error"] == "OSError: NVML unavailable with details"
    assert rpc.queue.get(record.run_id) == record
    assert rpc.queue.events() == events_before


def test_show_preserves_job_when_probe_returns_unhealthy_snapshot(tmp_path) -> None:
    probe = ExplanationProbe(gpu_snapshot(healthy=False))
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=lambda: (True, "ready"),
        resource_probe=probe,
    )
    record = rpc.queue.enqueue(
        JobSpec.model_validate(job_spec()),
        run_id="waiting-run",
        now=NOW,
    )

    response = rpc.handle(envelope("show", {"run_id": record.run_id, "explain": True}))

    assert response["ok"] is True
    assert response["result"]["job"] == record.model_dump(
        mode="json", exclude={"exit_code", "launch_error"}
    )
    assert response["result"]["explanation"]["code"] == "unknown"
    assert response["result"]["explanation"]["details"] == {"error": "GPU probe failed"}


def test_show_explains_nonqueued_states_without_worker_or_gpu_probes(tmp_path) -> None:
    probe = ExplanationProbe(error=AssertionError("GPU probe should not run"))

    def unexpected_readiness() -> tuple[bool, str]:
        raise AssertionError("readiness should not be checked")

    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        clock=lambda: NOW,
        readiness_probe=unexpected_readiness,
        resource_probe=probe,
    )

    def leased_job(run_id: str, gpu_index: int):
        queued = rpc.queue.enqueue(
            JobSpec.model_validate(job_spec(job_id=run_id)),
            run_id=run_id,
            now=NOW,
        )
        return rpc.queue.lease(
            queued.run_id,
            server_id="gpu01",
            gpu_indices=[gpu_index],
            worker_id="worker-1",
            ttl=timedelta(seconds=30),
            expected_revision=queued.revision,
            now=NOW,
        )

    leased = leased_job("leased-run", 0)
    starting = rpc.queue.transition(
        leased_job("starting-run", 1).run_id,
        JobStatus.STARTING,
        now=NOW,
    )
    running = rpc.queue.transition(
        rpc.queue.transition(
            leased_job("running-run", 2).run_id,
            JobStatus.STARTING,
            now=NOW,
        ).run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=4242, started_at=NOW),
        now=NOW,
    )
    cancelling_running = rpc.queue.transition(
        rpc.queue.transition(
            leased_job("cancel-requested-run", 3).run_id,
            JobStatus.STARTING,
            now=NOW,
        ).run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=4243, started_at=NOW),
        now=NOW,
    )
    cancel_requested = rpc.queue.request_cancel(cancelling_running.run_id, now=NOW)
    failing_running = rpc.queue.transition(
        rpc.queue.transition(
            leased_job("failed-run", 4).run_id,
            JobStatus.STARTING,
            now=NOW,
        ).run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=4244, started_at=NOW),
        now=NOW,
    )
    failed = rpc.queue.transition(
        failing_running.run_id,
        JobStatus.FAILED,
        failure_reason="process exited with code 17",
        now=NOW,
    )
    cancelled_queued = rpc.queue.enqueue(
        JobSpec.model_validate(job_spec(job_id="cancelled")),
        run_id="cancelled-run",
        now=NOW,
    )
    cancelled = rpc.queue.request_cancel(cancelled_queued.run_id, now=NOW)

    records = (leased, starting, running, cancel_requested, failed, cancelled)
    explanations = {
        record.status: rpc.handle(envelope("show", {"run_id": record.run_id, "explain": True}))[
            "result"
        ]["explanation"]
        for record in records
    }

    assert set(explanations) == {
        JobStatus.LEASED,
        JobStatus.STARTING,
        JobStatus.RUNNING,
        JobStatus.CANCEL_REQUESTED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    for status, explanation in explanations.items():
        assert explanation["code"] == status.value
        assert explanation["waiting"] is (status in {JobStatus.LEASED, JobStatus.STARTING})
    assert explanations[JobStatus.RUNNING]["next_action"] == "inspect_logs"
    assert explanations[JobStatus.FAILED]["details"]["failure_reason"] == (
        "process exited with code 17"
    )
    assert explanations[JobStatus.FAILED]["next_action"] == "inspect_logs"
    assert probe.calls == 0


def test_logs_return_bounded_tail_across_console_log_files(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01", max_log_tail_bytes=10)
    submit(rpc)
    older = rpc.storage.prepare_log_file("run-001/console.log.1")
    current = rpc.storage.prepare_log_file("run-001/console.log")
    older.write_bytes(b"older\n")
    current.write_bytes(b"newest\n")
    os.utime(older, ns=(1, 1))
    os.utime(current, ns=(2, 2))

    response = rpc.handle(envelope("logs", {"run_id": "run-001", "tail_bytes": 10}))

    assert response["ok"] is True
    result = response["result"]
    assert result.pop("modified_at") == current.stat().st_mtime
    assert result.pop("sampled_at") >= current.stat().st_mtime
    assert isinstance(result.pop("utc_offset_seconds"), (float, int))
    assert result.pop("source") == str(current)
    assert result == {
        "run_id": "run-001",
        "content": "er\nnewest\n",
        "returned_bytes": 10,
        "truncated": True,
        "files": ["console.log.1", "console.log"],
    }


def test_logs_reject_oversized_tail_and_path_escape(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01", max_log_tail_bytes=8)
    submit(rpc)

    oversized = rpc.handle(envelope("logs", {"run_id": "run-001", "tail_bytes": 9}))
    traversal = rpc.handle(envelope("logs", {"run_id": "../outside", "tail_bytes": 8}))

    assert oversized["error"]["code"] == "invalid_params"
    assert traversal["error"]["code"] == "invalid_params"


def test_logs_never_follow_symlinks(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    submit(rpc)
    external = tmp_path.parent / "private-experiment.log"
    external.write_text("do not expose", encoding="utf-8")
    run_directory = rpc.logs_directory / "run-001"
    run_directory.mkdir(mode=0o700)
    (run_directory / "console.log").symlink_to(external)

    response = rpc.handle(envelope("logs", {"run_id": "run-001"}))

    assert response["ok"] is False
    assert response["error"]["code"] == "log_not_found"
    assert "do not expose" not in json.dumps(response)


def test_health_reports_layout_usage_and_injected_budgets(tmp_path) -> None:
    rpc = WorkerRpc(
        tmp_path,
        server_id="gpu01",
        total_budget_bytes=10_000,
        log_budget_bytes=7_000,
        readiness_probe=lambda: (True, "ready"),
    )

    response = rpc.handle(envelope("health"))

    assert response["ok"] is True
    assert response["result"]["status"] == "ok"
    assert response["result"]["server_id"] == "gpu01"
    assert response["result"]["durable"] is True
    assert response["result"]["worker"] == {"ready": True, "reason": "ready"}
    assert response["result"]["storage"]["total_budget_bytes"] == 10_000
    assert response["result"]["storage"]["log_budget_bytes"] == 7_000
    assert response["result"]["paths"] == {
        "queue": "queue",
        "runs": "runs",
        "logs": "logs",
    }


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "invalid_envelope"),
        (envelope("terminate", {"run_id": "run-001"}), "method_not_found"),
        ({**envelope("health"), "extra": True}, "invalid_envelope"),
        ({**envelope("health"), "params": []}, "invalid_envelope"),
    ],
)
def test_fixed_envelope_rejects_unsupported_requests(tmp_path, payload, code) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    response = rpc.handle(payload)

    assert response["ok"] is False
    assert response["error"]["code"] == code


def test_invalid_json_and_invalid_job_spec_have_stable_errors(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    invalid_json = rpc.handle_json("{")
    invalid_spec = submit(rpc, spec={"id": "missing-most-fields"})

    assert invalid_json["error"]["code"] == "invalid_envelope"
    assert invalid_spec["error"]["code"] == "invalid_params"
    assert invalid_spec["error"]["details"]["errors"]
    json.dumps(invalid_json)
    json.dumps(invalid_spec)


def test_invalid_utf8_has_the_same_stable_error_envelope(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    response = rpc.handle_json(b'\xff{"method":"health"}')

    assert response["api_version"] == RPC_API_VERSION
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_envelope"


def test_direct_parameter_validation_uses_explicit_exception(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    with pytest.raises(InvalidParams, match="unexpected parameters"):
        rpc.health({"unexpected": True})


def test_cancel_is_durable_and_idempotent(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    submit(rpc)

    first = rpc.handle(envelope("cancel", {"run_id": "run-001", "reason": "superseded"}))
    second = rpc.handle(envelope("cancel", {"run_id": "run-001"}))

    assert first["ok"] is True
    assert first["result"]["durable"] is True
    assert first["result"]["job"]["status"] == "cancelled"
    assert second["result"]["job"] == first["result"]["job"]
    assert len(rpc.queue.events()) == 2


def test_cancel_unknown_job_returns_defined_error(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    response = rpc.handle(envelope("cancel", {"run_id": "missing"}))

    assert response["ok"] is False
    assert response["error"]["code"] == "job_not_found"


def test_submit_and_health_fail_closed_without_a_live_worker(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")

    health = rpc.handle(envelope("health"))
    response = rpc.handle(envelope("submit", {"run_id": "stopped-run", "job_spec": job_spec()}))

    assert health["ok"] is True
    assert health["result"]["status"] == "unavailable"
    assert health["result"]["durable"] is False
    assert response["ok"] is False
    assert response["error"]["code"] == "worker_unavailable"
    assert rpc.queue.list() == []


@pytest.mark.parametrize(
    "params",
    [
        {"limit": True},
        {"limit": 0},
        {"limit": 102},
        {"limit": "50"},
        {"project": []},
        {"project": ""},
        {"cursor": []},
        {"cursor": "invalid"},
    ],
)
def test_history_rejects_invalid_params(tmp_path, params) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    result = rpc.handle(envelope("history", params))
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_params"


def test_summary_list_requires_a_boolean(tmp_path) -> None:
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    result = rpc.handle(envelope("list", {"summary": "true"}))
    assert result["error"]["code"] == "invalid_params"


def test_warm_health_counts_jobs_without_reading_any_job_json(tmp_path, monkeypatch):
    from gpuops.core.queue import QueueStore

    rpc = WorkerRpc(tmp_path, server_id="gpu01", readiness_probe=lambda: (True, "ready"))
    for i in range(20):
        rpc.queue.enqueue(
            JobSpec.model_validate(job_spec(job_id=f"health-{i}")), run_id=f"health-{i}"
        )
        rpc.queue.request_cancel(f"health-{i}")
    first = rpc.health({})
    assert first["queue"]["jobs"] == 20
    assert first["storage"]["total_bytes"] == rpc.storage.usage().total_bytes

    def no_records(*args, **kwargs):
        raise AssertionError("health must not deserialize queue records")

    monkeypatch.setattr(QueueStore, "_all_unlocked", no_records)
    monkeypatch.setattr(QueueStore, "list", no_records)
    monkeypatch.setattr(QueueStore, "summaries", no_records)
    reopened = WorkerRpc(tmp_path, server_id="gpu01", readiness_probe=lambda: (True, "ready"))
    response = reopened.handle(envelope("health"))
    assert response["ok"] is True
    assert response["result"]["queue"] == {"jobs": 20}


def test_empty_rotated_log_does_not_redate_previous_progress(tmp_path):
    rpc = WorkerRpc(tmp_path, server_id="gpu01")
    submit(rpc)
    previous = rpc.storage.prepare_log_file("run-001/console.log.1")
    previous.write_text("epoch=2/3 seconds=100\n")
    current = rpc.storage.prepare_log_file("run-001/console.log")
    current.write_text("")
    os.utime(previous, (100, 100))
    os.utime(current, (200, 200))
    result = rpc.handle(envelope("logs", {"run_id": "run-001"}))["result"]
    assert result["modified_at"] == 100
    assert result["content"] == previous.read_text()
    assert result["source"] == str(previous)
