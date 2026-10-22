from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gpuops.control import DefinitiveRemoteSubmissionError, RemoteTarget
from gpuops.models import JobSpec
from gpuops.ssh import SSHResult
from gpuops.transport import (
    RemoteSubmissionRejected,
    RemoteWorkerError,
    SshWorkerTransport,
)
from gpuops.worker.rpc import RPC_API_VERSION


class FakeSsh:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.calls = []

    async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
        request = json.loads(stdin) if stdin else None
        self.calls.append((host, command, request, timeout_seconds))
        payload = self.response_factory(request)
        return SSHResult(host, 0, json.dumps(payload), "", 0.01)


class PrettyWorkerSsh:
    async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
        del command, stdin, timeout_seconds
        return SSHResult(
            host,
            0,
            json.dumps({"ok": True, "status": "running", "ready": True}, indent=2),
            "",
            0.01,
        )


def spec(host: str = "gpu01") -> JobSpec:
    return JobSpec(
        id="transport-test",
        project="tests",
        command=["python", "train.py"],
        working_directory="/work",
        resources={"host": host},
    )


def response(request, result):
    return {
        "api_version": RPC_API_VERSION,
        "request_id": request["request_id"],
        "ok": True,
        "result": result,
    }


def job_payload(run_id: str, *, status: str = "queued") -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "spec": spec().model_dump(mode="json"),
        "status": status,
        "revision": 0,
        "created_at": now,
        "updated_at": now,
        "allocation": None,
        "launch": None,
        "finished_at": None,
        "failure_reason": None,
    }


def test_submit_uses_fixed_remote_command_and_validates_record() -> None:
    job_spec = spec()

    def factory(request):
        if request["method"] == "health":
            return response(
                request,
                {
                    "status": "ok",
                    "durable": True,
                    "worker": {"ready": True, "reason": "ready"},
                },
            )
        record = {
            "run_id": request["params"]["run_id"],
            "spec": request["params"]["job_spec"],
            "status": "queued",
            "revision": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "allocation": None,
            "launch": None,
            "finished_at": None,
            "failure_reason": None,
        }
        return response(request, {"durable": True, "job": record})

    ssh = FakeSsh(factory)
    transport = SshWorkerTransport(ssh=ssh)
    target = RemoteTarget("gpu01", "lab-gpu01")

    import asyncio

    record = asyncio.run(transport.submit(target, "stable-run", job_spec))

    assert record.run_id == "stable-run"
    assert ssh.calls[1][1] == [
        "~/.local/bin/gpuops",
        "remote",
        "--server-id",
        "gpu01",
    ]
    assert [call[2]["method"] for call in ssh.calls] == ["health", "submit"]


def test_worker_command_accepts_pretty_printed_json_document() -> None:
    import asyncio

    result = asyncio.run(
        SshWorkerTransport(ssh=PrettyWorkerSsh()).worker_command(
            RemoteTarget("gpu01", "lab-gpu01"),
            "status",
        )
    )

    assert result == {"ok": True, "status": "running", "ready": True}


def test_submit_rejects_a_stale_or_degraded_worker_before_enqueue() -> None:
    def factory(request):
        return response(
            request,
            {
                "status": "unavailable",
                "durable": False,
                "worker": {"ready": False, "reason": "heartbeat is stale"},
            },
        )

    import asyncio

    with pytest.raises(RemoteSubmissionRejected, match="heartbeat is stale") as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).submit(
                RemoteTarget("gpu01", "lab-gpu01"),
                "stale-run",
                spec(),
            )
        )
    assert captured.value.code == "worker_unavailable"
    assert isinstance(captured.value, RemoteWorkerError)
    assert isinstance(captured.value, DefinitiveRemoteSubmissionError)


def test_submit_preserves_definitive_remote_rejection_code() -> None:
    def factory(request):
        if request["method"] == "health":
            return response(
                request,
                {
                    "status": "ok",
                    "durable": True,
                    "worker": {"ready": True, "reason": "ready"},
                },
            )
        return {
            "api_version": RPC_API_VERSION,
            "request_id": request["request_id"],
            "ok": False,
            "error": {"code": "invalid_params", "message": "invalid job_spec"},
        }

    import asyncio

    with pytest.raises(RemoteSubmissionRejected) as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).submit(
                RemoteTarget("gpu01", "lab-gpu01"),
                "invalid-run",
                spec(),
            )
        )

    assert captured.value.code == "invalid_params"
    assert isinstance(captured.value, RemoteWorkerError)


def test_submit_keeps_protocol_errors_ambiguous() -> None:
    def factory(request):
        if request["method"] == "health":
            return response(
                request,
                {
                    "status": "ok",
                    "durable": True,
                    "worker": {"ready": True, "reason": "ready"},
                },
            )
        return {
            "api_version": "unexpected-version",
            "request_id": request["request_id"],
            "ok": False,
            "error": {"code": "invalid_params", "message": "invalid job_spec"},
        }

    import asyncio

    with pytest.raises(RemoteWorkerError) as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).submit(
                RemoteTarget("gpu01", "lab-gpu01"),
                "unclear-run",
                spec(),
            )
        )

    assert captured.value.code == "protocol_error"
    assert not isinstance(captured.value, RemoteSubmissionRejected)


def test_submit_keeps_storage_quota_errors_ambiguous() -> None:
    def factory(request):
        if request["method"] == "health":
            return response(
                request,
                {
                    "status": "ok",
                    "durable": True,
                    "worker": {"ready": True, "reason": "ready"},
                },
            )
        return {
            "api_version": RPC_API_VERSION,
            "request_id": request["request_id"],
            "ok": False,
            "error": {
                "code": "storage_quota_exceeded",
                "message": "event capacity exceeded",
            },
        }

    import asyncio

    with pytest.raises(RemoteWorkerError) as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).submit(
                RemoteTarget("gpu01", "lab-gpu01"),
                "quota-run",
                spec(),
            )
        )

    assert captured.value.code == "storage_quota_exceeded"
    assert not isinstance(captured.value, RemoteSubmissionRejected)


def test_missing_job_returns_none() -> None:
    def factory(request):
        return {
            "api_version": RPC_API_VERSION,
            "request_id": request["request_id"],
            "ok": False,
            "error": {"code": "job_not_found", "message": "missing"},
        }

    import asyncio

    result = asyncio.run(
        SshWorkerTransport(ssh=FakeSsh(factory)).get_job(
            RemoteTarget("gpu01", "lab-gpu01"), "missing-run"
        )
    )

    assert result is None


def test_job_detail_requests_explanation_and_preserves_probe_failure() -> None:
    explanation = {
        "waiting": True,
        "code": "resource_probe_failed",
        "message": "GPU telemetry is unavailable",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "details": {"error": "nvidia-smi timed out"},
        "next_action": "Retry after the host probe recovers",
    }

    def factory(request):
        return response(
            request,
            {"job": job_payload(request["params"]["run_id"]), "explanation": explanation},
        )

    import asyncio

    ssh = FakeSsh(factory)
    detail = asyncio.run(
        SshWorkerTransport(ssh=ssh).get_job_detail(
            RemoteTarget("gpu01", "lab-gpu01"), "waiting-run"
        )
    )

    assert detail is not None
    assert detail.record.run_id == "waiting-run"
    assert detail.explanation == explanation
    assert ssh.calls[0][2]["params"] == {"run_id": "waiting-run", "explain": True}


def test_job_detail_falls_back_to_plain_show_for_an_old_worker() -> None:
    def factory(request):
        if request["params"].get("explain") is True:
            return {
                "api_version": RPC_API_VERSION,
                "request_id": request["request_id"],
                "ok": False,
                "error": {"code": "invalid_params", "message": "unexpected parameter: explain"},
            }
        job = job_payload(request["params"]["run_id"])
        job["updated_at"] = "2000-01-01T00:00:00+00:00"
        return response(request, {"job": job})

    import asyncio

    ssh = FakeSsh(factory)
    detail = asyncio.run(
        SshWorkerTransport(ssh=ssh).get_job_detail(RemoteTarget("gpu01", "lab-gpu01"), "legacy-run")
    )

    assert detail is not None
    assert detail.record.run_id == "legacy-run"
    assert detail.explanation["code"] == "unknown"
    assert detail.explanation["observed_at"] != detail.record.updated_at.isoformat()
    assert [call[2]["params"] for call in ssh.calls] == [
        {"run_id": "legacy-run", "explain": True},
        {"run_id": "legacy-run"},
    ]


def test_job_detail_degrades_a_malformed_explanation_to_unknown() -> None:
    def factory(request):
        return response(
            request,
            {
                "job": job_payload(request["params"]["run_id"]),
                "explanation": {"waiting": "yes", "message": ["invalid"]},
            },
        )

    import asyncio

    detail = asyncio.run(
        SshWorkerTransport(ssh=FakeSsh(factory)).get_job_detail(
            RemoteTarget("gpu01", "lab-gpu01"), "malformed-explanation"
        )
    )

    assert detail is not None
    assert detail.record.run_id == "malformed-explanation"
    assert detail.explanation["code"] == "unknown"


def test_job_detail_rejects_a_record_for_another_run() -> None:
    def factory(request):
        return response(
            request,
            {"job": job_payload("different-run"), "explanation": None},
        )

    import asyncio

    with pytest.raises(RemoteWorkerError, match="different run_id") as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).get_job_detail(
                RemoteTarget("gpu01", "lab-gpu01"), "requested-run"
            )
        )
    assert captured.value.code == "protocol_error"


def test_cancel_sends_fixed_rpc_and_validates_durable_record() -> None:
    def factory(request):
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "run_id": request["params"]["run_id"],
            "spec": spec().model_dump(mode="json"),
            "status": "cancelled",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
            "allocation": None,
            "launch": None,
            "finished_at": now,
            "failure_reason": None,
        }
        return response(request, {"durable": True, "job": record})

    import asyncio

    ssh = FakeSsh(factory)
    record = asyncio.run(
        SshWorkerTransport(ssh=ssh).cancel(
            RemoteTarget("gpu01", "lab-gpu01"),
            "stable-run",
            reason="superseded",
        )
    )

    assert record.status.value == "cancelled"
    assert ssh.calls[0][2]["method"] == "cancel"
    assert ssh.calls[0][2]["params"] == {
        "run_id": "stable-run",
        "reason": "superseded",
    }


def test_summary_fallback_keeps_status_filter_on_older_workers() -> None:
    import asyncio

    from gpuops.models import JobStatus

    def factory(request):
        if request["params"].get("summary"):
            return {
                "api_version": RPC_API_VERSION,
                "request_id": request["request_id"],
                "ok": False,
                "error": {"code": "invalid_params", "message": "unknown summary parameter"},
            }
        return response(request, {"jobs": [job_payload("queued-run")]})

    ssh = FakeSsh(factory)
    jobs = asyncio.run(
        SshWorkerTransport(ssh=ssh).list_summaries(
            RemoteTarget("gpu01", "lab-gpu01"), statuses={JobStatus.QUEUED}
        )
    )
    assert jobs[0].run_id == "queued-run"
    assert not hasattr(jobs[0], "spec")
    assert [call[2]["params"] for call in ssh.calls] == [
        {"summary": True, "statuses": ["queued"]},
        {"statuses": ["queued"]},
    ]


def test_host_inspection_only_requests_active_records() -> None:
    import asyncio

    from gpuops.models import ACTIVE_JOB_STATUSES

    class Collector:
        async def collect(self, host, *, timeout_seconds):
            return {"collected_at": datetime.now(timezone.utc).isoformat(), "gpus": []}, 0.01

    def factory(request):
        if request["method"] == "health":
            return response(
                request,
                {"status": "ok", "durable": True, "worker": {"ready": True, "reason": "ready"}},
            )
        assert request["method"] == "list"
        assert request["params"] == {
            "statuses": sorted(status.value for status in ACTIVE_JOB_STATUSES)
        }
        return response(request, {"jobs": [job_payload("active-run")]})

    result = asyncio.run(
        SshWorkerTransport(ssh=FakeSsh(factory), collector=Collector()).inspect(
            RemoteTarget("gpu01", "lab-gpu01")
        )
    )
    assert [record.run_id for record in result.queue] == ["active-run"]
