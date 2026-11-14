from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gpuops import __version__
from gpuops.control import DefinitiveRemoteSubmissionError, RemoteTarget
from gpuops.models import JobSpec, JobStatus
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
    assert all(call[2]["release"] == __version__ for call in ssh.calls)


def test_submit_sends_email_off_without_capability_probe() -> None:
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
        return response(
            request,
            {
                "durable": True,
                "job": job_payload(request["params"]["run_id"])
                | {"spec": request["params"]["job_spec"]},
            },
        )

    import asyncio

    job_spec = spec().model_copy(update={"metadata": {"gpuops.email": "off"}})
    ssh = FakeSsh(factory)
    record = asyncio.run(
        SshWorkerTransport(ssh=ssh).submit(
            RemoteTarget("gpu01", "lab-gpu01"), "silent-run", job_spec
        )
    )

    assert record.spec.metadata["gpuops.email"] == "off"
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


@pytest.mark.parametrize("matching", [True, False])
def test_start_checks_installed_release_without_requiring_a_running_worker(matching) -> None:
    import asyncio

    def factory(request):
        if request is None:
            return {"ok": True, "status": "starting"}
        assert request["method"] == "health"
        assert request["release"] == __version__
        if matching:
            return response(request, {"status": "unavailable", "worker": {"ready": False}})
        return {
            "api_version": RPC_API_VERSION,
            "ok": False,
            "error": {"code": "invalid_envelope", "message": "unexpected envelope fields: release"},
        }

    ssh = FakeSsh(factory)
    transport = SshWorkerTransport(ssh=ssh)
    target = RemoteTarget("gpu01", "lab-gpu01")
    if matching:
        assert asyncio.run(transport.worker_command(target, "start"))["status"] == "starting"
        assert ssh.calls[1][1] == [
            "~/.local/bin/gpuops",
            "worker",
            "start",
            "--server-id",
            "gpu01",
            "--json",
        ]
    else:
        with pytest.raises(RemoteWorkerError) as caught:
            asyncio.run(transport.worker_command(target, "start"))
        assert caught.value.code == "version_mismatch"
        assert len(ssh.calls) == 1
        asyncio.run(transport.worker_command(target, "stop"))
        assert ssh.calls[1][1] == ["~/.local/bin/gpuops", "worker", "stop", "--json"]


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


@pytest.mark.parametrize("source", ["older_envelope", "installed_cli", "running_worker"])
def test_submit_classifies_release_mismatch_as_definitive(source) -> None:
    import asyncio

    def factory(request):
        assert request["release"] == __version__
        error = "GPUOps versions must match; run gpuops deploy install --server gpu01"
        if source == "running_worker":
            return response(
                request,
                {
                    "status": "unavailable",
                    "durable": False,
                    "worker": {"ready": False, "error_code": "version_mismatch", "reason": error},
                },
            )
        if source == "installed_cli":
            return {
                "api_version": RPC_API_VERSION,
                "request_id": request["request_id"],
                "ok": False,
                "error": {"code": "version_mismatch", "message": error},
            }
        return {
            "api_version": RPC_API_VERSION,
            "ok": False,
            "error": {"code": "invalid_envelope", "message": "unexpected envelope fields: release"},
        }

    ssh = FakeSsh(factory)
    with pytest.raises(
        RemoteSubmissionRejected, match="gpuops deploy install --server gpu01"
    ) as caught:
        asyncio.run(
            SshWorkerTransport(ssh=ssh).submit(
                RemoteTarget("gpu01", "lab-gpu01"), "new-run", spec()
            )
        )

    assert caught.value.code == "version_mismatch"
    assert [call[2]["method"] for call in ssh.calls] == ["health"]


def test_logs_missing_sample_time_is_a_protocol_error() -> None:
    import asyncio

    ssh = FakeSsh(
        lambda request: response(
            request,
            {
                "content": "",
                "source": None,
                "modified_at": None,
                "truncated": False,
                "utc_offset_seconds": 0,
            },
        )
    )
    with pytest.raises(RemoteWorkerError, match="sampled_at") as caught:
        asyncio.run(SshWorkerTransport(ssh=ssh).get_logs(RemoteTarget("gpu01", "lab-gpu01"), "run"))
    assert caught.value.code == "protocol_error"


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


def test_job_detail_rejects_a_malformed_explanation() -> None:
    def factory(request):
        return response(
            request,
            {
                "job": job_payload(request["params"]["run_id"]),
                "explanation": {"waiting": "yes", "message": ["invalid"]},
            },
        )

    import asyncio

    with pytest.raises(RemoteWorkerError, match="invalid job explanation") as captured:
        asyncio.run(
            SshWorkerTransport(ssh=FakeSsh(factory)).get_job_detail(
                RemoteTarget("gpu01", "lab-gpu01"), "malformed-explanation"
            )
        )
    assert captured.value.code == "protocol_error"


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


@pytest.mark.parametrize("method", ["list", "show"])
def test_query_surfaces_invalid_params_without_retrying(method) -> None:
    import asyncio

    def factory(request):
        assert request["method"] == method
        return {
            "api_version": RPC_API_VERSION,
            "request_id": request["request_id"],
            "ok": False,
            "error": {"code": "invalid_params", "message": "query parameters are invalid"},
        }

    ssh = FakeSsh(factory)
    transport = SshWorkerTransport(ssh=ssh)
    target = RemoteTarget("gpu01", "lab-gpu01")
    with pytest.raises(RemoteWorkerError, match="query parameters are invalid") as captured:
        if method == "list":
            asyncio.run(transport.list_summaries(target, statuses={JobStatus.QUEUED}))
        else:
            asyncio.run(transport.get_job_detail(target, "queued-run"))
    assert captured.value.code == "invalid_params"
    assert len(ssh.calls) == 1


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
