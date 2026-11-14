from __future__ import annotations

import json
from io import BytesIO, TextIOWrapper

import pytest

from gpuops import __version__
from gpuops.cli import main
from gpuops.control import ReconcileReport
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now


def test_validate_job_manifest(capsys, tmp_path) -> None:
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps(
            {
                "api_version": "gpuops/v1",
                "id": "smoke-test",
                "project": "tests",
                "command": ["python", "train.py"],
                "working_directory": "/work",
            }
        ),
        encoding="utf-8",
    )

    assert main(["validate", str(manifest), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_validate_job_manifest_reports_structured_error(capsys, tmp_path) -> None:
    manifest = tmp_path / "job.json"
    manifest.write_text('{"id":"bad id"}', encoding="utf-8")

    assert main(["validate", str(manifest), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False
    assert "validation error" in result["error"]


def test_validate_minimal_exact_gpu_manifest(capsys, tmp_path) -> None:
    manifest = tmp_path / "job.json"
    spec = JobSpec(
        id="minimal", project="tests", command=["true"], working_directory="/work"
    ).model_dump(mode="json")
    spec["resources"] = {"host": "gpu01", "gpus": [1, 0]}
    manifest.write_text(json.dumps(spec))

    assert main(["validate", str(manifest), "--json"]) == 0
    resources = json.loads(capsys.readouterr().out)["job"]["resources"]
    assert resources["gpu_count"] == 2
    assert resources["gpus"] == [0, 1]


def test_validate_rejects_unsupported_tracking(capsys, tmp_path) -> None:
    spec = JobSpec(
        id="tracking",
        project="tests",
        command=["true"],
        working_directory="/work",
        tracking={"progress_adapter": "imaginary"},
    )
    manifest = tmp_path / "job.json"
    manifest.write_text(spec.model_dump_json())

    assert main(["validate", str(manifest), "--json"]) == 2
    assert "progress_adapter" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize(
    "command,selection",
    [("status", ["--server", "gpu02"]), ("doctor", ["--server", "gpu02"])],
)
def test_monitor_server_selection_uses_registry(monkeypatch, command, selection) -> None:
    captured = []

    async def run_remote(args, targets):
        captured.extend((target.id, target.ssh_host) for target in targets)
        return 0

    monkeypatch.setattr("gpuops.cli._run_remote_command", run_remote)
    assert main([command, *selection, "--json"]) == 0
    assert captured == [("gpu02", "lab-gpu02")]


@pytest.mark.parametrize("selection", [["--server", "gpu02"], ["--ssh-alias", "lab-gpu02"]])
def test_deployment_accepts_registered_server_or_direct_ssh_alias(
    monkeypatch, capsys, selection
) -> None:
    from dataclasses import dataclass

    calls = []

    @dataclass
    class Status:
        current_release: str = "test-release"

    class Manager:
        async def status(self, host):
            calls.append(host)
            return Status()

    monkeypatch.setattr("gpuops.deploy.RemoteDeploymentManager", Manager)
    assert main(["deploy", "status", *selection, "--json"]) == 0
    assert calls == ["lab-gpu02"]
    assert json.loads(capsys.readouterr().out)[0]["ok"] is True


def test_queue_server_selection_uses_shared_mapping(monkeypatch, capsys) -> None:
    calls = []

    class Transport:
        async def list_summaries(self, target, *, statuses):
            calls.append((target.server_id, target.ssh_host))
            return []

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: object())
    assert main(["queue", "--server", "gpu02", "--json"]) == 0
    assert calls == [("gpu02", "lab-gpu02")]
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_web_command_rejects_public_bind(capsys) -> None:
    assert main(["web", "--bind", "0.0.0.0"]) == 2
    assert "loopback" in capsys.readouterr().err


def test_web_command_rejects_non_positive_timeout(capsys) -> None:
    assert main(["web", "--timeout", "0"]) == 2
    assert "timeout must be positive" in capsys.readouterr().err


def test_worker_status_is_machine_readable(capsys, tmp_path) -> None:
    assert main(["worker", "status", "--state-dir", str(tmp_path), "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": True,
        "status": "stopped",
        "release": None,
        "ready": False,
        "reason": "worker process is stopped",
        "identity": None,
        "heartbeat": None,
    }


@pytest.mark.parametrize(
    "command,interval,as_json",
    [("run", "0", True), ("run", "-1", False), ("start", "-1", True), ("start", "0", False)],
)
def test_worker_rejects_non_positive_interval(
    command, interval, as_json, monkeypatch, capsys, tmp_path
) -> None:
    def unexpected_start(**kwargs):
        pytest.fail("an invalid interval must not start a Worker")

    monkeypatch.setattr(f"gpuops.worker.app.{command}_worker", unexpected_start)
    arguments = [
        "worker",
        command,
        "--server-id",
        "gpu01",
        "--state-dir",
        str(tmp_path / "state"),
        "--interval",
        interval,
    ]
    if as_json:
        arguments.append("--json")

    assert main(arguments) == 2
    captured = capsys.readouterr()
    if as_json:
        assert json.loads(captured.out) == {
            "ok": False,
            "error_code": "invalid_argument",
            "error": "--interval must be positive",
        }
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert captured.err == "gpuops: --interval must be positive\n"
    assert not (tmp_path / "state").exists()


def test_remote_rpc_reads_one_json_request(monkeypatch, capsys, tmp_path) -> None:
    request = {
        "api_version": "gpuops.worker.rpc/v1",
        "release": __version__,
        "method": "health",
        "params": {},
    }
    stream = TextIOWrapper(BytesIO(json.dumps(request).encode()), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", stream)

    assert (
        main(
            [
                "remote",
                "--server-id",
                "gpu01",
                "--state-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is True
    assert response["result"]["server_id"] == "gpu01"


def test_cancel_command_locates_job_and_forwards_reason(monkeypatch, capsys) -> None:
    spec = JobSpec(
        id="cancel-me",
        project="tests",
        command=["python", "train.py"],
        working_directory="/work",
        resources={"host": "gpu02", "gpu_count": 1},
    )
    queued = JobRecord(
        run_id="run-cancel-me",
        spec=spec,
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    class FakeTransport:
        def __init__(self) -> None:
            self.cancel_calls = []

        async def get_job(self, target, run_id):  # type: ignore[no-untyped-def]
            return queued if target.server_id == "gpu02" and run_id == queued.run_id else None

        async def cancel(self, target, run_id, *, reason=None):  # type: ignore[no-untyped-def]
            self.cancel_calls.append((target.server_id, run_id, reason))
            return queued.model_copy(
                update={"status": JobStatus.CANCELLED, "revision": queued.revision + 1}
            )

    transport = FakeTransport()
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: transport)
    monkeypatch.setattr(
        "gpuops.control.SubmissionJournal",
        lambda _path: object(),
    )

    assert (
        main(
            [
                "cancel",
                queued.run_id,
                "--reason",
                "superseded",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["server_id"] == "gpu02"
    assert payload["job"]["status"] == "cancelled"
    assert transport.cancel_calls == [("gpu02", queued.run_id, "superseded")]


@pytest.mark.parametrize(
    ("failures", "expected_code", "expected_ok", "expected_outcome"),
    [
        (set(), 0, True, "success"),
        ({"gpu02"}, 1, False, "partial"),
        ({"gpu01", "gpu02"}, 1, False, "failed"),
    ],
)
def test_queue_command_reports_aggregate_outcome(
    monkeypatch,
    capsys,
    failures,
    expected_code,
    expected_ok,
    expected_outcome,
) -> None:
    class FakeTransport:
        async def list_summaries(self, target, *, statuses):  # type: ignore[no-untyped-def]
            if target.server_id in failures:
                raise OSError(f"{target.server_id} unavailable")
            return []

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", FakeTransport)
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: object())

    assert main(["queue", "--json"]) == expected_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is expected_ok
    assert payload["outcome"] == expected_outcome
    assert [row["online"] for row in payload["servers"]] == [
        server_id not in failures for server_id in ("gpu01", "gpu02")
    ]


def test_reconcile_pending_is_not_reported_as_success(monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self, targets, transport, journal):  # type: ignore[no-untyped-def]
            del targets, transport, journal

        async def reconcile_pending(self):  # type: ignore[no-untyped-def]
            return ReconcileReport(
                accepted=("run-ok",),
                pending=("run-pending",),
                errors={"run-pending": "gpu02 unavailable"},
            )

    monkeypatch.setattr("gpuops.control.ControlClient", FakeClient)
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: object())
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: object())

    assert main(["reconcile", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["outcome"] == "partial"
    assert payload["pending"] == ["run-pending"]
    assert payload["errors"] == {"run-pending": "gpu02 unavailable"}


@pytest.mark.parametrize("mismatch", [False, True])
def test_job_lookup_distinguishes_unavailable_hosts_from_missing_job(
    monkeypatch, capsys, mismatch
) -> None:
    class FakeTransport:
        async def get_job(self, target, run_id):  # type: ignore[no-untyped-def]
            del run_id
            if target.server_id == "gpu02":
                if mismatch:
                    from gpuops.transport import RemoteWorkerError

                    raise RemoteWorkerError("version_mismatch", "upgrade gpu02")
                raise OSError("SSH timed out")
            return None

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", FakeTransport)
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: object())

    assert main(["job", "run-unknown", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    if mismatch:
        assert payload["error_code"] == "version_mismatch"
        assert "gpu02: upgrade gpu02" in payload["error"]
        return
    assert "could not be located" in payload["error"]
    assert "gpu02: SSH timed out" in payload["error"]
    assert "not found on: gpu01" in payload["error"]


@pytest.mark.parametrize("arguments", [["queue", "--all"], ["status", "--host", "gpu01=host"]])
def test_removed_cli_options_are_rejected(arguments, capsys):
    assert main([*arguments, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_argument"
