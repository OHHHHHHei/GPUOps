from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from conftest import TEST_TARGETS

from gpuops.cli import main
from gpuops.config import ServerTarget, local_target, select_client_targets
from gpuops.control import RemoteTarget, SubmissionJournal
from gpuops.execution import HostClient
from gpuops.models import JobRecord, JobSpec, utc_now
from gpuops.monitor.collector import RemoteCollector
from gpuops.monitor.service import ServerResult
from gpuops.transport import JobDetail, JobHistory, SshWorkerTransport


class FakeProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"ok", stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.input: bytes | None = None

    async def communicate(self, data: bytes | None = None):
        self.input = data
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.returncode = -9


def install_context(monkeypatch, home: Path, server_id: str = "gpu01") -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    context = home / ".local" / "share" / "gpuops" / "host.json"
    context.parent.mkdir(parents=True)
    context.write_text(json.dumps({"server_id": server_id}), encoding="utf-8")
    return context


def test_local_target_is_defined_only_by_valid_install_context(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert local_target() is None

    context = home / ".local" / "share" / "gpuops" / "host.json"
    context.parent.mkdir(parents=True)
    context.write_text('{"server_id":"gpu01"}', encoding="utf-8")
    assert local_target() == TEST_TARGETS[0]

    context.write_text('{"server_id":"other"}', encoding="utf-8")
    assert local_target() == ServerTarget("other", "other", "other")

    context.write_text('{"server_id":"auto"}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.*server_id"):
        local_target()

    context.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="host.json|JSON|context"):
        local_target()


def test_client_target_defaults_are_local_on_server_and_all_on_mac(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert select_client_targets() == TEST_TARGETS

    install_context(monkeypatch, home, "gpu02")
    assert select_client_targets() == (TEST_TARGETS[1],)
    assert select_client_targets(["gpu01"]) == (TEST_TARGETS[0],)
    with pytest.raises(ValueError, match="unknown"):
        select_client_targets(["no-such-server"])


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"label": "Installed GPU"}, ServerTarget("gpu01", "Installed GPU", "lab-gpu01")),
        ({"ssh_host": "installed-route"}, ServerTarget("gpu01", "GPU 01", "installed-route")),
        (
            {"label": "Installed GPU", "ssh_host": "installed-route"},
            ServerTarget("gpu01", "Installed GPU", "installed-route"),
        ),
    ],
)
def test_installation_fields_take_precedence_over_registry(
    tmp_path, monkeypatch, fields, expected
) -> None:
    context = install_context(monkeypatch, tmp_path / "home")
    context.write_text(json.dumps({"server_id": "gpu01", **fields}), encoding="utf-8")
    assert local_target() == expected
    assert select_client_targets(["gpu01"]) == (expected,)


def test_legacy_local_identity_works_without_registry(
    tmp_path, monkeypatch, server_registry
) -> None:
    server_registry.unlink()
    install_context(monkeypatch, tmp_path / "home")
    expected = ServerTarget("gpu01", "gpu01", "gpu01")
    assert local_target() == expected
    assert select_client_targets() == (expected,)
    assert select_client_targets(["gpu01"]) == (expected,)


@pytest.mark.parametrize(
    ("server_id", "ssh_host", "has_registry"),
    [
        ("gpu01", "lab-gpu01", True),
        ("lab-a100", "lab-a100", True),
        ("gpu01", "gpu01", False),
    ],
)
def test_local_python_collector_does_not_require_ssh(
    tmp_path, monkeypatch, server_registry, server_id, ssh_host, has_registry
) -> None:
    if not has_registry:
        server_registry.unlink()
    home = tmp_path / "server-home"
    install_context(monkeypatch, home, server_id)
    target = select_client_targets([server_id])[0]
    assert target.ssh_host == ssh_host
    process = FakeProcess(stdout=b'{"gpus":[]}\n')

    async def run():
        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
        ) as create:
            result = await HostClient(ssh_binary="definitely-unavailable").run_python(
                target.ssh_host, "print('collector')"
            )
        assert create.await_args.args == (sys.executable, "-I", "-u", "-")
        assert process.input == b"print('collector')"
        assert result.stdout == '{"gpus":[]}\n'

    asyncio.run(run())


@pytest.mark.parametrize("has_registry", [True, False])
def test_local_worker_command_uses_installed_binary_and_preserves_stdin(
    tmp_path, monkeypatch, server_registry, has_registry
) -> None:
    if not has_registry:
        server_registry.unlink()
    home = tmp_path / "server-home"
    install_context(monkeypatch, home)
    target = select_client_targets(["gpu01"])[0]
    process = FakeProcess(returncode=7, stderr=b"worker stopped")
    payload = b'{"request":"unchanged"}\n'

    async def run():
        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
        ) as create:
            result = await HostClient(ssh_binary="must-not-run").run_command(
                target.ssh_host,
                ["~/.local/bin/gpuops", "remote", "--server-id", "gpu01"],
                stdin=payload,
            )
        assert create.await_count == 1
        assert create.await_args.args == (
            str(home / ".local" / "bin" / "gpuops"),
            "remote",
            "--server-id",
            "gpu01",
        )
        assert process.input == payload
        assert result.returncode == 7

    asyncio.run(run())


def test_peer_host_still_routes_through_ssh(tmp_path, monkeypatch) -> None:
    install_context(monkeypatch, tmp_path / "server-home", "gpu01")
    process = FakeProcess(stdout=b'{"ok":true}')

    async def run():
        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
        ) as create:
            await HostClient(ssh_binary="custom-ssh").run_command(
                "lab-gpu02", ["~/.local/bin/gpuops", "worker", "status", "--json"]
            )
        argv = create.await_args.args
        assert argv[0] == "custom-ssh"
        assert "lab-gpu02" in argv
        assert argv[-1] == "~/.local/bin/gpuops worker status --json"

    asyncio.run(run())


def test_default_transports_use_host_client(tmp_path, monkeypatch) -> None:
    install_context(monkeypatch, tmp_path / "home")
    transport = SshWorkerTransport()
    collector = RemoteCollector()
    assert isinstance(transport.ssh, HostClient)
    assert isinstance(transport.collector.ssh, HostClient)
    assert isinstance(collector.ssh, HostClient)


@pytest.mark.parametrize(
    ("arguments", "method"),
    [
        (["queue", "--json"], "list_summaries"),
        (["history", "--json"], "history"),
        (["fleet", "status", "--json"], "worker_command"),
        (["job", "one-run", "--json"], "get_job_detail"),
        (["logs", "one-run", "--json"], "get_logs"),
        (["cancel", "one-run", "--json"], "cancel"),
    ],
)
def test_unqualified_control_commands_only_contact_local_server(
    tmp_path, monkeypatch, capsys, arguments, method
) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    calls: list[tuple[str, str]] = []
    spec = _spec("gpu01")

    class Transport:
        async def list_summaries(self, target, *, statuses):
            calls.append(("list_summaries", target.server_id))
            return []

        async def worker_command(self, target, action):
            calls.append(("worker_command", target.server_id))
            return {"ok": True, "ready": False}

        async def history(self, target, *, limit, project, cursor):
            calls.append(("history", target.server_id))
            return JobHistory([], [])

        async def get_job(self, target, run_id):
            calls.append(("get_job", target.server_id))
            return JobRecord(run_id=run_id, spec=spec)

        async def get_logs(self, target, run_id, *, tail_bytes):
            calls.append(("get_logs", target.server_id))
            return {"content": ""}

        async def get_job_detail(self, target, run_id):
            calls.append(("get_job_detail", target.server_id))
            record = JobRecord(run_id=run_id, spec=spec)
            return JobDetail(record, {"waiting": False})

        async def cancel(self, target, run_id, *, reason=None):
            calls.append(("cancel", target.server_id))
            return JobRecord(run_id=run_id, spec=spec)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(arguments) == 0
    json.loads(capsys.readouterr().out)
    assert (method, "gpu01") in calls
    assert {server_id for _called, server_id in calls} == {"gpu01"}


@pytest.mark.parametrize("command", ["status", "doctor"])
def test_unqualified_monitor_commands_only_contact_local_server(
    tmp_path, monkeypatch, capsys, command
) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu02")
    seen = []

    class Monitor:
        def __init__(self, targets, *, timeout_seconds):
            seen.extend(target.id for target in targets)
            self.targets = targets

        async def collect_all(self):
            return [ServerResult(target, True, data={"gpus": []}) for target in self.targets]

        async def doctor_all(self):
            return [
                ServerResult(target, True, data={"gpustat_ok": True})
                for target in self.targets
            ]

    class Transport:
        def __init__(self, *, timeout_seconds):
            pass

        async def worker_command(self, target, action):
            return {
                "ready": True,
                "reason": "ready",
                "heartbeat": {"server_id": target.server_id},
            }

    monkeypatch.setattr("gpuops.cli.MonitorService", Monitor)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main([command, "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert seen == ["gpu02"]


def test_explicit_peer_server_is_available_from_server_context(
    tmp_path, monkeypatch, capsys
) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    calls = []

    class Transport:
        async def list_summaries(self, target, *, statuses):
            calls.append(target.server_id)
            return []

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["queue", "--server", "gpu02", "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert calls == ["gpu02"]


def test_explicit_peer_job_command_does_not_search_local_server(
    tmp_path, monkeypatch, capsys
) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    calls = []
    record = JobRecord(run_id="peer-job", spec=_spec("gpu02"))

    class Transport:
        async def get_job_detail(self, target, run_id):
            calls.append((target.server_id, run_id))
            return JobDetail(record, {"waiting": False})

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["job", "peer-job", "--server", "gpu02", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["server_id"] == "gpu02"
    assert calls == [("gpu02", "peer-job")]


def test_auto_submit_on_server_queues_without_gpu_inspection(tmp_path, monkeypatch, capsys) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    manifest = tmp_path / "auto.json"
    spec = _spec("auto").model_copy(
        update={"resources": _spec("auto").resources.model_copy(update={"gpu_count": 1})}
    )
    manifest.write_text(spec.model_dump_json(), encoding="utf-8")
    inspected = []

    class Transport:
        async def inspect(self, target):
            inspected.append(target.server_id)
            raise AssertionError("single-host enqueue must not depend on GPU telemetry")

        async def get_job(self, target, run_id):
            return None

        async def submit(self, target, run_id, frozen):
            return JobRecord(run_id=run_id, spec=frozen)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    monkeypatch.setattr(
        "gpuops.control.SubmissionJournal", lambda _path: SubmissionJournal(tmp_path / "journal")
    )
    assert main(["submit", str(manifest), "--run-id", "auto-local", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["server_id"] == "gpu01"
    assert inspected == []


def test_submit_explicit_manifest_host_is_not_narrowed_to_local_server(
    tmp_path, monkeypatch, capsys
) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    manifest = tmp_path / "job.json"
    spec = _spec("gpu02")
    manifest.write_text(spec.model_dump_json(), encoding="utf-8")
    calls = []

    class Transport:
        async def get_job(self, target, run_id):
            calls.append(("get", target.server_id))
            return None

        async def submit(self, target, run_id, frozen):
            calls.append(("submit", target.server_id))
            return JobRecord(run_id=run_id, spec=frozen)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    monkeypatch.setattr(
        "gpuops.control.SubmissionJournal", lambda _path: SubmissionJournal(tmp_path / "journal")
    )
    assert main(["submit", str(manifest), "--run-id", "peer-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_id"] == "gpu02"
    assert calls == [("get", "gpu02"), ("submit", "gpu02")]


def test_reconcile_keeps_frozen_peer_target_accessible(tmp_path, monkeypatch, capsys) -> None:
    install_context(monkeypatch, tmp_path / "home", "gpu01")
    journal = SubmissionJournal(tmp_path / "journal")
    spec = _spec("gpu02")
    journal.create_pending(
        run_id="frozen-peer",
        target=RemoteTarget("gpu02", "lab-gpu02"),
        request_spec=spec,
        frozen_spec=spec,
        now=utc_now(),
    )
    calls = []

    class Transport:
        async def get_job(self, target, run_id):
            calls.append(("get", target.server_id))
            return None

        async def submit(self, target, run_id, frozen):
            calls.append(("submit", target.server_id))
            return JobRecord(run_id=run_id, spec=frozen)

    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: journal)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["reconcile", "frozen-peer", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_id"] == "gpu02"
    assert calls == [("get", "gpu02"), ("submit", "gpu02")]


def _spec(host: str) -> JobSpec:
    return JobSpec(
        id="local-access",
        project="tests",
        command=["true"],
        working_directory="/work",
        resources={"host": host},
    )
