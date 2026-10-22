from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import TEST_TARGETS

from gpuops import servers
from gpuops.cli import main
from gpuops.config import (
    ServerTarget,
    load_targets,
    local_target,
    register_target,
    select_client_targets,
    select_targets,
    unregister_target,
)
from gpuops.models import ACTIVE_JOB_STATUSES
from gpuops.ssh import SSHResult

ROOT = Path(__file__).resolve().parents[1]


def test_registry_starts_empty_and_preserves_existing_labels(server_registry) -> None:
    server_registry.unlink()
    assert load_targets() == ()
    target = ServerTarget("lab-a100", "实验室", "lab-a100")
    register_target(target)
    register_target(ServerTarget(target.id, "renamed", target.ssh_host))
    assert select_targets() == (target,)
    assert select_targets([target.id]) == (target,)
    assert json.loads(server_registry.read_text())["servers"] == [asdict(target)]
    unregister_target(target.id)
    assert load_targets() == ()


@pytest.mark.parametrize(
    "target",
    [ServerTarget("gpu01", "other", "new-route"), ServerTarget("duplicate", "other", "lab-gpu01")],
)
def test_registry_rejects_conflicting_routes_without_overwriting(server_registry, target) -> None:
    before = server_registry.read_text()
    with pytest.raises(ValueError, match="already registered"):
        register_target(target)
    assert server_registry.read_text() == before


@pytest.mark.parametrize("payload", ["not json", "[]", '{"servers":[{"id":"auto"}]}'])
def test_malformed_registry_is_not_silently_replaced(server_registry, payload) -> None:
    server_registry.write_text(payload)
    with pytest.raises(ValueError, match="invalid server registry"):
        register_target(ServerTarget("lab", "lab", "lab"))
    assert server_registry.read_text() == payload


def test_arbitrary_local_identity_does_not_need_controller_registry(
    tmp_path, monkeypatch, server_registry
) -> None:
    server_registry.unlink()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / ".local/share/gpuops/host.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"server_id":"lab-a100","ssh_host":"lab-route"}')
    expected = ServerTarget("lab-a100", "lab-a100", "lab-route")
    assert local_target() == expected
    assert select_client_targets() == (expected,)
    assert select_client_targets(["lab-a100"]) == (expected,)


@pytest.fixture
def connection(monkeypatch):
    success = SSHResult("lab", 0, "", "", 0.01)
    ssh = SimpleNamespace(
        run_python=AsyncMock(return_value=success), run_command=AsyncMock(return_value=success)
    )
    manager = SimpleNamespace(
        status=AsyncMock(
            return_value=SimpleNamespace(
                installed=False,
                worker_ready=False,
                server_id=None,
            )
        ),
        deploy=AsyncMock(),
    )
    transport = SimpleNamespace(
        worker_command=AsyncMock(return_value={"ok": True, "ready": True, "status": "running"}),
        list_summaries=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(servers, "SSHClient", lambda: ssh)
    monkeypatch.setattr(servers, "RemoteDeploymentManager", lambda **kwargs: manager)
    monkeypatch.setattr(servers, "SshWorkerTransport", lambda **kwargs: transport)
    return SimpleNamespace(ssh=ssh, manager=manager, transport=transport)


def test_add_installs_new_alias_and_registers_after_readiness(connection, server_registry) -> None:
    result = asyncio.run(servers.add_server("lab-a100", project_root=ROOT))
    connection.manager.deploy.assert_awaited_once_with("lab-a100", ROOT, server_id="lab-a100")
    assert result["installed"] and result["ready"]
    assert select_targets(["lab-a100"]) == (ServerTarget("lab-a100", "lab-a100", "lab-a100"),)
    connection.ssh.run_command.assert_not_awaited()


def test_add_without_checkout_passes_installed_distribution_to_deployer(connection) -> None:
    assert asyncio.run(servers.add_server("lab-a100"))["ready"]
    connection.manager.deploy.assert_awaited_once_with("lab-a100", None, server_id="lab-a100")
    connection.ssh.run_command.assert_not_awaited()


def test_repeat_add_reuses_ready_worker_and_preserves_legacy_identity(connection) -> None:
    connection.manager.status.return_value = SimpleNamespace(
        installed=True,
        worker_ready=True,
        server_id="gpu01",
    )
    for _ in range(2):
        result = asyncio.run(servers.add_server("lab-gpu01", project_root=ROOT))
        assert result["server"] == asdict(TEST_TARGETS[0])
        assert result["installed"] is False
    connection.manager.deploy.assert_not_awaited()
    assert [call.args[1] for call in connection.transport.worker_command.await_args_list] == [
        "status",
        "status",
    ]
    assert load_targets() == TEST_TARGETS


def test_add_starts_existing_stopped_worker_without_deployment(connection) -> None:
    connection.manager.status.return_value = SimpleNamespace(
        installed=True,
        worker_ready=False,
        server_id="lab-a100",
    )
    asyncio.run(servers.add_server("lab-a100", project_root=ROOT))
    connection.manager.deploy.assert_not_awaited()
    assert [call.args[1] for call in connection.transport.worker_command.await_args_list] == [
        "start",
        "status",
    ]


@pytest.mark.parametrize("identity", ["another-id", "gpu02"])
def test_add_does_not_mutate_installed_server_with_identity_conflict(connection, identity) -> None:
    connection.manager.status.return_value = SimpleNamespace(
        installed=True,
        worker_ready=True,
        server_id=identity,
    )
    with pytest.raises(ValueError, match="identifies as"):
        asyncio.run(servers.add_server("lab-gpu01", project_root=ROOT))
    connection.manager.deploy.assert_not_awaited()
    connection.transport.worker_command.assert_not_awaited()
    connection.ssh.run_command.assert_not_awaited()
    assert load_targets() == TEST_TARGETS


@pytest.mark.parametrize("stage", ["prerequisites", "deployment", "readiness"])
def test_failed_add_keeps_registration_unchanged(connection, stage, monkeypatch) -> None:
    monkeypatch.setattr(servers, "WORKER_START_TIMEOUT_SECONDS", 0)
    if stage == "prerequisites":
        connection.ssh.run_python.return_value = SSHResult("lab", 1, "", "needs Linux", 0.01)
    elif stage == "deployment":
        connection.manager.deploy.side_effect = RuntimeError("deployment failed")
    elif stage == "readiness":
        connection.transport.worker_command.return_value = {"ok": True, "ready": False}
    with pytest.raises(RuntimeError):
        asyncio.run(servers.add_server("lab-a100", project_root=ROOT))
    assert load_targets() == TEST_TARGETS


def test_add_waits_for_slow_worker_start(connection, monkeypatch) -> None:
    connection.manager.status.return_value = SimpleNamespace(
        installed=True,
        worker_ready=False,
        server_id="lab-a100",
    )
    connection.transport.worker_command.side_effect = [
        {"ok": True, "status": "starting", "ready": False},
        {"ok": True, "status": "starting", "ready": False},
        {"ok": True, "status": "running", "ready": True},
    ]
    monkeypatch.setattr(servers.asyncio, "sleep", AsyncMock())
    assert asyncio.run(servers.add_server("lab-a100", project_root=ROOT))["ready"]
    assert connection.transport.worker_command.await_count == 3


def test_remove_refuses_active_jobs_without_stopping(connection) -> None:
    connection.transport.list_summaries.return_value = [object()]
    with pytest.raises(ValueError, match="active jobs"):
        asyncio.run(servers.remove_server("gpu01"))
    connection.transport.worker_command.assert_not_awaited()
    assert load_targets() == TEST_TARGETS


def test_remove_stops_empty_worker_and_preserves_other_servers(connection) -> None:
    connection.transport.worker_command.side_effect = [
        {"status": "running"},
        {"ok": True},
        {"status": "stopped"},
    ]
    result = asyncio.run(servers.remove_server("gpu01"))
    assert result == {"removed": "gpu01", "remote_data_preserved": True}
    assert load_targets() == (TEST_TARGETS[1],)
    assert [call.args[1] for call in connection.transport.worker_command.await_args_list] == [
        "status",
        "stop",
        "status",
    ]
    assert connection.transport.list_summaries.await_count == 2
    assert connection.transport.list_summaries.await_args.kwargs == {
        "statuses": ACTIVE_JOB_STATUSES
    }


def test_remove_restarts_worker_if_new_job_arrives_during_stop(connection) -> None:
    connection.transport.list_summaries.side_effect = [[], [object()]]
    connection.transport.worker_command.side_effect = [
        {"status": "running"},
        {"ok": True},
        {"status": "stopped"},
        {"ready": True},
    ]
    with pytest.raises(ValueError, match="new active jobs"):
        asyncio.run(servers.remove_server("gpu01"))
    assert connection.transport.worker_command.await_args.args[1] == "start"
    assert load_targets() == TEST_TARGETS


def test_server_cli_reports_failure_and_empty_list(connection, server_registry, capsys) -> None:
    server_registry.unlink()
    assert main(["server", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "servers": []}
    assert main(["server", "add", "invalid alias", "--json"]) == 1


@pytest.mark.parametrize("command", ["status", "doctor"])
def test_empty_fleet_has_actionable_guidance(server_registry, capsys, command) -> None:
    server_registry.unlink()
    assert main([command, "--json"]) == 1
    assert "gpuops server add" in capsys.readouterr().out


def test_server_cli_accepts_new_alias(connection, capsys) -> None:
    assert main(["server", "add", "lab-a100", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["server"]["id"] == "lab-a100"
    assert main(["server", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["servers"][-1]["id"] == "lab-a100"
