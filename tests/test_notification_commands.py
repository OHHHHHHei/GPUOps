from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from conftest import TEST_TARGETS

from gpuops.cli import main
from gpuops.notifications import commands, config_path
from gpuops.ssh import SSHResult

_SMTP = {
    "host": "smtp.example.org",
    "sender": "sender@example.org",
    "recipient": "receiver@example.org",
    "username": "test-smtp-user",
    "password": "test-smtp-password",
}


def test_local_notification_lifecycle_needs_no_ssh_or_worker(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    identity = home / ".local/share/gpuops/host.json"
    identity.parent.mkdir(parents=True)
    identity.write_text('{"server_id":"gpu01"}', encoding="utf-8")
    state = tmp_path / "worker-state"
    monkeypatch.setattr(commands, "DEFAULT_STATE_DIRECTORY", state)

    async def no_ssh(*args, **kwargs):
        raise AssertionError("this installation does not need SSH")

    monkeypatch.setattr(commands.HostClient, "run_command", no_ssh)
    sends = []

    def fake_test(config, server_id):
        sends.append((config.smtp.host, server_id))

    monkeypatch.setattr(commands, "send_test", fake_test)

    def run(*argv, exit_code=0):
        assert main(["notify", *argv, "--json"]) == exit_code
        captured = capsys.readouterr()
        for secret in (_SMTP["password"], _SMTP["username"]):
            assert secret not in captured.out + captured.err
        payload = json.loads(captured.out)
        assert [row["server_id"] for row in payload["servers"]] == ["gpu01"]
        return payload["servers"][0]

    status = run("status")["result"]
    assert not status["configured"] and not status["enabled"]
    assert status["cursor"] is None
    assert run("test", exit_code=1)["error_code"] == "notification_not_configured"
    assert not config_path().exists()

    smtp_file = tmp_path / "smtp.json"
    smtp_file.write_text(json.dumps(_SMTP), encoding="utf-8")
    configured = run("configure", str(smtp_file))["result"]
    assert configured["configured"] and configured["enabled"]
    assert run("status")["result"]["enabled"]
    assert sends == []
    assert run("test")["result"]["accepted"] is True
    assert sends == [("smtp.example.org", "gpu01")]

    disabled = run("disable")["result"]
    assert disabled["configured"] and not disabled["enabled"]
    assert not state.exists()


def test_configure_stdin_reports_each_servers_result(monkeypatch, capsys):
    body = json.dumps(_SMTP).encode()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(body)))
    calls = []

    class FakeHostClient:
        local = None

        async def run_command(self, host, command, *, stdin, timeout_seconds):
            calls.append((host, command, stdin))
            if host == TEST_TARGETS[0].ssh_host:
                return SSHResult(host, 0, '{"ok":true,"result":{"enabled":true}}', "", 0)
            return SSHResult(
                host,
                1,
                "",
                "Error while finding module specification for 'gpuops.notifications.commands' "
                "(ModuleNotFoundError: No module named 'gpuops.notifications')",
                0,
            )

    monkeypatch.setattr(commands, "HostClient", FakeHostClient)
    assert (
        main(["notify", "configure", "-", "--server", "gpu01", "--server", "gpu02", "--json"]) == 1
    )
    captured = capsys.readouterr()
    assert _SMTP["password"] not in captured.out + captured.err
    result = json.loads(captured.out)
    assert result["outcome"] == "partial" and not result["ok"]
    assert result["servers"][0]["result"]["enabled"]
    assert result["servers"][1]["error_code"] == "notification_unsupported"
    assert "gpuops deploy install --server gpu02" in result["servers"][1]["error"]
    for (host, command, stdin), target in zip(calls, TEST_TARGETS, strict=True):
        assert host == target.ssh_host
        assert command == [
            "~/.local/share/gpuops/current/venv/bin/python",
            "-I",
            "-m",
            "gpuops.notifications.commands",
            "configure",
            "--server-id",
            target.id,
        ]
        assert stdin == body


def test_invalid_notification_settings_do_not_expose_input_or_contact_server(
    tmp_path, monkeypatch, capsys
):
    class FakeHostClient:
        local = None

        async def run_command(self, *args, **kwargs):
            raise AssertionError("invalid settings must not be sent")

    monkeypatch.setattr(commands, "HostClient", FakeHostClient)
    invalid = tmp_path / "smtp.json"
    invalid.write_text(json.dumps({**_SMTP, "port": _SMTP["password"]}), encoding="utf-8")
    assert main(["notify", "configure", str(invalid), "--server", "gpu01", "--json"]) == 1
    captured = capsys.readouterr()
    assert _SMTP["password"] not in captured.out + captured.err
    payload = json.loads(captured.out)
    assert payload["error_code"] == "invalid_notification_config"
    assert "port" in payload["error"]


def test_installed_notification_helper_reads_only_smtp_json_from_stdin(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gpuops.notifications.commands",
            "configure",
            "--server-id",
            "gpu01",
        ],
        input=json.dumps(_SMTP),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    assert json.loads(result.stdout)["result"]["enabled"]
    for secret in (_SMTP["password"], _SMTP["username"]):
        assert secret not in result.stdout + result.stderr
    saved = json.loads((tmp_path / ".config/gpuops/notifications.json").read_text(encoding="utf-8"))
    assert saved["smtp"]["password"] == _SMTP["password"]
