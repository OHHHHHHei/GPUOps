from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gpuops import agent_skill
from gpuops.bundled import agent_files
from gpuops.cli import main
from gpuops.ssh import SSHResult


def test_optional_local_skill_install_uses_codex_home(tmp_path, monkeypatch, capsys) -> None:
    codex_home = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    unrelated = codex_home / "skills/unrelated/SKILL.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep this skill")
    assert main(["skill", "install", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    root = Path(result["installed"])
    assert root == codex_home / "skills/gpu-infra"
    for relative, contents in agent_files().items():
        assert (root / relative).read_text() == contents
    assert unrelated.read_text() == "keep this skill"


def test_remote_skill_installer_writes_bundle_as_literal_text(tmp_path) -> None:
    files = agent_files()
    files["examples/quoted.txt"] = "literal $(whoami), `hostname`, 'quotes' and 中文\n"
    result = subprocess.run(
        [sys.executable, "-c", agent_skill._INSTALL_SKILL],
        input=json.dumps(files),
        text=True,
        capture_output=True,
        env={"HOME": str(tmp_path)},
        check=True,
    )
    root = Path(json.loads(result.stdout)["installed"])
    assert root == tmp_path / ".codex/skills/gpu-infra"
    for relative, contents in files.items():
        assert (root / relative).read_text() == contents


def test_remote_skill_install_routes_to_registered_server(monkeypatch) -> None:
    host = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SSHResult(
                "lab-gpu01",
                0,
                '{"installed":"/home/user/.codex/skills/gpu-infra"}',
                "",
                0.1,
            )
        )
    )
    monkeypatch.setattr(agent_skill, "HostClient", lambda: host)
    result = asyncio.run(agent_skill.install_skill("gpu01"))
    assert result["server_id"] == "gpu01"
    assert host.run_command.await_args.args[0] == "lab-gpu01"
    assert json.loads(host.run_command.await_args.kwargs["stdin"]) == agent_files()


def test_skill_failure_does_not_remove_server_registration(monkeypatch, server_registry, capsys):
    before = server_registry.read_text()
    host = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SSHResult(
                "lab-gpu01",
                1,
                "",
                "permission denied",
                0.1,
            )
        )
    )
    monkeypatch.setattr(agent_skill, "HostClient", lambda: host)
    assert main(["skill", "install", "--server", "gpu01", "--json"]) == 1
    assert "permission denied" in json.loads(capsys.readouterr().out)["error"]
    assert server_registry.read_text() == before


def test_version_is_local_without_servers(tmp_path, monkeypatch, server_registry, capsys):
    monkeypatch.chdir(tmp_path)
    server_registry.unlink()
    assert main(["version", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["cli_api_version"] == "gpuops.cli/v1"
    assert value["job_api_version"] == "gpuops/v1"
    assert value["worker_api_version"] == "gpuops.worker.rpc/v1"
    assert value["version"]
