from __future__ import annotations

import asyncio
import tarfile
from pathlib import Path
from types import SimpleNamespace

from gpuops import __version__
from gpuops.bundled import agent_files, project_files
from gpuops.deploy import REMOTE_DEPLOY_SCRIPT, REMOTE_PREPARE_SCRIPT, RemoteDeploymentManager
from gpuops.deployment.artifacts import build_source_archive
from gpuops.ssh import SSHResult


def test_bundled_agent_files_resolve_the_skill_references() -> None:
    bundled = agent_files()
    assert {
        "SKILL.md",
        "docs/AGENT_GUIDE.md",
        "docs/CLI_CONTRACT.md",
        "docs/PROGRESS.md",
        "examples/job.example.json",
        "examples/progress.example.py",
    } <= bundled.keys()
    assert all(content.strip() for content in bundled.values())
    assert 'version = "' + __version__ + '"' in project_files()["pyproject.toml"]


def test_source_archive_restores_build_sources_without_installed_caches(tmp_path: Path) -> None:
    artifact = build_source_archive(tmp_path)
    assert artifact.filename == f"gpuops-{__version__}.tar.gz"
    assert artifact.size_bytes == artifact.path.stat().st_size
    with tarfile.open(artifact.path) as archive:
        names = {item.name.removeprefix(f"gpuops-{__version__}/") for item in archive}
        assert {
            "pyproject.toml", "README.md", "LICENSE", "src/gpuops/worker/runtime.py",
            "src/gpuops/deployment/_bootstrap/manage.py", "src/gpuops/bundled.py",
            "src/gpuops/web/static/app.js", "skills/gpu-infra/SKILL.md",
            "docs/CLI_CONTRACT.md", "examples/progress.example.py",
        } <= names
        assert not any("__pycache__" in name or "/_bundled/" in name for name in names)
        assert all(item.isfile() for item in archive.getmembers())


def test_deploy_default_uploads_current_client_without_local_build_tool(tmp_path: Path) -> None:
    commands = []
    ssh_calls = []
    release_id = None

    def runner(command, *, cwd):
        commands.append(command)
        assert command[0] == "scp"
        with tarfile.open(command[-2]) as archive:
            assert any(item.name.endswith("src/gpuops/worker/runtime.py") for item in archive)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    class SSH:
        async def run_command(self, host, command, *, stdin, timeout_seconds):
            nonlocal release_id
            script = stdin.decode("utf-8")
            ssh_calls.append(script)
            if script == REMOTE_PREPARE_SCRIPT:
                release_id = command[3]
                assert command[-1] == f"gpuops-{__version__}.tar.gz"
                payload = '{"ok":true,"available_upload_bytes":10485760}'
            else:
                assert script == REMOTE_DEPLOY_SCRIPT
                payload = (
                    '{"ok":true,"action":"deploy","changed":true,"status":'
                    '{"installed":true,"current_release":"' + release_id + '"}}'
                )
            return SSHResult(host, 0, payload, "", 0.1)

    result = asyncio.run(RemoteDeploymentManager(ssh=SSH(), command_runner=runner).deploy("gpu"))
    assert result.status.current_release == release_id
    assert len(commands) == 1
    assert ssh_calls == [REMOTE_PREPARE_SCRIPT, REMOTE_DEPLOY_SCRIPT]
