from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import zipfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuops.core.queue import QueueStore
from gpuops.deploy import (
    REMOTE_ABORT_SCRIPT,
    REMOTE_CACHE_BUDGET_BYTES,
    REMOTE_DEPLOY_SCRIPT,
    REMOTE_INSTALL_BUDGET_BYTES,
    REMOTE_PREPARE_SCRIPT,
    REMOTE_ROLLBACK_SCRIPT,
    REMOTE_STATE_BUDGET_BYTES,
    REMOTE_STATUS_SCRIPT,
    REMOTE_TOTAL_BUDGET_BYTES,
    REMOTE_UNINSTALL_SCRIPT,
    DeployError,
    RemoteDeploymentManager,
    build_wheel,
    copy_wheel,
)
from gpuops.models import JobSpec
from gpuops.ssh import SSHResult


def command_result(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeCommandRunner:
    def __init__(self, wheel_content: bytes = b"wheel-content"):
        self.calls = []
        self.wheel_content = wheel_content
        self.fail_build = False
        self.fail_scp = False

    def __call__(self, command, *, cwd):
        self.calls.append((list(command), cwd))
        if command[0:2] == ["uv", "build"]:
            if self.fail_build:
                return command_result(1, stderr="build broke")
            output = Path(command[command.index("--out-dir") + 1])
            (output / "gpuops-0.1.0-py3-none-any.whl").write_bytes(self.wheel_content)
            return command_result()
        if command[0] == "scp":
            return command_result(1, stderr="copy broke") if self.fail_scp else command_result()
        raise AssertionError(command)


class FakeSSH:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def run_python(self, host, script, *, timeout_seconds):
        self.calls.append((host, script, timeout_seconds, None))
        return self._response(host, script)

    async def run_command(self, host, command, *, stdin, timeout_seconds):
        script = stdin.decode("utf-8")
        self.calls.append((host, script, timeout_seconds, list(command)))
        return self._response(host, script)

    def _response(self, host, script):
        response = self.responses[script]
        if isinstance(response, Exception):
            raise response
        payload, returncode = response if isinstance(response, tuple) else (response, 0)
        return SSHResult(
            host=host,
            returncode=returncode,
            stdout=json.dumps(payload) + "\n",
            stderr="remote stderr" if returncode else "",
            duration_seconds=0.1,
        )


def status_payload(release=None):
    releases = (
        [{"release_id": release, "installed_at": "2026-09-19T00:00:00+00:00", "size_bytes": 42}]
        if release
        else []
    )
    return {
        "installed": release is not None,
        "current_release": release,
        "previous_release": None,
        "releases": releases,
        "share_bytes": 42 if release else 0,
        "budget_bytes": 128 * 1024 * 1024,
        "cache_bytes": 0,
        "state_bytes": 0,
        "total_bytes": 42 if release else 0,
        "total_budget_bytes": 1024 * 1024 * 1024,
        "worker_ready": release is not None,
        "worker_release": release,
        "executable": "/home/researcher/.local/bin/gpuops" if release else None,
    }


def action_payload(action, release=None, changed=True):
    return {
        "ok": True,
        "action": action,
        "changed": changed,
        "message": action,
        "status": status_payload(release),
    }


def run_isolated_bootstrap(script: str, home: Path, *arguments: str):
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
    }
    return subprocess.run(
        [sys.executable, "-I", "-S", "-u", "-", *arguments],
        input=script,
        text=True,
        capture_output=True,
        cwd=home,
        env=environment,
        check=False,
    )


def rollback_namespace(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(home))
    namespace = {"__name__": "bootstrap_test"}
    exec(compile(REMOTE_ROLLBACK_SCRIPT, "<remote-rollback>", "exec"), namespace)
    active_id = "a" * 64
    target_id = "b" * 64
    for release_id, installed_at in (
        (active_id, "2026-09-21T00:00:00+00:00"),
        (target_id, "2026-09-20T00:00:00+00:00"),
    ):
        release = home / ".local/share/gpuops/releases" / release_id
        (release / "venv/bin").mkdir(parents=True)
        (release / "venv/bin/gpuops").touch()
        (release / "release.json").write_text(
            json.dumps({"installed_at": installed_at}), encoding="utf-8"
        )
    current = home / ".local/share/gpuops/current"
    current.symlink_to("releases/" + active_id)
    namespace["deployment_server_id"] = lambda: "gpu01"
    namespace["ensure_executable_link"] = lambda: None
    namespace["stop_worker"] = lambda: None
    namespace["start_worker"] = lambda _release_id, _server_id: None
    namespace["prune_releases"] = lambda _keep: None
    namespace["ensure_owned_budget"] = lambda: None
    namespace["live_release_ids"] = lambda: set()
    return namespace, active_id, target_id


def install_python_wrapper(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_build_wheel_uses_injected_runner_and_assigns_release_id(tmp_path: Path) -> None:
    runner = FakeCommandRunner(b"known wheel")
    artifact = build_wheel(tmp_path, tmp_path / "dist", command_runner=runner)
    assert artifact.filename == "gpuops-0.1.0-py3-none-any.whl"
    assert len(artifact.release_id) == 32
    assert all(character in "0123456789abcdef" for character in artifact.release_id)
    assert runner.calls[0][0][0:2] == ["uv", "build"]
    assert runner.calls[0][1] == tmp_path.resolve()


def test_remote_budget_partitions_have_a_strict_one_gib_ceiling() -> None:
    assert REMOTE_INSTALL_BUDGET_BYTES == 2 * REMOTE_CACHE_BUDGET_BYTES
    assert REMOTE_STATE_BUDGET_BYTES + REMOTE_INSTALL_BUDGET_BYTES == 1024**3
    assert REMOTE_TOTAL_BUDGET_BYTES == 1024**3


def test_build_failure_is_reported_without_guessing_an_artifact(tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    runner.fail_build = True
    with pytest.raises(DeployError, match="build broke"):
        build_wheel(tmp_path, tmp_path / "dist", command_runner=runner)


def test_copy_uses_fixed_cache_and_release_id(tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    artifact = build_wheel(tmp_path, tmp_path / "dist", command_runner=runner)
    deployment_id = "d" * 32
    copy_wheel(
        "lab-gpu01",
        artifact,
        deployment_id=deployment_id,
        command_runner=runner,
    )
    command = runner.calls[-1][0]
    assert command[:2] == ["scp", "-q"]
    assert "BatchMode=yes" in command
    assert "ConnectTimeout=10" in command
    assert "--" in command
    assert command[-1] == (
        f"lab-gpu01:~/.cache/gpuops/uploads/{artifact.release_id}/{deployment_id}/{artifact.filename}"
    )


@pytest.mark.parametrize("host", ["-oBad=yes", "host name", "host:2222", "host;touch_bad", ""])
def test_invalid_host_is_rejected_before_any_command(host, tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    manager = RemoteDeploymentManager(ssh=FakeSSH({}), command_runner=runner)
    with pytest.raises(ValueError):
        asyncio.run(manager.deploy(host, tmp_path))
    assert runner.calls == []


@pytest.mark.parametrize("server_id", ["", "auto", "-node", "node name", "a" * 97, "node/other"])
def test_invalid_server_id_is_rejected_before_any_command(
    server_id: str, tmp_path: Path
) -> None:
    runner = FakeCommandRunner()
    manager = RemoteDeploymentManager(ssh=FakeSSH({}), command_runner=runner)
    with pytest.raises(ValueError, match="server id"):
        asyncio.run(manager.deploy("ssh-gpu", tmp_path, server_id=server_id))
    assert runner.calls == []


def test_deploy_builds_prepares_uploads_and_runs_static_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"deploy wheel"
    release_id = "a" * 32
    monkeypatch.setattr(
        "gpuops.deployment.artifacts.uuid.uuid4",
        lambda: SimpleNamespace(hex=release_id),
    )
    runner = FakeCommandRunner(wheel)
    ssh = FakeSSH(
        {
            REMOTE_PREPARE_SCRIPT: (
                {"ok": True, "action": "prepare", "available_upload_bytes": 1024},
                0,
            ),
            REMOTE_DEPLOY_SCRIPT: action_payload("deploy", release_id),
        }
    )
    manager = RemoteDeploymentManager(ssh=ssh, command_runner=runner)
    result = asyncio.run(manager.deploy("lab-gpu01", tmp_path, server_id="gpu-node-03"))
    assert result.changed
    assert result.status.current_release == release_id
    assert [call[1] for call in ssh.calls] == [REMOTE_PREPARE_SCRIPT, REMOTE_DEPLOY_SCRIPT]
    assert ssh.calls[0][3][0:3] == ["python3", "-u", "-"]
    assert ssh.calls[1][3][0:3] == ["python3", "-u", "-"]
    assert ssh.calls[1][3][-2:] == ["gpu-node-03", "lab-gpu01"]
    assert runner.calls[-1][0][0] == "scp"


def test_remote_install_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    ssh = FakeSSH(
        {
            REMOTE_PREPARE_SCRIPT: (
                {"ok": True, "action": "prepare", "available_upload_bytes": 1024},
                0,
            ),
            REMOTE_DEPLOY_SCRIPT: (
                {"ok": False, "action": "deploy", "error": "pip install failed"},
                1,
            ),
        }
    )
    manager = RemoteDeploymentManager(ssh=ssh, command_runner=runner)
    with pytest.raises(DeployError, match="pip install failed"):
        asyncio.run(manager.deploy("lab-gpu01", tmp_path))


def test_deploy_rejects_upload_over_budget_before_scp(tmp_path: Path) -> None:
    runner = FakeCommandRunner(b"too large")
    ssh = FakeSSH(
        {
            REMOTE_PREPARE_SCRIPT: {
                "ok": True,
                "action": "prepare",
                "available_upload_bytes": 1,
            }
        }
    )

    with pytest.raises(DeployError, match="cannot fit"):
        asyncio.run(
            RemoteDeploymentManager(ssh=ssh, command_runner=runner).deploy("lab-gpu01", tmp_path)
        )

    assert all(call[0][0] != "scp" for call in runner.calls)


def test_failed_scp_aborts_only_its_upload(tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    runner.fail_scp = True
    ssh = FakeSSH(
        {
            REMOTE_PREPARE_SCRIPT: {
                "ok": True,
                "action": "prepare",
                "available_upload_bytes": 1024,
            },
            REMOTE_ABORT_SCRIPT: {"ok": True, "action": "abort"},
        }
    )

    with pytest.raises(DeployError, match="copy broke"):
        asyncio.run(
            RemoteDeploymentManager(ssh=ssh, command_runner=runner).deploy("lab-gpu01", tmp_path)
        )

    assert [call[1] for call in ssh.calls] == [REMOTE_PREPARE_SCRIPT, REMOTE_ABORT_SCRIPT]


def test_status_rollback_and_uninstall_use_fixed_scripts() -> None:
    current = "a" * 64
    previous = "b" * 64
    status = status_payload(current)
    status["server_id"] = "gpu-node-03"
    status["previous_release"] = previous
    status["releases"].append(
        {"release_id": previous, "installed_at": "2026-09-18T00:00:00+00:00", "size_bytes": 41}
    )
    ssh = FakeSSH(
        {
            REMOTE_STATUS_SCRIPT: {"ok": True, "action": "status", "status": status},
            REMOTE_ROLLBACK_SCRIPT: action_payload("rollback", previous),
            REMOTE_UNINSTALL_SCRIPT: action_payload("uninstall", None),
        }
    )
    manager = RemoteDeploymentManager(ssh=ssh, command_runner=FakeCommandRunner())
    read = asyncio.run(manager.status("lab-gpu02"))
    rolled_back = asyncio.run(manager.rollback("lab-gpu02"))
    removed = asyncio.run(manager.uninstall("lab-gpu02"))
    assert read.current_release == current
    assert read.previous_release == previous
    assert read.server_id == "gpu-node-03"
    assert rolled_back.status.current_release == previous
    assert removed.status.installed is False
    assert [call[1] for call in ssh.calls] == [
        REMOTE_STATUS_SCRIPT,
        REMOTE_ROLLBACK_SCRIPT,
        REMOTE_UNINSTALL_SCRIPT,
    ]


def test_remote_programs_compile_and_status_runs_without_gpuops_imports(tmp_path: Path) -> None:
    for script in (
        REMOTE_PREPARE_SCRIPT,
        REMOTE_ABORT_SCRIPT,
        REMOTE_DEPLOY_SCRIPT,
        REMOTE_STATUS_SCRIPT,
        REMOTE_ROLLBACK_SCRIPT,
        REMOTE_UNINSTALL_SCRIPT,
    ):
        compile(script, "<remote-deploy>", "exec")
        assert "shell=True" not in script

    completed = run_isolated_bootstrap(REMOTE_STATUS_SCRIPT, tmp_path)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["action"] == "status"
    assert payload["status"]["installed"] is False


def test_rollback_rejects_incompatible_queue_before_worker_or_symlink_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, active_id, target_id = rollback_namespace(tmp_path, monkeypatch)
    jobs = tmp_path / ".local/state/gpuops/queue/jobs"
    jobs.mkdir(parents=True)
    (jobs / "new-schema.json").write_text("{}\n", encoding="utf-8")
    target_python = (
        tmp_path / ".local/share/gpuops/releases" / target_id / "venv/bin/python"
    )
    target_python.write_text(
        "#!/bin/sh\necho 'gpu_uuids is not permitted' >&2\nexit 2\n",
        encoding="utf-8",
    )
    target_python.chmod(0o700)
    namespace["ensure_executable_link"] = lambda: (_ for _ in ()).throw(
        AssertionError("executable link changed before compatibility check")
    )
    namespace["stop_worker"] = lambda: (_ for _ in ()).throw(
        AssertionError("worker stopped before compatibility check")
    )
    namespace["atomic_symlink"] = lambda *_args: (_ for _ in ()).throw(
        AssertionError("release symlink changed before compatibility check")
    )

    with pytest.raises(RuntimeError, match="cannot read persisted GPUOps queue records"):
        namespace["rollback"]()

    assert namespace["current_release"]() == active_id


def test_rollback_allows_compatible_persisted_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _active_id, target_id = rollback_namespace(tmp_path, monkeypatch)
    store = QueueStore(tmp_path / ".local/state/gpuops/queue")
    record = store.enqueue(
        JobSpec(
            id="rollback-compatible",
            project="tests",
            command=["python", "train.py"],
            working_directory="/work",
            resources={"host": "gpu01", "gpu_count": 1},
        )
    )
    store.lease(
        record.run_id,
        server_id="gpu01",
        gpu_indices=[3],
        gpu_uuids={3: "GPU-compatible"},
        worker_id="worker",
        ttl=timedelta(minutes=1),
        expected_revision=record.revision,
    )
    install_python_wrapper(
        tmp_path / ".local/share/gpuops/releases" / target_id / "venv/bin/python"
    )

    changed, message = namespace["rollback"]()

    assert changed is True
    assert message == "rolled back to " + target_id
    assert namespace["current_release"]() == target_id


def test_rollback_allows_empty_queue_without_target_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _active_id, target_id = rollback_namespace(tmp_path, monkeypatch)

    changed, _message = namespace["rollback"]()

    assert changed is True
    assert namespace["current_release"]() == target_id


def test_rollback_rechecks_queue_after_stopping_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, active_id, _target_id = rollback_namespace(tmp_path, monkeypatch)
    checks = 0
    lifecycle = []

    def check(_release_id):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("queue changed during rollback")

    namespace["ensure_rollback_queue_compatible"] = check
    namespace["stop_worker"] = lambda: lifecycle.append("stop")
    namespace["start_worker"] = lambda release_id, _server_id: lifecycle.append(
        "start:" + release_id
    )

    with pytest.raises(RuntimeError, match="queue changed during rollback"):
        namespace["rollback"]()

    assert checks == 2
    assert namespace["current_release"]() == active_id
    assert lifecycle == ["stop", "stop", "start:" + active_id]


def test_remote_worker_lifecycle_allows_one_bounded_slow_iteration() -> None:
    namespace = {"__name__": "bootstrap_test"}
    exec(compile(REMOTE_STATUS_SCRIPT, "<remote-status>", "exec"), namespace)

    from gpuops.worker.app import WORKER_HEARTBEAT_MAX_AGE_SECONDS

    # The standalone bootstrap cannot import the installed Worker during repair.
    assert namespace["HEARTBEAT_MAX_AGE_SECONDS"] == WORKER_HEARTBEAT_MAX_AGE_SECONDS
    assert namespace["WORKER_LIFECYCLE_TIMEOUT_SECONDS"] >= WORKER_HEARTBEAT_MAX_AGE_SECONDS


def test_remote_install_creates_runnable_venv_and_ships_bootstrap_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).parents[1]
    artifact = build_wheel(project_root, tmp_path / "dist")
    with zipfile.ZipFile(artifact.path) as wheel:
        members = set(wheel.namelist())
    assert {
        "gpuops/deployment/_bootstrap/prepare.py",
        "gpuops/deployment/_bootstrap/abort.py",
        "gpuops/deployment/_bootstrap/manage.py",
    } <= members

    deployment_id = "d" * 32
    upload_directory = tmp_path / ".cache/gpuops/uploads" / artifact.release_id / deployment_id
    upload_directory.mkdir(parents=True)
    (upload_directory / artifact.filename).write_bytes(artifact.path.read_bytes())
    monkeypatch.setenv("HOME", str(tmp_path))
    namespace = {"__name__": "bootstrap_test"}
    original_argv = sys.argv
    original_umask = os.umask(0o077)
    try:
        monkeypatch.setattr("socket.gethostname", lambda: "login-node")
        sys.argv = ["-", artifact.release_id, deployment_id, "gpu-node-03", "ssh-gpu"]
        exec(compile(REMOTE_DEPLOY_SCRIPT, "<remote-deploy>", "exec"), namespace)
        namespace["worker_snapshot"] = lambda _server_id=None: {
            "process_running": False,
            "ready": False,
            "reason": "worker heartbeat is stale",
            "pid": None,
            "server_id": "gpu02",
        }
        initial_starts = []
        namespace["start_worker"] = lambda release_id, server_id: initial_starts.append(
            (release_id, server_id)
        )
        deployed = namespace["run_action"]()
    finally:
        sys.argv = original_argv
        os.umask(original_umask)

    assert deployed["ok"] is True
    assert deployed["status"]["current_release"] == artifact.release_id
    assert initial_starts == [(artifact.release_id, "gpu-node-03")]
    identity = tmp_path / ".local/share/gpuops/host.json"
    assert json.loads(identity.read_text(encoding="utf-8")) == {
        "server_id": "gpu-node-03", "ssh_host": "ssh-gpu"
    }
    assert deployed["status"]["server_id"] == "gpu-node-03"
    release = tmp_path / ".local/share/gpuops/releases" / artifact.release_id
    assert "include-system-site-packages = false" in (release / "venv/pyvenv.cfg").read_text()
    executable = release / "venv/bin/gpuops"
    assert executable.exists()
    assert not (release / ".installing").exists()
    launcher = executable.read_text(encoding="utf-8")
    assert str(release / "venv/bin/python") in launcher
    invoked = subprocess.run(
        [str(executable), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invoked.returncode == 0, invoked.stderr
    assert "Local GPU compute control plane" in invoked.stdout

    clean_home = tmp_path / "clean-home"
    clean_home.mkdir()
    clean_cwd = tmp_path / "clean-cwd"
    clean_cwd.mkdir()
    installed_status = subprocess.run(
        [
            str(release / "venv/bin/python"),
            "-I",
            "-c",
            "from gpuops.deploy import REMOTE_STATUS_SCRIPT; exec(REMOTE_STATUS_SCRIPT)",
        ],
        text=True,
        capture_output=True,
        cwd=clean_cwd,
        env={
            "HOME": str(clean_home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
        },
        check=False,
    )
    assert installed_status.returncode == 0, installed_status.stderr
    installed_lines = installed_status.stdout.splitlines()
    assert len(installed_lines) == 1
    assert json.loads(installed_lines[0])["action"] == "status"

    upload_directory.mkdir(parents=True)
    (upload_directory / artifact.filename).write_bytes(artifact.path.read_bytes())
    starts = []
    namespace["worker_snapshot"] = lambda _server_id=None: {
        "process_running": False,
        "ready": False,
        "reason": "worker process identity is unavailable",
        "pid": None,
        "server_id": None,
    }
    namespace["start_worker"] = lambda release_id, server_id: starts.append(
        (release_id, server_id)
    )
    redeployed = namespace["run_action"]()

    assert redeployed["changed"] is False
    assert starts == [(artifact.release_id, "gpu-node-03")]
    assert json.loads(identity.read_text(encoding="utf-8")) == {
        "server_id": "gpu-node-03", "ssh_host": "ssh-gpu"
    }
    assert namespace["deployment_server_id"]() == "gpu-node-03"

    previous_id = "b" * 64
    previous = tmp_path / ".local/share/gpuops/releases" / previous_id
    (previous / "venv/bin").mkdir(parents=True)
    (previous / "venv/bin/gpuops").touch()
    (previous / "release.json").write_text(
        json.dumps({"installed_at": "2026-09-20T00:00:00+00:00"}),
        encoding="utf-8",
    )
    starts.clear()
    namespace["ACTION"] = "rollback"
    rolled_back = namespace["run_action"]()

    assert rolled_back["ok"] is True
    assert rolled_back["status"]["current_release"] == previous_id
    assert starts == [(previous_id, "gpu-node-03")]
    assert json.loads(identity.read_text(encoding="utf-8")) == {
        "server_id": "gpu-node-03", "ssh_host": "ssh-gpu"
    }

    namespace["ACTION"] = "uninstall"
    removed = namespace["run_action"]()
    assert removed["ok"] is True
    assert removed["changed"] is True
    assert not identity.exists()


def test_failed_fresh_install_does_not_write_host_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"failed deployment"
    release_id = "f" * 32
    deployment_id = "e" * 32
    upload_directory = tmp_path / ".cache/gpuops/uploads" / release_id / deployment_id
    upload_directory.mkdir(parents=True)
    (upload_directory / "gpuops-0.1.0-py3-none-any.whl").write_bytes(wheel)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("socket.gethostname", lambda: "gpu02")
    namespace = {"__name__": "bootstrap_test"}
    original_argv = sys.argv
    original_umask = os.umask(0o077)
    try:
        sys.argv = ["-", release_id, deployment_id, "arbitrary-node", "ssh-alias"]
        exec(compile(REMOTE_DEPLOY_SCRIPT, "<remote-deploy>", "exec"), namespace)
        namespace["release_complete"] = lambda _release_id: True
        namespace["ensure_executable_link"] = lambda: None
        namespace["start_worker"] = lambda _release_id, _server_id: (_ for _ in ()).throw(
            RuntimeError("worker failed")
        )
        with pytest.raises(RuntimeError, match="worker failed"):
            namespace["run_action"]()
    finally:
        sys.argv = original_argv
        os.umask(original_umask)

    assert not (tmp_path / ".local/share/gpuops/host.json").exists()


def test_unlisted_hostname_is_valid_fallback_and_ignores_worker_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("socket.gethostname", lambda: "login-node")
    namespace = {"__name__": "bootstrap_test"}
    exec(compile(REMOTE_STATUS_SCRIPT, "<remote-status>", "exec"), namespace)
    namespace["worker_snapshot"] = lambda _server_id=None: {"server_id": "gpu01"}

    assert namespace["deployment_server_id"]() == "login-node"


def test_remote_deployment_rejects_auto_as_installation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    namespace = {"__name__": "bootstrap_test"}
    exec(compile(REMOTE_STATUS_SCRIPT, "<remote-status>", "exec"), namespace)

    with pytest.raises(RuntimeError, match="requested server id is invalid"):
        namespace["deployment_server_id"]("auto")


def test_status_reads_legacy_host_identity_and_old_release_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, active_id, _ = rollback_namespace(tmp_path, monkeypatch)
    identity = tmp_path / ".local/share/gpuops/host.json"
    identity.write_text('{"server_id":"gpu01"}\n', encoding="utf-8")

    status = namespace["status"]()

    assert status["installed"] is True
    assert status["current_release"] == active_id
    assert status["server_id"] == "gpu01"


def test_deploy_rejects_existing_server_identity_before_worker_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_id = "d" * 32
    deployment_id = "e" * 32
    upload = (
        tmp_path / ".cache/gpuops/uploads" / release_id / deployment_id
        / "gpuops-0.1.0-py3-none-any.whl"
    )
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"wheel")
    share = tmp_path / ".local/share/gpuops"
    share.mkdir(parents=True)
    identity = share / "host.json"
    identity.write_text('{"server_id":"gpu01"}\n', encoding="utf-8")
    current = share / "current"
    current.symlink_to("releases/" + "a" * 64)
    monkeypatch.setenv("HOME", str(tmp_path))
    namespace = {"__name__": "bootstrap_test"}
    original_argv = sys.argv
    original_umask = os.umask(0o077)
    try:
        sys.argv = ["-", release_id, deployment_id, "other-node", "ssh-gpu"]
        exec(compile(REMOTE_DEPLOY_SCRIPT, "<remote-deploy>", "exec"), namespace)
        namespace["stop_worker"] = lambda: (_ for _ in ()).throw(
            AssertionError("worker stopped despite identity conflict")
        )
        with pytest.raises(RuntimeError, match="conflicts with requested server id"):
            namespace["deploy"]()
    finally:
        sys.argv = original_argv
        os.umask(original_umask)

    assert current.readlink() == Path("releases/" + "a" * 64)
    assert json.loads(identity.read_text(encoding="utf-8")) == {"server_id": "gpu01"}


def test_prepare_reserves_exact_upload_and_abort_removes_only_that_transaction(
    tmp_path: Path,
) -> None:
    release_id = "a" * 32
    deployment_id = "b" * 32
    wheel_name = "gpuops-0.1.0-py3-none-any.whl"
    prepared = run_isolated_bootstrap(
        REMOTE_PREPARE_SCRIPT,
        tmp_path,
        release_id,
        "17",
        deployment_id,
        wheel_name,
    )

    assert prepared.returncode == 0, prepared.stderr
    prepared_lines = prepared.stdout.splitlines()
    assert len(prepared_lines) == 1
    assert json.loads(prepared_lines[0])["action"] == "prepare"
    reserved = tmp_path / ".cache/gpuops/uploads" / release_id / deployment_id / wheel_name
    assert reserved.stat().st_size == 17

    overcommitted = run_isolated_bootstrap(
        REMOTE_PREPARE_SCRIPT,
        tmp_path,
        "c" * 32,
        str(128 * 1024 * 1024),
        "d" * 32,
        wheel_name,
    )
    assert overcommitted.returncode != 0
    assert "cannot fit" in overcommitted.stderr

    aborted = run_isolated_bootstrap(REMOTE_ABORT_SCRIPT, tmp_path, release_id, deployment_id)

    assert aborted.returncode == 0, aborted.stderr
    aborted_lines = aborted.stdout.splitlines()
    assert len(aborted_lines) == 1
    assert json.loads(aborted_lines[0])["action"] == "abort"
    assert not reserved.exists()
