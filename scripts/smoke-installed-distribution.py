#!/usr/bin/env python3
"""Build/install outside the checkout, then deploy through a local SSH substitute.

Run `uv run python scripts/smoke-installed-distribution.py`. An existing wheel can
be supplied with --wheel. --installer selects pip, uv, or both (the default).
--legacy-pip seeds the deployed Worker venv with pip 22.0.2 before installation.
Every installation and fake remote home is confined to a temporary directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

CLIENT_SMOKE = r'''
import io
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import gpuops
import gpuops.deployment.client as deployment_client
import gpuops.servers as servers
from gpuops.bundled import agent_files
from gpuops.cli import main
from gpuops.config import load_targets
from gpuops.deploy import REMOTE_DEPLOY_SCRIPT, REMOTE_STATUS_SCRIPT
from gpuops.ssh import SSHResult

workspace = Path(sys.argv[1])
legacy_pip = len(sys.argv) > 2 and sys.argv[2] == "legacy-pip"
home = workspace / "remote-home"
home.mkdir()
client_home = workspace / "client-home"
client_home.mkdir()
os.environ["HOME"] = str(client_home)
os.environ["CODEX_HOME"] = str(workspace / "codex")
assert "site-packages" in str(gpuops.__file__), gpuops.__file__
bundled = agent_files()
assert len(bundled) >= 6 and all(bundled.values())
from importlib.resources import files
assert files("gpuops").joinpath("web/static/app.js").is_file()
assert files("gpuops").joinpath("web/templates/index.html").is_file()

def cli(*arguments):
    completed = subprocess.run(
        [str(Path(sys.executable).with_name("gpuops")), *arguments],
        text=True, capture_output=True, cwd=workspace, check=True,
    )
    return json.loads(completed.stdout)

assert cli("version", "--json")["version"] == gpuops.__version__
example = workspace / "example-job.json"
example.write_text(bundled["examples/job.example.json"], encoding="utf-8")
assert cli("validate", str(example), "--json")["valid"]

LEGACY_PIP_SEED = r"""
import subprocess
from pathlib import Path
original_run = subprocess.run
def seed_legacy_pip(command, **kwargs):
    result = original_run(command, **kwargs)
    if result.returncode == 0 and command[1:3] == ["-m", "venv"]:
        seeded = original_run(
            [str(Path(command[-1]) / "bin/python"), "-m", "pip", "install",
             "--disable-pip-version-check", "pip==22.0.2"], **kwargs,
        )
        if seeded.returncode:
            return seeded
    return result
subprocess.run = seed_legacy_pip
"""

class LocalSSH:
    calls = []

    async def run_python(self, host, script, *, timeout_seconds):
        if script == servers._PREREQUISITES:
            # The fake host has no GPU; preflight is outside this install smoke.
            return SSHResult(host, 0, "", "", 0.0)
        return self._execute(host, script, [], timeout_seconds)

    async def run_command(self, host, command, *, stdin, timeout_seconds):
        return self._execute(host, stdin.decode("utf-8"), command[3:], timeout_seconds)

    def _execute(self, host, script, arguments, timeout_seconds):
        assert host == "smoke-local"
        self.calls.append(script)
        if script in {REMOTE_DEPLOY_SCRIPT, REMOTE_STATUS_SCRIPT}:
            # Run the real installer; substitute Worker lifecycle on this CPU host.
            # The optional legacy case only changes the pip seeded in the new venv.
            script = (
                "namespace = {'__name__': 'smoke_remote'}\n"
                "exec(" + repr(script) + ", namespace)\n"
                "namespace['stop_worker'] = lambda: None\n"
                "namespace['start_worker'] = lambda *_args: None\n"
                "namespace['worker_snapshot'] = lambda *_args: "
                "{'ready': True, 'pid': None}\n"
                + (LEGACY_PIP_SEED if legacy_pip else "")
                + "namespace['main']()\n"
            )
        result = subprocess.run(
            [sys.executable, "-I", "-u", "-", *arguments],
            input=script, text=True, capture_output=True, cwd=home,
            env={**os.environ, "HOME": str(home)}, timeout=timeout_seconds,
        )
        return SSHResult(host, result.returncode, result.stdout, result.stderr, 0.0)

uploads = []
def upload(command, *, cwd):
    assert command[0] == "scp", command  # The installed client must not run uv/build.
    source = Path(command[-2])
    assert source.name == "gpuops-" + gpuops.__version__ + ".tar.gz"
    destination = command[-1].split(":", 1)[1]
    target = home / destination.removeprefix("~/")
    shutil.copyfile(source, target)
    uploads.append(target)
    return SimpleNamespace(returncode=0, stdout="", stderr="")

ssh = LocalSSH()
class ReadyTransport:
    def __init__(self, **kwargs):
        pass

    async def worker_command(self, *_args):
        return {"ready": True}

servers.SSHClient = lambda: ssh
servers.SshWorkerTransport = ReadyTransport
deployment_client._default_command_runner = upload
with redirect_stdout(io.StringIO()) as output:
    code = main(["server", "add", "smoke-local"])
assert code == 0, output.getvalue()
result = json.loads(output.getvalue())
assert result["installed"] and result["ready"]
assert result["server"]["id"] == "smoke-local"
assert [target.ssh_host for target in load_targets()] == ["smoke-local"]
assert not (client_home / ".codex/skills/gpu-infra").exists()
assert not (workspace / "codex/skills/gpu-infra").exists()
assert len(uploads) == 1 and len(ssh.calls) == 3
release = (home / ".local/share/gpuops/current").resolve()
assert "include-system-site-packages = false" in (release / "venv/pyvenv.cfg").read_text()
remote_python = release / "venv/bin/python"
verified = subprocess.run(
    [str(remote_python), "-I", "-c",
     "import json,gpuops,pip; from gpuops.bundled import agent_files; "
     "from gpuops.deployment.artifacts import build_source_archive; "
     "from pathlib import Path; "
     "assert all(agent_files().values()); "
     "assert build_source_archive(Path.cwd()/'redeploy').path.is_file(); "
     "print(json.dumps({'version':gpuops.__version__,'module':gpuops.__file__,"
     "'pip_version':pip.__version__}))"],
    text=True, capture_output=True, cwd=workspace, check=True,
)
remote = json.loads(verified.stdout)
assert remote["version"] == gpuops.__version__
assert str(release) in remote["module"], remote
if legacy_pip:
    assert remote["pip_version"] == "22.0.2", remote
    print("source deployment with pip 22.0.2 passed")
skill = Path(cli("skill", "install")["installed"])
assert skill == workspace / "codex/skills/gpu-infra"
assert all((skill / name).is_file() for name in bundled)
for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", (skill / "SKILL.md").read_text()):
    target = match.group(1)
    if "://" not in target and not target.startswith("#"):
        assert (skill / target.split("#", 1)[0]).exists(), target
subprocess.run([str(release / "venv/bin/gpuops"), "--help"],
               cwd=workspace, check=True, capture_output=True)
print("installed client and deployed Worker OK (" + gpuops.__version__ + ")")
'''


def run(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {command!r}\n{result.stdout}\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--installer", choices=("pip", "uv", "both"), default="both")
    parser.add_argument("--legacy-pip", action="store_true")
    arguments = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    with tempfile.TemporaryDirectory(prefix="gpuops-installed-smoke-") as temporary:
        workspace = Path(temporary)
        wheel = arguments.wheel.resolve() if arguments.wheel else None
        if wheel is None:
            if not uv:
                parser.error("building the checkout requires uv; alternatively pass --wheel")
            run([uv, "build", "--wheel", "--out-dir", str(workspace / "dist"), str(project)],
                cwd=project)
            wheel, = (workspace / "dist").glob("*.whl")
        installers = ("pip", "uv") if arguments.installer == "both" else (arguments.installer,)
        for installer in installers:
            if installer == "uv" and not uv:
                parser.error("the uv installation smoke requires uv on PATH")
            trial = workspace / installer
            trial.mkdir()
            client = trial / "client"
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(client)
            python = client / "bin/python"
            if os.name == "nt":
                parser.error("the SSH Worker deployment smoke requires a POSIX host")
            command = (
                [uv, "pip", "install", "--python", str(python), str(wheel)]
                if installer == "uv"
                else [
                    str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel),
                ]
            )
            run(command, cwd=trial)
            run(
                [str(python), "-I", "-c", CLIENT_SMOKE, str(trial),
                 "legacy-pip" if arguments.legacy_pip else ""],
                cwd=trial,
            )
            print(f"{installer} distribution smoke passed")


if __name__ == "__main__":
    main()
