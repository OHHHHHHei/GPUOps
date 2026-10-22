"""Build and transfer an installable copy of the current GPUOps version."""

from __future__ import annotations

import io
import re
import subprocess
import tarfile
import uuid
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from gpuops import __version__
from gpuops.bundled import project_files
from gpuops.deployment.contracts import (
    REMOTE_CACHE_DIRECTORY,
    CommandResult,
    CommandRunner,
    DeployError,
    DeploymentArtifact,
    WheelArtifact,
)

_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")
_WHEEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl$")
_ARTIFACT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*\.(?:whl|tar\.gz)$")
_RELEASE_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{64})$")
_DEPLOYMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def build_artifact(
    output_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    command_runner: CommandRunner | None = None,
) -> DeploymentArtifact:
    """Package the installed client, or build an explicitly selected checkout."""
    if project_root is not None:
        return build_wheel(project_root, output_directory, command_runner=command_runner)
    return build_source_archive(output_directory)


def build_source_archive(output_directory: str | Path) -> DeploymentArtifact:
    """Restore a pip-installable source archive from installed code and metadata.

    GPUOps is pure Python. Shipping these files with their original pyproject lets
    pip use the standard build backend on the remote host, without a checkout,
    a local build tool, or a package-index lookup for GPUOps itself.
    """
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"gpuops-{__version__}.tar.gz"
    prefix = f"gpuops-{__version__}"
    with tarfile.open(archive, "w:gz") as handle:
        for name, content in project_files().items():
            _add_source_file(handle, f"{prefix}/{name}", content.encode("utf-8"))
        _add_package(handle, files("gpuops"), f"{prefix}/src/gpuops")
    return DeploymentArtifact(archive, uuid.uuid4().hex, archive.stat().st_size)


def _add_source_file(handle: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    handle.addfile(info, io.BytesIO(content))


def _add_package(handle: tarfile.TarFile, directory, prefix: str) -> None:
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name in {"__pycache__", "_bundled"} or entry.name.endswith((".pyc", ".pyo")):
            continue
        target = f"{prefix}/{entry.name}"
        if entry.is_dir():
            _add_package(handle, entry, target)
        elif entry.is_file():
            _add_source_file(handle, target, entry.read_bytes())


def build_wheel(
    project_root: str | Path,
    output_directory: str | Path,
    *,
    command_runner: CommandRunner | None = None,
) -> WheelArtifact:
    """Build exactly one wheel and assign it a unique release ID."""

    root = Path(project_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.whl"):
        stale.unlink()
    runner = command_runner or _default_command_runner
    command = ["uv", "build", "--wheel", "--out-dir", str(output), str(root)]
    result = runner(command, cwd=root)
    _require_local_success(result, "wheel build")
    wheels = sorted(output.glob("*.whl"))
    if len(wheels) != 1:
        raise DeployError(f"wheel build produced {len(wheels)} wheels; expected exactly one")
    wheel = wheels[0]
    if not _WHEEL_PATTERN.fullmatch(wheel.name):
        raise DeployError(f"unsafe wheel filename: {wheel.name!r}")
    return WheelArtifact(
        path=wheel,
        release_id=uuid.uuid4().hex,
        size_bytes=wheel.stat().st_size,
    )


def copy_artifact(
    host: str,
    artifact: DeploymentArtifact,
    *,
    deployment_id: str,
    command_runner: CommandRunner | None = None,
) -> None:
    """Copy an installable archive to the fixed GPUOps cache using system scp."""

    validate_host(host)
    if not _RELEASE_ID_PATTERN.fullmatch(artifact.release_id):
        raise ValueError("artifact release id is invalid")
    if not _ARTIFACT_PATTERN.fullmatch(artifact.filename):
        raise ValueError("artifact filename is unsafe")
    if not _DEPLOYMENT_ID_PATTERN.fullmatch(deployment_id):
        raise ValueError("deployment id is invalid")
    runner = command_runner or _default_command_runner
    destination = (
        f"{host}:{REMOTE_CACHE_DIRECTORY}/uploads/{artifact.release_id}/"
        f"{deployment_id}/{artifact.filename}"
    )
    result = runner(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            str(artifact.path),
            destination,
        ],
        cwd=None,
    )
    _require_local_success(result, "artifact upload")


copy_wheel = copy_artifact


def validate_host(host: str) -> None:
    if not _HOST_PATTERN.fullmatch(host):
        raise ValueError(
            "host must be a strict SSH alias or user@host without whitespace, ports, or options"
        )


def _default_command_runner(
    command: Sequence[str], *, cwd: str | Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_local_success(result: CommandResult, operation: str) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or f"exit status {result.returncode}").strip()
    raise DeployError(f"{operation} failed: {detail}")
