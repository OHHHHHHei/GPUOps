"""Shared deployment constants and value types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REMOTE_CACHE_DIRECTORY = "~/.cache/gpuops"
REMOTE_TOTAL_BUDGET_BYTES = 1024 * 1024 * 1024
REMOTE_SHARE_BUDGET_BYTES = 128 * 1024 * 1024
REMOTE_CACHE_BUDGET_BYTES = 128 * 1024 * 1024
REMOTE_INSTALL_BUDGET_BYTES = REMOTE_SHARE_BUDGET_BYTES + REMOTE_CACHE_BUDGET_BYTES
REMOTE_STATE_BUDGET_BYTES = REMOTE_TOTAL_BUDGET_BYTES - REMOTE_INSTALL_BUDGET_BYTES


class DeployError(RuntimeError):
    """A local build, transfer, or remote deployment action failed."""


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[..., CommandResult]


@dataclass(frozen=True, slots=True)
class DeploymentArtifact:
    """A wheel or standard source archive ready to upload and install with pip."""

    path: Path
    release_id: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return self.path.name


# Kept for callers that construct or inspect the original wheel-only artifact.
WheelArtifact = DeploymentArtifact


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    release_id: str
    installed_at: str | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DeploymentStatus:
    host: str
    installed: bool
    current_release: str | None
    previous_release: str | None
    releases: tuple[ReleaseInfo, ...]
    share_bytes: int
    budget_bytes: int
    cache_bytes: int
    state_bytes: int
    total_bytes: int
    total_budget_bytes: int
    worker_ready: bool
    worker_release: str | None
    executable: str | None
    server_id: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentResult:
    action: str
    host: str
    changed: bool
    message: str
    status: DeploymentStatus
