"""Durable records and filesystem storage for job supervisors."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpuops.contracts import is_valid_run_id

SCHEMA_VERSION = "gpuops-supervisor/v1"


class SupervisorError(RuntimeError):
    """Base class for supervisor failures."""


class LaunchInProgress(SupervisorError):
    """A supervisor owns the run lock but has not published a final receipt."""


class DefinitiveLaunchFailure(RuntimeError):
    """A launch failed before any supervisor process could take ownership."""


class IntentConflict(SupervisorError):
    """A run id was reused with different launch parameters."""


class InvalidReceipt(SupervisorError):
    """A receipt is missing, corrupt, or does not match its launch intent."""


class UnsafeSignal(SupervisorError):
    """The stored process credentials do not identify the current process."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int | None
    boot_id: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessIdentity:
        return cls(
            pid=int(value["pid"]),
            pgid=int(value["pgid"]),
            start_ticks=(
                int(value["start_ticks"]) if value.get("start_ticks") is not None else None
            ),
            boot_id=str(value["boot_id"]) if value.get("boot_id") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class SupervisorReceipt:
    run_id: str
    launch_token: str
    status: str
    accepted_at: str
    supervisor: ProcessIdentity
    child: ProcessIdentity | None = None
    child_started_at: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SupervisorReceipt:
        if value.get("schema") != SCHEMA_VERSION:
            raise InvalidReceipt("unsupported receipt schema")
        child = value.get("child")
        return cls(
            run_id=str(value["run_id"]),
            launch_token=str(value["launch_token"]),
            status=str(value["status"]),
            accepted_at=str(value["accepted_at"]),
            supervisor=ProcessIdentity.from_dict(value["supervisor"]),
            child=ProcessIdentity.from_dict(child) if child is not None else None,
            child_started_at=(
                str(value["child_started_at"])
                if value.get("child_started_at") is not None
                else None
            ),
        )

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "launch_token": self.launch_token,
            "status": self.status,
            "accepted_at": self.accepted_at,
            "supervisor": asdict(self.supervisor),
            "child": asdict(self.child) if self.child is not None else None,
            "child_started_at": self.child_started_at,
        }


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    run_id: str
    launch_token: str
    exit_code: int
    started_at: str | None
    finished_at: str
    launch_error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SupervisorResult:
        if value.get("schema") != SCHEMA_VERSION:
            raise SupervisorError("unsupported result schema")
        return cls(
            run_id=str(value["run_id"]),
            launch_token=str(value["launch_token"]),
            exit_code=int(value["exit_code"]),
            started_at=str(value["started_at"]) if value.get("started_at") else None,
            finished_at=str(value["finished_at"]),
            launch_error=str(value["launch_error"]) if value.get("launch_error") else None,
        )

    def as_json_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SupervisorPaths:
    directory: Path
    lock: Path
    intent: Path
    receipt: Path
    result: Path
    cancellation: Path
    console: Path

    @classmethod
    def create(
        cls,
        root: Path,
        run_id: str,
        *,
        console_root: Path | None = None,
    ) -> SupervisorPaths:
        if not is_valid_run_id(run_id):
            raise ValueError("invalid run_id")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        directory = root / run_id
        directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        console = (
            directory / "console.log"
            if console_root is None
            else console_root / run_id / "console.log"
        )
        return cls(
            directory=directory,
            lock=directory / "launch.lock",
            intent=directory / "intent.json",
            receipt=directory / "receipt.json",
            result=directory / "result.json",
            cancellation=directory / "cancellation.json",
            console=console,
        )


def requested_intent(
    run_id: str,
    argv: Sequence[str],
    cwd: str | Path,
    environment: Mapping[str, str] | None,
    console_path: Path,
    console_max_bytes: int,
    console_backups: int,
    reservation_path: Path | None = None,
) -> dict[str, Any]:
    arguments = list(argv)
    if not arguments or any(not isinstance(value, str) or "\x00" in value for value in arguments):
        raise ValueError("argv must contain strings without NUL bytes")
    working_directory = str(Path(cwd))
    if "\x00" in working_directory or not Path(working_directory).is_absolute():
        raise ValueError("cwd must be an absolute path without NUL bytes")
    env = dict(environment or {})
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or "\x00" in key
        or "=" in key
        or "\x00" in value
        for key, value in env.items()
    ):
        raise ValueError("environment must contain valid string keys and values")
    if console_max_bytes <= 0 or console_backups < 0:
        raise ValueError("console limits must be positive with non-negative backups")
    return {
        "run_id": run_id,
        "argv": arguments,
        "cwd": working_directory,
        "environment": env,
        "console_path": str(console_path),
        "console_max_bytes": console_max_bytes,
        "console_backups": console_backups,
        "reservation_path": str(reservation_path) if reservation_path is not None else None,
    }


def ensure_same_intent(existing: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    for key, value in requested.items():
        if existing.get(key) != value:
            raise IntentConflict(f"run id already has a different {key}")


def read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as error:
        raise SupervisorError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise SupervisorError(f"{path.name} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_aware_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise SupervisorError(f"cancellation {field_name} must be an ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise SupervisorError(f"cancellation {field_name} is invalid") from error
    if timestamp.tzinfo is None:
        raise SupervisorError(f"cancellation {field_name} must include a timezone")
    return timestamp


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lock_is_held(path: Path) -> bool:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)
