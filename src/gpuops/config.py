from __future__ import annotations

import fcntl
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import Field

from gpuops.models import StrictModel

# Shared defaults tolerate slow NVIDIA queries without adding job parameters.
GPU_QUERY_TIMEOUT_SECONDS = 25.0
MONITOR_TIMEOUT_SECONDS = 35.0


@dataclass(frozen=True, slots=True)
class ServerTarget:
    id: str
    label: str
    ssh_host: str


def validate_server_id(server_id: str) -> None:
    if (
        not isinstance(server_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", server_id)
        or server_id == "auto"
    ):
        raise ValueError(
            "server id must be a name of 1-96 letters, digits, dots, _ or - (not auto)"
        )


def registry_path() -> Path:
    return Path.home() / ".config/gpuops/servers.json"


def _target_from_record(value: object) -> ServerTarget:
    if not isinstance(value, dict):
        raise ValueError("server entry must be an object")
    server_id = value.get("id")
    validate_server_id(server_id)
    ssh_host = value.get("ssh_host")
    label = value.get("label", server_id)
    if not isinstance(ssh_host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", ssh_host):
        raise ValueError("server ssh_host must be an SSH alias or user@host")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("server label must be non-empty")
    return ServerTarget(server_id, label, ssh_host)


def load_targets() -> tuple[ServerTarget, ...]:
    path = registry_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("servers"), list):
            raise ValueError("expected an object containing a servers list")
        targets = tuple(_target_from_record(item) for item in value["servers"])
        if len({target.id for target in targets}) != len(targets):
            raise ValueError("duplicate server ids")
        if len({target.ssh_host for target in targets}) != len(targets):
            raise ValueError("duplicate SSH routes")
        return targets
    except FileNotFoundError:
        return ()
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"invalid server registry {path}: {error}") from error


def _update_registry(target: ServerTarget | None, remove_id: str | None = None) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        targets = list(load_targets())
        if target is not None:
            _target_from_record(asdict(target))
            existing = next((item for item in targets if item.id == target.id), None)
            if existing is not None:
                if existing.ssh_host != target.ssh_host:
                    raise ValueError(
                        f"server {target.id!r} is already registered at {existing.ssh_host}"
                    )
                return
            if any(item.ssh_host == target.ssh_host for item in targets):
                raise ValueError(f"SSH route {target.ssh_host!r} is already registered")
            targets.append(target)
        else:
            if not any(item.id == remove_id for item in targets):
                raise ValueError(f"unknown server: {remove_id}")
            targets = [item for item in targets if item.id != remove_id]
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(
                    {"servers": [asdict(item) for item in targets]},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def register_target(target: ServerTarget) -> None:
    _update_registry(target)


def unregister_target(server_id: str) -> None:
    _update_registry(None, remove_id=server_id)


def select_targets(server_ids: list[str] | None = None) -> tuple[ServerTarget, ...]:
    """Resolve logical server IDs using the same mapping for every interface."""
    targets = load_targets()
    if not server_ids:
        return targets
    unknown = set(server_ids) - {target.id for target in targets}
    if unknown:
        raise ValueError(f"unknown server IDs: {', '.join(sorted(unknown))}")
    return tuple(target for target in targets if target.id in server_ids)


def local_target() -> ServerTarget | None:
    """Read host.json, filling older identities from the registry or server ID."""
    path = Path.home() / ".local/share/gpuops/host.json"
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GPUOps installation identity: {path}") from error
    server_id = identity.get("server_id") if isinstance(identity, dict) else None
    try:
        validate_server_id(server_id)
        registered = None
        if "label" not in identity or "ssh_host" not in identity:
            registered = next((item for item in load_targets() if item.id == server_id), None)
        return _target_from_record(
            {
                "id": server_id,
                "label": identity.get("label", registered.label if registered else server_id),
                "ssh_host": identity.get(
                    "ssh_host", registered.ssh_host if registered else server_id
                ),
            }
        )
    except ValueError as error:
        raise ValueError(f"invalid GPUOps installation server_id in {path}: {error}") from error


def select_client_targets(server_ids: list[str] | None = None) -> tuple[ServerTarget, ...]:
    """Default server-side clients to this installation; controllers see the fleet."""
    local = local_target()
    if not server_ids:
        return (local,) if local is not None else select_targets()
    if local is None:
        return select_targets(server_ids)
    targets = (local,) + tuple(item for item in load_targets() if item.id != local.id)
    unknown = set(server_ids) - {item.id for item in targets}
    if unknown:
        raise ValueError(f"unknown server ids: {', '.join(sorted(unknown))}")
    return tuple(item for item in targets if item.id in server_ids)


class SchedulerSettings(StrictModel):
    lease_ttl_seconds: float = Field(default=30.0, gt=0)
    idle_confirmation_delay_seconds: float = Field(default=2.0, ge=0)
    max_snapshot_age_seconds: float = Field(default=15.0, gt=0)


class WorkerSettings(StrictModel):
    server_id: str
    worker_id: str
    state_directory: Path = Path("~/.local/state/gpuops").expanduser()
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
