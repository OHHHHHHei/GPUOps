"""Linux process identity and owned process-group inspection."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path

from gpuops.worker.supervisor_records import ProcessIdentity, UnsafeSignal


@dataclass(frozen=True, slots=True)
class ProcfsRecord:
    """Raw stat fields and boot ID shared by process identity adapters."""

    pid: int
    stat_fields: tuple[str, ...]
    boot_id: str


def read_procfs_record(
    pid: int,
    proc_root: str | Path = "/proc",
    *,
    suppress_os_errors: bool = False,
) -> ProcfsRecord | None:
    """Read one process stat record and boot ID with caller-selected I/O policy."""

    if pid <= 0:
        return None
    root = Path(proc_root)
    try:
        stat = (root / str(pid) / "stat").read_text(encoding="utf-8")
        boot_id = (root / "sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        stat_fields = tuple(_proc_stat_fields(stat))
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    except OSError:
        if suppress_os_errors:
            return None
        raise
    return ProcfsRecord(
        pid=pid,
        stat_fields=stat_fields,
        boot_id=boot_id,
    )


def _proc_stat_fields(stat: str) -> list[str]:
    """Return proc stat fields beginning with the process state."""

    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process stat has no closing command delimiter")
    return stat[closing_parenthesis + 2 :].split()


def parse_proc_group_fields(fields: tuple[str, ...] | list[str]) -> tuple[str, int]:
    """Parse process state and process-group ID from split proc stat fields."""

    return fields[0], int(fields[2])


def parse_proc_group(stat: str) -> tuple[str, int]:
    """Parse process state and process-group ID from Linux proc stat."""

    return parse_proc_group_fields(_proc_stat_fields(stat))


def parse_proc_start_ticks(fields: tuple[str, ...] | list[str]) -> int:
    """Parse the process start time from split Linux proc stat fields."""

    return int(fields[19])


def read_process_identity(
    pid: int,
    *,
    proc_root: str | Path = "/proc",
) -> ProcessIdentity | None:
    """Read PID-reuse-resistant supervisor identity fields from procfs."""

    record = read_procfs_record(pid, proc_root, suppress_os_errors=True)
    if record is None:
        return None
    try:
        _state, pgid = parse_proc_group_fields(record.stat_fields)
        start_ticks = parse_proc_start_ticks(record.stat_fields)
    except (ValueError, IndexError):
        return None
    return ProcessIdentity(
        pid=record.pid,
        pgid=pgid,
        start_ticks=start_ticks,
        boot_id=record.boot_id,
    )


def process_identity_matches(
    identity: ProcessIdentity,
    *,
    proc_root: str | Path = "/proc",
) -> bool:
    """Return true only when all persistent process credentials still match."""

    if identity.start_ticks is None or not identity.boot_id:
        return False
    current = read_process_identity(identity.pid, proc_root=proc_root)
    return current == identity


def process_group_state(identity: ProcessIdentity, proc_root: str | Path) -> str:
    """Confirm the original process group is empty before treating it as gone."""

    leader_state = _process_identity_state(identity, proc_root)
    if leader_state == "replaced":
        # A PGID equal to the reused PID belongs to the replacement. The
        # original group could not retain that identifier across PID reuse.
        return "gone"
    if leader_state not in {"gone", "zombie"}:
        return leader_state
    root = Path(proc_root)
    if not root.exists():
        return "gone"
    if identity.start_ticks is None or not identity.boot_id:
        return "unknown"
    try:
        current_boot_id = root.joinpath("sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return "unknown"
    if current_boot_id != identity.boot_id:
        return "gone"
    try:
        process_directories = list(root.iterdir())
    except OSError:
        return "unknown"
    unreadable = False
    for process_directory in process_directories:
        if not process_directory.name.isdigit():
            continue
        try:
            stat = process_directory.joinpath("stat").read_text(encoding="utf-8")
            state, pgid = parse_proc_group(stat)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            unreadable = True
            continue
        if pgid == identity.pgid and state != "Z":
            return "members"
    return "unknown" if unreadable else "gone"


def signal_process(
    identity: ProcessIdentity,
    sig: int = signal.SIGTERM,
    *,
    proc_root: str | Path = "/proc",
) -> None:
    """Signal a process group only after validating all stored credentials."""

    if identity.pid <= 1 or identity.pgid != identity.pid:
        raise UnsafeSignal("refusing to signal a non-session-leader process group")
    if not process_identity_matches(identity, proc_root=proc_root):
        raise UnsafeSignal("process credentials no longer match")
    os.killpg(identity.pgid, sig)


def signal_owned_process_group(
    identity: ProcessIdentity,
    sig: int,
    *,
    proc_root: str | Path,
) -> None:
    """Signal an owned group while its leader or original descendants remain."""

    if identity.pid <= 1 or identity.pgid != identity.pid:
        raise UnsafeSignal("refusing to signal a non-session-leader process group")
    if identity.start_ticks is None or not identity.boot_id:
        raise UnsafeSignal("process group identity is incomplete")
    if process_group_state(identity, proc_root) not in {"matched", "members"}:
        raise UnsafeSignal("process group ownership can no longer be verified")
    os.killpg(identity.pgid, sig)


def identity_with_fallback(
    pid: int,
    *,
    fallback_pgid: int | None = None,
) -> ProcessIdentity:
    identity = read_process_identity(pid)
    if identity is not None:
        return identity
    return ProcessIdentity(
        pid=pid,
        pgid=fallback_pgid if fallback_pgid is not None else os.getpgid(pid),
        start_ticks=None,
        boot_id=None,
    )


def current_process_identity() -> ProcessIdentity:
    return identity_with_fallback(os.getpid())


def _process_identity_state(identity: ProcessIdentity, proc_root: str | Path) -> str:
    """Classify a stored identity without treating unreadable credentials as gone."""

    process_directory = Path(proc_root) / str(identity.pid)
    if not process_directory.exists():
        return "gone"
    if identity.start_ticks is None or not identity.boot_id:
        return "unknown"
    record = read_procfs_record(identity.pid, proc_root, suppress_os_errors=True)
    if record is None:
        return "unknown"
    try:
        state, pgid = parse_proc_group_fields(record.stat_fields)
        start_ticks = parse_proc_start_ticks(record.stat_fields)
    except (ValueError, IndexError):
        return "unknown"
    current = ProcessIdentity(
        pid=record.pid,
        pgid=pgid,
        start_ticks=start_ticks,
        boot_id=record.boot_id,
    )
    if current != identity:
        return "replaced"
    return "zombie" if state == "Z" else "matched"
