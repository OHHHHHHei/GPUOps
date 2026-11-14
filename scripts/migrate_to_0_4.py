#!/usr/bin/env python3
"""One-time GPUOps 0.4 data cutover. Run from the Mac during the maintenance window."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

STATE_DIRECTORY = Path(".local/state/gpuops")
BACKUP_DIRECTORY = Path("gpuops-0.4-backup")
REGISTRY = Path(".config/gpuops/servers.json")
JOBS = Path("queue/jobs")
JOURNAL = Path("control-journal")
EVENTS = Path("followups/records")
GROUPS = Path("followups/groups")
LAUNCH_AGENTS = ("io.gpuops.worker-alerts", "io.gpuops.followups", "io.gpuops.background")
ACTIVE = frozenset({"leased", "starting", "running", "cancel_requested"})
PHASES = ("check", "backup", "convert", "check-restore", "restore")


def object_value(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def convert_record(value: dict[str, Any], directory: Path) -> None:
    def spec(value: Any, field: str) -> None:
        object_value(value, field).pop("tracking", None)

    def job(value: Any, field: str) -> dict[str, Any]:
        record = object_value(value, field)
        spec(record.get("spec"), f"{field}.spec")
        record.setdefault("exit_code", None)
        record.setdefault("launch_error", None)
        return record

    if directory == JOBS:
        job(value, "job")
    elif directory == JOURNAL:
        spec(value.get("request_spec"), "request_spec")
        spec(value.get("frozen_spec"), "frozen_spec")
        if value.get("remote_record") is not None:
            job(value["remote_record"], "remote_record")
    elif directory == EVENTS:
        observed = job(value["job"], "job") if value.get("job") is not None else None
        outcomes = value.setdefault("outcomes", [])
        if not isinstance(outcomes, list):
            raise ValueError("outcomes must be a JSON list")
        if (
            not outcomes
            and observed is not None
            and value.get("group_id") is None
            and value.get("mode", "per_job") == "per_job"
        ):
            outcomes.append(
                {
                    "server_id": value["server_id"],
                    "run_id": observed["run_id"],
                    "status": observed["status"],
                    "exit_code": observed["exit_code"],
                    "failure_reason": observed.get("failure_reason"),
                    "remote_workspace": observed["spec"]["working_directory"],
                    "project_workspace": value["workspace"],
                }
            )
    elif directory == GROUPS:
        members = value.get("members")
        if not isinstance(members, list):
            raise ValueError("members must be a JSON list")
        for member in members:
            member = object_value(member, "members[]")
            if member.get("job") is not None:
                job(member["job"], "members[].job")


def atomic_write(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(payload)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def record_paths(root: Path, directories: tuple[Path, ...]) -> Iterator[Path]:
    for directory in directories:
        for path in sorted((root / directory).glob("*.json")):
            yield path.relative_to(root)


def check_launch_agents() -> None:
    for label in LAUNCH_AGENTS:
        service = f"gui/{os.getuid()}/{label}"
        result = subprocess.run(
            ["/bin/launchctl", "print", service], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            raise RuntimeError(f"LaunchAgent {service} is loaded; boot it out before continuing")
        if "Could not find service" not in result.stderr:
            raise RuntimeError(f"cannot check LaunchAgent {service}: {result.stderr.strip()}")


class Machine:
    def __init__(self, role: str, state: Path, backup: Path):
        self.role = role
        self.state = state.expanduser().resolve()
        self.backup = backup.expanduser().resolve()
        if self.backup == self.state or self.state in self.backup.parents:
            raise ValueError("backup directory must be outside the GPUOps state directory")
        self.snapshot = self.backup / "records"
        self.directories = (JOBS, JOURNAL) if role == "server" else (JOURNAL, EVENTS, GROUPS)

    @contextmanager
    def stopped(self) -> Iterator[None]:
        if self.role == "mac" and sys.platform == "darwin":
            check_launch_agents()
        locks = [("control-journal/journal.lock", "the submission journal is in use")]
        if self.role == "server":
            locks[:0] = [
                ("worker.lock", "the Worker is still running"),
                ("queue/queue.lock", "the queue is in use"),
            ]
        else:
            # Event and group changes share the event store's lock.
            locks.append(("followups/records/followups.lock", "follow-up records are in use"))
        with ExitStack() as stack:
            for relative, error in locks:
                path = self.state / relative
                try:
                    handle = stack.enter_context(path.open("rb"))
                except FileNotFoundError:
                    # The live writer creates its own lock. The runbook pauses writers.
                    continue
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as cause:
                    raise RuntimeError(f"{error}: {path}") from cause
            if self.role == "server":
                for relative in record_paths(self.state, (JOBS,)):
                    record = self.read_record(relative)
                    if record.get("status") in ACTIVE:
                        raise RuntimeError(f"{relative}: job is {record['status']}")
            yield

    def read_record(self, relative: Path) -> dict[str, Any]:
        path = self.state / relative
        try:
            return object_value(json.loads(path.read_bytes()), "record")
        except (UnicodeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    def converted_records(self) -> Iterator[tuple[Path, bytes | None]]:
        for relative in record_paths(self.state, self.directories):
            path = self.state / relative
            try:
                value = self.read_record(relative)
                original = json.dumps(value)
                convert_record(value, relative.parent)
                if json.dumps(value) == original:
                    yield relative, None
                else:
                    yield (
                        relative,
                        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(),
                    )
            except (KeyError, UnicodeError, ValueError) as error:
                raise ValueError(f"{path}: {error}") from error

    def manifest(self, files: list[Path]) -> dict[str, Any]:
        return {
            "role": self.role,
            "state": str(self.state),
            "directories": [path.as_posix() for path in self.directories],
            "files": [path.as_posix() for path in files],
        }

    def backed_up_records(self) -> list[Path]:
        path = self.snapshot / "manifest.json"
        value = object_value(json.loads(path.read_bytes()), "backup manifest")
        if any(
            value.get(key) != expected
            for key, expected in self.manifest([]).items()
            if key != "files"
        ):
            raise ValueError(f"{path}: backup is for a different set of state directories")
        files = value.get("files")
        if not isinstance(files, list) or any(not isinstance(file, str) for file in files):
            raise ValueError(f"{path}: backup files must be a list of paths")
        paths = [Path(file) for file in files]
        for relative in paths:
            if relative.parent not in self.directories or relative.suffix != ".json":
                raise ValueError(f"{path}: unmanaged backup path {relative}")
            if not (self.snapshot / relative).is_file():
                raise ValueError(f"{path}: missing backup file {relative}")
        return paths

    def create_backup(self) -> tuple[int, bool]:
        if self.snapshot.exists():
            return len(self.backed_up_records()), False
        self.backup.mkdir(mode=0o700, parents=True, exist_ok=True)
        files = list(record_paths(self.state, self.directories))
        with tempfile.TemporaryDirectory(prefix=".records-", dir=self.backup) as directory:
            temporary = Path(directory)
            for relative in files:
                destination = temporary / relative
                destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                shutil.copyfile(self.state / relative, destination)
            (temporary / "manifest.json").write_text(
                json.dumps(self.manifest(files), indent=2) + "\n", encoding="utf-8"
            )
            temporary.rename(self.snapshot)
        return len(files), True

    def remove_index(self) -> None:
        if self.role == "server":
            for name in ("queue-index.sqlite3", "queue-index.sqlite3-journal"):
                (self.state / "queue" / name).unlink(missing_ok=True)

    def run(self, phase: str) -> dict[str, Any]:
        with self.stopped():
            if phase == "check":
                return {"records": sum(1 for _ in self.converted_records())}
            if phase == "backup":
                count, created = self.create_backup()
                return {"records": count, "created": created}
            originals = self.backed_up_records()
            if phase == "check-restore":
                return {"records": len(originals)}
            if phase == "restore":
                extra = set(record_paths(self.state, self.directories)).difference(originals)
                for relative in originals:
                    path = self.state / relative
                    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                    atomic_write(path, (self.snapshot / relative).read_bytes())
                for relative in sorted(extra):
                    (self.state / relative).unlink()
                self.remove_index()
                return {"records": len(originals), "removed": len(extra)}
            count = 0
            for relative, payload in self.converted_records():
                if payload is not None:
                    atomic_write(self.state / relative, payload)
                    count += 1
            self.remove_index()
            if self.role == "server":
                (self.state / "gpu-sample.lock").unlink(missing_ok=True)
            return {"records": count}


def ssh_targets(registry: Path) -> list[tuple[str, str]]:
    try:
        value = object_value(json.loads(registry.read_bytes()), "server registry")
    except FileNotFoundError:
        return []
    servers = value.get("servers")
    if not isinstance(servers, list):
        raise ValueError(f"{registry}: expected a servers list")
    targets: list[tuple[str, str]] = []
    for value in servers:
        server = object_value(value, "server registry entry")
        server_id = server.get("id")
        host = server.get("ssh_host")
        if not isinstance(server_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", server_id
        ):
            raise ValueError(f"{registry}: invalid server id")
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            raise ValueError(f"{registry}: invalid SSH destination for {server_id}")
        targets.append((server_id, host))
    if len({server for server, _ in targets}) != len(targets) or len(
        {host for _, host in targets}
    ) != len(targets):
        raise ValueError(f"{registry}: duplicate server ID or SSH destination")
    return targets


def run_remote(host: str, source: bytes, phase: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            host,
            "python3",
            "-u",
            "-",
            "--on-server",
            phase,
        ],
        input=source,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "SSH failed")
    return object_value(json.loads(result.stdout), "remote result")


def main() -> int:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check every host without writing")
    mode.add_argument("--backup-only", action="store_true", help="check and back up every host")
    mode.add_argument(
        "--restore", action="store_true", help="restore the first backup on every host"
    )
    mode.add_argument("--on-server", choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--registry", type=Path, default=home / REGISTRY)
    parser.add_argument("--state-dir", type=Path, default=home / STATE_DIRECTORY)
    parser.add_argument("--backup-dir", type=Path, default=home / BACKUP_DIRECTORY)
    parser.add_argument("--local-only", choices=("mac", "server"), help="use only local state")
    args = parser.parse_args()

    try:
        if args.on_server:
            machine = Machine("server", args.state_dir, args.backup_dir)
            print(json.dumps(machine.run(args.on_server)))
            return 0
        role = args.local_only or "mac"
        local = Machine(role, args.state_dir, args.backup_dir)
        remotes = [] if args.local_only else ssh_targets(args.registry.expanduser())
        source = Path(__file__).read_bytes() if remotes else b""
        if args.restore:
            phases = ("check-restore", "restore")
        elif args.check:
            phases = ("check",)
        elif args.backup_only:
            phases = ("check", "backup")
        else:
            phases = ("check", "backup", "convert")
        for phase in phases:
            try:
                result = local.run(phase)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"{phase} on local {role}: {error}") from error
            print(f"{phase} local {role}: {json.dumps(result)}", flush=True)
            for server, host in remotes:
                try:
                    result = run_remote(host, source, phase)
                except (OSError, RuntimeError, ValueError) as error:
                    raise RuntimeError(f"{phase} on {server} ({host}): {error}") from error
                print(f"{phase} {server}: {json.dumps(result)}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
