"""Bounded, read-only log discovery sent over SSH by the Web console.

This module uses only the standard library; it also runs without GPUOps installed.
"""

import datetime
import json
import os
import stat
import time
from pathlib import Path

PROC = Path("/proc")
FILE_FLAGS = {"--log_file", "--log-file", "--logfile", "--log_path", "--log-path"}
DIR_FLAGS = {"--log_dir", "--log-dir", "--logging_dir", "--logging-dir"}
HEAD_BYTES = 8192
TAIL_BYTES = 49152
MAX_FILES = 16


def identity(pid):
    value = (PROC / str(pid) / "stat").read_text()
    fields = value[value.rfind(")") + 2 :].split()
    return int(fields[19])


def managed_log(pid):
    """Follow a bounded parent chain to the existing GPUOps launch receipt.

    GPUOps children write to a pipe, whose reader saves console.log. Matching
    the receipt to both processes also covers torchrun/DDP descendants without
    searching the queue or experiment directories.
    """
    child = None
    for _ in range(8):
        if pid <= 1:
            break
        root = PROC / str(pid)
        value = (root / "stat").read_text()
        fields = value[value.rfind(")") + 2 :].split()
        ticks = int(fields[19])
        with (root / "cmdline").open("rb") as handle:
            argv = handle.read(32768).decode("utf-8", errors="replace").split("\0")
        supervisor = any(
            arg.endswith("/gpuops/worker/supervisor.py") or arg == "gpuops.worker.supervisor"
            for arg in argv
        )
        if child and supervisor and "_supervise" in argv and "--intent" in argv:
            path = Path(argv[argv.index("--intent") + 1])
            intent = read_json(path)
            receipt = read_json(path.with_name("receipt.json"))
            boot_id = (PROC / "sys/kernel/random/boot_id").read_text().strip()
            if (
                receipt["supervisor"]["pid"] == pid
                and receipt["supervisor"]["boot_id"] == boot_id
                and receipt["supervisor"]["start_ticks"] == ticks
                and receipt["child"]["pid"] == child[0]
                and receipt["child"]["boot_id"] == boot_id
                and receipt["child"]["start_ticks"] == child[1]
                and receipt["launch_token"] == intent["launch_token"]
                and receipt["run_id"] == intent["run_id"]
                and identity(pid) == ticks
                and identity(child[0]) == child[1]
            ):
                return {"path": intent["console_path"], "run_id": receipt["run_id"]}
            return None
        child = (pid, ticks)
        pid = int(fields[1])
    return None


def read_json(path):
    # Launch metadata is small. Never block on a pipe or read an unbounded file.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("not a regular launch record")
        return json.loads(handle.read(65536))


def log_paths(pid, argv):
    """Only explicit log arguments and regular stdout/stderr destinations."""
    paths = []
    errors = []
    root = PROC / str(pid)
    for index, argument in enumerate(argv):
        flag, separator, value = argument.partition("=")
        if flag not in FILE_FLAGS | DIR_FLAGS:
            continue
        if not separator:
            value = argv[index + 1] if index + 1 < len(argv) else ""
        if not value or value.startswith("--"):
            continue
        try:
            path = Path(value)
            if not path.is_absolute():
                path = Path(os.readlink(root / "cwd")) / path
            if flag in FILE_FLAGS:
                paths.append(str(path))
                continue
            # A few regular log files in this exact directory, never a recursive scan.
            entries = []
            with os.scandir(path) as scan:
                for count, entry in enumerate(scan):
                    if count >= 32:
                        break
                    if entry.name.endswith((".log", ".out", ".txt")):
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISREG(info.st_mode):
                            entries.append((entry.name == "train.log", info.st_mtime, entry.path))
            paths.extend(entry[2] for entry in sorted(entries, reverse=True)[:2])
        except OSError as error:
            errors.append(type(error).__name__)
    for descriptor in ("1", "2"):
        try:
            target = os.readlink(root / "fd" / descriptor)
            if target.startswith("/") and not target.endswith(" (deleted)"):
                paths.append(target)
        except OSError:
            pass
    return list(dict.fromkeys(paths))[:3], errors


def read_log(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            return {"error": "NotRegularFile"}
        head = handle.read(HEAD_BYTES)
        offset = max(len(head), info.st_size - TAIL_BYTES)
        handle.seek(offset)
        tail = handle.read(TAIL_BYTES)
    if offset > len(head):
        # Do not parse a partial record at the tail boundary.
        boundary = min((i for i in (tail.find(b"\n"), tail.find(b"\r")) if i >= 0), default=-1)
        tail = tail[boundary + 1 :] if boundary >= 0 else b""
    return {
        "content": (head + (b"\n" if offset > len(head) else b"") + tail).decode(
            "utf-8", errors="replace"
        ),
        "modified_at": info.st_mtime,
        "truncated": offset > len(head),
    }


def collect(requests):
    now = time.time()
    uptime = float((PROC / "uptime").read_text().split()[0])
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    processes, logs = [], {}
    for request in requests[:32]:
        pid, ticks = request["pid"], request["start_time_ticks"]
        row = {"pid": pid, "start_time_ticks": ticks, "paths": []}
        try:
            if identity(pid) != ticks:
                row["error"] = "ProcessChanged"
                processes.append(row)
                continue
            with (PROC / str(pid) / "cmdline").open("rb") as handle:
                argv = handle.read(32768).split(b"\0")
            row["argv"] = [part.decode("utf-8", errors="replace") for part in argv if part]
            row["started_at"] = now - uptime + ticks / ticks_per_second
            paths, errors = log_paths(pid, row["argv"])
            try:
                managed = managed_log(pid)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                managed = None
            if managed:
                paths = [managed["path"]]
                row["run_id"] = managed["run_id"]
            row["discovery_errors"] = errors
            for path in paths:
                if path not in logs:
                    if len(logs) >= MAX_FILES:
                        continue
                    try:
                        logs[path] = read_log(path)
                    except OSError as error:
                        logs[path] = {"error": type(error).__name__}
                row["paths"].append(path)
            if identity(pid) != ticks:
                row["error"] = "ProcessChanged"
                row["paths"] = []
        except OSError as error:
            row["error"] = type(error).__name__
        processes.append(row)
    return {
        "sampled_at": now,
        "utc_offset_seconds": datetime.datetime.now().astimezone().utcoffset().total_seconds(),
        "processes": processes,
        "logs": logs,
    }
