"""Detached child lifecycle and bounded console capture for job supervision."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any

from gpuops.worker import process_identity
from gpuops.worker import supervisor_records as records


def run_supervisor(intent_path: Path, lock_fd: int) -> int:
    """Run the detached supervisor process for one persisted launch intent."""

    intent = records.read_optional_json(intent_path)
    if intent is None:
        return 125
    run_directory = intent_path.parent
    receipt_path = run_directory / "receipt.json"
    result_path = run_directory / "result.json"
    launch_token = str(intent["launch_token"])
    run_id = str(intent["run_id"])
    accepted_at = records.utc_now()
    supervisor_identity = process_identity.current_process_identity()
    receipt = records.SupervisorReceipt(
        run_id=run_id,
        launch_token=launch_token,
        status="accepted",
        accepted_at=accepted_at,
        supervisor=supervisor_identity,
    )
    child: subprocess.Popen[bytes] | None = None
    pending_signal: int | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        pending_signal = signum
        if child is not None and child.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records.atomic_write_json(receipt_path, receipt.as_json_dict())
        started_at: str | None = None
        launch_error: str | None = None
        try:
            environment = os.environ.copy()
            environment.update(intent["environment"])
            child = subprocess.Popen(
                intent["argv"],
                cwd=intent["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
            started_at = records.utc_now()
            # start_new_session guarantees the child's PGID equals its PID,
            # even when a very short command exits before procfs can be read.
            child_identity = process_identity.identity_with_fallback(
                child.pid,
                fallback_pgid=child.pid,
            )
            receipt = records.SupervisorReceipt(
                run_id=run_id,
                launch_token=launch_token,
                status="running",
                accepted_at=accepted_at,
                supervisor=supervisor_identity,
                child=child_identity,
                child_started_at=started_at,
            )
            records.atomic_write_json(receipt_path, receipt.as_json_dict())
            if pending_signal is not None:
                os.killpg(child.pid, pending_signal)
            if child.stdout is None:
                raise records.SupervisorError("child stdout pipe was not created")
            pump_console(
                child.stdout,
                Path(intent["console_path"]),
                int(intent["console_max_bytes"]),
                int(intent["console_backups"]),
            )
            exit_code = child.wait()
        except (OSError, ValueError, TypeError, records.SupervisorError) as error:
            launch_error = f"{type(error).__name__}: {error}"
            exit_code = 127
            if child is not None and child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()

        result = records.SupervisorResult(
            run_id=run_id,
            launch_token=launch_token,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=records.utc_now(),
            launch_error=launch_error,
        )
        finished_receipt = records.SupervisorReceipt(
            run_id=receipt.run_id,
            launch_token=receipt.launch_token,
            status="finished",
            accepted_at=receipt.accepted_at,
            supervisor=receipt.supervisor,
            child=receipt.child,
            child_started_at=receipt.child_started_at,
        )
        records.atomic_write_json(receipt_path, finished_receipt.as_json_dict())
        reservation_path = intent.get("reservation_path")
        if reservation_path:
            Path(reservation_path).unlink(missing_ok=True)
        # The result is the final completion marker. Readers that observe it
        # can rely on the finished receipt and quota release already existing.
        records.atomic_write_json(result_path, result.as_json_dict())
        return 0
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def pump_console(stream: Any, path: Path, max_bytes: int, backups: int) -> None:
    """Copy a child pipe promptly into bounded, rotated console files."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = path.open("ab", buffering=0)
    os.chmod(path, 0o600)
    size = path.stat().st_size
    read_chunk = getattr(stream, "read1", stream.read)
    try:
        while True:
            # BufferedReader.read(size) waits for the requested size or EOF on
            # a pipe. read1() returns after one underlying read, making small
            # flushed writes visible while the child is still running.
            chunk = read_chunk(64 * 1024)
            if not chunk:
                break
            offset = 0
            while offset < len(chunk):
                if size >= max_bytes:
                    handle.close()
                    _rotate_console(path, backups)
                    handle = path.open("ab", buffering=0)
                    os.chmod(path, 0o600)
                    size = 0
                length = min(max_bytes - size, len(chunk) - offset)
                handle.write(chunk[offset : offset + length])
                size += length
                offset += length
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()


def _rotate_console(path: Path, backups: int) -> None:
    if backups == 0:
        path.unlink(missing_ok=True)
        return
    oldest = path.with_name(f"{path.name}.{backups}")
    oldest.unlink(missing_ok=True)
    for index in range(backups - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            os.replace(source, path.with_name(f"{path.name}.{index + 1}"))
    if path.exists():
        os.replace(path, path.with_name(f"{path.name}.1"))
