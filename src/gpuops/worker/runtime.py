"""Safe host-local process runtime for one GPUOps worker."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from gpuops.models import StrictModel, utc_now
from gpuops.worker.process_identity import parse_proc_start_ticks, read_procfs_record


class WorkerAlreadyRunning(RuntimeError):
    """The host-local worker lock is already owned."""


@dataclass(frozen=True)
class ProcessIdentity:
    """Linux process identity resistant to PID reuse and host reboot."""

    pid: int
    start_time_ticks: int
    boot_id: str

    def __post_init__(self) -> None:
        if self.pid <= 0 or self.start_time_ticks < 0 or not self.boot_id:
            raise ValueError("invalid process identity")


@dataclass(frozen=True)
class RuntimePaths:
    state_directory: Path
    lock_path: Path
    pid_path: Path
    heartbeat_path: Path

    @classmethod
    def from_state_directory(cls, state_directory: Path) -> RuntimePaths:
        root = state_directory.expanduser()
        return cls(
            state_directory=root,
            lock_path=root / "worker.lock",
            pid_path=root / "worker.pid.json",
            heartbeat_path=root / "worker-health.json",
        )


class WorkerHeartbeat(StrictModel):
    server_id: str
    worker_id: str
    pid: int
    status: str
    started_at: datetime
    heartbeat_at: datetime
    iteration: int
    tick_failures: int
    last_tick_started_at: datetime | None = None
    last_tick_finished_at: datetime | None = None
    last_tick_error: str | None = None


class WorkerProcessState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True)
class WorkerProcessStatus:
    state: WorkerProcessState
    identity: ProcessIdentity | None
    heartbeat: dict[str, Any] | None


class ProcessFactory(Protocol):
    def __call__(self, args: Sequence[str], **kwargs: Any) -> Any: ...


class RuntimeSampler(Protocol):
    def tick(self) -> None: ...

    def close(self) -> None: ...


class WorkerFileLock:
    """Non-blocking advisory lock retained for the runtime's full lifetime."""

    def __init__(self, path: Path):
        self.path = path
        self._file_descriptor: int | None = None

    def acquire(self) -> None:
        if self._file_descriptor is not None:
            raise RuntimeError("worker lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise WorkerAlreadyRunning(f"worker lock is held: {self.path}") from error
        self._file_descriptor = descriptor

    def release(self) -> None:
        if self._file_descriptor is None:
            return
        descriptor = self._file_descriptor
        self._file_descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> WorkerFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


class WorkerRuntime:
    """Own the worker loop and its metadata, but never experiment processes."""

    def __init__(
        self,
        *,
        server_id: str,
        worker_id: str,
        paths: RuntimePaths,
        tick: Callable[[], Any],
        interval_seconds: float = 5.0,
        clock: Callable[[], datetime] = utc_now,
        sleeper: Callable[[float], None] = time.sleep,
        getpid: Callable[[], int] = os.getpid,
        identity_reader: Callable[[int], ProcessIdentity | None] | None = None,
        stop_event: threading.Event | None = None,
        sampler_factory: Callable[[], RuntimeSampler] | None = None,
    ):
        if not server_id.strip() or not worker_id.strip():
            raise ValueError("server_id and worker_id cannot be empty")
        if interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative")
        self.server_id = server_id
        self.worker_id = worker_id
        self.paths = paths
        self.tick = tick
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.getpid = getpid
        self.identity_reader = identity_reader or read_process_identity
        self.stop_event = stop_event or threading.Event()
        self.sampler_factory = sampler_factory

    def stop(self) -> None:
        """Request loop shutdown without signalling or inspecting any experiment."""

        self.stop_event.set()

    def run_foreground(
        self,
        *,
        max_iterations: int | None = None,
        install_signal_handlers: bool = True,
    ) -> int:
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations cannot be negative")
        self._prepare_state_directory()
        lock = WorkerFileLock(self.paths.lock_path)
        with lock:
            pid = self.getpid()
            identity = self.identity_reader(pid)
            if identity is None:
                raise RuntimeError(f"cannot establish identity for worker PID {pid}")
            write_process_identity(self.paths.pid_path, identity)
            started_at = self.clock()
            heartbeat = WorkerHeartbeat(
                server_id=self.server_id,
                worker_id=self.worker_id,
                pid=pid,
                status="starting",
                started_at=started_at,
                heartbeat_at=started_at,
                iteration=0,
                tick_failures=0,
            )
            write_heartbeat(self.paths.heartbeat_path, heartbeat)
            sampler: RuntimeSampler | None = None
            previous_handlers: dict[int, Any] = {}
            try:
                if self.sampler_factory is not None:
                    sampler = self.sampler_factory()
                if install_signal_handlers:
                    previous_handlers = self._install_signal_handlers()
                while not self.stop_event.is_set() and (
                    max_iterations is None or heartbeat.iteration < max_iterations
                ):
                    tick_started_at = self.clock()
                    tick_error: str | None = None
                    if sampler is not None:
                        try:
                            sampler.tick()
                        except Exception as error:
                            tick_error = self._error_text(error)
                    try:
                        tick_result = self.tick()
                        worker_error = self._tick_result_error(tick_result)
                        if worker_error:
                            tick_error = (
                                f"{tick_error}; {worker_error}" if tick_error else worker_error
                            )
                    except Exception as error:
                        worker_error = self._error_text(error)
                        tick_error = (
                            f"{tick_error}; {worker_error}" if tick_error else worker_error
                        )
                    tick_finished_at = self.clock()
                    heartbeat = heartbeat.model_copy(
                        update={
                            "status": "degraded" if tick_error else "running",
                            "heartbeat_at": tick_finished_at,
                            "iteration": heartbeat.iteration + 1,
                            "tick_failures": heartbeat.tick_failures + (1 if tick_error else 0),
                            "last_tick_started_at": tick_started_at,
                            "last_tick_finished_at": tick_finished_at,
                            "last_tick_error": tick_error,
                        }
                    )
                    write_heartbeat(self.paths.heartbeat_path, heartbeat)
                    if self.stop_event.is_set() or (
                        max_iterations is not None and heartbeat.iteration >= max_iterations
                    ):
                        break
                    self.sleeper(self.interval_seconds)
            finally:
                try:
                    if sampler is not None:
                        sampler.close()
                finally:
                    self._restore_signal_handlers(previous_handlers)
                    stopped_at = self.clock()
                    stopped = heartbeat.model_copy(
                        update={"status": "stopped", "heartbeat_at": stopped_at}
                    )
                    try:
                        write_heartbeat(self.paths.heartbeat_path, stopped)
                    finally:
                        remove_process_identity(self.paths.pid_path, identity)
            return heartbeat.iteration

    def _prepare_state_directory(self) -> None:
        self.paths.state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.paths.state_directory, 0o700)

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}

        def request_stop(signum: int, frame: FrameType | None) -> None:
            del signum, frame
            self.stop()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    @staticmethod
    def _error_text(error: Exception) -> str:
        message = str(error).strip() or error.__class__.__name__
        return f"{error.__class__.__name__}: {message}".replace("\n", " ")[:500]

    @staticmethod
    def _tick_result_error(result: Any) -> str | None:
        if result is None:
            return None
        errors = getattr(result, "errors", None)
        if not isinstance(errors, Mapping) or not errors:
            return None
        details = "; ".join(
            f"{run_id}: {str(message).strip()}" for run_id, message in sorted(errors.items())
        )
        return f"job errors: {details}".replace("\n", " ")[:500]


def read_process_identity(pid: int, proc_root: Path = Path("/proc")) -> ProcessIdentity | None:
    """Read a Linux process identity without relying on the PID alone."""

    record = read_procfs_record(pid, proc_root)
    if record is None:
        return None
    try:
        return ProcessIdentity(
            pid=record.pid,
            start_time_ticks=parse_proc_start_ticks(record.stat_fields),
            boot_id=record.boot_id,
        )
    except (ValueError, IndexError):
        return None


def write_process_identity(path: Path, identity: ProcessIdentity) -> None:
    _atomic_write_json(path, asdict(identity))


def read_stored_process_identity(path: Path) -> ProcessIdentity | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid worker PID file: {path}") from error
    try:
        return ProcessIdentity(
            pid=int(raw["pid"]),
            start_time_ticks=int(raw["start_time_ticks"]),
            boot_id=str(raw["boot_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid worker PID file: {path}") from error


def remove_process_identity(path: Path, expected: ProcessIdentity) -> None:
    try:
        current = read_stored_process_identity(path)
    except ValueError:
        return
    if current != expected:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def inspect_worker(
    paths: RuntimePaths,
    *,
    identity_reader: Callable[[int], ProcessIdentity | None] | None = None,
) -> WorkerProcessStatus:
    reader = identity_reader or read_process_identity
    heartbeat = _read_json_object(paths.heartbeat_path)
    try:
        stored = read_stored_process_identity(paths.pid_path)
    except ValueError:
        return WorkerProcessStatus(WorkerProcessState.INVALID, None, heartbeat)
    if stored is None:
        return WorkerProcessStatus(WorkerProcessState.STOPPED, None, heartbeat)
    current = reader(stored.pid)
    if current != stored:
        return WorkerProcessStatus(WorkerProcessState.STALE, stored, heartbeat)
    return WorkerProcessStatus(WorkerProcessState.RUNNING, stored, heartbeat)


def request_worker_stop(
    paths: RuntimePaths,
    *,
    identity_reader: Callable[[int], ProcessIdentity | None] | None = None,
    signal_sender: Callable[[int, int], None] = os.kill,
) -> bool:
    """Signal only a verified worker PID; never its process group or experiments."""

    status = inspect_worker(paths, identity_reader=identity_reader)
    if status.state != WorkerProcessState.RUNNING or status.identity is None:
        return False
    signal_sender(status.identity.pid, signal.SIGTERM)
    return True


def write_heartbeat(path: Path, heartbeat: WorkerHeartbeat) -> None:
    _atomic_write_json(path, heartbeat.model_dump(mode="json"))


def spawn_detached_worker(
    argv: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    process_factory: ProcessFactory = subprocess.Popen,
) -> int:
    """Start one detached worker process without a shell, privilege, or service manager."""

    arguments = _validated_argv(argv)
    if Path(arguments[0]).name == "sudo":
        raise ValueError("detached workers cannot be started through sudo")
    if cwd is not None and "\x00" in str(cwd):
        raise ValueError("cwd cannot contain NUL bytes")
    environment = None if env is None else _validated_environment(env)

    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stdout_descriptor = os.open(stdout_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    os.fchmod(stdout_descriptor, 0o600)
    stdout_handle = os.fdopen(stdout_descriptor, "ab", buffering=0)
    stderr_handle = None
    try:
        if stderr_path is not None:
            stderr_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            stderr_descriptor = os.open(stderr_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            os.fchmod(stderr_descriptor, 0o600)
            stderr_handle = os.fdopen(stderr_descriptor, "ab", buffering=0)
        process = process_factory(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle if stderr_handle is not None else subprocess.STDOUT,
            cwd=str(cwd) if cwd is not None else None,
            env=environment,
            start_new_session=True,
            close_fds=True,
            shell=False,
        )
        pid = int(process.pid)
        if pid <= 0:
            raise RuntimeError("detached worker did not return a valid PID")
        return pid
    finally:
        stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _validated_argv(argv: Sequence[str]) -> list[str]:
    if not argv:
        raise ValueError("worker argv cannot be empty")
    arguments = list(argv)
    if any(
        not isinstance(argument, str) or not argument or "\x00" in argument
        for argument in arguments
    ):
        raise ValueError("worker argv must contain non-empty strings without NUL bytes")
    return arguments


def _validated_environment(env: Mapping[str, str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("environment keys and values must be strings")
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ValueError("environment contains an invalid key or value")
        environment[key] = value
    return environment
