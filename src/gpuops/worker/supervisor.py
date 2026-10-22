"""Durable, at-most-once process supervision for queued jobs.

The worker that requests a launch may disappear immediately after dispatch. A
small detached supervisor therefore owns the process, console log, and durable
receipt. The per-run flock is inherited by the supervisor and remains held for
the full lifetime of the job.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gpuops.core.storage import StorageBudget
from gpuops.worker import process_identity, supervisor_child
from gpuops.worker import supervisor_records as records
from gpuops.worker.process_identity import (
    process_identity_matches,
    read_process_identity,
    signal_owned_process_group,
    signal_process,
)
from gpuops.worker.supervisor_records import (
    DefinitiveLaunchFailure,
    IntentConflict,
    InvalidReceipt,
    LaunchInProgress,
    ProcessIdentity,
    SupervisorError,
    SupervisorPaths,
    SupervisorReceipt,
    SupervisorResult,
    UnsafeSignal,
)

__all__ = [
    "DefinitiveLaunchFailure",
    "DurableJobSupervisor",
    "DurableSupervisorLauncher",
    "IntentConflict",
    "InvalidReceipt",
    "LaunchInProgress",
    "ProcessIdentity",
    "SupervisorError",
    "SupervisorPaths",
    "SupervisorReceipt",
    "SupervisorResult",
    "UnsafeSignal",
    "process_identity_matches",
    "read_process_identity",
    "signal_owned_process_group",
    "signal_process",
]


class DurableJobSupervisor:
    """Launch and reconcile detached supervisors using durable run artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        storage: StorageBudget | None = None,
        receipt_timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.02,
        cancel_grace_seconds: float = 10.0,
        proc_root: str | Path = "/proc",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            receipt_timeout_seconds < 0
            or poll_interval_seconds <= 0
            or cancel_grace_seconds < 0
        ):
            raise ValueError("supervisor timeouts must be non-negative and polling positive")
        self.root = Path(root)
        self.storage = storage
        self.receipt_timeout_seconds = receipt_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.cancel_grace_seconds = cancel_grace_seconds
        self.proc_root = Path(proc_root)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._owned_supervisor_processes: dict[str, subprocess.Popen[bytes]] = {}

    def launch(
        self,
        run_id: str,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        environment: Mapping[str, str] | None = None,
        console_max_bytes: int = 4 * 1024 * 1024,
        console_backups: int = 2,
    ) -> SupervisorReceipt:
        paths = self._paths(run_id)
        reservation_path = (
            self.storage.reservation_directory / f"{run_id}.reserve"
            if self.storage is not None
            else None
        )
        requested = records.requested_intent(
            run_id,
            argv,
            cwd,
            environment,
            paths.console,
            console_max_bytes,
            console_backups,
            reservation_path,
        )

        existing_intent = records.read_optional_json(paths.intent)
        if existing_intent is not None:
            records.ensure_same_intent(existing_intent, requested)
            receipt = self.reconcile(run_id)
            if receipt is not None:
                if receipt.status in {"running", "finished"}:
                    return receipt
                return self._wait_for_receipt(paths, existing_intent, self.receipt_timeout_seconds)
            if paths.result.exists():
                raise InvalidReceipt("run has a result but no matching receipt")

        lock_fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(lock_fd, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self._wait_for_receipt(paths, requested, self.receipt_timeout_seconds)

            # Recheck after acquiring the lock because another supervisor may
            # have completed between the optimistic read and flock acquisition.
            existing_intent = records.read_optional_json(paths.intent)
            if existing_intent is None:
                if self.storage is not None:
                    self.storage.reserve_log_capacity(
                        run_id,
                        console_max_bytes * (console_backups + 1),
                    )
                intent = {
                    **requested,
                    "schema": records.SCHEMA_VERSION,
                    "launch_token": uuid.uuid4().hex,
                    "created_at": records.utc_now(),
                }
                try:
                    records.atomic_write_json(paths.intent, intent)
                except Exception:
                    if paths.intent.exists():
                        self._discard_unstarted_launch(
                            paths,
                            str(intent["launch_token"]),
                        )
                    else:
                        self.release_log_reservation(run_id)
                    raise
            else:
                records.ensure_same_intent(existing_intent, requested)
                intent = existing_intent

            receipt = self.reconcile(run_id)
            if receipt is not None:
                if receipt.status in {"running", "finished"}:
                    return receipt
                return self._wait_for_receipt(paths, intent, self.receipt_timeout_seconds)

            command = (
                sys.executable,
                str(Path(__file__).resolve()),
                "_supervise",
                "--intent",
                str(paths.intent),
                "--lock-fd",
                str(lock_fd),
            )
            try:
                supervisor_process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(lock_fd,),
                    start_new_session=True,
                )
                self._owned_supervisor_processes[run_id] = supervisor_process
            except (OSError, ValueError) as error:
                self._discard_unstarted_launch(paths, str(intent["launch_token"]))
                raise DefinitiveLaunchFailure(f"could not start supervisor: {error}") from error
            # The inherited descriptor keeps the flock alive after this method
            # closes its local copy, including if the worker itself exits.
            return self._wait_for_receipt(paths, intent, self.receipt_timeout_seconds)
        finally:
            os.close(lock_fd)

    def reconcile(self, run_id: str) -> SupervisorReceipt | None:
        self._reap_owned_supervisor(run_id)
        paths = self._paths(run_id)
        intent = records.read_optional_json(paths.intent)
        receipt_data = records.read_optional_json(paths.receipt)
        if receipt_data is None:
            return None
        if intent is None or receipt_data.get("launch_token") != intent.get("launch_token"):
            raise InvalidReceipt("receipt launch token does not match intent")
        receipt = SupervisorReceipt.from_dict(receipt_data)
        if receipt.run_id != run_id:
            raise InvalidReceipt("receipt run id does not match its path")
        return receipt

    def poll(self, run_id: str) -> SupervisorResult | None:
        self._reap_owned_supervisor(run_id)
        paths = self._paths(run_id)
        data = records.read_optional_json(paths.result)
        if data is None:
            return None
        result = SupervisorResult.from_dict(data)
        intent = records.read_optional_json(paths.intent)
        if (
            intent is None
            or result.run_id != run_id
            or result.launch_token != intent.get("launch_token")
        ):
            raise SupervisorError("result does not match launch intent")
        return result

    def release_log_reservation(self, run_id: str) -> None:
        if self.storage is not None:
            self.storage.release_log_capacity(run_id)

    def _discard_unstarted_launch(self, paths: SupervisorPaths, launch_token: str) -> None:
        """Remove artifacts only while the caller owns the run lock pre-dispatch."""

        if paths.receipt.exists() or paths.result.exists():
            raise LaunchInProgress("launch artifacts appeared before cleanup")
        intent = records.read_optional_json(paths.intent)
        if intent is None or intent.get("launch_token") != launch_token:
            raise LaunchInProgress("launch intent changed before cleanup")
        paths.cancellation.unlink(missing_ok=True)
        paths.intent.unlink()
        records.fsync_directory(paths.directory)
        self.release_log_reservation(paths.directory.name)

    def wait(self, run_id: str, *, timeout_seconds: float = 30.0) -> SupervisorResult:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        paths = self._paths(run_id)
        deadline = time.monotonic() + timeout_seconds
        while True:
            result = self.poll(run_id)
            if (
                result is not None
                and not records.lock_is_held(paths.lock)
                and self._reap_owned_supervisor(run_id)
            ):
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run {run_id!r} did not finish in time")
            time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))

    def signal(self, run_id: str, sig: int = signal.SIGTERM) -> None:
        receipt = self.reconcile(run_id)
        if receipt is None:
            raise InvalidReceipt("cannot signal a run without a receipt")
        signal_process(receipt.supervisor, sig, proc_root=self.proc_root)

    def request_cancel(self, run_id: str, *, expected_launch_token: str | None = None) -> bool:
        """Terminate only verified process groups and report when all have exited."""

        result = self.poll(run_id)
        if (
            result is not None
            and expected_launch_token is not None
            and result.launch_token != expected_launch_token
        ):
            raise InvalidReceipt("job launch token does not match supervisor result")

        paths = self._paths(run_id)
        intent = records.read_optional_json(paths.intent)
        receipt = self.reconcile(run_id)
        if receipt is None:
            if result is not None:
                raise InvalidReceipt("completed run has no matching supervisor receipt")
            if intent is None:
                completed = self._confirm_no_launch_artifacts(paths)
                if completed:
                    self.release_log_reservation(run_id)
                return completed
            if (
                expected_launch_token is not None
                and intent.get("launch_token") != expected_launch_token
            ):
                raise InvalidReceipt("job launch token does not match supervisor intent")
            return False
        if expected_launch_token is not None and receipt.launch_token != expected_launch_token:
            raise InvalidReceipt("job launch token does not match supervisor receipt")
        if result is not None and result.launch_token != receipt.launch_token:
            raise InvalidReceipt("supervisor result does not match receipt")
        if receipt.status not in {"accepted", "running", "finished"}:
            raise InvalidReceipt(f"receipt has unsupported status {receipt.status!r}")

        now = self.clock()
        if now.tzinfo is None:
            raise SupervisorError("cancellation clock must return a timezone-aware datetime")
        cancellation, created = self._cancellation_record(paths, receipt, now)
        supervisor_state = process_identity.process_group_state(receipt.supervisor, self.proc_root)
        child_state = (
            process_identity.process_group_state(receipt.child, self.proc_root)
            if receipt.child is not None
            else "gone"
        )
        if supervisor_state == "gone" and child_state == "gone":
            self.release_log_reservation(run_id)
            return True

        kill_after = records.parse_aware_datetime(cancellation.get("kill_after"), "kill_after")
        phase = cancellation.get("phase")
        if now >= kill_after:
            if phase != "kill_sent":
                killing = {**cancellation, "phase": "kill_pending"}
                records.atomic_write_json(paths.cancellation, killing)
                if not self._signal_verified_groups(
                    receipt,
                    supervisor_state,
                    child_state,
                    signal.SIGKILL,
                    include_child=True,
                ):
                    return False
                cancellation = {
                    **killing,
                    "phase": "kill_sent",
                    "kill_sent_at": now.isoformat(),
                }
                records.atomic_write_json(paths.cancellation, cancellation)
            return False

        if created or phase == "term_pending":
            if not self._signal_verified_groups(
                receipt,
                supervisor_state,
                child_state,
                signal.SIGTERM,
                include_child=False,
            ):
                return False
            cancellation = {
                **cancellation,
                "phase": "term_sent",
                "term_sent_at": now.isoformat(),
            }
            records.atomic_write_json(paths.cancellation, cancellation)
        return False

    def _cancellation_record(
        self,
        paths: SupervisorPaths,
        receipt: SupervisorReceipt,
        now: datetime,
    ) -> tuple[dict[str, Any], bool]:
        existing = records.read_optional_json(paths.cancellation)
        if existing is not None:
            if (
                existing.get("schema") != records.SCHEMA_VERSION
                or existing.get("run_id") != receipt.run_id
                or existing.get("launch_token") != receipt.launch_token
                or existing.get("phase")
                not in {"term_pending", "term_sent", "kill_pending", "kill_sent"}
            ):
                raise SupervisorError("cancellation record does not match supervisor receipt")
            records.parse_aware_datetime(existing.get("requested_at"), "requested_at")
            records.parse_aware_datetime(existing.get("kill_after"), "kill_after")
            return existing, False

        cancellation = {
            "schema": records.SCHEMA_VERSION,
            "run_id": receipt.run_id,
            "launch_token": receipt.launch_token,
            "requested_at": now.isoformat(),
            "kill_after": (now + timedelta(seconds=self.cancel_grace_seconds)).isoformat(),
            "phase": "term_pending",
        }
        records.atomic_write_json(paths.cancellation, cancellation)
        return cancellation, True

    def _signal_verified_groups(
        self,
        receipt: SupervisorReceipt,
        supervisor_state: str,
        child_state: str,
        sig: int,
        *,
        include_child: bool,
    ) -> bool:
        identities: list[ProcessIdentity] = []
        if include_child:
            # Killing an accepted supervisor before it publishes a child can
            # orphan a child created concurrently in its independent session.
            if receipt.child is None and receipt.status != "finished":
                return False
            if receipt.child is not None and child_state == "unknown":
                return False
            if receipt.child is not None and child_state in {"matched", "members"}:
                identities.append(receipt.child)
            if supervisor_state in {"matched", "members"}:
                identities.append(receipt.supervisor)
        elif supervisor_state == "matched":
            identities.append(receipt.supervisor)
            # A live supervisor forwards TERM while its child leader is alive.
            # Once that leader is a zombie or gone, only a direct group signal
            # can reach surviving descendants.
            if receipt.child is not None and child_state == "members":
                identities.append(receipt.child)
        else:
            if receipt.child is not None and child_state in {"matched", "members"}:
                identities.append(receipt.child)
            if supervisor_state == "members":
                identities.append(receipt.supervisor)
        if not identities:
            return False
        for identity in identities:
            try:
                signal_owned_process_group(identity, sig, proc_root=self.proc_root)
            except (ProcessLookupError, UnsafeSignal):
                return False
        return True

    def process_groups_gone(self, receipt: SupervisorReceipt) -> bool:
        self._reap_owned_supervisor(receipt.run_id)
        supervisor_state = process_identity.process_group_state(receipt.supervisor, self.proc_root)
        child_state = (
            process_identity.process_group_state(receipt.child, self.proc_root)
            if receipt.child is not None
            else "gone"
        )
        return (
            supervisor_state == "gone"
            and child_state == "gone"
            and self._reap_owned_supervisor(receipt.run_id)
        )

    def _reap_owned_supervisor(self, run_id: str) -> bool:
        """Reap an owned supervisor and report whether no owned handle remains."""

        process = self._owned_supervisor_processes.get(run_id)
        if process is None:
            return True
        if process.poll() is None:
            return False
        if self._owned_supervisor_processes.get(run_id) is process:
            del self._owned_supervisor_processes[run_id]
        return run_id not in self._owned_supervisor_processes

    @staticmethod
    def _confirm_no_launch_artifacts(paths: SupervisorPaths) -> bool:
        """Prove no launch owns the run lock before completing cancellation."""

        lock_fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(lock_fd, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return (
                records.read_optional_json(paths.intent) is None
                and records.read_optional_json(paths.receipt) is None
                and records.read_optional_json(paths.result) is None
            )
        finally:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _paths(self, run_id: str) -> SupervisorPaths:
        console_root = self.storage.logs_directory if self.storage is not None else None
        return SupervisorPaths.create(self.root, run_id, console_root=console_root)

    def _wait_for_receipt(
        self,
        paths: SupervisorPaths,
        intent: Mapping[str, Any],
        timeout_seconds: float,
    ) -> SupervisorReceipt:
        deadline = time.monotonic() + timeout_seconds
        expected_token = intent.get("launch_token")
        while True:
            self._reap_owned_supervisor(paths.directory.name)
            if expected_token is None:
                persisted_intent = records.read_optional_json(paths.intent)
                if persisted_intent is not None:
                    records.ensure_same_intent(persisted_intent, intent)
                    expected_token = persisted_intent.get("launch_token")
            data = records.read_optional_json(paths.receipt)
            if data is not None:
                if expected_token is None or data.get("launch_token") != expected_token:
                    raise InvalidReceipt("receipt launch token does not match intent")
                receipt = SupervisorReceipt.from_dict(data)
                if receipt.status in {"running", "finished"}:
                    return receipt
            if time.monotonic() >= deadline:
                raise LaunchInProgress(
                    f"launch for {intent['run_id']!r} has no final receipt yet; reconcile later"
                )
            time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))


class DurableSupervisorLauncher:
    """Adapter implementing the Launcher protocol used by :class:`Worker`."""

    def __init__(self, supervisor: DurableJobSupervisor) -> None:
        self.supervisor = supervisor

    def start(self, job: Any) -> Any:
        environment = dict(job.spec.runtime.environment)
        if job.allocation is not None:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                job.allocation.gpu_uuids.get(index, str(index))
                for index in job.allocation.gpu_indices
            )
        argv = list(job.spec.command)
        if job.spec.runtime.conda_environment:
            argv = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                job.spec.runtime.conda_environment,
                *argv,
            ]
        receipt = self.supervisor.launch(
            job.run_id,
            argv,
            cwd=job.spec.working_directory,
            environment=environment,
        )
        return self._launch_info(receipt)

    def reconcile(self, job: Any) -> Any | None:
        receipt = self.supervisor.reconcile(job.run_id)
        if receipt is None:
            return None
        return self._launch_info(receipt)

    def poll(self, job: Any) -> int | None:
        result, receipt = self._result_and_receipt(job)
        if result is not None:
            if not self.supervisor.process_groups_gone(receipt):
                return None
            self.supervisor.release_log_reservation(job.run_id)
        if result is None:
            if receipt is None:
                raise InvalidReceipt("running job has no supervisor receipt")
            if job.launch is not None and job.launch.external_id != receipt.launch_token:
                raise InvalidReceipt("job launch token does not match supervisor receipt")
            if receipt.status == "finished":
                raise SupervisorError("supervisor finished without a durable result")
            supervisor_state = process_identity.process_group_state(
                receipt.supervisor,
                self.supervisor.proc_root,
            )
            child_state = (
                process_identity.process_group_state(receipt.child, self.supervisor.proc_root)
                if receipt.child is not None
                else "gone"
            )
            if supervisor_state != "gone" or child_state != "gone":
                return None
            self.supervisor.release_log_reservation(job.run_id)
            return 125
        return result.exit_code

    def result(self, job: Any) -> SupervisorResult | None:
        """Return only an observed, identity-matched result, including after cancel."""

        result, _receipt = self._result_and_receipt(job)
        return result

    def _result_and_receipt(
        self, job: Any
    ) -> tuple[SupervisorResult | None, SupervisorReceipt | None]:
        result = self.supervisor.poll(job.run_id)
        receipt = self.supervisor.reconcile(job.run_id)
        if result is None:
            return None, receipt
        if receipt is None or receipt.launch_token != result.launch_token:
            raise InvalidReceipt("supervisor result has no matching receipt")
        if job.launch is not None and job.launch.external_id != result.launch_token:
            raise InvalidReceipt("job launch token does not match supervisor result")
        return result, receipt

    def cancel(self, job: Any) -> bool:
        expected_token = job.launch.external_id if job.launch is not None else None
        return self.supervisor.request_cancel(
            job.run_id,
            expected_launch_token=expected_token,
        )

    @staticmethod
    def _launch_info(receipt: SupervisorReceipt) -> Any:
        from gpuops.models import LaunchInfo

        process = receipt.child or receipt.supervisor
        started_at = receipt.child_started_at or receipt.accepted_at
        return LaunchInfo(
            pid=process.pid,
            started_at=datetime.fromisoformat(started_at),
            external_id=receipt.launch_token,
        )



def _build_internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["_supervise"])
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--lock-fd", required=True, type=int)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_internal_parser().parse_args(argv)
    return supervisor_child.run_supervisor(arguments.intent, arguments.lock_fd)


if __name__ == "__main__":
    raise SystemExit(_main())
