from __future__ import annotations

from typing import Protocol

from pydantic import Field

from gpuops.core.queue import ConcurrentUpdate, QueueStore
from gpuops.core.scheduler import Scheduler
from gpuops.models import JobRecord, JobStatus, LaunchInfo, StrictModel, utc_now
from gpuops.worker.supervisor import SupervisorError


class Launcher(Protocol):
    """Execution boundary implemented by the remote worker runtime."""

    # Non-SupervisorError exceptions from start() must prove no child was launched.
    def start(self, job: JobRecord) -> LaunchInfo: ...

    def reconcile(self, job: JobRecord) -> LaunchInfo | None: ...

    def poll(self, job: JobRecord) -> int | None: ...

    def cancel(self, job: JobRecord) -> bool: ...


class WorkerTickResult(StrictModel):
    leased: list[str] = Field(default_factory=list)
    started: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


class Worker:
    """One deterministic worker iteration, suitable for a loop or an external timer."""

    def __init__(self, store: QueueStore, scheduler: Scheduler, launcher: Launcher):
        self.store = store
        self.scheduler = scheduler
        self.launcher = launcher

    def tick(self) -> WorkerTickResult:
        result = WorkerTickResult()
        self._reconcile_cancellations(result)
        self._reconcile_starting(result)
        self._reconcile_running(result)
        decisions = self.scheduler.tick()
        result.leased.extend(decision.run_id for decision in decisions)
        for decision in decisions:
            self._start_leased(decision.run_id, decision.lease_token, result)
        return result

    def _reconcile_cancellations(self, result: WorkerTickResult) -> None:
        for record in self.store.list({JobStatus.CANCEL_REQUESTED}):
            if record.allocation is None or (
                record.allocation.server_id != self.scheduler.server_id
                or record.allocation.worker_id != self.scheduler.worker_id
            ):
                continue
            try:
                completed = self.launcher.cancel(record)
                if not completed:
                    continue
                exit_code, launch_error = self._execution_result(record)
                self.store.transition(
                    record.run_id,
                    JobStatus.CANCELLED,
                    expected_status=JobStatus.CANCEL_REQUESTED,
                    expected_revision=record.revision,
                    exit_code=exit_code,
                    launch_error=launch_error,
                    details={"cancel_completed": True},
                )
                result.cancelled.append(record.run_id)
            except ConcurrentUpdate:
                continue
            except Exception as error:
                result.errors[record.run_id] = f"cancel failed: {error}"

    def _reconcile_starting(self, result: WorkerTickResult) -> None:
        for record in self.store.list({JobStatus.STARTING}):
            if record.allocation is None or (
                record.allocation.server_id != self.scheduler.server_id
                or record.allocation.worker_id != self.scheduler.worker_id
            ):
                continue
            try:
                launch = self.launcher.reconcile(record)
            except Exception as error:
                result.errors[record.run_id] = f"launch reconciliation failed: {error}"
                continue
            if launch is None:
                try:
                    launch = self.launcher.start(record)
                except SupervisorError as error:
                    result.errors[record.run_id] = f"launch recovery pending: {error}"
                    continue
                except Exception as error:
                    result.errors[record.run_id] = f"launch recovery failed: {error}"
                    state_error = self._fail_starting(record.run_id, error)
                    if state_error is not None:
                        result.errors[record.run_id] += (
                            f"; failed to record launch failure: {state_error}"
                        )
                    continue
            try:
                self.store.transition(
                    record.run_id,
                    JobStatus.RUNNING,
                    expected_status=JobStatus.STARTING,
                    expected_revision=record.revision,
                    launch=launch,
                )
                result.started.append(record.run_id)
            except Exception as error:
                result.errors[record.run_id] = f"launch recovery failed: {error}"

    def _start_leased(self, run_id: str, lease_token: str, result: WorkerTickResult) -> None:
        process_started = False
        try:
            leased = self.store.get(run_id)
            if leased.allocation is None or leased.allocation.lease_token != lease_token:
                raise ConcurrentUpdate("lease token changed before launch")
            if leased.allocation.lease_expires_at <= self.scheduler.clock():
                raise ConcurrentUpdate("lease expired before launch")
            starting = self.store.transition(
                run_id,
                JobStatus.STARTING,
                expected_status=JobStatus.LEASED,
                expected_revision=leased.revision,
            )
            launch = self.launcher.start(starting)
            process_started = True
            self.store.transition(
                run_id,
                JobStatus.RUNNING,
                expected_status=JobStatus.STARTING,
                expected_revision=starting.revision,
                launch=launch,
            )
            result.started.append(run_id)
        except SupervisorError as error:
            result.errors[run_id] = str(error)
        except Exception as error:
            result.errors[run_id] = str(error)
            if process_started:
                # Keep the allocation reserved until a recovery mechanism can
                # reconcile the child without risking a duplicate launch.
                return
            try:
                current = self.store.get(run_id)
                if current.status in {JobStatus.LEASED, JobStatus.STARTING}:
                    self.store.transition(
                        run_id,
                        JobStatus.FAILED,
                        expected_status=current.status,
                        expected_revision=current.revision,
                        failure_reason=f"launch failed: {error}",
                        launch_error=f"{type(error).__name__}: {error}",
                    )
            except ConcurrentUpdate:
                pass
            except Exception as state_error:
                result.errors[run_id] += (
                    "; failed to record launch failure: "
                    f"{state_error.__class__.__name__}: {state_error}"
                )

    def _fail_starting(self, run_id: str, error: Exception) -> str | None:
        try:
            current = self.store.get(run_id)
            if current.status == JobStatus.STARTING:
                self.store.transition(
                    run_id,
                    JobStatus.FAILED,
                    expected_status=JobStatus.STARTING,
                    expected_revision=current.revision,
                    failure_reason=f"launch failed: {error}",
                    launch_error=f"{type(error).__name__}: {error}",
                )
        except ConcurrentUpdate:
            return None
        except Exception as state_error:
            return f"{state_error.__class__.__name__}: {state_error}"
        return None

    def _reconcile_running(self, result: WorkerTickResult) -> None:
        for record in self.store.list({JobStatus.RUNNING}):
            if record.allocation is None or (
                record.allocation.server_id != self.scheduler.server_id
                or record.allocation.worker_id != self.scheduler.worker_id
            ):
                continue
            try:
                exit_code = self.launcher.poll(record)
                if exit_code is None:
                    continue
                observed_exit_code, launch_error = self._execution_result(
                    record, fallback_exit_code=exit_code
                )
            except Exception as error:
                result.errors[record.run_id] = f"poll failed: {error}"
                continue
            target = JobStatus.SUCCEEDED if exit_code == 0 else JobStatus.FAILED
            try:
                self.store.transition(
                    record.run_id,
                    target,
                    expected_status=JobStatus.RUNNING,
                    expected_revision=record.revision,
                    failure_reason=(
                        None if exit_code == 0 else f"process exited with code {exit_code}"
                    ),
                    exit_code=observed_exit_code,
                    launch_error=launch_error,
                    now=utc_now(),
                    details={"exit_code": exit_code},
                )
            except ConcurrentUpdate:
                continue
            result.completed.append(record.run_id)

    def _execution_result(
        self, record: JobRecord, *, fallback_exit_code: int | None = None
    ) -> tuple[int | None, str | None]:
        # Durable launchers distinguish an observed result from a synthetic
        # failure code when the supervisor disappeared without writing one.
        read_result = getattr(self.launcher, "result", None)
        if read_result is None:
            if fallback_exit_code is None and record.launch is not None:
                fallback_exit_code = self.launcher.poll(record)
            return fallback_exit_code, None
        completed = read_result(record)
        if completed is None:
            return None, None
        return completed.exit_code, completed.launch_error
