from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError, create_model

from gpuops.control import ControlClient, RemoteTarget, SubmissionJournal
from gpuops.core.queue import ConcurrentUpdate, QueueStore
from gpuops.models import JobRecord, JobSpec, JobStatus, LaunchInfo, StrictModel
from gpuops.ssh import SSHResult
from gpuops.transport import SshWorkerTransport
from gpuops.worker.service import Worker
from gpuops.worker.supervisor import DurableJobSupervisor, DurableSupervisorLauncher
from gpuops.worker.supervisor_records import (
    ProcessIdentity,
    SupervisorPaths,
    SupervisorReceipt,
    SupervisorResult,
    atomic_write_json,
)

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
RESULT_FIELDS = {"exit_code", "launch_error"}
# Preserve the pre-result field set and its extra="forbid" validation.
LegacyJobRecord = create_model(
    "LegacyJobRecord",
    __base__=StrictModel,
    **{
        name: (field.annotation, field)
        for name, field in JobRecord.model_fields.items()
        if name not in RESULT_FIELDS
    },
)


def spec() -> JobSpec:
    return JobSpec(
        id="results", project="tests", command=["python", "train.py"],
        working_directory="/work", resources={"host": "gpu01"},
    )


def starting_record(store: QueueStore) -> JobRecord:
    queued = store.enqueue(spec(), run_id="result-run", now=NOW)
    leased = store.lease(
        queued.run_id, server_id="gpu01", gpu_indices=[0], worker_id="worker",
        ttl=timedelta(minutes=5), expected_revision=0, now=NOW,
    )
    return store.transition(leased.run_id, JobStatus.STARTING, now=NOW)


def idle_scheduler():
    return SimpleNamespace(server_id="gpu01", worker_id="worker", tick=lambda: [])


def finished_supervisor(tmp_path, record, exit_code, launch_error=None):
    paths = SupervisorPaths.create(tmp_path / "runs", record.run_id)
    receipt = SupervisorReceipt(
        run_id=record.run_id, launch_token="token", status="finished",
        accepted_at=NOW.isoformat(),
        supervisor=ProcessIdentity(pid=43210, pgid=43210, start_ticks=1, boot_id="boot"),
        child=ProcessIdentity(pid=43211, pgid=43211, start_ticks=2, boot_id="boot"),
        child_started_at=NOW.isoformat(),
    )
    atomic_write_json(paths.intent, {"run_id": record.run_id, "launch_token": "token"})
    atomic_write_json(paths.receipt, receipt.as_json_dict())
    if exit_code is not None:
        result = SupervisorResult(
            run_id=record.run_id, launch_token="token", exit_code=exit_code,
            started_at=NOW.isoformat(), finished_at=(NOW + timedelta(seconds=1)).isoformat(),
            launch_error=launch_error,
        )
        atomic_write_json(paths.result, result.as_json_dict())
    else:
        # Missing results are possible after a supervisor exits unexpectedly.
        atomic_write_json(paths.receipt, {**receipt.as_json_dict(), "status": "running"})
    return DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "missing-proc")


def test_active_queue_files_remain_readable_by_strict_legacy_workers(tmp_path) -> None:
    store = QueueStore(tmp_path)
    record = store.enqueue(spec(), run_id="active", now=NOW)

    def assert_legacy_readable():
        payload = json.loads((store.jobs_directory / "active.json").read_text())
        assert RESULT_FIELDS.isdisjoint(payload)
        assert LegacyJobRecord.model_validate(payload).run_id == "active"

    assert_legacy_readable()
    record = store.lease(
        record.run_id, server_id="gpu01", gpu_indices=[0], worker_id="worker",
        ttl=timedelta(minutes=5), expected_revision=0, now=NOW,
    )
    assert_legacy_readable()
    record = store.transition(record.run_id, JobStatus.STARTING, now=NOW)
    assert_legacy_readable()
    record = store.transition(
        record.run_id, JobStatus.RUNNING, launch=LaunchInfo(pid=43211, started_at=NOW), now=NOW,
    )
    assert_legacy_readable()
    store.request_cancel(record.run_id, now=NOW)
    assert_legacy_readable()


@pytest.mark.parametrize(("status", "exit_code", "launch_error", "expected"), [
    (JobStatus.RUNNING, 0, None, JobStatus.SUCCEEDED),
    (JobStatus.RUNNING, 23, None, JobStatus.FAILED),
    (JobStatus.RUNNING, -signal.SIGTERM, None, JobStatus.FAILED),
    (JobStatus.RUNNING, 127, "FileNotFoundError: missing executable", JobStatus.FAILED),
    (JobStatus.STARTING, 127, "FileNotFoundError: missing executable", JobStatus.FAILED),
    (JobStatus.CANCEL_REQUESTED, -signal.SIGTERM, None, JobStatus.CANCELLED),
    (JobStatus.CANCEL_REQUESTED, 0, None, JobStatus.CANCELLED),
    (JobStatus.CANCEL_REQUESTED, None, None, JobStatus.CANCELLED),
    (JobStatus.RUNNING, None, None, JobStatus.FAILED),
])
def test_worker_persists_observed_results_after_recovery_and_cancellation(
    tmp_path, status, exit_code, launch_error, expected,
) -> None:
    store = QueueStore(tmp_path / "queue")
    record = starting_record(store)
    if status != JobStatus.STARTING:
        record = store.transition(
            record.run_id, JobStatus.RUNNING, now=NOW,
            launch=LaunchInfo(pid=43211, started_at=NOW, external_id="token"),
        )
    if status == JobStatus.CANCEL_REQUESTED:
        record = store.request_cancel(record.run_id, now=NOW)
    supervisor = finished_supervisor(tmp_path, record, exit_code, launch_error)
    worker = Worker(store, idle_scheduler(), DurableSupervisorLauncher(supervisor))

    tick = worker.tick()

    assert tick.errors == {}
    persisted = QueueStore(tmp_path / "queue").get(record.run_id)
    assert persisted.status == expected
    assert persisted.exit_code == exit_code
    assert persisted.launch_error == launch_error
    assert store.request_cancel(record.run_id) == persisted
    if status == JobStatus.STARTING:
        assert tick.started == [record.run_id]
    if exit_code is not None or launch_error is not None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            LegacyJobRecord.model_validate_json(
                (store.jobs_directory / f"{record.run_id}.json").read_text()
            )
    else:
        LegacyJobRecord.model_validate_json(
            (store.jobs_directory / f"{record.run_id}.json").read_text()
        )


def test_completion_race_retries_durable_result_without_relaunch(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path / "queue")
    record = starting_record(store)
    store.transition(
        record.run_id, JobStatus.RUNNING, now=NOW,
        launch=LaunchInfo(pid=43211, started_at=NOW, external_id="token"),
    )
    supervisor = finished_supervisor(tmp_path, record, 17)
    original_transition = store.transition

    def race_once(run_id, target, **kwargs):
        monkeypatch.setattr(store, "transition", original_transition)
        raise ConcurrentUpdate("retry terminal write")

    monkeypatch.setattr(store, "transition", race_once)
    worker = Worker(store, idle_scheduler(), DurableSupervisorLauncher(supervisor))
    assert worker.tick().completed == []
    assert store.get(record.run_id).status == JobStatus.RUNNING

    restarted = DurableJobSupervisor(tmp_path / "runs", proc_root=tmp_path / "missing-proc")
    worker = Worker(store, idle_scheduler(), DurableSupervisorLauncher(restarted))
    assert worker.tick().completed == [record.run_id]
    assert store.get(record.run_id).exit_code == 17


@pytest.mark.parametrize(("exit_code", "expected"), [(0, 0), (None, 125)])
def test_launcher_poll_reads_the_matching_receipt_once(
    tmp_path, monkeypatch, exit_code, expected,
) -> None:
    store = QueueStore(tmp_path / "queue")
    record = starting_record(store)
    supervisor = finished_supervisor(tmp_path, record, exit_code)
    reconcile = supervisor.reconcile
    reads = []

    def read_receipt(run_id):
        reads.append(run_id)
        return reconcile(run_id)

    monkeypatch.setattr(supervisor, "reconcile", read_receipt)

    assert DurableSupervisorLauncher(supervisor).poll(record) == expected
    assert reads == [record.run_id]


def test_recovery_records_definitive_launch_error_without_inventing_exit_code(tmp_path) -> None:
    store = QueueStore(tmp_path)
    record = starting_record(store)

    class UnavailableLauncher:
        def reconcile(self, record):
            return None

        def start(self, record):
            raise OSError("cannot spawn")

    result = Worker(store, idle_scheduler(), UnavailableLauncher()).tick()

    assert record.run_id in result.errors
    failed = store.get(record.run_id)
    assert failed.status == JobStatus.FAILED
    assert failed.exit_code is None
    assert failed.launch_error == "OSError: cannot spawn"


class RpcSsh:
    def __init__(self, rpc):
        self.rpc = rpc
        self.requests = []

    async def run_command(self, host, command, *, stdin=b"", timeout_seconds=0):
        request = json.loads(stdin)
        self.requests.append(request)
        return SSHResult(host, 0, json.dumps(self.rpc.handle(request)), "", 0.01)


def test_v1_wire_records_stay_legacy_compatible_while_show_exposes_results(tmp_path) -> None:
    from gpuops.worker.rpc import WorkerRpc

    rpc = WorkerRpc(tmp_path, server_id="gpu01", readiness_probe=lambda: (True, "ready"))
    record = starting_record(rpc.queue)
    failed = rpc.queue.transition(
        record.run_id, JobStatus.FAILED, now=NOW, exit_code=127,
        launch_error="FileNotFoundError: missing executable",
    )

    for payload in (
        rpc.submit({"run_id": record.run_id, "job_spec": spec().model_dump(mode="json")})["job"],
        rpc.show({"run_id": record.run_id})["job"],
        rpc.cancel({"run_id": record.run_id})["job"],
        rpc.list({})["jobs"][0],
    ):
        assert RESULT_FIELDS.isdisjoint(payload)
        assert LegacyJobRecord.model_validate(payload).status == JobStatus.FAILED

    ssh = RpcSsh(rpc)
    transport = SshWorkerTransport(ssh=ssh)
    target = RemoteTarget("gpu01", "gpu01")
    detail = asyncio.run(transport.get_job_detail(target, record.run_id))
    assert detail is not None
    assert detail.record == failed
    assert detail.record.model_dump(mode="json")["exit_code"] == 127
    assert asyncio.run(transport.get_job(target, record.run_id)) == failed
    assert ssh.requests[0]["params"] == {"run_id": record.run_id, "explain": True}
    assert ssh.requests[1]["params"] == {"run_id": record.run_id}
    assert asyncio.run(transport.submit(target, record.run_id, spec())) == failed
    assert asyncio.run(transport.cancel(target, record.run_id)) == failed


def test_new_client_reconciles_with_old_rpc_without_new_request_fields(tmp_path) -> None:
    from gpuops.worker.rpc import RPC_API_VERSION

    class LegacyRpc:
        record = None

        def handle(self, request):
            assert request["api_version"] == RPC_API_VERSION == "gpuops.worker.rpc/v1"
            method, params = request["method"], request["params"]
            if method == "health":
                assert params == {}
                result = {"status": "ok", "durable": True, "worker": {"ready": True}}
            elif method == "show":
                assert set(params) == {"run_id"}
                if self.record is None:
                    return {
                        "api_version": RPC_API_VERSION, "request_id": request["request_id"],
                        "ok": False, "error": {"code": "job_not_found"},
                    }
                result = {"job": self.record}
            else:
                assert method == "submit"
                assert set(params) == {"run_id", "job_spec"}
                self.record = JobRecord(
                    run_id=params["run_id"], spec=JobSpec.model_validate(params["job_spec"]),
                ).model_dump(mode="json", exclude=RESULT_FIELDS)
                # The old worker accepted the run, but its submit reply was lost.
                raise TimeoutError("submit reply lost")
            return {
                "api_version": RPC_API_VERSION, "request_id": request["request_id"],
                "ok": True, "result": result,
            }

    ssh = RpcSsh(LegacyRpc())
    client = ControlClient(
        [RemoteTarget("gpu01", "gpu01")], SshWorkerTransport(ssh=ssh),
        SubmissionJournal(tmp_path / "journal"), clock=lambda: NOW,
    )

    submitted = asyncio.run(client.submit(spec(), run_id="old-worker-run"))
    reconciled = asyncio.run(client.reconcile(submitted.run_id))

    assert submitted.record.exit_code is None
    assert submitted.record.launch_error is None
    assert reconciled.record == submitted.record
    assert sum(request["method"] == "submit" for request in ssh.requests) == 1
    saved = json.loads((tmp_path / "journal" / "old-worker-run.json").read_text())
    assert RESULT_FIELDS.isdisjoint(saved["remote_record"])
    LegacyJobRecord.model_validate(saved["remote_record"])


def test_old_terminal_record_keeps_unknown_results_null() -> None:
    record = JobRecord.model_validate({
        "run_id": "legacy-failed", "spec": spec().model_dump(mode="json"),
        "status": "failed", "finished_at": NOW,
        "failure_reason": "process exited with code 17",
    })

    assert record.exit_code is None
    assert record.launch_error is None
    assert record.model_dump(mode="json")["exit_code"] is None
