from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpuops.control import (
    ControlClient,
    DefinitiveRemoteSubmissionError,
    HostSelectionError,
    JournalState,
    RemoteHostState,
    RemoteTarget,
    SubmissionConflict,
    SubmissionJournal,
    SubmissionNotAccepted,
    SubmissionNotFound,
    SubmissionState,
    SubmissionUncertain,
    generate_run_id,
)
from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.models import JobRecord, JobSpec

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
TARGET_01 = RemoteTarget("gpu01", "lab-gpu01")
TARGET_02 = RemoteTarget("gpu02", "lab-gpu02")


def manifest(*, host: str = "auto", identifier: str = "ranking-seed-42") -> JobSpec:
    return JobSpec(
        id=identifier,
        project="ranking",
        command=["python", "train.py", "--seed", "42"],
        working_directory="/data/researcher/ranking",
        resources={"host": host, "gpu_count": 1, "memory_mib": 12000},
    )


def gpu(index: int, *, idle: bool = True, free_mib: int = 24000) -> GpuResource:
    total = 24576
    return GpuResource(
        index=index,
        uuid=f"GPU-{index}",
        memory_total_mib=total,
        memory_used_mib=total - free_mib,
        utilization_percent=0 if idle else 90,
        process_count=0 if idle else 1,
        idle=idle,
    )


def host_state(
    server_id: str,
    *gpus: GpuResource,
    queue: tuple[JobRecord, ...] = (),
) -> RemoteHostState:
    return RemoteHostState(
        snapshot=ServerResourceSnapshot(
            server_id=server_id,
            sampled_at=NOW,
            gpus=list(gpus),
        ),
        queue=queue,
    )


class FakeTransport:
    def __init__(self, states: dict[str, RemoteHostState | Exception]):
        self.states = states
        self.jobs: dict[tuple[str, str], JobRecord] = {}
        self.inspect_calls: list[str] = []
        self.submit_calls: list[tuple[str, str, JobSpec]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.accept_then_raise: set[str] = set()
        self.accept_then_error: dict[str, Exception] = {}
        self.reject_submissions: set[str] = set()
        self.definitive_rejections: dict[str, DefinitiveRemoteSubmissionError] = {}
        self.offline_get: set[str] = set()

    async def inspect(self, target: RemoteTarget) -> RemoteHostState:
        self.inspect_calls.append(target.server_id)
        state = self.states[target.server_id]
        if isinstance(state, Exception):
            raise state
        return state

    async def submit(self, target: RemoteTarget, run_id: str, spec: JobSpec) -> JobRecord:
        self.submit_calls.append((target.server_id, run_id, spec))
        definitive_rejection = self.definitive_rejections.get(target.server_id)
        if definitive_rejection is not None:
            raise definitive_rejection
        if target.server_id in self.reject_submissions:
            raise TimeoutError("submit response lost")
        key = (target.server_id, run_id)
        record = self.jobs.setdefault(
            key,
            JobRecord(
                run_id=run_id,
                spec=spec,
                created_at=NOW,
                updated_at=NOW,
            ),
        )
        if target.server_id in self.accept_then_raise:
            raise TimeoutError("response lost after commit")
        accepted_error = self.accept_then_error.get(target.server_id)
        if accepted_error is not None:
            raise accepted_error
        return record

    async def get_job(self, target: RemoteTarget, run_id: str) -> JobRecord | None:
        self.get_calls.append((target.server_id, run_id))
        if target.server_id in self.offline_get:
            raise ConnectionError("host offline")
        return self.jobs.get((target.server_id, run_id))


def client(tmp_path: Path, transport: FakeTransport) -> ControlClient:
    return ControlClient(
        [TARGET_02, TARGET_01],
        transport,
        SubmissionJournal(tmp_path / "journal"),
        clock=lambda: NOW,
    )


def test_run_id_is_stable_for_same_manifest_and_submission_time() -> None:
    first = generate_run_id(manifest(), NOW)
    second = generate_run_id(manifest(), NOW)
    changed = generate_run_id(manifest(identifier="another-job"), NOW)

    assert first == second
    assert first != changed
    assert first.startswith("ranking-seed-42-")
    assert len(first) <= 180


def test_explicit_host_is_fixed_without_auto_inspection(tmp_path: Path) -> None:
    transport = FakeTransport(
        {"gpu01": ConnectionError("unused"), "gpu02": ConnectionError("unused")}
    )

    result = asyncio.run(
        client(tmp_path, transport).submit(manifest(host="gpu02"), run_id="fixed-run")
    )

    assert result.target == TARGET_02
    assert result.spec.resources.host == "gpu02"
    assert transport.inspect_calls == []
    assert transport.submit_calls[0][:2] == ("gpu02", "fixed-run")


def test_new_unsupported_tracking_is_rejected_before_remote_calls(tmp_path: Path) -> None:
    spec = manifest(host="gpu01")
    spec.tracking.progress_adapter = "unimplemented"
    transport = FakeTransport({})
    control = client(tmp_path, transport)

    with pytest.raises(ValueError, match="tracking.progress_adapter"):
        asyncio.run(control.submit(spec, run_id="unsupported"))
    assert transport.inspect_calls == transport.get_calls == transport.submit_calls == []
    assert control.journal.get("unsupported") is None


def test_single_target_auto_queues_without_gpu_inspection(tmp_path: Path) -> None:
    transport = FakeTransport({"gpu01": TimeoutError("GPU query stuck")})
    control = ControlClient(
        [TARGET_01], transport, SubmissionJournal(tmp_path / "journal"), clock=lambda: NOW
    )

    result = asyncio.run(control.submit(manifest(), run_id="local-auto"))

    assert result.target == TARGET_01
    assert result.record.status.value == "queued"
    assert result.spec.resources.host == "gpu01"
    assert transport.inspect_calls == []


def test_single_target_auto_still_requires_a_ready_worker(tmp_path: Path) -> None:
    transport = FakeTransport({})
    transport.definitive_rejections["gpu01"] = DefinitiveRemoteSubmissionError(
        "worker_unavailable", "worker heartbeat is stale"
    )
    control = ControlClient(
        [TARGET_01], transport, SubmissionJournal(tmp_path / "journal"), clock=lambda: NOW
    )

    with pytest.raises(SubmissionNotAccepted) as captured:
        asyncio.run(control.submit(manifest(), run_id="local-unready"))

    assert captured.value.code == "worker_unavailable"
    assert transport.inspect_calls == []


def test_accepted_legacy_tracking_can_be_retried(tmp_path: Path) -> None:
    spec = manifest(host="gpu01")
    spec.tracking.progress_adapter = "legacy"
    transport = FakeTransport({})
    control = client(tmp_path, transport)
    control.journal.create_pending(
        run_id="legacy", target=TARGET_01, request_spec=spec, frozen_spec=spec, now=NOW
    )
    record = JobRecord(run_id="legacy", spec=spec, created_at=NOW, updated_at=NOW)
    control.journal.mark_accepted("legacy", record, now=NOW)

    result = asyncio.run(control.submit(spec, run_id="legacy"))
    assert result.record == record
    assert transport.get_calls == transport.submit_calls == []


def test_auto_uses_live_gpu_and_queue_state_deterministically(tmp_path: Path) -> None:
    queued = JobRecord(
        run_id="existing-queued-run",
        spec=manifest(host="gpu01", identifier="existing-job"),
        created_at=NOW,
        updated_at=NOW,
    )
    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0), gpu(1), queue=(queued,)),
            "gpu02": host_state("gpu02", gpu(0)),
        }
    )

    result = asyncio.run(client(tmp_path, transport).submit(manifest(), run_id="auto-run"))

    assert sorted(transport.inspect_calls) == ["gpu01", "gpu02"]
    assert result.target == TARGET_02
    assert result.spec.resources.host == "gpu02"
    journal = SubmissionJournal(tmp_path / "journal").get("auto-run")
    assert journal is not None
    assert journal.state == JournalState.ACCEPTED
    assert journal.frozen_spec.resources.host == "gpu02"


def test_auto_tie_breaks_by_server_id_and_ignores_offline_host(tmp_path: Path) -> None:
    tied = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0)),
        }
    )
    tied_result = asyncio.run(client(tmp_path / "tie", tied).submit(manifest(), run_id="tie-run"))
    assert tied_result.target == TARGET_01

    partial = FakeTransport(
        {
            "gpu01": ConnectionError("gpu01 offline"),
            "gpu02": host_state("gpu02", gpu(0)),
        }
    )
    partial_result = asyncio.run(
        client(tmp_path / "partial", partial).submit(manifest(), run_id="partial-run")
    )
    assert partial_result.target == TARGET_02

    both_offline = FakeTransport(
        {"gpu01": ConnectionError("offline"), "gpu02": TimeoutError("offline")}
    )
    with pytest.raises(HostSelectionError):
        asyncio.run(
            client(tmp_path / "offline", both_offline).submit(manifest(), run_id="offline-run")
        )


def test_ambiguous_submit_reconciles_same_host_and_run_id(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0, idle=False)),
        }
    )
    transport.accept_then_raise.add("gpu01")
    control = client(tmp_path, transport)

    result = asyncio.run(control.submit(manifest(), run_id="unclear-run"))
    retried = asyncio.run(control.submit(manifest(), run_id="unclear-run"))

    assert result.reconciled is True
    assert retried.run_id == result.run_id == "unclear-run"
    assert retried.target == result.target == TARGET_01
    assert {(host, run_id) for host, run_id, _ in transport.submit_calls} == {
        ("gpu01", "unclear-run")
    }
    assert all(host == "gpu01" for host, run_id in transport.get_calls if run_id == "unclear-run")
    assert len(transport.submit_calls) == 1


def test_quota_error_after_remote_commit_reconciles_as_accepted(tmp_path: Path) -> None:
    class QuotaAfterCommit(RuntimeError):
        code = "storage_quota_exceeded"

    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0, idle=False)),
        }
    )
    transport.accept_then_error["gpu01"] = QuotaAfterCommit("event capacity exceeded")

    result = asyncio.run(
        client(tmp_path, transport).submit(manifest(), run_id="quota-after-commit")
    )

    assert result.run_id == "quota-after-commit"
    assert result.reconciled is True
    assert result.record == transport.jobs[("gpu01", "quota-after-commit")]
    assert SubmissionJournal(tmp_path / "journal").get("quota-after-commit").state == (
        JournalState.ACCEPTED
    )


def test_pending_journal_survives_restart_and_retries_same_submission(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0, idle=False)),
        }
    )
    transport.reject_submissions.add("gpu01")
    journal = SubmissionJournal(tmp_path / "journal")
    first_client = ControlClient([TARGET_01, TARGET_02], transport, journal, clock=lambda: NOW)

    with pytest.raises(SubmissionUncertain) as captured:
        asyncio.run(first_client.submit(manifest(), run_id="restart-run"))
    assert captured.value.run_id == "restart-run"
    assert captured.value.server_id == "gpu01"
    assert captured.value.code == "submission_uncertain"
    assert captured.value.submission_state == SubmissionState.UNKNOWN
    pending = journal.get("restart-run")
    assert pending is not None
    assert pending.state == JournalState.PENDING
    assert pending.server_id == "gpu01"

    transport.reject_submissions.clear()
    restarted = ControlClient(
        [TARGET_02],
        transport,
        SubmissionJournal(tmp_path / "journal"),
        clock=lambda: NOW + timedelta(minutes=1),
    )
    result = asyncio.run(restarted.reconcile("restart-run"))

    assert result.run_id == "restart-run"
    assert result.target == TARGET_01
    assert result.reconciled is True
    retry_hosts = [host for host, run_id, _ in transport.submit_calls if run_id == "restart-run"]
    assert all(host == "gpu01" for host in retry_hosts)
    assert {run_id for _, run_id, _ in transport.submit_calls} == {"restart-run"}
    assert journal.get("restart-run").state == JournalState.ACCEPTED


def test_definitive_remote_rejection_preserves_pending_identity_and_metadata(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0, idle=False)),
        }
    )
    transport.definitive_rejections["gpu01"] = DefinitiveRemoteSubmissionError(
        "invalid_params",
        "job_spec validation failed",
    )
    control = client(tmp_path, transport)

    with pytest.raises(SubmissionNotAccepted) as captured:
        asyncio.run(control.submit(manifest(), run_id="rejected-run"))

    error = captured.value
    assert error.run_id == "rejected-run"
    assert error.server_id == "gpu01"
    assert error.code == "invalid_params"
    assert error.submission_state == SubmissionState.NOT_ACCEPTED
    assert len(transport.get_calls) == 1
    pending = control.journal.get("rejected-run")
    assert pending is not None
    assert pending.state == JournalState.PENDING
    assert pending.target == TARGET_01


def test_reconcile_reports_conflicting_remote_run_as_not_accepted(tmp_path: Path) -> None:
    transport = FakeTransport({})
    control = client(tmp_path, transport)
    saved_spec = manifest(host="gpu01")
    control.journal.create_pending(
        run_id="conflicting-run",
        target=TARGET_01,
        request_spec=saved_spec,
        frozen_spec=saved_spec,
        now=NOW,
    )
    transport.jobs[("gpu01", "conflicting-run")] = JobRecord(
        run_id="conflicting-run",
        spec=manifest(host="gpu01", identifier="different-job"),
        created_at=NOW,
        updated_at=NOW,
    )

    with pytest.raises(SubmissionNotAccepted) as captured:
        asyncio.run(control.reconcile("conflicting-run"))

    assert captured.value.code == "run_id_conflict"
    assert captured.value.submission_state == SubmissionState.NOT_ACCEPTED
    assert transport.submit_calls == []
    assert control.journal.get("conflicting-run").state == JournalState.PENDING


def test_wrong_run_id_from_remote_lookup_remains_unknown(tmp_path: Path) -> None:
    transport = FakeTransport({})
    control = client(tmp_path, transport)
    saved_spec = manifest(host="gpu01")
    control.journal.create_pending(
        run_id="requested-run",
        target=TARGET_01,
        request_spec=saved_spec,
        frozen_spec=saved_spec,
        now=NOW,
    )
    transport.jobs[("gpu01", "requested-run")] = JobRecord(
        run_id="unrelated-run",
        spec=saved_spec,
        created_at=NOW,
        updated_at=NOW,
    )

    with pytest.raises(SubmissionUncertain) as captured:
        asyncio.run(control.reconcile("requested-run"))

    assert captured.value.code == "remote_protocol_error"
    assert captured.value.submission_state == SubmissionState.UNKNOWN
    assert transport.submit_calls == []
    assert control.journal.get("requested-run").state == JournalState.PENDING


def test_targeted_reconcile_requires_an_existing_journal_identity(tmp_path: Path) -> None:
    control = client(tmp_path, FakeTransport({}))

    with pytest.raises(SubmissionNotFound) as captured:
        asyncio.run(control.reconcile("missing-run"))

    assert captured.value.run_id == "missing-run"
    assert captured.value.code == "submission_not_found"


def test_concurrent_reconciliation_keeps_one_remote_run_identity(tmp_path: Path) -> None:
    class RacingTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__({})
            self.initial_lookups = 0

        async def get_job(self, target: RemoteTarget, run_id: str) -> JobRecord | None:
            self.get_calls.append((target.server_id, run_id))
            key = (target.server_id, run_id)
            if run_id == "racing-run" and key not in self.jobs and self.initial_lookups < 2:
                self.initial_lookups += 1
                while self.initial_lookups < 2:
                    await asyncio.sleep(0)
                return None
            return self.jobs.get(key)

    transport = RacingTransport()
    journal = SubmissionJournal(tmp_path / "journal")
    saved_spec = manifest(host="gpu01")
    journal.create_pending(
        run_id="racing-run",
        target=TARGET_01,
        request_spec=saved_spec,
        frozen_spec=saved_spec,
        now=NOW,
    )
    first = ControlClient([TARGET_01], transport, journal, clock=lambda: NOW)
    second = ControlClient([TARGET_02], transport, journal, clock=lambda: NOW)

    async def reconcile_both():
        return await asyncio.gather(
            first.reconcile("racing-run"),
            second.reconcile("racing-run"),
        )

    results = asyncio.run(reconcile_both())

    assert {result.run_id for result in results} == {"racing-run"}
    assert {result.target for result in results} == {TARGET_01}
    assert set(transport.jobs) == {("gpu01", "racing-run")}
    assert {run_id for _server_id, run_id, _spec in transport.submit_calls} == {"racing-run"}
    assert journal.get("racing-run").state == JournalState.ACCEPTED


def test_same_manifest_without_explicit_run_id_creates_intentional_new_runs(
    tmp_path: Path,
) -> None:
    transport = FakeTransport({})
    timestamps = iter(
        [
            NOW,
            NOW,
            NOW + timedelta(microseconds=1),
            NOW + timedelta(microseconds=1),
        ]
    )
    control = ControlClient(
        [TARGET_01],
        transport,
        SubmissionJournal(tmp_path / "journal"),
        clock=lambda: next(timestamps),
    )
    spec = manifest(host="gpu01")

    first = asyncio.run(control.submit(spec))
    second = asyncio.run(control.submit(spec))

    assert first.run_id != second.run_id
    assert {run_id for _server_id, run_id, _spec in transport.submit_calls} == {
        first.run_id,
        second.run_id,
    }


def test_explicit_run_id_still_rejects_a_different_manifest(tmp_path: Path) -> None:
    transport = FakeTransport({})
    control = client(tmp_path, transport)
    asyncio.run(control.submit(manifest(host="gpu01"), run_id="fixed-identity"))

    with pytest.raises(SubmissionConflict):
        asyncio.run(
            control.submit(
                manifest(host="gpu01", identifier="different-job"),
                run_id="fixed-identity",
            )
        )


def test_remote_acceptance_survives_local_journal_mark_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport({})
    control = client(tmp_path, transport)
    spec = manifest(host="gpu01")
    control.journal.create_pending(
        run_id="accepted-warning",
        target=TARGET_01,
        request_spec=spec,
        frozen_spec=spec,
        now=NOW,
    )

    def fail_mark_accepted(*args, **kwargs):
        del args, kwargs
        raise OSError("disk sync failed")

    monkeypatch.setattr(control.journal, "mark_accepted", fail_mark_accepted)

    result = asyncio.run(control.reconcile("accepted-warning"))

    assert result.record.run_id == "accepted-warning"
    assert result.journal_warning is not None
    assert "disk sync failed" in result.journal_warning
    assert control.journal.get("accepted-warning").state == JournalState.PENDING


def test_reconcile_offline_entry_does_not_block_other_host(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "gpu01": host_state("gpu01", gpu(0)),
            "gpu02": host_state("gpu02", gpu(0)),
        }
    )
    journal = SubmissionJournal(tmp_path / "journal")
    first_spec = manifest(host="gpu01", identifier="first-job")
    second_spec = manifest(host="gpu02", identifier="second-job")
    journal.create_pending(
        run_id="six-pending",
        target=TARGET_01,
        request_spec=first_spec,
        frozen_spec=first_spec,
        now=NOW,
    )
    journal.create_pending(
        run_id="eleven-pending",
        target=TARGET_02,
        request_spec=second_spec,
        frozen_spec=second_spec,
        now=NOW,
    )
    transport.offline_get.add("gpu01")

    report = asyncio.run(
        ControlClient(
            [TARGET_01, TARGET_02], transport, journal, clock=lambda: NOW
        ).reconcile_pending()
    )

    assert report.accepted == ("eleven-pending",)
    assert report.pending == ("six-pending",)
    assert "six-pending" in report.errors
    assert report.outcomes == {
        "eleven-pending": SubmissionState.ACCEPTED,
        "six-pending": SubmissionState.UNKNOWN,
    }
    assert report.error_codes == {"six-pending": "lookup_failed"}
    assert journal.get("eleven-pending").state == JournalState.ACCEPTED
    assert journal.get("six-pending").state == JournalState.PENDING
