from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from gpuops.config import SchedulerSettings
from gpuops.core.queue import QueueStore
from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.core.scheduler import Scheduler
from gpuops.core.scheduling import ScheduleBlocker, assess_schedule_window
from gpuops.models import JobSpec, JobStatus, LaunchInfo
from gpuops.worker import Worker
from gpuops.worker.supervisor import LaunchInProgress

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def gpu(index: int, *, idle: bool = True, free_mib: int = 24000) -> GpuResource:
    total = 24576
    return GpuResource(
        index=index,
        uuid=f"GPU-{index}",
        memory_total_mib=total,
        memory_used_mib=total - free_mib,
        utilization_percent=0 if idle else 80,
        process_count=0 if idle else 1,
        idle=idle,
    )


def snapshot(*gpus: GpuResource, healthy: bool = True) -> ServerResourceSnapshot:
    return ServerResourceSnapshot(
        server_id="gpu01",
        sampled_at=NOW,
        healthy=healthy,
        error=None if healthy else "NVML failed",
        gpus=list(gpus),
    )


class Probe:
    def __init__(self, *snapshots: ServerResourceSnapshot):
        self.snapshots = deque(snapshots)
        self.calls = 0

    def snapshot(self) -> ServerResourceSnapshot:
        self.calls += 1
        return self.snapshots.popleft()


class PairProvider:
    def __init__(
        self,
        pair: tuple[ServerResourceSnapshot, ServerResourceSnapshot] | None,
    ):
        self.pair = pair
        self.calls = 0

    def confirmation_samples(
        self,
    ) -> tuple[ServerResourceSnapshot, ServerResourceSnapshot] | None:
        self.calls += 1
        return self.pair


def job(
    identifier: str,
    *,
    count: int = 1,
    exact: list[int] | None = None,
    memory_mib: int = 0,
    host: str = "auto",
    priority: str = "normal",
) -> JobSpec:
    return JobSpec(
        id=identifier,
        project="tests",
        command=["python", "train.py"],
        working_directory="/work",
        resources={
            "host": host,
            "gpu_count": count,
            "gpus": exact,
            "memory_mib": memory_mib,
        },
        schedule={"priority": priority},
    )


def scheduler(store: QueueStore, probe: Probe) -> Scheduler:
    return Scheduler(
        store,
        probe,
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=lambda: NOW,
    )


def cached_scheduler(store: QueueStore, provider: PairProvider, *, clock=lambda: NOW) -> Scheduler:
    return Scheduler(
        store,
        Probe(),
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=clock,
        confirmation_provider=provider,
    )


def cached_pair(
    *,
    first_at: datetime = NOW - timedelta(seconds=2),
    second_at: datetime = NOW,
    first_gpu: GpuResource | None = None,
    second_gpu: GpuResource | None = None,
) -> tuple[ServerResourceSnapshot, ServerResourceSnapshot]:
    return (
        snapshot(first_gpu or gpu(0)).model_copy(update={"sampled_at": first_at}),
        snapshot(second_gpu or gpu(0)).model_copy(update={"sampled_at": second_at}),
    )


def test_requires_two_idle_confirmations(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("confirmation"), now=NOW)
    probe = Probe(snapshot(gpu(0)), snapshot(gpu(0, idle=False)))

    assert scheduler(store, probe).tick() == []
    assert probe.calls == 2
    assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_cached_confirmation_pair_can_only_be_consumed_once(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(job("first"), now=NOW)
    provider = PairProvider(cached_pair())
    candidate_scheduler = cached_scheduler(store, provider)

    assert [decision.run_id for decision in candidate_scheduler.tick()] == [first.run_id]
    store.request_cancel(first.run_id, now=NOW)
    second = store.enqueue(job("second"), now=NOW)

    assert candidate_scheduler.tick() == []
    assert store.get(second.run_id).status == JobStatus.QUEUED
    assert provider.calls == 2


def test_cached_confirmation_rejects_invalid_time_and_identity_pairs(tmp_path) -> None:
    changed = gpu(0).model_copy(update={"uuid": "GPU-replaced"})
    cases = {
        "same-sample": cached_pair(first_at=NOW, second_at=NOW),
        "insufficient-gap": cached_pair(first_at=NOW - timedelta(seconds=1)),
        "excessive-gap": cached_pair(first_at=NOW - timedelta(seconds=46)),
        "stale-latest": cached_pair(
            first_at=NOW - timedelta(seconds=18),
            second_at=NOW - timedelta(seconds=16),
        ),
        "future-latest": cached_pair(
            first_at=NOW - timedelta(seconds=1),
            second_at=NOW + timedelta(seconds=1),
        ),
        "device-change": cached_pair(first_gpu=gpu(0), second_gpu=changed),
    }

    for name, pair in cases.items():
        store = QueueStore(tmp_path / name)
        queued = store.enqueue(job(name), now=NOW)

        assert cached_scheduler(store, PairProvider(pair)).tick() == []
        assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_cached_confirmation_allows_slow_historical_first_sample(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("historical-first"), now=NOW)
    provider = PairProvider(cached_pair(first_at=NOW - timedelta(seconds=40)))

    decisions = cached_scheduler(store, provider).tick()

    assert [decision.run_id for decision in decisions] == [queued.run_id]


@pytest.mark.parametrize("sibling_change", ["error", "disappear", "replace"])
def test_card_faults_do_not_block_healthy_card_leases(tmp_path, sibling_change) -> None:
    store = QueueStore(tmp_path)
    blocked = store.enqueue(job("bad-card", exact=[3]), now=NOW)
    runnable = store.enqueue(job("healthy-card", exact=[4]), now=NOW)
    first = snapshot(gpu(3), gpu(4)).model_copy(update={"sampled_at": NOW - timedelta(seconds=5)})
    bad_card = gpu(3).model_copy(update={
        "error": "query failed" if sibling_change == "error" else None,
        "uuid": "GPU-replaced" if sibling_change == "replace" else "GPU-3",
    })
    second = snapshot(gpu(4)) if sibling_change == "disappear" else snapshot(bad_card, gpu(4))

    decisions = cached_scheduler(store, PairProvider((first, second))).tick()

    assert [decision.run_id for decision in decisions] == [runnable.run_id]
    assert store.get(blocked.run_id).status == JobStatus.QUEUED
    assert store.get(runnable.run_id).allocation.gpu_uuids == {4: "GPU-4"}


@pytest.mark.parametrize("latest_offset", [-16, 1, -5])
def test_publication_cannot_refresh_or_duplicate_card_evidence(tmp_path, latest_offset) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("needs-fresh-card"), now=NOW)
    first = snapshot(gpu(4).model_copy(update={"sampled_at": NOW - timedelta(seconds=5)}))
    first = first.model_copy(update={"sampled_at": NOW - timedelta(seconds=1)})
    second = snapshot(gpu(4).model_copy(update={
        "sampled_at": NOW + timedelta(seconds=latest_offset),
    }))

    assert cached_scheduler(store, PairProvider((first, second))).tick() == []
    assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_independent_card_confirmations_survive_fast_publication_and_sibling_use(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = snapshot(*[
        gpu(index).model_copy(update={"sampled_at": NOW - timedelta(seconds=5)})
        for index in [4, 5]
    ]).model_copy(update={"sampled_at": NOW - timedelta(seconds=1)})
    second = snapshot(*[
        gpu(index).model_copy(update={"sampled_at": NOW}) for index in [4, 5]
    ])
    candidate = cached_scheduler(store, PairProvider((first, second)))
    first_job = store.enqueue(job("on-four", exact=[4]), now=NOW)
    assert [decision.run_id for decision in candidate.tick()] == [first_job.run_id]
    second_job = store.enqueue(job("on-five", exact=[5]), now=NOW)
    assert [decision.run_id for decision in candidate.tick()] == [second_job.run_id]
    store.request_cancel(first_job.run_id, now=NOW)
    store.enqueue(job("cannot-reuse-four", exact=[4]), now=NOW)
    assert candidate.tick() == []


def test_cached_confirmation_rechecks_schedule_window_before_leasing(tmp_path) -> None:
    store = QueueStore(tmp_path)
    constrained = JobSpec.model_validate(
        {
            **job("cached-deadline").model_dump(mode="json"),
            "schedule": {
                "expected_duration": "10s",
                "deadline": NOW + timedelta(seconds=20),
            },
        }
    )
    queued = store.enqueue(constrained, now=NOW)
    times = iter((NOW, NOW + timedelta(seconds=11)))

    assert cached_scheduler(
        store,
        PairProvider(cached_pair()),
        clock=lambda: next(times),
    ).tick() == []
    assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_device_identity_must_match_across_confirmations(tmp_path) -> None:
    store = QueueStore(tmp_path)
    store.enqueue(job("device-change"), now=NOW)
    changed = gpu(0).model_copy(update={"uuid": "replacement-device"})
    probe = Probe(snapshot(gpu(0)), snapshot(changed))

    assert scheduler(store, probe).tick() == []


def test_fail_closed_on_unhealthy_or_stale_snapshot(tmp_path) -> None:
    unhealthy_store = QueueStore(tmp_path / "unhealthy")
    unhealthy_store.enqueue(job("unhealthy"), now=NOW)
    unhealthy_probe = Probe(snapshot(gpu(0), healthy=False))
    assert scheduler(unhealthy_store, unhealthy_probe).tick() == []

    stale_store = QueueStore(tmp_path / "stale")
    stale_store.enqueue(job("stale"), now=NOW)
    stale = snapshot(gpu(0)).model_copy(update={"sampled_at": NOW - timedelta(minutes=1)})
    stale_probe = Probe(stale)
    assert scheduler(stale_store, stale_probe).tick() == []


def test_fifo_assignment_and_disjoint_parallel_jobs(tmp_path) -> None:
    store = QueueStore(tmp_path)
    first = store.enqueue(job("first", exact=[0]), now=NOW)
    second = store.enqueue(job("second", exact=[1]), now=NOW + timedelta(seconds=1))
    probe = Probe(snapshot(gpu(0), gpu(1)), snapshot(gpu(0), gpu(1)))

    decisions = scheduler(store, probe).tick()

    assert [decision.run_id for decision in decisions] == [first.run_id, second.run_id]
    assert [decision.gpu_indices for decision in decisions] == [[0], [1]]


def test_older_higher_priority_job_gets_auto_gpu_first(tmp_path) -> None:
    store = QueueStore(tmp_path)
    normal = store.enqueue(job("normal"), now=NOW)
    high = store.enqueue(job("high", priority="high"), now=NOW + timedelta(seconds=1))
    probe = Probe(snapshot(gpu(0)), snapshot(gpu(0)))

    decisions = scheduler(store, probe).tick()

    assert [decision.run_id for decision in decisions] == [high.run_id]
    assert store.get(normal.run_id).status == JobStatus.QUEUED


def test_exact_gpu_and_memory_requirements_are_enforced(tmp_path) -> None:
    store = QueueStore(tmp_path)
    store.enqueue(job("too-large", exact=[0], memory_mib=20000), now=NOW)
    store.enqueue(job("fits", exact=[1], memory_mib=12000), now=NOW + timedelta(seconds=1))
    available = snapshot(gpu(0, free_mib=16000), gpu(1, free_mib=16000))
    probe = Probe(available, available)

    decisions = scheduler(store, probe).tick()

    assert len(decisions) == 1
    assert decisions[0].gpu_indices == [1]


def test_snapshot_is_validated_after_probe_finishes(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("probe-latency"), run_id="probe-latency", now=NOW)
    current_time = [NOW]

    class AdvancingProbe:
        def snapshot(self) -> ServerResourceSnapshot:
            current_time[0] += timedelta(milliseconds=500)
            return snapshot(gpu(0)).model_copy(update={"sampled_at": current_time[0]})

    delayed_scheduler = Scheduler(
        store,
        AdvancingProbe(),
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=lambda: current_time[0],
        sleeper=lambda _: None,
    )

    decisions = delayed_scheduler.tick()

    assert [decision.run_id for decision in decisions] == [queued.run_id]


def test_slow_confirmation_can_use_historical_first_snapshot(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("stale-confirmation"), now=NOW)
    current_time = [NOW]

    class SlowProbe:
        calls = 0

        def snapshot(self) -> ServerResourceSnapshot:
            self.calls += 1
            current_time[0] += timedelta(seconds=1 if self.calls == 1 else 16)
            return snapshot(gpu(0)).model_copy(update={"sampled_at": current_time[0]})

    slow_probe = SlowProbe()
    delayed_scheduler = Scheduler(
        store,
        slow_probe,
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=lambda: current_time[0],
        sleeper=lambda _: None,
    )

    decisions = delayed_scheduler.tick()

    assert [decision.run_id for decision in decisions] == [queued.run_id]
    assert slow_probe.calls == 2
    assert store.get(queued.run_id).status == JobStatus.LEASED


def test_slow_confirmation_cannot_lease_from_stale_latest_snapshot(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("stale-latest"), now=NOW)
    current_time = [NOW]

    class StaleSecondProbe:
        calls = 0

        def snapshot(self) -> ServerResourceSnapshot:
            self.calls += 1
            if self.calls == 1:
                return snapshot(gpu(0))
            current_time[0] += timedelta(seconds=16)
            return snapshot(gpu(0))

    slow_probe = StaleSecondProbe()
    delayed_scheduler = Scheduler(
        store,
        slow_probe,
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=lambda: current_time[0],
        sleeper=lambda _: None,
    )

    assert delayed_scheduler.tick() == []
    assert slow_probe.calls == 2
    assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_schedule_window_is_rechecked_after_slow_gpu_queries(tmp_path) -> None:
    store = QueueStore(tmp_path)
    constrained = JobSpec.model_validate(
        {
            **job("slow-deadline").model_dump(mode="json"),
            "schedule": {
                "expected_duration": "10s",
                "deadline": NOW + timedelta(seconds=20),
            },
        }
    )
    queued = store.enqueue(constrained, now=NOW)
    current_time = [NOW]

    class AdvancingProbe:
        def snapshot(self) -> ServerResourceSnapshot:
            current_time[0] += timedelta(seconds=6)
            return snapshot(gpu(0)).model_copy(update={"sampled_at": current_time[0]})

    delayed_scheduler = Scheduler(
        store,
        AdvancingProbe(),
        server_id="gpu01",
        worker_id="worker-1",
        settings=SchedulerSettings(idle_confirmation_delay_seconds=0),
        clock=lambda: current_time[0],
        sleeper=lambda _: None,
    )

    assert delayed_scheduler.tick() == []
    assert store.get(queued.run_id).status == JobStatus.QUEUED


def test_jobs_for_other_server_are_ignored(tmp_path) -> None:
    store = QueueStore(tmp_path)
    store.enqueue(job("other", host="gpu02"), now=NOW)
    probe = Probe(snapshot(gpu(0)))

    assert scheduler(store, probe).tick() == []
    assert probe.calls == 0


def test_job_that_cannot_meet_deadline_is_not_leased(tmp_path) -> None:
    store = QueueStore(tmp_path)
    constrained = JobSpec.model_validate(
        {
            **job("deadline").model_dump(mode="json"),
            "schedule": {
                "priority": "normal",
                "expected_duration": "2h",
                "deadline": NOW + timedelta(hours=1),
            },
        }
    )
    store.enqueue(constrained, now=NOW)
    probe = Probe(snapshot(gpu(0)))

    assert scheduler(store, probe).tick() == []
    assert probe.calls == 0


def test_scheduler_candidate_filter_uses_shared_schedule_window_assessment(tmp_path) -> None:
    store = QueueStore(tmp_path)
    future = store.enqueue(
        JobSpec.model_validate(
            {
                **job("future").model_dump(mode="json"),
                "schedule": {"not_before": NOW + timedelta(hours=1)},
            }
        ),
        run_id="future",
        now=NOW,
    )
    deadline = store.enqueue(
        JobSpec.model_validate(
            {
                **job("deadline-shared").model_dump(mode="json"),
                "schedule": {
                    "expected_duration": "2h",
                    "deadline": NOW + timedelta(hours=1),
                },
            }
        ),
        run_id="deadline-shared",
        now=NOW,
    )
    other_host = store.enqueue(
        job("other-shared", host="gpu02"),
        run_id="other-shared",
        now=NOW,
    )
    eligible = store.enqueue(job("eligible-shared"), run_id="eligible-shared", now=NOW)
    candidate_filter = scheduler(store, Probe())

    assessments = {
        record.run_id: assess_schedule_window(record, server_id="gpu01", now=NOW)
        for record in (future, deadline, other_host, eligible)
    }

    assert assessments["future"].blocker == ScheduleBlocker.NOT_BEFORE
    assert assessments["future"].available_at == NOW + timedelta(hours=1)
    assert assessments["deadline-shared"].blocker == ScheduleBlocker.DEADLINE
    assert assessments["other-shared"].blocker == ScheduleBlocker.HOST_MISMATCH
    assert assessments["eligible-shared"].eligible is True
    assert {
        record.run_id: candidate_filter._is_candidate(record, NOW)
        for record in (future, deadline, other_host, eligible)
    } == {run_id: assessment.eligible for run_id, assessment in assessments.items()}


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.exit_codes: dict[str, int | None] = {}
        self.cancelled: list[str] = []
        self.cancel_ready: dict[str, bool] = {}

    def start(self, record):
        self.started.append(record.run_id)
        self.exit_codes[record.run_id] = None
        return LaunchInfo(pid=4242, started_at=NOW)

    def reconcile(self, record):
        del record
        return None

    def poll(self, record):
        return self.exit_codes[record.run_id]

    def cancel(self, record):
        self.cancelled.append(record.run_id)
        return self.cancel_ready.get(record.run_id, False)


class RecordingLauncher(FakeLauncher):
    def __init__(self) -> None:
        super().__init__()
        self.polled: list[str] = []

    def poll(self, record):
        self.polled.append(record.run_id)
        return super().poll(record)


def test_worker_tick_uses_injected_launcher_and_reconciles(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("worker"), now=NOW)
    probe = Probe(snapshot(gpu(0)), snapshot(gpu(0)))
    fake = FakeLauncher()
    worker = Worker(store, scheduler(store, probe), fake)

    first_tick = worker.tick()
    assert first_tick.started == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.RUNNING

    fake.exit_codes[queued.run_id] = 0
    second_tick = worker.tick()
    assert second_tick.completed == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.SUCCEEDED


def test_worker_does_not_poll_another_worker_process(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("foreign"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="other-worker",
        ttl=timedelta(minutes=5),
        expected_revision=queued.revision,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING)
    store.transition(
        starting.run_id,
        JobStatus.RUNNING,
        launch=LaunchInfo(pid=123, started_at=NOW),
    )
    fake = RecordingLauncher()
    fake.exit_codes[queued.run_id] = 0
    worker = Worker(store, scheduler(store, Probe()), fake)

    worker.tick()

    assert fake.polled == []
    assert store.get(queued.run_id).status == JobStatus.RUNNING


def test_worker_keeps_starting_reserved_if_running_write_fails(tmp_path, monkeypatch) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("uncertain-start"), now=NOW)
    probe = Probe(snapshot(gpu(0)), snapshot(gpu(0)))
    fake = FakeLauncher()
    worker = Worker(store, scheduler(store, probe), fake)
    original_transition = store.transition

    def fail_running_transition(run_id, target, **kwargs):
        if target == JobStatus.RUNNING:
            raise OSError("simulated disk failure")
        return original_transition(run_id, target, **kwargs)

    monkeypatch.setattr(store, "transition", fail_running_transition)
    result = worker.tick()

    assert fake.started == [queued.run_id]
    assert queued.run_id in result.errors
    assert store.get(queued.run_id).status == JobStatus.STARTING


def test_worker_recovers_starting_job_from_durable_launch(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("recover"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker-1",
        ttl=timedelta(minutes=5),
        expected_revision=queued.revision,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING, now=NOW)
    fake = FakeLauncher()
    recovered = LaunchInfo(pid=9876, started_at=NOW, external_id="stable-token")
    fake.reconcile = lambda record: recovered
    worker = Worker(store, scheduler(store, Probe()), fake)

    result = worker.tick()

    assert result.started == [starting.run_id]
    assert store.get(starting.run_id).status == JobStatus.RUNNING
    assert store.get(starting.run_id).launch == recovered


def test_worker_reconciles_expired_starting_job_instead_of_releasing_gpu(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("expired-start"), now=NOW - timedelta(minutes=2))
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker-1",
        ttl=timedelta(seconds=30),
        expected_revision=queued.revision,
        now=NOW - timedelta(minutes=2),
    )
    store.transition(leased.run_id, JobStatus.STARTING, now=NOW - timedelta(minutes=2))
    fake = FakeLauncher()
    worker = Worker(store, scheduler(store, Probe()), fake)

    worker.tick()

    record = store.get(queued.run_id)
    assert record.status == JobStatus.RUNNING
    assert fake.started == [queued.run_id]


def test_launch_in_progress_keeps_starting_until_receipt_is_reconciled(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("slow-receipt"), now=NOW)

    class SlowReceiptLauncher(FakeLauncher):
        receipt: LaunchInfo | None = None

        def start(self, record):
            self.started.append(record.run_id)
            raise LaunchInProgress("receipt is not durable yet")

        def reconcile(self, record):
            del record
            return self.receipt

    fake = SlowReceiptLauncher()
    worker = Worker(
        store,
        scheduler(store, Probe(snapshot(gpu(0)), snapshot(gpu(0)))),
        fake,
    )

    uncertain = worker.tick()
    assert queued.run_id in uncertain.errors
    assert store.get(queued.run_id).status == JobStatus.STARTING
    assert fake.started == [queued.run_id]

    fake.receipt = LaunchInfo(pid=8765, started_at=NOW, external_id="durable-token")
    recovered = worker.tick()
    assert recovered.started == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.RUNNING
    assert fake.started == [queued.run_id]


def test_definitive_launch_failure_marks_starting_failed(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("definite-failure"), now=NOW)

    class FailingLauncher(FakeLauncher):
        def start(self, record):
            del record
            raise OSError("exec is unavailable")

    fake = FailingLauncher()
    worker = Worker(
        store,
        scheduler(store, Probe(snapshot(gpu(0)), snapshot(gpu(0)))),
        fake,
    )

    failed = worker.tick()
    record = store.get(queued.run_id)
    assert queued.run_id in failed.errors
    assert record.status == JobStatus.FAILED
    assert record.failure_reason == "launch failed: exec is unavailable"
    assert record.launch_error == "OSError: exec is unavailable"
    assert record.exit_code is None


def test_worker_keeps_cancel_pending_until_launcher_has_durable_result(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("cancel-running"), now=NOW)
    fake = FakeLauncher()
    worker = Worker(
        store,
        scheduler(store, Probe(snapshot(gpu(0)), snapshot(gpu(0)))),
        fake,
    )
    worker.tick()
    store.request_cancel(queued.run_id, now=NOW)

    pending = worker.tick()
    assert pending.cancelled == []
    assert store.get(queued.run_id).status == JobStatus.CANCEL_REQUESTED

    fake.cancel_ready[queued.run_id] = True
    completed = worker.tick()
    assert completed.cancelled == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.CANCELLED
    assert fake.cancelled == [queued.run_id, queued.run_id]


def test_worker_does_not_restart_cancelled_starting_job_without_receipt(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("cancel-starting"), now=NOW)
    leased = store.lease(
        queued.run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker-1",
        ttl=timedelta(minutes=1),
        expected_revision=queued.revision,
        now=NOW,
    )
    starting = store.transition(leased.run_id, JobStatus.STARTING, now=NOW)
    store.request_cancel(starting.run_id, now=NOW)
    fake = FakeLauncher()
    worker = Worker(store, scheduler(store, Probe()), fake)

    result = worker.tick()

    assert result.cancelled == []
    assert fake.started == []
    assert fake.cancelled == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.CANCEL_REQUESTED


def test_cancel_race_after_launch_does_not_duplicate_start(tmp_path) -> None:
    store = QueueStore(tmp_path)
    queued = store.enqueue(job("cancel-race"), now=NOW)

    class RaceLauncher(FakeLauncher):
        def start(self, record):
            launch = super().start(record)
            store.request_cancel(record.run_id, now=NOW)
            self.cancel_ready[record.run_id] = True
            return launch

    fake = RaceLauncher()
    worker = Worker(
        store,
        scheduler(store, Probe(snapshot(gpu(0)), snapshot(gpu(0)))),
        fake,
    )

    raced = worker.tick()
    assert queued.run_id in raced.errors
    assert store.get(queued.run_id).status == JobStatus.CANCEL_REQUESTED

    completed = worker.tick()
    assert completed.cancelled == [queued.run_id]
    assert fake.started == [queued.run_id]
    assert store.get(queued.run_id).status == JobStatus.CANCELLED
