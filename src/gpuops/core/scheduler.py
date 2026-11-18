from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import Field

from gpuops.config import SchedulerSettings
from gpuops.core.queue import ConcurrentUpdate, QueueStore, ResourceConflict
from gpuops.core.resources import GpuResource, ResourceProbe, ServerResourceSnapshot
from gpuops.core.scheduling import (
    ALLOCATED_STATUSES,
    assess_schedule_window,
    reserved_gpu_indices,
)
from gpuops.models import JobRecord, JobStatus, StrictModel, utc_now


class LeaseDecision(StrictModel):
    run_id: str
    server_id: str
    gpu_indices: list[int] = Field(min_length=1)
    lease_token: str


class ConfirmationProvider(Protocol):
    """Provide two ordered observations without performing work inline."""

    def confirmation_samples(
        self,
    ) -> tuple[ServerResourceSnapshot, ServerResourceSnapshot] | None: ...


class Scheduler:
    """Conservative per-server scheduler; it only creates leases and never starts processes."""

    def __init__(
        self,
        store: QueueStore,
        probe: ResourceProbe,
        *,
        server_id: str,
        worker_id: str,
        settings: SchedulerSettings | None = None,
        clock: Callable[[], datetime] = utc_now,
        sleeper: Callable[[float], None] = time.sleep,
        confirmation_provider: ConfirmationProvider | None = None,
        gpu_ready: Callable[[GpuResource], bool] | None = None,
    ):
        self.store = store
        self.probe = probe
        self.server_id = server_id
        self.worker_id = worker_id
        self.settings = settings or SchedulerSettings()
        self.clock = clock
        self.sleeper = sleeper
        self.confirmation_provider = confirmation_provider
        # Checked only for GPUs about to be leased, so idle servers run no CUDA checks.
        self.gpu_ready = gpu_ready
        self._consumed_samples: dict[str, datetime] = {}

    def tick(self) -> list[LeaseDecision]:
        now = self.clock()
        self.store.release_expired_leases(now)
        candidates = [
            record
            for record in self.store.list({JobStatus.QUEUED})
            if self._is_candidate(record, now)
        ]
        if not candidates:
            return []

        if self.confirmation_provider is not None:
            pair = self.confirmation_provider.confirmation_samples()
            if pair is None:
                return []
            first, second = pair
            first_now = self.clock()
            if not self._valid_cached_confirmation_pair(first, second, first_now):
                return []
            candidates = [record for record in candidates if self._is_candidate(record, first_now)]
            if not candidates:
                return []
        else:
            first = self.probe.snapshot()
            first_now = self.clock()
            if not self._usable(first, first_now):
                return []
            candidates = [record for record in candidates if self._is_candidate(record, first_now)]
            if not candidates:
                return []
            if self.settings.idle_confirmation_delay_seconds:
                self.sleeper(self.settings.idle_confirmation_delay_seconds)
            second = self.probe.snapshot()
        second_now = self.clock()
        # The first sample is historical confirmation; the latest sample must remain fresh.
        if not self._usable(second, second_now):
            return []
        candidates = [record for record in candidates if self._is_candidate(record, second_now)]
        if not candidates:
            return []

        active = self.store.list(set(ALLOCATED_STATUSES))
        reserved = reserved_gpu_indices(
            active, second.gpus, server_id=self.server_id, now=second_now
        )
        available = set(first.by_index()).intersection(second.by_index()).difference(reserved)
        decisions: list[LeaseDecision] = []
        for record in candidates:
            lease_now = self.clock()
            if not self._usable(second, lease_now):
                break
            if not self._is_candidate(record, lease_now):
                continue
            confirmed = self._confirmed_devices(first, second, lease_now)
            selected = self._select_ready(record, first, second, available, confirmed)
            if selected is None:
                continue
            if self.gpu_ready is not None:
                # CUDA checks can take seconds; lease only with still-fresh evidence.
                lease_now = self.clock()
                if not self._usable(second, lease_now):
                    break
                if not self._is_candidate(record, lease_now) or not set(selected).issubset(
                    self._confirmed_devices(first, second, lease_now)
                ):
                    continue
            try:
                leased = self.store.lease(
                    record.run_id,
                    server_id=self.server_id,
                    gpu_indices=selected,
                    gpu_uuids={index: second.by_index()[index].uuid for index in selected},
                    worker_id=self.worker_id,
                    ttl=timedelta(seconds=self.settings.lease_ttl_seconds),
                    expected_revision=record.revision,
                    now=lease_now,
                )
            except (ConcurrentUpdate, ResourceConflict):
                continue
            available.difference_update(selected)
            for index in selected:
                gpu = second.by_index()[index]
                self._consumed_samples[gpu.uuid] = gpu.sampled_at or second.sampled_at
            assert leased.allocation is not None
            decisions.append(
                LeaseDecision(
                    run_id=leased.run_id,
                    server_id=self.server_id,
                    gpu_indices=selected,
                    lease_token=leased.allocation.lease_token,
                )
            )
        return decisions

    def _valid_cached_confirmation_pair(
        self,
        first: ServerResourceSnapshot,
        second: ServerResourceSnapshot,
        now: datetime,
    ) -> bool:
        gap = (second.sampled_at - first.sampled_at).total_seconds()
        if not 0 < gap <= 45:
            return False
        if not first.healthy or first.error or first.server_id != self.server_id:
            return False
        return self._usable(second, now)

    def _confirmed_devices(
        self,
        first: ServerResourceSnapshot,
        second: ServerResourceSnapshot,
        now: datetime,
    ) -> set[int]:
        first_by_index = first.by_index()
        second_by_index = second.by_index()
        confirmed = set()
        for index in set(first_by_index).intersection(second_by_index):
            earlier, latest = first_by_index[index], second_by_index[index]
            if earlier.uuid != latest.uuid or earlier.error or latest.error:
                continue
            first_at = earlier.sampled_at or first.sampled_at
            second_at = latest.sampled_at or second.sampled_at
            if not 0 <= (now - second_at).total_seconds() <= self.settings.max_snapshot_age_seconds:
                continue
            if (
                self.confirmation_provider is not None
                or latest.sampled_at is not None
                or earlier.sampled_at is not None
            ):
                if not 2 <= (second_at - first_at).total_seconds() <= 45:
                    continue
                consumed = self._consumed_samples.get(latest.uuid)
                if consumed is not None and second_at <= consumed:
                    continue
            confirmed.add(index)
        return confirmed

    def _is_candidate(self, record: JobRecord, now: datetime) -> bool:
        return assess_schedule_window(record, server_id=self.server_id, now=now).eligible

    def _usable(self, snapshot: ServerResourceSnapshot, now: datetime) -> bool:
        return snapshot.is_usable(
            self.server_id,
            self.settings.max_snapshot_age_seconds,
            now,
        )

    def _select_ready(
        self,
        record: JobRecord,
        first: ServerResourceSnapshot,
        second: ServerResourceSnapshot,
        available: set[int],
        confirmed: set[int],
    ) -> list[int] | None:
        """Select GPUs, dropping any that fail the CUDA check and choosing again."""
        while True:
            selected = self._select(record, first, second, available.intersection(confirmed))
            if selected is None or self.gpu_ready is None:
                return selected
            gpus = second.by_index()
            unready = {index for index in selected if not self.gpu_ready(gpus[index])}
            if not unready:
                return selected
            available.difference_update(unready)

    @staticmethod
    def _select(
        record: JobRecord,
        first: ServerResourceSnapshot,
        second: ServerResourceSnapshot,
        available: set[int],
    ) -> list[int] | None:
        request = record.spec.resources
        first_by_index = first.by_index()
        second_by_index = second.by_index()
        eligible = [
            index
            for index in sorted(available)
            if first_by_index[index].can_allocate(request.memory_mib)
            and second_by_index[index].can_allocate(request.memory_mib)
        ]
        if request.gpus is not None:
            return list(request.gpus) if set(request.gpus).issubset(eligible) else None
        if len(eligible) < request.gpu_count:
            return None
        return eligible[: request.gpu_count]
