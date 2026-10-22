"""Nonblocking, per-device GPU sampling and fail-closed cache access."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpuops.core.resources import GpuResource, ServerResourceSnapshot
from gpuops.models import utc_now
from gpuops.worker.devices import GpuDevice, discover_devices
from gpuops.worker.runtime import _atomic_write_json

CACHE_NAME = "gpu-sample.json"
LOCK_NAME = "gpu-sample.lock"
FRESH_MAX_AGE_SECONDS = 15.0
CONFIRMATION_MIN_GAP_SECONDS = 2.0
CONFIRMATION_MAX_GAP_SECONDS = 45.0
SUCCESS_INTERVAL_SECONDS = 5.0
FAILURE_BACKOFF_MAX_SECONDS = 60.0
PARENT_SAMPLE_TIMEOUT_SECONDS = 7.0
TIMEOUT_BACKOFF_INITIAL_SECONDS = 60.0
TIMEOUT_BACKOFF_MAX_SECONDS = 300.0
DISCOVERY_INTERVAL_SECONDS = 30.0
MAX_JSON_BYTES = 2 * 1024 * 1024


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        raw = os.read(descriptor, MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            return None
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        os.close(descriptor)
    return value if isinstance(value, dict) else None


def _validated_current_snapshot(
    envelope: dict[str, Any], expected_server: str
) -> ServerResourceSnapshot | None:
    try:
        snapshot = ServerResourceSnapshot.model_validate(envelope["snapshot"])
        data = envelope["data"]
        error_code = envelope["error_code"]
    except (KeyError, ValueError, TypeError):
        return None
    if snapshot.server_id != expected_server:
        return None
    if not snapshot.healthy:
        return snapshot if data is None and isinstance(error_code, str) else None
    if error_code is not None or not isinstance(data, dict):
        return None
    if not isinstance(data.get("gpus"), list):
        return None
    collected_at = data.get("collected_at")
    if not isinstance(collected_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed != snapshot.sampled_at:
        return None
    return snapshot


@dataclass(slots=True)
class _DeviceState:
    device: GpuDevice
    process: subprocess.Popen[bytes] | None = None
    started_at: float | None = None
    timed_out: bool = False
    attempt_id: str | None = None
    next_attempt_at: float = 0.0
    failure_backoff: float = SUCCESS_INTERVAL_SECONDS
    timeout_backoff: float = TIMEOUT_BACKOFF_INITIAL_SECONDS
    latest: dict[str, Any] | None = None


class AsyncGpuSampler:
    """Advance independent one-shot samplers without blocking the worker loop."""

    def __init__(
        self,
        state_directory: Path,
        server_id: str,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        clock: Callable[[], datetime] = utc_now,
        device_discovery: Callable[[], list[GpuDevice]] = discover_devices,
    ) -> None:
        if not server_id.strip():
            raise ValueError("server_id cannot be empty")
        self.state_directory = state_directory.expanduser()
        self.server_id = server_id
        self.cache_path = self.state_directory / CACHE_NAME
        self.lock_path = self.state_directory / LOCK_NAME
        self.monotonic_clock = monotonic_clock
        self.clock = clock
        self.device_discovery = device_discovery
        self._devices: dict[str, _DeviceState] = {}
        self._retired_processes: list[subprocess.Popen[bytes]] = []
        self._next_discovery_at = 0.0
        self._inventory_error: str | None = None
        self._first_tick = True

    @property
    def processes(self) -> tuple[subprocess.Popen[bytes], ...]:
        return tuple(
            state.process for state in self._devices.values() if state.process is not None
        )

    def tick(self) -> None:
        self.state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._first_tick:
            self.cache_path.unlink(missing_ok=True)
            self._first_tick = False
        now = self.monotonic_clock()
        if now >= self._next_discovery_at:
            self._refresh_inventory(now)
        self._reap_retired_processes()
        for state in tuple(self._devices.values()):
            self._poll(state, now)
        due_retries = []
        for state in tuple(self._devices.values()):
            if state.process is not None or now < state.next_attempt_at:
                continue
            if state.latest is not None and state.latest.get("error_code") is not None:
                due_retries.append(state)
            else:
                self._start_if_owner(state, now)
        for state in sorted(
            due_retries,
            key=lambda item: (item.next_attempt_at, item.device.index),
        ):
            if self._start_if_owner(state, now):
                break
        self._publish()

    def close(self) -> None:
        for state in self._devices.values():
            state.process = None
            state.started_at = None
            state.attempt_id = None
        self._retired_processes.clear()

    def _refresh_inventory(self, now: float) -> None:
        self._next_discovery_at = now + DISCOVERY_INTERVAL_SECONDS
        try:
            devices = self.device_discovery()
            if not isinstance(devices, list):
                raise TypeError("device discovery returned invalid inventory")
        except Exception as error:
            self._inventory_error = str(error).strip() or error.__class__.__name__
            self._retire_states(self._devices.values())
            self._devices.clear()
            return
        if not devices:
            self._inventory_error = "no NVIDIA devices discovered"
            self._retire_states(self._devices.values())
            self._devices.clear()
            return
        self._inventory_error = None
        old = self._devices
        updated: dict[str, _DeviceState] = {}
        for device in devices:
            key = self._device_key(device)
            state = old.get(key)
            if state is None or state.device != device:
                state = _DeviceState(device=device)
            updated[key] = state
        self._retire_states(
            state
            for state in old.values()
            if updated.get(self._device_key(state.device)) is not state
        )
        self._devices = updated

    def _retire_states(self, states: Iterable[_DeviceState]) -> None:
        for state in states:
            if state.process is not None:
                self._retired_processes.append(state.process)

    def _reap_retired_processes(self) -> None:
        self._retired_processes = [
            process for process in self._retired_processes if process.poll() is None
        ]

    def _poll(self, state: _DeviceState, now: float) -> None:
        process = state.process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            if (
                not state.timed_out
                and state.started_at is not None
                and now - state.started_at >= PARENT_SAMPLE_TIMEOUT_SECONDS
            ):
                state.timed_out = True
                state.latest = self._device_error(
                    state.device, "gpu_query_timeout", "GPU sampling deadline exhausted"
                )
            return
        state.process = None
        state.started_at = None
        if state.timed_out:
            state.timed_out = False
            state.attempt_id = None
            self._schedule_timeout(state, now)
            return
        result = self._consume_result(state) if return_code == 0 else None
        if result is None:
            state.latest = self._device_error(
                state.device, "gpu_query_failed", f"GPU sampler child exited {return_code}"
            )
            self._schedule_failure(state, now)
        elif result.get("error_code") is None:
            state.latest = result
            state.failure_backoff = SUCCESS_INTERVAL_SECONDS
            state.timeout_backoff = TIMEOUT_BACKOFF_INITIAL_SECONDS
            state.next_attempt_at = now + SUCCESS_INTERVAL_SECONDS
        else:
            state.latest = result
            if result.get("error_code") == "gpu_query_timeout":
                self._schedule_timeout(state, now)
            else:
                self._schedule_failure(state, now)
        state.attempt_id = None

    def _start_if_owner(self, state: _DeviceState, now: float) -> bool:
        # Probe the legacy barrier only; per-device children must never inherit it.
        barrier_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(barrier_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        finally:
            os.close(barrier_fd)
        lock_fd = os.open(self._device_lock_path(state.device), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            result_path = self._result_path(state.device)
            result_path.unlink(missing_ok=True)
            attempt_id = uuid.uuid4().hex
            argv = [
                sys.executable,
                "-m",
                "gpuops.worker.sampling_child",
                "--server-id", self.server_id,
                "--result", str(result_path),
                "--lock-fd", str(lock_fd),
                "--attempt-id", attempt_id,
                "--index", str(state.device.index),
                "--uuid", state.device.uuid,
                "--pci-bus", state.device.pci_bus,
            ]
            try:
                state.process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(lock_fd,),
                    start_new_session=True,
                )
            except OSError as error:
                state.latest = self._device_error(
                    state.device,
                    "gpu_query_failed",
                    f"cannot start GPU sampler: {str(error).strip() or error.__class__.__name__}",
                )
                self._schedule_failure(state, now)
                return True
            state.started_at = now
            state.timed_out = False
            state.attempt_id = attempt_id
            return True
        finally:
            os.close(lock_fd)

    def _consume_result(self, state: _DeviceState) -> dict[str, Any] | None:
        raw = _read_json_object(self._result_path(state.device))
        if raw is None or state.attempt_id is None or raw.get("attempt_id") != state.attempt_id:
            return None
        try:
            device = GpuDevice(**raw["device"])
        except (KeyError, TypeError, ValueError):
            return None
        if device != state.device:
            return None
        if raw.get("error_code") is None:
            try:
                snapshot = ServerResourceSnapshot.model_validate(raw["snapshot"])
                data = raw["data"]
            except (KeyError, TypeError, ValueError):
                return None
            observed_at = self.clock()
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            if (
                not snapshot.healthy
                or snapshot.server_id != self.server_id
                or len(snapshot.gpus) != 1
                or snapshot.gpus[0].index != device.index
                or snapshot.gpus[0].uuid != device.uuid
                or snapshot.gpus[0].sampled_at is None
                or snapshot.gpus[0].sampled_at != snapshot.sampled_at
                or snapshot.sampled_at > observed_at
                or not isinstance(data, dict)
                or not isinstance(data.get("gpus"), list)
                or len(data["gpus"]) != 1
                or data.get("collected_at") != snapshot.sampled_at.isoformat()
            ):
                return None
            raw_gpu = data["gpus"][0]
            if (
                not isinstance(raw_gpu, dict)
                or raw_gpu.get("index") != device.index
                or raw_gpu.get("uuid") != device.uuid
            ):
                return None
        else:
            if raw.get("error_code") not in {
                "gpu_query_failed", "gpu_query_timeout",
                "gpu_device_unavailable", "gpu_cuda_unavailable",
            }:
                return None
            if not isinstance(raw.get("error"), str):
                return None
            try:
                failed = ServerResourceSnapshot.model_validate(raw["snapshot"])
            except (KeyError, TypeError, ValueError):
                return None
            if failed.healthy or failed.server_id != self.server_id or failed.gpus:
                return None
            raw["sampled_at"] = failed.sampled_at.isoformat()
        return raw

    def _publish(self) -> None:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self._inventory_error is not None:
            snapshot = ServerResourceSnapshot(
                server_id=self.server_id,
                sampled_at=now,
                healthy=False,
                error=f"GPU discovery failed: {self._inventory_error}",
                gpus=[],
            )
            _atomic_write_json(self.cache_path, {
                "snapshot": snapshot.model_dump(mode="json"), "previous": None,
                "data": None, "error_code": "gpu_discovery_failed",
            })
            return
        gpus: list[GpuResource] = []
        raw_gpus: list[dict[str, Any]] = []
        for state in sorted(self._devices.values(), key=lambda item: item.device.index):
            latest = state.latest
            if latest is not None and latest.get("error_code") is None:
                sample = ServerResourceSnapshot.model_validate(latest["snapshot"])
                gpu = sample.gpus[0]
                gpus.append(gpu)
                raw_gpu = dict(latest["data"]["gpus"][0])
                raw_gpu.update({
                    "sampled_at": gpu.sampled_at.isoformat() if gpu.sampled_at else None,
                    "error": None, "error_code": None,
                })
                raw_gpus.append(raw_gpu)
                continue
            error = (
                str(latest.get("error"))
                if latest is not None
                else "GPU sample has not completed"
            )
            code = (
                str(latest.get("error_code"))
                if latest is not None
                else "gpu_sample_pending"
            )
            gpus.append(GpuResource(
                index=state.device.index, uuid=state.device.uuid, idle=False,
                sampled_at=self._error_sampled_at(latest), error=error,
            ))
            raw_gpus.append({
                "index": state.device.index, "uuid": state.device.uuid,
                "pci.bus_id": state.device.pci_bus, "memory.total": None,
                "memory.used": None, "utilization.gpu": None, "processes": None,
                "sampled_at": latest.get("sampled_at") if latest is not None else None,
                "error": error, "error_code": code,
            })
        snapshot = ServerResourceSnapshot(
            server_id=self.server_id, sampled_at=now, healthy=True, error=None, gpus=gpus
        )
        old = self._read_cache()
        previous = None
        if old is not None:
            old_snapshot = _validated_current_snapshot(old, self.server_id)
            if old_snapshot is not None and old_snapshot.healthy:
                try:
                    older = ServerResourceSnapshot.model_validate(old["previous"])
                except (KeyError, ValueError, TypeError):
                    older = None
                older_by_index = (
                    older.by_index()
                    if older is not None and older.healthy and older.server_id == self.server_id
                    else {}
                )
                old_by_index = old_snapshot.by_index()
                previous_gpus = []
                for gpu in gpus:
                    prior = old_by_index.get(gpu.index)
                    if prior is None or prior.uuid != gpu.uuid:
                        previous_gpus.append(gpu)
                    elif prior.sampled_at != gpu.sampled_at:
                        previous_gpus.append(prior)
                    else:
                        earlier = older_by_index.get(gpu.index)
                        previous_gpus.append(
                            earlier if earlier is not None and earlier.uuid == gpu.uuid else prior
                        )
                previous = old_snapshot.model_copy(update={"gpus": previous_gpus}).model_dump(
                    mode="json"
                )
        _atomic_write_json(self.cache_path, {
            "snapshot": snapshot.model_dump(mode="json"), "previous": previous,
            "data": {"gpus": raw_gpus, "collected_at": now.isoformat()}, "error_code": None,
        })

    def _read_cache(self) -> dict[str, Any] | None:
        return _read_json_object(self.cache_path)

    def _schedule_failure(self, state: _DeviceState, now: float) -> None:
        state.next_attempt_at = now + state.failure_backoff
        state.failure_backoff = min(state.failure_backoff * 2, FAILURE_BACKOFF_MAX_SECONDS)

    def _schedule_timeout(self, state: _DeviceState, now: float) -> None:
        state.next_attempt_at = now + state.timeout_backoff
        state.timeout_backoff = min(
            state.timeout_backoff * 2, TIMEOUT_BACKOFF_MAX_SECONDS
        )

    @staticmethod
    def _device_key(device: GpuDevice) -> str:
        value = device.pci_bus.strip().lower()
        parts = value.split(":")
        if len(parts) == 3:
            domain, bus, slot = parts
            return f"{domain.zfill(8)}:{bus.zfill(2)}:{slot}"
        return value

    def _path_token(self, device: GpuDevice) -> str:
        return hashlib.sha256(self._device_key(device).encode()).hexdigest()[:20]

    def _device_lock_path(self, device: GpuDevice) -> Path:
        return self.state_directory / f"gpu-sample-{self._path_token(device)}.lock"

    def _result_path(self, device: GpuDevice) -> Path:
        return self.state_directory / f"gpu-sample-{self._path_token(device)}.result.json"

    def _device_error(self, device: GpuDevice, code: str, message: str) -> dict[str, Any]:
        sampled_at = self.clock()
        if sampled_at.tzinfo is None:
            sampled_at = sampled_at.replace(tzinfo=timezone.utc)
        return {
            "device": {"index": device.index, "uuid": device.uuid, "pci_bus": device.pci_bus},
            "sampled_at": sampled_at.isoformat(), "error": message, "error_code": code,
        }

    @staticmethod
    def _error_sampled_at(latest: dict[str, Any] | None) -> datetime | None:
        if latest is None or not isinstance(latest.get("sampled_at"), str):
            return None
        try:
            value = datetime.fromisoformat(latest["sampled_at"].replace("Z", "+00:00"))
        except ValueError:
            return None
        return value if value.tzinfo is not None else None


class CachedGpuProbe:
    """Read validated scheduler snapshots from the shared sampler cache."""

    def __init__(
        self,
        state_directory: Path,
        server_id: str,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.cache_path = state_directory.expanduser() / CACHE_NAME
        self.server_id = server_id
        self.clock = clock

    def snapshot(self) -> ServerResourceSnapshot:
        now = self.clock()
        cache = self._read_cache()
        if cache is not None:
            snapshot = _validated_current_snapshot(cache, self.server_id)
            if snapshot is not None:
                age = (
                    now.astimezone(timezone.utc)
                    - snapshot.sampled_at.astimezone(timezone.utc)
                ).total_seconds()
                if 0 <= age <= FRESH_MAX_AGE_SECONDS:
                    return snapshot.with_fresh_gpus(now, FRESH_MAX_AGE_SECONDS)
        return self._unhealthy(now, "GPU sample cache is unavailable, invalid, or stale")

    def confirmation_samples(self) -> tuple[ServerResourceSnapshot, ServerResourceSnapshot] | None:
        now = self.clock()
        cache = self._read_cache()
        if cache is None:
            return None
        latest = _validated_current_snapshot(cache, self.server_id)
        if latest is None or not latest.healthy:
            return None
        try:
            previous = ServerResourceSnapshot.model_validate(cache["previous"])
        except (KeyError, ValueError, TypeError):
            return None
        latest = latest.with_fresh_gpus(now, FRESH_MAX_AGE_SECONDS)
        if not latest.is_usable(self.server_id, FRESH_MAX_AGE_SECONDS, now):
            return None
        if not previous.healthy or previous.error or previous.server_id != self.server_id:
            return None
        gap = (latest.sampled_at - previous.sampled_at).total_seconds()
        per_device_timestamps = all(
            gpu.sampled_at is not None for gpu in [*previous.gpus, *latest.gpus]
        )
        minimum_gap = 0.0 if per_device_timestamps else CONFIRMATION_MIN_GAP_SECONDS
        if not minimum_gap < gap <= CONFIRMATION_MAX_GAP_SECONDS:
            return None
        return previous, latest

    def _read_cache(self) -> dict[str, Any] | None:
        return _read_json_object(self.cache_path)

    def _unhealthy(self, now: datetime, message: str) -> ServerResourceSnapshot:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return ServerResourceSnapshot(
            server_id=self.server_id, sampled_at=now, healthy=False, error=message, gpus=[]
        )
