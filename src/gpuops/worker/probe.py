"""Host-local GPU resource probe for the worker scheduler."""

from __future__ import annotations

import csv
import json
import os
import pwd
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from math import isfinite
from pathlib import Path
from typing import Any

from gpuops.config import GPU_QUERY_TIMEOUT_SECONDS
from gpuops.core.resources import (
    DEFAULT_IDLE_MEMORY_BASELINE_MIB,
    DEFAULT_IDLE_MEMORY_TOLERANCE_MIB,
    DEFAULT_IDLE_UTILIZATION_THRESHOLD,
    GpuResource,
    GpuTelemetry,
    ResourceEligibilityPolicy,
    ServerResourceSnapshot,
)
from gpuops.models import utc_now
from gpuops.worker.devices import GpuDevice

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class GpuProbeSample:
    snapshot: ServerResourceSnapshot
    data: dict[str, Any] | None
    error_code: str | None


class ProbeCommandError(RuntimeError):
    """A local GPU command was unavailable or returned unusable data."""


class ProbeBudgetExhausted(ProbeCommandError):
    """The shared sampling deadline was consumed."""


class DeviceUnavailable(ProbeCommandError):
    """The discovered PCI device is no longer readable."""


class CudaUnavailable(ProbeCommandError):
    """An idle GPU cannot be initialized through CUDA."""


def check_pci_config(
    device: GpuDevice, sysfs_root: Path = Path("/sys/bus/pci/devices")
) -> None:
    """Read only the target device's PCI identity before touching NVIDIA's driver."""

    try:
        domain, bus, slot = device.pci_bus.lower().split(":")
        address = f"{int(domain, 16):04x}:{bus}:{slot}"
        if int(domain, 16) > 0xFFFF:
            raise ValueError("PCI domain is out of range")
        path = sysfs_root / address / "config"
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            identity = os.read(descriptor, 4)
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as error:
        raise DeviceUnavailable(
            f"PCI config is unreadable for {device.pci_bus}: {error}"
        ) from error
    if len(identity) != 4:
        raise DeviceUnavailable(f"PCI config is incomplete for {device.pci_bus}")
    vendor = int.from_bytes(identity[:2], "little")
    product = int.from_bytes(identity[2:], "little")
    if vendor != 0x10DE or product in {0, 0xFFFF}:
        raise DeviceUnavailable(
            f"PCI device is unavailable at {device.pci_bus} "
            f"(vendor={vendor:04x}, device={product:04x})"
        )


class LocalGpuProbe:
    """Read local GPU telemetry and check device readiness.

    Production per-device sampling checks PCI config first, then scoped
    nvidia-smi telemetry, then CUDA initialization for an idle GPU. The legacy
    host-wide sample prefers gpustat and falls back to nvidia-smi. A GPU is idle
    only when it has no compute processes, low utilization, and used memory
    within the calibrated baseline plus tolerance.
    """

    def __init__(
        self,
        server_id: str,
        *,
        runner: CommandRunner = subprocess.run,
        clock: Callable[[], datetime] = utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sample_timeout_seconds: float = GPU_QUERY_TIMEOUT_SECONDS,
        idle_utilization_threshold: float = DEFAULT_IDLE_UTILIZATION_THRESHOLD,
        idle_memory_baseline_mib: int = DEFAULT_IDLE_MEMORY_BASELINE_MIB,
        idle_memory_tolerance_mib: int = DEFAULT_IDLE_MEMORY_TOLERANCE_MIB,
        pci_checker: Callable[[GpuDevice], None] | None = None,
    ):
        if not server_id.strip():
            raise ValueError("server_id cannot be empty")
        if sample_timeout_seconds <= 0:
            raise ValueError("sample_timeout_seconds must be positive")
        policy = ResourceEligibilityPolicy(
            idle_utilization_threshold=idle_utilization_threshold,
            idle_memory_baseline_mib=idle_memory_baseline_mib,
            idle_memory_tolerance_mib=idle_memory_tolerance_mib,
        )
        self.server_id = server_id
        self.runner = runner
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.sample_timeout_seconds = sample_timeout_seconds
        self.idle_utilization_threshold = idle_utilization_threshold
        self.idle_memory_baseline_mib = idle_memory_baseline_mib
        self.idle_memory_tolerance_mib = idle_memory_tolerance_mib
        self.eligibility_policy = policy
        self.pci_checker = pci_checker if pci_checker is not None else check_pci_config

    def snapshot(self) -> ServerResourceSnapshot:
        return self.sample().snapshot

    def sample(self) -> GpuProbeSample:
        deadline = self.monotonic_clock() + self.sample_timeout_seconds
        errors: list[str] = []
        try:
            snapshot, data = self._snapshot_from_gpustat(deadline)
            self._require_budget(deadline)
            return GpuProbeSample(snapshot=snapshot, data=data, error_code=None)
        except ProbeBudgetExhausted as error:
            errors.append(f"gpustat: {self._error_text(error)}")
            return GpuProbeSample(
                snapshot=self._unhealthy_snapshot(errors),
                data=None,
                error_code="gpu_query_timeout",
            )
        except Exception as error:
            errors.append(f"gpustat: {self._error_text(error)}")

        try:
            snapshot, data = self._snapshot_from_nvidia_smi(deadline)
            self._require_budget(deadline)
            return GpuProbeSample(snapshot=snapshot, data=data, error_code=None)
        except ProbeBudgetExhausted as error:
            errors.append(f"nvidia-smi: {self._error_text(error)}")
            error_code = "gpu_query_timeout"
        except Exception as error:
            errors.append(f"nvidia-smi: {self._error_text(error)}")
            error_code = "gpu_query_failed"

        return GpuProbeSample(
            snapshot=self._unhealthy_snapshot(errors),
            data=None,
            error_code=error_code,
        )

    def sample_device(self, device: GpuDevice) -> GpuProbeSample:
        """Sample one verified device without issuing any host-wide query."""

        deadline = self.monotonic_clock() + self.sample_timeout_seconds
        stage = f"pci[{device.pci_bus}]"
        try:
            self.pci_checker(device)
            self._require_budget(deadline)
            stage = f"nvidia-smi[{device.pci_bus}]"
            snapshot, data = self._snapshot_from_nvidia_device(device, deadline)
            if snapshot.gpus[0].idle:
                stage = f"cuda[{device.uuid}]"
                self._check_cuda(device, deadline)
            self._require_budget(deadline)
            return GpuProbeSample(snapshot=snapshot, data=data, error_code=None)
        except ProbeBudgetExhausted as error:
            error_code = "gpu_query_timeout"
            error_text = self._error_text(error)
        except DeviceUnavailable as error:
            error_code = "gpu_device_unavailable"
            error_text = self._error_text(error)
        except CudaUnavailable as error:
            error_code = "gpu_cuda_unavailable"
            error_text = self._error_text(error)
        except Exception as error:
            error_code = "gpu_query_failed"
            error_text = self._error_text(error)
        message = f"{stage}: {error_text}"
        return GpuProbeSample(
            snapshot=self._unhealthy_snapshot([message]),
            data=None,
            error_code=error_code,
        )

    def _unhealthy_snapshot(self, errors: list[str]) -> ServerResourceSnapshot:
        return ServerResourceSnapshot(
            server_id=self.server_id,
            sampled_at=self.clock(),
            healthy=False,
            error="; ".join(errors),
            gpus=[],
        )

    def _snapshot_from_gpustat(
        self, deadline: float
    ) -> tuple[ServerResourceSnapshot, dict[str, Any]]:
        completed = self._run(["gpustat", "--json"], deadline)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ProbeCommandError("returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ProbeCommandError("returned an invalid JSON object")
        raw_gpus = payload.get("gpus")
        if not isinstance(raw_gpus, list) or not raw_gpus:
            raise ProbeCommandError("reported no GPUs")

        gpus: list[GpuResource] = []
        for raw_gpu in raw_gpus:
            if not isinstance(raw_gpu, dict):
                raise ProbeCommandError("returned an invalid GPU entry")
            processes = raw_gpu.get("processes")
            if not isinstance(processes, list):
                raise ProbeCommandError("did not report GPU processes")
            for process in processes:
                if isinstance(process, dict):
                    self._enrich_process(process)
            utilization = self._number(raw_gpu.get("utilization.gpu"), "utilization.gpu")
            process_count = len(processes)
            memory_used_mib = self._integer(raw_gpu.get("memory.used"), "memory.used")
            gpus.append(
                GpuResource(
                    index=self._integer(raw_gpu.get("index"), "index"),
                    uuid=self._text(raw_gpu.get("uuid"), "uuid"),
                    memory_total_mib=self._integer(raw_gpu.get("memory.total"), "memory.total"),
                    memory_used_mib=memory_used_mib,
                    utilization_percent=utilization,
                    process_count=process_count,
                    idle=self._is_idle(process_count, utilization, memory_used_mib),
                )
            )
        snapshot = self._healthy_snapshot(gpus)
        payload["collected_at"] = snapshot.sampled_at.isoformat()
        return snapshot, payload

    def _snapshot_from_nvidia_smi(
        self, deadline: float
    ) -> tuple[ServerResourceSnapshot, dict[str, Any]]:
        gpu_result = self._run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        process_result = self._run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        process_pids = self._parse_process_pids(process_result.stdout)
        gpus: list[GpuResource] = []
        raw_gpus: list[dict[str, Any]] = []
        for row in csv.reader(StringIO(gpu_result.stdout)):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise ProbeCommandError("returned an invalid GPU CSV row")
            index, uuid, total, used, utilization = (value.strip() for value in row)
            processes = [self._process_payload(pid) for pid in process_pids.get(uuid, [])]
            process_count = len(processes)
            utilization_value = self._number(utilization, "utilization.gpu")
            memory_used_mib = self._integer(used, "memory.used")
            gpus.append(
                GpuResource(
                    index=self._integer(index, "index"),
                    uuid=self._text(uuid, "uuid"),
                    memory_total_mib=self._integer(total, "memory.total"),
                    memory_used_mib=memory_used_mib,
                    utilization_percent=utilization_value,
                    process_count=process_count,
                    idle=self._is_idle(process_count, utilization_value, memory_used_mib),
                )
            )
            raw_gpus.append(
                {
                    "index": self._integer(index, "index"),
                    "uuid": self._text(uuid, "uuid"),
                    "name": "NVIDIA GPU",
                    "memory.total": self._integer(total, "memory.total"),
                    "memory.used": memory_used_mib,
                    "utilization.gpu": utilization_value,
                    "processes": processes,
                }
            )
        if not gpus:
            raise ProbeCommandError("reported no GPUs")
        unknown_uuids = set(process_pids).difference(gpu.uuid for gpu in gpus)
        if unknown_uuids:
            raise ProbeCommandError("process query referenced an unknown GPU")
        snapshot = self._healthy_snapshot(gpus)
        return snapshot, {"gpus": raw_gpus, "collected_at": snapshot.sampled_at.isoformat()}

    def _snapshot_from_nvidia_device(
        self, device: GpuDevice, deadline: float
    ) -> tuple[ServerResourceSnapshot, dict[str, Any]]:
        gpu_result = self._run(
            [
                "nvidia-smi",
                "-i",
                device.pci_bus,
                "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,"
                "utilization.gpu,name,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        rows = [
            [value.strip() for value in row]
            for row in csv.reader(StringIO(gpu_result.stdout))
            if row and any(value.strip() for value in row)
        ]
        if len(rows) != 1 or len(rows[0]) != 10:
            raise ProbeCommandError("returned an invalid scoped GPU CSV result")
        index, uuid, pci_bus, total, used, utilization, name, temperature, power, power_limit = (
            rows[0]
        )
        parsed_index = self._integer(index, "index")
        parsed_uuid = self._text(uuid, "uuid")
        parsed_pci = self._text(pci_bus, "pci.bus_id")
        if (
            parsed_index != device.index
            or parsed_uuid != device.uuid
            or self._normalize_pci(parsed_pci) != self._normalize_pci(device.pci_bus)
        ):
            raise ProbeCommandError("scoped GPU identity did not match discovery")
        memory_total_mib = self._integer(total, "memory.total")
        memory_used_mib = self._integer(used, "memory.used")
        utilization_value = self._number(utilization, "utilization.gpu")
        if memory_used_mib > memory_total_mib:
            raise ProbeCommandError("memory.used cannot exceed memory.total")
        if utilization_value > 100:
            raise ProbeCommandError("utilization.gpu cannot exceed 100")

        process_result = self._run(
            [
                "nvidia-smi",
                "-i",
                device.pci_bus,
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        process_pids = self._parse_process_pids(process_result.stdout)
        if set(process_pids).difference({device.uuid}):
            raise ProbeCommandError("process query referenced a foreign GPU")
        processes = [self._process_payload(pid) for pid in process_pids.get(device.uuid, [])]
        process_count = len(processes)
        sampled_at = self.clock()
        gpu = GpuResource(
            index=device.index,
            uuid=device.uuid,
            memory_total_mib=memory_total_mib,
            memory_used_mib=memory_used_mib,
            utilization_percent=utilization_value,
            process_count=process_count,
            idle=self._is_idle(process_count, utilization_value, memory_used_mib),
            sampled_at=sampled_at,
        )
        snapshot = ServerResourceSnapshot(
            server_id=self.server_id,
            sampled_at=sampled_at,
            healthy=True,
            gpus=[gpu],
        )
        raw_gpu = {
            "index": device.index,
            "uuid": device.uuid,
            "pci.bus_id": parsed_pci,
            "name": self._optional_text(name),
            "memory.total": memory_total_mib,
            "memory.used": memory_used_mib,
            "utilization.gpu": utilization_value,
            "temperature.gpu": self._optional_number(temperature),
            "power.draw": self._optional_number(power),
            "power.limit": self._optional_number(power_limit),
            "processes": processes,
        }
        return snapshot, {"gpus": [raw_gpu], "collected_at": snapshot.sampled_at.isoformat()}

    def _check_cuda(self, device: GpuDevice, deadline: float) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device.uuid
        try:
            completed = self._run(
                [sys.executable, "-m", "gpuops.worker.cuda_check", device.uuid],
                deadline,
                env=environment,
            )
        except ProbeBudgetExhausted:
            raise
        except ProbeCommandError as error:
            raise CudaUnavailable(str(error)) from error
        if completed.stdout.strip() != device.uuid:
            raise CudaUnavailable("CUDA returned a different device identity")

    def _is_idle(
        self,
        process_count: int,
        utilization_percent: float,
        memory_used_mib: int,
    ) -> bool:
        return self.eligibility_policy.is_allocatable(
            GpuTelemetry(
                process_count=process_count,
                utilization_percent=utilization_percent,
                memory_used_mib=memory_used_mib,
            )
        )

    def _healthy_snapshot(self, gpus: list[GpuResource]) -> ServerResourceSnapshot:
        return ServerResourceSnapshot(
            server_id=self.server_id,
            sampled_at=self.clock(),
            healthy=True,
            gpus=sorted(gpus, key=lambda gpu: gpu.index),
        )

    def _run(
        self,
        command: Sequence[str],
        deadline: float,
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        remaining = deadline - self.monotonic_clock()
        if remaining <= 0:
            raise ProbeBudgetExhausted("sampling deadline exhausted")
        try:
            kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": remaining,
            }
            if env is not None:
                kwargs["env"] = env
            completed = self.runner(list(command), **kwargs)
        except subprocess.TimeoutExpired as error:
            raise ProbeBudgetExhausted("sampling deadline exhausted") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise ProbeCommandError(str(error)) from error
        if self.monotonic_clock() >= deadline:
            raise ProbeBudgetExhausted("sampling deadline exhausted")
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "command failed").strip()
            raise ProbeCommandError(message[:300])
        return completed

    def _require_budget(self, deadline: float) -> None:
        if self.monotonic_clock() >= deadline:
            raise ProbeBudgetExhausted("sampling deadline exhausted")

    @staticmethod
    def _parse_process_pids(output: str) -> dict[str, list[int]]:
        processes: dict[str, list[int]] = {}
        for row in csv.reader(StringIO(output)):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ProbeCommandError("returned an invalid process CSV row")
            uuid, pid = (value.strip() for value in row)
            if not uuid or not pid.isdigit():
                raise ProbeCommandError("returned an invalid process entry")
            processes.setdefault(uuid, []).append(int(pid))
        return processes

    def _process_payload(self, pid: int) -> dict[str, Any]:
        process: dict[str, Any] = {"pid": pid, "command": ""}
        self._enrich_process(process)
        return process

    @staticmethod
    def _enrich_process(process: dict[str, Any], proc_root: Path = Path("/proc")) -> None:
        fallback = process.get("cmdline") or process.get("command")
        process["cmdline"] = fallback if isinstance(fallback, str) else ""
        for field in ("ppid", "pgrp", "started_at", "elapsed_seconds", "start_time_ticks"):
            process.setdefault(field, None)
        process.setdefault("username", None)
        try:
            pid = int(process.get("pid"))
            parts = [
                part
                for part in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            if parts:
                process["cmdline"] = " ".join(
                    part.decode("utf-8", errors="replace") for part in parts
                )
            stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            process["ppid"] = int(fields[1])
            process["pgrp"] = int(fields[2])
            start_ticks = int(fields[19])
            process["start_time_ticks"] = start_ticks
            uptime = float((proc_root / "uptime").read_text(encoding="utf-8").split()[0])
            elapsed = max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
            process["elapsed_seconds"] = round(elapsed, 3)
            process["started_at"] = datetime.fromtimestamp(
                time.time() - elapsed, tz=timezone.utc
            ).isoformat()
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
            OSError,
            IndexError,
        ):
            pass
        with suppress(
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
            OSError,
            KeyError,
        ):
            uid = (proc_root / str(process.get("pid"))).stat().st_uid
            process["username"] = pwd.getpwuid(uid).pw_name

    @staticmethod
    def _integer(value: Any, field: str) -> int:
        number = LocalGpuProbe._number(value, field)
        if not number.is_integer():
            raise ProbeCommandError(f"{field} must be an integer")
        return int(number)

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if value is None or isinstance(value, bool):
            raise ProbeCommandError(f"{field} is unavailable")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ProbeCommandError(f"{field} is invalid") from error
        if number < 0:
            raise ProbeCommandError(f"{field} cannot be negative")
        return number

    @staticmethod
    def _text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ProbeCommandError(f"{field} is unavailable")
        return value.strip()

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text or text.lower() in {"n/a", "[not supported]", "not supported"}:
            return None
        return text

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        text = LocalGpuProbe._optional_text(value)
        if text is None:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
        return number if isfinite(number) and number >= 0 else None

    @staticmethod
    def _normalize_pci(value: str) -> str:
        parts = value.strip().lower().split(":")
        if len(parts) == 3:
            domain, bus, slot = parts
            return f"{domain.zfill(8)}:{bus.zfill(2)}:{slot}"
        return value.strip().lower()

    @staticmethod
    def _error_text(error: Exception) -> str:
        message = str(error).strip() or error.__class__.__name__
        return message.replace("\n", " ")[:300]
