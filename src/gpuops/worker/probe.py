"""Host-local GPU resource probe for the worker scheduler.

The per-device sampler runs this module in a short-lived subprocess every few
seconds, so it imports only the standard library. Callers that want the typed
snapshot read ``GpuProbeSample.snapshot``, which validates on access.
"""

from __future__ import annotations

import csv
import os
import pwd
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpuops.gpu_policy import (
    DEFAULT_IDLE_MEMORY_BASELINE_MIB,
    DEFAULT_IDLE_MEMORY_TOLERANCE_MIB,
    DEFAULT_IDLE_UTILIZATION_THRESHOLD,
    GPU_QUERY_TIMEOUT_SECONDS,
    validate_idle_limits,
)
from gpuops.worker.devices import GpuDevice
from gpuops.worker.process_identity import read_process_identity

if TYPE_CHECKING:
    from gpuops.core.resources import ServerResourceSnapshot

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class GpuProbeSample:
    snapshot_payload: dict[str, Any]
    data: dict[str, Any] | None
    error_code: str | None

    @property
    def snapshot(self) -> ServerResourceSnapshot:
        from gpuops.core.resources import ServerResourceSnapshot

        return ServerResourceSnapshot.model_validate(self.snapshot_payload)


class ProbeCommandError(RuntimeError):
    """A local GPU command was unavailable or returned unusable data."""


class ProbeBudgetExhausted(ProbeCommandError):
    """The shared sampling deadline was consumed."""


class DeviceUnavailable(ProbeCommandError):
    """The discovered PCI device is no longer readable."""


def check_pci_config(device: GpuDevice, sysfs_root: Path = Path("/sys/bus/pci/devices")) -> None:
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
    """Read one GPU's telemetry after checking its PCI identity.

    Production per-device sampling checks PCI config first, then scoped
    nvidia-smi telemetry. A GPU is idle only when it has no compute processes,
    low utilization, and used memory within the calibrated baseline plus
    tolerance. CUDA readiness is checked separately, just before leasing.
    """

    def __init__(
        self,
        server_id: str,
        *,
        runner: CommandRunner = subprocess.run,
        clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sample_timeout_seconds: float = GPU_QUERY_TIMEOUT_SECONDS,
        idle_utilization_threshold: float = DEFAULT_IDLE_UTILIZATION_THRESHOLD,
        idle_memory_baseline_mib: int = DEFAULT_IDLE_MEMORY_BASELINE_MIB,
        idle_memory_tolerance_mib: int = DEFAULT_IDLE_MEMORY_TOLERANCE_MIB,
        pci_checker: Callable[[GpuDevice], None] | None = None,
        proc_root: Path = Path("/proc"),
    ):
        if not server_id.strip():
            raise ValueError("server_id cannot be empty")
        if sample_timeout_seconds <= 0:
            raise ValueError("sample_timeout_seconds must be positive")
        validate_idle_limits(
            idle_utilization_threshold, idle_memory_baseline_mib, idle_memory_tolerance_mib
        )
        self.proc_root = proc_root
        self.server_id = server_id
        self.runner = runner
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.sample_timeout_seconds = sample_timeout_seconds
        self.idle_utilization_threshold = idle_utilization_threshold
        self.idle_memory_baseline_mib = idle_memory_baseline_mib
        self.idle_memory_tolerance_mib = idle_memory_tolerance_mib
        self.pci_checker = pci_checker if pci_checker is not None else check_pci_config

    def sample_device(self, device: GpuDevice) -> GpuProbeSample:
        """Sample one verified device without issuing any host-wide query."""

        deadline = self.monotonic_clock() + self.sample_timeout_seconds
        stage = f"pci[{device.pci_bus}]"
        try:
            self.pci_checker(device)
            self._require_budget(deadline)
            stage = f"nvidia-smi[{device.pci_bus}]"
            snapshot, data = self._snapshot_from_nvidia_device(device, deadline)
            self._require_budget(deadline)
            return GpuProbeSample(snapshot_payload=snapshot, data=data, error_code=None)
        except ProbeBudgetExhausted as error:
            error_code = "gpu_query_timeout"
            error_text = self._error_text(error)
        except DeviceUnavailable as error:
            error_code = "gpu_device_unavailable"
            error_text = self._error_text(error)
        except Exception as error:
            error_code = "gpu_query_failed"
            error_text = self._error_text(error)
        message = f"{stage}: {error_text}"
        return GpuProbeSample(
            snapshot_payload=self._unhealthy_snapshot([message]),
            data=None,
            error_code=error_code,
        )

    def _unhealthy_snapshot(self, errors: list[str]) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "sampled_at": self.clock(),
            "healthy": False,
            "error": "; ".join(errors),
            "gpus": [],
        }

    def _snapshot_from_nvidia_device(
        self, device: GpuDevice, deadline: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Read the processes first: while models load, an earlier managed-memory
        # observation leaves more of their reservation charged against the card.
        process_result = self._run(
            [
                "nvidia-smi",
                "-i",
                device.pci_bus,
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        processes = self._processes_on_device(device.uuid, process_result.stdout)
        group_memory = self._group_memory(processes)
        process_count = len(processes)
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

        sampled_at = self.clock()
        gpu = {
            "index": device.index,
            "uuid": device.uuid,
            "memory_total_mib": memory_total_mib,
            "memory_used_mib": memory_used_mib,
            "utilization_percent": utilization_value,
            "process_count": process_count,
            "group_memory": group_memory,
            "idle": self._is_idle(process_count, utilization_value, memory_used_mib),
            "sampled_at": sampled_at,
        }
        snapshot = {
            "server_id": self.server_id,
            "sampled_at": sampled_at,
            "healthy": True,
            "error": None,
            "gpus": [gpu],
        }
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
        return snapshot, {"gpus": [raw_gpu], "collected_at": sampled_at.isoformat()}

    def _is_idle(
        self,
        process_count: int,
        utilization_percent: float,
        memory_used_mib: int,
    ) -> bool:
        # Mirrors ResourceEligibilityPolicy.classify for already-validated values.
        return (
            process_count == 0
            and utilization_percent <= self.idle_utilization_threshold
            and memory_used_mib <= self.idle_memory_baseline_mib + self.idle_memory_tolerance_mib
        )

    def _run(self, command: Sequence[str], deadline: float) -> subprocess.CompletedProcess[str]:
        remaining = deadline - self.monotonic_clock()
        if remaining <= 0:
            raise ProbeBudgetExhausted("sampling deadline exhausted")
        try:
            completed = self.runner(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=remaining,
            )
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

    def _processes_on_device(self, uuid: str, output: str) -> list[dict[str, Any]]:
        processes: dict[int, dict[str, Any]] = {}
        for row in csv.reader(StringIO(output)):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 3:
                raise ProbeCommandError("returned an invalid process CSV row")
            gpu_uuid, pid_text, memory = (value.strip() for value in row)
            if gpu_uuid != uuid:
                raise ProbeCommandError("process query referenced a foreign GPU")
            if not pid_text.isdigit():
                raise ProbeCommandError("returned an invalid process PID")
            pid = int(pid_text)
            if pid in processes:
                continue
            process: dict[str, Any] = {
                "pid": pid,
                "command": "",
                "gpu_memory_usage": int(memory) if memory.isdigit() else None,
            }
            self._enrich_process(process, self.proc_root)
            processes[pid] = process
        return list(processes.values())

    def _group_memory(self, processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        totals: dict[int, int] = {}
        for process in processes:
            pgid, memory = process["pgrp"], process["gpu_memory_usage"]
            if pgid is not None and memory is not None:
                totals[pgid] = totals.get(pgid, 0) + memory

        groups = []
        for pgid, memory in totals.items():
            leader = read_process_identity(pgid, proc_root=self.proc_root)
            if leader and leader.pgid == pgid and leader.boot_id and leader.start_ticks is not None:
                groups.append(
                    {
                        "pgid": pgid,
                        "leader_start_ticks": leader.start_ticks,
                        "boot_id": leader.boot_id,
                        "memory_mib": memory,
                    }
                )
        return groups

    @staticmethod
    def _enrich_process(process: dict[str, Any], proc_root: Path = Path("/proc")) -> None:
        fallback = process.get("cmdline") or process.get("command")
        process["cmdline"] = fallback if isinstance(fallback, str) else ""
        for field in ("ppid", "pgrp", "started_at", "elapsed_seconds", "start_time_ticks"):
            process.setdefault(field, None)
        process.setdefault("username", None)
        try:
            raw_pid: Any = process.get("pid")
            pid = int(raw_pid)
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
