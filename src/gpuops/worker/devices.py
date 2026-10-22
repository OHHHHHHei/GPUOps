"""Discover NVIDIA devices without relying on a host-wide GPU query."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PCI_BUS_PATTERN = re.compile(
    r"(?:[0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]"
)


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    uuid: str
    pci_bus: str


def discover_devices(
    proc_root: Path = Path("/proc/driver/nvidia/gpus"),
) -> list[GpuDevice]:
    """Return procfs device candidates without invoking potentially wedged drivers."""

    try:
        information_files = sorted(proc_root.glob("*/information"))
    except OSError:
        return []

    devices: list[GpuDevice] = []
    for information_path in information_files:
        candidate = _read_candidate(information_path)
        if candidate is None:
            continue
        minor, proc_uuid, proc_pci = candidate
        devices.append(GpuDevice(index=minor, uuid=proc_uuid, pci_bus=proc_pci))

    _reject_duplicates(devices)
    return sorted(devices, key=lambda device: device.index)


def _read_candidate(path: Path) -> tuple[int, str, str] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if separator:
            fields[name.strip()] = value.strip()
    uuid = fields.get("GPU UUID", "")
    pci_bus = fields.get("Bus Location", "")
    try:
        minor = int(fields["Device Minor"])
    except (KeyError, ValueError):
        return None
    if (
        minor < 0
        or not uuid.startswith("GPU-")
        or PCI_BUS_PATTERN.fullmatch(pci_bus) is None
    ):
        return None
    return minor, uuid, pci_bus


def _normalize_pci(value: str) -> str:
    parts = value.strip().lower().split(":")
    if len(parts) == 3:
        domain, bus, slot = parts
        return f"{domain.zfill(8)}:{bus.zfill(2)}:{slot}"
    return value.strip().lower()


def _reject_duplicates(devices: list[GpuDevice]) -> None:
    for field in ("index", "uuid", "pci_bus"):
        values = [
            _normalize_pci(device.pci_bus) if field == "pci_bus" else getattr(device, field)
            for device in devices
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate NVIDIA device {field}")
