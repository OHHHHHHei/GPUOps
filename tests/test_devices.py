from __future__ import annotations

from pathlib import Path

import pytest

from gpuops.worker.devices import GpuDevice, discover_devices


def write_information(root: Path, directory: str, content: str) -> None:
    path = root / directory
    path.mkdir(parents=True)
    (path / "information").write_text(content, encoding="utf-8")


def test_discover_devices_reads_procfs_without_driver_queries(tmp_path: Path) -> None:
    write_information(
        tmp_path,
        "0000:81:00.0",
        "GPU UUID: GPU-B\nBus Location: 00000000:81:00.0\nDevice Minor: 6\n",
    )
    write_information(
        tmp_path,
        "0000:17:00.0",
        "GPU UUID: GPU-A\nBus Location: 00000000:17:00.0\nDevice Minor: 3\n",
    )
    write_information(tmp_path, "broken", "GPU UUID: GPU-C\nDevice Minor: nope\n")
    write_information(
        tmp_path,
        "unavailable",
        "GPU UUID: N/A\nBus Location: N/A\nDevice Minor: 7\n",
    )

    assert discover_devices(tmp_path) == [
        GpuDevice(index=3, uuid="GPU-A", pci_bus="00000000:17:00.0"),
        GpuDevice(index=6, uuid="GPU-B", pci_bus="00000000:81:00.0"),
    ]


def test_discover_devices_rejects_duplicate_procfs_identity(tmp_path: Path) -> None:
    body = "GPU UUID: GPU-A\nBus Location: 00000000:17:00.0\nDevice Minor: 3\n"
    write_information(tmp_path, "one", body)
    write_information(tmp_path, "two", body)

    with pytest.raises(ValueError, match="duplicate NVIDIA device"):
        discover_devices(tmp_path)
