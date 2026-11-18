from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpuops.worker.devices import GpuDevice
from gpuops.worker.probe import (
    DeviceUnavailable,
    LocalGpuProbe,
    check_pci_config,
)

NOW = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)
DEVICE = GpuDevice(index=5, uuid="GPU-ABCDE", pci_bus="00000000:81:00.0")


def completed(command: list[str], stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def device_probe(server_id: str, *, runner, **kwargs) -> LocalGpuProbe:
    return LocalGpuProbe(server_id, runner=runner, pci_checker=lambda _device: None, **kwargs)


def test_sample_device_uses_only_scoped_queries_and_returns_one_gpu() -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 6.0
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, NVIDIA A100, 42, 58.5, 250\n",
            )
        return completed(command, "")

    sample = device_probe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        monotonic_clock=lambda: 10.0,
        sample_timeout_seconds=6.0,
    ).sample_device(DEVICE)

    assert sample.snapshot.healthy is True
    assert len(sample.snapshot.gpus) == 1
    assert sample.snapshot.gpus[0].index == 5
    assert sample.snapshot.gpus[0].sampled_at == sample.snapshot.sampled_at == NOW
    assert sample.data is not None
    assert len(sample.data["gpus"]) == 1
    assert sample.data["gpus"][0]["name"] == "NVIDIA A100"
    assert sample.data["gpus"][0]["temperature.gpu"] == 42
    assert sample.data["gpus"][0]["power.draw"] == 58.5
    assert sample.data["gpus"][0]["power.limit"] == 250
    assert all(call[:3] == ["nvidia-smi", "-i", DEVICE.pci_bus] for call in calls)
    assert len(calls) == 2
    assert calls[0][3] == "--query-compute-apps=gpu_uuid,pid,used_gpu_memory"
    assert calls[1][3].startswith("--query-gpu=")


def test_sample_records_deduplicated_process_memory_by_leader_identity(tmp_path: Path) -> None:
    (tmp_path / "sys/kernel/random").mkdir(parents=True)
    (tmp_path / "sys/kernel/random/boot_id").write_text("boot-A")
    (tmp_path / "uptime").write_text("100 0")
    for pid, group, ticks in [
        (123, 123, 15),
        (456, 123, 25),
        (500, 123, 35),
        (770, 700, 40),  # Missing group leader.
        (800, 800, 60),  # Another job under the same user.
    ]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"job")
        fields = ["S", "1", str(group)] + ["0"] * 16 + [str(ticks)]
        (directory / "stat").write_text(f"{pid} (job) {' '.join(fields)}")

    def runner(command, **kwargs):
        del kwargs
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 0000:81:00.0, 24576, 8000, 70, GPU, 45, 60, 250\n",
            )
        return completed(
            command,
            "GPU-ABCDE, 123, 1000\nGPU-ABCDE, 456, 2000\nGPU-ABCDE, 456, 2000\n"
            "GPU-ABCDE, 500, N/A\nGPU-ABCDE, 770, 100\nGPU-ABCDE, 800, 300\n",
        )

    sample = device_probe(
        "gpu01", runner=runner, clock=lambda: NOW, proc_root=tmp_path
    ).sample_device(DEVICE)

    gpu = sample.snapshot.gpus[0]
    assert gpu.process_count == 5
    assert [group.model_dump() for group in gpu.group_memory] == [
        {"pgid": 123, "leader_start_ticks": 15, "boot_id": "boot-A", "memory_mib": 3000},
        {"pgid": 800, "leader_start_ticks": 60, "boot_id": "boot-A", "memory_mib": 300},
    ]
    assert sample.data is not None
    assert [p["gpu_memory_usage"] for p in sample.data["gpus"][0]["processes"]] == [
        1000,
        2000,
        None,
        100,
        300,
    ]
    assert all("username" in process for process in sample.data["gpus"][0]["processes"])


@pytest.mark.parametrize(
    "gpu_row",
    [
        "4, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
        "5, GPU-FOREIGN, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
        "5, GPU-ABCDE, 00000000:82:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
        "5, GPU-ABCDE, 00000000:81:00.0, N/A, 318, 0, GPU, 45, 60, 250\n",
        "",
    ],
)
def test_sample_device_rejects_identity_mismatch_and_incomplete_rows(gpu_row: str) -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        return completed(command, gpu_row if "--query-gpu" in command[3] else "")

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.snapshot.healthy is False
    assert sample.snapshot.gpus == []
    assert sample.data is None
    assert sample.error_code == "gpu_query_failed"
    assert len(calls) == 2


def test_sample_device_rejects_foreign_process_uuid() -> None:
    def runner(command, **kwargs):
        del kwargs
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
            )
        return completed(command, "GPU-FOREIGN, 123, 8\n")

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.snapshot.healthy is False
    assert sample.data is None
    assert "foreign GPU" in (sample.snapshot.error or "")


def test_sample_device_failure_never_invokes_host_query() -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 6.0)

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.error_code == "gpu_query_timeout"
    assert sample.snapshot.healthy is False
    assert sample.snapshot.gpus == []
    assert sample.data is None
    assert len(calls) == 1
    assert calls[0][:3] == ["nvidia-smi", "-i", DEVICE.pci_bus]


def test_sample_device_shares_deadline_across_its_two_scoped_queries() -> None:
    current = [10.0]
    timeouts: list[float] = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if "--query-gpu" in command[3]:
            current[0] = 13.0
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
            )
        current[0] = 12.0
        return completed(command, "")

    sample = device_probe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        monotonic_clock=lambda: current[0],
        sample_timeout_seconds=6.0,
    ).sample_device(DEVICE)

    assert sample.snapshot.healthy is True
    assert timeouts == pytest.approx([6.0, 4.0])


def test_sample_device_tolerates_unavailable_optional_display_fields() -> None:
    def runner(command, **kwargs):
        del kwargs
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, "
                "N/A, N/A, [Not Supported], not supported\n",
            )
        return completed(command, "")

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.snapshot.healthy is True
    assert sample.error_code is None
    assert sample.data is not None
    raw = sample.data["gpus"][0]
    assert raw["name"] is None
    assert raw["temperature.gpu"] is None
    assert raw["power.draw"] is None
    assert raw["power.limit"] is None


@pytest.mark.parametrize(
    ("identity", "available"),
    [
        (bytes.fromhex("de10a110"), True),
        (bytes.fromhex("ffffa110"), False),
        (bytes.fromhex("de10ffff"), False),
        (bytes.fromhex("00000000"), False),
        (bytes.fromhex("de10"), False),
    ],
)
def test_pci_guard_checks_only_target_config(tmp_path, identity: bytes, available: bool) -> None:
    directory = tmp_path / "0000:81:00.0"
    directory.mkdir()
    (directory / "config").write_bytes(identity)
    if available:
        check_pci_config(DEVICE, tmp_path)
    else:
        with pytest.raises(DeviceUnavailable):
            check_pci_config(DEVICE, tmp_path)


def test_pci_guard_prevents_driver_query_when_device_is_missing(tmp_path) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        raise AssertionError("driver must not be queried")

    sample = LocalGpuProbe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        pci_checker=lambda device: check_pci_config(device, tmp_path),
    ).sample_device(DEVICE)

    assert sample.error_code == "gpu_device_unavailable"
    assert sample.snapshot.healthy is False
    assert "PCI config is unreadable" in (sample.snapshot.error or "")
    assert calls == []


def test_idle_device_sampling_runs_only_scoped_nvidia_queries() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command[0])
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
            )
        return completed(command, "")

    sample = LocalGpuProbe(
        "gpu01", runner=runner, clock=lambda: NOW, pci_checker=lambda _device: None
    ).sample_device(DEVICE)

    # CUDA is checked just before leasing, not on every sample of an idle GPU.
    assert sample.error_code is None
    assert sample.snapshot.gpus[0].idle is True
    assert calls == ["nvidia-smi", "nvidia-smi"]


def test_busy_device_skips_cuda_check() -> None:
    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 8000, 70, GPU, 45, 60, 250\n",
            )
        return completed(command, "GPU-ABCDE, 123, 8000\n")

    sample = LocalGpuProbe(
        "gpu01", runner=runner, clock=lambda: NOW, pci_checker=lambda _device: None
    ).sample_device(DEVICE)

    assert sample.error_code is None
    assert sample.snapshot.gpus[0].idle is False


def test_process_query_that_exhausts_the_budget_times_out() -> None:
    current = [10.0]

    def runner(command, **kwargs):
        if "--query-gpu" in command[3]:
            current[0] += 1.0
            return completed(
                command, "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n"
            )
        current[0] += 6.0
        return completed(command, "")

    sample = device_probe(
        "gpu01",
        runner=runner,
        clock=lambda: NOW,
        monotonic_clock=lambda: current[0],
        sample_timeout_seconds=6.0,
    ).sample_device(DEVICE)

    assert sample.error_code == "gpu_query_timeout"
    assert sample.data is None
