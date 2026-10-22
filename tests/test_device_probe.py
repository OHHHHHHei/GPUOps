from __future__ import annotations

import subprocess
from datetime import datetime, timezone

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
    def scoped_runner(command, **run_kwargs):
        if command[1:3] == ["-m", "gpuops.worker.cuda_check"]:
            assert run_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == DEVICE.uuid
            return completed(command, f"{DEVICE.uuid}\n")
        return runner(command, **run_kwargs)

    return LocalGpuProbe(
        server_id, runner=scoped_runner, pci_checker=lambda _device: None, **kwargs
    )


def test_sample_device_uses_only_scoped_queries_and_returns_one_gpu() -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == pytest.approx(6.0, abs=0.01)
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, "
                "NVIDIA A100, 42, 58.5, 250\n",
            )
        return completed(command, "")

    sample = device_probe(
        "gpu01", runner=runner, clock=lambda: NOW, sample_timeout_seconds=6.0
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
    assert not any(call[0] == "gpustat" for call in calls)


def test_sample_device_enriches_only_matching_processes() -> None:
    def runner(command, **kwargs):
        del kwargs
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 0000:81:00.0, 24576, 8000, 70, GPU, 45, 60, 250\n",
            )
        return completed(command, "GPU-ABCDE, 123\nGPU-ABCDE, 456\n")

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.snapshot.gpus[0].process_count == 2
    assert sample.data is not None
    assert [process["pid"] for process in sample.data["gpus"][0]["processes"]] == [123, 456]
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
        return completed(command, gpu_row)

    sample = device_probe("gpu01", runner=runner, clock=lambda: NOW).sample_device(DEVICE)

    assert sample.snapshot.healthy is False
    assert sample.snapshot.gpus == []
    assert sample.data is None
    assert sample.error_code == "gpu_query_failed"
    assert len(calls) == 1


def test_sample_device_rejects_foreign_process_uuid() -> None:
    def runner(command, **kwargs):
        del kwargs
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
            )
        return completed(command, "GPU-FOREIGN, 123\n")

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
            current[0] = 12.0
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
            )
        current[0] = 13.0
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


@pytest.mark.parametrize(
    ("cuda_result", "message"),
    [
        ((1, "", "cuInit failed with CUDA error 3"), "cuInit failed"),
        ((0, "GPU-FOREIGN\n", ""), "different device identity"),
    ],
)
def test_idle_device_requires_cuda_for_same_uuid(cuda_result, message) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            if "--query-gpu" in command[3]:
                return completed(
                    command,
                    "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
                )
            return completed(command, "")
        assert command[1:3] == ["-m", "gpuops.worker.cuda_check"]
        assert command[-1] == kwargs["env"]["CUDA_VISIBLE_DEVICES"] == DEVICE.uuid
        code, stdout, stderr = cuda_result
        return completed(command, stdout=stdout, stderr=stderr, code=code)

    sample = LocalGpuProbe(
        "gpu01", runner=runner, clock=lambda: NOW, pci_checker=lambda _device: None
    ).sample_device(DEVICE)

    assert sample.error_code == "gpu_cuda_unavailable"
    assert sample.snapshot.gpus == []
    assert message in (sample.snapshot.error or "")
    assert len(calls) == 3


def test_busy_device_skips_cuda_check() -> None:
    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        if "--query-gpu" in command[3]:
            return completed(
                command,
                "5, GPU-ABCDE, 00000000:81:00.0, 24576, 8000, 70, GPU, 45, 60, 250\n",
            )
        return completed(command, "GPU-ABCDE, 123\n")

    sample = LocalGpuProbe(
        "gpu01", runner=runner, clock=lambda: NOW, pci_checker=lambda _device: None
    ).sample_device(DEVICE)

    assert sample.error_code is None
    assert sample.snapshot.gpus[0].idle is False


def test_cuda_check_uses_remaining_shared_deadline() -> None:
    current = [10.0]
    timeouts = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if command[0] == "nvidia-smi":
            current[0] += 1.0
            if "--query-gpu" in command[3]:
                return completed(
                    command,
                    "5, GPU-ABCDE, 00000000:81:00.0, 24576, 318, 0, GPU, 45, 60, 250\n",
                )
            return completed(command, "")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    sample = LocalGpuProbe(
        "gpu01", runner=runner, clock=lambda: NOW,
        monotonic_clock=lambda: current[0], sample_timeout_seconds=6.0,
        pci_checker=lambda _device: None,
    ).sample_device(DEVICE)

    assert sample.error_code == "gpu_query_timeout"
    assert timeouts == pytest.approx([6.0, 5.0, 4.0])
