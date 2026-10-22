"""One-shot device sampler whose inherited descriptor owns the query lock."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpuops.worker.devices import GpuDevice
from gpuops.worker.probe import LocalGpuProbe
from gpuops.worker.runtime import _atomic_write_json

DEVICE_QUERY_BUDGET_SECONDS = 6.0


def run_sample(
    *,
    server_id: str,
    result_path: Path,
    lock_fd: int,
    attempt_id: str,
    device: GpuDevice,
) -> None:
    os.fstat(lock_fd)

    def locked_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, pass_fds=(lock_fd,), **kwargs)

    sample = LocalGpuProbe(
        server_id,
        runner=locked_runner,
        sample_timeout_seconds=DEVICE_QUERY_BUDGET_SECONDS,
    ).sample_device(device)
    payload: dict[str, Any] = {
        "attempt_id": attempt_id,
        "device": {
            "index": device.index,
            "uuid": device.uuid,
            "pci_bus": device.pci_bus,
        },
        "snapshot": sample.snapshot.model_dump(mode="json"),
        "data": sample.data,
        "error_code": sample.error_code,
    }
    if sample.error_code is not None:
        payload["error"] = sample.snapshot.error or "GPU query failed"
    _atomic_write_json(result_path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--pci-bus", required=True)
    arguments = parser.parse_args(argv)
    run_sample(
        server_id=arguments.server_id,
        result_path=arguments.result,
        lock_fd=arguments.lock_fd,
        attempt_id=arguments.attempt_id,
        device=GpuDevice(
            index=arguments.index,
            uuid=arguments.uuid,
            pci_bus=arguments.pci_bus,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
