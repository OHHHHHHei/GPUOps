"""Copyable, standard-library-only GPU Ops progress emitter and smoke demo."""

from __future__ import annotations

import json
import time


class ProgressEmitter:
    """Emit optional GPUOPS_PROGRESS records to stdout."""

    def __init__(self) -> None:
        self._key: tuple[str, int | None] | None = None
        self._last_current: int | None = None
        self._phase_started = time.monotonic()

    def emit(self, phase: str, current: int, total: int | None) -> None:
        key = (phase, total)
        now_monotonic = time.monotonic()
        if key != self._key or (self._last_current is not None and current < self._last_current):
            self._phase_started = now_monotonic

        record = {
            "phase": phase,
            "current": current,
            "total": total,
            "elapsed_seconds": now_monotonic - self._phase_started,
            "timestamp": time.time(),
        }
        print(
            "GPUOPS_PROGRESS " + json.dumps(record, separators=(",", ":")),
            flush=True,
        )
        self._key = key
        self._last_current = current


def demo() -> None:
    """Run a tiny no-GPU example with realistic cumulative training counts."""

    epochs = 3
    batches_per_epoch = 262
    total_batches = epochs * batches_per_epoch
    report_every = 131
    progress = ProgressEmitter()
    progress.emit("training", 0, total_batches)

    for epoch in range(epochs):
        for batch_index in range(batches_per_epoch):
            completed = epoch * batches_per_epoch + batch_index + 1
            if completed % report_every == 0 or completed == total_batches:
                progress.emit("training", completed, total_batches)


if __name__ == "__main__":
    demo()
