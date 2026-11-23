# Structured Progress

For training entry points you write or maintain, include machine-readable
progress in the program's normal stdout. GPUOps reads it from the captured
console log to estimate training completion in the Web GPU overview and
managed job detail. GPUOps also accepts jobs with ordinary logs.

## Record Format

Write one complete line for each report:

```text
GPUOPS_PROGRESS {"phase":"training","current":262,"total":786,"elapsed_seconds":83.4,"timestamp":1789952400.2}
```

The prefix is exactly `GPUOPS_PROGRESS ` (including the trailing space). The
rest of the line is one JSON object with exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `phase` | string | Name of the current phase. It must be 1-80 characters. |
| `current` | nonnegative integer | Work completed since the start of the whole declared phase. |
| `total` | positive integer or `null` | Total work in that phase, or `null` when the total is genuinely unknown. |
| `elapsed_seconds` | finite nonnegative number | `time.monotonic() - phase_start`, using one baseline for the phase/reporting session. |
| `timestamp` | finite number | `time.time()` at emission, as Unix seconds. |

The provided emitter takes `phase`, `current`, and `total`, tracks
`elapsed_seconds`, and adds `timestamp` automatically. It also emits compact JSON
and flushes stdout. See the copyable, standard-library-only
[example emitter](../examples/progress.example.py).

Run its no-GPU smoke example from the repository root:

```bash
python examples/progress.example.py
```

It runs immediately and emits seven records for 3 x 262 = 786 cumulative batches.

Use a cumulative counter for the declared phase. For example, training with
three epochs of 262 batches has `total = 786`; the first batch of epoch two is
`current = 263`. Emit the starting count when the phase begins. When resuming
from a checkpoint, emit the restored count and keep the same full-phase total.

## ETA Semantics

GPU Ops needs at least two consistent reports whose `current` and
`elapsed_seconds` both strictly increase to measure recent speed. Samples remain
consistent while `phase` and `total` stay the same and the counter and monotonic
clock do not regress. A phase change, total change, counter reset, or elapsed-time
reset starts a new measurement; samples from the previous sequence are not mixed
into its ETA. The report timestamp anchors the ETA, so unrelated log writes and
file modification-time changes do not move it.

The phase name controls the displayed scope:

- `train` and `training` describe the whole training phase. Their ETA is the end
  of training, not necessarily the end of the process or managed job.
- Every other name produces an explicitly labelled phase ETA. It must not be
  presented as whole-training or whole-job completion.

When `current == total`, that phase is complete. GPU Ops does not invent an ETA
for a later phase. A record with `total: null`, no measured advance, inconsistent
samples, or no report for more than ten minutes has unknown ETA. If the amount
of work cannot be known, report `null`; do not manufacture a total just to
display an estimate.

Structured records are evidence, not scheduling promises. They do not change
queue state, reserve a GPU, set a time limit, or replace
`schedule.expected_duration`. No manifest field, submission argument, service,
SDK, or package installation is required.

## Reporting Guidance

Report at phase transitions and about every 10-30 seconds during longer work.
Do not print on every batch when batches are fast. In distributed training,
only rank 0 should print the record. Always flush it so the managed log sees the
sample promptly.

The counter should represent stable completed work. If batch sizes or step cost
vary widely, choose a unit whose progress is meaningful or leave the total
unknown. Metrics, checkpoints, and configuration discovery are separate from
this contract and should remain in the experiment's existing logging.

## Minimal Integration

For a fresh, single-process run, copy `ProgressEmitter` into the experiment
and call it from the loop. A framework callback can use the trainer's global
step and total optimizer steps in the same way.

```python
import time

progress = ProgressEmitter()
total_batches = epochs * len(train_loader)
progress.emit("training", 0, total_batches)
next_update = time.monotonic() + 20

for epoch in range(epochs):
    for batch_index, batch in enumerate(train_loader):
        train_step(batch)
        completed = epoch * len(train_loader) + batch_index + 1
        now = time.monotonic()
        if now >= next_update or completed == total_batches:
            progress.emit("training", completed, total_batches)
            next_update = now + 20
```

Do not modify a third-party trainer or an already-running experiment solely to
force this format.
