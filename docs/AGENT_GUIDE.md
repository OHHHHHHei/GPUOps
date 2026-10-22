# GPUOps Agent Guide

Use GPUOps from any terminal Agent that can run commands and parse JSON. The
CLI is the integration surface; a Codex skill is optional. Install the CLI with
the [project setup](https://github.com/OHHHHHHei/GPUOps/blob/main/README.md#install), then read
[CLI contract](CLI_CONTRACT.md) for versioning, result shapes, and exit behavior.
Use `gpuops version --json` on the current control client to identify it before
integrating. Older installed server CLIs may lack `version` and `server list`;
continue using their existing `status`, `queue`, and `submit` commands and manage
installation from the control client.

Submission validates the manifest and checks Worker readiness automatically.
Once an experiment is authorized and its code, environment, and data are ready
on an authorized server, submit it with automatic GPU allocation. Choosing a
currently idle GPU ID is not a separate approval step.

On the control machine, use the installed `gpuops` command from any directory.
In a development checkout, use `uv run gpuops`. On an installed compute server,
use `~/.local/bin/gpuops` as the Worker Unix account from any experiment directory.
The shorter command works when `~/.local/bin` is on `PATH`; installation does
not configure that `PATH` entry.

## Normal Workflow

| Operation | Command | Read this result |
| --- | --- | --- |
| Inspect resources | `gpuops status --json` | Each server's GPU data and `worker.ready`. |
| Submit | `gpuops submit job.json --json` | Save `run_id`; confirm `submission_state: accepted`. |
| Inspect active work | `gpuops queue --json` | Active summaries in `servers[].jobs` and per-server errors. |
| Inspect completed work | `gpuops history --json` | Recent terminal summaries in `jobs`, plus `next_cursor`. |
| Understand a job | `gpuops job RUN_ID --json` | `job.status`, `explanation.code`, `message`, and `next_action`. |
| Read output | `gpuops logs RUN_ID --json` | Bounded console tail; a queued job may have no log yet. |
| Cancel managed work | `gpuops cancel RUN_ID --json` | `cancel_requested` or `cancelled`; query again for completion. |

To connect a server, run `gpuops server add SSH_ALIAS --json` on the control
machine. GPUOps reuses a ready installation, starts a stopped Worker, or
installs a missing Worker using the locally installed package. It waits for
readiness before creating a registration; failure creates no new registry
entry. Reusing a ready Worker does not upgrade its CLI. An existing remote
`host.json` preserves its server ID; otherwise the SSH alias becomes the ID.
Read `server.id` from the result and `gpuops server list --json` for the current
fleet. A fresh installation has an empty registry at
`~/.config/gpuops/servers.json`.

Registration does not install any Agent skill. Codex users can opt in with
`gpuops skill install --json` for the current account, or
`gpuops skill install --server ID --json` for a registered server's SSH account.
The default destination is `~/.codex/skills/gpu-infra`; if `CODEX_HOME` is set in
the target command environment, it is `$CODEX_HOME/skills/gpu-infra`. Other
terminal Agents use this guide and the CLI directly.

To disconnect a server, use `gpuops server remove ID --json`. It refuses active
managed jobs, then stops the Worker and removes the local registration while
preserving the remote installation, history, and experiment data. Add
`--server ID` to queries when you want a narrower scope.

`status` retains its monitoring exit code: zero means GPU collection succeeded,
not that scheduling is ready. Inspect each server's `worker.ready` explicitly.
Worker failure leaves successful GPU data visible, and one offline server does
not remove the other server's results.

Command defaults depend on where the CLI runs. The control machine selects all
registered servers for unqualified `status`, `doctor`, `queue`, `history`, and `fleet`, and
searches that fleet for unqualified `job`, `logs`, and `cancel`. An installed
server selects only the identity persisted in `~/.local/share/gpuops/host.json` for those
commands, even when the Worker is stopped or the current directory changes.
`--server` explicitly selects another server and requires its configured peer
SSH connection.

The current server is contacted through the same fixed JSON RPC locally under
the Worker Unix account; GPUOps does not SSH back to itself. Submission has no
new server flag: the manifest's explicit `resources.host` remains authoritative
and can select a peer with working SSH. `resources.host: "auto"` considers all
registered servers from the control machine but only the current server by
default when invoked from an installed server. Reconciliation follows the destination already frozen in the
local submission journal, even when it falls outside the current default query
scope.

When there is only one candidate host (the normal server-local case), `auto`
queues on that host without requiring a GPU sample. Explicit-host submission
works the same way. Worker readiness is still required; actual GPU allocation
waits for two reliable idle samples. Multi-host automatic placement still needs
fresh GPU evidence to choose a host. `status`, `doctor`, and job explanations
read the Worker's shared sample; repeating them does not force a new GPU query.

## Active Jobs and History

`queue` returns only queued, leased, starting, running, and cancellation-pending
jobs. Entries are small summaries: `run_id`, `project`, `status`, GPU IDs,
priority, and timestamps. Full commands, environment settings, and logs are read
with `job RUN_ID` and `logs RUN_ID` when needed. `queue --all` explicitly retains
the previous full-record output for existing scripts.

`history` returns succeeded, failed, and cancelled summaries across the selected hosts,
newest completion first. Each entry includes `server_id` and `run_id`. Defaults
need no flags: the first page contains up to 50 jobs. To filter a project, use
`history --project NAME --json`. To continue, pass the response's `next_cursor`
to `history --cursor CURSOR --json`, keeping the same project/server filters.
`--limit` optionally changes page size within 1–100. A null cursor ends paging.

If a host is unavailable, successful host data remains visible with
`outcome: partial`, `ok: false`, and exit code 1. History suppresses its next
cursor in that case to avoid skipping records; retry the same page after the
host recovers. An empty history page with failed hosts is not proof that no
jobs exist. Queries never create or replay a submission journal.

## Describe the Experiment Once

```json
{
  "api_version": "gpuops/v1",
  "id": "baseline-seed-42",
  "project": "example",
  "command": ["python", "train.py", "--seed", "42"],
  "working_directory": "/srv/experiments/example",
  "resources": {"host": "gpu-server", "gpu_count": 1, "memory_mib": 18000}
}
```

For the normal case, set `host` to an authorized, prepared server, omit `gpus`,
and set `gpu_count` to the needed number (or omit it for the default of 1).
GPUOps selects eligible GPUs when the job can run. Do not copy currently idle
GPU IDs into `gpus` just to make a submission. If the user explicitly requires
physical GPU IDs, use `gpus`; its length supplies `gpu_count`, and an explicit
count must match. Confirm only when the host or resource scale would exceed the
user's authorization, or a material unknown affects whether the experiment can
run as intended. Ordinary GPU selection needs no new confirmation.

Use `host: "auto"` only when GPUOps may choose the destination server too. On
the control machine, every registered candidate must be authorized for the
experiment and have the code, environment, data, and checkpoints ready. On an
installed server, `auto` considers only that server by default. The chosen destination is frozen at submission;
reconciliation does not move the job to another server.

`memory_mib` means required free memory **per GPU**, not a runtime memory cap.
`expected_duration` is optional and supplies ETA evidence, not a time limit.
Commands are argument arrays; `working_directory` is an absolute remote path.
Set `runtime.conda_environment` or `runtime.environment` only when the experiment
needs them. The example assumes the code and environment already exist on the
destination.
GPU Ops does not synchronize code, environments, datasets, or checkpoints; the
Agent remains responsible for preparing them on the destination before
submission.

## Optional Progress Reporting

When creating training code that the Agent controls, `GPUOPS_PROGRESS` stdout
records can improve ETA when the work has a meaningful counter.
Use the standard-library [example emitter](../examples/progress.example.py) and
follow the [structured progress contract](PROGRESS.md). Existing runs and
third-party trainers do not need to be changed solely for compliance.

Report cumulative work across the entire declared phase, normally every 10-30
seconds and at phase transitions. Only rank 0 should print in distributed runs.
`train` and `training` mean the whole training phase; other names receive an
explicit phase ETA. Unknown totals, stale sequences, sequences with no measured
advance, and fewer than two consistent samples remain unknown rather than
receiving a fabricated estimate. This reporting requires no manifest option or
extra package.

## Recover the Same Submission

A submission error includes `error_code`. `submission_state: not_accepted` means
this request was rejected; `unknown` means acceptance has not been established.
Preserve its run ID and use `gpuops reconcile RUN_ID --json` after resolving the
reported issue. The saved request and frozen server are reused, whether that
destination is local or reached over SSH. If no output arrived,
`gpuops reconcile --list --json` lists unresolved local identities without SSH
or resubmission. A `journal_warning` accompanies accepted work whose local
bookkeeping needs repair; retain that ID as well.

Each `submit` without `--run-id` creates a new experiment, even for identical
input. To change a rejected manifest, make a corrected new submission. Do not
automatically retry permanent errors or replace an uncertain submission with a
new one. `reconcile` without a run ID explicitly retries **all** pending entries.

## Interpret Job Explanations

| Code | Meaning and action |
| --- | --- |
| `not_before` | Wait until `details.available_at`. |
| `deadline` | The current schedule cannot meet the deadline; review the request. |
| `gpu_occupied` | Requested capacity is in use or reserved; wait. |
| `insufficient_free_memory` | Idle GPUs lack the requested free memory; review the request or wait. |
| `worker_unavailable` | Inspect/repair the Worker with `fleet status` / `fleet start`. |
| `failed` | Inspect the recorded failure reason and managed logs before another run. |
| `unknown` | Evidence is incomplete, mixed, stale, or unsupported by the installed Worker. |

Explanations describe evidence at `observed_at`; a single GPU sample cannot
promise an immediate start. The scheduler still confirms availability twice.
Queries never retry submissions or alter queued jobs. Unknown external-process
ETA remains unknown. Cancellation only applies to GPUOps-managed jobs; do not
kill an unmanaged process or bypass the queue with direct SSH execution.

For installation and repair, see
[Remote Worker](https://github.com/OHHHHHHei/GPUOps/blob/main/docs/REMOTE_WORKER.md). For the acceptance procedure, see
[Roadmap](https://github.com/OHHHHHHei/GPUOps/blob/main/docs/ROADMAP.md). Published checks and limits are summarized in
[Release validation](https://github.com/OHHHHHHei/GPUOps/blob/main/docs/VALIDATION.md).

## Account and Ownership Scope

One Worker and its queue coordinate jobs under one trusted Unix account on a
server. Other users' visible compute processes make a GPU unavailable, but
GPUOps cannot enforce an exclusive reservation against those users or programs
that launch outside its queue. Do not treat the account boundary as a
multi-tenant authorization service. Cancellation controls only verified
GPUOps-managed processes.
