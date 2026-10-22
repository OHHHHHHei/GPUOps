---
name: gpu-infra
description: Inspect GPU resources and manage queued experiments on registered Linux NVIDIA servers through GPUOps. Use for GPU status, process ownership, submission, queues, managed logs, progress estimates, recovery, or cancellation.
---

# GPU Infra

Use the installed `gpuops` CLI and `--json` for automation. It works from any
experiment directory. On an installed compute server as the Worker Unix account,
use `~/.local/bin/gpuops` if it is not on `PATH`; current-server operations run
locally without SSH. A source-checkout workflow can use `uv run gpuops`.

Read [Agent guide](docs/AGENT_GUIDE.md) for workflows and
[CLI contract](docs/CLI_CONTRACT.md) for result/error semantics. On the current
control client, use `gpuops version --json` for version discovery. Older server
CLIs may lack `version` and `server list`; continue using `status`, `queue`, and
`submit` there and manage installation from the control client. These references
ship with this skill. No MCP server or Agent-specific runtime is needed.

## Discover and Inspect

Use `server list --json` for registered hosts. A fresh registry is empty;
`server add SSH_ALIAS --json` connects an existing OpenSSH alias, installing or
starting a Worker if needed. Add does not install a skill or prepare experiment
assets. Reusing a ready Worker does not upgrade its CLI.

Unqualified queries cover registered servers on a control machine and only the
current server on an installed server. `--server ID` selects a host explicitly;
a peer needs configured SSH access. Submission follows `resources.host` in the
manifest. Automatic host selection considers the registered fleet on the control
machine and the current server by default on an installed server.

- `status --json`: GPU/process observations and independent `worker.ready`.
  Monitoring exit code zero does not prove Worker readiness.
- `doctor --json` and `fleet status --json`: connectivity and Worker diagnostics.
  Read operations use shared GPU samples and do not start NVIDIA probes.
- `queue --json`: active summaries. `history --json`: recent terminal jobs,
  paginated with `next_cursor`; keep the same filters when continuing. A partial
  result retains successful hosts and is not proof that failed hosts are empty.
- `job RUN_ID --json` and `logs RUN_ID --tail-bytes 65536 --json`: one job's
  state, explanation, next action, and bounded output. A queued job may have no
  log yet. Unknown or stale evidence stays unknown.

## Submit Within the Authorized Scope

Prepare the experiment's code, environment, data, and checkpoints on the chosen
server. GPUOps does not synchronize them. Use a `gpuops/v1` manifest with an
argv command and an absolute remote `working_directory`.

Default to automatic GPU allocation: set the prepared, authorized
`resources.host`, omit `resources.gpus`, and request the needed `gpu_count`
(default one). Set realistic `memory_mib`, which means required free memory per
GPU. Authorization to run an experiment includes selecting eligible physical
GPUs within that host/count scope; do not ask for GPU-ID confirmation or pin
currently idle IDs. Insufficient capacity means waiting in the queue.

Honor explicit physical GPU choices with `resources.gpus`. Use `host: "auto"`
only when every candidate is authorized and prepared. Ask only when execution
would exceed the established host/resource scope or a material requirement is
missing. Idle capacity alone does not authorize a new experiment.

Use `submit job.json --json` and retain `run_id` and `submission_state`.
Validation and Worker-readiness checks are automatic. `validate` is optional.
A ready Worker can queue an explicit-host job, or a sole-candidate automatic
job, without GPU telemetry; launch still waits for reliable samples.

## Recover and Cancel

`accepted` means durable acceptance; `not_accepted` means rejection; `unknown`
means acceptance is unresolved. Recover with `reconcile RUN_ID --json` on the
submitting machine. If no Run ID was printed, `reconcile --list --json` lists
pending local requests without contacting servers. Recovery keeps the saved
manifest and destination. Control and server clients share the remote queue
but have separate local journals.

Each fresh `submit` without `--run-id` creates a distinct experiment. Do not
replace an uncertain submission with a new one or automatically retry permanent
errors. Keep the Run ID when `journal_warning` accompanies accepted work.
`reconcile` without an ID explicitly retries all pending entries.

Use `cancel RUN_ID --reason TEXT --json` for managed jobs, then query for its
terminal state. Never edit queue files, adopt unmanaged processes, or kill a
process merely because it appears in `status`. Inspect Worker, queue, job, and
logs when scheduling fails; do not bypass the queue with direct execution
without explicit authorization.

One trusted Unix account shares the Worker queue. Other users' processes are
observed and avoided; GPUOps leases do not enforce exclusivity against them or
programs outside the queue. Accepted work continues while the control machine
is offline. Workers need an explicit `fleet start` after a server reboot.

## Optional ETA and Web

ETA requires evidence from readable progress logs or an optional
`schedule.expected_duration`. Do not infer release time from utilization or
process age. For new training code with a meaningful counter, see
[progress contract](docs/PROGRESS.md) and the standard-library
[emitter](examples/progress.example.py). Report cumulative phase work and use
rank 0 in distributed training. Existing and third-party code need no change.

The Web console starts with `web` and accepts loopback binds only. It displays
estimates with their source; estimates do not reserve GPUs or affect scheduling.
