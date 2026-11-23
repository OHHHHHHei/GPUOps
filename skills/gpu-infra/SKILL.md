---
name: gpu-infra
description: Inspect GPU resources and manage queued experiments on registered Linux NVIDIA servers through GPUOps. Use for GPU status, process ownership, submission, queues, managed logs, progress estimates, recovery, cancellation, or returning to the original chat when experiments finish.
---

# GPU Infra

Use the installed `gpuops` CLI and `--json` for automation. It works from any
experiment directory. On an installed compute server as the Worker Unix account,
use `~/.local/bin/gpuops` if it is not on `PATH`; current-server operations run
locally without SSH. A source-checkout workflow can use `uv run gpuops`.

Read [Agent guide](docs/AGENT_GUIDE.md) for workflows and
[CLI contract](docs/CLI_CONTRACT.md) for result/error semantics. On the current
control client, use `gpuops version --json` for version discovery. The control
client and every server must run the same release; on `version_mismatch`, run the
returned `gpuops deploy install` command from the control client. These references
ship with this skill. No MCP server or Agent-specific runtime is needed.

## Discover and Inspect

Use `server list --json` for registered hosts. A fresh registry is empty;
`server add SSH_ALIAS --json` connects an existing OpenSSH alias, installing or
starting a Worker if needed, and upgrades an installation that runs a different
release.

Unqualified queries cover registered servers on a control machine and only the
current server on an installed server. `--server ID` selects a host explicitly;
a peer needs configured SSH access. Submission follows `resources.host` in the
manifest. Automatic host selection considers the registered fleet on the control
machine and the current server by default on an installed server.

- `status --json`: GPU/process observations. Check each server's `worker.ready`
  for scheduling readiness.
- `doctor --json` and `fleet status --json`: connectivity and Worker diagnostics.
- `queue --json`: active summaries. `history --json`: recent terminal jobs,
  paginated with `next_cursor`; keep the same filters when continuing. A partial
  result lists failed hosts in its errors; query them again before concluding
  they have no jobs.
- `job RUN_ID --json` and `logs RUN_ID --tail-bytes 65536 --json`: one job's
  state, explanation, next action, and bounded output. A queued job may have no
  log yet. Unknown or stale evidence stays unknown.

## Submit Within the Authorized Scope

Prepare the experiment's code, environment, data, and checkpoints on the chosen
server before submitting. Use a `gpuops/v1` manifest with an argv command and an
absolute remote `working_directory`.

When writing or maintaining a training entry point, make the training process
emit [`GPUOPS_PROGRESS` records](docs/PROGRESS.md) to stdout from its loop or
trainer callback. Reuse existing structured output, or copy the standard-library
[ProgressEmitter](examples/progress.example.py) into the experiment. Emit the
starting step, then the actual cumulative step about every 10–30 seconds as
training advances, and once at the end. Count across all epochs, use `None`
when the total is unknown, and emit only on global rank 0 in distributed runs.

Default to automatic GPU allocation: set the prepared, authorized
`resources.host`, omit `resources.gpus`, and request the needed `gpu_count`
(default one). Jobs are exclusive by default and need `memory_mib` free on each
GPU. For jobs intended to share, set `resources.exclusive: false` and request
the expected per-card peak with a positive `memory_mib`. That memory stays
reserved from lease until release. Authorization to run an experiment includes selecting eligible physical
GPUs within that host/count scope; do not ask for GPU-ID confirmation or pin
currently idle IDs. Insufficient capacity means waiting in the queue.

Honor explicit physical GPU choices with `resources.gpus`. Use `host: "auto"`
only when every candidate is authorized and prepared. Ask only when execution
would exceed the established host/resource scope or a material requirement is
missing. Start new experiments only when the user asked for them, not because
GPUs are idle.

Use `submit job.json --json` and retain `run_id` and `submission_state`.
Validation and Worker-readiness checks are automatic. `validate` is optional.
A ready Worker can queue an explicit-host job, or a sole-candidate automatic
job, without GPU telemetry; launch waits for reliable samples.

## Return to the Original Chat (default)

After submitting experiments from a Codex Desktop, Claude Code Desktop, or
Antigravity chat, register a follow-up for them unless the user says not to.
GPUOps waits for the jobs and brings their results back to this chat, where you
analyze them with the existing context.

Register once, after every submission in the batch is `accepted`, with
`followup register FILE --json` (or `-` for stdin) and no `target`; GPUOps
binds the current chat. Write a `gpuops.followup/v1` document
([example](examples/followup.example.json)) with:

- `jobs`: the submitted `run_id`s with their `server_id`s.
- `mode`: `all_terminal` when the results are compared or analyzed together
  (a sweep, an ablation, baseline versus candidate); `per_job` when each result
  is handled on its own or the user asks for it.
- `action`: what to do with the results, taken from the conversation: which
  saved metrics or artifacts to read, what to compare them against, which
  research records to update, and what to report here.
- `context`: the research question and the file locations needed later.
- `notify`: `none` unless the user asked for email; `analysis_complete` sends
  one after you record the analysis.

If registration reports that the follow-up is unbound, this chat cannot be
resumed automatically; tell the user and continue without it. Do not wait in a
sleep loop or start a separate CLI conversation; the shared checker waits.

When a completion message arrives, run `followup acknowledge EVENT_ID --json`
first and continue only when `first_receipt` is true. Read the saved results,
perform the action, report the analysis here, then record
`followup complete EVENT_ID --summary TEXT --json`. See
[follow-ups](docs/FOLLOWUPS.md) for supported clients and commands.

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

One trusted Unix account shares the Worker queue. External compute processes
block exclusive jobs. Shared jobs may run alongside them when the remaining
memory budget permits. Accepted work continues while the control machine is offline.
Workers need an explicit `fleet start` after a server reboot.

## Web and ETA

The loopback Web console (`gpuops web`) shows estimates from progress logs or
an optional `schedule.expected_duration`, labelled with their source. GPU
utilization and process age alone cannot predict a release time.
