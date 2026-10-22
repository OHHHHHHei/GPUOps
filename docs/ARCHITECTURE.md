# Architecture

## Scope

GPU Ops supports a POSIX control plane and host-local Agent access to the
same execution state. The Worker continuously samples GPUs; monitoring reads its
shared result. Once a job is durably accepted,
its worker and detached supervisor can continue without the submitting CLI.

```text
Control-machine Agent                   Installed server Agent
  gpuops CLI + Web console          ~/.local/bin/gpuops
  server registry + journal         submission journal
             |                                 |
             | OpenSSH                         | local collector / JSON RPC
      registered servers                current server
      worker + queue + supervisor       worker + queue + supervisor
```

The control machine has two SSH operation paths for each registered server:

1. `status`, `doctor`, and the Web GPU overview send a standard-library Python
   cache reader over SSH. They require an installed, running Worker to refresh
   the shared GPU sample and never start NVIDIA queries themselves.
2. Submission, queues, logs, cancellation, and fleet operations invoke the
   installed `~/.local/bin/gpuops` over SSH. The background worker advances the
   durable queue but does not serve a socket.

The control-machine registry at `~/.config/gpuops/servers.json` maps server IDs
to SSH aliases. It begins empty on a fresh setup. The CLI and Web read this
registry for fleet selection; the Web reloads changes without requiring a
restart and renders the physical GPU indices reported by each server.

`server add SSH_ALIAS` inspects the remote installation before changing it. A
ready installation is reused without upgrading its CLI, a stopped Worker is
started, and a missing installation is deployed from the local package. Add
waits for Worker readiness before registration, leaving the local registry unchanged on failure. A
pre-existing remote `host.json` determines the stable server ID; otherwise the
SSH alias supplies it. Registration does not install an Agent skill.
`server remove ID` requires no active managed jobs,
stops the Worker, and removes the local registry entry while preserving remote
installation, history, and experiment data. It does not drain jobs.

On an installed server, `~/.local/share/gpuops/host.json` persists the current
server identity independently of the working directory and Worker process
state. The CLI routes current-host collection, Worker lifecycle, and fixed JSON
RPC operations directly through a `HostClient` under the same Unix account; it
does not SSH to itself. An explicitly selected peer continues through its
configured SSH alias. Both transports preserve the same RPC request and response
contract. Successful installation writes the identity file, and uninstall
removes it.

## Components

### Client Control Plane

- `MonitorService` collects selected servers concurrently and isolates failures.
  Remote reads have a 35-second operation timeout; GPU queries run separately
  in per-device Worker sampler children with a six-second budget per card.
  `ServerResult.online` still means a successful current collection;
  `error_code` distinguishes GPU failure/timeout, SSH connection failure, and
  an operation timeout whose connectivity is unknown.
- `status.py` reads Worker readiness alongside monitoring without discarding
  GPU data when a Worker is unavailable. It uses the read-only `worker status`
  command, which shares the server-side readiness check with submission, and
  verifies that the heartbeat belongs to the selected server.
- `ControlClient` selects among multiple hosts using fresh GPU snapshots and
  queue state. An explicit or sole candidate host needs only Worker readiness
  to accept a job into its queue.
- `SubmissionJournal` atomically records `pending` and `accepted` submissions;
  an uncertain response is reconciled against the same host and run ID.
- FastAPI serves a loopback-only dashboard. It combines shared GPU monitoring with
  remote queue/job/log/cancel data and polls every five seconds. GPU snapshots,
  queue data, health, and process ETA have independent request guards and render
  as they arrive. Slow or failed streams retain their last view with an error.
  The provider shares a response cache across requests. The host validates its
  samples against its own clock with a 15-second freshness limit. Web comparisons
  with the control machine's clock allow five seconds of skew in either direction
  (an age from -5 to 20 seconds); the response cache still expires after 15 seconds.
  This display tolerance does not alter host validation or scheduling checks.
  Expired local samples trigger a new shared-cache read before being reported
  as stale; failed reads are cached for
  at most one second to coalesce requests without delaying the next poll's recovery.
  Delayed samples appear in amber, while query and connection failures remain red.
  After a failed
  collection, the GPU view can retain the last successful sample with its
  timestamp and a stale label. It marks those GPUs unknown/offline, excludes
  them from idle counts, and suppresses process ETA. This display cache is
  isolated from allocation. All consumers read the same Worker sample; stale
  display data never becomes scheduling evidence.

### Deployment Code

`gpuops.deploy` is the public import surface for deployment. Implementation
modules live under `gpuops.deployment`:

- `contracts.py`: deployment results, artifact metadata, errors, and budget constants.
- `artifacts.py`: installed-package archive creation or explicit source wheel
  construction, then transfer to the private remote cache.
- `client.py`: SSH action orchestration and response decoding.
- `bootstrap.py`: load the packaged bootstrap sources and select a fixed action.
- `_bootstrap/prepare.py`, `abort.py`, and `manage.py`: standalone remote programs.

The bootstrap sources, optional Codex skill, and its reference files ship inside
the wheel as package resources. Default deployment packages the installed GPUOps
files into a pip-installable source archive, so an installed CLI needs neither
a source checkout nor a local wheel builder to install a missing Worker. An
explicit project root instead builds a wheel from that source.

The bootstrap sources ship as ordinary Python files. They are
sent over stdin to the remote Python interpreter and use only the standard
library, because installation and repair must work before GPUOps is importable.
Their process inspection cannot depend on the installed Worker implementation.
Changes to these sources should be verified from a built wheel as well as from
the development checkout.

Each new deployment gets a random `release_id` used for its release directory,
upload path, status, and rollback. Older release IDs remain valid identifiers.

### Remote Worker

- A single-instance worker loop uses `flock`, a PID identity file, and an atomic
  heartbeat.
- A filesystem queue stores one JSON record per job plus a bounded, rotatable
  event stream.
- The scheduler uses two distinct successful observations of the same GPU index
  and UUID before leasing. The per-card observations must be at least two seconds
  and at most 45 seconds apart; the latest must be at most 15 seconds old. A GPU must have no
  visible compute process, acceptable utilization and calibrated idle-memory
  use, enough requested free memory, and no conflicting lease.
- A lease persists the selected index-to-UUID mapping. The supervisor launches
  the job with those UUIDs in `CUDA_VISIBLE_DEVICES`, so an index reorder between
  observation and launch does not redirect the job to another physical card.
- `core/scheduling.py` shares the schedule-window predicate with read-only job
  explanations. `worker/explanations.py` combines that predicate with readiness,
  one current GPU snapshot, and reservations. `worker/rpc.py` exposes it through
  optional `show {explain: true}` without changing persisted records. A single
  observation can identify blockers but cannot guarantee allocation.
- A detached per-job supervisor owns the launch lock, child process group,
  bounded console log, receipt, cancellation record, and final result.
- The worker retains handles to supervisors it launches and reaps them during
  reconciliation. Completion checks ignore zombie processes while still
  checking for surviving members of the managed process groups.

The deployment is a Python virtual environment under the SSH user's home. It
uses neither Docker nor a container runtime, and it does not require `sudo`,
systemd, or root-owned files.

### Process Management Code

The worker process owns the queue loop; detached supervisors own experiment
processes. Their implementation files follow those responsibilities:

- `worker/runtime.py`: the Worker lock, heartbeat, PID file, and start/stop operations.
- `worker/sampling.py`: procfs inventory, independent per-device sampling
  lifecycles, locks and failure backoff, aggregate cache publication, and cached
  resource reads. It also checks the legacy host-wide lock as a rollout barrier.
- `worker/sampling_child.py`: one isolated attempt for one PCI device; that
  device's lock is inherited by NVIDIA command children so a stuck query remains
  the sole owner for that card even across a Worker restart.
- `worker/supervisor.py`: launch, reconcile, poll, and cancel orchestration, plus
  the existing public imports and internal subprocess entrypoint.
- `worker/process_identity.py`: procfs parsing, identity matching, and inspection
  and signaling of owned process groups.
- `worker/supervisor_records.py`: receipt/result types, run paths, and durable
  record reads and writes.
- `worker/supervisor_child.py`: detached child execution and bounded console capture.

Runtime identities serialize `start_time_ticks`; supervisor identities serialize
`pgid` and `start_ticks`, with nullable credentials for a process that exits before
inspection. Shared procfs mechanics do not merge those two persisted schemas.
The supervisor protocol remains `gpuops-supervisor/v1`. Changes to these files
must retain old-record recovery and be exercised on Linux, including from an
installed wheel outside the source checkout.

## Trust Boundaries

- A Worker and its queue belong to one trusted Unix account. All Agents using
  that account share its execution authority and job state; GPUOps is not a
  multi-tenant authorization boundary.
- Other users' visible GPU processes are observed and avoided. GPUOps cannot
  reserve a GPU against programs that bypass its queue.
- SSH is the only network transport used by GPU Ops.
- The Web server rejects non-loopback bind addresses.
- SSH keys and aliases remain in the user's OpenSSH configuration.
- Remote RPC accepts one fixed JSON request on stdin and returns one JSON line;
  it never evaluates a caller-provided shell command.
- Job commands are argv arrays and launch with `shell=False` as the SSH user.
- GPU Ops signals only a stored supervisor/child identity after checking PID,
  process group, Linux start ticks, boot ID, and launch token as applicable.
- An unknown, stale, inconsistent, or partially persisted state fails closed.

## Worker Readiness Gate

The existence of a PID is not enough. A worker is ready only when all of these
conditions hold:

- the stored PID, Linux start ticks, and boot ID match the live process;
- the heartbeat parses and has status `running`;
- heartbeat PID and expected server ID match;
- the timestamp is timezone-aware, no more than five seconds in the future, and
  no more than 75 seconds old.

The Worker discovers candidate index, UUID, and PCI identities from procfs; this
control-loop step does not call NVIDIA tooling. It samples each candidate in a
separate child with a six-second monotonic budget and PCI-scoped commands, then
accepts the result only when the returned index, UUID, and PCI identity match.
Each card has its own inherited query lock. Ordinary failure backoff is capped
at 60 seconds; timeouts retry after 60 to 300 seconds from actual child exit,
so time spent stuck in the driver does not consume the cooldown. One timeout
does not block cache publication or the control loop.
The legacy host-wide lock is checked as a migration barrier. A query still in
kernel sleep retains its per-card lock until it exits, including across a Worker
restart. Each attempt first checks the target PCI configuration; unavailable
devices fail before a NVIDIA query. Idle telemetry also requires bounded CUDA
initialization and UUID verification in a child before it becomes allocatable.
This does not create a CUDA context or run computation. Failed cards continue
retrying and recover automatically on fresh success, with no permanent
quarantine file or manual unblocking step. Recovery still needs two new idle
observations before allocation. Separate processes cannot isolate a shared
NVIDIA kernel lock, repair the driver, or prevent other tools from querying the
same faulty device.

Only completed control loops refresh the heartbeat. Stalled control-loop
detection still uses the 75-second threshold. Each card retains its actual sample
timestamp when other card results are published; aggregate publication does not
renew scheduling evidence. The latest per-card observation must be at most 15
seconds old at allocation, and schedule eligibility is checked immediately before
each lease. Cached observation pairs used for allocation cannot be reused for new
leases on later ticks.

A tick that raises an exception or returns per-job errors writes a `degraded`
heartbeat. `last_tick_error` contains a bounded error summary and `tick_failures`
counts failed iterations. A subsequent successful tick restores `running`.
Client-side submit checks health before RPC submission, and the remote submit
handler repeats the readiness check before creating a queue record.

## State Ownership

The Worker atomically publishes an aggregate current snapshot and its prior
snapshot in its state directory. Each GPU carries its own sample timestamp and
error state, allowing two independent idle observations to be checked per card.
UI polling state stays in the Web process. Each machine
running the control CLI keeps the small durable submission journal needed to
reconcile uncertain transport responses:

```text
~/.local/state/gpuops/control-journal/
```

The remote layout is user-owned and private:

```text
~/.local/bin/gpuops -> ../share/gpuops/current/venv/bin/gpuops
~/.local/share/gpuops/
  host.json
  current -> releases/<release_id>
  releases/<release_id>/
~/.cache/gpuops/uploads/
~/.local/state/gpuops/
  worker.pid.json
  worker-health.json
  gpu-sample.json
  gpu-sample.lock
  gpu-sample-<pci-token>.lock
  gpu-sample-<pci-token>.result.json
  queue/jobs/<run-id>.json
  queue/events.jsonl
  runs/<run-id>/{intent,receipt,result,cancellation}.json
  logs/worker/worker.log
  logs/<run-id>/console.log*
```

Queue writes, receipts, results, heartbeats, and journal writes use atomic
replacement. Queue operations and deployment operations have separate file
locks. Durable state is authoritative; query indexes are only caches.

## Storage Budget

GPU Ops enforces a strict 1 GiB total across its remote share, cache, and state
roots. The deployment reserves at most 256 MiB for installation plus cache,
with a 128 MiB share cap. Mutable state is capped at 768 MiB, including a 640
MiB log budget.

Worker logs rotate. Each job reserves bounded console capacity (4 MiB times the
current file plus two backups by default). Cleanup is restricted to GPU
Ops-owned logs and event archives; queue records and experiment data are not
cleanup candidates. If capacity cannot be proved, new work fails closed while
already-running supervisors continue.

Datasets, checkpoints, project logs, and other files written in the experiment
working directory are outside both this budget and GPU Ops cleanup ownership.

## Scheduling And Lifecycle

The control machine resolves `resources.host = "auto"` using usable snapshots and queue
state from all registered servers. One unreachable server does not block
selection of the other. A server-local Agent defaults automatic placement to
its current server. An explicit manifest host remains authoritative and can
select a peer when its SSH alias is configured. In every case, the choice is
frozen into the manifest and journaled before submission; reconciliation
searches the journal's frozen destination even when it is outside the current
default query scope.

An explicit host bypasses auto-placement but does not bypass the worker health
gate. `resources.memory_mib` defaults to 1 GiB of required free memory per GPU;
real workloads should declare their requirement explicitly. Exclusive
allocation among jobs in this Worker queue is the only supported mode. It does
not enforce exclusivity against other Unix accounts or programs outside GPUOps.

```text
QUEUED -> LEASED -> STARTING -> RUNNING -> SUCCEEDED
   |         |          |          |
   +-------> CANCELLED   +----------+-> CANCEL_REQUESTED -> CANCELLED
   +---------+----------+-------------------------------> FAILED
```

Queued or leased cancellation completes immediately. Starting or running work
moves to `CANCEL_REQUESTED`; the supervisor forwards `SIGTERM` to its managed
child, persists the grace deadline, and may escalate to `SIGKILL` after ten
seconds. Completion is recorded only after the verified managed process groups
are gone. Cancellation never targets unmanaged processes.

Stopping the background worker signals only the worker PID. Detached experiment
supervisors continue, and a restarted worker reconciles their receipts/results.

## ETA Contract

- Managed job details reuse the already-requested bounded console logs and the
  GPU overview's progress parser. Supported log progress takes precedence;
  phase estimates explicitly remain phase estimates.
- When no structured record is recognized, a running job can fall back to
  `started_at + expected_duration`, labelled as a submission-time estimate with
  low confidence. Past estimates are unknown. A recognized structured record
  with unknown ETA blocks this fallback.
- Queue rows reuse log samples from visited running jobs for at most 60 seconds,
  with a bounded in-memory cache. Queue reads never fetch logs for every job.
- Waiting jobs have no predicted start/end: GPU availability, start windows,
  deadlines, and other users' work make the former serial estimate misleading.
  Job explanations still describe the actual scheduling blockers.
- Cancelling jobs have no training ETA. Terminal jobs use durable `finished_at`.
- The Web GPU overview can estimate a managed or unmanaged process's current
  training from readable logs. CLI GPU monitoring does not calculate log ETA.

These values are planning aids, not reservations or completion guarantees.

The separate `/api/v1/process-etas` endpoint keeps log SSH reads out of the GPU
snapshot response. The browser attaches estimates only when server, PID, and
process start ticks match, and shows progress/sample times in the tooltip.

`web/log_eta.py` samples logs through the existing SSH transport on demand, with
a 30-second in-memory cache keyed by PID and process start ticks. The standalone
`web/_log_sample.py` discovers explicit log arguments and regular stdout/stderr
destinations. It reads only a small header and tail (8/48 KiB), caps discovery at
32 processes and 16 files per host, and reads shared DDP logs once. For a managed
child or DDP descendant whose stdout is a pipe, it validates the bounded ancestor
chain and launch receipt, then reads the existing regular `console.log`; it never
reads the pipe itself. It does not recursively search project directories.
Sampling failures leave GPU monitoring available; no remote service or additional
Worker deployment is needed.

`web/progress.py` first interprets optional `GPUOPS_PROGRESS` records, then legacy
completed-epoch timings, per-epoch duration summaries, timestamped step counts,
or tqdm-style progress bars with speeds when no structured prefix is present.
A recognized but invalid, stale, complete, or insufficient structured record
stays unknown and blocks a misleading legacy or submitted-duration fallback.
Progress without whole-training evidence is labelled as a stage estimate.
Explicit record or log-line timestamps anchor observations; safe untimestamped
legacy progress uses the log modification time, never the polling time. Old-run
logs, stale progress, completed counters, anomalous timestamps, and estimates in
the past return unknown. Managed `logs` RPC responses supply source path,
modification time, sampling time, and server UTC offset; older Workers remain
usable but may lack the anchors needed for an estimate. Source paths and
observation times accompany estimates, while raw sampled log contents remain in
the short-lived local cache. Estimates never reserve GPUs, affect scheduling,
inspect future shell queues, or change job state. See [Structured progress](PROGRESS.md)
for the record contract and emitter.

## Compatibility

The CLI and versioned JSON output are the terminal-Agent integration surface.
See [CLI contract](CLI_CONTRACT.md) for compatibility rules. The optional Codex
skill is installed separately with `gpuops skill install`, locally or with
`--server ID` for a registered remote account; `server add` has no skill side
effect.

This repository is a clean implementation. It does not import, load, or depend
on any earlier project-specific dashboard or scheduler. Research projects
integrate through `gpuops/v1` manifests, argv commands, environment values, and
optional structured `GPUOPS_PROGRESS` records.

## Queue and history reads

The Web current-jobs view requests active `JobSummary` records through
`list(summary=true, statuses=...)`. It never requests the full historical queue.
`history` returns at most 101 terminal summaries per host, filtered by project
and ordered by completion time and Run ID. The Web merges host pages into 50
rows by default; its cursor carries `(finished_at, run_id, server_id)` so ties
across hosts and newly completed jobs do not shift subsequent pages. A partial
fleet response exposes host errors and disables further paging until refreshed.
Full commands and bounded logs are fetched only for the selected job. Detail
lookup reads that Run ID directly on each host instead of enumerating history.

`queue/queue-index.sqlite3` is a derived query index using Python's built-in
SQLite support. `queue/jobs/<run-id>.json` remains the source of truth, including
submission deduplication and recovery. Under the existing queue lock, current
writes update the index after the authoritative JSON replacement. Reads compare
a cheap directory stat signature; missing/stale indexes rebuild from the JSON
records once. Older releases also replace job files atomically, so rollback
writes invalidate the index automatically when the new code returns. Manual
in-place edits of task files are unsupported. The index counts toward the
existing storage budget. If it cannot be stored, reads temporarily rebuild the
same queries in memory from the authoritative records, preserving scheduling
at the cost of slower reads. It requires no separate service, user setting, or
record deletion policy.

The CLI `queue` requests the same active summaries. `queue --all` preserves the
old full-record listing; `history` and the Web reuse `queries.collect_history`
for cross-host ordering, pagination, and partial failure behavior. Query-only
commands do not initialize the local submission journal. Host selection reads
only active queue records. RPC health preserves `queue.jobs` as the total job
count, using an indexed aggregate rather than parsing all job JSON or summaries.
Missing/stale indexes still rebuild from authoritative records; storage byte
accounting continues to inspect filesystem metadata under the existing budget.
