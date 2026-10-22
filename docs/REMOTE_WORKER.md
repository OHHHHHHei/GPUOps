# Remote Worker Contract

The remote worker is a small process running as the existing SSH user on each
compute server. It opens no network port, needs no elevated privileges, and can
advance accepted jobs while the control machine is offline.

The implementation uses a per-release Python virtual environment that does not
inherit system Python packages. Its dependencies are installed inside that
environment, so older system libraries do not interfere with package builds.
It does not
use Docker, `sudo`, systemd, or a root-owned daemon. GPU Ops also does not install
a reboot-start entry; after a server reboot, the operator must start the worker
again.

The installed `~/.local/bin/gpuops` is also the server-side Agent entry point.
It reads the installation identity from `~/.local/share/gpuops/host.json`, so
the current server remains known independently of the working directory and
Worker lifecycle. For operations targeting that server, the CLI uses the same
fixed JSON RPC locally as the Worker Unix account and does not SSH back to
itself. Use the full path from arbitrary experiment directories unless
`~/.local/bin` is already on `PATH`; deployment does not promise to configure
`PATH`.

## Deployment Boundary

Use live `deploy status`, `fleet status`, and `status` results for installation,
Worker readiness, and GPU telemetry. These are separate observations. [Release validation](VALIDATION.md) summarizes controlled checks; those checks
do not certify a new server or assert current health.

Upgrade clients and Workers together when changing shared GPU sampling behavior;
older clients may query GPUs directly. Monitoring uses the deployment's default
state directory. A manually started Worker with a custom `--state-dir` does not
feed fleet monitoring.

The Worker queue is shared by one trusted Unix account. It observes and avoids
visible external GPU processes, including other users' work, but does not
provide OS-enforced exclusivity against those users or programs outside the
queue. Use separate accounts as separate trust boundaries, rather than treating
GPUOps as a multi-tenant service.

Prerequisites for adding a server:

- working SSH/scp access through a configured OpenSSH alias;
- remote Linux with an installed NVIDIA driver;
- remote Python 3.10 or newer with virtual-environment support;
- a pip-accessible source or cache for GPU Ops dependencies;
- NVIDIA procfs device information and `nvidia-smi` available to the worker;
- enough quota within the 1 GiB GPU Ops budget;
- no unmanaged file at `~/.local/bin/gpuops`, which deployment refuses to
  replace.

For ordinary access, use the installed CLI on the control machine:

```bash
gpuops server add SSH_ALIAS --json
gpuops server list --json
```

`server add` inspects the remote installation, reuses it if the Worker is
ready, starts a stopped Worker, and deploys only if GPUOps is missing. It waits
up to 75 seconds for slow startup and creates no new registry entry on failure.
Repeating add for an already ready server does not restart its Worker or upgrade
its installed CLI. An existing `host.json` keeps its ID; otherwise the SSH alias
supplies the ID. The local registry is `~/.config/gpuops/servers.json` and starts
empty.

Registration does not install an Agent skill. Optionally run
`gpuops skill install --json` on the current machine, or
`gpuops skill install --server ID --json` for a registered server's SSH account.
The default skill destination is `~/.codex/skills/gpu-infra`; `CODEX_HOME` in the
target command environment changes it to `$CODEX_HOME/skills/gpu-infra`. Any
terminal Agent can use the CLI and [Agent guide](AGENT_GUIDE.md) directly.

For lower-level maintenance:

```bash
gpuops doctor --json
gpuops deploy status --json
# Install or upgrade registered targets from the local GPUOps package:
gpuops deploy install --json
```

An installed CLI can deploy without a source checkout: GPUOps packages its
installed Python files and resources as a standard pip-installable source
archive, uploads it, and installs it remotely. Remote dependency installation
still needs an accessible source or cache. GPUOps itself does not need a
published PyPI package. When developing from source, build and deploy an
explicit project root instead:

```bash
uv run gpuops deploy install --project-root . --json
```

To operate on a subset, repeat `--server ID` as needed:

```bash
gpuops deploy install --server gpu-server --json
gpuops deploy status --server gpu-server --json
```

Deployment assigns a fresh random `release_id`, copies the package artifact
with system `scp` to a private upload path, creates the virtual environment at
its release path behind an incomplete marker, installs the artifact, atomically
switches `current`, and starts a Worker. Older release IDs
remain usable as identifiers for rollback and status.
A failed switch attempts to restore the prior release and worker.

## Required Verification

After installation, do not treat SSH success alone as worker readiness. Run:

```bash
gpuops deploy status --json
gpuops fleet status --json
gpuops status --json
```

For every intended host, verify:

- `installed` is true and `current_release` identifies the intended release;
- `worker_release` matches the active release;
- `worker_ready` or `ready` is true;
- total owned bytes remain below the reported 1 GiB budget;
- the shared GPU snapshot contains the expected devices and process details,
  or reports unavailable/stale telemetry independently of Worker readiness.

The worker is ready only when its stored PID identity matches `/proc`, its
heartbeat status is `running`, its PID and server ID match, and its
timezone-aware heartbeat is not more than five seconds in the future or more
than 75 seconds old. Missing, stale, `starting`, `degraded`, or inconsistent
health fails closed.
Both the client transport and remote submission handler enforce readiness.

The control loop polls independent sampling children without waiting for them.
Each GPU has a six-second budget for device checks, PCI-scoped `nvidia-smi`
queries, and an idle-only CUDA initialization check.
This isolates the sampling processes, but it does not isolate NVIDIA driver
locks. Controlled acceptance observed one targeted device query stalling sibling
queries in the shared NVIDIA driver; process isolation cannot remove that limit.
Heartbeat, cancellation,
and completion reconciliation remain outside those sampling children, but
process isolation alone does not guarantee independent telemetry. Heartbeats
still come only from completed control loops; a stalled control loop takes up
to 75 seconds to become unready. Deployment also allows 75 seconds for graceful
stop and readiness.

Only one query attempt per physical GPU may run at a time. Its PCI-based lock is
inherited by NVIDIA commands, so a query stuck in kernel sleep prevents another
attempt on that card after a Worker restart. Successful attempts are followed by
a five-second interval; ordinary failures back off independently up to 60
seconds. Query timeouts use a separate 60-to-300-second cooldown, measured from
the sampling child's actual exit, so a long kernel stall cannot consume the
cooldown and trigger an immediate repeat. Successful sampling resets both.
Due failed-device retries are spread across control-loop ticks, ordered by their
due time; healthy periodic sampling continues independently.
`status`, `doctor`, the Web overview, job
explanations, and the scheduler read the shared cache without triggering probes.
A missing Worker/cache requires installation or startup; monitoring no longer
falls back to independent queries.

### Automatic GPU Recovery

Every discovered card remains eligible for automatic rechecking. There is no
permanent fault list or manual unblocking step. The old `gpu-quarantine.json`
file is no longer read; archive it when upgrading from the quarantine release.

Each attempt first reads the target PCI device's configuration. A missing or
unreadable device, or an invalid vendor/device identity, fails that card before
any NVIDIA query is issued. A later attempt reads it again, so restoration can
be discovered without invoking a driver that already cannot reach the card.

After successful telemetry, a card that appears idle must also pass CUDA driver
initialization in a bounded child process, with visibility restricted to its
UUID and the returned CUDA UUID verified. This catches readable telemetry with
unusable CUDA initialization. It creates no CUDA context and runs no experiment
or compute workload; it does not certify that every future CUDA operation will
succeed. Occupied cards retain ordinary telemetry without this extra check.

Device and CUDA failures are shown separately in the Web console. Failed cards
remain unavailable to the scheduler and retry with the same per-card backoff.
A successful retry clears the current error automatically and resets the normal
sampling interval. A still-running timed-out query retains its lock; automatic
retry never means stacking another query onto it or forcibly resetting a GPU.

A failed or over-budget card cannot be allocated. The scheduler requires two
distinct successful idle observations per selected card, at least two and at most
45 seconds apart, and checks the latest sample's 15-second freshness and the
job's schedule window before leasing. A used pair cannot allocate more work
on a later tick. The Web's last-good display cache never grants allocation.
These changes contain blocked telemetry; they do not repair the NVIDIA driver.

Before unattended use, complete the canary checks in
[ROADMAP.md](ROADMAP.md#remote-acceptance-gate).

## Installation Layout

```text
~/.local/bin/gpuops -> ../share/gpuops/current/venv/bin/gpuops
~/.local/share/gpuops/
  host.json
  deploy.lock
  current -> releases/<release_id>
  releases/<release_id>/venv/
~/.cache/gpuops/uploads/
~/.local/state/gpuops/
  worker.lock
  worker.pid.json
  worker-health.json
  gpu-sample.json
  gpu-sample.lock
  queue/
  runs/
  logs/
```

Files and directories are private to the user. Release switches and state
updates use atomic replacement; deployment and queue mutation use file locks.

## Worker Lifecycle

The host-local commands are:

```text
gpuops worker run --server-id ID [--once] [--interval SECONDS]
gpuops worker start --server-id ID [--interval SECONDS]
gpuops worker stop
gpuops worker status
```

Normal control-machine operation uses `gpuops fleet start|stop|status`, which invokes
those fixed commands over SSH for registered targets. On an installed
server, the unqualified fleet command targets only that server and uses the
local lifecycle implementation; `--server` can select a configured peer over
SSH.

Only one worker loop may hold the worker lock. `start` launches a detached user
process and waits for a fresh heartbeat. `stop` verifies the worker identity and
signals only that PID; it does not signal job supervisors or experiment process
groups. Therefore stopping or upgrading the worker does not terminate an
already-running managed experiment.

The worker is not automatically restarted after a server reboot. Use:

```bash
gpuops fleet start --json
```

## Submission Boundary

The control machine resolves `resources.host = "auto"` across all registered servers. A
server-local Agent considers only its current server by default, and can queue
there even while GPU samples are unavailable. Actual launch still waits for
reliable samples. Multi-host automatic placement needs fresh evidence to choose
a host. Use it only when every candidate is authorized for the experiment and
its code, environment, data, and checkpoints are ready there. An explicit
manifest host is fixed directly and can select a configured peer over SSH;
submission has no new server-selection flag. In all cases, the chosen host is
written into the manifest and local journal before queue mutation.
Reconciliation searches the journal's frozen destination or destinations even
when they are outside the current default query scope, and does not recalculate
placement.

Submission is idempotent:

- the first request atomically creates the queued record;
- a retry with the same run ID and identical frozen manifest returns that
  record;
- reuse of a run ID with different content is rejected;
- an ambiguous submission result is queried on the same host and run ID, never retried
  on a different host.

Only a successful response containing `durable: true` means the job was
accepted. A pending local journal entry is not proof of remote acceptance; use
`gpuops reconcile --json`.

The one-shot `gpuops remote --server-id ID` RPC accepts only the versioned fixed
methods `health`, `submit`, `list`, `history`, `show`, `logs`, and `cancel`. It
reads JSON from stdin, writes one JSON response, and opens no listening socket.
SSH and host-local transports use this same contract.

## GPU And Launch Boundary

The worker discovers cards from procfs and validates their index, UUID, and PCI
identity through device-scoped `nvidia-smi`. It marks a GPU idle only when:

- no compute process is reported;
- utilization is within the configured idle threshold;
- used memory is within the calibrated baseline plus tolerance.

The default idle-memory limit is 384 MiB plus 256 MiB tolerance. A process-free
GPU above that limit is treated as residual/non-idle. The scheduler reads two
distinct observations of each selected card, 2–45 seconds apart, with the latest
no more than 15 seconds old. Updating another card never refreshes that evidence.
The scheduler also requires the requested amount of
free memory. A manifest that omits `memory_mib` requests at least 1 GiB free per
GPU; production manifests should state the real requirement.

After leasing, the worker writes a durable launch intent and starts a detached
per-job supervisor. The supervisor holds a run-specific lock, launches the argv
command with `shell=False`, injects confirmed UUIDs into `CUDA_VISIBLE_DEVICES`, captures bounded
combined stdout/stderr, and atomically writes receipt and result files.
Older leases without UUID mappings retain their original index binding.

New allocations persist their UUID mapping. Rollback validates queue records
with the target release before switching; releases that cannot read these records
are refused. Use an updated client for rollback rather than bypassing that check.

Receipts bind the run ID and launch token to supervisor/child process identity.
Recovery never launches a second child when ownership is ambiguous. A missing,
corrupt, or mismatched receipt fails closed.

## Cancellation Boundary

`gpuops cancel RUN_ID` persists an idempotent request.

- Queued and leased jobs become `CANCELLED` without launching.
- Starting and running jobs become `CANCEL_REQUESTED`.
- The worker verifies the launch token and process identities, then asks the
  supervisor to forward `SIGTERM` to the managed child process group.
- The grace deadline is persisted. After ten seconds, the worker may use
  `SIGKILL`, but only against verified managed process groups.
- The job becomes `CANCELLED` only after those groups are gone.

If identity is incomplete, reused, or inconsistent, cancellation remains
pending or fails. GPU Ops never adopts or kills an unmanaged GPU process.

## Storage Budget

GPU Ops may own at most 1 GiB on each host across:

```text
~/.local/share/gpuops
~/.local/state/gpuops
~/.cache/gpuops
```

The partition is 768 MiB for mutable state and 256 MiB for installation plus
cache, with a 128 MiB share-directory cap. State includes a 640 MiB log budget.
Each managed console log is bounded and rotated; worker logs rotate as well.
Cleanup may remove old GPU Ops logs and archived events, but not queue records,
receipts, active log reservations, or experiment-owned files.

If the budget cannot fit a deployment, durable record, or log reservation, that
operation fails closed. Already-running supervisors are not stopped to reclaim
space.

Experiment working directories, datasets, checkpoints, and application logs are
outside the budget and outside GPU Ops cleanup ownership. GPU Ops does not
synchronize these assets or the experiment code and environment; the submitting
Agent must prepare them on the chosen server.

## Rollback And Uninstall

```bash
gpuops deploy rollback --json
gpuops deploy uninstall --json
```

Rollback switches to the retained previous release and verifies that its worker
becomes ready. Uninstall stops the worker and removes only managed release/cache
files, the expected symlink, and `host.json`. It refuses to proceed while a
deployment upload or managed job supervisor is still active, and it does not
delete the durable state directory or experiment data.

`gpuops server remove ID --json` is a separate local disconnect operation. It
refuses removal while any active managed job remains, then stops the Worker and
removes the local registry entry. It retains the remote installation, job
history, queue records, and experiment assets. It does not drain the queue or
wait for running jobs to finish; inspect `queue --server ID` and complete or
cancel the work before removing the server.
