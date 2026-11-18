# GPUOps

[English](README.md) | [简体中文](README.zh-CN.md)

GPUOps monitors Linux NVIDIA servers and runs experiments through a durable
queue. People and terminal Agents use the same CLI to inspect GPUs, submit work,
read bounded logs, recover uncertain submissions, and cancel managed jobs. An
optional Web console runs on your control machine.

GPUOps uses your existing OpenSSH access. Each server runs one user-level Worker
with no listening port. Accepted work continues while the control machine is
offline. The supported setup is a POSIX control machine and Linux NVIDIA servers;
Windows control machines and non-NVIDIA accelerators are outside this release.

## Install

You need Python 3.10+ and OpenSSH (`ssh` and `scp`) on the control machine.
The following download commands apply when v0.5.0 is published; before publication,
use the source installation below. Install the [v0.5.0 release](https://github.com/OHHHHHHei/GPUOps/releases/tag/v0.5.0)
with [uv](https://docs.astral.sh/uv/getting-started/installation/), including the
optional Web console:

```bash
uv tool install 'gpuops[web] @ https://github.com/OHHHHHHei/GPUOps/releases/download/v0.5.0/gpuops-0.5.0-py3-none-any.whl'
gpuops version --json
```

This installs a standalone tool; subsequent `gpuops` commands work from any
directory. If uv reports that its executable directory is not on `PATH`, run
`uv tool update-shell` and reopen the terminal.

Alternatively, download the wheel from the release and install it in your own
virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install './gpuops-0.5.0-py3-none-any.whl[web]'
.venv/bin/gpuops version --json
```

GPUOps is distributed through GitHub Releases and is not published to PyPI.
The installed client does not need uv to deploy Workers. For development:

```bash
git clone https://github.com/OHHHHHHei/GPUOps.git
cd GPUOps
uv sync --all-extras --frozen
uv run gpuops version --json
```

Prefix the following commands with `uv run` when using that development setup.
To upgrade an existing 0.3 installation, follow the one-time
[0.4 upgrade guide](docs/UPGRADE_0_4.md).

## Connect a server

Each server needs:

- Linux, an NVIDIA driver, and `nvidia-smi`.
- Python 3.10+ with `venv`.
- An accessible Python dependency source or cache.
- Home-directory quota for GPUOps' 1 GiB owned-data budget.
- An OpenSSH alias that already works for your account.

```bash
ssh gpu-server
```

Then, on the control machine:

```bash
gpuops server add gpu-server --json
gpuops server list --json
gpuops status --json
```

`server add` reuses a ready installation of the same release, starts a stopped
Worker, and installs the local GPUOps package when the server has no installation
or runs a different release. It adds the registry entry after the Worker is
ready. The alias becomes the server ID unless an existing installation already
has an identity; use the returned `server.id` in job manifests.

`status` reports GPU telemetry and `worker.ready` separately. A ready Worker can
accept an explicit-host job while telemetry is unavailable; the queued job waits
for reliable GPU observations before launch.

## Submit an experiment

Prepare the code, runtime environment, and data on your chosen server. Save a
manifest such as `job.json`, replacing the working directory, command, server ID,
and free-memory requirement with your experiment's values:

```json
{
  "api_version": "gpuops/v1",
  "id": "baseline-seed-42",
  "project": "example",
  "command": ["python", "train.py", "--seed", "42"],
  "working_directory": "/srv/experiments/example",
  "resources": {
    "host": "gpu-server",
    "gpu_count": 1,
    "memory_mib": 18000
  }
}
```

```bash
gpuops submit job.json --json
gpuops job RUN_ID --json
gpuops logs RUN_ID --tail-bytes 65536 --json
gpuops queue --json
gpuops history --json
```

Save the returned `run_id` and check `submission_state: "accepted"`. Submission
validates the manifest automatically. `gpuops validate job.json --json` is
available when you want to check it without submitting.

Omit `resources.gpus` for normal use: the Worker chooses eligible physical GPUs
within the requested host and count. Use `gpus` when you require particular
physical indices. By default jobs run exclusively and require `memory_mib` free
per GPU. For sharing, set `"exclusive": false` and request a positive per-GPU
`memory_mib`. The Worker reserves that amount until release and admits up to two
managed jobs per card by default. GPU memory usage is unrestricted at runtime.

`host: "auto"` also chooses the server. Use it only when every candidate
is authorized and has the experiment assets prepared. Host selection is frozen
before submission and recovery keeps that destination.

If a submission response is unclear, recover the saved request:

```bash
gpuops reconcile RUN_ID --json
# If the CLI exited before showing a Run ID:
gpuops reconcile --list --json
```

Each new `submit` without `--run-id` creates a distinct experiment. Recover an
uncertain request with `reconcile` to preserve its identity. To cancel managed
work, use `gpuops cancel RUN_ID --reason "superseded" --json` and query until it
reaches a terminal state.

## Use any terminal Agent

Any Agent that can invoke a terminal and parse JSON can use GPUOps. Give it the
[Agent guide](docs/AGENT_GUIDE.md) and [CLI contract](docs/CLI_CONTRACT.md). There
is no required Agent vendor, SDK, or MCP server.

Codex users can optionally install the bundled skill on the current machine, or
under the SSH account of a registered server:

```bash
gpuops skill install --json
gpuops skill install --server gpu-server --json
```

The destination is `~/.codex/skills/gpu-infra`, or `$CODEX_HOME/skills/gpu-infra`
when `CODEX_HOME` is set in that account. Agents running as the Worker's Unix
account on a server can call `~/.local/bin/gpuops` from any experiment directory
without a source checkout or an SSH connection back to that server.

## Open the Web console

```bash
gpuops web
```

Open `http://127.0.0.1:8787`. The console shows physical GPU indices, process
ownership, active jobs, paginated history, job explanations, and bounded logs.
`gpuops web --demo` shows deterministic example data. The Web server accepts
loopback binds only; use a trusted tunnel when you need remote access.

ETA is optional. Readable training logs may provide a phase or training estimate,
and `schedule.expected_duration` can supply a manual fallback for managed jobs.
For new training code, [structured progress](docs/PROGRESS.md) provides a small
stdout protocol and an emitter without extra dependencies. Missing, stale, or
insufficient evidence stays unknown.

## Get email when a job finishes

Configure SMTP once per server to receive email when a managed job succeeds,
fails, or is cancelled. The Worker sends the email, so your control machine can
be offline. Existing job manifests need no changes.

```bash
gpuops notify configure smtp.json --server gpu-server --json
gpuops notify test --server gpu-server --json
```

See [Job completion email](docs/NOTIFICATIONS.md) for the SMTP settings and how
to check status or disable email.

For a job whose only human email should be the later follow-up analysis, submit
it with `gpuops submit job.json --email off --json`, then select
`analysis_complete` in the follow-up registration.

## Return to the original chat (experimental)

When registered experiments finish, GPUOps can return to the Codex Desktop,
Claude Code Desktop, or Antigravity chat that registered them. That chat keeps
its context, reads the saved results, performs the requested analysis, and
reports there.

Edit [`examples/followup.example.json`](examples/followup.example.json) and
register it from the original chat:

```bash
gpuops followup register examples/followup.example.json --json
```

GPUOps detects the client and session, so the file needs no `target`. Jobs may
span servers. `per_job` emits one event as each member succeeds, fails, or is
cancelled; `all_terminal` waits for every member and emits one grouped event.
Chat continuation and human email (`none`, `terminal`, or `analysis_complete`)
are chosen independently. The registering Agent supplies the action and context;
GPUOps observes the jobs and delivers the events.

In Codex, `gpuops followup add RUN_ID --action '...' --json` is a shortcut for a
single job. See [follow-ups](docs/FOLLOWUPS.md) for tested client versions,
receive/complete ACKs, email setup, and external adapters.

## Get Worker alerts on macOS

Get a desktop notification when a Worker stops and another when it is ready
again:

```bash
gpuops alerts enable
gpuops alerts test
```

If SSH fails, the status is reported as unknown and no stop notification is
sent. To also receive email, reuse the SMTP file from job completion setup:

```bash
gpuops alerts enable --email-config smtp.json
gpuops alerts test --json
```

Desktop and email delivery are tracked independently, and a failed channel
retries on later checks while the event remains relevant. Use
`gpuops alerts enable --no-email` to return to desktop-only alerts,
`gpuops alerts status --json` to see the latest observations and delivery
errors, and `gpuops alerts disable` to pause both channels. See
[Worker alert email](docs/NOTIFICATIONS.md#worker-alert-email-on-macos) for
details.

## Background checker

Worker alerts and chat follow-ups share one macOS checker. It runs every 60
seconds while either feature has work, batches each server's queries into one
SSH call per round, and unloads when neither needs it. It runs while you are logged in; your Mac must be
awake and able to reach the servers.

`gpuops background status --json` shows which features are enabled and the last
round's SSH calls and errors. `gpuops alerts disable` and
`gpuops followup disable` pause each feature without affecting the other.

## Operating boundary and maintenance

The queue coordinates jobs sharing one trusted Unix account on each server.
External GPU processes block exclusive jobs. Shared jobs account for their
observed usage and can run alongside them if memory is available. The Worker's
leases coordinate jobs in that account. Other users can start programs directly
at any time.
Cancellation only signals identity-verified managed processes.

GPUOps confirms GPU capacity twice before allocation and binds launches to device
UUIDs. Failed devices are retried automatically. GPU telemetry comes from the
Worker's shared samples; repeated queries do not start extra probes.

The Worker needs no Docker, `sudo`, systemd, or server-side HTTP service. It does
not start automatically after a server reboot; use `gpuops fleet start --json`.
Stopping a Worker leaves its experiment supervisors running. GPUOps owns at
most 1 GiB per server for installation, cache, durable state, and bounded logs;
experiment outputs and datasets stay with your project.

The control machine, server CLIs, and running Workers use the same release; a
mismatch returns `version_mismatch`. To upgrade a registered server:

```bash
gpuops deploy install --server gpu-server --json
gpuops deploy status --server gpu-server --json
```

Other maintenance commands are `gpuops deploy rollback|uninstall` and
`gpuops fleet start|stop|status`. To disconnect a server after its managed work
has finished, use `gpuops server remove gpu-server --json`; this stops its Worker
and removes the local registration while preserving its installation, records,
and experiment data. It refuses active jobs. See
[Remote Worker](docs/REMOTE_WORKER.md) for deployment and rollback details.

## Documentation and development

- [Agent workflow](docs/AGENT_GUIDE.md) and [CLI compatibility contract](docs/CLI_CONTRACT.md)
- [Architecture](docs/ARCHITECTURE.md) and [Remote Worker operations](docs/REMOTE_WORKER.md)
- [Roadmap and acceptance procedure](docs/ROADMAP.md)
- [Release validation](docs/VALIDATION.md): CI coverage, real-server checks, and limitations
- [Contributing](CONTRIBUTING.md) and [Changelog](CHANGELOG.md)

Run the same checks as CI, then build the wheel:

```bash
uv sync --all-extras --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
node --test tests/test_web_polling.cjs
uv build --wheel
```

Browser polling tests require Node.js 22+. Linux process tests use synthetic GPU
observations and do not require physical GPUs. GPUOps is released under the
[MIT License](LICENSE).
