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
Install the [v0.2.1 release](https://github.com/OHHHHHHei/GPUOps/releases/tag/v0.2.1)
with [uv](https://docs.astral.sh/uv/getting-started/installation/), including the
optional Web console:

```bash
uv tool install 'gpuops[web] @ https://github.com/OHHHHHHei/GPUOps/releases/download/v0.2.1/gpuops-0.2.1-py3-none-any.whl'
gpuops version --json
```

This installs a standalone tool; subsequent `gpuops` commands work from any
directory. If uv reports that its executable directory is not on `PATH`, run
`uv tool update-shell` and reopen the terminal.

Alternatively, download the wheel from the release and install it in your own
virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install './gpuops-0.2.1-py3-none-any.whl[web]'
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

## Connect a server

Each target needs Linux, an NVIDIA driver, `nvidia-smi`, Python 3.10+ with `venv`,
an accessible Python dependency source or cache, and available home-directory
quota for GPUOps' 1 GiB owned-data budget. Use an OpenSSH alias that already
works for your account:

```bash
ssh gpu-server
```

Then, on the control machine:

```bash
gpuops server add gpu-server --json
gpuops server list --json
gpuops status --json
```

`server add` reuses a ready installation, starts a stopped Worker, or installs a
missing Worker from the local GPUOps package. It waits for readiness before
adding a registry entry. Reusing a ready installation does not upgrade it. The
alias becomes the server ID unless an existing installation already has an
identity; use the returned `server.id` in job manifests. A fresh registry is
empty. Registration does not install an Agent skill or copy your experiment
code, environment, or data.

`status` reports GPU telemetry and `worker.ready` separately. A ready Worker can
accept an explicit-host job while telemetry is unavailable; the queued job waits
for reliable GPU observations before launch. Monitoring exit code zero alone
does not prove scheduling readiness.

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
physical indices. `memory_mib` is required free memory per GPU, not a runtime
limit. `host: "auto"` also chooses the server; use it only when every candidate
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

Codex users can optionally install the bundled skill on the current machine:

```bash
gpuops skill install --json
```

Or install it under the SSH account of a registered server:

```bash
gpuops skill install --server gpu-server --json
```

The default destination is `~/.codex/skills/gpu-infra` in the selected account.
When `CODEX_HOME` is set in that account's command environment, the destination
is `$CODEX_HOME/skills/gpu-infra`. The remote operation is optional; connect a
server first if it is not registered. Agents
running as the Worker Unix account on an installed server can use
`~/.local/bin/gpuops` from any experiment directory without a source checkout or
an SSH connection back to the current server. Existing older server CLIs may
lack `version` or `server list`; they can continue using `status`, `queue`, and
`submit`. Use the control client for version discovery and installation
maintenance.

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
insufficient evidence stays unknown. ETA does not reserve GPUs or control
scheduling.

## Operating boundary

The queue coordinates jobs sharing one trusted Unix account on each server.
The Worker avoids GPUs occupied by visible unmanaged processes, including other
users' work. Its leases coordinate GPUOps jobs in that account; they do not
enforce exclusivity against other Unix users or programs that bypass GPUOps.
Cancellation only signals identity-verified managed processes.

GPUOps confirms idle GPUs twice before allocation and binds launches to device
UUIDs. Failed devices are retried automatically, but GPUOps cannot repair an
NVIDIA driver or isolate its shared kernel locks. GPU telemetry always comes
from the Worker's shared samples; repeated queries do not start extra probes.

The Worker needs no Docker, `sudo`, systemd, or server-side HTTP service. It does
not start automatically after a server reboot; use `gpuops fleet start --json`.
Stopping a Worker leaves its experiment supervisors running. GPUOps owns at
most 1 GiB per server for installation, cache, durable state, and bounded logs;
experiment outputs and datasets remain your responsibility.

For maintenance, use `gpuops deploy status|install|rollback|uninstall` and
`gpuops fleet start|stop|status`. To disconnect a server after its active managed
work has finished, use `gpuops server remove ID --json`; this stops its Worker
and removes the local registration while preserving its installation, records,
and experiment data. It refuses active jobs and does not wait for them to drain.
See [Remote Worker](docs/REMOTE_WORKER.md) for deployment and rollback details.

## Documentation and development

- [Agent workflow](docs/AGENT_GUIDE.md) and [CLI compatibility contract](docs/CLI_CONTRACT.md)
- [Architecture](docs/ARCHITECTURE.md) and [Remote Worker operations](docs/REMOTE_WORKER.md)
- [Roadmap and acceptance procedure](docs/ROADMAP.md)
- [Release validation](docs/VALIDATION.md): CI coverage, real-server checks, and limitations
- [Contributing](CONTRIBUTING.md) and [Changelog](CHANGELOG.md)

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
node --test tests/test_web_polling.cjs
uv build --wheel
```

Browser polling tests require Node.js 22+. Linux process tests use synthetic GPU
observations and do not require physical GPUs. Real deployment and CUDA
acceptance are separate from local test results. GPUOps is released under the
[MIT License](LICENSE).
