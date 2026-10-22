# Project Status and Roadmap

GPUOps focuses on monitoring and queued experiments on Linux NVIDIA servers,
controlled from a POSIX machine. Experiment code, environments, data, metrics,
and checkpoints remain in the experiment project. The
[Agent guide](AGENT_GUIDE.md) describes daily use; the
[CLI contract](CLI_CONTRACT.md) defines terminal-Agent integration.

## Release Status

GPUOps 0.2.0 is the first public release, distributed through
[GitHub Releases](https://github.com/OHHHHHHei/GPUOps/releases).
It provides a general SSH-alias setup flow, deployment from an installed package,
and a documented terminal-Agent CLI contract. The optional Codex skill adapts
that same interface.

[Release validation](VALIDATION.md) covers Linux CI, independent installations,
and controlled real-server execution. Every new server still needs its own
acceptance check. GPUOps is not published to PyPI.

## Existing Capabilities

- Shared GPU observations, independent per-device sampling and retries, device
  UUID binding, PCI checks, and idle-only CUDA initialization checks.
- Durable queued submissions, bounded logs, managed cancellation, detached
  supervisors, Worker recovery, and reconciliation of uncertain submissions.
- Registered SSH servers and direct local CLI access on an installed server.
- Active-job summaries, paginated history, project filtering, and explanations
  for waiting or failed work.
- A loopback Web console with independent refresh of GPU, queue, health, and
  optional log-based progress estimates.

Queue coordination belongs to one trusted Unix account. Other users' processes
are observed and avoided, without enforced GPU exclusivity against them.
Device retries cannot repair an NVIDIA driver or isolate its shared kernel
locks. ETA remains a planning aid and does not reserve resources.

## Follow-up Work

- Validate the complete CUDA lifecycle on each new physical server using the
  procedure below. Historical checks on another machine do not transfer.
- Integrate optional structured progress in a controlled training run and compare
  its phase estimate with observed completion and GPU release, including
  validation, checkpoint saving, and early stopping.
- Improve Web queue progress without requiring each job detail to be opened,
  using bounded log reads and the existing progress evidence rules. Queue rows
  currently reuse samples from recently visited running jobs.

## Outside the Current Release

- MCP servers, Kubernetes integration, Windows control machines, and
  non-NVIDIA accelerators.
- OS-level GPU isolation, a multi-tenant authorization service, and control of
  unmanaged processes.
- Code/data/environment synchronization and metric or checkpoint management.
- Guaranteed queued start times, advance reservations, and ETA calibration from
  job history. Existing schedule windows are eligibility checks, not resource
  availability promises.
- Automatic reboot startup and automatic reruns of interrupted experiments.
- A project-specific log-parser plugin framework; the small stdout progress
  protocol and generic log heuristics remain the integration path.

## Remote Acceptance Gate

Use this procedure when changing remote execution or deployment behavior.
Documentation and local Web changes do not require Worker installation or GPU
consumption. Use an authorized test server and experiment.

1. Read `doctor`, `deploy status`, and `status --json`. Record the release,
   Worker readiness, and GPU observations. After deployment, confirm installed
   and Worker releases, fresh readiness, and the reported storage budget through
   the public CLI.
2. For a controlled canary with an explicit host and a one-GPU request, allow
   automatic allocation and verify the assigned physical device against the
   device visible to CUDA. Check live logs, completion without a following
   queued job, same-Run-ID deduplication, and allocation reuse.
3. Check running cancellation and Worker stop/restart recovery while preserving
   unrelated processes. Confirm that stopping the Worker leaves the child
   running and that restart recovers its original result.
4. For deployment changes, check rollback/restoration. Add the date, release,
   run IDs, results, and unverified checks to a private acceptance record.
   Share a redacted summary when contributing validation evidence.

An unavailable GPU or ambiguous result leaves its check unverified. Use public
GPUOps operations rather than editing queue files or bypassing the Worker.
