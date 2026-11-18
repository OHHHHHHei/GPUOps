# Changelog

Changes are recorded here by release. See [release validation](docs/VALIDATION.md)
for tested behavior and limits.

## 0.5.0 — Unreleased

- Allow jobs to share a GPU with `resources.exclusive: false`. The Worker
  reserves each job's per-card `memory_mib`, accounts for external usage, and
  defaults to two managed jobs per GPU. Status, queue, and job queries show
  the allocations and per-job requests.
- Upgrade the controller and Workers together. Version 0.5 reads existing 0.4
  records. Version 0.4 rejects records containing `exclusive: false` or the
  new launch fields `start_ticks` and `boot_id`. An exclusive job started on
  0.5 can therefore block rollback to 0.4 even after the job finishes.

## 0.4.0 — Unreleased

- Sample each GPU every five seconds as designed (it had slipped to ten) with
  sampling children that import only the standard library, and check CUDA just
  before leasing instead of on every idle sample. An idle 8-GPU server spends
  roughly a quarter of the previous CPU on sampling.
- Remove the queue event log that nothing read, skip storage scans below the
  cleanup watermark, skip fsync for heartbeat and GPU sample files, and drop
  the extra health call before each submission.
- Fix `reconcile` on an installed server or a controller with no registered
  servers. Use one GPU reservation rule, including UUIDs and expired leases, for
  scheduling, job explanations, and automatic host selection.
- Stop Web console polling while its tab is hidden and turn off its access log.

- Resume original Claude Code Desktop and Antigravity chats through the shared
  checker. Infer desktop/session identity during registration and receipt;
  return `first_receipt` to distinguish new work from a duplicate message.
  See [tested desktop boundaries](docs/FOLLOWUPS.md).

- Require one published release across controllers, server CLIs and running
  Workers. Report a version mismatch instead of probing optional features or
  retrying older RPC variants; `server add` upgrades a different installed release.
- Return and persist complete execution-result fields. Remove the unused
  host-wide GPU probe and migration sampling barrier; launches require GPU UUIDs.
- Replace hidden host arguments with `submit --server` and `deploy --ssh-alias`;
  remove `queue --all`, and report `doctor.telemetry_ok`. Remove the `--host`
  aliases: use `--server` for `job`, `logs` and `cancel`, and `web --bind`.
- Add `deploy install --no-start` to install a release while keeping the Worker
  stopped, as used by the 0.4 upgrade.

- Remove the unused `tracking` field from `gpuops/v1` manifests. The [0.4
  upgrade](docs/UPGRADE_0_4.md) converts saved records and fills in explicit
  result fields and single-job follow-up outcomes.

- Add lightweight `watch` batches and share one on-demand macOS checker between
  Worker alerts and follow-ups. Expose independent switches and last-round
  statistics; network delivery no longer holds the follow-up record lock.

- Add optional email to macOS Worker stop/recovery alerts. Reuse an SMTP settings
  file with `gpuops alerts enable --email-config`; desktop and email delivery
  retry independently. No changes to experiment manifests or remote Workers.

## 0.3.0 — 2026-11-07

- Add optional server-side completion email for succeeded, failed, and cancelled
  jobs. Configure SMTP once with `gpuops notify`; experiment manifests stay the
  same. Sending runs outside the scheduling loop and retries from saved progress.
- Start notifications from first enable, skipping older history; preserve delivery
  progress across Worker restarts. SMTP acceptance does not guarantee inbox delivery.
- Include notification setup with the installed Agent guides and test the command
  from independently installed packages.
- Include the macOS Worker stopped/recovered desktop alerts added after 0.2.1.

## 0.2.1 — 2026-10-23

- Fix history table project cells losing table layout on desktop, which shifted
  their text and row borders. Preserve long-name truncation and the stacked
  layout on narrow screens.

## 0.2.0 — 2026-10-23

First public release.

- Make first-time installation, SSH-alias registration, and experiment
  submission the primary project workflow.
- Support a control CLI installed from a built wheel, with deployment and
  optional skill resources available outside a source checkout.
- Separate server registration from optional Codex skill installation.
- Document the terminal-Agent CLI contract and expose machine-readable version
  information.
- Return observed process exit codes and launch errors as structured task results,
  preserving the RPC v1 job shape for older clients. Unknown historical results
  remain null. Persisted terminal results require a compatible Worker on rollback.
- Test installed clients outside the checkout with both pip and uv, including
  source-archive deployment and optional skill installation.
- Isolate Worker environments from system Python packages, fixing source
  installation on hosts whose system `packaging` predates SPDX license support.
  Exercise source deployment with pip 22.0.2 in Linux CI.
- Keep storage accounting working when a sampling or atomic-update file disappears
  during enumeration, instead of failing an otherwise valid queue operation.
- Use generic server and experiment examples, document the shared Unix-account
  trust boundary, and adopt the MIT License.

## 0.1.0 development baseline

- GPU monitoring from shared Worker samples, independent per-device retries,
  and UUID-bound allocation.
- Durable queued jobs, bounded logs, cancellation, Worker recovery, and
  submission reconciliation.
- Active-job summaries, paginated history, job explanations, and a loopback Web
  console with optional log-based progress estimates.
- Registered SSH servers and server-local access through the installed CLI.

The baseline describes implemented functionality; it is not a claim that a
package was published to PyPI or that every server passed live acceptance.
