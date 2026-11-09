# Changelog

Changes are recorded here by release. See [release validation](docs/VALIDATION.md)
for tested behavior and limits.

## Unreleased

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
