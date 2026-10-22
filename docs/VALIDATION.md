# Release Validation

GPUOps 0.2.0 has been validated with automated tests and a controlled reinstall
on a shared Linux NVIDIA server. This summary omits private host identities,
usernames, experiment paths, and process records.

## Automated Checks

The CI workflow runs the Python suite and frontend polling tests on Linux with
Python 3.10 and 3.12. Installed-client checks cover Linux 3.10, Linux 3.12, and
macOS 3.12 through both pip and uv.

The distribution checks install a wheel outside the source checkout, deploy a
Worker source archive into a separate environment, verify bundled resources,
and exercise optional skill installation. Linux 3.10 additionally tests source
installation using pip 22.0.2. Worker environments do not inherit system Python
packages. These automated tests need no GPU; they use synthetic observations
for scheduling and real local processes for lifecycle checks.

## Controlled Real-server Checks

A previously installed Worker was uninstalled and reinstalled through its SSH
alias using a wheel-installed client outside the checkout. Historical state was
retained and the server-local CLI worked from an unrelated directory.

A one-GPU manifest with automatic allocation exercised CUDA context creation,
a 1 MiB host/device memory round trip, and cleanup. The CUDA-visible device UUID
matched the Worker assignment. The following checks passed:

- Growing logs and normal completion without a following task.
- Same-Run-ID submission without a second launch, followed by allocation reuse.
- Managed cancellation with SIGTERM, CUDA cleanup, and an observed exit result.
- Child completion while the Worker was stopped; restart collected the original
  result without rerunning the experiment.
- A subsequent CUDA task after recovery.
- Intentional nonzero exit and a structured missing-executable launch error.
- Rollback and restoration between compatible release candidates.

Pre-existing GPU processes retained their process identities. No unrelated
training was cancelled, and the other server's Worker was left unchanged.
No acceptance tasks remained active after the checks.

## Findings Included in This Release

- Isolating Worker virtual environments fixed source builds that accidentally
  imported an older system `packaging` module.
- Storage accounting now tolerates sampling files disappearing between
  enumeration and size reads. Other I/O errors continue to propagate.

## Limits

These checks cover a small CUDA workload on an existing server with retained
history. They do not establish full-memory behavior, training throughput,
server-reboot recovery, or compatibility with every NVIDIA driver. A new host
needs the [remote acceptance procedure](ROADMAP.md#remote-acceptance-gate).
Rollback to an older Worker that cannot read new persisted records remains
subject to the existing compatibility check.
