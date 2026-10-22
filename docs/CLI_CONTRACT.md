# Agent CLI contract

GPUOps accepts prepared commands and manages their GPU execution lifecycle. Any
Agent with terminal access can use this interface. The optional Codex skill is
guidance, not a runtime dependency. Commands execute with the SSH account's
permissions; GPUOps does not sandbox untrusted code.

## Versions and compatibility

Run `gpuops version --json` locally to discover these independent versions:

```json
{
  "version": "0.2.0",
  "cli_api_version": "gpuops.cli/v1",
  "job_api_version": "gpuops/v1",
  "worker_api_version": "gpuops.worker.rpc/v1"
}
```

`version` identifies the installed client release. It is not evidence of the
release running on a server; use `gpuops deploy status --server ID --json` for
remote deployment information. `worker_api_version` identifies the wire protocol
the client implements, not every optional feature available on an older Worker.

The CLI contract preserves the command-specific JSON shapes below rather than
wrapping existing arrays in a new envelope. Within `gpuops.cli/v1`, documented
fields retain their meanings and types. Clients should tolerate additional
response fields and handle unknown error/explanation codes as an unknown case.
A required-field removal, type change, or changed meaning requires a new CLI
contract version. New optional fields may appear in a later release. Job
manifests remain strict: unrecognized input fields are rejected.

## Output and exit status

With `--json`, stdout contains one JSON document, possibly formatted over several
lines. Human-facing output without `--json` is not the automation contract.

- Exit **0** means the requested operation succeeded. `job` can return 0 for a
  successfully queried job whose status is `failed`; inspect the job itself.
- Exit **1** means an operation failed, or a fleet query returned partial results.
  Parse the available JSON before deciding what to do next.
- Exit **2** indicates command-line parsing/selection errors. Some validation
  failures detected during execution use exit 1; `error_code`, not the numeric
  exit status alone, classifies the failure.
- `status` and `doctor` retain their monitoring exit semantics. A successful GPU
  observation does not establish scheduling readiness; read `worker.ready` in
  `status`, or let `submit` perform the readiness check.

The normal error object contains `ok: false`, `error_code`, and a human-readable
`error`. Submission errors additionally expose `submission_state` and, once
assigned, `run_id` and `server_id`. Do not parse the English error message as a
machine-readable category. Known codes include `invalid_argument`,
`invalid_manifest`, `host_unavailable`, `worker_unavailable`, `run_id_conflict`,
`job_not_found`, `storage_quota_exceeded`, and `io_error`. Other codes remain
possible; automatic retry is not implied by an unfamiliar code.

## Core operations

| Command with `--json` | Stable result fields / shape | Interpretation |
| --- | --- | --- |
| `version` | Version object above | Local discovery; no network calls. |
| `server list` | `ok`, `servers[]` with `id`, `label`, `ssh_host` | Local controller registry. |
| `server add SSH_ALIAS` | `ok`, `server`, `installed`, `ready` | Registers only after Worker readiness; `installed: false` means a deployment was reused. |
| `submit JOB_FILE` | `ok`, `run_id`, `server_id`, `submission_state`, `durable`, `reconciled`, `job` | Keep the Run ID. Acceptance means durably queued, not necessarily running. |
| `queue` | `ok`, `outcome`, `view: "active"`, `servers[]` with `server_id`, `online`, `jobs` | Active summaries only; a failed server carries its error rather than proving an empty queue. |
| `history` | `ok`, `outcome`, `jobs`, `projects`, `servers`, `next_cursor` | Recent terminal summaries. Continue using the same filters. |
| `job RUN_ID` | `ok`, `server_id`, `job`, `explanation` | Full task state, execution result when available, and current waiting/failure explanation. |
| `logs RUN_ID` | `ok`, `server_id`, `content` and log metadata | Bounded retained output; not a promise of the entire original log. |
| `cancel RUN_ID` | `ok`, `server_id`, `durable`, `job` | A durable cancellation request. Query until terminal. |
| `reconcile RUN_ID` | Same accepted result as submit, or a submission error | Reuses the saved request and original destination. |
| `reconcile --list` | `ok`, `pending` | Local unresolved submissions; no network calls. |
| `status` | Array of server observations, each with `server`, `online`, `data`, `worker` | `worker.ready` is independent of GPU collection. |

Fleet operations use `outcome: "success"`, `"partial"`, or `"failed"`.
Partial results have `ok: false` and exit 1 while retaining successful hosts.
History suppresses `next_cursor` on partial results to avoid skipping records.
`queue --all` is a compatibility view of full job records, not the active-summary
shape. Raw GPU telemetry and human messages may gain device-specific fields;
Agents should use documented state/readiness and job fields for decisions.

## Job lifecycle and execution results

Active states are `queued`, `leased`, `starting`, `running`, and
`cancel_requested`. Terminal states are `succeeded`, `failed`, and `cancelled`.
`job.status` is authoritative; a progress record or ETA is not completion.

Full records returned by `job` include nullable `exit_code` and `launch_error`:

- `exit_code: 0` is an observed successful process exit.
- Positive exit codes identify process failure; a negative POSIX return code
  indicates termination by that signal number.
- `launch_error` describes a captured failure to start or supervise the process.
  A nonzero exit code by itself does not distinguish a launch error from an
  application error.
- `null` means no structured evidence is available. This includes older Worker
  responses, old records, queued jobs cancelled before launch, and a supervisor
  that disappeared without recording its result. Do not reconstruct an exit
  code by parsing `failure_reason`.

`failure_reason` remains a human-readable explanation. Summary/history records
need not carry execution-result fields; query `job RUN_ID` for details.

RPC v1 keeps its original nested `job` record shape for older clients. Updated
Workers include an optional sibling `execution_result` in `show`, `submit`, and
`cancel` responses;
updated clients expose those fields in the CLI job record. This does not require
a new RPC request or forced upgrade of an already healthy Worker.

New terminal records can persist these result fields. An older release with a
strict record model may not read them. Deployment rollback checks persisted
record compatibility before switching Workers and refuses incompatible rollback.
Do not manually rewrite records to force an old Worker to start.

## Submission identity and recovery

`submission_state` is one of:

- `accepted`: the remote queue confirmed durability. Keep this Run ID even if a
  `journal_warning` says local bookkeeping needs repair.
- `not_accepted`: this attempt was definitively rejected. Inspect its error;
  permanent input errors need a corrected request, not blind retries.
- `unknown`: acceptance was not established. Reconcile the existing Run ID;
  creating another submission can run the experiment twice.

`gpuops submit JOB_FILE --run-id ID --json` reuses an explicit identity. Retrying
the same identity and payload is idempotent. Reusing an identity with different
input is a conflict. A new `submit` without `--run-id` creates a new experiment,
even if its manifest is identical. Run IDs and records are preserved in history.

`reconcile` uses the journal on the client that submitted the task. Mac/Linux
controller and server-local journals are separate. A successful submission is
not dependent on the submitting Agent remaining connected. Worker recovery
reattaches to recorded execution; it does not automatically rerun an interrupted
experiment from the beginning.

## Resource and trust boundary

Use one Worker and queue per trusted Unix account on a server. Its leases
coordinate the tasks in that queue. GPUOps observes other users' GPU processes
but cannot atomically reserve hardware against another user's Worker or a
directly launched command. `CUDA_VISIBLE_DEVICES` selects devices for cooperating
programs; it is not a permission boundary or system-wide exclusive GPU lock.

The Agent prepares code, runtime environments, data, and checkpoints before
submission. GPUOps schedules the command already present on the selected server.
Keep automatic GPU selection as the default, and use the resource scope the
user has authorized. No additional per-GPU authorization step is required.
