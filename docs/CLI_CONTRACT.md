# Agent CLI contract

GPUOps accepts prepared commands and manages their GPU execution lifecycle. Any
Agent with terminal access can use this interface. The optional Codex skill is
guidance, not a runtime dependency. Commands execute with the SSH account's
permissions; GPUOps does not sandbox untrusted code.

## Versions and compatibility

Run `gpuops version --json` locally to discover these independent versions:

```json
{
  "version": "0.3.0",
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
`job_not_found`, `storage_quota_exceeded`, `unsupported_option`, and `io_error`.
Other codes remain possible; automatic retry is not implied by an unfamiliar
code.

For `worker run` and `worker start`, a zero or negative `--interval` is rejected
before starting a Worker. With `--json`, stdout contains
`{"ok": false, "error_code": "invalid_argument", "error": "--interval must be positive"}`,
stderr is empty, and the exit status is 2. Without `--json`, the same error is
printed as human-readable text on stderr.

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

On macOS, `alerts status --json` reports whether the local poller is loaded
(`enabled`), the last completed check (`checked_at`, nullable), and the last
observations (`servers`). Each observation has a `server_id` and a `status`:
`ready`, `stopped`, `not_ready`, or `unknown` when the status query failed.
Check `checked_at` before treating the observations as current. This command
reads local state only; use `fleet status` for a fresh query.

`alerts enable --email-config CONFIG_FILE` adds local Worker stop/recovery email
using the same SMTP file format as `notify configure`. `alerts enable --no-email`
removes local email settings while keeping desktop alerts. These options are
mutually exclusive and valid only with `enable` (other actions exit 2 with
`invalid_argument`). Plain `enable` preserves saved settings; `disable` stops
both channels and retains email settings.

Alert results include `email_configured`, `email_recipient` (nullable), and
`email_error` (nullable, for configuration errors or test delivery failure).
Server observations retain `stop_notified` for desktop delivery and add
`email_stop_notified`, plus optional `notification_error` and `email_error` for
the latest delivery attempt. Delivery is independently deduplicated per channel.
`alerts test --json` attempts both configured channels and returns `ok`,
`notification_submitted`, optional `notification_error`, and `email_accepted`
when email is configured. It exits 1 on a channel or email configuration failure.
SMTP acceptance does not prove inbox delivery. Credentials are omitted.

`notify configure CONFIG_FILE`, `notify status`, `notify test`, and `notify disable`
manage optional server-side completion email. With `--json`, they return `ok`,
`outcome`, and `servers[]`, with `server_id`, `online`, `ok`, and either `result`
or `error_code`/`error` per server. They use the same target defaults as other
fleet commands. `configure -` reads SMTP settings from stdin.

Notification status includes `configured`, `enabled`, `enabled_since`, public
SMTP settings, `cursor`, `retrying_run_id`, `last_error`, `next_attempt_at`,
`last_sent_at`, `sent_count`, `config_error`, and `state_error`. It reads saved
configuration and delivery state; it does not prove the Worker is running.
Credentials are omitted. A test result of `accepted: true` means SMTP accepted
the message, not that it reached an inbox. See [notification setup](NOTIFICATIONS.md).
Email is independent of job submission and never changes the job's result or
Worker readiness.

`submit JOB_FILE --email default|off` overrides the job manifest's
`metadata["gpuops.email"]`. `off` suppresses server-side completion email for
that job without affecting follow-up email. The client requires the remote
Worker health response to advertise `capabilities.job_email_policy: true`;
otherwise submission is definitively rejected before enqueue with
`unsupported_option` and the Worker must be upgraded. `default` removes the
per-job override and uses the Worker's configured policy. Automatic placement
does not filter candidates by this optional capability: upgrade all eligible
Workers before using `off`, or set `resources.host` to an upgraded Worker.

## Experimental completion groups and session follow-ups

`followup add RUN_ID --action TEXT --json` registers one job for the current
Codex Desktop chat and enables a local macOS checker. Routing is persisted on
this client. `--thread UUID`, `--workspace DIR`, and `--server ID` provide
explicit context when it cannot be inferred.

`followup register FILE --json` accepts one strict `gpuops.followup/v1` JSON
document (or `-` for stdin). It contains `jobs[]` (`run_id` and optional
`server_id`), a `target` (`harness`, optional `session_id`, optional
`workspace`), `mode`, `action`, string-valued `context`, `resume`, `notify`, and
optional `name`. `mode` is `per_job` or `all_terminal`. Success, failure, and
cancellation all satisfy a member's terminal boundary. Members may resolve to
different servers.

`resume` independently controls Harness delivery. `notify` is `none`,
`terminal`, or `analysis_complete`; the last requires session continuation and
a completed analysis summary. `resume: false` records an observation without
Harness delivery. Non-`none` registration requires client-side follow-up SMTP
settings installed with `followup configure --email-config FILE --json` (or the
local Worker-alert SMTP fallback).

Registration enables the macOS LaunchAgent. `followup check --json` publicly
runs one check, but registration still requires macOS; this is not yet a public
Linux manual-scheduling workflow. `followup capabilities --json` returns the
registration/event API versions, modes, notification phases, adapter
capabilities, email configuration state, `background_platform`, and
`codex_unloaded_session_wakeup: false`. `followup schema --json` returns the
registration JSON Schema.

`followup adapter NAME FILE --json` installs an external adapter definition
with `command[]` and `idempotent`. The adapter receives one
`gpuops.completion/v1` JSON document on stdin and must return a
`gpuops.delivery/v1` acceptance document with matching stable `event_id` and a
durable `receipt`. The request includes `group_id`, target, mode, action,
context, job snapshots, message, and `trigger_job`; `trigger_job` identifies the
member for `per_job` and is `null` for `all_terminal`. Adapter acceptance is not
model receipt or completion. The exact request/response contract, timeout, and
idempotency limits are in [the follow-up contract](FOLLOWUPS.md). A configured
adapter does not establish native Claude Code, DeepSeek, or original-session
support; custom continuity is reported as `unverified`.

`add`, `register`, `show ID`, `acknowledge ID`,
`complete ID --summary TEXT`, `stop ID`, and `retry ID` return `ok` plus the
result. `show` and `stop` accept group or event IDs; ACK, completion, and retry
use event IDs. For an external Harness, ACK/completion/retry accept `--session`
and require it to equal the registered session. `--session` validates routing;
it does not establish a connection.

`list` returns `ok`, `followups[]` (without saved job/message bodies),
`groups[]`, and `invalid_records[]` (`path` and `error`; other records still
run). Event records add group, mode, context, outcomes, resume/notify, email
(`email_sent_at`, `email_error`, `email_attempts`, `email_next_attempt_at`),
receipt, and summary fields. Group summaries and their event rows omit saved
job/message bodies; `show EVENT_ID` returns the full event. Group summaries
expose members, planned/created events, `all_terminal`, `completed`,
`completion_kind`, and aggregate status. Explicitly stopping the group yields
`stopped`; stopping only events derives `delivery_stopped` when all are stopped
or `partially_stopped` while another planned event remains active.
`queued` means only delivery acceptance; `received` and `completed` are
explicit target-session ACKs. `observed` means no Harness continuation was
requested. None of these states changes experiment status.

Uncertain Codex or non-idempotent adapter delivery requires inspection before
explicit retry. An adapter marked `idempotent` is trusted to deduplicate the
stable event ID and may be retried automatically; GPUOps does not verify that
claim. The existing Codex Desktop loaded-chat limitation remains: queue
acceptance does not wake an unloaded original chat. New grouped paths are
locally tested but have not received a new live deployment acceptance.

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
