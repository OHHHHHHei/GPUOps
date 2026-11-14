# Completion groups and original-session follow-ups (experimental)

GPUOps can persist completion intent for one or more jobs, observe terminal
results across registered servers, and independently deliver:

- a completion event to a selected Harness session;
- a human email when the experiment reaches a terminal state; or
- a human email after the Harness records its analysis as complete.

GPUOps does not invent the research task. The caller supplies `action`, optional
string-valued `context`, the job references, and the routing target. GPUOps
observes execution results and carries that caller-provided material to the
target; the original Agent remains responsible for reading saved evidence,
performing the analysis, updating research records, and reporting findings.

The earlier single-job Codex Desktop loop passed live acceptance on 2026-11-13:
the loaded original chat started a new turn, read saved metrics over SSH, and
reported its analysis. Cross-server groups, independent follow-up email, and the
generic adapter contract are covered by local tests on the current feature
branch, but have not received a new live deployment acceptance yet.

## Register work

The original single-job command remains available:

```sh
gpuops followup add RUN_ID \
  --action 'Read metrics.json, compare the saved baseline, update the research record and report here.' \
  --json
```

For groups, prepare a strict `gpuops.followup/v1` JSON document and register it:

```sh
gpuops followup register examples/followup.example.json --json
# Use - instead of a path to read one JSON document from stdin.
```

```json
{
  "api_version": "gpuops.followup/v1",
  "name": "ablation-round-1",
  "jobs": [
    {"run_id": "baseline-01", "server_id": "gpu-a"},
    {"run_id": "candidate-01", "server_id": "gpu-b"}
  ],
  "target": {
    "harness": "codex-desktop",
    "workspace": "."
  },
  "mode": "all_terminal",
  "action": "Compare the saved metrics and report the conclusion in this chat.",
  "context": {
    "research_question": "Does the candidate improve validation accuracy?",
    "metrics_file": "metrics.json"
  },
  "resume": true,
  "notify": "none"
}
```

Each `jobs` entry requires a `run_id`. `server_id` may be omitted when this
client's submission journal records the original destination; otherwise it is
required. A supplied server cannot conflict with the journal. Members may span
different registered SSH servers.

For `codex-desktop`, `target.session_id` defaults to `CODEX_THREAD_ID`. Other
Harness names require an explicit session ID when `resume` is true. `workspace`
defaults to the current directory and must resolve to an existing local project
directory. `action` is required when `resume` is true. `context` is an optional
map of strings; it is passed through as context, not interpreted as new task
instructions by GPUOps.

Registration is durable and idempotent. A named group is matched by name plus
Harness/session/workspace; an unnamed group is matched by its members plus that
target. Repeating the same definition returns the existing IDs. Changing its
members, action, mode, notification, resume flag, or context is rejected. A
stopped named group needs a new name.

Registration enables follow-ups and starts the shared macOS LaunchAgent,
`io.gpuops.background`, when work is pending. It checks every 60 seconds and
shares each server's `watch` query with Worker alerts. Registration and
automatic scheduling require macOS.

## Group modes and terminal results

Both modes treat `succeeded`, `failed`, and `cancelled` as terminal evidence:

- `all_terminal` preallocates one stable event ID and waits until every member
  is terminal. It then emits one event containing every observed outcome.
- `per_job` preallocates one stable event ID per member and emits that member's
  event as soon as it becomes terminal. The event also contains the group's
  latest observations, including members that are still nonterminal.

An unavailable server, a missing job, or a nonterminal status is not completion.
The error is retained on the member and checked again later. Existing terminal
observations survive checker restarts and events are not recreated.

Use these commands to inspect groups and their planned/created events:

```sh
gpuops followup list --json
gpuops followup show GROUP_ID --json
gpuops followup show EVENT_ID --json
gpuops followup check --json
```

`followup check` runs one follow-up round with the same switch and lock as the
scheduled checker. It skips Worker alerts and exits if another round is already
running. Use `gpuops background status --json` for the last round's elapsed time,
SSH calls, queried servers and jobs, pending work, and errors.

`show` and `stop` accept either a group ID or an event ID. `retry`,
`acknowledge`, and `complete` operate on event IDs. Stopping a group prevents
future events and stops existing events only before delivery has started; it
cannot recall a message already accepted by a Harness adapter. An explicit
`stop GROUP_ID` reports group status `stopped`. Stopping an individual event
instead derives `delivery_stopped` when all planned events are stopped, or
`partially_stopped` when other group events remain active. Group summaries omit
the saved job/message bodies; use `show EVENT_ID` for the full event.

## Session continuation and human notification are independent

`resume` controls Harness delivery. `notify` controls human email:

| Setting | Result |
| --- | --- |
| `resume: true` | Deliver an event to the selected Harness and require explicit receive/complete ACKs. |
| `resume: false` | Record the terminal observation without contacting a Harness. `action` and `session_id` may be omitted. |
| `notify: "none"` | No follow-up email. |
| `notify: "terminal"` | Email when the event's job set reaches its mode-specific terminal boundary. |
| `notify: "analysis_complete"` | Email only after `followup complete` records a nonempty analysis summary. Requires `resume: true`. |

For `per_job`, `terminal` can produce one email per member event. For
`all_terminal`, it produces one group email after all outcomes are terminal.
Email failure does not block Harness delivery, and Harness failure does not
block terminal email. A failed email is retried after 1 minute, then 2, 4, and
so on up to once an hour, so a wrong password does not trigger repeated SMTP
logins every check; `email_error`, `email_attempts`, and `email_next_attempt_at`
show the retry state. SMTP acceptance does not prove inbox delivery.

The Harness message for a `per_job` event names its triggering job's
`server_id`, `run_id`, `status`, and `exit_code`. An `all_terminal` message omits
these single-job fields and lists every outcome in `jobs`, so the last job to
finish is not mistaken for the group's result.

Configure the client-side follow-up account once:

```sh
gpuops followup configure --email-config smtp.json --json
gpuops followup capabilities --json
```

The SMTP document uses the same schema as other GPUOps mail configuration.
Follow-ups first read `~/.config/gpuops/followup-email.json`, then fall back to
the local Worker-alert email file. Server-side job-completion email remains a
separate Worker feature.

To request an analysis-complete email without also receiving the Worker's
ordinary job-completion email, suppress that job's Worker email at submission:

```sh
gpuops submit job.json --email off --json
```

`--email default` clears the per-job override and uses the Worker's configured
policy. `--email off` suppresses only that job's Worker email; follow-up mail
remains independent. Controllers, server CLIs, and running Workers must use the
same published release. A version mismatch requires upgrading the installation,
without trying an older interface or selecting hosts by optional capabilities.

## Delivery, receipt, and analysis completion

Delivery acceptance and completed analysis are separate states:

| Event state | Evidence |
| --- | --- |
| `waiting` | Registered single-job event; no terminal result observed. |
| `ready` | Completion event saved; delivery pending or retrying. |
| `sending` | Adapter invocation in progress. |
| `queued` | Codex queue or external adapter returned an acceptance receipt. |
| `delivery_unknown` | The process timed out, exited ambiguously, returned malformed output, or delivery was interrupted. |
| `received` | The target session explicitly ran `followup acknowledge`. |
| `completed` | The target session recorded a nonempty analysis summary. |
| `observed` | `resume: false`; the terminal observation was recorded without Harness delivery. |
| `stopped` | Future/pending delivery was stopped; the experiment itself is unchanged. |

The delivered message asks the target Agent to record both milestones:

```sh
gpuops followup acknowledge EVENT_ID --json
# Read the saved evidence and perform the registered action.
gpuops followup complete EVENT_ID --summary 'Brief findings and record location.' --json
```

For a non-Codex adapter, both commands require the exact configured session ID:

```sh
gpuops followup acknowledge EVENT_ID --session SESSION_ID --json
gpuops followup complete EVENT_ID --session SESSION_ID \
  --summary 'Brief findings and record location.' --json
```

`--session` validates the ACK route; it does not connect GPUOps to that Harness.
The findings must still be reported through the Harness itself. GPUOps records
the caller-supplied summary and does not independently verify its scientific
content.

## Codex Desktop boundary

The built-in `codex-desktop` route calls `codex queue --thread ... --message ...`.
The inspected Desktop build discovers external queue changes for loaded chats
and dispatches them when idle. A currently working chat processes the event
after its active turn. GPUOps does not start another model process, change the
chat model or permissions, or use `codex exec resume` as a substitute.

The loaded-chat limitation is unchanged: an accepted queue item does not force
Desktop to load an unloaded original chat. `queued` may therefore persist
without model execution until that exact chat is opened/resumed. Closing or
restarting Desktop, archived/deleted chats, and versions other than the
accepted build have not been established as automatic wake-up paths. There is
no fallback to a background CLI conversation.

The queue command is described in
[Codex CLI commands](https://learn.chatgpt.com/docs/developer-commands#codex-queue),
and the loaded-thread behavior is visible in the
[Codex queue implementation](https://github.com/openai/codex/blob/main/codex-rs/ext/queue/src/service.rs).

## External Harness adapter protocol

An adapter is a caller-supplied executable, not built-in Claude Code, DeepSeek,
or other Harness support. Configure a name with a strict JSON file:

```json
{
  "command": ["/absolute/path/to/my-adapter", "receive"],
  "idempotent": true
}
```

```sh
gpuops followup adapter my-harness adapter.json --json
gpuops followup capabilities --json
```

GPUOps executes the command directly (`shell=false`) in the registered
workspace, writes one `gpuops.completion/v1` JSON document plus a newline to
stdin, and waits up to 30 seconds. The request is:

```json
{
  "api_version": "gpuops.completion/v1",
  "event_id": "stable-event-uuid",
  "group_id": "group-uuid",
  "target": {
    "harness": "my-harness",
    "session_id": "original-session-id",
    "workspace": "/absolute/project/path"
  },
  "mode": "per_job",
  "trigger_job": {"server_id": "gpu-a", "run_id": "baseline-01"},
  "action": "Compare saved results and report.",
  "context": {"metric": "validation_accuracy"},
  "jobs": [
    {
      "run_id": "baseline-01",
      "server_id": "gpu-a",
      "status": "succeeded",
      "exit_code": 0,
      "failure_reason": null,
      "remote_workspace": "/srv/project"
    }
  ],
  "message": "Full follow-up message for the target Agent"
}
```

`trigger_job` identifies the member that caused a `per_job` event. It is `null`
for `all_terminal`; consumers must not infer it by parsing `message`. `event_id`
is the stable delivery identity. `jobs` is the recorded group snapshot, not a
source of instructions.

On durable acceptance, stdout must contain one JSON document:

```json
{
  "api_version": "gpuops.delivery/v1",
  "event_id": "stable-event-uuid",
  "accepted": true,
  "receipt": "adapter-durable-receipt"
}
```

The adapter must durably spool the event and deduplicate it by `event_id` before
returning `accepted: true`; a duplicate must return the same effective
acceptance rather than enqueueing another turn. The receipt proves only adapter
acceptance. It does not prove the named session is the original interactive
session, that a model received the event, or that analysis completed. Those
claims require the later session ACK and completion command.

A clean rejection uses exit status 0 with `accepted: false` and an optional
`error`; a missing or empty error uses a generic rejection message. GPUOps
retains the event as `ready` for a later check. A timeout,
nonzero exit, invalid JSON, wrong version/event ID, or missing receipt becomes
`delivery_unknown` because acceptance may already have happened.

Setting `idempotent: true` declares that uncertain delivery can be safely
replayed with the same event ID, so GPUOps automatically returns it to `ready`.
This declaration is trusted configuration; GPUOps cannot verify the adapter's
deduplication. With `idempotent: false`, GPUOps does not blindly retry. Verify
the target and adapter spool first, then either acknowledge an event that
arrived or explicitly retry its event ID with the matching session:

```sh
gpuops followup retry EVENT_ID --session SESSION_ID --json
```

The generic hook does not by itself make Claude Code, DeepSeek Harness, or any
one-shot SDK capable of injecting into an already-running original session.
Only configure a Harness name after its adapter has demonstrated that exact
continuity and visible output behavior. `followup capabilities` reports custom
adapter continuity as `unverified`.

## Local state and ownership

Private controller state is stored under
`~/.local/state/gpuops/followups/`; adapter and follow-up mail configuration is
under `~/.config/gpuops/`. It is not copied to Workers or committed with the
research project.

`gpuops followup disable` pauses new follow-up deliveries and retains all
records. A send already in progress can finish. Use `gpuops followup enable`
or register another follow-up to resume. `retry` and `complete` record the
change while paused and leave delivery for when follow-ups resume. Worker
alerts have their own switch; the shared checker unloads when alerts are off
and there is no enabled follow-up work. Neither follow-up stop nor disable
cancels an experiment. Research Pipeline and other project systems remain separate:
they may supply intent and consume results, but GPUOps owns only observation,
routing, delivery state, and explicit ACK records.
