# Original-chat follow-ups (experimental)

This first stage follows **one job in one local Codex Desktop chat**. It uses
Codex's persistent message queue, not `codex exec resume`, a second model
session, or scheduled model polling. The single-job loop passed live acceptance
on 2026-11-13: the original Desktop chat automatically started a new turn, read
saved metrics over SSH, and reported its analysis. A queue receipt alone still
does not prove that a later follow-up has executed.

The acceptance used Codex CLI 0.167.1 and Desktop's bundled CLI
0.167.0-alpha.4.1. A small scalar-optimization job completed under the existing
GPUOps Worker. The ordinary macOS checker detected its terminal record and
queued one message; after the active turn finished, the same chat continued
without a separate CLI model session. It read the saved 25-step result
(`initial_loss: 4.0`, `final_loss: 0.00005708990770823891`) without rerunning
training. This tests completion and continuation, not GPU-training performance.

## Register from the original chat

After `gpuops submit` accepts a job, the Agent can register the requested action:

```sh
gpuops followup add RUN_ID --action 'Read metrics.json, compare the saved baseline, update the research record and report the findings here.' --json
```

GPUOps reads the exact chat UUID from `CODEX_THREAD_ID`, the project workspace
from the current directory, and the server/SSH destination from its local
submission journal. `--workspace DIR` overrides the project directory;
`--thread UUID` explicitly identifies a chat; `--server ID` is needed only when
this client did not submit the job. A saved submission's server cannot be
overridden. Repeating the same registration returns the existing follow-up.

The first registration enables a macOS LaunchAgent that checks every 30 seconds.
It requires an installed Codex CLI whose `codex queue --help` shows the queue
command (older CLIs are rejected at registration), the same Codex home used
by Desktop, and SSH access to the experiment server. No Worker upgrade or new
job-manifest fields are needed. Register only when the user requested follow-up.

The poller reads the registered job directly. `succeeded`, `failed`, and
`cancelled` all trigger a message. A missing job or unavailable server is not
treated as completion. Once it observes a terminal result, GPUOps saves the
message locally before handing it to Codex. Ordinary SSH/CLI failures retry on
later checks. It never reads experiment logs as instructions or starts a model
while waiting.

## Delivery and analysis are different states

```sh
gpuops followup list --json
gpuops followup show FOLLOWUP_ID --json
```

| State | Evidence |
| --- | --- |
| `waiting` | Registered; no terminal job result observed yet. |
| `ready` | Completion message saved; delivery pending or retrying. |
| `sending` | Delivery command in progress. |
| `queued` | Codex accepted the message into its persistent queue. |
| `received` | The original chat ran `followup acknowledge`. |
| `completed` | The original chat recorded its analysis summary. |
| `delivery_unknown` | Delivery was interrupted or timed out; acceptance is uncertain. |
| `stopped` | Follow-up stopped before delivery; the experiment is unchanged. |

The completion message asks the original Agent to run:

```sh
gpuops followup acknowledge FOLLOWUP_ID --json
# Read saved metrics/artifacts and perform the registered action.
gpuops followup complete FOLLOWUP_ID --summary 'Brief findings and research-record location.' --json
```

These commands require the original `CODEX_THREAD_ID`. They do not claim that
GPUOps independently verified the model's analysis. The findings must also be
reported in the same chat. A completed record tells the Agent not to analyze
the same notification again.

Accepted messages are not sent again. The CLI does not expose a stable delivery
idempotency key, so GPUOps does not blindly resend after an ambiguous timeout or
crash. Inspect the original chat and its queued messages first. Only if the
message was not accepted, run `gpuops followup retry FOLLOWUP_ID --json` from the
original chat. If it arrived, acknowledge it instead. This is not an exactly-once
delivery guarantee.

`gpuops followup stop FOLLOWUP_ID --json` stops a waiting/pending follow-up. It
cannot recall a message already handed to Codex. Running `add` again for the
same job and chat re-registers a stopped follow-up, optionally with a new
action. `gpuops followup disable`
unloads the local checker and retains records; a later `add` enables it again.
Neither command cancels experiments.

## Desktop boundary

The inspected Codex Desktop build includes a queue watcher that discovers
external queue changes for **loaded chats**, then dispatches them when idle.
A currently working chat processes its queued follow-up after the current turn.
GPUOps does not start a separate daemon, take over the thread, or change the
chat's model, permissions, or working directory. The registered workspace is
included in the message for subsequent commands.

When the Mac is asleep/offline, checks resume after it wakes and reconnects.
Codex can retain an accepted message while Desktop is closed. **An unloaded
chat is not automatically loaded by this integration**: opening/resuming that
original chat may be necessary. `queued` can therefore persist without model
execution. Closing/restarting Desktop, archived/deleted chats, and other Codex
versions are not covered by the initial live acceptance. There is no silent
fallback to a background CLI conversation. Desktop app tools are not used as an
external callback API.

The queue command and its `--resume` limitations are documented in
[Codex CLI commands](https://learn.chatgpt.com/docs/developer-commands#codex-queue).
The loaded-thread behavior is described by the
[Codex queue implementation](https://github.com/openai/codex/blob/main/codex-rs/ext/queue/src/service.rs).
This integration intentionally omits `--resume`: a separate local daemon could
run the thread outside the Desktop server whose behavior we are validating.

## Responsibility and next stages

GPUOps owns task observations, saved routing, delivery, and receipt records.
Research Pipeline/the originating Agent supplies the purpose, action, and
artifact locations, then performs the agreed analysis and documentation using
saved metrics. The queue message retains that original research context.

Existing server-side completion email and Worker alerts are independent and
unchanged. A follow-up does not disable them or send an additional email.
"Email only after analysis" is not implemented by this first stage.

The next stage can extend local registration to explicit cross-server groups
and per-job/all-terminal delivery. All-terminal
must include failure and cancellation. Other Harness adapters and independently
selectable post-analysis notifications follow later; none is claimed here.

Private local state lives under `~/.local/state/gpuops/followups/`; it is not
copied to Workers or committed with the research project.
