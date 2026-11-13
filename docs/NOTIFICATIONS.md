# Job completion email

GPUOps can email you when a managed job succeeds, fails, or is cancelled. Email
is off by default and configured once per server. You do not need to change your
job JSON. The Worker sends email from the GPU server, so it works while your Mac
or other control machine is offline.

Use a GPUOps build that includes `gpuops notify` on the controller and each
server. To upgrade an older registered server from that controller, run
`gpuops deploy install --server gpu-server --json`. The server must be able to
reach the configured SMTP host and port.

Save your SMTP settings in a private file such as `smtp.json` (permissions
`0600`), substituting the values provided by your SMTP service:

```json
{
  "host": "smtp.example.org",
  "port": 465,
  "security": "ssl",
  "username": "gpuops@example.org",
  "password": "replace-with-your-app-password",
  "sender": "gpuops@example.org",
  "recipient": "researcher@example.org"
}
```

`host`, `sender`, and `recipient` are required. `security` defaults to `ssl`
(TLS from the start), and `port` defaults to `465`. Set `security` to `starttls`
and specify the port if your provider requires STARTTLS. Omit both `username`
and `password` if your SMTP relay does not require authentication.

```bash
gpuops notify configure smtp.json --server gpu-server --json
gpuops notify status --server gpu-server --json
gpuops notify test --server gpu-server --json
```

Use `configure -` to read the same JSON from standard input, for example from a
secret manager. SMTP credentials are read from the file or pipe, sent to remote
servers over SSH standard input, and stored in the Worker account's private
`~/.config/gpuops/notifications.json`. They are not included in command output.

Repeat `--server ID` to select several servers. With no `--server`, the control
machine targets all registered servers; on an installed server the command
defaults to that server and runs locally. `--json` reports an overall `outcome`
(`success`, `partial`, or `failed`) and the result for each server. A partial or
failed operation exits nonzero.

The Worker checks for newly finished jobs about every 30 seconds. Enabling email
starts from that point and does not send mail for jobs that finished before it.
If the Worker stops, it resumes checking when restarted and sends eligible mail
once it confirms the jobs' final states. If SMTP delivery fails, the Worker keeps
the pending job and retries later. `notify status` shows whether email is enabled,
when the Worker last sent a message, and any pending retry or recent error.

`notify test` sends one test message per selected server using its saved SMTP
settings. SMTP acceptance confirms that the server accepted the message for
delivery; check your inbox or spam folder to verify it arrived. If the SMTP
server accepts a job notification but its confirmation is lost, a retry can
send a duplicate. The same can happen after a restart if SMTP accepted a message
but GPUOps could not save its delivery cursor. While the Worker stays running,
it retries saving that cursor before sending further mail.

Run `gpuops notify disable --server gpu-server --json` to stop automatic email.
The Worker picks up this change on its next check, normally within about 30
seconds. A message already being sent may finish. The settings stay saved for
testing. Run `configure` again to re-enable email; only jobs that finish after
re-enabling are eligible. Running `configure` while email is already enabled
updates SMTP settings without resetting the delivery position.

These commands manage the standard installed Worker and its default
`~/.local/state/gpuops` directory. A manually launched development Worker with
`worker run --state-dir` writes delivery state in that custom directory;
`notify status` does not read that custom state.

## Choosing email timing for a task group

Use `gpuops submit job.json --email off --json` to suppress the server's
completion email for that job. The same policy can be saved in the manifest as
`"metadata": {"gpuops.email": "off"}`. Omit it, or use `default`, to retain the
server's existing notification policy. Other jobs and Worker alerts are unaffected.
Older Workers that cannot honor this policy reject `off` before submission;
upgrade the server's GPUOps installation first.

[Session follow-ups](FOLLOWUPS.md) provide separate client-side group emails at
`terminal` or `analysis_complete`. For “analyze the whole group, then email me,”
submit each member with `--email off` and register the group with
`notify: "analysis_complete"`. Registration does not change a submitted job's
email policy or recall an already sent email. These group emails require the
Mac follow-up checker, while ordinary job emails continue to run on the server.

## Worker alert email on macOS

Worker stop/recovery alerts are sent by the Mac's existing alert poller, because
a stopped Worker cannot send its own email. They are separate from the
server-side job completion emails above. Reuse the same SMTP settings file:

```bash
gpuops alerts enable --email-config smtp.json
gpuops alerts test --json
gpuops alerts status --json
```

The settings are copied to `~/.config/gpuops/worker-alerts-email.json` with
permissions `0600`; credentials are omitted from command output. This local
configuration does not change the remote Workers or their job email settings.
`alerts enable` without options retains any saved email settings. Email is off
until configured. Use `alerts enable --no-email` to remove the local email
settings and keep desktop notifications, or `alerts disable` to stop both
channels while retaining the email settings for the next `enable`.

Once a minute, the poller queries all registered Workers over SSH. A confirmed
`stopped` or `stale` Worker produces a stop alert; recovery to `ready` produces
a recovery alert. Failed queries (`unknown`) and other non-ready states do not
produce stop alerts. A stopped Worker does not mean its already-started
experiments have stopped. No Worker is automatically restarted by these alerts.

Desktop and email delivery each keep their own per-server state. Successful
delivery suppresses repeats of the same stop/recovery event; a failed channel
retries on subsequent checks. If a stop email has not succeeded before recovery,
neither an expired stop email nor an unmatched recovery email is sent. Recovery
email is retried only while the Worker remains ready. Delivery errors appear
in `alerts status`; malformed email settings do not prevent desktop alerts.
SMTP acceptance and saving local delivery state are separate operations, so
lost SMTP confirmation or a failed state write can lead to duplicate email.

`alerts test` attempts both configured channels independently and exits nonzero
if either fails. `notification_submitted` means macOS accepted the desktop
notification; `email_accepted` means SMTP accepted the email, not that it reached
your inbox. The test never stops a Worker or experiment.

The Mac must be logged in, awake, and able to reach the SSH servers and SMTP
service. Checks continue with the Web console closed and resume after login.
`alerts status` reads the latest saved check; it is not a fresh remote query.
