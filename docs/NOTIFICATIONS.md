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
