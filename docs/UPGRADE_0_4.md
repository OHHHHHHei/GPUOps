# Upgrade a GPUOps fleet to 0.4

GPUOps 0.4 changes the saved job format and the Mac background service. Upgrade
the Mac and all registered servers in one maintenance window. Keep the source
checkout containing `scripts/migrate_to_0_4.py` until the upgrade is accepted;
the script is separate from the installed wheel.

## 1. Prepare and pause writers

- Build or obtain the 0.4 wheel, including all the cutover changes. On the Mac,
  install it in a separate environment, leaving the active installation alone.
  The examples below use `$candidate` for the path to that environment's
  `gpuops` executable. Confirm its version with `"$candidate" version --json`.
- Run `gpuops deploy status --json` from the current Mac installation. Record
  each server ID, SSH destination, actual `current_release`, and whether its
  Worker is running. Save the old
  Mac wheel and the actual artifacts for the installed server releases, or keep
  a copy of each installed release directory outside the GPUOps storage budget.
  A release may be an unreleased build carrying the same version string as a
  published wheel; keep the exact build. Keep the server registry fixed during
  the window.
- Pause submissions, cancels, reconciles, follow-up writes and manual checks
  on the Mac and server-local clients. Let existing jobs finish. Drain the
  production queue before starting 0.4 if you need rollback to remain safe:
  restoring a pre-upgrade queued record after 0.4 has run it can cause it to
  run again. Keep production writers paused through acceptance.
- Record which Mac LaunchAgents are loaded using
  `launchctl print "gui/$(id -u)/io.gpuops.worker-alerts"` and
  `launchctl print "gui/$(id -u)/io.gpuops.followups"`. Boot out the loaded
  agents with `launchctl bootout "gui/$(id -u)/LABEL"`, substituting the actual
  label. If `io.gpuops.background` is loaded from a prior attempt, boot it out
  too. The old `alerts disable` command deletes the alert state; use
  `launchctl bootout` for this window.
- Save the two original plists from `~/Library/LaunchAgents/` and
  `~/.local/state/gpuops/alerts.json` in a private directory, if present.
  Record which files were absent. If `~/.config/gpuops/background.json`
  exists from a previous setup, save it as well. Keep the old plists out of
  `~/Library/LaunchAgents/` until acceptance or restoration so they cannot
  reload at login. Existing SMTP settings remain in place.
- With the old Mac CLI, run `gpuops fleet stop --json` and wait for all Workers
  to stop. If any Worker cannot be stopped, end the window without converting
  records and restart only the services you stopped.

## 2. Check, back up and convert

From the Mac checkout, as the same user who runs GPUOps:

```sh
python3 scripts/migrate_to_0_4.py --check
python3 scripts/migrate_to_0_4.py
```

The check reads the Mac and each registered SSH destination without changing
files. A loaded LaunchAgent, a Worker holding its runtime lock, or a queue
record in `leased`, `starting`, `running` or `cancel_requested` blocks the
cutover. The second command repeats the check, completes the first JSON backup
on every host, and then converts. `--backup-only` can run the first two phases
without converting. Each host prints the number of records in each phase.

The backup lives in `~/gpuops-0.4-backup/records/` on each machine. It contains
a manifest and the original bytes of the managed JSON files. It does not
contain executables, plists, alerts, logs or results. Keep the first backup
throughout this maintenance window. If conversion is interrupted, leave
services stopped, fix the problem and run the same command again; it reuses
the first backup. If you abandon the window and resume production writes,
archive that window's backup before beginning a fresh upgrade.

The conversion removes `tracking` from job specs, adds explicit null result
fields where they were absent, and fills in outcomes for single-job follow-ups.
Existing observed results, submission IDs and delivery receipts are preserved.
Remove `tracking` from any saved input manifests you plan to submit again;
the script only converts GPUOps state files.

## 3. Install and verify the candidate while stopped

From the candidate Mac environment, install the matching candidate remotely:

```sh
"$candidate" deploy install --no-start --json
```

This selects the new release on each successfully installed server but leaves
Workers stopped. Check every result. Before starting services:

- On each server run `~/.local/bin/gpuops version --json` and
  `~/.local/bin/gpuops reconcile --list --json`. The reconcile command parses
  the whole local journal and only lists pending entries.
- On each server, use the deployed environment's Python to parse every queue
  record and rebuild the derived index while the Worker is stopped:

  ```sh
  ~/.local/share/gpuops/current/venv/bin/python - <<'PY'
  from pathlib import Path
  from gpuops.core.queue import QueueStore

  queue = QueueStore(Path.home() / ".local/state/gpuops/queue")
  print("Parsed job records:", len(queue.list()))
  print("Indexed job records:", queue.count())
  PY
  ```

- On the Mac, run `"$candidate" reconcile --list --json` and
  `"$candidate" followup list --json`. The latter checks both events and groups;
  confirm `invalid_records` is empty. Fix any read failure before starting
  Workers or enabling the Mac background service.

## 4. Start and accept

Switch the Mac's usual `gpuops` command to the verified candidate. Run
`gpuops fleet start --json` and verify all Workers are ready and running the
new release. Check that history pagination and an older job's details work.

Run `gpuops alerts enable` and/or `gpuops followup enable` according to the old
settings you recorded. Use a disposable test job to check follow-up delivery,
original-chat acknowledgment and completion, and email if configured.
`gpuops background status --json` shows background queries; verify that a
round needs at most one query per server. If Worker alerts were enabled, stop
and start one idle test Worker and verify both notifications. Keep production
writers paused until these checks pass.

After acceptance, resume production writes. Remove the temporary migration
backups, the saved old plists, and each retained old release identified during
preparation. Remove the migration script from the source checkout when the
cutover is complete.

## Restore before acceptance

If a cutover fails, keep writers paused. Boot out
`io.gpuops.background` and move its plist out of `~/Library/LaunchAgents/`.
Let disposable test jobs finish, or cancel them and wait for them to become
terminal. Then use `"$candidate" fleet stop --json` to stop all Workers and
wait for them to exit. The restore command
checks that the same services are stopped and that all hosts have a backup
before restoring:

```sh
python3 scripts/migrate_to_0_4.py --restore
```

Use this after the backup phase has succeeded on every host. If a check or
backup failed before conversion began, the original records are already in
place. Restore replaces managed records with the first backup, removes
managed JSON files created since then, and deletes the derived queue index
so the restored release can rebuild it. Logs and results remain on disk.
The script does not change the installed release or start any service.
It does not undo emails already sent or restore queue event logs and notification
cursors changed by acceptance jobs. Those traces are separate from the saved
job records and remain available after restoration.

Compare `deploy status --json` against the original release IDs recorded
earlier. For each host whose `current_release` changed, run
`"$candidate" deploy rollback --server SERVER_ID --json` only if its
`previous_release` is the original release ID and its Worker was originally
running: rollback starts the restored Worker. For an originally stopped Worker,
repoint `current` to the saved original release while keeping it stopped, using
the manual restoration path below. If that release is no longer
available as the rollback target, restore the saved release directory to its
original path under `~/.local/share/gpuops/releases/` and repoint
`~/.local/share/gpuops/current` to it while the Worker is stopped. If you
saved the exact artifact instead, use it to reinstall that build before
starting the host.

Use the original Mac installation or reinstall the saved original Mac
wheel. Restart any previously running Workers still stopped. Put the original
alert state, background settings and plists back from step 1; remove state or
settings created during the cutover if those files were originally absent.
Bootstrap only the previously loaded agents using
`launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/LABEL.plist"`.
Resume production writers after verifying the restored fleet.

## Mac data copy validation

On 2026-11-14, 468 submission journal entries, 6 follow-up events, and 2
follow-up groups were copied from an existing Mac installation. All 476
records converted and passed validation with the 0.4 data models. A second
conversion made no changes. Restore reproduced all 476 original files byte
for byte. The exercise used only temporary copies; no live state was locked
or changed, and no services or remote servers were contacted.
