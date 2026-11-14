"""Shared macOS scheduling for Worker alerts and session follow-ups."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import plistlib
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Protocol

from pydantic import StrictBool

from gpuops import alerts
from gpuops.cli_errors import CommandError, error_payload, print_error
from gpuops.config import select_targets
from gpuops.control import RemoteTarget
from gpuops.file_io import atomic_write_bytes
from gpuops.followup_delivery import AdapterSettings, DeliveryRouter, load_adapters
from gpuops.followup_groups import (
    GroupStore,
    apply_group_observations,
    pending_group_events,
    pending_group_jobs,
)
from gpuops.followups import (
    CodexQueue,
    FollowupStore,
    JobObservations,
    apply_followup_observations,
    deliver_followups,
    pending_delivery,
    pending_email,
    pending_followup_jobs,
    recover_interrupted,
)
from gpuops.models import JobRecord, StrictModel, utc_now
from gpuops.transport import SshWorkerTransport
from gpuops.worker.rpc import MAX_WATCH_JOBS

LABEL = "io.gpuops.background"
INTERVAL_SECONDS = 60
PAUSED_MESSAGE = "Follow-ups are paused; run gpuops followup enable to process them."


class Switches(StrictModel):
    alerts: StrictBool = False
    followups: StrictBool = False


class WatchTransport(Protocol):
    async def watch(self, target: RemoteTarget, run_ids: list[str]) -> dict[str, Any]: ...


def _launchctl(*args: str, check: bool = True) -> bool:
    result = subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True, timeout=15)
    if result.returncode and check:
        raise RuntimeError(result.stderr.strip() or f"launchctl {args[0]} failed")
    return result.returncode == 0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode())


def _retryable_harnesses(adapters: dict[str, AdapterSettings]) -> set[str]:
    return {name for name, settings in adapters.items() if settings.idempotent}


@dataclass
class Round:
    started: float = field(default_factory=monotonic)
    targets: int = 0
    jobs: int = 0
    ssh_calls: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    jobs_observed: JobObservations = field(default_factory=dict)
    workers: dict[RemoteTarget, dict[str, Any]] = field(default_factory=dict)
    worker_errors: dict[RemoteTarget, str] = field(default_factory=dict)

    def error(
        self,
        scope: str,
        error: Exception,
        *,
        target: RemoteTarget | None = None,
        run_id: str | None = None,
    ) -> None:
        row = {"scope": scope, **error_payload(error)}
        row.pop("ok", None)
        if target is not None:
            row.update(server_id=target.server_id, ssh_host=target.ssh_host)
        if run_id is not None:
            row["run_id"] = run_id
        self.errors.append(row)

    async def collect(
        self, transport: WatchTransport, requests: dict[RemoteTarget, set[str]]
    ) -> None:
        self.targets = len(requests)
        self.jobs = sum(len(ids) for ids in requests.values())
        await asyncio.gather(
            *(
                self._watch_target(transport, target, sorted(ids))
                for target, ids in requests.items()
            )
        )

    async def _watch_target(
        self, transport: WatchTransport, target: RemoteTarget, run_ids: list[str]
    ) -> None:
        # Empty batches still read Worker status for alerts.
        for offset in range(0, len(run_ids) or 1, MAX_WATCH_JOBS):
            batch = run_ids[offset : offset + MAX_WATCH_JOBS]
            self.ssh_calls += 1
            try:
                payload = await transport.watch(target, batch)
                jobs = payload["jobs"]
                worker = payload["worker"]
                self.workers.setdefault(target, worker)
            except Exception as error:
                self.error("watch", error, target=target)
                self.worker_errors[target] = str(error)
                self.jobs_observed.update(((target, run_id), str(error)) for run_id in batch)
                continue
            if worker.get("error_code") == "version_mismatch":
                mismatch = CommandError("version_mismatch", worker["reason"])
                self.error("worker", mismatch, target=target)
                self.worker_errors[target] = str(mismatch)
            for run_id in batch:
                try:
                    value = jobs[run_id]
                    if value is None:
                        raise ValueError("registered job was not found on its original server")
                    if isinstance(value, dict) and "error" in value:
                        detail = value["error"]
                        raise CommandError(detail["code"], detail["message"])
                    job = JobRecord.model_validate(value)
                    if job.run_id != run_id:
                        raise ValueError("observed job does not match the requested run_id")
                    self.jobs_observed[target, run_id] = job
                except (KeyError, TypeError, ValueError) as error:
                    self.error("job", error, target=target, run_id=run_id)
                    self.jobs_observed[target, run_id] = str(error)


class Background:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home()
        self.root = self.home / ".local/state/gpuops"
        self.config = self.home / ".config/gpuops/background.json"
        self.followup_root = self.root / "followups"
        self.alert_state = self.root / "alerts.json"
        self.stats = self.root / "background-last.json"
        self.plist = self.home / "Library/LaunchAgents" / f"{LABEL}.plist"

    def event_store(self) -> FollowupStore:
        return FollowupStore(self.followup_root / "records")

    def group_store(self) -> GroupStore:
        return GroupStore(self.followup_root / "groups")

    @property
    def domain(self) -> str:
        return f"gui/{os.getuid()}"

    @property
    def service(self) -> str:
        return f"{self.domain}/{LABEL}"

    def switches(self) -> Switches:
        try:
            return Switches.model_validate_json(self.config.read_bytes())
        except FileNotFoundError:
            return Switches()

    def set_switch_locked(self, feature: Literal["alerts", "followups"], enabled: bool) -> None:
        values = self.switches().model_dump()
        values[feature] = enabled
        _write_json(self.config, values)

    def alerts_enabled(self) -> bool:
        return self.switches().alerts

    def followups_enabled(self) -> bool:
        return self.switches().followups

    def enable_followups_locked(self, events: FollowupStore) -> None:
        self.set_switch_locked("followups", True)
        self.sync_locked(events)

    @contextmanager
    def round_lock(self) -> Iterator[bool]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / "background-round.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def is_loaded(self) -> bool:
        return sys.platform == "darwin" and _launchctl("print", self.service, check=False)

    def sync_locked(self, events: FollowupStore) -> None:
        """Update launchd after a command, with the same lock used by round finalization."""
        if sys.platform != "darwin":
            return
        switches = self.switches()
        if switches.alerts or (
            switches.followups and self.pending(events, self.group_store())["total"] > 0
        ):
            if not self.is_loaded():
                config = {
                    "Label": LABEL,
                    "ProgramArguments": [sys.executable, "-m", "gpuops", "background", "_run"],
                    "EnvironmentVariables": {
                        "PATH": (
                            f"{self.home / '.local/bin'}:/opt/homebrew/bin:/usr/local/bin:"
                            "/usr/bin:/bin:/usr/sbin:/sbin"
                        )
                    },
                    "RunAtLoad": True,
                    "StartInterval": INTERVAL_SECONDS,
                    "LimitLoadToSessionType": "Aqua",
                }
                self.plist.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(self.plist, plistlib.dumps(config))
                _launchctl("bootstrap", self.domain, str(self.plist))
        else:
            # A command never unloads a checker that is still delivering or writing results.
            with self.round_lock() as idle:
                if idle:
                    self._unload()

    def _unload(self) -> None:
        # Unlink before bootout: launchd can terminate the current checker.
        self.plist.unlink(missing_ok=True)
        if self.is_loaded():
            _launchctl("bootout", self.service)

    @staticmethod
    def pending(events: FollowupStore, groups: GroupStore) -> dict[str, int]:
        records = events.list()
        retryable = (
            _retryable_harnesses(load_adapters())
            if any(
                row.status == "delivery_unknown" and row.harness != "codex-desktop"
                for row in records
            )
            else set()
        )
        group_records = groups.list()
        waiting = len({*pending_followup_jobs(records), *pending_group_jobs(group_records, events)})
        deliveries = sum(pending_delivery(row, retryable) for row in records)
        deliveries += pending_group_events(group_records, events)
        emails = sum(pending_email(row) for row in records)
        return {
            "waiting_jobs": waiting,
            "events": deliveries,
            "emails": emails,
            "total": waiting + deliveries + emails,
        }

    def status(self) -> dict[str, Any]:
        try:
            last_round = json.loads(self.stats.read_text())
        except FileNotFoundError:
            last_round = None
        return {
            "ok": True,
            "enabled": self.switches().model_dump(),
            "loaded": self.is_loaded(),
            "last_round": last_round,
        }

    async def run_round(self, *, followup_only: bool = False) -> dict[str, Any]:
        with self.round_lock() as acquired:
            if not acquired:
                return {"checked": False, "reason": "round_in_progress"}
            return await self._run_round(followup_only=followup_only)

    async def _run_round(self, *, followup_only: bool) -> dict[str, Any]:
        run = Round()
        requests: dict[RemoteTarget, set[str]] = {}
        labels: dict[RemoteTarget, str] = {}
        events = self.event_store()
        groups = self.group_store()
        with events.lock():
            switches = self.switches()
            if switches.alerts and not followup_only:
                try:
                    labels = {
                        RemoteTarget(target.id, target.ssh_host): target.label
                        for target in select_targets()
                    }
                    for target in labels:
                        requests[target] = set()
                except Exception as error:
                    run.error("alerts", error)
            if switches.followups:
                try:
                    records, invalid = events.scan()
                    group_records, invalid_groups = groups.scan()
                    for row in [*invalid, *invalid_groups]:
                        run.errors.append({"scope": "followups", **row})
                    recover_interrupted(events, records)
                    for target, run_id in [
                        *pending_followup_jobs(records),
                        *pending_group_jobs(group_records, events),
                    ]:
                        requests.setdefault(target, set()).add(run_id)
                except Exception as error:
                    run.error("followups", error)

        if requests:
            await run.collect(SshWorkerTransport(timeout_seconds=10), requests)

        if switches.followups:
            try:
                with events.lock():
                    if self.followups_enabled():
                        apply_group_observations(groups, events, run.jobs_observed)
                        apply_followup_observations(events, run.jobs_observed)
            except Exception as error:
                run.error("followups", error)
        if labels:
            try:
                alerts.process_observations(
                    self.alert_state,
                    events.lock,
                    self.alerts_enabled,
                    [
                        alerts.observe(
                            target.server_id,
                            label,
                            run.workers.get(target),
                            run.worker_errors.get(target),
                        )
                        for target, label in labels.items()
                    ],
                )
            except Exception as error:
                run.error("alerts", error)

        if switches.followups:
            try:
                adapters = load_adapters()
            except (OSError, ValueError) as error:
                run.error("followups", error)
                adapters = {}
            try:
                try:
                    settings = json.loads((self.followup_root / "codex.json").read_bytes())
                    queue = CodexQueue(**settings)
                except FileNotFoundError:
                    queue = None
                except (OSError, ValueError, TypeError) as error:
                    run.error("followups", error)
                    queue = None
                deliver_followups(
                    events,
                    DeliveryRouter(queue, adapters),
                    self.followups_enabled,
                    _retryable_harnesses(adapters),
                )
            except Exception as error:
                run.error("followups", error)

        with events.lock():
            pending = None
            try:
                pending = self.pending(events, groups)
            except Exception as error:
                run.error("followups", error)
            latest = self.switches()
            stats = {
                "checked_at": utc_now().isoformat(),
                "elapsed_seconds": round(monotonic() - run.started, 3),
                "followup_only": followup_only,
                "ssh_calls": run.ssh_calls,
                "targets": run.targets,
                "jobs": run.jobs,
                "pending": pending,
                "errors": run.errors,
            }
            _write_json(self.stats, stats)
            if not latest.alerts and (not latest.followups or (pending and not pending["total"])):
                # A launchd-started round may be terminated by bootout, so write results first.
                self._unload()
        return {
            "checked": not (followup_only and not switches.followups),
            **(
                {"reason": "followups_paused"} if (followup_only and not switches.followups) else {}
            ),
            **stats,
        }


def run_command(args: argparse.Namespace) -> int:
    background = Background()
    try:
        if args.background_command == "status":
            payload = background.status()
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"Background checker: {'loaded' if payload['loaded'] else 'unloaded'}.")
                for name, enabled in payload["enabled"].items():
                    print(f"{name}: {'enabled' if enabled else 'paused'}")
                if payload["last_round"] is not None:
                    print(json.dumps(payload["last_round"], ensure_ascii=False, indent=2))
        else:
            asyncio.run(background.run_round())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    return 0
