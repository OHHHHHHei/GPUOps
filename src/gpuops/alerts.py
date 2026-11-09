"""Mac-side desktop and email notifications for stopped Workers."""

from __future__ import annotations

import asyncio
import json
import os
import plistlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpuops.cli_errors import CommandError
from gpuops.config import select_targets
from gpuops.control import RemoteTarget
from gpuops.file_io import atomic_write_bytes
from gpuops.notifications.config import SMTPSettings
from gpuops.notifications.mail import (
    MailDeliveryError,
    send_mail,
    worker_alert_message,
    worker_alert_test_message,
)
from gpuops.queries import fleet_worker_action
from gpuops.transport import SshWorkerTransport

LABEL = "io.gpuops.worker-alerts"
INTERVAL_SECONDS = 60


def _email_path() -> Path:
    return Path.home() / ".config/gpuops/worker-alerts-email.json"


def _read_email(path: Path) -> SMTPSettings:
    try:
        return SMTPSettings.model_validate_json(path.read_bytes())
    except ValueError:
        raise ValueError("invalid Worker alert SMTP settings") from None


def _email_settings() -> tuple[SMTPSettings | None, str | None]:
    try:
        return _read_email(_email_path()), None
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, "cannot read Worker alert SMTP settings"
    except ValueError as error:
        return None, str(error)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, json.dumps(state, ensure_ascii=False).encode("utf-8"))


def _launchctl(*args: str, check: bool = True) -> bool:
    result = subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True, timeout=15)
    if result.returncode and check:
        raise RuntimeError(result.stderr.strip() or f"launchctl {args[0]} failed")
    return result.returncode == 0


def _notify(title: str, message: str) -> None:
    result = subprocess.run(
        ["/usr/bin/osascript", "-", title, message],
        input=(
            "on run argv\n"
            "display notification (item 2 of argv) with title (item 1 of argv)\n"
            "end run\n"
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "macOS notification failed")


def _local_time(value: str) -> str:
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
    except ValueError:
        return value


def _observe(
    row: dict[str, Any],
    label: str,
    previous: dict[str, Any],
    settings: SMTPSettings | None,
    detected_at: str,
) -> dict[str, Any]:
    result = row.get("result", {})
    heartbeat = result.get("heartbeat") or {}
    stopped = result.get("status") in {"stopped", "stale"}
    ready = result.get("ready") is True
    if not row["online"]:
        status = "unknown"
    elif stopped:
        status = "stopped"
    elif ready:
        status = "ready"
    else:
        status = "not_ready"

    observed = {
        "server_id": row["server_id"],
        "label": label,
        "status": status,
        "reason": row.get("error") or result.get("reason"),
        "last_heartbeat": heartbeat.get("heartbeat_at"),
        "last_worker_error": heartbeat.get("last_tick_error"),
        "stop_notified": previous.get("stop_notified", False),
        "email_stop_notified": previous.get("email_stop_notified", False),
    }
    try:
        if status == "stopped" and not observed["stop_notified"]:
            details = ["Worker stopped."]
            if observed["last_heartbeat"]:
                details.append(f"Last heartbeat: {_local_time(observed['last_heartbeat'])}")
            if observed["last_worker_error"]:
                details.append(f"Last Worker error: {observed['last_worker_error']}")
            _notify(f"GPUOps: {label}", " ".join(details)[:320])
            observed["stop_notified"] = True
        elif status == "ready" and observed["stop_notified"]:
            _notify(f"GPUOps: {label}", "Worker recovered and is ready to schedule.")
            observed["stop_notified"] = False
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        observed["notification_error"] = str(error)
    if settings is not None and (
        (status == "stopped" and not observed["email_stop_notified"])
        or (status == "ready" and observed["email_stop_notified"])
    ):
        try:
            send_mail(
                settings,
                worker_alert_message(
                    settings,
                    server_id=row["server_id"],
                    label=label,
                    recovered=status == "ready",
                    detected_at=detected_at,
                    last_heartbeat=observed["last_heartbeat"],
                    last_worker_error=observed["last_worker_error"],
                ),
            )
            observed["email_stop_notified"] = status == "stopped"
        except MailDeliveryError as error:
            observed["email_error"] = str(error)
    return observed


def _read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"checked_at": None, "servers": []}


def _check(path: Path) -> dict[str, Any]:
    targets = select_targets()
    labels = {target.id: target.label for target in targets}
    previous = {row["server_id"]: row for row in _read_state(path)["servers"]}
    rows = asyncio.run(
        fleet_worker_action(
            SshWorkerTransport(timeout_seconds=10),
            [RemoteTarget(target.id, target.ssh_host) for target in targets],
            "status",
        )
    )
    settings, email_error = _email_settings()
    checked_at = datetime.now(timezone.utc).isoformat()
    state = {
        "checked_at": checked_at,
        "email_error": email_error,
        "servers": [
            _observe(
                row,
                labels[row["server_id"]],
                previous.get(row["server_id"], {}),
                settings,
                checked_at,
            )
            for row in rows
        ],
    }
    _save_state(path, state)
    return state


def run_alerts(
    action: str, *, as_json: bool, email_config: Path | None = None, no_email: bool = False
) -> int:
    if sys.platform != "darwin":
        raise CommandError("unsupported_platform", "Worker alerts currently require macOS")
    path = Path.home() / ".local/state/gpuops/alerts.json"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    if action == "_check":
        _check(path)
        return 0
    if action == "enable":
        if not select_targets():
            raise CommandError("no_servers", "Register a server with gpuops server add first")
        if email_config is not None:
            configured_settings = _read_email(email_config.expanduser())
            destination = _email_path()
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_write_bytes(
                destination, configured_settings.model_dump_json(indent=2).encode("utf-8")
            )
        elif no_email:
            _email_path().unlink(missing_ok=True)
            state = _read_state(path)
            for row in state["servers"]:
                row["email_stop_notified"] = False
                row.pop("email_error", None)
            _save_state(path, state)
    settings, email_error = _email_settings()
    if action == "test":
        payload: dict[str, Any] = {"ok": True, "notification_submitted": False}
        try:
            _notify("GPUOps", "Test notification: Worker alerts are available on this Mac.")
            payload["notification_submitted"] = True
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            payload.update(ok=False, notification_error=str(error))
        if settings is not None:
            try:
                send_mail(settings, worker_alert_test_message(settings))
                payload["email_accepted"] = True
            except MailDeliveryError as error:
                payload.update(ok=False, email_accepted=False, email_error=str(error))
        elif email_error:
            payload.update(ok=False, email_error=email_error)
        message = "Test notification attempted. Check the channel results below."
    else:
        enabled = _launchctl("print", service, check=False)
        if action == "enable":
            config = plistlib.dumps(
                {
                    "Label": LABEL,
                    "ProgramArguments": [sys.executable, "-m", "gpuops", "alerts", "_check"],
                    "RunAtLoad": True,
                    "StartInterval": INTERVAL_SECONDS,
                    "LimitLoadToSessionType": "Aqua",
                }
            )
            if not enabled or not plist.exists() or plist.read_bytes() != config:
                if enabled:
                    _launchctl("bootout", service)
                plist.parent.mkdir(parents=True, exist_ok=True)
                plist.write_bytes(config)
                _launchctl("bootstrap", domain, str(plist))
                enabled = True
        elif action == "disable":
            if enabled:
                _launchctl("bootout", service)
            plist.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            enabled = False
        payload = {"ok": True, "enabled": enabled, **_read_state(path)}
        message = f"Worker alerts {'enabled' if enabled else 'disabled'}."
        if action == "status":
            checked_at = payload["checked_at"]
            if checked_at:
                message += f" Last check: {_local_time(checked_at)}."
            for row in payload["servers"]:
                detail = row["reason"] if row["reason"] != row["status"] else None
                detail = f" — {detail}" if detail else ""
                if row.get("notification_error"):
                    detail += f" — Notification error: {row['notification_error']}"
                if row.get("email_error"):
                    detail += f" — Email error: {row['email_error']}"
                message += f"\n{row['label']}: {row['status']}{detail}"
    payload["email_configured"] = settings is not None
    payload["email_recipient"] = settings.recipient if settings else None
    payload["email_error"] = (
        email_error or payload.get("email_error") if action == "test" else email_error
    )
    message += f"\nEmail: {settings.recipient}" if settings else "\nEmail: not configured."
    if payload["email_error"]:
        message += f"\nEmail error: {payload['email_error']}"
    if action == "test":
        message += f"\nDesktop submitted: {payload['notification_submitted']}"
        if settings:
            message += f"\nEmail accepted: {payload['email_accepted']}"
        if payload.get("notification_error"):
            message += f"\nNotification error: {payload['notification_error']}"
    print(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else message)
    return 0 if payload["ok"] else 1
