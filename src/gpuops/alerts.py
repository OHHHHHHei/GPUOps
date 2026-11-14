"""Mac-side desktop and email notifications for stopped Workers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

from gpuops.cli_errors import CommandError
from gpuops.config import select_targets
from gpuops.file_io import atomic_write_bytes
from gpuops.models import utc_now
from gpuops.notifications.config import SMTPSettings
from gpuops.notifications.mail import (
    MailDeliveryError,
    send_mail,
    worker_alert_message,
    worker_alert_test_message,
)


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


def observe(
    server_id: str,
    label: str,
    worker: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    worker = worker or {}
    if error is not None or not worker:
        status = "unknown"
    elif worker.get("process") in {"stopped", "stale"}:
        status = "stopped"
    elif worker.get("ready") is True:
        status = "ready"
    else:
        status = "not_ready"
    return {
        "server_id": server_id,
        "label": label,
        "status": status,
        "reason": error or worker.get("reason"),
        "last_heartbeat": worker.get("heartbeat_at"),
        "last_worker_error": worker.get("last_tick_error"),
    }


def _desktop_message(row: dict[str, Any]) -> str:
    if row["status"] == "ready":
        return "Worker recovered and is ready to schedule."
    details = ["Worker stopped."]
    if row["last_heartbeat"]:
        details.append(f"Last heartbeat: {_local_time(row['last_heartbeat'])}")
    if row["last_worker_error"]:
        details.append(f"Last Worker error: {row['last_worker_error']}")
    return " ".join(details)[:320]


def _channel_due(row: dict[str, Any], field: str) -> bool:
    return (row["status"] == "stopped" and not row[field]) or (
        row["status"] == "ready" and row[field]
    )


def process_observations(
    path: Path,
    lock: Callable[[], AbstractContextManager[None]],
    enabled: Callable[[], bool],
    rows: list[dict[str, Any]],
) -> None:
    checked_at = utc_now().isoformat()
    with lock():
        if not enabled():
            return
        _, email_error = _email_settings()
        previous = {row["server_id"]: row for row in _read_state(path)["servers"]}
        for row in rows:
            old = previous.get(row["server_id"], {})
            row["stop_notified"] = old.get("stop_notified", False)
            row["email_stop_notified"] = old.get("email_stop_notified", False)
        _save_state(path, {"checked_at": checked_at, "email_error": email_error, "servers": rows})

    for index, row in enumerate(rows):
        for channel in ("desktop", "email"):
            email = channel == "email"
            field = "email_stop_notified" if email else "stop_notified"
            error_field = "email_error" if email else "notification_error"
            with lock():
                if not enabled():
                    return
                observed = _read_state(path)["servers"][index]
                settings = _email_settings()[0] if email else None
                if not _channel_due(observed, field) or (email and settings is None):
                    continue
            error_message = None
            try:
                if settings is not None:
                    send_mail(
                        settings,
                        worker_alert_message(
                            settings,
                            server_id=row["server_id"],
                            label=row["label"],
                            recovered=row["status"] == "ready",
                            detected_at=checked_at,
                            last_heartbeat=row["last_heartbeat"],
                            last_worker_error=row["last_worker_error"],
                        ),
                    )
                else:
                    _notify(f"GPUOps: {row['label']}", _desktop_message(row))
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                error_message = str(error)

            with lock():
                if email and settings != _email_settings()[0]:
                    # A changed or removed email configuration has its own delivery state.
                    continue
                latest = _read_state(path)
                current = latest["servers"][index]
                if error_message is None:
                    current[field] = row["status"] == "stopped"
                else:
                    current[error_field] = error_message
                _save_state(path, latest)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"checked_at": None, "servers": []}


def run_alerts(
    action: str, *, as_json: bool, email_config: Path | None = None, no_email: bool = False
) -> int:
    if sys.platform != "darwin":
        raise CommandError("unsupported_platform", "Worker alerts currently require macOS")
    from gpuops.background import Background

    background = Background()
    path = background.alert_state
    if action == "enable":
        if not select_targets():
            raise CommandError("no_servers", "Register a server with gpuops server add first")
        configured = _read_email(email_config.expanduser()) if email_config is not None else None
        events = background.event_store()
        with events.lock():
            if configured is not None:
                destination = _email_path()
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic_write_bytes(destination, configured.model_dump_json(indent=2).encode())
            elif no_email:
                _email_path().unlink(missing_ok=True)
                state = _read_state(path)
                for row in state["servers"]:
                    row["email_stop_notified"] = False
                    row.pop("email_error", None)
                _save_state(path, state)
            background.set_switch_locked("alerts", True)
            background.sync_locked(events)
    elif action == "disable":
        events = background.event_store()
        with events.lock():
            background.set_switch_locked("alerts", False)
            background.sync_locked(events)
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
        enabled = background.alerts_enabled()
        payload = {"ok": True, "enabled": enabled, **_read_state(path)}
        message = f"Worker alerts {'enabled' if enabled else 'disabled'}."
        if action == "status":
            checked_at = payload["checked_at"]
            if checked_at:
                message += f" Last check: {_local_time(checked_at)}."
            for row in payload["servers"]:
                detail = row.get("reason") if row.get("reason") != row["status"] else None
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
