"""Opt-in macOS notifications for stopped Workers on registered servers."""

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
from gpuops.queries import fleet_worker_action
from gpuops.transport import SshWorkerTransport

LABEL = "io.gpuops.worker-alerts"
INTERVAL_SECONDS = 60


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


def _observe(row: dict[str, Any], label: str, stop_notified: bool) -> dict[str, Any]:
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
        "stop_notified": stop_notified,
    }
    try:
        if status == "stopped" and not stop_notified:
            details = ["Worker stopped."]
            if observed["last_heartbeat"]:
                details.append(f"Last heartbeat: {_local_time(observed['last_heartbeat'])}")
            if observed["last_worker_error"]:
                details.append(f"Last Worker error: {observed['last_worker_error']}")
            _notify(f"GPUOps: {label}", " ".join(details)[:320])
            observed["stop_notified"] = True
        elif status == "ready" and stop_notified:
            _notify(f"GPUOps: {label}", "Worker recovered and is ready to schedule.")
            observed["stop_notified"] = False
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        observed["notification_error"] = str(error)
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
    state = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "servers": [
            _observe(
                row,
                labels[row["server_id"]],
                previous.get(row["server_id"], {}).get("stop_notified", False),
            )
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return state


def run_alerts(action: str, *, as_json: bool) -> int:
    if sys.platform != "darwin":
        raise CommandError("unsupported_platform", "Worker alerts currently require macOS")
    path = Path.home() / ".local/state/gpuops/alerts.json"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    if action == "_check":
        _check(path)
        return 0
    if action == "test":
        _notify("GPUOps", "Test notification: Worker alerts are available on this Mac.")
        payload: dict[str, Any] = {"ok": True, "notification_submitted": True}
        message = "Test notification sent. Check macOS Notification Center."
    else:
        enabled = _launchctl("print", service, check=False)
        if action == "enable":
            if not select_targets():
                raise CommandError("no_servers", "Register a server with gpuops server add first")
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
                message += f"\n{row['label']}: {row['status']}{detail}"
    print(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else message)
    return 0
