"""Configure server-side email locally or over the registered SSH routes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gpuops.cli_errors import CommandError, error_payload, print_error
from gpuops.config import (
    MONITOR_TIMEOUT_SECONDS,
    ServerTarget,
    select_client_targets,
    validate_server_id,
)
from gpuops.execution import HostClient
from gpuops.notifications import (
    MailDeliveryError,
    NotificationConfigError,
    SMTPSettings,
    configure,
    disable,
    load_config,
    notification_status,
    send_test,
)
from gpuops.ssh import SSHError
from gpuops.worker.app import DEFAULT_STATE_DIRECTORY

_ACTIONS = ("configure", "status", "test", "disable")
_REMOTE_PYTHON = "~/.local/share/gpuops/current/venv/bin/python"
_REMOTE_MODULE = "gpuops.notifications.commands"


def _read_settings(raw: bytes) -> SMTPSettings:
    try:
        return SMTPSettings.model_validate_json(raw)
    except ValidationError as error:
        fields = sorted(
            {
                item["loc"][0]
                for item in error.errors(include_input=False)
                if item["loc"] and item["loc"][0] in SMTPSettings.model_fields
            }
        )
        details = f"; check {', '.join(fields)}" if fields else ""
        raise CommandError(
            "invalid_notification_config", f"expected a JSON object with SMTP settings{details}"
        ) from None


def _local_response(
    action: str, server_id: str, settings: SMTPSettings | None = None
) -> dict[str, Any]:
    try:
        if action == "configure":
            assert settings is not None
            configure(settings)
        elif action == "disable":
            disable()
        elif action == "test":
            config = load_config()
            if config.smtp is None:
                raise CommandError(
                    "notification_not_configured", "configure SMTP settings before sending a test"
                )
            send_test(config, server_id)
            return {"ok": True, "result": {"accepted": True}}
        return {"ok": True, "result": notification_status(DEFAULT_STATE_DIRECTORY)}
    except NotificationConfigError as error:
        return error_payload(CommandError("invalid_notification_config", str(error)))
    except MailDeliveryError as error:
        return error_payload(CommandError("notification_delivery_failed", str(error)))
    except (CommandError, OSError) as error:
        return error_payload(error)


async def _remote_response(
    client: HostClient, target: ServerTarget, action: str, body: bytes
) -> dict[str, Any]:
    command = await client.run_command(
        target.ssh_host,
        [_REMOTE_PYTHON, "-I", "-m", _REMOTE_MODULE, action, "--server-id", target.id],
        stdin=body,
        timeout_seconds=MONITOR_TIMEOUT_SECONDS,
    )
    if command.returncode == 255:
        raise CommandError("host_unavailable", "could not connect to the server over SSH")
    if command.returncode != 0:
        if ("No module named" in command.stderr and "gpuops.notifications" in command.stderr) or (
            command.returncode == 127 and "venv/bin/python" in command.stderr
        ):
            raise CommandError(
                "notification_unsupported",
                "server does not have email notification support; upgrade with "
                f"gpuops deploy install --server {target.id} --json",
            )
        raise CommandError("notification_command_failed", "server notification command failed")
    try:
        response = json.loads(command.stdout)
    except ValueError:
        response = None
    if isinstance(response, dict):
        if response.get("ok") is True and isinstance(response.get("result"), dict):
            return {"ok": True, "result": response["result"]}
        if (
            response.get("ok") is False
            and isinstance(response.get("error_code"), str)
            and isinstance(response.get("error"), str)
        ):
            return error_payload(CommandError(response["error_code"], response["error"]))
    raise CommandError("notification_invalid_response", "server returned an invalid response")


async def run_notification_command(args: argparse.Namespace) -> int:
    """Apply one notification action and report each selected server's outcome."""
    action = args.notify_action
    settings = None
    body = b""
    try:
        targets = select_client_targets(args.server)
        if not targets:
            raise CommandError("invalid_argument", "no servers registered; use gpuops server add")
        client = HostClient()
        if action == "configure":
            body = (
                sys.stdin.buffer.read()
                if args.config_file == "-"
                else Path(args.config_file).expanduser().read_bytes()
            )
            settings = _read_settings(body)
    except (OSError, ValueError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1

    async def execute(target: ServerTarget) -> dict[str, Any]:
        online = True
        try:
            if client.local is not None and target.id == client.local.id:
                response = await asyncio.to_thread(_local_response, action, target.id, settings)
            else:
                response = await _remote_response(client, target, action, body)
        except CommandError as error:
            online = error.code != "host_unavailable"
            response = error_payload(error)
        except (OSError, SSHError) as error:
            online = False
            response = error_payload(CommandError("host_unavailable", str(error)))
        return {"server_id": target.id, "online": online, **response}

    rows = await asyncio.gather(*(execute(target) for target in targets))
    success_count = sum(row["ok"] for row in rows)
    outcome = "success" if success_count == len(rows) else "partial" if success_count else "failed"
    payload = {"ok": outcome == "success", "outcome": outcome, "servers": rows}
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            server_id = row["server_id"]
            if not row["ok"]:
                print(f"{server_id}: {row['error']}", file=sys.stderr)
            elif action == "test":
                print(f"{server_id}: test message accepted by SMTP")
            else:
                print(f"{server_id}: {json.dumps(row['result'], ensure_ascii=False)}")
    return 0 if payload["ok"] else 1


def _run_installed_command() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=_ACTIONS)
    parser.add_argument("--server-id", required=True)
    args = parser.parse_args()
    try:
        validate_server_id(args.server_id)
        settings = _read_settings(sys.stdin.buffer.read()) if args.action == "configure" else None
        response = _local_response(args.action, args.server_id, settings)
    except (OSError, ValueError) as error:
        response = error_payload(error)
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_installed_command())
