"""Email and external-harness delivery for persisted follow-up events.

An external adapter must durably spool an accepted event and deduplicate it by
``event_id`` before returning a receipt.  The receipt confirms only that the
adapter accepted the event; it does not mean that a model received or completed
the requested work.  This protocol does not turn a one-shot CLI or SDK call into
the original interactive harness session.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from gpuops.file_io import atomic_write_bytes
from gpuops.models import StrictModel
from gpuops.notifications.config import SMTPSettings
from gpuops.notifications.mail import followup_message, send_mail

if TYPE_CHECKING:
    from gpuops.followups import CodexQueue, Followup


ADAPTER_API_VERSION = "gpuops.completion/v1"
DELIVERY_API_VERSION = "gpuops.delivery/v1"
ADAPTER_TIMEOUT_SECONDS = 30
_ADAPTER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class DeliveryUncertain(RuntimeError):
    """The adapter may have accepted an event without returning a valid receipt."""


class AdapterSettings(StrictModel):
    command: list[str] = Field(min_length=1)
    idempotent: bool = False

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value[0]:
            raise ValueError("adapter executable must be non-empty")
        if any("\x00" in argument for argument in value):
            raise ValueError("adapter command arguments must contain no NUL")
        return value


def _email_path() -> Path:
    return Path.home() / ".config/gpuops/followup-email.json"


def _worker_alert_email_path() -> Path:
    return Path.home() / ".config/gpuops/worker-alerts-email.json"


def _adapter_path() -> Path:
    return Path.home() / ".config/gpuops/followup-adapters.json"


def _read_email_file(path: Path) -> SMTPSettings:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError:
        raise OSError(f"cannot read follow-up SMTP settings: {path}") from None
    try:
        return SMTPSettings.model_validate_json(raw)
    except ValueError:
        raise ValueError(f"invalid follow-up SMTP settings: {path}") from None


def configure_email(path: Path) -> SMTPSettings:
    """Install SMTP settings from ``path`` without attempting a delivery."""

    settings = _read_email_file(path.expanduser())
    destination = _email_path()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_bytes(
        destination,
        (settings.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    return settings


def read_email_settings() -> SMTPSettings | None:
    """Read dedicated settings, falling back to the existing Worker-alert account."""

    try:
        return _read_email_file(_email_path())
    except FileNotFoundError:
        try:
            return _read_email_file(_worker_alert_email_path())
        except FileNotFoundError:
            return None


def send_followup_email(followup: Followup) -> None:
    settings = read_email_settings()
    if settings is None:
        raise RuntimeError("follow-up email is not configured")
    if followup.notify == "terminal":
        subject = (
            "[GPUOps] Experiment finished"
            if followup.mode == "per_job"
            else "[GPUOps] Experiments finished"
        )
        body = "\n".join(
            [
                f"Follow-up event: {followup.id}",
                f"Group: {followup.group_id or followup.id}",
                *(
                    [f"Completed job: {followup.server_id}/{followup.run_id}"]
                    if followup.mode == "per_job"
                    else []
                ),
                f"Requested action: {followup.action}",
                "",
                "Experiment outcomes:",
                json.dumps(followup.outcomes, ensure_ascii=False, indent=2, default=str),
            ]
        )
    elif followup.notify == "analysis_complete":
        if not followup.summary:
            raise ValueError("analysis-complete email requires a summary")
        subject = "[GPUOps] Analysis complete"
        body = "\n".join(
            [
                f"Follow-up event: {followup.id}",
                f"Group: {followup.group_id or followup.id}",
                "",
                "Analysis summary:",
                followup.summary,
            ]
        )
    else:
        raise ValueError("follow-up email notification is disabled")
    send_mail(
        settings,
        followup_message(settings, event_id=followup.id, subject=subject, body=body),
    )


def _validate_adapter_name(name: str) -> str:
    if name == "codex-desktop":
        raise ValueError("codex-desktop is a reserved built-in harness")
    if not _ADAPTER_NAME.fullmatch(name):
        raise ValueError("adapter name must use lowercase letters, digits, '.', '_', or '-'")
    return name


def _read_adapter_settings(path: Path) -> AdapterSettings:
    try:
        raw = path.read_bytes()
    except OSError:
        raise OSError(f"cannot read adapter settings: {path}") from None
    try:
        return AdapterSettings.model_validate_json(raw)
    except ValueError:
        raise ValueError(f"invalid adapter settings: {path}") from None


def load_adapters() -> dict[str, AdapterSettings]:
    path = _adapter_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError
        adapters: dict[str, AdapterSettings] = {}
        for name, value in raw.items():
            adapters[_validate_adapter_name(name)] = AdapterSettings.model_validate(value)
        return adapters
    except FileNotFoundError:
        return {}
    except OSError:
        raise OSError(f"cannot read follow-up adapter configuration: {path}") from None
    except (json.JSONDecodeError, ValueError):
        raise ValueError(f"invalid follow-up adapter configuration: {path}") from None


def configure_adapter(name: str, file: Path) -> AdapterSettings:
    """Install a validated adapter definition without executing its command."""

    name = _validate_adapter_name(name)
    settings = _read_adapter_settings(file.expanduser())
    adapters = load_adapters()
    adapters[name] = settings
    destination = _adapter_path()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        adapter_name: adapter.model_dump(mode="json")
        for adapter_name, adapter in sorted(adapters.items())
    }
    atomic_write_bytes(
        destination,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return settings


def adapter_capabilities() -> list[dict[str, object]]:
    capabilities: list[dict[str, object]] = [
        {
            "name": "codex-desktop",
            "idempotent": False,
            "builtin": True,
            "session_continuity": "native",
        }
    ]
    capabilities.extend(
        {
            "name": name,
            "idempotent": settings.idempotent,
            "builtin": False,
            "session_continuity": "unverified",
        }
        for name, settings in sorted(load_adapters().items())
    )
    return capabilities


def _adapter_request(followup: Followup) -> dict[str, Any]:
    return {
        "api_version": ADAPTER_API_VERSION,
        "event_id": followup.id,
        "group_id": followup.group_id,
        "target": {
            "harness": followup.harness,
            "session_id": followup.thread_id,
            "workspace": followup.workspace,
        },
        "mode": followup.mode,
        "trigger_job": (
            {"server_id": followup.server_id, "run_id": followup.run_id}
            if followup.mode == "per_job"
            else None
        ),
        "action": followup.action,
        "context": followup.context,
        "jobs": followup.outcomes,
        "message": followup.message,
    }


class DeliveryRouter:
    def __init__(self, codex: CodexQueue | None, adapters: dict[str, AdapterSettings]):
        self.codex = codex
        self.adapters = adapters

    def send(self, followup: Followup) -> subprocess.CompletedProcess[str]:
        if followup.harness == "codex-desktop":
            if self.codex is None:
                raise OSError("Codex desktop delivery is not available")
            return self.codex.send(followup)

        settings = self.adapters.get(followup.harness)
        if settings is None:
            raise OSError(f"follow-up adapter is not configured: {followup.harness}")
        request = json.dumps(_adapter_request(followup), ensure_ascii=False, default=str) + "\n"
        try:
            result = subprocess.run(
                settings.command,
                input=request,
                cwd=followup.workspace,
                capture_output=True,
                text=True,
                timeout=ADAPTER_TIMEOUT_SECONDS,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise DeliveryUncertain(
                "adapter delivery timed out; delivery may have succeeded"
            ) from None
        if result.returncode != 0:
            raise DeliveryUncertain(
                f"adapter exited with status {result.returncode}; delivery may have succeeded"
            )
        try:
            response = json.loads(result.stdout)
            if not isinstance(response, dict):
                raise ValueError
            if response.get("api_version") != DELIVERY_API_VERSION:
                raise ValueError
            if response.get("event_id") != followup.id:
                raise ValueError
            accepted = response.get("accepted")
            if not isinstance(accepted, bool):
                raise ValueError
            if not accepted:
                error = response.get("error")
                if not isinstance(error, str) or not error.strip():
                    error = "adapter rejected delivery"
                return subprocess.CompletedProcess(settings.command, 1, "", error.strip())
            receipt = response.get("receipt")
            if not isinstance(receipt, str) or not receipt.strip():
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise DeliveryUncertain(
                "adapter returned an invalid receipt; delivery may have succeeded"
            ) from None
        return subprocess.CompletedProcess(settings.command, 0, receipt.strip(), "")

    def can_retry_uncertain(self, followup: Followup) -> bool:
        if followup.harness == "codex-desktop":
            return False
        settings = self.adapters.get(followup.harness)
        return settings.idempotent if settings is not None else False
