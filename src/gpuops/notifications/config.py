from __future__ import annotations

from datetime import datetime
from email.errors import HeaderParseError
from email.headerregistry import Address
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from gpuops.file_io import atomic_write_bytes
from gpuops.models import StrictModel, utc_now


class NotificationConfigError(ValueError):
    pass


class _InvalidNotificationConfig(NotificationConfigError):
    pass


class SMTPSettings(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    host: str
    port: int = Field(default=465, ge=1, le=65535)
    security: Literal["ssl", "starttls"] = "ssl"
    username: str | None = Field(default=None, min_length=1, repr=False)
    password: str | None = Field(default=None, min_length=1, repr=False)
    sender: str
    recipient: str

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if not value or any(char.isspace() or char == "\x00" for char in value):
            raise ValueError("SMTP host must be a hostname or IP address")
        return value

    @field_validator("sender", "recipient")
    @classmethod
    def validate_address(cls, value: str) -> str:
        try:
            address = Address(addr_spec=value)
        except (ValueError, HeaderParseError):
            raise ValueError("must be a single email address") from None
        if not address.username or not address.domain:
            raise ValueError("must be a single email address")
        return address.addr_spec

    @model_validator(mode="after")
    def validate_credentials(self) -> SMTPSettings:
        if (self.username is None) != (self.password is None):
            raise ValueError("SMTP username and password must be provided together")
        return self


class NotificationConfig(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    enabled: bool = False
    enabled_since: AwareDatetime | None = None
    smtp: SMTPSettings | None = None

    @model_validator(mode="after")
    def validate_enabled(self) -> NotificationConfig:
        if self.enabled and (self.enabled_since is None or self.smtp is None):
            raise ValueError("enabled notifications require SMTP settings and an enable time")
        return self


def config_path() -> Path:
    return Path.home() / ".config/gpuops/notifications.json"


def load_config(path: Path | None = None) -> NotificationConfig:
    path = path if path is not None else config_path()
    try:
        return NotificationConfig.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return NotificationConfig()
    except OSError:
        raise NotificationConfigError(f"cannot read notification configuration: {path}") from None
    except ValueError:
        raise _InvalidNotificationConfig(f"invalid notification configuration: {path}") from None


def configure(
    settings: SMTPSettings, path: Path | None = None, now: datetime | None = None
) -> NotificationConfig:
    path = path if path is not None else config_path()
    try:
        previous = load_config(path)
    except _InvalidNotificationConfig:
        previous = NotificationConfig()
    config = NotificationConfig(
        enabled=True,
        enabled_since=previous.enabled_since if previous.enabled else (now or utc_now()),
        smtp=settings,
    )
    _write_json(path, config)
    return config


def disable(path: Path | None = None) -> NotificationConfig:
    path = path if path is not None else config_path()
    previous = load_config(path)
    config = NotificationConfig(smtp=previous.smtp)
    if previous.enabled:
        _write_json(path, config)
    return config


def _write_json(path: Path, model: StrictModel) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    atomic_write_bytes(path, (model.model_dump_json(indent=2) + "\n").encode("utf-8"))
