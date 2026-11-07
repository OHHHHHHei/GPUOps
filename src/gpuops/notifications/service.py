from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, Field

from gpuops.core.queue import QueueStore
from gpuops.models import StrictModel, utc_now
from gpuops.notifications.config import (
    NotificationConfig,
    NotificationConfigError,
    SMTPSettings,
    _write_json,
    load_config,
)
from gpuops.notifications.mail import delivery_error_message, job_message, send_mail

_POLL_SECONDS = 30
_RETRY_DELAY = timedelta(seconds=60)
_BATCH_SIZE = 20
_LOG = logging.getLogger(__name__)


class _NotificationStateError(ValueError):
    pass


class _NotificationState(StrictModel):
    enabled_since: AwareDatetime | None = None
    cursor: tuple[AwareDatetime, str] | None = None
    retrying_run_id: str | None = None
    last_error: str | None = None
    next_attempt_at: AwareDatetime | None = None
    last_sent_at: AwareDatetime | None = None
    sent_count: int = Field(default=0, ge=0)


def _load_state(path: Path) -> _NotificationState:
    try:
        return _NotificationState.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return _NotificationState()
    except OSError:
        raise _NotificationStateError(f"cannot read notification state: {path}") from None
    except ValueError:
        raise _NotificationStateError(f"invalid notification state: {path}") from None


def _active_state(config: NotificationConfig, saved: _NotificationState) -> _NotificationState:
    if not config.enabled or config.enabled_since is None:
        return _NotificationState()
    if saved.enabled_since != config.enabled_since or saved.cursor is None:
        return _NotificationState(
            enabled_since=config.enabled_since,
            cursor=(config.enabled_since, ""),
        )
    return saved


def notification_status(state_directory: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Read the effective settings and delivery state without exposing credentials."""
    config_error = state_error = None
    try:
        config = load_config(config_path)
    except NotificationConfigError as error:
        config_error = str(error)
        config = NotificationConfig()
    try:
        state = _load_state(state_directory / "notifications.json")
    except _NotificationStateError as error:
        state_error = str(error)
        state = _NotificationState()
    active = _active_state(config, state)
    return {
        **active.model_dump(mode="json"),
        "configured": config.smtp is not None,
        "enabled": config.enabled,
        "smtp": (
            config.smtp.model_dump(mode="json", exclude={"username", "password"})
            if config.smtp
            else None
        ),
        "config_error": config_error,
        "state_error": state_error,
    }


class JobNotifier:
    """Deliver newly terminal jobs from the queue independently of worker ticks."""

    def __init__(
        self,
        store: QueueStore,
        server_id: str,
        state_directory: Path,
        *,
        config_path: Path | None = None,
        clock: Callable[[], datetime] = utc_now,
        sender: Callable[[SMTPSettings, EmailMessage], None] = send_mail,
    ) -> None:
        self.store = store
        self.server_id = server_id
        self.state_path = state_directory / "notifications.json"
        self.config_path = config_path
        self.clock = clock
        self.sender = sender
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_accepted: _NotificationState | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpuops-job-notifier", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Let an in-flight send finish if it can; the worker must exit promptly.
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(_POLL_SECONDS)

    def check_once(self) -> int:
        """Deliver up to one batch, isolating notification errors from the worker."""
        if self._stop.is_set():
            return 0
        try:
            return self._deliver_batch()
        except (NotificationConfigError, _NotificationStateError) as error:
            _LOG.warning("Job notification check failed: %s", error)
        except Exception as error:
            _LOG.warning("Job notification check failed (%s)", type(error).__name__)
        return 0

    def _deliver_batch(self) -> int:
        config = load_config(self.config_path)
        if not config.enabled or config.smtp is None:
            self._pending_accepted = None
            return 0
        pending = self._pending_accepted
        if pending is not None and pending.enabled_since == config.enabled_since:
            self._save_accepted(pending)
            state = pending
        else:
            self._pending_accepted = None
            state = _active_state(config, _load_state(self.state_path))
        if state.next_attempt_at is not None and self.clock() < state.next_attempt_at:
            return 0
        assert state.cursor is not None
        delivered = 0
        for job in self.store.terminal_after(state.cursor, limit=_BATCH_SIZE):
            if self._stop.is_set():
                break
            if delivered and load_config(self.config_path) != config:
                break
            record = self.store.get(job.run_id)
            message = job_message(config.smtp, self.server_id, record)
            if self._stop.is_set():
                break
            try:
                self.sender(config.smtp, message)
            except Exception as error:
                state = state.model_copy(
                    update={
                        "retrying_run_id": job.run_id,
                        "last_error": delivery_error_message(error),
                        "next_attempt_at": self.clock() + _RETRY_DELAY,
                    }
                )
                _write_json(self.state_path, state)
                break
            assert job.finished_at is not None
            state = state.model_copy(
                update={
                    "cursor": (job.finished_at, job.run_id),
                    "retrying_run_id": None,
                    "last_error": None,
                    "next_attempt_at": None,
                    "last_sent_at": self.clock(),
                    "sent_count": state.sent_count + 1,
                }
            )
            self._save_accepted(state)
            delivered += 1
        return delivered

    def _save_accepted(self, state: _NotificationState) -> None:
        # A failed disk write must not make this worker resend an accepted message.
        self._pending_accepted = state
        _write_json(self.state_path, state)
        self._pending_accepted = None
