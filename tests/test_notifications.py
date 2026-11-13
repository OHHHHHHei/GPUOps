from __future__ import annotations

import errno
import json
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from gpuops.core.queue import QueueStore
from gpuops.models import JobSpec, JobStatus, LaunchInfo
from gpuops.notifications import (
    JobNotifier,
    MailDeliveryError,
    SMTPSettings,
    configure,
    disable,
    load_config,
    mail,
    notification_status,
    send_test,
    service,
)

NOW = datetime(2026, 11, 7, tzinfo=timezone.utc)
SECOND = timedelta(seconds=1)
PASSWORD = "private-app-password"
USERNAME = "private-account-name"


@pytest.fixture
def settings() -> SMTPSettings:
    return SMTPSettings(
        host="smtp.example.com",
        sender="jobs@example.com",
        recipient="researcher@example.com",
        username=USERNAME,
        password=PASSWORD,
    )


@pytest.fixture
def spec() -> JobSpec:
    return JobSpec(
        id="train",
        project="experiments",
        command=["python", "private-command.py"],
        working_directory="/work",
        runtime={"environment": {"API_KEY": "private-environment"}},
        metadata={"name": "Train baseline"},
    )


@pytest.fixture
def queue(tmp_path: Path) -> QueueStore:
    return QueueStore(tmp_path / "queue")


def _cancel(queue: QueueStore, spec: JobSpec, run_id: str, *, at: datetime) -> None:
    queue.enqueue(spec, run_id=run_id, now=at - SECOND)
    queue.request_cancel(run_id, now=at)


def _start(queue: QueueStore, spec: JobSpec, run_id: str, *, at: datetime) -> None:
    queued = queue.enqueue(spec, run_id=run_id, now=at)
    queue.lease(
        run_id,
        server_id="gpu01",
        gpu_indices=[0],
        worker_id="worker",
        ttl=timedelta(minutes=5),
        expected_revision=queued.revision,
        now=at,
    )
    queue.transition(run_id, JobStatus.STARTING, now=at)
    queue.transition(run_id, JobStatus.RUNNING, launch=LaunchInfo(pid=123, started_at=at), now=at)


def _run_id(message: EmailMessage) -> str:
    return next(
        line.removeprefix("Run ID: ")
        for line in message.get_content().splitlines()
        if line.startswith("Run ID: ")
    )


def test_first_enable_and_restart_deliver_only_new_terminal_jobs(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config" / "notifications.json"
    _cancel(queue, spec, "old", at=NOW - SECOND)
    _start(queue, spec, "running-at-enable", at=NOW - 3 * SECOND)
    configure(settings, config_file, NOW)
    messages = []

    def notifier(store):
        return JobNotifier(
            store,
            "gpu01",
            tmp_path,
            config_path=config_file,
            clock=lambda: NOW + 10 * SECOND,
            sender=lambda _, message: messages.append(message),
        )

    assert notifier(queue).check_once() == 0
    queue.transition("running-at-enable", JobStatus.SUCCEEDED, exit_code=0, now=NOW + SECOND)
    _start(queue, spec, "failed", at=NOW + 2 * SECOND)
    queue.transition(
        "failed",
        JobStatus.FAILED,
        now=NOW + 4 * SECOND,
        failure_reason="worker lost process\nprivate full log" + "x" * 500,
    )
    _cancel(queue, spec, "cancelled", at=NOW + 5 * SECOND)
    assert notifier(queue).check_once() == 3
    assert notifier(QueueStore(queue.root)).check_once() == 0
    assert [_run_id(message) for message in messages] == [
        "running-at-enable",
        "failed",
        "cancelled",
    ]

    success, failure, cancelled = (message.get_content() for message in messages)
    assert "Status: succeeded" in success
    assert "Server: gpu01" in success and "Project: experiments" in success
    assert "Job: Train baseline" in success and "Exit code: 0" in success
    assert "Duration: 0:00:04" in success and "Started: 2026-11-06" in success
    assert "Finished: 2026-11-07" in success
    assert "gpuops logs --server gpu01 -- running-at-enable" in success
    assert "Status: failed" in failure and "Reason: worker lost process" in failure
    assert "Status: cancelled" in cancelled and "Started: not started" in cancelled
    assert "Duration:" not in cancelled
    assert not any(
        secret in str(message)
        for message in messages
        for secret in (PASSWORD, "private-command.py", "private-environment", "private full log")
    )
    status = notification_status(tmp_path, config_file)
    assert status["sent_count"] == 3
    assert status["last_sent_at"] is not None
    assert status["cursor"][1] == "cancelled"
    assert config_file.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "notifications.json").stat().st_mode & 0o777 == 0o600
    assert PASSWORD not in json.dumps(status) and USERNAME not in json.dumps(status)


def test_failed_delivery_retries_after_restart(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config" / "notifications.json"
    configure(settings, config_file, NOW)
    _cancel(queue, spec, "first", at=NOW + SECOND)
    _cancel(queue, spec, "second", at=NOW + 2 * SECOND)
    now = NOW + 3 * SECOND
    attempted = []

    def send(_, message):
        attempted.append(_run_id(message))
        if len(attempted) == 1:
            raise smtplib.SMTPAuthenticationError(535, PASSWORD.encode())

    def notifier():
        return JobNotifier(
            queue, "gpu01", tmp_path, config_path=config_file, clock=lambda: now, sender=send
        )

    cursor = notification_status(tmp_path, config_file)["cursor"]
    assert notifier().check_once() == 0
    status = notification_status(tmp_path, config_file)
    assert status["cursor"] == cursor
    assert status["retrying_run_id"] == "first"
    assert status["last_error"] == "SMTP authentication failed (status 535)"
    assert status["next_attempt_at"] is not None
    assert notifier().check_once() == 0
    assert attempted == ["first"]

    now += timedelta(minutes=1)
    assert notifier().check_once() == 2
    assert attempted == ["first", "first", "second"]
    status = notification_status(tmp_path, config_file)
    assert status["cursor"][1] == "second" and status["sent_count"] == 2
    assert status["last_error"] is None and status["next_attempt_at"] is None
    assert status["retrying_run_id"] is None


def test_email_off_advances_past_retry_and_delivers_the_next_job(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config" / "notifications.json"
    configure(settings, config_file, NOW)
    silent = spec.model_copy(update={"metadata": {**spec.metadata, "gpuops.email": "off"}})
    _cancel(queue, silent, "silent", at=NOW + SECOND)
    _cancel(queue, spec, "next", at=NOW + 2 * SECOND)
    service._write_json(
        tmp_path / "notifications.json",
        service._NotificationState(
            enabled_since=NOW,
            cursor=(NOW, ""),
            retrying_run_id="silent",
            last_error="previous failure",
            next_attempt_at=NOW + timedelta(minutes=1),
        ),
    )
    attempted = []
    notifier = JobNotifier(
        queue,
        "gpu01",
        tmp_path,
        config_path=config_file,
        clock=lambda: NOW + 3 * SECOND,
        sender=lambda _, message: attempted.append(_run_id(message)),
    )

    assert notifier.check_once() == 1
    assert attempted == ["next"]
    status = notification_status(tmp_path, config_file)
    assert status["cursor"][1] == "next"
    assert status["sent_count"] == 1
    assert status["retrying_run_id"] is None
    assert status["last_error"] is None
    assert status["next_attempt_at"] is None


def test_failed_state_write_does_not_resend_an_accepted_mail(
    tmp_path,
    queue,
    spec,
    settings,
    monkeypatch,
):
    config_file = tmp_path / "config.json"
    configure(settings, config_file, NOW)
    _cancel(queue, spec, "first", at=NOW + SECOND)
    _cancel(queue, spec, "second", at=NOW + 2 * SECOND)
    cursor = notification_status(tmp_path, config_file)["cursor"]
    attempted = []
    writes = []
    disk_full = True
    write = service._write_json

    def write_state(path, state):
        writes.append(state.cursor[1])
        if disk_full:
            raise OSError(errno.ENOSPC, "No space left on device")
        write(path, state)

    def send(_, message):
        run_id = _run_id(message)
        if run_id == "second":
            status = notification_status(tmp_path, config_file)
            assert status["cursor"][1] == "first" and status["sent_count"] == 1
        attempted.append(run_id)

    monkeypatch.setattr(service, "_write_json", write_state)
    notifier = JobNotifier(queue, "gpu01", tmp_path, config_path=config_file, sender=send)
    for _ in range(3):
        notifier.check_once()
    assert attempted == ["first"]
    assert writes == ["first", "first", "first"]
    assert notification_status(tmp_path, config_file)["cursor"] == cursor

    disk_full = False
    assert notifier.check_once() == 1
    assert attempted == ["first", "second"]
    assert writes[-2:] == ["first", "second"]
    status = notification_status(tmp_path, config_file)
    assert status["cursor"][1] == "second" and status["sent_count"] == 2


def test_delivery_pages_in_finish_order_with_ties(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config" / "notifications.json"
    configure(settings, config_file, NOW)
    run_ids = [f"run-{index:02d}" for index in range(23)]
    for run_id in reversed(run_ids):
        _cancel(queue, spec, run_id, at=NOW + SECOND)
    attempted = []
    notifier = JobNotifier(
        queue,
        "gpu01",
        tmp_path,
        config_path=config_file,
        sender=lambda _, message: attempted.append(_run_id(message)),
    )
    assert notifier.check_once() == 20
    assert notifier.check_once() == 3
    assert notifier.check_once() == 0
    assert attempted == run_ids


def test_reconfigure_keeps_start_and_reenable_skips_history(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config" / "notifications.json"
    assert not load_config(config_file).enabled
    assert not disable(config_file).enabled and not config_file.exists()
    configure(settings, config_file, NOW)
    updated = settings.model_copy(update={"password": "new-password"})
    assert configure(updated, config_file, NOW + SECOND).enabled_since == NOW
    assert load_config(config_file).smtp == updated
    assert disable(config_file).smtp == updated
    _cancel(queue, spec, "disabled", at=NOW + 2 * SECOND)
    attempted = []
    notifier = JobNotifier(
        queue,
        "gpu01",
        tmp_path,
        config_path=config_file,
        sender=lambda _, message: attempted.append(_run_id(message)),
    )
    assert notifier.check_once() == 0
    assert configure(updated, config_file, NOW + 3 * SECOND).enabled_since == NOW + 3 * SECOND
    _cancel(queue, spec, "reenabled", at=NOW + 4 * SECOND)
    assert notifier.check_once() == 1
    assert attempted == ["reenabled"]


def test_disabling_during_a_send_stops_the_rest_of_the_batch(tmp_path, queue, spec, settings):
    config_file = tmp_path / "config.json"
    configure(settings, config_file, NOW)
    _cancel(queue, spec, "first", at=NOW + SECOND)
    _cancel(queue, spec, "second", at=NOW + 2 * SECOND)
    attempted = []

    def send(_, message):
        attempted.append(_run_id(message))
        disable(config_file)

    notifier = JobNotifier(queue, "gpu01", tmp_path, config_path=config_file, sender=send)
    assert notifier.check_once() == 1
    assert attempted == ["first"]
    assert notifier.check_once() == 0


def test_background_recovers_from_config_and_smtp_errors(
    tmp_path,
    queue,
    spec,
    settings,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(service, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(service, "_RETRY_DELAY", timedelta(0))
    config_file = tmp_path / "config.json"
    config_file.write_text(f'{{"smtp": {{"password": "{PASSWORD}"}}}}')
    _cancel(queue, spec, "run", at=NOW + SECOND)
    checked, sent = Event(), Event()
    attempted = []

    def send(_, message):
        attempted.append(_run_id(message))
        if len(attempted) == 1:
            raise TimeoutError(PASSWORD)
        sent.set()

    notifier = JobNotifier(queue, "gpu01", tmp_path, config_path=config_file, sender=send)
    check_once = notifier.check_once

    def observed_check():
        check_once()
        checked.set()

    monkeypatch.setattr(notifier, "check_once", observed_check)
    notifier.start()
    try:
        assert checked.wait(2)
        status = notification_status(tmp_path, config_file)
        assert status["config_error"] and PASSWORD not in status["config_error"]
        assert attempted == []
        configure(settings, config_file, NOW)
        assert sent.wait(2)
    finally:
        notifier.close()
    assert attempted == ["run", "run"]
    assert notification_status(tmp_path, config_file)["sent_count"] == 1
    assert PASSWORD not in caplog.text


def test_blocked_smtp_does_not_hold_queue_lock_and_close_stops_next_mail(
    tmp_path,
    queue,
    spec,
    settings,
):
    config_file = tmp_path / "config.json"
    configure(settings, config_file, NOW)
    _cancel(queue, spec, "first", at=NOW + SECOND)
    _cancel(queue, spec, "second", at=NOW + 2 * SECOND)
    entered, release, queue_changed = Event(), Event(), Event()
    attempted = []

    def send(_, message):
        attempted.append(_run_id(message))
        entered.set()
        assert release.wait(5)

    def enqueue():
        queue.enqueue(spec, run_id="while-sending", now=NOW + 3 * SECOND)
        queue_changed.set()

    notifier = JobNotifier(queue, "gpu01", tmp_path, config_path=config_file, sender=send)
    queue_thread = Thread(target=enqueue, daemon=True)
    close_thread = Thread(target=notifier.close, daemon=True)
    notifier.start()
    try:
        assert entered.wait(2)
        queue_thread.start()
        assert queue_changed.wait(2)
        close_thread.start()
        assert notifier._stop.wait(2)
    finally:
        release.set()
        notifier.close()
    assert attempted == ["first"]
    assert notification_status(tmp_path, config_file)["cursor"][1] == "first"


@pytest.mark.parametrize(("security", "port"), [("ssl", 465), ("starttls", 587)])
def test_mail_uses_verified_tls_and_one_envelope_recipient(settings, security, port, monkeypatch):
    client = Mock()
    connect = Mock(return_value=client)
    monkeypatch.setattr(mail.smtplib, "SMTP_SSL" if security == "ssl" else "SMTP", connect)
    settings = settings.model_copy(update={"security": security, "port": port})
    config = mail.NotificationConfig(smtp=settings)
    send_test(config, "gpu01")
    assert connect.call_args.args == (settings.host, port)
    assert connect.call_args.kwargs["timeout"] == 10
    if security == "ssl":
        context = connect.call_args.kwargs["context"]
        client.starttls.assert_not_called()
    else:
        context = client.starttls.call_args.kwargs["context"]
        assert [call[0] for call in client.mock_calls] == [
            "starttls",
            "login",
            "send_message",
            "close",
        ]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    client.login.assert_called_once_with(USERNAME, PASSWORD)
    message = client.send_message.call_args.args[0]
    assert list(message["To"].addresses)[0].addr_spec == settings.recipient
    assert len(message["To"].addresses) == 1
    assert "Test notification (gpu01)" in message["Subject"]
    client.close.assert_called_once()

    client.login.side_effect = smtplib.SMTPAuthenticationError(535, PASSWORD.encode())
    with pytest.raises(MailDeliveryError, match="SMTP authentication failed") as raised:
        send_test(config, "gpu01")
    assert PASSWORD not in str(raised.value)


def test_smtp_settings_reject_multiple_recipients_without_echoing_credentials(settings):
    with pytest.raises(ValidationError, match="single email address") as raised:
        SMTPSettings.model_validate(
            {
                **settings.model_dump(),
                "recipient": "one@example.com, two@example.com",
            }
        )
    assert PASSWORD not in str(raised.value) and PASSWORD not in repr(settings)
