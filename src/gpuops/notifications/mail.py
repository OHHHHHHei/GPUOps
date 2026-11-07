from __future__ import annotations

import shlex
import smtplib
import ssl
from datetime import timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid

from gpuops.models import JobRecord, utc_now
from gpuops.notifications.config import NotificationConfig, NotificationConfigError, SMTPSettings

SMTP_TIMEOUT_SECONDS = 10


class MailDeliveryError(RuntimeError):
    pass


def _message(settings: SMTPSettings, subject: str, body: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = settings.recipient
    message["Date"] = format_datetime(utc_now())
    message["Message-ID"] = make_msgid(domain="gpuops.local")
    message["Subject"] = subject
    message.set_content(body)
    return message


def job_message(settings: SMTPSettings, server_id: str, record: JobRecord) -> EmailMessage:
    finished = record.finished_at
    if finished is None:
        raise ValueError("cannot notify about an unfinished job")
    server = " ".join(server_id.split())
    name = " ".join((record.spec.metadata.get("name") or record.spec.id).split())[:120]
    project = " ".join(record.spec.project.split())
    status = record.status.value
    details = [
        f"Server: {server}",
        f"Project: {project}",
        f"Job: {name}",
        f"Run ID: {record.run_id}",
        f"Status: {status}",
        f"Started: {record.launch.started_at.astimezone(timezone.utc).isoformat()}"
        if record.launch
        else "Started: not started",
        f"Finished: {finished.astimezone(timezone.utc).isoformat()}",
    ]
    if record.launch is not None:
        duration = max(0, int((finished - record.launch.started_at).total_seconds()))
        details.append(f"Duration: {timedelta(seconds=duration)}")
    reason = record.launch_error or (record.failure_reason if record.exit_code is None else None)
    if record.exit_code is not None:
        details.append(f"Exit code: {record.exit_code}")
    if reason and reason.strip():
        details.append(f"Reason: {reason.strip().splitlines()[0][:180]}")
    logs_command = shlex.join(["gpuops", "logs", "--server", server_id, "--", record.run_id])
    return _message(
        settings,
        f"[GPUOps] {status}: {name} ({server})",
        "\n".join([*details, "", f"View logs: {logs_command}"]),
    )


def send_mail(settings: SMTPSettings, message: EmailMessage) -> None:
    try:
        context = ssl.create_default_context()
        client: smtplib.SMTP
        if settings.security == "ssl":
            client = smtplib.SMTP_SSL(
                settings.host, settings.port, timeout=SMTP_TIMEOUT_SECONDS, context=context
            )
        else:
            client = smtplib.SMTP(settings.host, settings.port, timeout=SMTP_TIMEOUT_SECONDS)
        try:
            if settings.security == "starttls":
                client.starttls(context=context)
            if settings.username is not None:
                assert settings.password is not None
                client.login(settings.username, settings.password)
            client.send_message(message)
        finally:
            client.close()
    except (OSError, ValueError) as error:
        raise MailDeliveryError(delivery_error_message(error)) from None


def delivery_error_message(error: Exception) -> str:
    if isinstance(error, MailDeliveryError):
        return str(error)
    if isinstance(error, smtplib.SMTPAuthenticationError):
        return f"SMTP authentication failed (status {error.smtp_code})"
    if isinstance(error, smtplib.SMTPResponseException):
        return f"SMTP server rejected the request (status {error.smtp_code})"
    if isinstance(error, smtplib.SMTPRecipientsRefused):
        return "SMTP server refused the recipient"
    if isinstance(error, ssl.SSLCertVerificationError):
        return "SMTP TLS certificate verification failed"
    if isinstance(error, TimeoutError):
        return "SMTP connection or delivery timed out"
    return f"SMTP delivery failed ({type(error).__name__})"


def send_test(config: NotificationConfig, server_id: str) -> None:
    if config.smtp is None:
        raise NotificationConfigError("email notifications are not configured")
    settings = config.smtp
    server = " ".join(server_id.split())
    message = _message(
        settings,
        f"[GPUOps] Test notification ({server})",
        f"This is a test of GPUOps job notifications from {server}.",
    )
    send_mail(settings, message)
