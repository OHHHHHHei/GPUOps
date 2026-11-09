import json
import stat
from pathlib import Path

import pytest

from gpuops import alerts
from gpuops.cli import main
from gpuops.config import ServerTarget
from gpuops.notifications.config import SMTPSettings
from gpuops.notifications.mail import MailDeliveryError


@pytest.fixture(autouse=True)
def isolated_email(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "_email_path", lambda: tmp_path / "config" / "email.json")

    def unexpected_mail(*args):
        pytest.fail("tests must explicitly stub email delivery")

    monkeypatch.setattr(alerts, "send_mail", unexpected_mail)


@pytest.fixture
def settings():
    return SMTPSettings(
        host="smtp.example.com",
        sender="sender@example.com",
        recipient="reader@example.com",
        username="test-user",
        password="synthetic-secret",
    )


@pytest.fixture
def fleet(tmp_path, monkeypatch, settings):
    alerts._email_path().parent.mkdir()
    alerts._email_path().write_text(settings.model_dump_json())
    desktop, emails = [], []
    row = {"server_id": "gpu01", "online": True, "result": {"ready": True}}

    async def collect(_transport, targets, action):
        assert [target.server_id for target in targets] == ["gpu01", "gpu02"]
        assert action == "status"
        return [row, {"server_id": "gpu02", "online": True, "result": {"ready": True}}]

    monkeypatch.setattr(
        alerts,
        "select_targets",
        lambda: [
            ServerTarget("gpu01", "Server 1", "one"),
            ServerTarget("gpu02", "Server 2", "two"),
        ],
    )
    monkeypatch.setattr(alerts, "fleet_worker_action", collect)
    monkeypatch.setattr(alerts, "_notify", lambda title, text: desktop.append((title, text)))
    monkeypatch.setattr(alerts, "send_mail", lambda _settings, message: emails.append(message))

    def poll(status):
        row["online"] = status != "unknown"
        row["result"] = {
            "status": "running" if status in {"ready", "not_ready"} else status,
            "ready": status == "ready",
            "heartbeat": {"heartbeat_at": "2026-11-09T00:00:00Z", "last_tick_error": "disk full"},
        }
        return alerts._check(tmp_path / "alerts.json")

    return poll, desktop, emails


@pytest.mark.parametrize(
    "email_enabled,stopped", [(False, "stopped"), (True, "stopped"), (True, "stale")]
)
def test_stop_and_recovery_are_persisted_and_deduplicated(fleet, tmp_path, email_enabled, stopped):
    poll, desktop, emails = fleet
    if not email_enabled:
        alerts._email_path().unlink()
    for status in ["ready", "unknown", stopped, stopped, "unknown", "not_ready", "ready", "ready"]:
        state = poll(status)
        assert [row["status"] for row in state["servers"]] == [
            "stopped" if status == "stale" else status,
            "ready",
        ]
    assert len(desktop) == 2
    assert desktop[0][0] == desktop[1][0] == "GPUOps: Server 1"
    assert "disk full" in desktop[0][1] and "Last heartbeat:" in desktop[0][1]
    assert "recovered" in desktop[1][1]
    assert len(emails) == (2 if email_enabled else 0)
    if email_enabled:
        assert "Worker stopped: Server 1 (gpu01)" in emails[0]["Subject"]
        assert "Worker recovered: Server 1 (gpu01)" in emails[1]["Subject"]
        body = emails[0].get_content()
        assert "Detected at:" in body and "2026-11-09T00:00:00Z" in body
        assert "disk full" in body and "does not imply" in body
    assert state["servers"][0]["email_stop_notified"] is False
    assert stat.S_IMODE((tmp_path / "alerts.json").stat().st_mode) == 0o600


@pytest.mark.parametrize("channel", ["desktop", "email"])
@pytest.mark.parametrize("status", ["stopped", "ready"])
def test_failed_channel_retries_without_repeating_successful_channel(
    fleet, monkeypatch, channel, status
):
    poll, desktop, emails = fleet
    if status == "ready":
        poll("stopped")
    desktop.clear()
    emails.clear()
    attempts = []
    field = "notification_error" if channel == "desktop" else "email_error"
    function = "_notify" if channel == "desktop" else "send_mail"
    original = getattr(alerts, function)

    def fail_once(*args):
        attempts.append(True)
        if len(attempts) == 1:
            if channel == "email":
                raise MailDeliveryError("SMTP delivery timed out")
            raise RuntimeError("desktop delivery failed")
        original(*args)

    monkeypatch.setattr(alerts, function, fail_once)
    first = poll(status)["servers"][0]
    assert field in first
    assert len(desktop) == (channel == "email")
    assert len(emails) == (channel == "desktop")
    second = poll(status)["servers"][0]
    poll(status)
    assert field not in second
    assert len(attempts) == 2
    assert len(desktop) == len(emails) == 1


def test_recovery_does_not_send_an_expired_failed_stop_email(fleet, monkeypatch):
    poll, desktop, emails = fleet
    attempts = []

    def fail(*args):
        attempts.append(True)
        raise MailDeliveryError("SMTP delivery timed out")

    monkeypatch.setattr(alerts, "send_mail", fail)
    poll("stopped")
    poll("unknown")
    recovered = poll("ready")["servers"][0]
    assert len(attempts) == 1
    assert not emails and len(desktop) == 2
    assert recovered["email_stop_notified"] is False
    assert "email_error" not in recovered


def test_legacy_desktop_state_does_not_suppress_new_email_channel(fleet, tmp_path):
    poll, desktop, emails = fleet
    path = tmp_path / "alerts.json"
    path.write_text(json.dumps({"servers": [{"server_id": "gpu01", "stop_notified": True}]}))
    poll("stopped")
    assert not desktop and len(emails) == 1


def test_invalid_email_config_does_not_block_desktop(fleet):
    poll, desktop, emails = fleet
    alerts._email_path().write_text('{"password":"synthetic-secret"}')
    state = poll("stopped")
    assert len(desktop) == 1 and not emails
    assert state["email_error"] == "invalid Worker alert SMTP settings"
    assert "synthetic-secret" not in json.dumps(state)


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(alerts.sys, "platform", "darwin")
    monkeypatch.setattr(
        alerts, "select_targets", lambda: [ServerTarget("gpu01", "Server 1", "one")]
    )
    monkeypatch.setattr(alerts, "_notify", lambda *_: None)
    calls = []

    def launchctl(*args, **kwargs):
        calls.append(args)
        return args[0] != "print"

    monkeypatch.setattr(alerts, "_launchctl", launchctl)
    return calls


def test_email_configuration_is_private_and_can_be_disabled_separately(
    tmp_path, local_app, settings, capsys
):
    source = tmp_path / "smtp.json"
    source.write_text(settings.model_dump_json())
    assert main(["alerts", "enable", "--email-config", str(source), "--json"]) == 0
    output = capsys.readouterr().out
    assert "synthetic-secret" not in output and "test-user" not in output
    assert json.loads(output)["email_recipient"] == settings.recipient
    assert SMTPSettings.model_validate_json(alerts._email_path().read_bytes()) == settings
    assert stat.S_IMODE(alerts._email_path().stat().st_mode) == 0o600
    assert any(call[0] == "bootstrap" for call in local_app)
    path = tmp_path / ".local/state/gpuops/alerts.json"
    alerts._save_state(
        path,
        {
            "checked_at": None,
            "servers": [
                {
                    "server_id": "gpu01",
                    "stop_notified": True,
                    "email_stop_notified": True,
                    "email_error": "previous failure",
                }
            ],
        },
    )
    assert main(["alerts", "enable", "--no-email", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["enabled"] and not payload["email_configured"]
    assert not alerts._email_path().exists()
    assert payload["servers"][0]["stop_notified"] is True
    assert payload["servers"][0]["email_stop_notified"] is False
    assert "email_error" not in payload["servers"][0]


def test_invalid_email_input_does_not_replace_saved_settings(tmp_path, local_app, settings, capsys):
    destination = alerts._email_path()
    destination.parent.mkdir()
    saved = settings.model_dump_json()
    destination.write_text(saved)
    source = tmp_path / "invalid.json"
    source.write_text('{"password":"synthetic-secret"}')
    assert main(["alerts", "enable", "--email-config", str(source), "--json"]) == 1
    output = capsys.readouterr().out
    assert "synthetic-secret" not in output
    assert json.loads(output)["ok"] is False
    assert destination.read_text() == saved
    assert not local_app


@pytest.mark.parametrize(
    "action,option",
    [
        ("test", ["--email-config", "smtp.json"]),
        ("status", ["--no-email"]),
        ("disable", ["--email-config", "smtp.json"]),
    ],
)
def test_email_options_only_apply_to_enable(action, option, capsys):
    assert main(["alerts", action, *option, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_argument"


@pytest.mark.parametrize("failed", ["desktop", "email"])
def test_test_command_attempts_both_channels(local_app, settings, monkeypatch, capsys, failed):
    alerts._email_path().parent.mkdir()
    alerts._email_path().write_text(settings.model_dump_json())
    attempts = []

    def desktop(*args):
        attempts.append("desktop")
        if failed == "desktop":
            raise RuntimeError("desktop unavailable")

    def email(_settings, message):
        attempts.append("email")
        assert message["Subject"] == "[GPUOps] Worker alert test (Mac)"
        assert "no Worker or experiment was stopped" in message.get_content()
        if failed == "email":
            raise MailDeliveryError("SMTP delivery timed out")

    monkeypatch.setattr(alerts, "_notify", desktop)
    monkeypatch.setattr(alerts, "send_mail", email)
    assert main(["alerts", "test", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["notification_submitted"] is (failed != "desktop")
    assert payload["email_accepted"] is (failed != "email")
    assert attempts == ["desktop", "email"]


def test_status_reports_fixed_config_without_waiting_for_next_poll(
    local_app, settings, capsys, tmp_path
):
    alerts._email_path().parent.mkdir()
    alerts._email_path().write_text(settings.model_dump_json())
    alerts._save_state(
        tmp_path / ".local/state/gpuops/alerts.json",
        {
            "checked_at": None,
            "servers": [],
            "email_error": "previous config error",
        },
    )
    assert main(["alerts", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["email_error"] is None
