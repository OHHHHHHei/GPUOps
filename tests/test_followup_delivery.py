from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpuops.followup_delivery import (
    AdapterSettings,
    DeliveryRouter,
    DeliveryUncertain,
    adapter_capabilities,
    configure_adapter,
    configure_email,
    read_email_settings,
    send_followup_email,
)
from gpuops.followups import Followup
from gpuops.notifications.config import SMTPSettings


@pytest.fixture
def followup(tmp_path: Path) -> Followup:
    return Followup(
        id="00000000-0000-4000-8000-000000000001",
        run_id="experiment-1",
        server_id="gpu01",
        ssh_host="lab",
        harness="test-harness",
        thread_id="original-session",
        workspace=str(tmp_path),
        action="Compare the experiments.",
        group_id="group-1",
        mode="all_terminal",
        context={"question": "Which run won?"},
        outcomes=[{"run_id": "experiment-1", "status": "succeeded", "metric": 0.9}],
        message="Analyze saved metrics.",
    )


@pytest.fixture
def smtp() -> SMTPSettings:
    return SMTPSettings(
        host="smtp.example.com",
        sender="gpuops@example.com",
        recipient="researcher@example.com",
        username="private-user",
        password="private-password",
    )


def _install_adapter(tmp_path: Path, monkeypatch, *, idempotent: bool = False) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "adapter.json"
    source.write_text(
        AdapterSettings(
            command=["/tools/adapter", "--receive", ""], idempotent=idempotent
        ).model_dump_json()
    )
    configure_adapter("test-harness", source)


def test_adapter_ack_contract_and_duplicate_event_id(tmp_path, monkeypatch, followup):
    _install_adapter(tmp_path, monkeypatch, idempotent=True)
    requests = []

    def run(command, **kwargs):
        assert command == ["/tools/adapter", "--receive", ""]
        assert kwargs["cwd"] == followup.workspace
        assert kwargs["shell"] is False
        request = json.loads(kwargs["input"])
        requests.append(request)
        response = {
            "api_version": "gpuops.delivery/v1",
            "event_id": request["event_id"],
            "accepted": True,
            "receipt": "durable-receipt",
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(subprocess, "run", run)
    router = DeliveryRouter(None)

    first = router.send(followup)
    second = router.send(followup)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout == "durable-receipt"
    assert [request["event_id"] for request in requests] == [followup.id, followup.id]
    assert requests[0] == {
        "api_version": "gpuops.completion/v1",
        "event_id": followup.id,
        "group_id": "group-1",
        "target": {
            "harness": "test-harness",
            "session_id": "original-session",
            "workspace": followup.workspace,
        },
        "mode": "all_terminal",
        "trigger_job": None,
        "action": "Compare the experiments.",
        "context": {"question": "Which run won?"},
        "jobs": followup.outcomes,
        "message": "Analyze saved metrics.",
    }
    assert router.can_retry_uncertain(followup)
    followup.mode = "per_job"
    router.send(followup)
    assert requests[-1]["trigger_job"] == {"server_id": "gpu01", "run_id": "experiment-1"}
    assert adapter_capabilities() == [
        {
            "name": "codex-desktop",
            "idempotent": False,
            "builtin": True,
            "session_continuity": "native",
        },
        {
            "name": "test-harness",
            "idempotent": True,
            "builtin": False,
            "session_continuity": "unverified",
        },
    ]
    assert (
        stat.S_IMODE((tmp_path / ".config/gpuops/followup-adapters.json").stat().st_mode) == 0o600
    )


def test_adapter_rejection_and_uncertain_delivery(tmp_path, monkeypatch, followup):
    _install_adapter(tmp_path, monkeypatch)
    router = DeliveryRouter(None)

    def result(payload, returncode=0):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda command, **kwargs: subprocess.CompletedProcess(
                command,
                returncode,
                json.dumps(payload) if not isinstance(payload, str) else payload,
                "",
            ),
        )

    result(
        {
            "api_version": "gpuops.delivery/v1",
            "event_id": followup.id,
            "accepted": False,
            "error": "receiver is full",
        }
    )
    rejected = router.send(followup)
    assert rejected.returncode == 1 and rejected.stderr == "receiver is full"

    for payload, returncode in [
        ("not json", 0),
        (
            {
                "api_version": "gpuops.delivery/v1",
                "event_id": "wrong-event",
                "accepted": True,
                "receipt": "receipt",
            },
            0,
        ),
        ("", 7),
    ]:
        result(payload, returncode)
        with pytest.raises(DeliveryUncertain):
            router.send(followup)
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("adapter", 30))
    )
    with pytest.raises(DeliveryUncertain, match="timed out"):
        router.send(followup)
    assert not router.can_retry_uncertain(followup)


def test_email_configuration_fallback_and_both_message_types(tmp_path, monkeypatch, followup, smtp):
    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "smtp.json"
    source.write_text(smtp.model_dump_json())
    fallback = tmp_path / ".config/gpuops/worker-alerts-email.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(smtp.model_dump_json())
    assert read_email_settings() == smtp

    configure_email(source)
    destination = tmp_path / ".config/gpuops/followup-email.json"
    assert read_email_settings() == smtp
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600

    sent = []
    monkeypatch.setattr(
        "gpuops.followup_delivery.send_mail",
        lambda settings, message: sent.append(message),
    )
    followup.notify = "terminal"
    send_followup_email(followup)
    followup.notify = "analysis_complete"
    followup.summary = "The grouped run improved accuracy."
    send_followup_email(followup)

    terminal, analysis = sent
    assert terminal["Subject"] == "[GPUOps] Experiments finished"
    assert "experiment-1" in terminal.get_content() and '"metric": 0.9' in terminal.get_content()
    assert analysis["Subject"] == "[GPUOps] Analysis complete"
    assert "The grouped run improved accuracy." in analysis.get_content()
    expected_message_id = f"<{followup.id}@gpuops.local>"
    assert terminal["Message-ID"] == analysis["Message-ID"] == expected_message_id


def test_email_without_configuration_is_readable(tmp_path, monkeypatch, followup):
    monkeypatch.setenv("HOME", str(tmp_path))
    followup.notify = "terminal"
    with pytest.raises(RuntimeError, match="not configured"):
        send_followup_email(followup)
