"""Public registration, delivery, receipt and completion across module boundaries."""

import json
import subprocess
import sys
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from gpuops.cli import main
from gpuops.followups import FollowupStore
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now


@pytest.fixture
def client(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_THREAD_ID", "00000000-0000-4000-8000-000000000001")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("gpuops.followup_commands.prepare_delivery", Mock())
    monkeypatch.setattr("gpuops.background._launchctl", Mock(return_value=False))
    smtp = tmp_path / "smtp.json"
    smtp.write_text(
        json.dumps(
            {
                "host": "smtp.example.com",
                "sender": "gpuops@example.com",
                "recipient": "researcher@example.com",
            }
        )
    )

    def command(*args, code=0):
        assert main(["followup", *args, "--json"]) == code
        return json.loads(capsys.readouterr().out)

    command("configure", "--email-config", str(smtp))
    return command


def job(run_id, status):
    return JobRecord(
        run_id=run_id,
        spec=JobSpec(id=run_id, project="tests", command=["true"], working_directory="/research"),
        status=status,
        finished_at=utc_now() if status.terminal else None,
    )


def test_cross_server_group_through_external_adapter_and_post_analysis_email(
    client, tmp_path, monkeypatch
):
    # A real subprocess spools once, loses its first receipt, then acknowledges the same ID.
    adapter = tmp_path / "adapter.py"
    adapter.write_text("""import json, pathlib, sys
event = json.load(sys.stdin)
path = pathlib.Path(event["event_id"] + ".accepted.json")
if not path.exists():
    path.write_text(json.dumps(event))
    sys.exit(7)
print(json.dumps({"api_version": "gpuops.delivery/v1", "event_id": event["event_id"],
                  "accepted": True, "receipt": str(path)}))
""")
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps({"command": [sys.executable, str(adapter)], "idempotent": True}))
    client("adapter", "test-harness", str(config))
    assert client("capabilities")["codex_unloaded_session_wakeup"] is False
    assert "jobs" in client("schema")["schema"]["properties"]
    request = tmp_path / "followup.json"
    request.write_text(
        json.dumps(
            {
                "jobs": [
                    {"run_id": "a", "server_id": "gpu01"},
                    {"run_id": "b", "server_id": "gpu02"},
                    {"run_id": "c", "server_id": "gpu02"},
                ],
                "target": {"harness": "test-harness", "session_id": "original-session"},
                "action": "Compare saved results",
                "context": {"question": "Compare variants"},
                "notify": "analysis_complete",
            }
        )
    )
    registered = client("register", str(request))
    assert client("register", str(request))["id"] == registered["id"]
    event_id = registered["events"][0]["id"]
    remaining = {
        "a": iter([JobStatus.SUCCEEDED]),
        "b": iter([JobStatus.QUEUED, JobStatus.FAILED]),
        "c": iter([JobStatus.CANCELLED]),
    }

    async def batch(target, ids):
        return {
            "server_id": target.server_id,
            "worker": {},
            "jobs": {
                run_id: job(run_id, next(remaining[run_id])).model_dump(mode="json")
                for run_id in ids
            },
        }

    watch = AsyncMock(side_effect=batch)
    monkeypatch.setattr("gpuops.background.SshWorkerTransport", lambda **_: Mock(watch=watch))
    email = Mock(side_effect=[OSError("SMTP offline"), None])
    monkeypatch.setattr("gpuops.followup_delivery.send_mail", email)

    client("check")
    assert client("show", registered["id"])["all_terminal"] is False
    assert not list(tmp_path.glob("*.accepted.json"))
    client("check")
    assert client("show", event_id)["status"] == "delivery_unknown"
    client("check")
    queued = client("show", event_id)
    assert queued["status"] == "queued" and queued["received_at"] is None
    email.assert_not_called()
    [accepted] = tmp_path.glob("*.accepted.json")
    envelope = json.loads(accepted.read_text())
    assert envelope["event_id"] == event_id and envelope["trigger_job"] is None
    assert envelope["target"]["session_id"] == "original-session"
    assert [row["status"] for row in envelope["jobs"]] == ["succeeded", "failed", "cancelled"]
    assert envelope["context"] == {"question": "Compare variants"}
    assert client("acknowledge", event_id, "--session", "wrong", code=1)["ok"] is False
    assert client("acknowledge", event_id, "--session", "original-session")["status"] == "received"
    client("complete", event_id, "--session", "original-session", "--summary", " ", code=1)
    client(
        "complete",
        event_id,
        "--session",
        "original-session",
        "--summary",
        "A succeeded; B failed; C cancelled.",
    )
    client("check")
    failed_mail = client("show", event_id)
    assert failed_mail["status"] == "completed" and failed_mail["email_error"] == "SMTP offline"
    client("check")
    assert email.call_count == 1  # still inside the retry delay
    retry_at = utc_now() + timedelta(minutes=2)
    monkeypatch.setattr("gpuops.followups.utc_now", lambda: retry_at)
    client("check")
    assert client("show", event_id)["email_sent_at"] is not None
    assert client("show", registered["id"])["status"] == "analysis_complete"
    assert watch.await_count == 3 and email.call_count == 2
    assert "A succeeded; B failed; C cancelled." in email.call_args.args[1].get_content()


@pytest.mark.parametrize("resume", [True, False])
def test_terminal_email_failure_does_not_block_model_or_repeat_accepted_delivery(
    client, tmp_path, monkeypatch, resume
):
    request = tmp_path / "followup.json"
    request.write_text(
        json.dumps(
            {
                "jobs": [{"run_id": "a", "server_id": "gpu01"}],
                "action": "Read saved results",
                "resume": resume,
                "notify": "terminal",
            }
        )
    )
    group = client("register", str(request))
    event_id = group["events"][0]["id"]
    transport = Mock(
        watch=AsyncMock(
            return_value={
                "server_id": "gpu01",
                "worker": {},
                "jobs": {"a": job("a", JobStatus.SUCCEEDED).model_dump(mode="json")},
            }
        )
    )
    monkeypatch.setattr("gpuops.background.SshWorkerTransport", lambda **_: transport)
    delivery = Mock(send=Mock(return_value=subprocess.CompletedProcess([], 0, "accepted", "")))
    monkeypatch.setattr("gpuops.background.DeliveryRouter", lambda _: delivery)
    email = Mock(side_effect=[OSError("offline"), None])
    monkeypatch.setattr("gpuops.followup_delivery.send_mail", email)
    client("check")
    event = client("show", event_id)
    assert event["status"] == ("queued" if resume else "observed")
    assert event["email_error"] == "offline"
    assert delivery.send.call_count == int(resume)
    if resume:
        client("check")
        assert email.call_count == 1  # still inside the retry delay
        retry_at = utc_now() + timedelta(minutes=2)
        monkeypatch.setattr("gpuops.followups.utc_now", lambda: retry_at)
        client("check")
        assert email.call_count == 2
        assert client("show", event_id)["email_sent_at"] is not None
        assert delivery.send.call_count == 1
    else:
        # Stopping a human-only event suppresses its pending email retry.
        client("stop", event_id)
        client("check")
        assert email.call_count == 1
        assert client("show", event_id)["status"] == "stopped"
    assert transport.watch.await_count == 1
    store = FollowupStore(tmp_path / ".local/state/gpuops/followups/records")
    assert len(store.list()) == 1
