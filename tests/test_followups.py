import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gpuops import followup_commands
from gpuops.cli import main
from gpuops.followups import CodexQueue, Followup, FollowupStore, check_followups
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now


@pytest.fixture
def followup(tmp_path):
    return Followup(
        run_id="experiment-1",
        server_id="gpu01",
        ssh_host="lab",
        thread_id="original-chat",
        workspace=str(tmp_path),
        action="Read metrics.json, compare the baseline and report in this chat.",
    )


def job(status):
    return JobRecord(
        run_id="experiment-1",
        spec=JobSpec(
            id="experiment",
            project="research",
            command=["python", "train.py"],
            working_directory="/data/research",
        ),
        status=status,
        finished_at=utc_now() if status.terminal else None,
    )


@pytest.mark.parametrize("terminal", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED])
def test_completion_survives_offline_restart_and_is_delivered_once(tmp_path, followup, terminal):
    store = FollowupStore(tmp_path / "records")
    store.add(followup)
    transport = Mock(
        get_job=AsyncMock(side_effect=[OSError("offline"), job(JobStatus.QUEUED), job(terminal)])
    )
    delivery = Mock(send=Mock(return_value=subprocess.CompletedProcess([], 0, "queue receipt", "")))
    for _ in range(4):
        asyncio.run(check_followups(FollowupStore(store.directory), transport, delivery))
    saved = store.get(followup.id)
    assert transport.get_job.call_count == 3
    assert delivery.send.call_count == 1
    assert saved.status == "queued" and saved.received_at is None
    assert saved.job.status == terminal and saved.error is None
    assert followup.id in saved.message and followup.action in saved.message
    assert saved.receipt == "queue receipt"


def test_unavailable_cli_retries_persisted_message_without_requerying_job(tmp_path, followup):
    store = FollowupStore(tmp_path / "records")
    store.add(followup)
    transport = Mock(get_job=AsyncMock(return_value=job(JobStatus.SUCCEEDED)))
    delivery = Mock(
        send=Mock(
            side_effect=[
                FileNotFoundError("Codex offline"),
                subprocess.CompletedProcess([], 0, "accepted", ""),
            ]
        )
    )
    asyncio.run(check_followups(store, transport, delivery))
    first = store.get(followup.id)
    assert first.status == "ready" and first.message is not None
    asyncio.run(check_followups(store, transport, delivery))
    assert store.get(followup.id).message == first.message
    assert transport.get_job.call_count == 1 and delivery.send.call_count == 2


@pytest.mark.parametrize("interrupted", [False, True])
def test_uncertain_delivery_does_not_blindly_repeat(tmp_path, followup, interrupted):
    store = FollowupStore(tmp_path / "records")
    if interrupted:
        followup.status = "sending"
    store.add(followup)
    transport = Mock(get_job=AsyncMock(return_value=job(JobStatus.SUCCEEDED)))
    delivery = Mock(send=Mock(side_effect=subprocess.TimeoutExpired("codex", 30)))
    for _ in range(2):
        asyncio.run(check_followups(store, transport, delivery))
    assert store.get(followup.id).status == "delivery_unknown"
    assert delivery.send.call_count == (0 if interrupted else 1)


def test_registration_and_original_chat_receipt_are_distinct_from_analysis(
    tmp_path, followup, monkeypatch
):
    store = FollowupStore(tmp_path / "records")
    store.add(followup)
    assert store.add(followup.model_copy(update={"id": "another-id"})).id == followup.id
    with pytest.raises(ValueError, match="different follow-up"):
        store.add(followup.model_copy(update={"action": "different"}))
    followup.status = "queued"
    store.save(followup)
    monkeypatch.setenv("CODEX_THREAD_ID", "other-chat")
    with pytest.raises(ValueError, match="original Codex"):
        store.acknowledge(followup.id)
    monkeypatch.setenv("CODEX_THREAD_ID", followup.thread_id)
    received = store.acknowledge(followup.id)
    assert received.status == "received" and received.completed_at is None
    completed = store.acknowledge(followup.id, summary="Loss decreased.")
    assert completed.status == "completed" and completed.summary == "Loss decreased."
    assert store.acknowledge(followup.id) == completed


def test_queue_targets_exact_chat_and_home_without_starting_a_model(followup, monkeypatch):
    followup.message = "completion message"
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, "accepted", ""))
    monkeypatch.setattr(subprocess, "run", runner)
    CodexQueue("/tools/codex", "/codex-home").send(followup)
    args, kwargs = runner.call_args
    assert args[0] == [
        "/tools/codex",
        "queue",
        "--thread",
        "original-chat",
        "--message",
        "completion message",
    ]
    assert kwargs["cwd"] == followup.workspace
    assert kwargs["env"]["CODEX_HOME"] == "/codex-home"


def test_cli_stops_only_unsubmitted_followup_not_experiment(
    tmp_path, monkeypatch, capsys, followup
):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = FollowupStore(tmp_path / ".local/state/gpuops/followups/records")
    store.add(followup)
    assert main(["followup", "stop", followup.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stopped"
    followup.status = "queued"
    store.save(followup)
    assert main(["followup", "stop", followup.id, "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert store.get(followup.id).status == "queued"


def test_cli_registration_uses_original_chat_and_journal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    thread_id = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setenv("CODEX_THREAD_ID", thread_id)
    monkeypatch.chdir(tmp_path)
    setup = Mock()
    monkeypatch.setattr(followup_commands, "enable", setup)
    monkeypatch.setattr(
        followup_commands,
        "SubmissionJournal",
        lambda _: Mock(get=Mock(return_value=SimpleNamespace(server_id="gpu01", ssh_host="lab"))),
    )
    args = ["followup", "add", "experiment-1", "--action", "Read the saved metrics", "--json"]
    assert main(args + ["--server", "gpu02"]) == 1
    assert "conflicts" in json.loads(capsys.readouterr().out)["error"]
    setup.assert_not_called()
    assert main(args) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["thread_id"] == thread_id and row["workspace"] == str(tmp_path.resolve())
    assert row["ssh_host"] == "lab" and row["status"] == "waiting"
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["id"] == row["id"]


def test_unknown_delivery_can_be_explicitly_retried_from_original_chat(
    tmp_path, monkeypatch, capsys, followup
):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = FollowupStore(tmp_path / ".local/state/gpuops/followups/records")
    followup.status = "delivery_unknown"
    store.add(followup)
    monkeypatch.setenv("CODEX_THREAD_ID", "other-chat")
    assert main(["followup", "retry", followup.id, "--json"]) == 1
    capsys.readouterr()
    assert store.get(followup.id).status == "delivery_unknown"
    monkeypatch.setenv("CODEX_THREAD_ID", followup.thread_id)
    assert main(["followup", "retry", followup.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
