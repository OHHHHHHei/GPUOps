import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gpuops import background, followup_commands
from gpuops.cli import main
from gpuops.followups import (
    CodexQueue,
    Followup,
    FollowupStore,
    completion_message,
    job_outcome,
)
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now


@pytest.fixture(autouse=True)
def local_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(background, "_launchctl", Mock(return_value=False))


@pytest.fixture
def checker(monkeypatch):
    bg = background.Background()
    store = bg.event_store()
    with store.lock():
        bg.set_switch_locked("followups", True)

    def setup(delivery, responses):
        def watch(target, ids):
            assert ids == ["experiment-1"]
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return {
                "server_id": target.server_id,
                "worker": {},
                "jobs": {ids[0]: result.model_dump(mode="json")},
            }

        transport = Mock(watch=AsyncMock(side_effect=watch))
        monkeypatch.setattr(background, "SshWorkerTransport", lambda **_: transport)
        monkeypatch.setattr(background, "DeliveryRouter", lambda *_: delivery)
        return transport, lambda: asyncio.run(bg.run_round(followup_only=True))

    return store, setup


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
def test_completion_survives_offline_restart_and_is_delivered_once(checker, followup, terminal):
    store, setup = checker
    store.add(followup)
    delivery = Mock(send=Mock(return_value=subprocess.CompletedProcess([], 0, "queue receipt", "")))
    transport, check = setup(
        delivery, iter([OSError("offline"), job(JobStatus.QUEUED), job(terminal)])
    )
    for _ in range(4):
        check()
    saved = store.get(followup.id)
    assert transport.watch.call_count == 3
    assert delivery.send.call_count == 1
    assert saved.status == "queued" and saved.received_at is None
    assert saved.job.status == terminal and saved.error is None
    assert saved.outcomes == [
        {
            "server_id": "gpu01",
            "run_id": "experiment-1",
            "status": terminal.value,
            "exit_code": None,
            "failure_reason": None,
            "remote_workspace": "/data/research",
            "project_workspace": followup.workspace,
        }
    ]
    details = json.loads(saved.message.split("\n", 2)[2].split("\n\n", 1)[0])
    assert details["jobs"] == saved.outcomes and details["status"] == terminal.value
    assert followup.id in saved.message and followup.action in saved.message
    assert saved.receipt == "queue receipt"
    assert saved.outcomes[0]["status"] == terminal.value
    assert saved.outcomes[0]["project_workspace"] == followup.workspace


def test_unavailable_cli_retries_persisted_message_without_requerying_job(checker, followup):
    store, setup = checker
    store.add(followup)
    delivery = Mock(
        send=Mock(
            side_effect=[
                FileNotFoundError("Codex offline"),
                subprocess.CompletedProcess([], 0, "accepted", ""),
            ]
        )
    )
    transport, check = setup(delivery, iter([job(JobStatus.SUCCEEDED)]))
    check()
    first = store.get(followup.id)
    assert first.status == "ready" and first.message is not None
    check()
    assert store.get(followup.id).message == first.message
    assert transport.watch.call_count == 1 and delivery.send.call_count == 2


@pytest.mark.parametrize("interrupted", [False, True])
def test_uncertain_delivery_does_not_blindly_repeat(checker, followup, interrupted):
    store, setup = checker
    if interrupted:
        followup.status = "sending"
    store.add(followup)
    delivery = Mock(send=Mock(side_effect=subprocess.TimeoutExpired("codex", 30)))
    _, check = setup(delivery, iter([job(JobStatus.SUCCEEDED)]))
    for _ in range(2):
        check()
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
    monkeypatch.setattr(followup_commands, "prepare_delivery", setup)
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


TOP_LEVEL_HELP = (
    "Codex CLI\n\nUsage: codex [OPTIONS] [PROMPT]\n\n"
    "  queue  Queue a message for an existing session\n"
)
QUEUE_HELP = "Usage: codex queue [OPTIONS] --thread <THREAD> --message <TEXT>\n"


@pytest.mark.parametrize("supported", [False, True])
def test_prepare_delivery_requires_the_queue_subcommand_help(tmp_path, monkeypatch, supported):
    # Unknown subcommands print Codex's top-level help and still exit 0.
    help_text = QUEUE_HELP if supported else TOP_LEVEL_HELP
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["queue", "--help"]:
            return subprocess.CompletedProcess(command, 0, help_text, "")
        return subprocess.CompletedProcess(command, int(command[1] == "print"), "", "")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(followup_commands.sys, "platform", "darwin")
    monkeypatch.setattr(followup_commands.shutil, "which", lambda _name: "/tools/codex")
    monkeypatch.setattr(followup_commands.subprocess, "run", run)
    settings = tmp_path / "codex.json"

    if not supported:
        with pytest.raises(ValueError, match="does not support queue"):
            followup_commands.prepare_delivery(tmp_path)
        assert not settings.exists()
    else:
        followup_commands.prepare_delivery(tmp_path)
        assert json.loads(settings.read_text())["executable"] == "/tools/codex"
    assert calls == [["/tools/codex", "queue", "--help"]]


def test_unreadable_record_does_not_block_other_followups(checker, capsys, followup):
    store, setup = checker
    store.add(followup)
    unreadable = store.directory / "00000000-0000-4000-8000-0000000000ff.json"
    unreadable.write_text('{"unexpected": true}')
    delivery = Mock(send=Mock(return_value=subprocess.CompletedProcess([], 0, "accepted", "")))

    _, check = setup(delivery, iter([job(JobStatus.SUCCEEDED)]))
    check()

    assert store.get(followup.id).status == "queued"
    assert main(["followup", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in listing["followups"]] == [followup.id]
    assert [row["path"] for row in listing["invalid_records"]] == [str(unreadable)]


def test_stopped_followup_can_be_registered_again_without_requerying_job(checker, followup):
    store, setup = checker
    store.add(followup)
    # Stopped after the job finished, before delivery started.
    followup.status = "stopped"
    followup.job = job(JobStatus.SUCCEEDED)
    followup.outcomes = [job_outcome(followup.server_id, followup.run_id, followup.job)]
    followup.message = completion_message(followup)
    store.save(followup)

    again = store.add(followup.model_copy(update={"id": "another-id", "action": "New action"}))

    assert again.id == followup.id
    assert again.status == "ready" and again.action == "New action"
    assert again.job == followup.job
    delivery = Mock(send=Mock(return_value=subprocess.CompletedProcess([], 0, "accepted", "")))
    transport, check = setup(delivery, iter(()))
    check()
    transport.watch.assert_not_called()
    saved = store.get(followup.id)
    assert saved.status == "queued" and "New action" in saved.message


def test_failed_email_backs_off_exponentially_up_to_one_hour(tmp_path, monkeypatch, followup):
    from gpuops import followups as module

    store = FollowupStore(tmp_path / "records")
    now = utc_now()
    monkeypatch.setattr(module, "utc_now", lambda: now)
    logins = []

    def sender(_followup):
        logins.append(now)
        raise OSError("535 authentication failed")

    followup.status = "observed"
    followup.job = job(JobStatus.SUCCEEDED)
    followup.notify = "terminal"
    store.add(followup)
    delays = []
    for _ in range(9):
        module._deliver_email(store, followup.id, sender, lambda: True)
        module._deliver_email(
            store, followup.id, sender, lambda: True
        )  # inside the delay: no new login
        followup = store.get(followup.id)
        delays.append((followup.email_next_attempt_at - now).total_seconds() / 60)
        now = followup.email_next_attempt_at

    assert len(logins) == 9
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60, 60]
    saved = store.get(followup.id)
    assert saved.email_attempts == 9 and saved.email_error == "535 authentication failed"
