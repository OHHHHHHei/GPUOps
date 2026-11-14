import asyncio
import json
import plistlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

from gpuops import alerts, background, followup_commands, followup_delivery
from gpuops.cli import main
from gpuops.control import RemoteTarget
from gpuops.followup_groups import RemoteJob, create_group
from gpuops.followups import Followup, completion_message
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now
from gpuops.transport import RemoteWorkerError


class Launchd:
    def __init__(self):
        self.loaded = False
        self.changes = []

    def __call__(self, action, *args, check=True):
        if action == "print":
            return self.loaded
        assert action in {"bootstrap", "bootout"}
        assert self.loaded == (action == "bootout")
        self.loaded = not self.loaded
        self.changes.append(action)
        return True


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(background.sys, "platform", "darwin")
    launchd = Launchd()
    monkeypatch.setattr(background, "_launchctl", launchd)
    monkeypatch.setattr(followup_commands, "prepare_delivery", Mock())
    bg = background.Background()
    transport = Mock(watch=AsyncMock(side_effect=AssertionError("unexpected SSH")))
    monkeypatch.setattr(background, "SshWorkerTransport", lambda **_: transport)
    monkeypatch.setattr(alerts, "_notify", Mock())
    monkeypatch.setattr(followup_delivery, "send_followup_email", Mock())
    return bg, launchd, transport


def run(bg):
    return asyncio.run(bg.run_round())


def job(run_id, status=JobStatus.SUCCEEDED):
    return JobRecord(
        run_id=run_id,
        spec=JobSpec(id="spec", project="test", command=["true"], working_directory="/remote"),
        status=status,
        finished_at=utc_now() if status.terminal else None,
    )


def response(target, jobs=None, *, process="running", ready=True):
    return {
        "server_id": target.server_id,
        "worker": {"process": process, "ready": ready, "reason": process},
        "jobs": jobs or {},
    }


def event(bg, run_id, *, status="waiting", notify="none", server_id="gpu01", ssh_host="lab-gpu01"):
    row = Followup(
        run_id=run_id,
        server_id=server_id,
        ssh_host=ssh_host,
        thread_id="original",
        workspace=str(bg.home),
        action="Read metrics",
        status=status,
        notify=notify,
    )
    if status != "waiting":
        row.job = job(run_id)
        row.message = completion_message(row)
    bg.event_store().add(row)
    return row


def enable_followups(bg):
    store = bg.event_store()
    with store.lock():
        bg.enable_followups_locked(store)


def test_status_is_read_only_and_missing_config_defaults_off(app, capsys):
    bg, launchd, transport = app
    assert main(["background", "status", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": True,
        "enabled": {"alerts": False, "followups": False},
        "loaded": False,
        "last_round": None,
    }
    assert not list(bg.home.iterdir())
    assert launchd.changes == []
    transport.watch.assert_not_called()


def test_scheduler_follows_two_independent_switches_and_pending_work(app, capsys):
    bg, launchd, _transport = app
    assert main(["followup", "enable", "--json"]) == 0
    assert not launchd.loaded
    event(bg, "pending")
    assert main(["followup", "enable"]) == 0
    assert launchd.changes == ["bootstrap"]
    config = plistlib.loads(bg.plist.read_bytes())
    assert config["Label"] == "io.gpuops.background"
    assert config["ProgramArguments"][-3:] == ["gpuops", "background", "_run"]
    assert config["StartInterval"] == 60

    assert main(["alerts", "enable"]) == 0
    assert main(["followup", "disable"]) == 0
    assert launchd.loaded and bg.alerts_enabled() and not bg.followups_enabled()
    assert main(["followup", "enable"]) == 0
    assert main(["alerts", "disable"]) == 0
    assert launchd.loaded and not bg.alerts_enabled() and bg.followups_enabled()
    assert main(["followup", "disable"]) == 0
    assert not launchd.loaded and not bg.plist.exists()
    assert launchd.changes == ["bootstrap", "bootout"]
    assert bg.event_store().list()[0].status == "waiting"
    capsys.readouterr()
    assert main(["followup", "check", "--json"]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["reason"] == "followups_paused" and paused["ssh_calls"] == 0


@pytest.mark.parametrize("action", ["retry", "complete"])
@pytest.mark.parametrize("enabled", [False, True])
def test_retry_and_completion_email_schedule_without_changing_switch(app, capsys, action, enabled):
    bg, launchd, transport = app
    if enabled:
        assert main(["followup", "enable"]) == 0
        capsys.readouterr()
    row = event(
        bg,
        "pending",
        status="delivery_unknown" if action == "retry" else "queued",
        notify="none" if action == "retry" else "analysis_complete",
    )
    assert not launchd.loaded
    command = ["followup", action, row.id, "--session", "original", "--json"]
    if action == "complete":
        command.extend(["--summary", "saved analysis"])
    assert main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == ("ready" if action == "retry" else "completed")
    assert launchd.loaded == enabled == bg.followups_enabled()
    assert ("background_message" in result) != enabled
    if not enabled:
        assert "gpuops followup enable" in result["background_message"]
    transport.watch.assert_not_called()


def test_enabling_adapter_retries_loads_an_idle_checker(app, capsys):
    bg, launchd, transport = app
    row = event(bg, "pending", status="delivery_unknown")
    row.harness = "external"
    bg.event_store().save(row)
    enable_followups(bg)
    assert not launchd.loaded

    settings = bg.home / "adapter.json"
    settings.write_text(json.dumps({"command": ["/fake/adapter"], "idempotent": True}))
    assert main(["followup", "adapter", "external", str(settings), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["idempotent"]
    assert launchd.loaded and launchd.changes == ["bootstrap"]
    transport.watch.assert_not_called()


def test_uncertain_delivery_uses_one_adapter_configuration_per_round(app, monkeypatch):
    bg, launchd, transport = app
    settings = bg.home / "adapter.json"
    settings.write_text(json.dumps({"command": ["/fake/old"], "idempotent": True}))
    assert main(["followup", "adapter", "external", str(settings)]) == 0
    for run_id in ("first", "second"):
        row = event(bg, run_id, status="delivery_unknown")
        row.harness = "external"
        bg.event_store().save(row)
    enable_followups(bg)
    assert launchd.loaded

    attempted = []

    def deliver(command, **_):
        attempted.append(command)
        if len(attempted) == 1:
            settings.write_text(json.dumps({"command": ["/fake/new"], "idempotent": False}))
            assert main(["followup", "adapter", "external", str(settings)]) == 0
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(followup_delivery.subprocess, "run", deliver)
    stats = run(bg)
    assert attempted == [["/fake/old"], ["/fake/old"]]
    assert stats["pending"]["events"] == 0
    assert not launchd.loaded
    assert {row.status for row in bg.event_store().list()} == {"delivery_unknown"}
    transport.watch.assert_not_called()


def test_shared_watch_deduplicates_groups_and_uses_full_job_records(app, monkeypatch):
    bg, _, transport = app
    single = event(bg, "shared")
    groups, events = bg.group_store(), bg.event_store()
    for name in ["one", "two"]:
        create_group(
            groups,
            events,
            [RemoteJob(run_id="shared", server_id="gpu01", ssh_host="lab-gpu01")],
            thread_id="original",
            workspace=str(bg.home),
            action="Read metrics",
            name=name,
        )
    running = event(bg, "running")
    missing = event(bg, "missing")
    invalid = event(bg, "invalid")
    enable_followups(bg)
    assert main(["alerts", "enable"]) == 0

    def watch(target, ids):
        if target.server_id == "gpu02":
            assert ids == []
            raise RemoteWorkerError("version_mismatch", "Upgrade GPU 02")
        assert ids == ["invalid", "missing", "running", "shared"]
        return response(
            target,
            {
                "shared": job("shared").model_dump(mode="json"),
                "running": job("running", JobStatus.QUEUED).model_dump(mode="json"),
                "missing": None,
                "invalid": {"error": {"code": "job_read_failed", "message": "Invalid record"}},
            },
            process="stale",
            ready=False,
        )

    transport.watch.side_effect = watch
    send = Mock(return_value=subprocess.CompletedProcess([], 0, "receipt", ""))
    monkeypatch.setattr(background, "DeliveryRouter", lambda *_: Mock(send=send))
    stats = run(bg)
    assert (stats["ssh_calls"], stats["targets"], stats["jobs"]) == (2, 2, 4)
    assert stats["pending"] == {"waiting_jobs": 3, "events": 0, "emails": 0, "total": 3}
    assert bg.status()["last_round"]["ssh_calls"] == 2
    assert {error["error_code"] for error in stats["errors"]} >= {
        "version_mismatch",
        "job_read_failed",
    }
    assert events.get(single.id).status == "queued" and send.call_count == 3
    assert events.get(running.id).job.status == JobStatus.QUEUED
    assert "not found" in events.get(missing.id).error
    assert events.get(invalid.id).error == "Invalid record"
    worker_alerts = alerts._read_state(bg.alert_state)["servers"]
    assert [row["status"] for row in worker_alerts] == ["stopped", "unknown"]
    assert worker_alerts[1]["reason"] == "Upgrade GPU 02"
    alerts._notify.assert_called_once()


def test_running_worker_version_mismatch_does_not_discard_jobs(app, monkeypatch):
    bg, _, transport = app
    row = event(bg, "terminal")
    enable_followups(bg)
    assert main(["alerts", "enable"]) == 0
    send = Mock(return_value=subprocess.CompletedProcess([], 0, "receipt", ""))
    monkeypatch.setattr(background, "DeliveryRouter", lambda *_: Mock(send=send))

    def watch(target, ids):
        if not ids:
            return response(target)
        payload = response(
            target, {"terminal": job("terminal").model_dump(mode="json")}, ready=False
        )
        payload["worker"].update(error_code="version_mismatch", reason="Upgrade Worker")
        return payload

    transport.watch.side_effect = watch
    stats = run(bg)
    worker = alerts._read_state(bg.alert_state)["servers"][0]
    assert worker["status"] == "unknown" and worker["reason"] == "Upgrade Worker"
    assert stats["errors"] == [
        {
            "scope": "worker",
            "server_id": "gpu01",
            "ssh_host": "lab-gpu01",
            "error_code": "version_mismatch",
            "error": "Upgrade Worker",
        }
    ]
    alerts._notify.assert_not_called()
    assert bg.event_store().get(row.id).status == "queued"
    send.assert_called_once()


def test_batches_limited_to_500_and_route_is_part_of_identity():
    targets = [RemoteTarget("same-id", "route-a"), RemoteTarget("same-id", "route-b")]
    ids = {f"job-{n:03}" for n in range(501)}
    first_batches_started = set()

    async def watch(target, batch):
        first_batches_started.add(target)
        await asyncio.sleep(0)  # Let the other target start without waiting for this batch.
        assert first_batches_started == set(targets)
        return response(target, {run_id: None for run_id in batch})

    transport = Mock(watch=AsyncMock(side_effect=watch))
    round_ = background.Round()
    asyncio.run(round_.collect(transport, {targets[0]: ids, targets[1]: {"job-000"}}))
    batches = [(call.args[0], len(call.args[1])) for call in transport.watch.await_args_list]
    assert batches == [(targets[0], 500), (targets[1], 1), (targets[0], 1)]
    assert round_.ssh_calls == 3 and round_.targets == 2 and round_.jobs == 502


def test_stop_during_watch_wins_and_does_not_block(app, capsys):
    bg, launchd, transport = app
    row = event(bg, "pending")
    groups, events = bg.group_store(), bg.event_store()
    group = create_group(
        groups,
        events,
        [RemoteJob(run_id="pending", server_id="gpu01", ssh_host="lab-gpu01")],
        thread_id="original",
        workspace=str(bg.home),
        action="Read metrics",
    )
    enable_followups(bg)
    entered, release = Event(), Event()

    async def watch(target, ids):
        entered.set()
        assert release.wait(3)
        return response(target, {ids[0]: job(ids[0]).model_dump(mode="json")})

    transport.watch.side_effect = watch
    with ThreadPoolExecutor(max_workers=2) as pool:
        polling = pool.submit(run, bg)
        assert entered.wait(3)
        try:
            assert pool.submit(main, ["followup", "stop", row.id, "--json"]).result(2) == 0
            assert pool.submit(main, ["followup", "stop", group.id, "--json"]).result(2) == 0
            assert launchd.loaded
        finally:
            release.set()
        stats = polling.result(3)
    assert stats["ssh_calls"] == 1 and stats["pending"]["total"] == 0
    assert events.get(row.id).status == "stopped" and events.get(row.id).job is None
    assert groups.get(group.id).stopped and groups.get(group.id).members[0].job is None
    assert [record.id for record in events.list()] == [row.id]
    assert launchd.changes == ["bootstrap", "bootout"]
    capsys.readouterr()


@pytest.mark.parametrize("ack", ["acknowledge", "complete"])
@pytest.mark.parametrize("returncode", [0, 7])
def test_sending_can_be_confirmed_and_only_one_round_makes_claim(
    app, monkeypatch, capsys, ack, returncode
):
    bg, launchd, transport = app
    row = event(bg, "claim", status="ready")
    waiting = event(bg, "another", status="ready")
    enable_followups(bg)
    entered, release = Event(), Event()

    def send(followup):
        # IDs are sorted; only the first claim is allowed after disabling follow-ups.
        entered.set()
        assert release.wait(3)
        code = returncode if followup.id == first.id else 0
        return subprocess.CompletedProcess([], code, "receipt", "rejected")

    sender = Mock(side_effect=send)
    monkeypatch.setattr(background, "DeliveryRouter", lambda *_: Mock(send=sender))
    first, second = sorted([row, waiting], key=lambda record: record.id)
    store = bg.event_store()
    with ThreadPoolExecutor(max_workers=2) as pool:
        polling = pool.submit(run, bg)
        assert entered.wait(3)
        try:
            assert store.get(first.id).status == "sending"
            assert pool.submit(main, ["followup", "check", "--json"]).result(2) == 0
            assert json.loads(capsys.readouterr().out)["reason"] == "round_in_progress"
            assert store.get(first.id).status == "sending"
            command = ["followup", ack, first.id, "--session", "original", "--json"]
            if ack == "complete":
                command.extend(["--summary", "saved analysis"])
            assert pool.submit(main, command).result(2) == 0
            assert pool.submit(main, ["followup", "disable"]).result(2) == 0
            assert launchd.loaded
        finally:
            release.set()
        polling.result(3)
    assert sender.call_count == 1
    confirmed = store.get(first.id)
    assert confirmed.status == ("received" if ack == "acknowledge" else "completed")
    assert confirmed.received_at is not None
    assert confirmed.summary == ("saved analysis" if ack == "complete" else None)
    assert confirmed.error == ("rejected" if returncode else None)
    assert confirmed.receipt == (None if returncode else "receipt")
    assert bool(confirmed.queued_at) == (returncode == 0)
    assert store.get(second.id).status == "ready"
    assert launchd.changes == ["bootstrap", "bootout"]
    assert main(["followup", "enable"]) == 0
    run(bg)
    assert sender.call_count == 2 and not launchd.loaded
    transport.watch.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_smtp_writeback_preserves_concurrent_completion(app, monkeypatch, fails):
    bg, launchd, transport = app
    row = event(bg, "mail", status="queued", notify="terminal")
    enable_followups(bg)
    entered, release = Event(), Event()

    def send(_):
        entered.set()
        assert release.wait(3)
        if fails:
            raise RuntimeError("SMTP offline")

    monkeypatch.setattr(followup_delivery, "send_followup_email", send)
    store = bg.event_store()
    with ThreadPoolExecutor(max_workers=2) as pool:
        polling = pool.submit(run, bg)
        assert entered.wait(3)
        try:
            completed = pool.submit(
                store.acknowledge, row.id, summary="saved analysis", session_id="original"
            ).result(2)
            assert completed.status == "completed"
            assert pool.submit(main, ["followup", "disable"]).result(2) == 0
            assert launchd.loaded
        finally:
            release.set()
        polling.result(3)
    saved = store.get(row.id)
    assert saved.status == "completed" and saved.summary == "saved analysis"
    assert saved.email_attempts == int(fails)
    assert bool(saved.email_next_attempt_at) == fails
    assert bool(saved.email_sent_at) != fails
    assert launchd.changes == ["bootstrap", "bootout"]
    transport.watch.assert_not_called()


def test_disable_alerts_between_desktop_and_email_prevents_email(app, monkeypatch):
    bg, launchd, transport = app
    assert main(["alerts", "enable"]) == 0
    transport.watch.side_effect = lambda target, ids: response(
        target, process="stopped", ready=False
    )
    settings = Mock()
    monkeypatch.setattr(alerts, "_email_settings", lambda: (settings, None))
    monkeypatch.setattr(alerts, "send_mail", Mock())
    entered, release = Event(), Event()

    def notify(*_):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(alerts, "_notify", notify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        polling = pool.submit(run, bg)
        assert entered.wait(3)
        try:
            assert pool.submit(main, ["alerts", "disable"]).result(2) == 0
            assert launchd.loaded
        finally:
            release.set()
        polling.result(3)
    alerts.send_mail.assert_not_called()
    assert alerts._read_state(bg.alert_state)["servers"][0]["stop_notified"]
    assert launchd.changes == ["bootstrap", "bootout"]


def test_registration_during_self_unload_reloads_agent(app, monkeypatch, capsys):
    bg, launchd, _transport = app
    assert main(["alerts", "enable"]) == 0
    capsys.readouterr()
    with bg.event_store().lock():
        bg.set_switch_locked("alerts", False)
    entered, registering, release = Event(), Event(), Event()

    def launchctl(action, *args, **kwargs):
        if action == "bootout":
            entered.set()
            assert release.wait(3)
        return launchd(action, *args, **kwargs)

    monkeypatch.setattr(background, "_launchctl", launchctl)
    monkeypatch.setenv("CODEX_THREAD_ID", "00000000-0000-4000-8000-000000000001")
    command = ["followup", "add", "new", "--action", "Read metrics", "--server", "gpu01", "--json"]

    def register():
        registering.set()
        return main(command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        polling = pool.submit(run, bg)
        assert entered.wait(3)
        adding = pool.submit(register)
        assert registering.wait(3)
        release.set()
        polling.result(3)
        assert adding.result(3) == 0
    assert launchd.loaded and launchd.changes == ["bootstrap", "bootout", "bootstrap"]
    assert bg.followups_enabled() and not bg.alerts_enabled()
    assert bg.event_store().list()[0].run_id == "new"
    assert json.loads(capsys.readouterr().out)["run_id"] == "new"


@pytest.mark.parametrize("broken", ["alerts", "followups"])
def test_feature_error_does_not_block_other_feature_or_stats(app, monkeypatch, broken):
    bg, _, transport = app
    row = event(bg, "pending")
    enable_followups(bg)
    assert main(["alerts", "enable"]) == 0
    transport.watch.side_effect = lambda target, ids: response(
        target,
        {run_id: job(run_id).model_dump(mode="json") for run_id in ids},
        process="stopped",
        ready=False,
    )
    send = Mock(return_value=subprocess.CompletedProcess([], 0, "receipt", ""))
    monkeypatch.setattr(background, "DeliveryRouter", lambda *_: Mock(send=send))
    error = Mock(side_effect=OSError(f"{broken} failed"))
    if broken == "alerts":
        monkeypatch.setattr(alerts, "process_observations", error)
    else:
        monkeypatch.setattr(background, "apply_followup_observations", error)
    stats = run(bg)
    assert stats["errors"][0]["scope"] == broken
    assert bg.status()["last_round"]["errors"] == stats["errors"]
    assert bg.event_store().get(row.id).status == ("queued" if broken == "alerts" else "waiting")
    assert alerts._notify.call_count == (2 if broken == "followups" else 0)


@pytest.mark.parametrize("failure", ["invalid_utf8_output", "decode_error"])
def test_failure_after_adapter_started_is_never_resent(app, monkeypatch, failure):
    bg, _, transport = app
    spool = bg.home / "spool"
    spool.mkdir()
    adapter = bg.home / "adapter.py"
    # The adapter accepts the event, then prints bytes that are not UTF-8.
    adapter.write_text(
        "import pathlib, sys, uuid\n"
        f"pathlib.Path({str(spool)!r}, uuid.uuid4().hex).touch()\n"
        "sys.stdout.buffer.write(b'\\xff\\xfe accepted')\n"
    )
    settings = bg.home / "adapter.json"
    settings.write_text(json.dumps({"command": [sys.executable, str(adapter)]}))
    assert main(["followup", "adapter", "external", str(settings)]) == 0
    row = event(bg, "accepted", status="ready")
    row.harness = "external"
    bg.event_store().save(row)
    enable_followups(bg)
    if failure == "decode_error":
        real_run = followup_delivery.subprocess.run

        def run_then_fail_decoding(*args, **kwargs):
            real_run(*args, **kwargs)
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        monkeypatch.setattr(followup_delivery.subprocess, "run", run_then_fail_decoding)

    run(bg)
    run(bg)

    assert len(list(spool.iterdir())) == 1
    assert bg.event_store().get(row.id).status == "delivery_unknown"
    transport.watch.assert_not_called()


def test_unexpected_error_in_one_event_does_not_stop_the_round(app, monkeypatch):
    bg, _, transport = app
    first = event(bg, "first", status="queued", notify="terminal")
    second = event(bg, "second", status="queued", notify="terminal")
    enable_followups(bg)
    sent = []

    def send(followup):
        if followup.id == first.id:
            raise KeyError("broken message template")
        sent.append(followup.id)

    monkeypatch.setattr(followup_delivery, "send_followup_email", send)

    stats = run(bg)

    assert sent == [second.id]
    assert bg.event_store().get(second.id).email_sent_at is not None
    assert [(row["scope"], row.get("followup_id")) for row in stats["errors"]] == [
        ("followups", first.id)
    ]
    transport.watch.assert_not_called()
