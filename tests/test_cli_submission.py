from __future__ import annotations

import json

import pytest

from gpuops.cli import main
from gpuops.control import RemoteTarget, SubmissionJournal
from gpuops.models import JobRecord, JobSpec, utc_now


@pytest.fixture
def submission_context(tmp_path, monkeypatch):
    journal = SubmissionJournal(tmp_path / "journal")
    monkeypatch.setattr("gpuops.control.SubmissionJournal", lambda _path: journal)
    spec = JobSpec(
        id="simple",
        project="tests",
        command=["true"],
        working_directory="/work",
        resources={"host": "gpu01", "gpus": [0, 1]},
    )
    manifest = tmp_path / "job.json"
    manifest.write_text(spec.model_dump_json())
    return journal, spec, manifest


def test_lost_ack_is_accepted_and_targeted_retry_does_not_resubmit(
    submission_context, monkeypatch, capsys
):
    _journal, _spec, manifest = submission_context

    class Transport:
        jobs = {}
        submissions = []

        async def get_job(self, target, run_id):
            return self.jobs.get(run_id)

        async def submit(self, target, run_id, spec):
            self.submissions.append((target.server_id, run_id))
            self.jobs[run_id] = JobRecord(run_id=run_id, spec=spec)
            raise ConnectionError("acknowledgement lost")

    transport = Transport()
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: transport)
    assert main(["submit", str(manifest), "--run-id", "one-run", "--json"]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["submission_state"] == "accepted"
    assert accepted["run_id"] == "one-run"
    assert main(["reconcile", "one-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "one-run"
    assert transport.submissions == [("gpu01", "one-run")]


def test_unknown_submission_has_identity_and_recovery_command(
    submission_context, monkeypatch, capsys
):
    journal, _spec, manifest = submission_context

    class Transport:
        async def get_job(self, target, run_id):
            raise ConnectionError("SSH unavailable")

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["submit", str(manifest), "--run-id", "uncertain", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["submission_state"] == "unknown"
    assert payload["run_id"] == "uncertain"
    assert payload["server_id"] == "gpu01"
    assert payload["error_code"]
    assert payload["recovery_command"] == ["gpuops", "reconcile", "uncertain", "--json"]
    assert journal.get("uncertain") is not None


def test_pending_list_recovers_saved_identity_without_transport(
    submission_context, monkeypatch, capsys
):
    journal, spec, _manifest = submission_context
    journal.create_pending(
        run_id="before-client-exit",
        target=RemoteTarget("gpu01", "lab-gpu01"),
        request_spec=spec,
        frozen_spec=spec,
        now=utc_now(),
    )
    before = (journal.directory / "before-client-exit.json").read_bytes()

    def no_transport():
        pytest.fail("listing pending submissions must not construct or call a transport")

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", no_transport)
    assert main(["reconcile", "--list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["pending"][0]["run_id"] == "before-client-exit"
    assert payload["pending"][0]["server_id"] == "gpu01"
    assert (journal.directory / "before-client-exit.json").read_bytes() == before


def test_targeted_reconcile_ignores_other_pending_entries(submission_context, monkeypatch, capsys):
    journal, spec, _manifest = submission_context
    for run_id in ("chosen", "leave-pending"):
        journal.create_pending(
            run_id=run_id,
            target=RemoteTarget("gpu01", "lab-gpu01"),
            request_spec=spec,
            frozen_spec=spec,
            now=utc_now(),
        )
    calls = []

    class Transport:
        async def get_job(self, target, run_id):
            calls.append(("get", run_id))
            return None

        async def submit(self, target, run_id, frozen):
            calls.append(("submit", run_id))
            return JobRecord(run_id=run_id, spec=frozen)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["reconcile", "chosen", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["submission_state"] == "accepted"
    assert {run_id for _action, run_id in calls} == {"chosen"}
    assert [entry.run_id for entry in journal.list_pending()] == ["leave-pending"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["status", "--server", "unknown"],
        ["submit"],
        ["reconcile", "one-run", "--list"],
        ["status", "--timeout", "0"],
    ],
)
def test_invalid_cli_arguments_still_produce_json(arguments, capsys):
    assert main([*arguments, "--json"]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_argument"
    assert captured.err == ""


def test_invalid_manifest_never_reports_accepted(submission_context, capsys):
    _journal, _spec, manifest = submission_context
    manifest.write_text('{"id":"incomplete"}')
    assert main(["submit", str(manifest), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "invalid_manifest"
    assert payload["submission_state"] == "not_accepted"


def test_definitive_rejection_retains_structured_reason(submission_context, monkeypatch, capsys):
    from gpuops.transport import RemoteSubmissionRejected

    _journal, _spec, manifest = submission_context

    class Transport:
        async def get_job(self, target, run_id):
            return None

        async def submit(self, target, run_id, spec):
            raise RemoteSubmissionRejected("worker_unavailable", "worker is stopped")

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["submit", str(manifest), "--run-id", "rejected", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["submission_state"] == "not_accepted"
    assert payload["error_code"] == "worker_unavailable"
    assert payload["run_id"] == "rejected"
    assert payload["server_id"] == "gpu01"


def test_local_journal_failure_does_not_hide_remote_acceptance(
    submission_context, monkeypatch, capsys
):
    journal, _spec, manifest = submission_context

    class Transport:
        async def get_job(self, target, run_id):
            return None

        async def submit(self, target, run_id, spec):
            return JobRecord(run_id=run_id, spec=spec)

    def disk_full(*args, **kwargs):
        raise OSError("local disk full")

    monkeypatch.setattr(journal, "mark_accepted", disk_full)
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    assert main(["submit", str(manifest), "--run-id", "accepted", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["submission_state"] == "accepted"
    assert "local disk full" in payload["journal_warning"]
    assert payload["run_id"] == "accepted"


def test_repeated_new_submits_remain_distinct_experiments(submission_context, monkeypatch, capsys):
    _journal, _spec, manifest = submission_context
    calls = []

    class Transport:
        async def get_job(self, target, run_id):
            return None

        async def submit(self, target, run_id, spec):
            calls.append(run_id)
            return JobRecord(run_id=run_id, spec=spec)

    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", Transport)
    for _ in range(2):
        assert main(["submit", str(manifest), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["submission_state"] == "accepted"
    assert len(set(calls)) == 2


def test_corrupt_existing_journal_does_not_mean_manifest_rejected(
    submission_context, monkeypatch, capsys
):
    journal, _spec, manifest = submission_context
    (journal.directory / "prior-run.json").write_text('{"state":"accepted"}')
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: object())

    assert main(["submit", str(manifest), "--run-id", "prior-run", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["submission_state"] == "unknown"
    assert payload["error_code"] == "invalid_state"
    assert payload["run_id"] == "prior-run"


def test_unsupported_new_options_are_definitively_rejected(submission_context, monkeypatch, capsys):
    journal, spec, manifest = submission_context
    spec.tracking.progress_adapter = "unsupported"
    manifest.write_text(spec.model_dump_json())
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: object())
    assert main(["submit", str(manifest), "--run-id", "new-run", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["submission_state"] == "not_accepted"
    assert payload["error_code"] == "unsupported_option"
    assert journal.get("new-run") is None


def test_invalid_run_id_is_rejected_before_submission(submission_context, monkeypatch, capsys):
    _journal, _spec, manifest = submission_context
    monkeypatch.setattr("gpuops.transport.SshWorkerTransport", lambda: object())
    assert main(["submit", str(manifest), "--run-id", "../bad", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "invalid_argument"
    assert payload["submission_state"] == "not_accepted"
    assert "recovery_command" not in payload
