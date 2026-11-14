import asyncio
import fcntl
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpuops.control import ControlClient, JournalEntry, RemoteTarget, SubmissionJournal
from gpuops.followup_groups import GroupRecord
from gpuops.followups import Followup
from gpuops.models import JobRecord, JobSpec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/migrate_to_0_4.py"
STATE = Path(".local/state/gpuops")
BACKUP = Path("gpuops-0.4-backup/records")
RUN_ID = "accepted-run"
SINGLE_ID = "00000000-0000-4000-8000-000000000001"
GROUP_ID = "00000000-0000-4000-8000-000000000002"
GROUP_EVENT = "00000000-0000-4000-8000-000000000003"
TIMESTAMP = "2026-11-07T02:12:00Z"


def write_record(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # Noncanonical encoding makes the byte-for-byte restore observable.
    path.write_bytes((json.dumps(value, separators=(", ", " : ")) + "\n\n").encode())
    return path


def json_files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    mac = tmp_path / "mac"
    gpu1 = tmp_path / "lab-gpu01"
    gpu2 = tmp_path / "lab-gpu02"
    manifest = {
        "api_version": "gpuops/v1",
        "id": "old-experiment",
        "project": "compatibility",
        "command": ["python", "train.py"],
        "working_directory": "/remote/experiment",
        "resources": {"host": "gpu01"},
        "metadata": {"tracking": "keep arbitrary user metadata"},
        "tracking": {"progress_adapter": "jsonl", "output_directory": "/old/results"},
    }
    accepted_record = {
        "run_id": RUN_ID,
        "spec": manifest,
        "status": "queued",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }
    old_result = {
        **accepted_record,
        "status": "failed",
        "finished_at": TIMESTAMP,
        "failure_reason": "training error",
        "exit_code": 23,
        "launch_error": "observed launch failure",
    }
    journal = {
        "run_id": RUN_ID,
        "server_id": "gpu01",
        "ssh_host": "lab-gpu01",
        "state": "accepted",
        "request_spec": {**manifest, "resources": {"host": "auto"}},
        "frozen_spec": manifest,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "remote_record": accepted_record,
    }
    single = {
        "id": SINGLE_ID,
        "run_id": RUN_ID,
        "server_id": "gpu01",
        "ssh_host": "lab-gpu01",
        "thread_id": "original-chat",
        "workspace": "/Users/researcher/experiment",
        "action": "Read the saved outcomes.",
        "status": "completed",
        "message": "Keep the original message and delivery state.",
        "receipt": "accepted",
        "job": old_result,
    }
    group = {
        "id": GROUP_ID,
        "event_id": GROUP_EVENT,
        "thread_id": single["thread_id"],
        "workspace": single["workspace"],
        "action": single["action"],
        "members": [
            {"server_id": "gpu01", "ssh_host": "lab-gpu01", "run_id": RUN_ID, "job": old_result},
            {"server_id": "gpu02", "ssh_host": "lab-gpu02", "run_id": RUN_ID, "job": None},
        ],
    }

    write_record(mac / STATE, f"control-journal/{RUN_ID}.json", journal)
    write_record(gpu1 / STATE, f"control-journal/{RUN_ID}.json", journal)
    write_record(gpu2 / STATE, f"control-journal/{RUN_ID}.json", journal)
    write_record(gpu1 / STATE, f"queue/jobs/{RUN_ID}.json", old_result)
    write_record(gpu2 / STATE, f"queue/jobs/{RUN_ID}.json", accepted_record)
    write_record(mac / STATE, f"followups/records/{SINGLE_ID}.json", single)
    write_record(mac / STATE, f"followups/groups/{GROUP_ID}.json", group)
    write_record(
        mac / STATE,
        f"followups/records/{GROUP_EVENT}.json",
        {
            **single,
            "id": GROUP_EVENT,
            "group_id": GROUP_ID,
            "mode": "all_terminal",
            "outcomes": [{"run_id": RUN_ID, "status": "failed", "measurement": 0.14}],
        },
    )
    for gpu in (gpu1, gpu2):
        (gpu / STATE / "queue/queue-index.sqlite3").write_bytes(b"derived")
        (gpu / STATE / "gpu-sample.lock").touch()
        write_record(gpu / STATE, "runs/results/saved.json", {"exit_code": 23})

    write_record(
        mac,
        ".config/gpuops/servers.json",
        {
            "servers": [
                {"id": "gpu01", "ssh_host": "lab-gpu01"},
                {"id": "gpu02", "ssh_host": "lab-gpu02"},
            ]
        },
    )
    ssh = tmp_path / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        + """\
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["MIGRATION_TEST_FLEET"])
host, _, _, _, _, phase = sys.argv[-6:]
with (root / "ssh.log").open("a") as log:
    log.write(f"{phase} {host}\\n")
if os.environ.get("MIGRATION_TEST_FAIL") == f"{phase} {host}":
    sys.exit("injected SSH failure")
process = subprocess.run(
    [sys.executable, *sys.argv[-4:]],
    env={**os.environ, "HOME": str(root / host)},
    input=sys.stdin.buffer.read(),
)
sys.exit(process.returncode)
"""
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{Path(sys.executable).parent}")
    monkeypatch.setenv("MIGRATION_TEST_FLEET", str(tmp_path))
    monkeypatch.setenv("HOME", str(mac))
    entrypoint = runpy.run_path(str(SCRIPT))["main"]
    # The fake fleet never inspects the Mac login session's real LaunchAgents.
    monkeypatch.setitem(entrypoint.__globals__, "check_launch_agents", lambda: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
        return entrypoint()

    return mac, gpu1, gpu2, run


@pytest.mark.parametrize("status", ["leased", "starting", "running", "cancel_requested"])
def test_fleet_preflight_is_read_only_when_a_remote_job_is_active(fleet, capsys, status):
    mac, gpu1, gpu2, run = fleet
    path = gpu2 / STATE / f"queue/jobs/{RUN_ID}.json"
    value = json.loads(path.read_bytes())
    write_record(gpu2 / STATE, path.relative_to(gpu2 / STATE), {**value, "status": status})
    before = {home: json_files(home) for home in (mac, gpu1, gpu2)}
    assert run() == 1
    assert f"job is {status}" in capsys.readouterr().err
    assert {home: json_files(home) for home in before} == before
    assert all(not (home / BACKUP.parent).exists() for home in before)
    assert all((home / STATE / "queue/queue-index.sqlite3").exists() for home in (gpu1, gpu2))


def test_all_backups_must_succeed_before_any_conversion(fleet, monkeypatch, capsys):
    mac, gpu1, gpu2, run = fleet
    before = {home: json_files(home / STATE) for home in (mac, gpu1, gpu2)}
    monkeypatch.setenv("MIGRATION_TEST_FAIL", "backup lab-gpu02")
    assert run() == 1
    assert "backup on gpu02" in capsys.readouterr().err
    assert {home: json_files(home / STATE) for home in before} == before
    assert (mac / BACKUP / "manifest.json").is_file()
    assert (gpu1 / BACKUP / "manifest.json").is_file()
    assert not (gpu2 / BACKUP).exists()
    assert "convert" not in (mac.parent / "ssh.log").read_text()


def test_worker_lock_blocks_both_conversion_and_restore(fleet, capsys):
    _, _, gpu2, run = fleet
    assert run("--backup-only") == 0
    with (gpu2 / STATE / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run("--check") == 1
        assert run("--restore") == 1
    assert capsys.readouterr().err.count("Worker is still running") == 2


@pytest.mark.parametrize(
    "label", ["io.gpuops.worker-alerts", "io.gpuops.followups", "io.gpuops.background"]
)
def test_loaded_launch_agents_block_preflight(monkeypatch, label):
    check = runpy.run_path(str(SCRIPT))["check_launch_agents"]

    def fake_run(command, **_kwargs):
        if command[-1].endswith(f"/{label}"):
            return subprocess.CompletedProcess(command, 0, "loaded", "")
        return subprocess.CompletedProcess(command, 113, "", "Could not find service")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=label):
        check()


def test_fleet_cutover_resumes_with_original_backups_and_restores_exact_records(
    fleet, monkeypatch, capsys
):
    mac, gpu1, gpu2, run = fleet
    homes = (mac, gpu1, gpu2)
    original = {home: json_files(home / STATE) for home in homes}

    # Simulate an interrupted fleet after some machines have converted.
    monkeypatch.setenv("MIGRATION_TEST_FAIL", "convert lab-gpu02")
    assert run() == 1
    assert "convert on gpu02" in capsys.readouterr().err
    assert all((home / BACKUP / "manifest.json").is_file() for home in homes)
    assert json_files(gpu2 / STATE) == original[gpu2]
    first_backup = {home: json_files(home / BACKUP) for home in homes}

    monkeypatch.delenv("MIGRATION_TEST_FAIL")
    assert run() == 0
    assert {home: json_files(home / BACKUP) for home in homes} == first_backup
    converted = {home: json_files(home / STATE) for home in homes}
    assert run() == 0
    assert {home: json_files(home / STATE) for home in homes} == converted

    for home in homes:
        data = converted[home][Path(f"control-journal/{RUN_ID}.json")]
        raw = json.loads(data)
        assert "tracking" not in raw["request_spec"]
        assert "tracking" not in raw["frozen_spec"]
        assert "tracking" not in raw["remote_record"]["spec"]
        assert raw["remote_record"]["exit_code"] is None
        assert raw["remote_record"]["launch_error"] is None
        entry = JournalEntry.model_validate_json(data)
        assert entry.run_id == RUN_ID
        assert entry.request_spec.metadata == {"tracking": "keep arbitrary user metadata"}

    for host, expected_exit, expected_error in (
        (gpu1, 23, "observed launch failure"),
        (gpu2, None, None),
    ):
        data = converted[host][Path(f"queue/jobs/{RUN_ID}.json")]
        raw = json.loads(data)
        assert raw["exit_code"] == expected_exit
        assert raw["launch_error"] == expected_error
        assert JobRecord.model_validate_json(data).spec.api_version == "gpuops/v1"
        assert not (host / STATE / "queue/queue-index.sqlite3").exists()
        assert not (host / STATE / "gpu-sample.lock").exists()

    single = Followup.model_validate_json(
        converted[mac][Path(f"followups/records/{SINGLE_ID}.json")]
    )
    assert single.outcomes == [
        {
            "server_id": "gpu01",
            "run_id": RUN_ID,
            "status": "failed",
            "exit_code": 23,
            "failure_reason": "training error",
            "remote_workspace": "/remote/experiment",
            "project_workspace": "/Users/researcher/experiment",
        }
    ]
    assert single.job is not None and single.job.launch_error == "observed launch failure"
    assert single.receipt == "accepted"
    assert single.message == "Keep the original message and delivery state."
    grouped_event = Followup.model_validate_json(
        converted[mac][Path(f"followups/records/{GROUP_EVENT}.json")]
    )
    assert grouped_event.outcomes == [{"run_id": RUN_ID, "status": "failed", "measurement": 0.14}]
    group = GroupRecord.model_validate_json(
        converted[mac][Path(f"followups/groups/{GROUP_ID}.json")]
    )
    assert group.members[0].job == single.job
    assert group.members[1].job is None

    journal = SubmissionJournal(mac / STATE / "control-journal")
    client = ControlClient([RemoteTarget("gpu01", "lab-gpu01")], Mock(spec=[]), journal)
    manifest = JobSpec(
        id="old-experiment",
        project="compatibility",
        command=["python", "train.py"],
        working_directory="/remote/experiment",
        metadata={"tracking": "keep arbitrary user metadata"},
    )
    retry = asyncio.run(client.submit(manifest, run_id=RUN_ID))
    assert retry.reconciled and retry.record.run_id == RUN_ID

    new_record = write_record(gpu2 / STATE, "queue/jobs/after-cutover.json", {"status": "queued"})
    log = gpu2 / STATE / "logs/new.log"
    log.parent.mkdir()
    log.write_text("Keep logs written after cutover.")
    new_result = write_record(gpu2 / STATE, "runs/results/new.json", {"exit_code": 0})
    index = gpu1 / STATE / "queue/queue-index.sqlite3"
    index.write_bytes(b"rebuilt")
    assert run("--restore") == 0
    assert not new_record.exists()
    assert not index.exists()
    assert log.read_text() == "Keep logs written after cutover."
    assert json.loads(new_result.read_text()) == {"exit_code": 0}
    actual = {home: json_files(home / STATE) for home in homes}
    del actual[gpu2][Path("runs/results/new.json")]
    assert actual == original
    assert {home: json_files(home / BACKUP) for home in homes} == first_backup
