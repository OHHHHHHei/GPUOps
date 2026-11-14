import json
from unittest.mock import Mock

import pytest

from gpuops.followup_groups import (
    GroupStore,
    RemoteJob,
    apply_group_observations,
    create_group,
    stop_group,
    summarize_group,
)
from gpuops.followups import FollowupStore, job_key
from gpuops.models import JobRecord, JobSpec, JobStatus, utc_now


def job(
    run_id: str,
    status: JobStatus,
    *,
    workspace: str | None = None,
    exit_code: int | None = None,
    failure_reason: str | None = None,
) -> JobRecord:
    return JobRecord(
        run_id=run_id,
        spec=JobSpec(
            id=f"spec-{run_id}",
            project="research",
            command=["python", "train.py"],
            working_directory=workspace or f"/remote/{run_id}",
        ),
        status=status,
        finished_at=utc_now() if status.terminal else None,
        exit_code=exit_code,
        failure_reason=failure_reason,
    )


def stores(tmp_path):
    return GroupStore(tmp_path / "root/groups"), FollowupStore(tmp_path / "root/events")


def register(groups, events, members, **kwargs):
    return create_group(
        groups,
        events,
        members,
        thread_id="chat-1",
        workspace="/local/project",
        action="Analyze the saved outcomes.",
        **kwargs,
    )


def observe(groups, events, *jobs):
    by_id = {job.run_id: job for job in jobs}
    observations = {
        job_key(member.server_id, member.ssh_host, member.run_id): by_id[member.run_id]
        for group in groups.list()
        for member in group.members
        if member.run_id in by_id
    }
    with events.lock():
        apply_group_observations(groups, events, observations)


def test_registration_is_idempotent_but_conflicts_and_stopped_names_are_explicit(tmp_path):
    groups, events = stores(tmp_path)
    members = [RemoteJob(run_id="a", server_id="gpu-a", ssh_host="host-a")]

    first = register(groups, events, members, name="round-1")
    again = register(groups, events, members, name="round-1")

    assert again == first
    assert again.event_id == first.event_id
    with pytest.raises(ValueError, match="different settings: action"):
        create_group(
            groups,
            events,
            members,
            thread_id="chat-1",
            workspace="/local/project",
            action="Do something different.",
            name="round-1",
        )
    stop_group(groups, events, first.id)
    with pytest.raises(ValueError, match="stopped"):
        register(groups, events, members, name="round-1")
    second = register(groups, events, members, name="round-2")
    assert second.id != first.id


def test_all_terminal_cross_host_waits_for_every_success_failure_and_cancel(tmp_path):
    groups, events = stores(tmp_path)
    record = register(
        groups,
        events,
        [
            RemoteJob(run_id="success", server_id="gpu-a", ssh_host="host-a"),
            RemoteJob(run_id="failure", server_id="gpu-b", ssh_host="host-b"),
            RemoteJob(run_id="cancel", server_id="gpu-c", ssh_host="host-c"),
        ],
    )
    observe(
        groups,
        events,
        job("success", JobStatus.SUCCEEDED, exit_code=0),
        job("failure", JobStatus.QUEUED),
        job("cancel", JobStatus.CANCELLED),
    )
    assert events.list() == []
    observe(
        groups,
        events,
        job("failure", JobStatus.FAILED, exit_code=7, failure_reason="out of memory"),
    )

    saved = groups.get(record.id)
    created = events.get(record.event_id)
    assert created.status == "ready"
    assert created.job.run_id == "failure" and created.job.status == JobStatus.FAILED
    assert [(row["run_id"], row["status"]) for row in created.outcomes] == [
        ("success", "succeeded"),
        ("failure", "failed"),
        ("cancel", "cancelled"),
    ]
    assert created.outcomes[1]["failure_reason"] == "out of memory"
    assert created.outcomes[2]["remote_workspace"] == "/remote/cancel"
    pending = summarize_group(saved, events)
    assert pending["all_terminal"] is True and pending["completed"] is False
    assert "job" not in pending["events"][0] and "message" not in pending["events"][0]
    created.status = "completed"
    events.save(created)
    complete = summarize_group(saved, events)
    assert complete["completed"] is True and complete["status"] == "analysis_complete"


def message_details(message: str) -> dict:
    return json.loads(message[message.index("\n{") + 1 : message.index("\n}\n") + 2])


@pytest.mark.parametrize("mode", ["all_terminal", "per_job"])
def test_message_reports_one_job_status_only_for_per_job_events(tmp_path, mode):
    groups, events = stores(tmp_path)
    record = register(
        groups,
        events,
        [
            RemoteJob(run_id="failed-first", server_id="gpu-a", ssh_host="host-a"),
            RemoteJob(run_id="succeeded-last", server_id="gpu-b", ssh_host="host-b"),
        ],
        mode=mode,
    )
    observe(
        groups,
        events,
        job("failed-first", JobStatus.FAILED, exit_code=1),
        job("succeeded-last", JobStatus.SUCCEEDED, exit_code=0),
    )

    event_ids = (
        [record.event_id] if mode == "all_terminal" else [m.event_id for m in record.members]
    )
    for event_id in event_ids:
        details = message_details(events.get(event_id).message)
        assert [(row["run_id"], row["status"]) for row in details["jobs"]] == [
            ("failed-first", "failed"),
            ("succeeded-last", "succeeded"),
        ]
        if mode == "all_terminal":
            # The last member to finish succeeded; that must not read as the group result.
            assert not {"run_id", "status", "exit_code"} & details.keys()
        else:
            assert details["status"] == events.get(event_id).job.status.value


def test_per_job_creates_each_event_without_waiting_and_never_duplicates(tmp_path):
    groups, events = stores(tmp_path)
    record = register(
        groups,
        events,
        [
            RemoteJob(run_id="first", server_id="gpu-a", ssh_host="host-a"),
            RemoteJob(run_id="later", server_id="gpu-b", ssh_host="host-b"),
        ],
        mode="per_job",
    )
    observe(
        groups,
        events,
        job("first", JobStatus.SUCCEEDED, exit_code=0),
        job("later", JobStatus.QUEUED),
    )
    first_event_id = record.members[0].event_id
    later_event_id = record.members[1].event_id
    first_event = events.get(first_event_id)
    assert first_event.job.run_id == "first"
    assert first_event.outcomes[1]["status"] == "queued"
    assert not (events.directory / f"{later_event_id}.json").exists()

    observe(groups, events, job("later", JobStatus.FAILED, exit_code=2, failure_reason="boom"))
    observe(groups, events, job("first", JobStatus.QUEUED))
    assert events.get(first_event_id) == first_event
    assert events.get(later_event_id).job.status == JobStatus.FAILED
    assert len(events.list()) == 2


def test_offline_observation_is_retained_and_restart_creates_event_once(tmp_path):
    groups, events = stores(tmp_path)
    record = register(
        groups,
        events,
        [RemoteJob(run_id="run", server_id="gpu-a", ssh_host="host-a")],
    )
    observe(groups, events, job("run", JobStatus.QUEUED))
    with events.lock():
        apply_group_observations(groups, events, {job_key("gpu-a", "host-a", "run"): "offline"})
    offline = groups.get(record.id).members[0]
    assert offline.job.status == JobStatus.QUEUED and offline.error == "offline"
    assert events.list() == []

    restarted_groups = GroupStore(groups.directory)
    restarted_groups.save = Mock(wraps=restarted_groups.save)
    restarted_events = FollowupStore(events.directory)
    for _ in range(2):
        observe(restarted_groups, restarted_events, job("run", JobStatus.SUCCEEDED, exit_code=0))
    assert events.get(record.event_id).job.status == JobStatus.SUCCEEDED
    assert restarted_groups.save.call_count == 1
    assert len(events.list()) == 1


def test_resume_false_records_observation_without_model_message(tmp_path):
    groups, events = stores(tmp_path)
    record = create_group(
        groups,
        events,
        [RemoteJob(run_id="run", server_id="gpu-a", ssh_host="host-a")],
        thread_id="",
        workspace="/local/project",
        action="",
        harness="email-only",
        resume=False,
        notify="terminal",
    )
    observe(groups, events, job("run", JobStatus.CANCELLED))

    event = events.get(record.event_id)
    summary = summarize_group(groups.get(record.id), events)
    assert event.status == "observed" and event.message is None and event.resume is False
    assert summary["completed"] is True
    assert summary["status"] == "observed"
    assert summary["completion_kind"] == "observation"
    event.status = "stopped"
    events.save(event)
    delivery_stopped = summarize_group(groups.get(record.id), events)
    assert delivery_stopped["status"] == "delivery_stopped"
    assert delivery_stopped["completed"] is False
    stop_group(groups, events, record.id)
    assert events.get(record.event_id).status == "stopped"


def test_stop_prevents_future_events_stops_ready_but_does_not_revoke_queued(tmp_path):
    groups, events = stores(tmp_path)
    record = register(
        groups,
        events,
        [
            RemoteJob(run_id="ready", server_id="gpu-a", ssh_host="host-a"),
            RemoteJob(run_id="queued", server_id="gpu-b", ssh_host="host-b"),
            RemoteJob(run_id="future", server_id="gpu-c", ssh_host="host-c"),
        ],
        mode="per_job",
    )
    observe(
        groups,
        events,
        job("ready", JobStatus.SUCCEEDED),
        job("queued", JobStatus.SUCCEEDED),
        job("future", JobStatus.QUEUED),
    )
    queued = events.get(record.members[1].event_id)
    queued.status = "queued"
    events.save(queued)
    ready = events.get(record.members[0].event_id)
    ready.status = "stopped"
    events.save(ready)
    assert summarize_group(groups.get(record.id), events)["status"] == "partially_stopped"

    summary = stop_group(groups, events, record.id)

    assert summary["status"] == "stopped"
    assert events.get(record.members[0].event_id).status == "stopped"
    assert events.get(record.members[1].event_id).status == "queued"
    observe(groups, events, job("future", JobStatus.SUCCEEDED))
    assert not (events.directory / f"{record.members[2].event_id}.json").exists()


def test_bad_group_record_is_isolated(tmp_path):
    groups, events = stores(tmp_path)
    valid = register(
        groups,
        events,
        [RemoteJob(run_id="run", server_id="gpu-a", ssh_host="host-a")],
    )
    bad = groups.directory / "00000000-0000-4000-8000-0000000000ff.json"
    bad.write_text('{"unexpected": true}')

    records, invalid = groups.scan()

    assert records == [valid]
    assert invalid[0]["path"] == str(bad)
