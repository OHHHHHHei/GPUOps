"""Managed and process views must agree on the same log evidence."""

from datetime import datetime, timedelta, timezone

import pytest
from test_web import managed_running_job

from gpuops.models import JobStatus, JobSummary
from gpuops.web.job_eta import job_estimate, managed_log_estimate
from gpuops.web.progress import estimate

NOW = datetime(2026, 9, 19, 10, tzinfo=timezone.utc)
LOG = "epoch=1/5 seconds=120\nepoch=2/5 seconds=100\n"


def evidence():
    return {
        "content": LOG,
        "modified_at": (NOW - timedelta(seconds=30)).timestamp(),
        "sampled_at": NOW.timestamp(),
        "utc_offset_seconds": 28800,
        "source": "/logs/run/console.log",
    }


def test_managed_and_process_estimates_share_timing_and_scope():
    record, response = managed_running_job(), evidence()
    managed = managed_log_estimate(record, response, NOW)
    sample = {**response, "logs": {response["source"]: response}}
    process = {"argv": record.spec.command, "started_at": record.launch.started_at.timestamp()}
    direct = estimate(sample, process, response["source"], now=NOW.timestamp())
    assert managed == direct
    assert managed["end_at"] == "2026-09-19T10:05:00+00:00"
    assert managed["source_kind"] == "logs"
    assert job_estimate(JobSummary.from_record(record), NOW, managed) == managed
    # Reading the unchanged file again must not move its predicted end forward.
    later = managed_log_estimate(
        record, {**response, "sampled_at": NOW.timestamp() + 60}, NOW + timedelta(seconds=60)
    )
    assert later["end_at"] == managed["end_at"]


def test_legacy_logs_need_explicit_timestamps_to_estimate_without_metadata():
    record = managed_running_job()
    assert managed_log_estimate(record, {"content": LOG}, NOW)["end_at"] is None
    content = "2026-09-19T09:58:00Z global_step 100/1000\n2026-09-19T09:59:00Z global_step 200/1000"
    assert managed_log_estimate(record, {"content": content}, NOW)["end_at"]
    naive = content.replace("Z", "")
    assert managed_log_estimate(record, {"content": naive}, NOW)["end_at"] is None


def test_phase_progress_is_not_presented_as_whole_job_completion():
    response = {**evidence(), "content": "200/1000 [01:40<06:40, 2.00it/s]"}
    eta = managed_log_estimate(managed_running_job(), response, NOW)
    assert eta["scope"] == "phase"
    assert "总训练时长未知" in eta["reason"]


def test_interleaved_epoch_durations_are_not_combined_into_one_job():
    response = {
        **evidence(),
        "content": "trial_a epoch=1/3 seconds=120\ntrial_b epoch=1/3 seconds=200",
    }
    assert managed_log_estimate(managed_running_job(), response, NOW)["end_at"] is None


def test_submitted_fallback_is_labelled_and_expires():
    summary = JobSummary.from_record(managed_running_job())
    eta = job_estimate(summary, NOW)
    assert eta["source_kind"] == "submitted"
    assert eta["confidence"] == "low"
    assert job_estimate(summary, NOW + timedelta(hours=4))["end_at"] is None


def test_recognized_structured_progress_without_eta_blocks_submitted_fallback():
    summary = JobSummary.from_record(managed_running_job())
    structured = {
        "format": "gpuops_progress",
        "end_at": None,
        "reason": "当前阶段总量未知，无法估计完成时间",
    }
    assert job_estimate(summary, NOW, structured) == structured


@pytest.mark.parametrize(
    "status", [JobStatus.QUEUED, JobStatus.LEASED, JobStatus.STARTING, JobStatus.CANCEL_REQUESTED]
)
def test_waiting_and_cancelling_jobs_never_inherit_a_training_eta(status):
    summary = JobSummary.from_record(managed_running_job()).model_copy(update={"status": status})
    eta = job_estimate(summary, NOW, {"end_at": "2099-01-01T00:00:00Z"})
    assert eta["end_at"] is None


def test_terminal_job_uses_actual_finish_even_if_logs_predict_later():
    summary = JobSummary.from_record(managed_running_job()).model_copy(
        update={"status": JobStatus.SUCCEEDED, "finished_at": NOW}
    )
    eta = job_estimate(summary, NOW, {"end_at": "2099-01-01T00:00:00Z"})
    assert eta["source_kind"] == "actual"
    assert eta["end_at"] == NOW.isoformat()


def test_truncated_first_progress_record_is_not_accepted_as_a_complete_line():
    response = {**evidence(), "content": "epoch=1/3 seconds=100", "truncated": True}
    assert managed_log_estimate(managed_running_job(), response, NOW)["end_at"] is None
