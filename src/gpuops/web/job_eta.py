"""Display-only managed-job estimates sharing the GPU overview's log parser."""

from __future__ import annotations

from datetime import datetime

from gpuops.models import JobRecord, JobStatus, JobSummary
from gpuops.web.progress import estimate, unknown


def managed_log_estimate(record: JobRecord, response: dict, now: datetime) -> dict:
    if record.launch is None:
        return unknown("作业尚未启动")
    source = response.get("source") or f"GPUOps 托管日志 · {record.run_id}"
    content = str(response.get("content") or "")
    if response.get("truncated"):
        # The RPC byte limit may begin inside a progress record.
        boundary = min((i for i in (content.find("\n"), content.find("\r")) if i >= 0), default=-1)
        content = content[boundary + 1 :] if boundary >= 0 else ""
    sample = {
        "sampled_at": response.get("sampled_at", now.timestamp()),
        "utc_offset_seconds": response.get("utc_offset_seconds"),
        "logs": {source: {"content": content, "modified_at": response.get("modified_at")}},
    }
    process = {"argv": record.spec.command, "started_at": record.launch.started_at.timestamp()}
    try:
        return estimate(sample, process, source, now=now.timestamp())
    except (ValueError, OverflowError):
        return unknown("无法解析日志中的进度数值")


def job_estimate(record: JobSummary, now: datetime, log_eta: dict | None = None) -> dict:
    if record.status.terminal:
        return {
            "end_at": record.finished_at.isoformat() if record.finished_at else None,
            "source_kind": "actual",
            "confidence": "high",
            "basis": "实际结束时间",
        }
    if record.status != JobStatus.RUNNING:
        reason = {
            JobStatus.QUEUED: "等待调度，开始时间取决于 GPU 空闲情况和调度条件",
            JobStatus.LEASED: "资源已分配，等待启动",
            JobStatus.STARTING: "正在启动",
            JobStatus.CANCEL_REQUESTED: "正在取消，等待进程退出",
        }[record.status]
        return unknown(reason)
    if log_eta and (log_eta.get("end_at") or log_eta.get("format") == "gpuops_progress"):
        return log_eta
    if record.started_at and record.expected_duration:
        end = record.started_at + record.expected_duration
        if end > now:
            return {
                "end_at": end.isoformat(),
                "source_kind": "submitted",
                "scope": "job",
                "confidence": "low",
                "basis": "实际启动时间 + 提交时填写的预计时长",
                "reason": "提交时估计，尚无可用的日志估时",
            }
    return log_eta or unknown("暂无足够的日志进度，提交时长也未提供有效估计")
