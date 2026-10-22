"""Parse the explicit GPUOps progress records written to job stdout."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone

PREFIX = "GPUOPS_PROGRESS "
FIELDS = {"phase", "current", "total", "elapsed_seconds", "timestamp"}


def _record(line: str) -> dict | None:
    payload = line[len(PREFIX) :].strip()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != FIELDS:
        return None
    phase = value["phase"]
    current = value["current"]
    total = value["total"]
    elapsed = value["elapsed_seconds"]
    timestamp = value["timestamp"]
    if not isinstance(phase, str) or not phase or len(phase) > 80:
        return None
    if type(current) is not int or current < 0:
        return None
    if total is not None and (type(total) is not int or total <= 0 or total < current):
        return None
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
    ):
        return None
    return {
        "phase": phase,
        "current": current,
        "total": total,
        "elapsed_seconds": float(elapsed),
        "timestamp": float(timestamp),
    }


def _iso(timestamp: float) -> str | None:
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _unknown(reason: str, *, path: str, sampled_at: float, record: dict | None = None) -> dict:
    result = {
        "label": "未知",
        "source_kind": "logs",
        "format": "gpuops_progress",
        "end_at": None,
        "confidence": "unknown",
        "scope": "phase",
        "basis": "GPUOps 结构化进度记录",
        "reason": reason,
        "source": path,
        "sampled_at": _iso(sampled_at),
    }
    if record is not None:
        result["scope"] = (
            "training" if record["phase"].lower() in {"train", "training"} else "phase"
        )
        result["progress"] = {
            "phase": record["phase"],
            "current": record["current"],
            "total": record["total"],
        }
        total = record["total"] if record["total"] is not None else "?"
        result["basis"] = f"{record['phase']} {record['current']}/{total}"
        result["observed_at"] = _iso(record["timestamp"])
    return result


def estimate(
    content: str,
    *,
    path: str,
    started_at: float,
    sampled_at: float,
    now: float,
) -> dict | None:
    """Return ``None`` only when the log contains no explicit progress prefix."""
    lines = [line for line in content.splitlines() if line.startswith(PREFIX)]
    if not lines:
        return None
    if not all(math.isfinite(value) for value in (started_at, sampled_at, now)):
        return _unknown("结构化进度的采样时间无效", path=path, sampled_at=now)

    records = [_record(line) for line in lines]
    latest = records[-1]
    if latest is None:
        return _unknown("最新 GPUOps 结构化进度记录无效", path=path, sampled_at=sampled_at)
    if latest["timestamp"] < started_at - 2:
        return _unknown(
            "结构化进度记录早于当前进程，无法确认属于本次训练",
            path=path,
            sampled_at=sampled_at,
        )
    if latest["timestamp"] > sampled_at + 5 or latest["timestamp"] > now + 5:
        return _unknown(
            "结构化进度记录时间异常",
            path=path,
            sampled_at=sampled_at,
        )
    if latest["total"] is None:
        return _unknown(
            "当前阶段总量未知，无法估计完成时间",
            path=path,
            sampled_at=sampled_at,
            record=latest,
        )
    if latest["current"] == latest["total"]:
        return _unknown(
            "当前阶段已完成，等待进程结束或下一阶段",
            path=path,
            sampled_at=sampled_at,
            record=latest,
        )
    if now - latest["timestamp"] > 600:
        return _unknown(
            "结构化进度已超过 10 分钟未更新",
            path=path,
            sampled_at=sampled_at,
            record=latest,
        )

    # Walk backwards only through the current reporting session. A malformed
    # record, phase/total change, or clock/counter reset is a hard boundary.
    series = [latest]
    for previous in reversed(records[:-1]):
        current = series[-1]
        if previous is None:
            break
        if previous["timestamp"] < started_at - 2:
            break
        if previous["phase"] != current["phase"] or previous["total"] != current["total"]:
            break
        if (
            previous["current"] > current["current"]
            or previous["elapsed_seconds"] > current["elapsed_seconds"]
            or previous["timestamp"] > current["timestamp"]
        ):
            break
        series.append(previous)
    series.reverse()
    deltas = [
        (b["elapsed_seconds"] - a["elapsed_seconds"]) / (b["current"] - a["current"])
        for a, b in zip(series, series[1:], strict=False)
        if b["current"] > a["current"] and b["elapsed_seconds"] > a["elapsed_seconds"]
    ]
    if not deltas:
        return _unknown(
            "结构化进度样本不足，至少需要两次递增报告",
            path=path,
            sampled_at=sampled_at,
            record=latest,
        )
    seconds = statistics.median(deltas[-5:])
    try:
        end = latest["timestamp"] + (latest["total"] - latest["current"]) * seconds
    except OverflowError:
        end = math.inf
    end_at = _iso(end) if math.isfinite(end) else None
    if end_at is None or end <= now:
        return _unknown(
            "已超过结构化进度推算时间，等待新进度",
            path=path,
            sampled_at=sampled_at,
            record=latest,
        )
    scope = "training" if latest["phase"].lower() in {"train", "training"} else "phase"
    return {
        "label": "日志估计" if scope == "training" else "阶段估计",
        "source_kind": "logs",
        "format": "gpuops_progress",
        "end_at": end_at,
        "confidence": "low",
        "scope": scope,
        "basis": (
            f"{latest['phase']} {latest['current']}/{latest['total']}"
            f" · 近期约 {seconds:.2f} 秒/单位"
        ),
        "reason": "当前训练估计，不包含后续任务"
        if scope == "training"
        else "仅当前阶段，总训练时长未知",
        "progress": {
            "phase": latest["phase"],
            "current": latest["current"],
            "total": latest["total"],
        },
        "source": path,
        "observed_at": _iso(latest["timestamp"]),
        "sampled_at": _iso(sampled_at),
    }
