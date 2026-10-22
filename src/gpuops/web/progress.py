"""Conservative progress estimates from bounded training-log samples."""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, timedelta, timezone

from gpuops.web.structured_progress import estimate as estimate_structured

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP = re.compile(
    r"^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}|"
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
)
EPOCH = re.compile(r"\bepoch\s*[:= ]\s*(\d+)(?:\s*/\s*(\d+))?", re.I)
STEP = re.compile(r"\b(global[_ ]?step|step|iteration|iter)\s*[:= ]\s*(\d+)\s*/\s*(\d+)", re.I)
BAR = re.compile(
    r"\b(\d+)\s*/\s*(\d+)\s*\[([\d:]+)<([\d:]+|\?),"
    r"\s*(\d+(?:\.\d+)?)\s*(it/s|s/it)",
    re.I,
)
EPOCH_DURATION = re.compile(
    r"\b(?:epoch_seconds|seconds|elapsed_sec)\s*[:=]\s*(\d+(?:\.\d+)?)\b", re.I
)


def unknown(reason: str) -> dict:
    return {"label": "未知", "end_at": None, "confidence": "unknown", "reason": reason}


def _total(argv: list[str], content: str, unit: str) -> int | None:
    flags = {"--epochs", "--num_epochs", "--num-epochs", "--max_epochs", "--max-epochs"}
    keys = "num_epochs|epochs|max_epochs"
    if unit == "step":
        flags = {"--max_steps", "--max-steps", "--total_steps", "--total-steps"}
        keys = "max_steps|total_steps"
    for index, argument in enumerate(argv):
        flag, separator, value = argument.partition("=")
        if flag in flags:
            value = value if separator else (argv[index + 1] if index + 1 < len(argv) else "")
            if value.isdigit() and int(value) > 0:
                return int(value)
    values = re.findall(rf"['\"](?:{keys})['\"]\s*:\s*(\d+)", content)
    return int(values[-1]) if values and int(values[-1]) > 0 else None


def _timestamp(line: str, offset: float | None) -> float | None:
    match = TIMESTAMP.match(line)
    if match is None:
        return None
    value = match[1]
    try:
        parsed = (
            datetime.strptime(value, "%m/%d/%y %H:%M:%S")
            if "/" in value
            else datetime.fromisoformat(value.replace("Z", "+00:00").replace(",", "."))
        )
        if parsed.tzinfo is None:
            if offset is None:
                return None
            parsed = parsed.replace(tzinfo=timezone(timedelta(seconds=offset)))
        return parsed.timestamp()
    except ValueError:
        return None


def estimate(sample: dict, process: dict, path: str, *, now: float) -> dict:
    """Estimate only observed progress; no inference from GPU load or process age."""
    log = sample["logs"][path]
    if log.get("error"):
        return unknown("日志不可读取")
    content = ANSI.sub("", log["content"])
    structured = estimate_structured(
        content,
        path=path,
        started_at=float(process["started_at"]),
        sampled_at=float(sample["sampled_at"]),
        now=float(now),
    )
    if structured is not None:
        return structured
    modified = log["modified_at"]
    if modified is not None and modified < process["started_at"] - 2:
        return unknown("日志早于当前进程，无法确认属于本次训练")
    if modified is not None and modified > sample["sampled_at"] + 5:
        return unknown("日志时间异常")
    argv = process["argv"]
    epochs = _total(argv, content, "epoch")
    total_steps = _total(argv, content, "step")
    epoch_points, step_points = [], []
    epoch_durations = []
    bar = None
    training_finished = False
    lines = [line for line in re.split(r"[\r\n]+", content) if line.strip()]
    for line_index, line in enumerate(lines):
        timestamp = _timestamp(line.strip(), sample["utc_offset_seconds"])
        if timestamp is not None and timestamp < process["started_at"] - 2:
            continue
        if re.search(r"\btraining (?:complete|finished)\b|\bearly stopping\b", line, re.I):
            training_finished = True
        epoch = EPOCH.search(line)
        if epoch and epoch[2]:
            epochs = int(epoch[2])
        duration = EPOCH_DURATION.search(line)
        if (
            epoch
            and duration
            and not re.search(r"\b(?:val|valid|validation|eval|evaluation|test)\b", line, re.I)
            and not re.search(r"\b(?:step|batch|microbatch)s?\s*[:=]\s*\d+\s*/", line, re.I)
        ):
            point = (int(epoch[1]), float(duration[1]), line_index)
            if epoch_durations and point[0] < epoch_durations[-1][0]:
                epoch_durations.clear()
            if (
                epoch_durations
                and point[0] == epoch_durations[-1][0]
                and point[1] != epoch_durations[-1][1]
            ):
                return unknown("同一 epoch 有多组耗时，无法确认属于同一训练")
            if not epoch_durations or point[0] != epoch_durations[-1][0]:
                epoch_durations.append(point)
        # A header saying "Epoch 3" does not prove that epoch 3 has finished.
        if (
            epoch
            and timestamp is not None
            and re.search(r"Average Loss|train time|\bdone\b|\bcompleted\b", line, re.I)
        ):
            point = (int(epoch[1]), timestamp)
            if epoch_points and point[0] < epoch_points[-1][0]:
                epoch_points.clear()
            if not epoch_points or point[0] != epoch_points[-1][0]:
                epoch_points.append(point)
        step = STEP.search(line)
        if step and timestamp is not None:
            point = (int(step[2]), timestamp, int(step[3]), step[1].lower())
            if step_points and (point[0] < step_points[-1][0] or point[2] != step_points[-1][2]):
                step_points.clear()
            if not step_points or point[0] != step_points[-1][0]:
                step_points.append(point)
        progress_bar = BAR.search(line)
        if progress_bar:
            bar = (progress_bar, timestamp, line_index)

    if training_finished:
        return unknown("日志显示训练已结束，进程可能仍在收尾")
    current, total, seconds, observed, scope, basis = 0, 0, 0.0, modified, "phase", ""
    if epochs and len(epoch_points) >= 2:
        current, observed = epoch_points[-1]
        total = epochs
        cycles = [
            (b[1] - a[1]) / (b[0] - a[0])
            for a, b in zip(epoch_points, epoch_points[1:], strict=False)
            if b[1] > a[1]
        ]
        if cycles:
            seconds = statistics.median(cycles[-3:])
        scope = "training"
        basis = f"已完成 {current}/{total} epoch · 近期约 {seconds / 60:.1f} 分/轮"
    elif epochs and epoch_durations and epoch_durations[-1][2] == len(lines) - 1:
        current = epoch_durations[-1][0]
        total = epochs
        seconds = statistics.median([point[1] for point in epoch_durations[-3:]])
        scope = "training"
        basis = f"已完成 {current}/{total} epoch · 日志报告约 {seconds / 60:.1f} 分/轮"
    elif len(step_points) >= 2:
        current, observed, total, name = step_points[-1]
        cycles = [
            (b[1] - a[1]) / (b[0] - a[0])
            for a, b in zip(step_points, step_points[1:], strict=False)
            if b[1] > a[1]
        ]
        if cycles:
            seconds = statistics.median(cycles[-5:])
        scope = "training" if name.startswith("global") or total_steps == total else "phase"
        basis = f"{current}/{total} step · 近期约 {seconds:.2f} 秒/步"
    elif bar:
        match, timestamp, line_index = bar
        if timestamp is None and line_index != len(lines) - 1:
            return unknown("进度条之后还有输出，无法确定进度更新时间")
        current, total = int(match[1]), int(match[2])
        speed = float(match[5])
        seconds = (1 / speed if match[6].lower() == "it/s" else speed) if speed > 0 else 0
        # Untimestamped bars are anchored to the file update, never the time of each poll.
        observed = timestamp if timestamp is not None else modified
        if total_steps == total and re.search(r"train|global", lines[line_index], re.I):
            scope = "training"
        basis = f"进度条 {current}/{total} · {speed:g} {match[6]}"
    else:
        return unknown("日志中还没有足够的可识别进度或速度")
    if observed is None:
        return unknown("日志缺少可信的更新时间，无法估计")
    if current >= total:
        return unknown("已完成当前进度，等待进程结束或下一阶段")
    if (
        current < 0
        or total <= 0
        or not math.isfinite(seconds)
        or seconds <= 0
        or observed > now + 5
    ):
        return unknown("进度信息不足或时间异常")
    stale_after = max(600, seconds * 3) if scope == "training" and epochs else 600
    if now - observed > stale_after:
        return unknown("训练进度已长时间未更新")
    end = observed + (total - current) * seconds
    if end <= now:
        return unknown("已超过日志推算时间，等待新进度")
    return {
        "label": "日志估计" if scope == "training" else "阶段估计",
        "source_kind": "logs",
        "end_at": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "confidence": "low",
        "scope": scope,
        "basis": basis,
        "source": path,
        "observed_at": datetime.fromtimestamp(observed, timezone.utc).isoformat(),
        "sampled_at": datetime.fromtimestamp(sample["sampled_at"], timezone.utc).isoformat(),
        "reason": (
            "当前训练估计，不包含后续任务" if scope == "training" else "仅当前阶段，总训练时长未知"
        ),
    }
