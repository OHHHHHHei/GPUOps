import json
from datetime import datetime, timezone

import pytest

from gpuops.web.progress import estimate

NOW = 2_000_000_000.0
PATH = "/logs/train.log"


def line(
    phase="train",
    current=10,
    total=100,
    elapsed_seconds=60.0,
    timestamp=NOW - 60,
):
    return "GPUOPS_PROGRESS " + json.dumps(
        {
            "phase": phase,
            "current": current,
            "total": total,
            "elapsed_seconds": elapsed_seconds,
            "timestamp": timestamp,
        }
    )


def predict(content, *, now=NOW, modified_at=None, started_at=NOW - 3600):
    sample = {
        "sampled_at": NOW,
        "utc_offset_seconds": 0,
        "logs": {PATH: {"content": content, "modified_at": modified_at}},
    }
    process = {"argv": ["python", "train.py"], "started_at": started_at}
    return estimate(sample, process, PATH, now=now)


def test_cumulative_training_records_use_recent_median_and_record_timestamp():
    content = "\n".join(
        [
            line(current=10, elapsed_seconds=50, timestamp=NOW - 150),
            line(current=20, elapsed_seconds=90, timestamp=NOW - 110),
            # A batch counter may reset between epochs; the explicit current remains cumulative.
            "epoch=2 batch=0/50",
            line(current=30, elapsed_seconds=150, timestamp=NOW - 50),
        ]
    )
    result = predict(content, modified_at=NOW + 10_000)
    assert result["format"] == "gpuops_progress"
    assert result["scope"] == "training"
    assert result["progress"] == {"phase": "train", "current": 30, "total": 100}
    assert result["observed_at"] == datetime.fromtimestamp(NOW - 50, timezone.utc).isoformat()
    # Median of 4 and 6 seconds/unit, anchored to the record timestamp.
    assert result["end_at"] == datetime.fromtimestamp(NOW + 300, timezone.utc).isoformat()
    assert "5.00 秒/单位" in result["basis"]


def test_polling_and_unrelated_file_updates_do_not_move_structured_eta():
    content = "\n".join(
        [
            line(current=10, elapsed_seconds=10, timestamp=NOW - 80),
            line(current=20, elapsed_seconds=20, timestamp=NOW - 70),
        ]
    )
    first = predict(content, modified_at=NOW - 5)
    later = predict(content, now=NOW + 5, modified_at=NOW + 20)
    assert first["end_at"] == later["end_at"]


@pytest.mark.parametrize(
    "boundary",
    [
        line(phase="validation", current=1, total=10, elapsed_seconds=1, timestamp=NOW - 20),
        line(current=1, total=200, elapsed_seconds=1, timestamp=NOW - 20),
        line(current=1, elapsed_seconds=1, timestamp=NOW - 20),
        line(current=30, elapsed_seconds=1, timestamp=NOW - 20),
    ],
)
def test_phase_total_and_elapsed_or_counter_resets_start_a_new_series(boundary):
    content = "\n".join(
        [
            line(current=10, elapsed_seconds=10, timestamp=NOW - 40),
            line(current=20, elapsed_seconds=20, timestamp=NOW - 30),
            boundary,
        ]
    )
    result = predict(content)
    assert result["format"] == "gpuops_progress"
    assert result["end_at"] is None
    assert "至少需要两次" in result["reason"]


def test_two_reports_after_a_reset_form_a_new_estimate():
    content = "\n".join(
        [
            line(current=80, elapsed_seconds=800, timestamp=NOW - 50),
            line(current=5, elapsed_seconds=5, timestamp=NOW - 20),
            line(current=10, elapsed_seconds=10, timestamp=NOW - 10),
        ]
    )
    result = predict(content)
    assert result["end_at"] == datetime.fromtimestamp(NOW + 80, timezone.utc).isoformat()


@pytest.mark.parametrize(
    "latest,reason",
    [
        (line(total=None), "总量未知"),
        (line(current=100, total=100), "当前阶段已完成"),
        (line(timestamp=NOW - 601), "超过 10 分钟"),
        ("GPUOPS_PROGRESS {bad json", "最新 GPUOps 结构化进度记录无效"),
    ],
)
def test_unknown_complete_stale_or_invalid_latest_never_revives_an_older_eta(latest, reason):
    content = "\n".join(
        [
            line(current=10, elapsed_seconds=10, timestamp=NOW - 100),
            line(current=20, elapsed_seconds=20, timestamp=NOW - 90),
            latest,
        ]
    )
    result = predict(content)
    assert result["format"] == "gpuops_progress"
    assert result["end_at"] is None
    assert reason in result["reason"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", ""),
        ("phase", "x" * 81),
        ("current", True),
        ("current", -1),
        ("total", 0),
        ("total", 9),
        ("elapsed_seconds", True),
        ("elapsed_seconds", float("nan")),
        ("timestamp", float("inf")),
        ("timestamp", 1e300),
    ],
)
def test_malformed_numeric_and_boolean_fields_do_not_crash(field, value):
    payload = {
        "phase": "train",
        "current": 10,
        "total": 100,
        "elapsed_seconds": 10,
        "timestamp": NOW - 10,
    }
    payload[field] = value
    result = predict("GPUOPS_PROGRESS " + json.dumps(payload))
    assert result["format"] == "gpuops_progress"
    assert result["end_at"] is None


def test_structured_timestamps_must_belong_to_current_process_and_sample():
    old = predict(line(timestamp=NOW - 100), started_at=NOW)
    assert old["end_at"] is None
    assert "早于当前进程" in old["reason"]

    future = predict(line(timestamp=NOW + 6))
    assert future["end_at"] is None
    assert "时间异常" in future["reason"]


@pytest.mark.parametrize("timestamp", [NOW - 4000, NOW + 6])
def test_rejected_timestamps_cannot_display_a_completed_phase(timestamp):
    result = predict(line(current=100, total=100, timestamp=timestamp))
    assert result["end_at"] is None
    assert "progress" not in result


def test_legacy_parser_is_used_only_when_no_structured_prefix_exists():
    first = datetime.fromtimestamp(NOW - 20, timezone.utc).isoformat()
    second = datetime.fromtimestamp(NOW - 10, timezone.utc).isoformat()
    content = f"{first} global_step 10/100\n{second} global_step 20/100"
    result = predict(content)
    assert result["end_at"] is not None
    assert "format" not in result


def test_similar_text_without_the_exact_prefix_does_not_claim_the_protocol():
    result = predict("GPUOPS_PROGRESSIVE " + line().split(" ", 1)[1])
    assert result["end_at"] is None
    assert "format" not in result
