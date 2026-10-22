"""Human-readable terminal rendering for monitoring results."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Any

from gpuops.core.resources import DEFAULT_RESOURCE_ELIGIBILITY_POLICY, telemetry_from_mapping

from .service import ServerResult


def format_status_table(results: Iterable[ServerResult]) -> str:
    headers = (
        "SERVER",
        "GPU",
        "STATE",
        "UTIL",
        "MEMORY",
        "USER",
        "PID",
        "COMMAND",
        "ELAPSED",
        "ETA",
    )
    rows: list[tuple[str, ...]] = []
    for result in results:
        if not result.online or result.data is None:
            rows.append(
                (
                    result.server.label,
                    "-",
                    "offline" if result.error_code in {None, "connection_failed"} else "unknown",
                    "-",
                    "-",
                    "-",
                    "-",
                    _truncate(result.error or "unknown error", 36),
                    "-",
                    "-",
                )
            )
            continue
        for gpu in result.data.get("gpus", []):
            rows.extend(_gpu_rows(result.server.label, gpu))
    return _render_table(headers, rows)


def format_doctor_table(results: Iterable[ServerResult]) -> str:
    headers = ("SERVER", "SSH", "PYTHON", "GPU SAMPLE", "DETAIL")
    rows = []
    for result in results:
        data = result.data or {}
        rows.append(
            (
                result.server.label,
                "ok" if result.online else "failed",
                str(data.get("python", "-")),
                "ok" if data.get("gpustat_ok") else "failed",
                _truncate(str(data.get("error") or result.error or ""), 56),
            )
        )
    return _render_table(headers, rows)


def _gpu_rows(server: str, gpu: dict[str, Any]) -> list[tuple[str, ...]]:
    raw_processes = gpu.get("processes")
    processes = raw_processes if isinstance(raw_processes, list) else []
    state = _idle_state(gpu)
    used = gpu.get("memory.used", "?")
    total = gpu.get("memory.total", "?")
    common = (
        server,
        str(gpu.get("index", "?")),
        state,
        _percent(gpu.get("utilization.gpu")),
        f"{used}/{total} MiB",
    )
    if not processes:
        return [common + ("-", "-", "-", "-", "-")]
    rows = []
    for position, process in enumerate(processes):
        prefix = common if position == 0 else ("", "", "", "", "")
        rows.append(
            prefix
            + (
                str(process.get("username", "?")),
                str(process.get("pid", "?")),
                _truncate(str(process.get("cmdline") or process.get("command") or "?"), 36),
                _duration(process.get("elapsed_seconds")),
                "unknown",
            )
        )
    return rows


def _idle_state(gpu: dict[str, Any]) -> str:
    eligibility = DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(telemetry_from_mapping(gpu))
    return "occupied" if eligibility.value == "busy" else eligibility.value


def _percent(value: Any) -> str:
    return f"{value}%" if value is not None else "?"


def _duration(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    seconds = max(0, int(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


def _render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        rows = [("-",) * len(headers)]
    widths = [
        max(_display_width(headers[index]), *(_display_width(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            value + " " * (widths[index] - _display_width(value)) for index, value in enumerate(row)
        ).rstrip()

    separator = render(tuple("-" * width for width in widths))
    return "\n".join((render(headers), separator, *(render(row) for row in rows)))


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1 for character in value
    )
