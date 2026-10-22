"""Web-only, short-lived log sampling and progress estimates."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from gpuops.monitor.service import ServerResult
from gpuops.ssh import SSHClient
from gpuops.web.progress import estimate, unknown

SAMPLE_SCRIPT = Path(__file__).with_name("_log_sample.py").read_text()


class LogEtaCollector:
    def __init__(self, *, ssh: SSHClient | None = None, timeout_seconds: float = 5.0):
        self.ssh = ssh or SSHClient()
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[float, list[dict], dict]] = {}
        self._lock = asyncio.Lock()

    async def estimates(self, results: list[ServerResult]) -> dict[str, dict[int, dict]]:
        async with self._lock:
            rows = await asyncio.gather(*(self._host(result) for result in results))
        return dict(rows)

    async def _host(self, result: ServerResult) -> tuple[str, dict[int, dict]]:
        server = result.server.id
        requests_by_pid = {
            process["pid"]: {"pid": process["pid"], "start_time_ticks": process["start_time_ticks"]}
            for gpu in (result.data or {}).get("gpus", [])
            for process in gpu.get("processes") or []
            if process.get("pid") and process.get("start_time_ticks") is not None
        }
        requests = sorted(requests_by_pid.values(), key=lambda value: value["pid"])[:32]
        if not result.online or not requests:
            self._cache.pop(server, None)
            return server, {}
        cached = self._cache.get(server)
        if cached is None or cached[1] != requests or time.monotonic() - cached[0] >= 30:
            try:
                script = "import json\nREQUESTS = json.loads(" + repr(json.dumps(requests)) + ")\n"
                script += SAMPLE_SCRIPT + "\nprint(json.dumps(collect(REQUESTS)))\n"
                response = await self.ssh.run_python(
                    result.server.ssh_host, script, timeout_seconds=self.timeout_seconds
                )
                if response.returncode != 0:
                    raise ValueError("log sampling failed")
                sample = json.loads(response.stdout)
            except Exception:
                sample = {"processes": [], "logs": {}}
            self._cache[server] = (time.monotonic(), requests, sample)
        sample = self._cache[server][2]
        estimates = {request["pid"]: unknown("日志采样暂不可用") for request in requests}
        for process in sample["processes"]:
            if process.get("error"):
                estimates[process["pid"]] = unknown("进程已变化或无法读取")
                continue
            candidates = []
            for path in process["paths"]:
                try:
                    candidates.append(estimate(sample, process, path, now=time.time()))
                except (ValueError, OverflowError):
                    candidates.append(unknown("无法解析日志中的进度数值"))
            candidates.sort(
                key=lambda value: (
                    value.get("format") == "gpuops_progress",
                    value.get("end_at") is not None,
                    value.get("scope") == "training",
                    value.get("observed_at", ""),
                ),
                reverse=True,
            )
            estimates[process["pid"]] = (
                candidates[0]
                if candidates
                else unknown(
                    "日志路径不可读取"
                    if process.get("discovery_errors")
                    else "未找到可读的训练日志"
                )
            )
        return server, estimates
