"""Data boundary used by the local web console."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol

from gpuops.config import SchedulerSettings
from gpuops.core.resources import (
    DEFAULT_RESOURCE_ELIGIBILITY_POLICY,
    normalize_nonnegative_number,
    telemetry_from_mapping,
)
from gpuops.history import decode_cursor, encode_cursor
from gpuops.monitor import MonitorService
from gpuops.monitor.service import ServerResult
from gpuops.web.log_eta import LogEtaCollector

JsonObject = dict[str, Any]

# The host validates sample age on its own clock; allow small host/Mac skew
# only when deciding whether to display that sample as current.
WEB_CLOCK_SKEW_SECONDS = 5.0


class DashboardProvider(Protocol):
    """Minimum service contract required by the web console."""

    async def get_snapshot(self) -> JsonObject:
        """Return the current multi-server GPU snapshot."""

    async def get_process_etas(self) -> JsonObject:
        """Return log estimates separately from live GPU telemetry."""

    async def get_queue(self) -> JsonObject:
        """Return queued and running jobs."""

    async def get_history(
        self, *, limit: int = 50, project: str | None = None, cursor: str | None = None
    ) -> JsonObject:
        """Return one page of completed jobs and available projects."""

    async def get_job(self, job_id: str) -> JsonObject | None:
        """Return one job including recent logs, or None when absent."""

    async def cancel_job(self, job_id: str) -> JsonObject | None:
        """Request cancellation of one managed job, or return None when absent."""

    async def health(self) -> JsonObject:
        """Return provider and upstream health information."""


class ProviderUnavailable(RuntimeError):
    """The requested provider operation cannot currently be completed."""


class ProviderConflict(RuntimeError):
    """Remote state conflicts with a unique job operation."""


class LiveProvider:
    """Translate live SSH snapshots into the stable dashboard contract."""

    def __init__(
        self,
        monitor: MonitorService | None = None,
        *,
        cache_seconds: float = 15.0,
        expected_gpu_count: int | None = None,
        log_eta: LogEtaCollector | None = None,
    ) -> None:
        self.monitor = monitor or MonitorService()
        self.cache_seconds = cache_seconds
        self.expected_gpu_count = expected_gpu_count
        self.log_eta = log_eta
        self._lock = asyncio.Lock()
        self._cached_at = 0.0
        self._cached_results: list[ServerResult] | None = None
        self._last_good_data: dict[str, JsonObject] = {}
        self._seen_gpu_indices: dict[str, set[int]] = {}

    async def _results(self) -> list[ServerResult]:
        cached = self._cached_if_fresh()
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._cached_if_fresh()
            if cached is not None:
                return cached
            self._cached_results = await self.monitor.collect_all()
            self._cached_results = self._fresh_results()
            self._cached_results = [
                self._preserve_and_remember_card_data(result)
                for result in self._cached_results
            ]
            self._cached_at = time.monotonic()
            return self._cached_results

    def _cached_if_fresh(self) -> list[ServerResult] | None:
        if self._cached_results is None:
            return None
        current_targets = getattr(self.monitor, "targets", None)
        if current_targets is not None:
            current_targets = tuple(current_targets)
            cached_targets = tuple(result.server for result in self._cached_results)
            if current_targets != cached_targets:
                current_by_id = {target.id: target for target in current_targets}
                for previous in cached_targets:
                    current = current_by_id.get(previous.id)
                    if current is None or current.ssh_host != previous.ssh_host:
                        self._last_good_data.pop(previous.id, None)
                        self._seen_gpu_indices.pop(previous.id, None)
                return None
        ttl = self.cache_seconds
        if any(
            not result.online or self._result_has_card_errors(result)
            for result in self._cached_results
        ):
            # Coalesce concurrent endpoints, but retry before the next browser poll.
            ttl = min(ttl, 1.0)
        now = datetime.now(timezone.utc)
        if time.monotonic() - self._cached_at < ttl and all(
            not result.online or self._sample_is_fresh(result, now)
            for result in self._cached_results
        ):
            return self._cached_results
        # Read the host cache before declaring an aged local sample unavailable.
        return None

    @staticmethod
    def _sample_is_fresh(result: ServerResult, now: datetime) -> bool:
        data = result.data or {}
        collected_at = data.get("collected_at")
        if collected_at is not None and not LiveProvider._timestamp_is_fresh(collected_at, now):
            return False
        for gpu in data.get("gpus", []):
            if LiveProvider._gpu_has_error(gpu) or "sampled_at" not in gpu:
                continue
            if not LiveProvider._timestamp_is_fresh(gpu.get("sampled_at"), now):
                return False
        return True

    @staticmethod
    def _timestamp_is_fresh(value: Any, now: datetime) -> bool:
        try:
            age = (now - datetime.fromisoformat(str(value))).total_seconds()
            max_age = SchedulerSettings().max_snapshot_age_seconds + WEB_CLOCK_SKEW_SECONDS
            return -WEB_CLOCK_SKEW_SECONDS <= age <= max_age
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _gpu_has_error(gpu: JsonObject) -> bool:
        return bool(gpu.get("error") or gpu.get("error_code"))

    @classmethod
    def _result_has_card_errors(cls, result: ServerResult) -> bool:
        return any(cls._gpu_has_error(gpu) for gpu in (result.data or {}).get("gpus", []))

    def _fresh_results(self) -> list[ServerResult]:
        """Reject samples that expired while reading the fleet's shared caches."""
        results = []
        now = datetime.now(timezone.utc)
        for result in self._cached_results or []:
            if result.online and isinstance(result.data, dict):
                stale_indices = {
                    gpu.get("index")
                    for gpu in result.data.get("gpus", [])
                    if not self._gpu_has_error(gpu)
                    and "sampled_at" in gpu
                    and not self._timestamp_is_fresh(gpu.get("sampled_at"), now)
                }
                if stale_indices:
                    data = deepcopy(result.data)
                    for gpu in data.get("gpus", []):
                        if gpu.get("index") not in stale_indices:
                            continue
                        gpu["error"] = "GPU sample timestamp is stale or invalid"
                        gpu["error_code"] = "telemetry_stale"
                    result = replace(result, data=data)
            if result.online and not self._sample_is_fresh(result, now):
                result = replace(
                    result, online=False, data=None, error_code="telemetry_stale",
                    error="The last Worker GPU sample is stale; waiting for a new sample.",
                )
            results.append(result)
        return results

    def _preserve_and_remember_card_data(self, result: ServerResult) -> ServerResult:
        if not result.online or not isinstance(result.data, dict):
            return result
        has_card_errors = self._result_has_card_errors(result)
        data = deepcopy(result.data) if has_card_errors else result.data
        previous = self._last_good_data.get(result.server.id, {})
        previous_gpus = {
            gpu.get("index"): gpu for gpu in previous.get("gpus", [])
            if isinstance(gpu, dict)
        }
        remembered = deepcopy(previous) if has_card_errors else deepcopy(data)
        remembered["collected_at"] = data.get("collected_at")
        remembered_gpus = dict(previous_gpus)
        rendered_gpus = []
        for gpu in data.get("gpus", []):
            index = gpu.get("index")
            if type(index) is int and index >= 0:
                self._seen_gpu_indices.setdefault(result.server.id, set()).add(index)
            if self._gpu_has_error(gpu):
                old = previous_gpus.get(gpu.get("index"))
                if old is not None:
                    preserved = deepcopy(old)
                    preserved.update(
                        {key: value for key, value in gpu.items() if value is not None}
                    )
                    if not gpu.get("processes"):
                        preserved["processes"] = deepcopy(old.get("processes"))
                    gpu = preserved
            else:
                remembered_gpus[gpu.get("index")] = deepcopy(gpu)
            rendered_gpus.append(gpu)
        if has_card_errors:
            data["gpus"] = rendered_gpus
        remembered["gpus"] = list(remembered_gpus.values())
        self._last_good_data[result.server.id] = remembered
        return replace(result, data=data) if has_card_errors else result

    async def get_snapshot(self) -> JsonObject:
        results = await self._results()
        generated_at = _now_iso()
        servers = [self._server_snapshot(result) for result in results]
        statuses = [gpu["status"] for server in servers for gpu in server["gpus"]]
        summary = {
            "total": len(statuses),
            "idle": statuses.count("idle"),
            "busy": statuses.count("busy"),
            "residual": statuses.count("residual"),
            "unknown": statuses.count("unknown"),
            "offline": statuses.count("offline"),
        }
        return {
            "generated_at": generated_at,
            "summary": summary,
            "servers": servers,
        }

    async def get_process_etas(self) -> JsonObject:
        results = await self._results()
        estimates = await self.log_eta.estimates(results) if self.log_eta else {}
        return {
            "updated_at": _now_iso(),
            "servers": [
                {
                    "id": result.server.id,
                    "processes": [
                        {
                            "pid": process["pid"],
                            "start_time_ticks": process["start_time_ticks"],
                            "eta": estimates.get(result.server.id, {}).get(process["pid"], {}),
                        }
                        for gpu in (result.data or {}).get("gpus", [])
                        if not self._gpu_has_error(gpu)
                        for process in gpu.get("processes") or []
                        if process.get("start_time_ticks") is not None
                    ],
                }
                for result in results
            ],
        }

    async def get_queue(self) -> JsonObject:
        return {"updated_at": _now_iso(), "mode": "monitor-only", "items": []}

    async def get_history(
        self, *, limit: int = 50, project: str | None = None, cursor: str | None = None
    ) -> JsonObject:
        decode_cursor(cursor)
        return {
            "updated_at": _now_iso(),
            "mode": "monitor-only",
            "items": [],
            "servers": [],
            "projects": [],
            "next_cursor": None,
        }

    async def get_job(self, job_id: str) -> JsonObject | None:
        return None

    async def cancel_job(self, job_id: str) -> JsonObject | None:
        del job_id
        raise ProviderUnavailable("job control is unavailable in monitor-only mode")

    async def health(self) -> JsonObject:
        results = await self._results()
        servers = [
            {
                "id": result.server.id,
                "status": self._server_snapshot(result)["status"],
            }
            for result in results
        ]
        health = {
            "status": (
                "ok" if servers and all(item["status"] == "online" for item in servers)
                else "degraded"
            ),
            "mode": "live",
            "checked_at": _now_iso(),
            "servers": servers,
        }
        if not servers:
            health["reason"] = "no_servers_registered"
        return health

    def _server_snapshot(self, result: ServerResult) -> JsonObject:
        error_code = result.error_code
        failed = not result.online
        stale_data = self._last_good_data.get(result.server.id) if failed else None
        data = result.data if isinstance(result.data, dict) else stale_data or {}
        stale = stale_data is not None
        connection_failed = failed and (error_code == "connection_failed" or error_code is None)
        failed_gpu_status = "offline" if connection_failed else "unknown"
        raw_gpus = {
            gpu["index"]: gpu for gpu in data.get("gpus", [])
            if isinstance(gpu, dict) and type(gpu.get("index")) is int and gpu["index"] >= 0
        }
        indices = (
            range(self.expected_gpu_count)
            if self.expected_gpu_count is not None
            else sorted(set(raw_gpus) | self._seen_gpu_indices.get(result.server.id, set()))
        )
        gpus = [
            self._gpu_snapshot(
                index,
                raw_gpus.get(index),
                result.error,
                forced_status=(
                    failed_gpu_status
                    if failed
                    else "unknown"
                    if self.expected_gpu_count is None and index not in raw_gpus
                    else None
                ),
            )
            for index in indices
        ]
        online_count = sum(gpu["status"] != "offline" for gpu in gpus)
        has_unknown = any(gpu["status"] == "unknown" for gpu in gpus)
        if failed:
            status = "offline" if connection_failed else "degraded"
        elif (not gpus or online_count == len(gpus)) and not has_unknown:
            status = "online"
        elif online_count:
            status = "degraded"
        else:
            status = "offline"
        return {
            "id": result.server.id,
            "label": result.server.label,
            "status": status,
            "stale": stale,
            "last_seen": data.get("collected_at") if (result.online or stale) else None,
            "error": result.error,
            "error_code": error_code,
            "gpus": gpus,
        }

    @staticmethod
    def _gpu_snapshot(
        index: int,
        raw: JsonObject | None,
        server_error: str | None,
        *,
        forced_status: str | None = None,
    ) -> JsonObject:
        if raw is None:
            return {
                "id": f"gpu-{index}",
                "index": index,
                "name": "Unknown",
                "status": forced_status or "offline",
                "utilization_percent": None,
                "memory_used_mib": None,
                "memory_total_mib": None,
                "temperature_c": None,
                "power_watts": None,
                "processes": [],
                "error": server_error or "GPU data unavailable",
            }

        raw_processes = raw.get("processes")
        processes = raw_processes if isinstance(raw_processes, list) else []
        telemetry = telemetry_from_mapping(raw)
        card_error = raw.get("error")
        card_error_code = raw.get("error_code")
        status = (
            forced_status
            or ("unknown" if card_error or card_error_code else None)
            or DEFAULT_RESOURCE_ELIGIBILITY_POLICY.classify(telemetry).value
        )
        return {
            "id": f"gpu-{index}",
            "index": index,
            "name": raw.get("name") or "Unknown",
            "status": status,
            "utilization_percent": telemetry.utilization_percent,
            "memory_used_mib": telemetry.memory_used_mib,
            "memory_total_mib": normalize_nonnegative_number(
                raw.get("memory.total"),
                integral=True,
            ),
            "temperature_c": normalize_nonnegative_number(raw.get("temperature.gpu")),
            "power_watts": normalize_nonnegative_number(raw.get("power.draw")),
            "processes": [LiveProvider._process_snapshot(process) for process in processes],
            "sampled_at": raw.get("sampled_at"),
            "error": card_error,
            "error_code": card_error_code,
        }

    @staticmethod
    def _process_snapshot(process: JsonObject) -> JsonObject:
        return {
            "pid": normalize_nonnegative_number(process.get("pid"), integral=True),
            "user": process.get("username") or "unknown",
            "command": process.get("cmdline") or process.get("command") or "unknown",
            "elapsed_seconds": normalize_nonnegative_number(process.get("elapsed_seconds")),
            "gpu_memory_mib": normalize_nonnegative_number(
                process.get("gpu_memory_usage"),
                integral=True,
            ),
            "start_time_ticks": process.get("start_time_ticks"),
            "job_id": None,
            "eta": {"label": "未知", "end_at": None, "confidence": "unknown"},
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DemoProvider:
    """Deterministic in-memory data for development and visual checks."""

    _NOW = "2026-09-19T14:32:00+08:00"

    def __init__(self) -> None:
        self._snapshot = self._build_snapshot()
        self._queue = self._build_queue()
        self._jobs = self._build_jobs()

    async def get_snapshot(self) -> JsonObject:
        return deepcopy(self._snapshot)

    async def get_process_etas(self) -> JsonObject:
        return {"updated_at": self._NOW, "servers": []}

    async def get_queue(self) -> JsonObject:
        payload = deepcopy(self._queue)
        for item in payload["items"]:
            item.pop("command", None)
        return payload

    async def get_history(
        self, *, limit: int = 50, project: str | None = None, cursor: str | None = None
    ) -> JsonObject:
        before = decode_cursor(cursor)
        items = [
            deepcopy(job)
            for job in self._jobs.values()
            if job["state"] in {"finished", "failed", "cancelled"}
        ]
        projects = sorted({item["project"] for item in items})

        def key(item: JsonObject) -> tuple[datetime, str, str]:
            return datetime.fromisoformat(item["finished_at"]), item["id"], item["server"]

        items = sorted(
            (
                item
                for item in items
                if (project is None or item["project"] == project)
                and (before is None or key(item) < before)
            ),
            key=key,
            reverse=True,
        )
        page = items[:limit]
        next_cursor = encode_cursor(key(page[-1])) if len(items) > limit else None
        for item in page:
            for field in (
                "command",
                "logs",
                "events",
                "working_directory",
                "created_by",
                "progress",
            ):
                item.pop(field, None)
        return {
            "updated_at": self._NOW,
            "mode": "demo",
            "items": page,
            "projects": projects,
            "next_cursor": next_cursor,
            "servers": [],
        }

    async def get_job(self, job_id: str) -> JsonObject | None:
        job = self._jobs.get(job_id)
        return deepcopy(job) if job is not None else None

    async def cancel_job(self, job_id: str) -> JsonObject | None:
        if job_id not in self._jobs:
            return None
        raise ProviderUnavailable("job control is unavailable in demo mode")

    async def health(self) -> JsonObject:
        return {
            "status": "ok",
            "mode": "demo",
            "checked_at": self._NOW,
            "servers": [
                {"id": "gpu01", "status": "online"},
                {"id": "gpu02", "status": "degraded"},
            ],
        }

    @classmethod
    def _process(
        cls,
        pid: int,
        user: str,
        command: str,
        elapsed_seconds: int,
        gpu_memory_mib: int,
        *,
        job_id: str | None = None,
        eta_label: str = "未知",
        eta_end_at: str | None = None,
        eta_confidence: str = "unknown",
    ) -> JsonObject:
        return {
            "pid": pid,
            "user": user,
            "command": command,
            "elapsed_seconds": elapsed_seconds,
            "gpu_memory_mib": gpu_memory_mib,
            "job_id": job_id,
            "eta": {
                "label": eta_label,
                "end_at": eta_end_at,
                "confidence": eta_confidence,
            },
        }

    @classmethod
    def _gpu(
        cls,
        index: int,
        status: str,
        utilization: int | None,
        memory_used: int | None,
        *,
        processes: list[JsonObject] | None = None,
        temperature: int | None = 37,
        error: str | None = None,
    ) -> JsonObject:
        return {
            "id": f"gpu-{index}",
            "index": index,
            "name": "NVIDIA GeForce RTX 3090",
            "status": status,
            "utilization_percent": utilization,
            "memory_used_mib": memory_used,
            "memory_total_mib": None if status == "offline" else 24576,
            "temperature_c": temperature,
            "power_watts": None if utilization is None else round(29.0 + utilization * 2.1, 1),
            "processes": processes or [],
            "error": error,
        }

    @classmethod
    def _build_snapshot(cls) -> JsonObject:
        ddp_command = (
            "python -m torch.distributed.run --nproc_per_node=4 train.py "
            "--config configs/ranking.yaml --seed 42"
        )
        ddp_processes = [
            [
                cls._process(
                    42110 + gpu,
                    "researcher",
                    ddp_command,
                    5_520,
                    19_340,
                    job_id="run-ranking-042",
                    eta_label="约 16:10",
                    eta_end_at="2026-09-19T16:10:00+08:00",
                    eta_confidence="medium",
                )
            ]
            for gpu in range(4)
        ]
        gpu01_gpus = [
            cls._gpu(
                index,
                "busy",
                88 + index * 2,
                19_700 + index * 110,
                processes=ddp_processes[index],
                temperature=67 + index,
            )
            for index in range(4)
        ]
        gpu01_gpus.extend(
            cls._gpu(index, "idle", 0, 318 + index, temperature=34 + index % 2)
            for index in range(4, 7)
        )
        gpu01_gpus.append(cls._gpu(7, "residual", 0, 3_284, temperature=39))

        eval_process = cls._process(
            88302,
            "researcher",
            "python evaluate.py --checkpoint outputs/final.pt --split test",
            1_860,
            8_412,
            eta_label="未知",
        )
        sweep_process = cls._process(
            91173,
            "researcher",
            "python scripts/sweep.py --dataset synthetic --trial 7",
            10_860,
            21_120,
            job_id="run-sweep-007",
            eta_label="约 15:25",
            eta_end_at="2026-09-19T15:25:00+08:00",
            eta_confidence="high",
        )
        gpu02_gpus = [
            cls._gpu(0, "idle", 0, 316, temperature=33),
            cls._gpu(1, "busy", 76, 8_730, processes=[eval_process], temperature=61),
            cls._gpu(2, "busy", 96, 21_438, processes=[sweep_process], temperature=72),
            cls._gpu(3, "idle", 1, 318, temperature=35),
            cls._gpu(4, "idle", 0, 316, temperature=34),
            cls._gpu(5, "idle", 0, 317, temperature=34),
            cls._gpu(6, "idle", 0, 315, temperature=33),
            cls._gpu(
                7,
                "offline",
                None,
                None,
                temperature=None,
                error="NVML device query timed out",
            ),
        ]
        return {
            "generated_at": cls._NOW,
            "summary": {
                "total": 16,
                "idle": 7,
                "busy": 7,
                "residual": 1,
                "unknown": 0,
                "offline": 1,
            },
            "servers": [
                {
                    "id": "gpu01",
                    "label": "GPU 01",
                    "status": "online",
                    "last_seen": cls._NOW,
                    "error": None,
                    "gpus": gpu01_gpus,
                },
                {
                    "id": "gpu02",
                    "label": "GPU 02",
                    "status": "degraded",
                    "last_seen": cls._NOW,
                    "error": "GPU 7 暂时无法读取，其他设备正常",
                    "gpus": gpu02_gpus,
                },
            ],
        }

    @classmethod
    def _build_queue(cls) -> JsonObject:
        return {
            "updated_at": cls._NOW,
            "items": [
                {
                    "id": "run-ranking-042",
                    "name": "Ranking baseline / seed 42",
                    "project": "ranking-study",
                    "state": "running",
                    "server": "gpu01",
                    "gpus": [0, 1, 2, 3],
                    "priority": "normal",
                    "submitted_at": "2026-09-19T12:58:00+08:00",
                    "estimated_start": "2026-09-19T13:00:00+08:00",
                    "estimated_end": "2026-09-19T16:10:00+08:00",
                    "eta_confidence": "medium",
                    "command": (
                        "python -m torch.distributed.run --nproc_per_node=4 train.py "
                        "--config configs/ranking.yaml --seed 42"
                    ),
                },
                {
                    "id": "run-sweep-007",
                    "name": "Synthetic data sweep / trial 7",
                    "project": "benchmark-study",
                    "state": "running",
                    "server": "gpu02",
                    "gpus": [2],
                    "priority": "high",
                    "submitted_at": "2026-09-19T11:28:00+08:00",
                    "estimated_start": "2026-09-19T11:31:00+08:00",
                    "estimated_end": "2026-09-19T15:25:00+08:00",
                    "eta_confidence": "high",
                    "command": "python scripts/sweep.py --dataset synthetic --trial 7",
                },
                {
                    "id": "run-ablation-013",
                    "name": "Ablation / no context gate",
                    "project": "ranking-study",
                    "state": "queued",
                    "server": "gpu01",
                    "gpus": [],
                    "priority": "normal",
                    "submitted_at": "2026-09-19T14:20:00+08:00",
                    "estimated_start": "2026-09-19T16:10:00+08:00",
                    "estimated_end": "2026-09-19T19:40:00+08:00",
                    "eta_confidence": "low",
                    "command": (
                        "python train.py --config configs/ablation.yaml --disable-context-gate"
                    ),
                },
            ],
        }

    @classmethod
    def _build_jobs(cls) -> dict[str, JsonObject]:
        jobs: dict[str, JsonObject] = {}
        for item in cls._build_queue()["items"]:
            logs = {
                "run-ranking-042": [
                    "14:31:44 epoch=8 step=1180/2400 loss=0.2841 gpu_mem=19.3GiB",
                    "14:31:49 epoch=8 step=1185/2400 loss=0.2798 gpu_mem=19.3GiB",
                    "14:31:54 epoch=8 step=1190/2400 loss=0.2812 gpu_mem=19.3GiB",
                ],
                "run-sweep-007": [
                    "14:31:31 trial=7 step=9210/11000 score=0.7432",
                    "14:31:43 trial=7 step=9220/11000 score=0.7434",
                    "14:31:55 trial=7 step=9230/11000 score=0.7435",
                ],
                "run-ablation-013": [
                    "14:20:00 submitted priority=normal requested_gpus=2",
                    "14:20:01 waiting for GPU 0-1 on gpu01",
                ],
            }[item["id"]]
            jobs[item["id"]] = {
                **item,
                "working_directory": f"/data/researcher/{item['project']}",
                "created_by": "researcher",
                "progress": (
                    {"current": 1190, "total": 2400, "unit": "step"}
                    if item["id"] == "run-ranking-042"
                    else None
                ),
                "logs": logs,
                "events": [
                    {
                        "at": item["submitted_at"],
                        "type": "submitted",
                        "message": "作业已进入调度队列",
                    }
                ],
            }
        for state, name, project, server in (
            ("finished", "Baseline / seed 41", "ranking-study", "gpu01"),
            ("failed", "Sweep / trial 6", "benchmark-study", "gpu02"),
            ("cancelled", "Ablation / old config", "ranking-study", "gpu01"),
        ):
            run_id = f"run-demo-{state}"
            jobs[run_id] = {
                "id": run_id,
                "name": name,
                "project": project,
                "state": state,
                "raw_state": "succeeded" if state == "finished" else state,
                "server": server,
                "gpus": [0],
                "priority": "normal",
                "submitted_at": "2026-09-19T10:00:00+08:00",
                "finished_at": "2026-09-19T12:00:00+08:00",
                "estimated_start": "2026-09-19T10:01:00+08:00",
                "estimated_end": "2026-09-19T12:00:00+08:00",
                "eta_confidence": "high",
                "command": "python train.py",
                "working_directory": f"/data/researcher/{project}",
                "created_by": "researcher",
                "progress": None,
                "logs": [f"Job {state}"],
                "events": [],
            }
        return jobs
