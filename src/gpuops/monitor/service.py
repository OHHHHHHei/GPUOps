"""Concurrent, fault-isolated collection across GPU servers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

import gpuops.config as config
from gpuops.config import MONITOR_TIMEOUT_SECONDS
from gpuops.config import ServerTarget as ServerTarget
from gpuops.ssh import SSHTimeoutError

from .collector import CollectionError, RemoteCollector


@dataclass(frozen=True, slots=True)
class ServerResult:
    server: ServerTarget
    online: bool
    latency_seconds: float | None = None
    data: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": asdict(self.server),
            "online": self.online,
            "latency_seconds": self.latency_seconds,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
        }


class MonitorService:
    def __init__(
        self,
        targets: tuple[ServerTarget, ...] | list[ServerTarget] | None = None,
        *,
        collector: RemoteCollector | None = None,
        timeout_seconds: float = MONITOR_TIMEOUT_SECONDS,
    ) -> None:
        self._targets = tuple(targets) if targets is not None else None
        self.collector = collector or RemoteCollector()
        self.timeout_seconds = timeout_seconds

    @property
    def targets(self) -> tuple[ServerTarget, ...]:
        """Use the current fleet registry unless targets were explicitly supplied."""
        return self._targets if self._targets is not None else config.select_targets()

    async def collect_all(self) -> list[ServerResult]:
        return await self._run_all(self.collector.collect)

    async def doctor_all(self) -> list[ServerResult]:
        return await self._run_all(self.collector.doctor)

    async def _run_all(
        self,
        operation: Callable[..., Awaitable[tuple[dict[str, Any], float]]],
    ) -> list[ServerResult]:
        async def run_one(target: ServerTarget) -> ServerResult:
            try:
                data, latency = await operation(
                    target.ssh_host, timeout_seconds=self.timeout_seconds
                )
                return ServerResult(server=target, online=True, latency_seconds=latency, data=data)
            except Exception as exc:
                # An operation timeout alone does not establish that SSH is down.
                code = (
                    exc.code
                    if isinstance(exc, CollectionError)
                    else "collection_timeout"
                    if isinstance(exc, SSHTimeoutError)
                    else "collection_error"
                )
                return ServerResult(
                    server=target,
                    online=False,
                    error=f"{type(exc).__name__}: {exc}",
                    error_code=code,
                )

        return list(await asyncio.gather(*(run_one(target) for target in self.targets)))
