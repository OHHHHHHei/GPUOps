"""Shared multi-server history queries for the CLI and Web console."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from gpuops.control import RemoteTarget
from gpuops.history import decode_cursor, encode_cursor
from gpuops.models import JobSummary
from gpuops.transport import JobHistory


class HistoryTransport(Protocol):
    async def history(
        self, target: RemoteTarget, *, limit: int, project: str | None, cursor: str | None
    ) -> JobHistory: ...


@dataclass(frozen=True)
class HostHistory:
    target: RemoteTarget
    jobs: tuple[JobSummary, ...] = ()
    projects: tuple[str, ...] = ()
    error: Exception | None = None


@dataclass(frozen=True)
class HistoryPage:
    items: list[tuple[RemoteTarget, JobSummary]]
    hosts: tuple[HostHistory, ...]
    projects: list[str]
    next_cursor: str | None


def validate_history_query(limit: int, project: str | None, cursor: str | None) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("history limit must be an integer between 1 and 100")
    if project is not None and (not isinstance(project, str) or not project):
        raise ValueError("project must be a non-empty string")
    decode_cursor(cursor)


async def collect_history(
    transport: HistoryTransport,
    targets: Sequence[RemoteTarget],
    *,
    limit: int = 50,
    project: str | None = None,
    cursor: str | None = None,
) -> HistoryPage:
    validate_history_query(limit, project, cursor)

    async def collect(target: RemoteTarget) -> HostHistory:
        try:
            result = await transport.history(
                target, limit=limit + 1, project=project, cursor=cursor
            )
            return HostHistory(target, tuple(result.jobs), tuple(result.projects))
        except Exception as error:
            return HostHistory(target, error=error)

    hosts = tuple(await asyncio.gather(*(collect(target) for target in targets)))
    entries = sorted(
        ((host.target, job) for host in hosts for job in host.jobs),
        key=lambda item: (item[1].finished_at, item[1].run_id, item[0].server_id),
        reverse=True,
    )
    items = entries[:limit]
    next_cursor = None
    # Paging through an incomplete fleet could skip records when a host returns.
    if len(entries) > limit and all(host.error is None for host in hosts):
        target, job = items[-1]
        next_cursor = encode_cursor((job.finished_at, job.run_id, target.server_id))
    return HistoryPage(
        items=items,
        hosts=hosts,
        projects=sorted({name for host in hosts for name in host.projects}),
        next_cursor=next_cursor,
    )
