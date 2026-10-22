from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpuops.core.storage import StorageBudget, StorageQuotaExceeded
from gpuops.models import PRIORITY_ORDER, JobRecord, JobStatus, JobSummary

INDEX_SCHEMA_VERSION = 1
INDEX_FILENAME = "queue-index.sqlite3"
_TERMINAL_STATUSES = (
    JobStatus.SUCCEEDED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class DirectorySignature:
    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def read(cls, directory: Path) -> DirectorySignature:
        metadata = directory.stat()
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
        )

    def encode(self) -> str:
        return json.dumps(
            [self.device, self.inode, self.size, self.modified_ns],
            separators=(",", ":"),
        )


class QueueIndex:
    """Rebuildable SQLite projection of authoritative per-job JSON records."""

    def __init__(self, root: Path, jobs_directory: Path, storage: StorageBudget) -> None:
        self.root = root
        self.jobs_directory = jobs_directory
        self.storage = storage
        self.path = root / INDEX_FILENAME
        self._known_signature: DirectorySignature | None = None

    def signature(self) -> DirectorySignature:
        return DirectorySignature.read(self.jobs_directory)

    def count(self, load_records: Callable[[], list[JobRecord]]) -> int:
        return int(self._query(load_records, "SELECT COUNT(*) FROM jobs", [])[0][0])

    def summaries(
        self,
        load_records: Callable[[], list[JobRecord]],
        statuses: set[JobStatus] | None = None,
    ) -> list[JobSummary]:
        parameters: list[object] = []
        where = ""
        if statuses is not None:
            if not statuses:
                return []
            values = sorted(status.value for status in statuses)
            where = f" WHERE status IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query = f"SELECT * FROM jobs{where} ORDER BY priority_rank ASC, created_us ASC, run_id ASC"
        return self._query_summaries(load_records, query, parameters)

    def history(
        self,
        load_records: Callable[[], list[JobRecord]],
        *,
        server_id: str,
        limit: int,
        project: str | None,
        before: tuple[datetime, str, str] | None,
    ) -> list[JobSummary]:
        if limit < 1:
            raise ValueError("history limit must be positive")
        clauses = ["finished_us IS NOT NULL", "status IN (?,?,?)"]
        parameters: list[object] = list(_TERMINAL_STATUSES)
        if project is not None:
            clauses.append("project = ?")
            parameters.append(project)
        if before is not None:
            finished_at, run_id, before_server_id = before
            finished_us = _datetime_microseconds(finished_at)
            comparator = "<=" if server_id < before_server_id else "<"
            clauses.append(f"(finished_us, run_id) {comparator} (?, ?)")
            parameters.extend([finished_us, run_id])
        parameters.append(limit)
        index_name = "jobs_project_history" if project is not None else "jobs_history"
        query = (
            f"SELECT * FROM jobs INDEXED BY {index_name} WHERE "
            + " AND ".join(clauses)
            + " ORDER BY finished_us DESC, run_id DESC LIMIT ?"
        )
        return self._query_summaries(load_records, query, parameters)

    def history_projects(
        self,
        load_records: Callable[[], list[JobRecord]],
    ) -> list[str]:
        query = (
            "SELECT DISTINCT project FROM jobs WHERE finished_us IS NOT NULL "
            "AND status IN (?,?,?) "
            "ORDER BY project ASC"
        )
        rows = self._query(load_records, query, list(_TERMINAL_STATUSES))
        return [str(row["project"]) for row in rows]

    def record_written(
        self,
        record: JobRecord,
        *,
        previous_signature: DirectorySignature,
        current_signature: DirectorySignature,
    ) -> None:
        if not self._is_current(previous_signature):
            self._known_signature = None
            return

        summary = JobSummary.from_record(record)
        try:
            growth = self._estimated_incremental_growth(summary)
            if growth:
                self.storage.ensure_capacity(additional_bytes=growth)
            with self._connect() as connection:
                self._upsert(connection, summary)
                self._set_metadata(connection, current_signature)
            os.chmod(self.path, 0o600)
            self._known_signature = current_signature
        except (OSError, sqlite3.DatabaseError, StorageQuotaExceeded):
            self.invalidate()

    def invalidate(self) -> None:
        self._known_signature = None
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-journal"),
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            with suppress(OSError):
                path.unlink()
        for path in self.root.glob(f".{INDEX_FILENAME}.*.tmp"):
            with suppress(OSError):
                path.unlink()

    def _query_summaries(
        self,
        load_records: Callable[[], list[JobRecord]],
        query: str,
        parameters: list[object],
    ) -> list[JobSummary]:
        return [_summary_from_row(row) for row in self._query(load_records, query, parameters)]

    def _query(
        self,
        load_records: Callable[[], list[JobRecord]],
        query: str,
        parameters: list[object],
    ) -> list[sqlite3.Row]:
        try:
            self._ensure_current(load_records)
            try:
                with self._connect(readonly=True) as connection:
                    return list(connection.execute(query, parameters))
            except (OSError, sqlite3.DatabaseError):
                self.invalidate()
                self._ensure_current(load_records)
                with self._connect(readonly=True) as connection:
                    return list(connection.execute(query, parameters))
        except (OSError, sqlite3.DatabaseError, StorageQuotaExceeded):
            self.invalidate()
            return self._query_memory(load_records, query, parameters)

    def _query_memory(
        self,
        load_records: Callable[[], list[JobRecord]],
        query: str,
        parameters: list[object],
    ) -> list[sqlite3.Row]:
        connection = self._memory_index(JobSummary.from_record(record) for record in load_records())
        try:
            return list(connection.execute(query, parameters))
        finally:
            connection.close()

    def _ensure_current(self, load_records: Callable[[], list[JobRecord]]) -> None:
        signature = self.signature()
        if self._is_current(signature):
            return
        records = load_records()
        current_signature = self.signature()
        self._rebuild((JobSummary.from_record(record) for record in records), current_signature)

    def _is_current(self, signature: DirectorySignature) -> bool:
        if self._known_signature == signature and self.path.is_file():
            return True
        if not self.path.is_file():
            return False
        try:
            with self._connect(readonly=True) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        except (OSError, sqlite3.DatabaseError):
            return False
        if metadata.get("schema_version") != str(INDEX_SCHEMA_VERSION):
            return False
        if metadata.get("jobs_signature") != signature.encode():
            return False
        self._known_signature = signature
        return True

    def _rebuild(
        self,
        summaries: Iterable[JobSummary],
        signature: DirectorySignature,
    ) -> None:
        memory = self._memory_index(summaries)
        try:
            self._set_metadata(memory, signature)
            memory.commit()
            page_size = int(memory.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(memory.execute("PRAGMA page_count").fetchone()[0])
            image_bytes = page_size * page_count
            try:
                previous_bytes = self.path.stat().st_size
            except FileNotFoundError:
                previous_bytes = 0
            self.storage.ensure_capacity(additional_bytes=max(0, image_bytes - previous_bytes))
            self.invalidate()
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.root,
                prefix=f".{INDEX_FILENAME}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            try:
                destination = sqlite3.connect(temporary_path)
                try:
                    self._configure(destination)
                    memory.backup(destination)
                    destination.commit()
                finally:
                    destination.close()
                os.chmod(temporary_path, 0o600)
                with temporary_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.path)
                _fsync_directory(self.root)
            finally:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()
        except (OSError, sqlite3.DatabaseError, StorageQuotaExceeded):
            self.invalidate()
            raise
        finally:
            memory.close()
        self._known_signature = signature

    def _memory_index(self, summaries: Iterable[JobSummary]) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        try:
            connection.row_factory = sqlite3.Row
            self._configure(connection)
            self._create_schema(connection)
            for summary in summaries:
                self._upsert(connection, summary)
            connection.commit()
        except BaseException:
            connection.close()
            raise
        return connection

    def _estimated_incremental_growth(self, summary: JobSummary) -> int:
        encoded_bytes = len(summary.model_dump_json().encode("utf-8"))
        try:
            with self._connect(readonly=True) as connection:
                row = connection.execute(
                    "SELECT payload_bytes FROM jobs WHERE run_id = ?",
                    (summary.run_id,),
                ).fetchone()
                page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
                free_pages = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        except (OSError, sqlite3.DatabaseError):
            return encoded_bytes * 6 + 32 * 1024
        previous_bytes = int(row[0]) if row is not None else 0
        estimated_bytes = max(0, encoded_bytes - previous_bytes) * 6
        estimated_bytes += page_size * (6 if row is None else 2)
        return max(0, estimated_bytes - free_pages * page_size)

    @contextmanager
    def _connect(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        if readonly:
            connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=5,
            )
            connection.execute("PRAGMA query_only = ON")
        else:
            connection = sqlite3.connect(self.path, timeout=5)
            self._configure(connection)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        except BaseException:
            if not readonly:
                connection.rollback()
            raise
        else:
            if not readonly:
                connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE jobs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                priority_rank INTEGER NOT NULL,
                gpus TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_us INTEGER NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                finished_us INTEGER,
                expected_duration_us INTEGER,
                payload_bytes INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX jobs_status_queue
                ON jobs(status, priority_rank, created_us, run_id);
            CREATE INDEX jobs_history
                ON jobs(finished_us DESC, run_id DESC)
                WHERE finished_us IS NOT NULL;
            CREATE INDEX jobs_project_history
                ON jobs(project, finished_us DESC, run_id DESC)
                WHERE finished_us IS NOT NULL;
            """
        )

    @staticmethod
    def _upsert(connection: sqlite3.Connection, summary: JobSummary) -> None:
        values = _summary_values(summary)
        connection.execute(
            """
            INSERT INTO jobs (
                run_id, name, project, status, priority, priority_rank, gpus,
                created_at, created_us, started_at, finished_at, finished_us,
                expected_duration_us, payload_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                name = excluded.name,
                project = excluded.project,
                status = excluded.status,
                priority = excluded.priority,
                priority_rank = excluded.priority_rank,
                gpus = excluded.gpus,
                created_at = excluded.created_at,
                created_us = excluded.created_us,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                finished_us = excluded.finished_us,
                expected_duration_us = excluded.expected_duration_us,
                payload_bytes = excluded.payload_bytes
            """,
            values,
        )

    @staticmethod
    def _set_metadata(
        connection: sqlite3.Connection,
        signature: DirectorySignature,
    ) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", str(INDEX_SCHEMA_VERSION)),
                ("jobs_signature", signature.encode()),
            ),
        )


def _summary_values(summary: JobSummary) -> tuple[object, ...]:
    encoded_bytes = len(summary.model_dump_json().encode("utf-8"))
    return (
        summary.run_id,
        summary.name,
        summary.project,
        summary.status.value,
        summary.priority.value,
        PRIORITY_ORDER[summary.priority],
        json.dumps(summary.gpus, separators=(",", ":")),
        summary.created_at.isoformat(),
        _datetime_microseconds(summary.created_at),
        summary.started_at.isoformat() if summary.started_at is not None else None,
        summary.finished_at.isoformat() if summary.finished_at is not None else None,
        (_datetime_microseconds(summary.finished_at) if summary.finished_at is not None else None),
        (
            _timedelta_microseconds(summary.expected_duration)
            if summary.expected_duration is not None
            else None
        ),
        encoded_bytes,
    )


def _summary_from_row(row: sqlite3.Row) -> JobSummary:
    expected_duration_us = row["expected_duration_us"]
    return JobSummary(
        run_id=row["run_id"],
        name=row["name"],
        project=row["project"],
        status=row["status"],
        priority=row["priority"],
        gpus=json.loads(row["gpus"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=(
            datetime.fromisoformat(row["started_at"]) if row["started_at"] is not None else None
        ),
        finished_at=(
            datetime.fromisoformat(row["finished_at"]) if row["finished_at"] is not None else None
        ),
        expected_duration=(
            timedelta(microseconds=int(expected_duration_us))
            if expected_duration_us is not None
            else None
        ),
    )


def _datetime_microseconds(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("history timestamps must include a timezone")
    delta = value.astimezone(timezone.utc) - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _timedelta_microseconds(value: timedelta) -> int:
    return value.days * 86_400_000_000 + value.seconds * 1_000_000 + value.microseconds


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
