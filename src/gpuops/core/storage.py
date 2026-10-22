from __future__ import annotations

import fcntl
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from gpuops.contracts import is_valid_run_id

GIB = 1024**3
DEFAULT_TOTAL_BUDGET_BYTES = GIB
DEFAULT_LOG_BUDGET_BYTES = GIB


class StorageError(RuntimeError):
    pass


class StorageQuotaExceeded(StorageError):
    pass


@dataclass(frozen=True)
class StorageUsage:
    total_bytes: int
    log_bytes: int
    event_archive_bytes: int


@dataclass(frozen=True)
class CleanupReport:
    before: StorageUsage
    after: StorageUsage
    removed_paths: tuple[str, ...]
    reclaimed_bytes: int


class StorageBudget:
    """Budget and cleanup policy for GPUOps-owned state only.

    Cleanup is deliberately limited to ``logs/`` and ``event-archive/`` below
    the state root. Queue records and arbitrary experiment outputs are never
    cleanup candidates.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        total_budget_bytes: int = DEFAULT_TOTAL_BUDGET_BYTES,
        log_budget_bytes: int = DEFAULT_LOG_BUDGET_BYTES,
        high_watermark: float = 0.90,
        target_watermark: float = 0.75,
    ) -> None:
        if total_budget_bytes <= 0 or log_budget_bytes <= 0:
            raise ValueError("storage budgets must be positive")
        if log_budget_bytes > total_budget_bytes:
            raise ValueError("log budget cannot exceed the total storage budget")
        if not 0 < target_watermark < high_watermark <= 1:
            raise ValueError("watermarks must satisfy 0 < target < high <= 1")

        self.root = Path(root)
        self.total_budget_bytes = total_budget_bytes
        self.log_budget_bytes = log_budget_bytes
        self.high_watermark = high_watermark
        self.target_watermark = target_watermark
        self.logs_directory = self.root / "logs"
        self.event_archive_directory = self.root / "event-archive"
        self.reservation_directory = self.root / ".log-reservations"
        self.lock_path = self.root / ".storage.lock"

        self._secure_directory(self.root)
        self._secure_directory(self.logs_directory)
        self._secure_directory(self.event_archive_directory)
        self._secure_directory(self.reservation_directory)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.lock_path, 0o600)

    @staticmethod
    def _secure_directory(path: Path) -> None:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def usage(self) -> StorageUsage:
        with self._lock():
            return self._usage_unlocked()

    def prepare_log_file(self, relative_path: str | Path) -> Path:
        """Create a private GPUOps log file and return its absolute path."""

        target = self._safe_log_path(relative_path)
        with self._lock():
            self._cleanup_unlocked(0, 0, set())
            current = self._usage_unlocked()
            if current.total_bytes > self.total_budget_bytes:
                raise StorageQuotaExceeded("GPUOps storage is already over its hard limit")
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            parent = target.parent
            while True:
                os.chmod(parent, 0o700)
                if parent == self.logs_directory:
                    break
                parent = parent.parent
            target.touch(mode=0o600, exist_ok=True)
            os.chmod(target, 0o600)
        return target

    def cleanup_high_watermark(
        self,
        *,
        additional_bytes: int = 0,
        additional_log_bytes: int = 0,
        protected_paths: Iterable[str | Path] = (),
    ) -> CleanupReport:
        self._validate_additional(additional_bytes, additional_log_bytes)
        protected = {self._resolved(path) for path in protected_paths}
        with self._lock():
            return self._cleanup_unlocked(additional_bytes, additional_log_bytes, protected)

    def reserve_log_capacity(self, key: str, size_bytes: int) -> Path:
        """Atomically reserve the maximum bytes a live log may consume."""

        if not is_valid_run_id(key):
            raise ValueError("invalid log reservation key")
        if size_bytes <= 0:
            raise ValueError("log reservation size must be positive")
        path = self.reservation_directory / f"{key}.reserve"
        with self._lock():
            if path.exists():
                if path.stat().st_size != size_bytes:
                    raise StorageError("log reservation already exists with a different size")
                return path
            report = self._cleanup_unlocked(size_bytes, size_bytes, set())
            if report.after.total_bytes + size_bytes > self.total_budget_bytes:
                raise StorageQuotaExceeded(
                    f"GPUOps storage would exceed {self.total_budget_bytes} bytes"
                )
            if report.after.log_bytes + size_bytes > self.log_budget_bytes:
                raise StorageQuotaExceeded(
                    f"GPUOps logs would exceed {self.log_budget_bytes} bytes"
                )
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                os.ftruncate(descriptor, size_bytes)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return path

    def release_log_capacity(self, key: str) -> None:
        if not is_valid_run_id(key):
            raise ValueError("invalid log reservation key")
        with self._lock():
            (self.reservation_directory / f"{key}.reserve").unlink(missing_ok=True)

    def ensure_capacity(
        self,
        *,
        additional_bytes: int = 0,
        additional_log_bytes: int = 0,
        protected_paths: Iterable[str | Path] = (),
    ) -> CleanupReport:
        """Clean at high water and reject writes that exceed a hard budget."""

        self._validate_additional(additional_bytes, additional_log_bytes)
        protected = {self._resolved(path) for path in protected_paths}
        with self._lock():
            report = self._cleanup_unlocked(additional_bytes, additional_log_bytes, protected)
            usage = report.after
            if usage.total_bytes + additional_bytes > self.total_budget_bytes:
                raise StorageQuotaExceeded(
                    f"GPUOps storage would exceed {self.total_budget_bytes} bytes"
                )
            if usage.log_bytes + additional_log_bytes > self.log_budget_bytes:
                raise StorageQuotaExceeded(
                    f"GPUOps logs would exceed {self.log_budget_bytes} bytes"
                )
            return report

    def _cleanup_unlocked(
        self,
        additional_bytes: int,
        additional_log_bytes: int,
        protected: set[Path],
    ) -> CleanupReport:
        before = self._usage_unlocked()
        removed: list[str] = []
        protected = protected.union(self._active_log_paths_unlocked())

        log_high = int(self.log_budget_bytes * self.high_watermark)
        if before.log_bytes + additional_log_bytes > log_high:
            log_target = max(
                0,
                int(self.log_budget_bytes * self.target_watermark) - additional_log_bytes,
            )
            self._remove_oldest_until(
                self._regular_files(self.logs_directory),
                lambda: self._directory_size(self.logs_directory),
                log_target,
                protected,
                removed,
            )

        current = self._usage_unlocked()
        total_high = int(self.total_budget_bytes * self.high_watermark)
        if current.total_bytes + additional_bytes > total_high:
            total_target = max(
                0,
                int(self.total_budget_bytes * self.target_watermark) - additional_bytes,
            )
            candidates = [
                *self._regular_files(self.logs_directory),
                *self._regular_files(self.event_archive_directory),
            ]
            self._remove_oldest_until(
                candidates,
                lambda: self._usage_unlocked().total_bytes,
                total_target,
                protected,
                removed,
            )

        self._remove_empty_directories(self.logs_directory)
        after = self._usage_unlocked()
        return CleanupReport(
            before=before,
            after=after,
            removed_paths=tuple(removed),
            reclaimed_bytes=max(0, before.total_bytes - after.total_bytes),
        )

    def _remove_oldest_until(
        self,
        candidates: Iterable[Path],
        current_size: Callable[[], int],
        target_bytes: int,
        protected: set[Path],
        removed: list[str],
    ) -> None:
        ordered = sorted(
            (path for path in candidates if self._resolved(path) not in protected),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        for path in ordered:
            if current_size() <= target_bytes:
                break
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(path.relative_to(self.root)))

    def _usage_unlocked(self) -> StorageUsage:
        return StorageUsage(
            total_bytes=self._directory_size(self.root),
            log_bytes=(
                self._directory_size(self.logs_directory)
                + self._directory_size(self.reservation_directory)
            ),
            event_archive_bytes=self._directory_size(self.event_archive_directory),
        )

    def _active_log_paths_unlocked(self) -> set[Path]:
        protected: set[Path] = set()
        for reservation in self.reservation_directory.glob("*.reserve"):
            run_id = reservation.name.removesuffix(".reserve")
            run_logs = self.logs_directory / run_id
            protected.update(self._resolved(path) for path in self._regular_files(run_logs))
        return protected

    @classmethod
    def _directory_size(cls, directory: Path) -> int:
        total = 0
        for path in cls._regular_files(directory):
            # Atomic state updates can remove a temporary file after enumeration.
            with suppress(FileNotFoundError):
                total += path.stat().st_size
        return total

    @staticmethod
    def _regular_files(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        files: list[Path] = []
        for root, directories, names in os.walk(directory, followlinks=False):
            root_path = Path(root)
            directories[:] = [
                name for name in directories if not (root_path / name).is_symlink()
            ]
            for name in names:
                path = root_path / name
                if path.is_symlink():
                    continue
                try:
                    if path.is_file():
                        files.append(path)
                except FileNotFoundError:
                    continue
        return files

    @staticmethod
    def _remove_empty_directories(root: Path) -> None:
        directories = [path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            with suppress(OSError):
                directory.rmdir()

    def _safe_log_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("log path must be a relative path within the GPUOps log directory")
        target = self._resolved(self.logs_directory / relative)
        if target == self._resolved(self.logs_directory) or self._resolved(
            self.logs_directory
        ) not in target.parents:
            raise ValueError("log path escapes the GPUOps log directory")
        return target

    @staticmethod
    def _resolved(path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    @staticmethod
    def _validate_additional(additional_bytes: int, additional_log_bytes: int) -> None:
        if additional_bytes < 0 or additional_log_bytes < 0:
            raise ValueError("additional byte counts cannot be negative")
        if additional_log_bytes > additional_bytes:
            raise ValueError("additional log bytes cannot exceed additional total bytes")
