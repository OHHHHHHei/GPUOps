"""Durable replacement of local state files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(
    path: Path, payload: bytes, *, prefix: str | None = None, durable: bool = True
) -> None:
    """Replace a file with private bytes in an existing directory.

    ``durable=False`` skips both fsyncs for short-lived status files that are
    rewritten every few seconds; the replacement is still atomic.
    """

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}." if prefix is None else prefix,
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        if durable:
            fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
