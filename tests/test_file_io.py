from __future__ import annotations

import errno
import stat

import pytest

from gpuops import file_io


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_write_preserves_bytes_and_private_permissions(tmp_path, existing):
    path = tmp_path / "record"
    if existing:
        path.write_bytes(b"old contents\n")
        path.chmod(0o644)
    payload = b"\x00\xff\r\n" + "训练\n".encode()

    file_io.atomic_write_bytes(path, payload)

    assert path.read_bytes() == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failure_before_replacement_preserves_old_file(tmp_path, monkeypatch, operation):
    path = tmp_path / "record"
    path.write_bytes(b"old contents\n")

    def disk_full(*args):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(file_io.os, operation, disk_full)

    with pytest.raises(OSError) as error:
        file_io.atomic_write_bytes(path, b"new contents\n")

    assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == b"old contents\n"
    assert list(tmp_path.iterdir()) == [path]


def test_directory_sync_error_leaves_complete_replacement(tmp_path, monkeypatch):
    path = tmp_path / "record"
    path.write_bytes(b"old contents\n")

    def sync_failed(directory):
        raise OSError(errno.EIO, "directory sync failed")

    monkeypatch.setattr(file_io, "fsync_directory", sync_failed)

    with pytest.raises(OSError) as error:
        file_io.atomic_write_bytes(path, b"new contents\n")

    assert error.value.errno == errno.EIO
    assert path.read_bytes() == b"new contents\n"
    assert list(tmp_path.iterdir()) == [path]
