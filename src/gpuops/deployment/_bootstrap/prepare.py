import fcntl
import json
import os
import re
import sys
from contextlib import suppress
from pathlib import Path

os.umask(0o077)
TOTAL_BUDGET = 1024 * 1024 * 1024
STATE_BUDGET = 768 * 1024 * 1024
INSTALL_BUDGET = 256 * 1024 * 1024
CACHE_BUDGET = 128 * 1024 * 1024
home = Path.home()
cache = Path.home() / ".cache" / "gpuops"
share = home / ".local" / "share" / "gpuops"
state = home / ".local" / "state" / "gpuops"
cache.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(cache, 0o700)
release_id = sys.argv[1] if len(sys.argv) > 1 else ""
declared_size = int(sys.argv[2]) if len(sys.argv) > 2 else -1
deployment_id = sys.argv[3] if len(sys.argv) > 3 else ""
artifact_name = sys.argv[4] if len(sys.argv) > 4 else ""
if not re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", release_id):
    raise RuntimeError("release id is invalid")
if (
    declared_size < 0
    or not re.fullmatch(r"[0-9a-f]{32}", deployment_id)
    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\.(?:whl|tar\.gz)", artifact_name)
):
    raise RuntimeError("deployment metadata is invalid")
upload = cache / "uploads" / release_id / deployment_id
share.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = share / "deploy.lock"
lock_path.touch(mode=0o600, exist_ok=True)
lock_handle = lock_path.open("r+")
fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
upload.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(upload, 0o700)


def directory_size(path):
    if not path.exists():
        return 0
    return sum(
        candidate.lstat().st_size for candidate in path.rglob("*") if not candidate.is_symlink()
    )


state_bytes = directory_size(state)
cache_bytes = directory_size(cache)
install_bytes = directory_size(share) + cache_bytes
if state_bytes > STATE_BUDGET:
    raise RuntimeError("GPUOps state directory exceeds its 768 MiB budget")
if (
    cache_bytes > CACHE_BUDGET
    or install_bytes > INSTALL_BUDGET
    or state_bytes + install_bytes > TOTAL_BUDGET
):
    raise RuntimeError("GPUOps owned directories exceed the global 1 GiB budget")
available = min(
    CACHE_BUDGET - cache_bytes,
    INSTALL_BUDGET - install_bytes,
    TOTAL_BUDGET - state_bytes - install_bytes,
)
if declared_size > available:
    upload.rmdir()
    for parent in (upload.parent, upload.parent.parent):
        with suppress(OSError):
            parent.rmdir()
    raise RuntimeError("GPUOps budget cannot fit the deployment artifact")
artifact = upload / artifact_name
descriptor = os.open(artifact, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    os.ftruncate(descriptor, declared_size)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(
    json.dumps(
        {
            "ok": True,
            "action": "prepare",
            "available_upload_bytes": max(0, available),
            "deployment_id": deployment_id,
        },
        separators=(",", ":"),
    )
)
